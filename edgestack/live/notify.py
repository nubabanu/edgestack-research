"""Pluggable at-least-once external notification channels."""

from __future__ import annotations

import asyncio
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Protocol

import httpx
from rich.console import Console

from edgestack.live.state import OutboxRecord
from edgestack.models import AlertEvent


class NotificationChannel(Protocol):
    """External delivery interface."""

    name: str

    async def send(self, event: AlertEvent, idempotency_key: str) -> str:
        """Send an event and return a provider receipt."""


class OutboxStore(Protocol):
    """Minimal transactional-outbox interface used by the dispatcher."""

    def lease_outbox(self, limit: int = 100) -> tuple[OutboxRecord, ...]:
        """Lease deliverable records."""

    def acknowledge(self, outbox_id: int, provider_receipt: str) -> None:
        """Record successful external delivery."""

    def retry(self, outbox_id: int, error: str) -> None:
        """Release a failed delivery for retry/dead-letter handling."""


@dataclass(slots=True)
class ConsoleChannel:
    """Always-available console/log channel."""

    name: str = "console"
    console: Console = field(default_factory=Console)

    async def send(self, event: AlertEvent, idempotency_key: str) -> str:
        """Print an event with its stable idempotency key."""

        self.console.print(
            f"[bold]{event.event_type}[/bold] {event.message} [{idempotency_key}]"
        )
        return idempotency_key


@dataclass(slots=True)
class InMemoryIdempotentChannel:
    """Recorded receiver used by restart tests."""

    name: str = "memory"
    received: dict[str, AlertEvent] = field(default_factory=dict)

    async def send(self, event: AlertEvent, idempotency_key: str) -> str:
        """Store each logical event at most once."""

        self.received.setdefault(idempotency_key, event)
        return idempotency_key


@dataclass(frozen=True, slots=True)
class WebhookChannel:
    """Generic JSON webhook channel."""

    url: str
    name: str = "webhook"
    timeout_seconds: float = 15.0

    async def send(self, event: AlertEvent, idempotency_key: str) -> str:
        """POST an event with an Idempotency-Key header."""

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.url,
                json={
                    "event_id": event.event_id,
                    "recommendation_id": event.recommendation_id,
                    "revision": event.revision,
                    "event_type": event.event_type,
                    "message": event.message,
                    "created_at": event.created_at.isoformat(),
                },
                headers={"Idempotency-Key": idempotency_key},
            )
            response.raise_for_status()
            return str(response.headers.get("x-request-id", idempotency_key))


@dataclass(frozen=True, slots=True)
class TelegramChannel:
    """Telegram Bot API channel; tokens must come from environment-backed config."""

    token: str
    chat_id: str
    name: str = "telegram"

    async def send(self, event: AlertEvent, idempotency_key: str) -> str:
        """Send a Telegram message with the stable event ID in its body."""

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": f"{event.message}\nID: {idempotency_key}",
                },
            )
            response.raise_for_status()
            payload = response.json()
        return str(payload.get("result", {}).get("message_id", idempotency_key))


@dataclass(frozen=True, slots=True)
class EmailChannel:
    """SMTP email channel with at-least-once semantics."""

    host: str
    port: int
    sender: str
    recipient: str
    username: str | None = None
    password: str | None = None
    starttls: bool = True
    name: str = "email"

    async def send(self, event: AlertEvent, idempotency_key: str) -> str:
        """Send email in a worker thread to avoid blocking the monitor."""

        message = EmailMessage()
        message["Subject"] = f"EdgeStack {event.event_type} [{idempotency_key}]"
        message["From"] = self.sender
        message["To"] = self.recipient
        message["Message-ID"] = f"<{idempotency_key}@edgestack.local>"
        message.set_content(event.message)

        def deliver() -> None:
            with smtplib.SMTP(self.host, self.port, timeout=20) as server:
                if self.starttls:
                    server.starttls()
                if self.username:
                    server.login(self.username, self.password or "")
                server.send_message(message)

        await asyncio.to_thread(deliver)
        return idempotency_key


async def dispatch_outbox(
    store: OutboxStore,
    channels: dict[str, NotificationChannel],
    *,
    limit: int = 100,
) -> int:
    """Lease, deliver, and acknowledge outbox records.

    ``store`` is duck-typed to keep notification transports independent of the
    SQLite implementation and easy to fault-inject in tests.
    """

    if limit <= 0:
        raise ValueError("dispatch limit must be positive")
    records = store.lease_outbox(limit=limit)
    delivered = 0
    for record in records:
        channel = channels.get(record.channel)
        if channel is None:
            store.retry(record.outbox_id, f"channel unavailable: {record.channel}")
            continue
        try:
            receipt = await channel.send(record.event, record.idempotency_key)
        except Exception as exc:  # transport boundary deliberately broad
            store.retry(record.outbox_id, repr(exc))
        else:
            store.acknowledge(record.outbox_id, receipt)
            delivered += 1
    return delivered
