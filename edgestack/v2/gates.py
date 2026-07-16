"""Fail-closed V2 data-entitlement gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edgestack.models import (
    DataTier,
    EstimateVintage,
    IntradayMarketRecord,
    MarketRecordKind,
    MembershipInterval,
    TickerValidityInterval,
)


class CapabilityStatus(StrEnum):
    """V2 capability result; unavailable is distinct from invalid evidence."""

    PASS = "PASS"
    FAIL = "FAIL"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class CapabilityGate:
    """One persisted-ready capability decision."""

    name: str
    status: CapabilityStatus
    reason: str
    observations: int


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """The three entitlement gates that control V2 promotion."""

    pit_membership: CapabilityGate
    estimate_vintages: CapabilityGate
    auction_execution: CapabilityGate

    @property
    def promotable(self) -> bool:
        """Return true only when every non-substitutable input passed."""

        return all(
            gate.status is CapabilityStatus.PASS
            for gate in (
                self.pit_membership,
                self.estimate_vintages,
                self.auction_execution,
            )
        )


def evaluate_capabilities(
    memberships: tuple[MembershipInterval, ...] = (),
    ticker_history: tuple[TickerValidityInterval, ...] = (),
    estimates: tuple[EstimateVintage, ...] = (),
    intraday: tuple[IntradayMarketRecord, ...] = (),
) -> CapabilityReport:
    """Evaluate genuine PIT, estimate-vintage, and auction coverage."""

    pit = _pit_gate(memberships, ticker_history)
    estimate = _estimate_gate(estimates)
    auction = _auction_gate(intraday)
    return CapabilityReport(pit, estimate, auction)


def _pit_gate(
    memberships: tuple[MembershipInterval, ...],
    ticker_history: tuple[TickerValidityInterval, ...],
) -> CapabilityGate:
    if not memberships or not ticker_history:
        return CapabilityGate(
            "PIT_MEMBERSHIP",
            CapabilityStatus.DATA_UNAVAILABLE,
            "Hash-pinned membership intervals and permanent-ID ticker history are required.",
            len(memberships),
        )
    if any(
        item.data_tier is not DataTier.POINT_IN_TIME
        or not item.security_id
        or item.available_at is None
        or item.fetched_at is None
        or not item.content_hash
        for item in memberships
    ):
        return CapabilityGate(
            "PIT_MEMBERSHIP",
            CapabilityStatus.FAIL,
            "Approximate, current-only, or incomplete membership records cannot pass PIT.",
            len(memberships),
        )
    mapped = {item.security_id for item in ticker_history}
    if not {str(item.security_id) for item in memberships}.issubset(mapped):
        return CapabilityGate(
            "PIT_MEMBERSHIP",
            CapabilityStatus.FAIL,
            "At least one member lacks a permanent-ID ticker-validity interval.",
            len(memberships),
        )
    return CapabilityGate(
        "PIT_MEMBERSHIP",
        CapabilityStatus.PASS,
        "Complete entitled membership and ticker intervals were supplied.",
        len(memberships),
    )


def _estimate_gate(estimates: tuple[EstimateVintage, ...]) -> CapabilityGate:
    if not estimates:
        return CapabilityGate(
            "ESTIMATE_VINTAGES",
            CapabilityStatus.DATA_UNAVAILABLE,
            "Historical consensus vintages known at each decision time are not configured.",
            0,
        )
    identities = {(item.estimate_id, item.revision) for item in estimates}
    if len(identities) != len(estimates) or any(
        not item.revision for item in estimates
    ):
        return CapabilityGate(
            "ESTIMATE_VINTAGES",
            CapabilityStatus.FAIL,
            "Estimate vintages contain duplicates or noncausal timestamps.",
            len(estimates),
        )
    return CapabilityGate(
        "ESTIMATE_VINTAGES",
        CapabilityStatus.PASS,
        "Historical estimate revisions retain their original knowledge timestamps.",
        len(estimates),
    )


def _auction_gate(records: tuple[IntradayMarketRecord, ...]) -> CapabilityGate:
    required = {
        MarketRecordKind.NBBO,
        MarketRecordKind.TRADE,
        MarketRecordKind.IMBALANCE,
        MarketRecordKind.AUCTION_PRINT,
    }
    if not records:
        return CapabilityGate(
            "AUCTION_EXECUTION",
            CapabilityStatus.DATA_UNAVAILABLE,
            "NBBO, trades, imbalances, and official auction prints are not configured.",
            0,
        )
    by_security: dict[str, set[MarketRecordKind]] = {}
    for record in records:
        by_security.setdefault(record.security_id, set()).add(record.kind)
    incomplete = [
        key for key, kinds in by_security.items() if not required.issubset(kinds)
    ]
    if incomplete:
        return CapabilityGate(
            "AUCTION_EXECUTION",
            CapabilityStatus.FAIL,
            f"Incomplete auction record kinds for {len(incomplete)} securities.",
            len(records),
        )
    return CapabilityGate(
        "AUCTION_EXECUTION",
        CapabilityStatus.PASS,
        "Every supplied finalist has NBBO/trade/imbalance/official-print evidence.",
        len(records),
    )
