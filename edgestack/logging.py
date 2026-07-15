"""Structured console/JSON logging configuration."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(*, json_output: bool = False, level: str = "INFO") -> None:
    """Configure stdlib and structlog once for CLI or long-lived service use."""

    logging.basicConfig(
        format="%(message)s", stream=sys.stderr, level=level.upper(), force=True
    )
    # Provider requests are represented by immutable payload metadata and phase
    # diagnostics. Per-request INFO lines swamp a 500-instrument campaign and can
    # expose signed query parameters from third-party clients.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    renderer = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
