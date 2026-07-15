from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from edgestack.cli import app
from edgestack.disclaimer import DISCLAIMER
from edgestack.models import GateStatus

runner = CliRunner()
CONFIG = Path("configs/smoke.yaml").resolve()


def _install_runner(monkeypatch: pytest.MonkeyPatch, runner_type: type[object]) -> None:
    module = ModuleType("edgestack.pipeline.runner")
    module.CampaignRunner = runner_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edgestack.pipeline.runner", module)


def test_help_lists_public_commands_and_disclaimer() -> None:
    result = runner.invoke(app, ["--help"])
    normalized_output = " ".join(result.output.split())

    assert result.exit_code == 0
    for command in (
        "ingest",
        "replicate",
        "discover",
        "validate",
        "report",
        "score",
        "live",
        "live-demo",
    ):
        assert command in result.output
    assert " ".join(DISCLAIMER.split()) in normalized_output


def test_ingest_accepts_iso_date_and_passes_a_date_to_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCampaignRunner:
        campaign_id = "cli-date"

        @classmethod
        def create(
            cls,
            config: object,
            *,
            campaign_id: str | None,
            as_of: date | None,
        ) -> FakeCampaignRunner:
            del config
            assert campaign_id == "cli-date"
            assert as_of == date(2024, 1, 31)
            return cls()

        def ingest(self) -> SimpleNamespace:
            return SimpleNamespace(status=GateStatus.PASS, summary="offline test")

    _install_runner(monkeypatch, FakeCampaignRunner)
    result = runner.invoke(
        app,
        [
            "ingest",
            "--config",
            str(CONFIG),
            "--campaign-id",
            "cli-date",
            "--as-of",
            "2024-01-31",
        ],
    )

    assert result.exit_code == 0
    assert "Campaign cli-date: data gate PASS" in result.output


@pytest.mark.parametrize("gate_status", [GateStatus.FAIL, GateStatus.BLOCKED])
def test_ingest_returns_nonzero_after_reporting_failed_persisted_gate(
    monkeypatch: pytest.MonkeyPatch, gate_status: GateStatus
) -> None:
    persisted: list[GateStatus] = []

    class FakeCampaignRunner:
        campaign_id = "failed-ingest"

        @classmethod
        def create(
            cls,
            config: object,
            *,
            campaign_id: str | None,
            as_of: date | None,
        ) -> FakeCampaignRunner:
            del config, campaign_id, as_of
            return cls()

        def ingest(self) -> SimpleNamespace:
            persisted.append(gate_status)
            return SimpleNamespace(status=gate_status, summary="diagnostic saved")

    _install_runner(monkeypatch, FakeCampaignRunner)
    result = runner.invoke(
        app,
        ["ingest", "--config", str(CONFIG), "--campaign-id", "failed-ingest"],
    )

    assert result.exit_code == 1
    assert persisted == [gate_status]
    assert f"data gate {gate_status.value}" in result.output
    assert "diagnostic saved" in result.output


@pytest.mark.parametrize(
    ("gate_status", "expected_exit_code"),
    [
        (GateStatus.PASS, 0),
        (GateStatus.FAIL, 1),
        (GateStatus.BLOCKED, 1),
    ],
)
def test_phase_exit_code_reflects_persisted_gate_status(
    monkeypatch: pytest.MonkeyPatch,
    gate_status: GateStatus,
    expected_exit_code: int,
) -> None:
    persisted: list[GateStatus] = []

    class FakeCampaignRunner:
        @classmethod
        def open(cls, config: object, campaign_id: str) -> FakeCampaignRunner:
            del config, campaign_id
            return cls()

        def replicate(self) -> SimpleNamespace:
            persisted.append(gate_status)
            return SimpleNamespace(status=gate_status, summary="evidence retained")

    _install_runner(monkeypatch, FakeCampaignRunner)
    result = runner.invoke(
        app,
        ["replicate", "--campaign", "gate-outcome", "--config", str(CONFIG)],
    )

    assert result.exit_code == expected_exit_code
    assert persisted == [gate_status]
    assert f"replicate: {gate_status.value}" in result.output
    assert "evidence retained" in result.output


@pytest.mark.parametrize("invalid_date", ["31-01-2024", "20240131", "2024-02-30"])
def test_ingest_rejects_non_iso_date(
    monkeypatch: pytest.MonkeyPatch, invalid_date: str
) -> None:
    class FakeCampaignRunner:
        @classmethod
        def create(cls, *args: object, **kwargs: object) -> FakeCampaignRunner:
            raise AssertionError(
                "invalid dates must be rejected before runner creation"
            )

    _install_runner(monkeypatch, FakeCampaignRunner)
    result = runner.invoke(
        app,
        ["ingest", "--config", str(CONFIG), "--as-of", invalid_date],
    )

    assert result.exit_code == 2
    assert "expected an ISO date in YYYY-MM-DD format" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["report", "--campaign", "c1", "--config", str(CONFIG)],
        [
            "report",
            "--campaign",
            "c1",
            "--config",
            str(CONFIG),
            "--provisional",
            "--finalize-holdout",
        ],
        ["score", "--campaign", "c1", "--config", str(CONFIG)],
    ],
)
def test_governance_flags_fail_before_campaign_access(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)


def test_prior_gate_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCampaignRunner:
        @classmethod
        def open(cls, config: object, campaign_id: str) -> FakeCampaignRunner:
            del config, campaign_id
            return cls()

        def replicate(self) -> None:
            raise RuntimeError("campaign has not passed prerequisite data gate")

    _install_runner(monkeypatch, FakeCampaignRunner)
    result = runner.invoke(
        app,
        ["replicate", "--campaign", "blocked", "--config", str(CONFIG)],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "prerequisite data gate" in str(result.exception)


def test_live_demo_covers_restart_without_duplicate_logical_events(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        ["live-demo", "--database", str(tmp_path / "live-demo.sqlite")],
    )

    assert result.exit_code == 0
    assert "'sent': 6" in result.output
    assert "'receiver_unique': 6" in result.output
