from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from edgestack.mobile.api import create_mobile_app
from edgestack.mobile.models import MobileSnapshot
from edgestack.mobile.service import MobileSnapshotService, SnapshotUnavailableError

TOKEN = "test-mobile-token-with-24-characters"


def test_demo_snapshot_is_strict_and_visibly_non_live(tmp_path: Path) -> None:
    snapshot = MobileSnapshotService(tmp_path, demo=True).load()
    assert snapshot.meta.mode == "DEMO"
    assert snapshot.meta.stale is True
    assert snapshot.model_status == "DEMO"
    assert snapshot.bias_tier == "SURVIVORSHIP_BIASED"
    assert "DEMO" in snapshot.watermark
    assert [item.rank for item in snapshot.recommendations] == [1, 2, 3, 4, 5]
    assert snapshot.portfolio.shorts_enabled is False


def test_mobile_api_requires_constant_bearer_and_sets_evidence_headers(
    tmp_path: Path,
) -> None:
    client = TestClient(
        create_mobile_app(
            artifact_root=tmp_path,
            bearer_token=TOKEN,
            demo=True,
        )
    )
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/mobile/snapshot").status_code == 401
    assert (
        client.get(
            "/api/v1/mobile/snapshot",
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )
    response = client.get(
        "/api/v1/mobile/snapshot",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200
    assert response.headers["etag"].startswith('"')
    assert response.headers["cache-control"] == "private, no-cache"
    assert response.json()["meta"]["schema_version"] == "1.0"
    assert "/orders" not in client.app.openapi()["paths"]


def test_non_demo_api_rejects_missing_or_short_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"24\+"):
        create_mobile_app(artifact_root=tmp_path, demo=False)
    with pytest.raises(ValueError, match=r"24\+"):
        create_mobile_app(
            artifact_root=tmp_path,
            bearer_token="short",
            demo=False,
        )


def test_campaign_identifier_cannot_traverse_artifact_root(tmp_path: Path) -> None:
    service = MobileSnapshotService(tmp_path, campaign_id="../outside")
    with pytest.raises(SnapshotUnavailableError, match="invalid campaign"):
        service.load()


def test_mobile_model_rejects_reordered_or_partial_rank_sequence(
    tmp_path: Path,
) -> None:
    payload = MobileSnapshotService(tmp_path, demo=True).load().model_dump(mode="json")
    payload["recommendations"][0]["rank"] = 2
    with pytest.raises(ValidationError, match="contiguous"):
        MobileSnapshot.model_validate(payload)


def test_promoted_model_requires_passed_holdout(tmp_path: Path) -> None:
    payload = MobileSnapshotService(tmp_path, demo=True).load().model_dump(mode="json")
    payload["model_status"] = "PROMOTED"
    payload["holdout"]["status"] = "FAIL"
    with pytest.raises(ValidationError, match="passed holdout"):
        MobileSnapshot.model_validate(payload)


def test_sealed_campaign_artifacts_are_verified_and_normalized(tmp_path: Path) -> None:
    campaign = tmp_path / "campaigns" / "sealed-001"
    holdout_dir = campaign / "holdout"
    live_dir = campaign / "live"
    holdout_dir.mkdir(parents=True)
    live_dir.mkdir()
    (holdout_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "holdout_pass": True,
                "second_evaluation": "FORBIDDEN_REPLAY_ONLY",
                "holdout_start": "2023-01-01",
                "holdout_end": "2025-12-31",
                "observations": 752,
                "expected_sessions": 752,
                "net_mean": 0.001,
                "benchmark_excess_mean": 0.0004,
                "terminal_net_wealth": 1.4,
                "terminal_benchmark_wealth": 1.2,
                "freeze_id": "freeze-001",
            }
        ),
        encoding="utf-8",
    )
    (live_dir / "2026-01-02-signal.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-01-02T21:00:00Z",
                "market_as_of": "2026-01-02_CLOSE",
                "bias_tier": "POINT_IN_TIME",
                "strategy": "TESTED_REVERSAL_BASKET",
                "data": {"source": "fixture"},
                "entry": {
                    "session": "2026-01-05",
                    "planned_submission_time": "15:45 America/New_York",
                    "no_chase": "Wait for the next completed-close scan.",
                    "cancel_if": ["quote is stale"],
                },
                "exit": {"session": "2026-01-12"},
                "portfolio": {
                    "paper_capital_usd": 100000,
                    "tested_new_account_gross_target": 0.5,
                    "tested_maximum_weight_per_name": 0.1,
                    "paper_risk_budget_per_name_usd": 500,
                },
                "candidates": [
                    {
                        "recommendation_id": "rec-001",
                        "rank": 1,
                        "symbol": "TEST",
                        "direction": "LONG",
                        "confidence_ordinal_not_probability": 75,
                        "signal_close_usd": 100,
                        "trailing_5_session_return": -0.1,
                        "risk_capped_reference_shares": 10,
                        "two_atr_reference_price_usd": 90,
                        "event_risk": "FIXTURE_ONLY",
                    }
                ],
                "shorts": [],
                "interpretation": "The basket is evaluated as a whole.",
            }
        ),
        encoding="utf-8",
    )

    snapshot = MobileSnapshotService(tmp_path, campaign_id="sealed-001").load()

    assert snapshot.model_status == "PROMOTED"
    assert snapshot.bias_tier == "POINT_IN_TIME"
    assert snapshot.holdout.status == "PASS"
    assert snapshot.holdout.result_sha256
    assert snapshot.recommendations[0].symbol == "TEST"
