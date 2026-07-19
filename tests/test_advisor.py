from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datetime import UTC, datetime
from pathlib import Path

import yaml

from edgestack.advisor import advise, validated_edge_tier
from edgestack.data.calendars import NYSECalendar
from edgestack.models import GateResult, GateStatus
from edgestack.storage.catalog import Catalog


@pytest.fixture(scope="module")
def planted_bars() -> pd.DataFrame:
    """Nine years of bars with Friday/November effects, all return overnight."""

    sessions = NYSECalendar().sessions("2015-01-02", "2023-12-29")
    rng = np.random.default_rng(20260716)
    returns = rng.normal(0.0002, 0.006, len(sessions))
    returns[sessions.dayofweek == 4] += 0.004
    returns[sessions.month == 11] += 0.002
    prices = 100.0 * np.cumprod(1.0 + returns)
    # The gap from the prior close carries the whole move; the session itself
    # is flat, making the closing auction the measurably better buy anchor.
    return pd.DataFrame(
        {
            "symbol": "GLD",
            "session": sessions,
            "open": prices,
            "close": prices,
            "adjusted_close": prices,
        }
    )


@pytest.fixture(scope="module")
def report(planted_bars: pd.DataFrame) -> dict:
    return advise(planted_bars, symbol="GLD")


def test_planted_friday_effect_is_a_family_significant_tailwind(
    report: dict,
) -> None:
    friday = report["all_conditions"]["weekday=FRI"]
    assert friday["classification"].startswith("TAILWIND_FAMILY_SIGNIFICANT")
    assert friday["bonferroni_p"] is not None and friday["bonferroni_p"] < 0.05
    assert report["tailwinds"][0]["name"] in {"weekday=FRI", "month=NOV"}


def test_every_condition_reports_its_dark_side(report: dict) -> None:
    for entry in report["all_conditions"].values():
        assert "regime_down_mean" in entry
        assert "regime_up_mean" in entry
        assert "max_drawdown_within_condition" in entry
        assert "worst_session" in entry


def test_weekly_window_buys_thursday_close_for_the_friday_effect(
    report: dict,
) -> None:
    week = report["timing"]["week"]
    assert week["best_target_session"] == "FRI"
    assert "THU" in week["buy"]
    assert week["worst_target_session"] != "FRI"
    assert "worst_time_to_buy" in week


def test_year_window_ranks_november_first(report: dict) -> None:
    year = report["timing"]["year"]
    assert year["best_month"] == "NOV"
    assert year["ranking"][0]["month"] == "NOV"
    assert year["worst_month"] != "NOV"


def test_combinations_are_cross_kind_and_counted_in_the_family(
    report: dict,
) -> None:
    assert report["combinations"], "strong planted effects must form pairs"
    for combo in report["combinations"]:
        left, right = combo["component_names"]
        assert report["all_conditions"][left]["kind"] != (
            report["all_conditions"][right]["kind"]
        )
        assert combo["verdict"] in {
            "INCREMENTAL_CANDIDATE",
            "NO_INCREMENTAL_VALUE",
            "GATING_TOO_THIN",
        }
        assert "component_overlap_fraction" in combo
    assert report["family_size_tested"] == len(report["all_conditions"]) + len(
        report["combinations"]
    )


def test_alignment_scan_scores_future_sessions(report: dict) -> None:
    alignment = report["alignment"]
    assert len(alignment["all_stars_aligned"]) == 5
    assert len(alignment["worst_sessions"]) == 5
    best = alignment["all_stars_aligned"][0]
    worst = alignment["worst_sessions"][0]
    assert best["alignment_score_daily"] >= worst["alignment_score_daily"]
    # Fridays dominate the planted tape, so the best future sessions are
    # Fridays.
    assert pd.Timestamp(best["session"]).dayofweek == 4


def test_buy_time_assessment_rates_a_friday_positive(
    planted_bars: pd.DataFrame,
) -> None:
    report = advise(
        planted_bars, symbol="GLD", buy_session=pd.Timestamp("2024-01-05").date()
    )
    assessment = report["buy_time_assessment"]
    assert assessment["buy_session"] == "2024-01-05"
    assert "weekday=FRI" in assessment["active_conditions"]
    assert assessment["overall_rating"] == "POSITIVE"
    assert assessment["if_you_trade_anyway"]


def test_buy_time_assessment_rejects_a_non_session(
    planted_bars: pd.DataFrame,
) -> None:
    report = advise(
        planted_bars, symbol="GLD", buy_session=pd.Timestamp("2024-01-06").date()
    )
    assert report["buy_time_assessment"]["status"] == "NOT_A_TRADING_SESSION"


def test_report_is_stamped_diagnostic_with_no_news_claim(report: dict) -> None:
    assert report["status"] == "DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER"
    assert report["news"]["status"] == "DATA_UNAVAILABLE"
    assert "intra-hour" in report["timing"]["execution"]["intraday_hours"]
    assert report["disclaimer"]
    assert any("SURVIVORSHIP" in item for item in report["provenance_warnings"])


def test_conditions_carry_ci_dsr_and_stability_fields(report: dict) -> None:
    friday = report["all_conditions"]["weekday=FRI"]
    assert friday["ci_lower_daily"] is not None and friday["ci_lower_daily"] > 0
    assert 0.0 <= friday["dsr_probability"] <= 1.0
    assert friday["dsr_probability"] > 0.9, "a strong planted edge survives DSR"
    assert friday["oos_sign_agreement"] is True
    assert friday["decayed_in_recent_third"] is False
    assert friday["classification"] == (
        "TAILWIND_FAMILY_SIGNIFICANT_CI_AND_OOS_CONFIRMED"
    )
    assert friday["reliability_weighted_daily"] == pytest.approx(
        friday["shrunk_mean_daily"] * friday["dsr_probability"]
    )


def test_weak_conditions_are_reliability_dampened(report: dict) -> None:
    for entry in report["all_conditions"].values():
        if entry["classification"] == "NEUTRAL":
            assert abs(entry["reliability_weighted_daily"]) <= abs(
                entry["shrunk_mean_daily"]
            ) + 1e-15


def test_validated_edge_tier_requires_passed_gates(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "spy-tom-edge-v1.yaml").write_text(
        yaml.safe_dump(
            {
                "campaign_id": "tom-test-001",
                "data": {"symbol": "SPY"},
                "strategy": {"name": "TOM"},
            }
        ),
        encoding="utf-8",
    )
    catalog = Catalog(tmp_path / "artifacts" / "edgestack.sqlite")
    catalog.create_campaign("tom-test-001", {"id": "tom-test-001"})
    tier = validated_edge_tier("SPY", root=tmp_path)
    assert tier["edges"][0]["status"] == "NOT_VALIDATED_GATES_ABSENT_OR_FAILED"
    for phase in ("edge_preholdout", "edge_holdout"):
        catalog.record_gate(
            GateResult(
                "tom-test-001", phase, GateStatus.PASS, datetime.now(UTC), "ok", {}
            )
        )
    tier = validated_edge_tier("SPY", root=tmp_path)
    edge = tier["edges"][0]
    assert edge["status"] == "VALIDATED_GATES_PASSED"
    assert edge["applies_to_symbol"] is True
    other = validated_edge_tier("GLD", root=tmp_path)
    assert other["edges"][0]["applies_to_symbol"] is False


def test_universe_scoped_edges_never_claim_offline_applicability(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "reversal-edge-v1.yaml").write_text(
        yaml.safe_dump(
            {
                "campaign_id": "rev-test-001",
                "data": {},
                "strategy": {"universe": "current S&P 500 equities"},
            }
        ),
        encoding="utf-8",
    )
    Catalog(tmp_path / "artifacts" / "edgestack.sqlite").create_campaign(
        "rev-test-001", {"id": "rev-test-001"}
    )
    tier = validated_edge_tier("GLD", root=tmp_path)
    assert tier["edges"][0]["applies_to_symbol"] == (
        "UNKNOWN_REQUIRES_CURRENT_MEMBERSHIP_CHECK"
    )


def test_validated_edge_tier_without_catalog_is_explicit(tmp_path: Path) -> None:
    assert validated_edge_tier("SPY", root=tmp_path)["status"] == (
        "NO_CAMPAIGN_CATALOG_AT_ROOT"
    )
    assert validated_edge_tier("SPY", root=None)["status"] == (
        "NOT_CHECKED_NO_ARTIFACT_ROOT"
    )


def test_anchor_assessment_detects_the_overnight_leg(report: dict) -> None:
    anchors = report["timing"]["anchors"]
    assert anchors["status"] == "TWO_ANCHORS_ONLY"
    assert anchors["legs"]["overnight"]["mean_daily"] > 0
    assert abs(anchors["legs"]["intraday"]["mean_daily"]) < 1e-12
    assert "CLOSE_AUCTION" in anchors["best_buy_anchor"]
    assert "OPEN_AUCTION" in anchors["matching_sell_anchor"]
    assert "DATA_UNAVAILABLE" in anchors["fifteen_minute_calendar"]


def test_hour_classification_maps_only_the_auctions() -> None:
    from edgestack.advisor import _classify_hour

    assert _classify_hour("09:45") == (
        "OPEN_AUCTION",
        "REQUESTED_TIME_IS_AN_EXECUTABLE_ANCHOR",
    )
    assert _classify_hour("15:50")[0] == "CLOSE_AUCTION"
    assert _classify_hour("12:00") == (
        None,
        "MID_SESSION_UNRATED_ONLY_THE_AUCTIONS_ARE_MEASURED",
    )
    assert _classify_hour("08:00")[1].startswith("PREMARKET")
    assert _classify_hour("18:00")[1].startswith("AFTER_HOURS")
    with pytest.raises(ValueError, match="HH:MM"):
        _classify_hour("noonish")


def test_choice_review_ranks_the_day_and_redirects_the_hour(
    planted_bars: pd.DataFrame,
) -> None:
    report = advise(
        planted_bars,
        symbol="GLD",
        buy_session=pd.Timestamp("2024-01-03").date(),  # a Wednesday
        buy_hour="12:00",
    )
    review = report["buy_time_assessment"]["choice_review"]
    rank = review["weekday_rank"]
    assert rank["chosen_weekday"] == "WED"
    assert rank["rank_of_5"] > 1, "Wednesday cannot outrank the planted Friday"
    assert "FRI" in rank["better_weekdays"]
    hour = review["hour_review"]
    assert hour["verdict"] == "MID_SESSION_UNRATED_ONLY_THE_AUCTIONS_ARE_MEASURED"
    assert "CLOSE_AUCTION" in hour["recommended_buy_anchor"]
    assert "DATA_UNAVAILABLE" in hour["finer_than_anchors"]
    plan = review["sell_plan_by_horizon"]
    assert set(plan) == {"day", "week", "month", "year", "always"}
    assert "OPEN_AUCTION" in plan["day"]["exit"]
    assert len(review["revalidation_schedule"]) == 3
    assert isinstance(review["better_upcoming_sessions"], list)


def test_alignment_calendar_rows_carry_win_scores(report: dict) -> None:
    calendar = report["alignment"]["calendar"]
    assert len(calendar) == report["alignment"]["sessions_scanned"]
    fridays = [row for row in calendar if row["weekday"] == "FRI"]
    others = [row for row in calendar if row["weekday"] != "FRI"]
    assert fridays and others
    for row in calendar:
        assert 0 <= row["win_score_0_100"] <= 100
        assert row["expected_daily_bp"] == pytest.approx(
            row["alignment_score_daily"] * 10_000.0
        )
    best_friday = max(row["win_score_0_100"] for row in fridays)
    median_other = sorted(row["win_score_0_100"] for row in others)[
        len(others) // 2
    ]
    assert best_friday > median_other


def test_alignment_scan_does_not_mislabel_a_partial_first_month(
    planted_bars: pd.DataFrame,
) -> None:
    # As-of mid-November: the first scanned sessions are mid-month and must
    # NOT be flagged turn-of-month just because the future index starts there.
    report = advise(planted_bars, symbol="GLD", as_of=pd.Timestamp("2023-11-15").date())
    calendar = report["alignment"]["calendar"]
    first_rows = [row for row in calendar if row["session"] <= "2023-11-27"]
    assert first_rows
    for row in first_rows:
        assert "turn_of_month" not in row["active_calendar_conditions"], row
    tom_rows = [
        row
        for row in calendar
        if "turn_of_month" in row["active_calendar_conditions"]
    ]
    # The genuine window (last session of November + first three of December)
    # is still found.
    assert {row["session"] for row in tom_rows} & {
        "2023-11-30", "2023-12-01", "2023-12-04", "2023-12-05"
    }


def test_advisor_requires_a_year_of_history() -> None:
    sessions = NYSECalendar().sessions("2023-01-03", "2023-06-30")
    bars = pd.DataFrame(
        {
            "symbol": "GLD",
            "session": sessions,
            "adjusted_close": np.linspace(100.0, 110.0, len(sessions)),
        }
    )
    with pytest.raises(ValueError, match="one year"):
        advise(bars, symbol="GLD")
