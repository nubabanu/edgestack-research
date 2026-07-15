from __future__ import annotations

import html as html_module
from datetime import date
from pathlib import Path

import pytest

from edgestack.config import EdgeStackConfig
from edgestack.disclaimer import DISCLAIMER
from edgestack.models import GateStatus
from edgestack.pipeline.runner import CampaignRunner


def test_failed_gate_emits_self_contained_html_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CampaignRunner.create(
        EdgeStackConfig(), campaign_id="diagnostic-test", as_of=date(2020, 1, 2)
    )

    result = runner._record_gate(
        "data", False, "frozen empirical miss", {"threshold_changed": False}
    )

    assert result.status is GateStatus.FAIL
    html_path = runner.campaign_root / "diagnostics/data_report.html"
    csv = runner.campaign_root / "diagnostics/data_report.csv"
    assert DISCLAIMER in html_module.unescape(html_path.read_text(encoding="utf-8"))
    assert DISCLAIMER in csv.read_text(encoding="utf-8")
    assert "frozen empirical miss" in html_path.read_text(encoding="utf-8")
