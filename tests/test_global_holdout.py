from __future__ import annotations

import pytest

from edgestack.edges.global_holdout import GlobalHoldoutLedger, global_scope_id


def test_global_scope_cannot_be_reopened_by_another_freeze(tmp_path) -> None:
    ledger = GlobalHoldoutLedger(tmp_path / "catalog.sqlite")
    scope = global_scope_id(
        program_id="program",
        market="XNYS",
        promotion_class="FINAL",
        start="2023-01-01",
        end="2025-12-31",
    )
    ledger.register(
        scope_id=scope,
        program_id="program",
        market="XNYS",
        promotion_class="FINAL",
        data_snapshot_id="snapshot",
        start="2023-01-01",
        end="2025-12-31",
    )
    consumed = ledger.consume(
        scope_id=scope, freeze_id="freeze-a", evaluator_sha256="evaluator"
    )
    assert consumed.state == "CONSUMED"
    with pytest.raises(RuntimeError, match="already consumed"):
        ledger.consume(
            scope_id=scope, freeze_id="freeze-b", evaluator_sha256="evaluator"
        )


def test_seal_is_single_state_transition(tmp_path) -> None:
    ledger = GlobalHoldoutLedger(tmp_path / "catalog.sqlite")
    scope = global_scope_id(
        program_id="program",
        market="XNYS",
        promotion_class="FINAL",
        start="2023-01-01",
        end="2025-12-31",
    )
    ledger.register(
        scope_id=scope,
        program_id="program",
        market="XNYS",
        promotion_class="FINAL",
        data_snapshot_id="snapshot",
        start="2023-01-01",
        end="2025-12-31",
    )
    ledger.consume(scope_id=scope, freeze_id="freeze", evaluator_sha256="eval")
    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")
    sealed = ledger.seal(
        scope_id=scope,
        freeze_id="freeze",
        result_sha256="result-hash",
        result_path=result,
    )
    assert sealed.state == "SEALED"
    assert sealed.result_sha256 == "result-hash"
    with pytest.raises(RuntimeError, match="consumed"):
        ledger.seal(
            scope_id=scope,
            freeze_id="freeze",
            result_sha256="different",
            result_path=result,
        )


def test_same_economic_window_cannot_be_reregistered_with_new_data_copy(
    tmp_path,
) -> None:
    ledger = GlobalHoldoutLedger(tmp_path / "catalog.sqlite")
    scope = global_scope_id(
        program_id="program",
        market="XNYS",
        promotion_class="FINAL",
        start="2023-01-01",
        end="2025-12-31",
    )
    ledger.register(
        scope_id=scope,
        program_id="program",
        market="XNYS",
        promotion_class="FINAL",
        data_snapshot_id="snapshot-a",
        start="2023-01-01",
        end="2025-12-31",
    )
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        ledger.register(
            scope_id=scope,
            program_id="program",
            market="XNYS",
            promotion_class="FINAL",
            data_snapshot_id="snapshot-b",
            start="2023-01-01",
            end="2025-12-31",
        )
