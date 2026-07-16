"""Crash-safe, gate-enforced EdgeStack campaign orchestration."""

from __future__ import annotations

import ast
import base64
import html
import json
import math
import re
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import pandas as pd

from edgestack.backtest.costs import CostModel
from edgestack.backtest.engine import vectorized_backtest
from edgestack.config import EdgeStackConfig, dump_resolved_config
from edgestack.data.cache import DataCache
from edgestack.data.calendars import NYSECalendar
from edgestack.data.quality import write_qa_report
from edgestack.disclaimer import DISCLAIMER
from edgestack.entrytiming.indicators import bollinger_pct_b, rsi
from edgestack.entrytiming.interaction_tests import (
    OverlayDecision,
    OverlayEvidence,
    interaction_decision,
)
from edgestack.evaluation.report import render_verdict_report
from edgestack.features.cross_sectional import canonical_features, decile_weights
from edgestack.models import (
    GateResult,
    GateStatus,
    HoldoutFreezeManifest,
    HypothesisSpec,
)
from edgestack.pipeline.campaign_data import (
    acquire_campaign_data,
    memberships_frame,
    synthetic_replication_inputs,
)
from edgestack.pipeline.gates import Gatekeeper
from edgestack.pipeline.holdout import HoldoutGuard
from edgestack.pipeline.research import (
    canonical_spec_payload,
    prepare_research,
    run_discovery,
    run_trial,
    spec_from_dict,
)
from edgestack.pipeline.validation_run import (
    final_records,
    records_from_payload,
    run_validation,
    serialize_records,
)
from edgestack.provenance import (
    canonical_sha256,
    runtime_manifest,
    sha256_file,
    source_tree_sha256,
)
from edgestack.reversal.dataset import build_cross_sectional_dataset
from edgestack.reversal.portfolio import run_reversal_grid
from edgestack.reversal.study import default_model_specs, run_model_study
from edgestack.scoring.stacking import StackResult, build_stack
from edgestack.stats.bootstrap import stationary_bootstrap_ci
from edgestack.stats.deflated_sharpe import deflated_sharpe_ratio
from edgestack.stats.multiple_testing import benjamini_hochberg
from edgestack.stats.tests import hac_mean_test
from edgestack.storage.artifacts import ArtifactStore
from edgestack.storage.catalog import Catalog
from edgestack.validation.cpcv import cpcv_pbo
from edgestack.validation.replication import run_replication_suite
from edgestack.validation.walkforward import expanding_walk_forward


class CampaignRunner:
    """Execute one immutable campaign through its persisted phase gates."""

    def __init__(
        self,
        config: EdgeStackConfig,
        campaign_id: str,
        *,
        as_of: date,
        holdout_start: date,
        workspace: Path,
    ) -> None:
        self.config = config
        self.campaign_id = campaign_id
        self.as_of = as_of
        self.holdout_start = holdout_start
        self.workspace = workspace
        root = _resolve(workspace, config.paths.root)
        self.raw_root = _resolve(root, config.paths.raw)
        self.canonical_root = _resolve(root, config.paths.canonical)
        self.artifacts_root = _resolve(root, config.paths.artifacts)
        self.catalog_path = _resolve(root, config.paths.catalog)
        self.campaign_root = self.artifacts_root / "campaigns" / campaign_id
        self.store = ArtifactStore(self.campaign_root)
        self.catalog = Catalog(self.catalog_path)
        self.gates = Gatekeeper(self.catalog, campaign_id)
        self.cache = DataCache(
            raw_root=self.raw_root,
            canonical_root=self.canonical_root,
            catalog_path=self.catalog_path,
        )

    @classmethod
    def create(
        cls,
        config: EdgeStackConfig,
        *,
        campaign_id: str | None = None,
        as_of: date | None = None,
    ) -> Self:
        """Register a campaign and freeze dates, seed, source, and lock identity."""

        workspace = Path.cwd().resolve()
        selected_as_of = _last_completed_session(as_of or config.as_of)
        holdout_start = (
            pd.Timestamp(selected_as_of)
            - pd.DateOffset(years=config.data.holdout_years)
        ).date()
        config_text = dump_resolved_config(config)
        config_sha = canonical_sha256(config.model_dump(mode="json"))
        identifier = campaign_id or (
            f"{config.profile}-{selected_as_of:%Y%m%d}-"
            f"{datetime.now(UTC):%H%M%S}-{config_sha[:8]}"
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", identifier):
            raise ValueError("campaign_id must be 3-128 safe filename characters")
        runner = cls(
            config,
            identifier,
            as_of=selected_as_of,
            holdout_start=holdout_start,
            workspace=workspace,
        )
        lock_path = workspace / "uv.lock"
        manifest = {
            "campaign_id": identifier,
            "created_at": datetime.now(UTC).isoformat(),
            "profile": config.profile,
            "as_of": selected_as_of.isoformat(),
            "holdout_start": holdout_start.isoformat(),
            "config_sha256": config_sha,
            "data_snapshot_id": "PENDING_INGEST",
            "source_tree_sha256": source_tree_sha256(workspace),
            "lock_sha256": sha256_file(lock_path) if lock_path.exists() else "MISSING",
            "seed": config.stats.seed,
            "evidence_protocol": config.protocol.model_dump(mode="json"),
            "non_promotable": config.profile == "smoke",
            "disclaimer": DISCLAIMER,
        }
        runner.catalog.create_campaign(identifier, manifest)
        runner._write_json("manifest", "manifest.json", manifest)
        runner._write_json(
            "resolved_config",
            "resolved_config.json",
            config.model_dump(mode="json"),
        )
        runner._write_json("runtime", "runtime.json", runtime_manifest())
        (runner.campaign_root / "resolved_config.yaml").write_text(
            config_text, encoding="utf-8"
        )
        return runner

    @classmethod
    def open(cls, config: EdgeStackConfig, campaign_id: str) -> Self:
        """Open a registered campaign and reject configuration drift."""

        workspace = Path.cwd().resolve()
        root = _resolve(workspace, config.paths.root)
        catalog = Catalog(_resolve(root, config.paths.catalog))
        manifest = catalog.campaign(campaign_id)
        if manifest is None:
            raise KeyError(f"unknown campaign: {campaign_id}")
        expected = canonical_sha256(config.model_dump(mode="json"))
        legacy_payload = config.model_dump(mode="json")
        legacy_payload.pop("protocol", None)
        legacy_expected = canonical_sha256(legacy_payload)
        legacy_match = (
            config.protocol.version == "FROZEN_V1"
            and "evidence_protocol" not in manifest
            and manifest.get("config_sha256") == legacy_expected
        )
        extension_payload = config.model_dump(mode="json")
        extension_payload.pop("reversal", None)
        extension_expected = canonical_sha256(extension_payload)
        extension_match = manifest.get("config_sha256") == extension_expected
        if (
            manifest.get("config_sha256") != expected
            and not legacy_match
            and not extension_match
        ):
            raise RuntimeError(
                "campaign config hash differs; use the exact frozen config or create a new campaign"
            )
        return cls(
            config,
            campaign_id,
            as_of=date.fromisoformat(str(manifest["as_of"])),
            holdout_start=date.fromisoformat(str(manifest["holdout_start"])),
            workspace=workspace,
        )

    def ingest(self) -> GateResult:
        """Acquire/cache data, persist provenance, and evaluate the data gate."""

        prior = self._prior("data")
        if prior is not None:
            return prior
        try:
            acquired = acquire_campaign_data(
                self.config,
                as_of=self.as_of,
                cache=self.cache,
            )
            self._write_parquet("canonical_bars", "data/bars.parquet", acquired.bars)
            self._write_parquet(
                "reference_factors", "data/factors.parquet", acquired.factors
            )
            self._write_parquet(
                "universe_memberships",
                "data/universe.parquet",
                memberships_frame(acquired.memberships),
            )
            qa_path = self.campaign_root / "data" / "qa.json"
            qa_path.parent.mkdir(parents=True, exist_ok=True)
            qa_sha = write_qa_report(acquired.qa, qa_path)
            self.catalog.record_artifact(self.campaign_id, "data_qa", qa_sha, qa_path)
            payload = acquired.evidence()
            payload["fomc_dates"] = [
                item.date().isoformat() for item in acquired.fomc_dates
            ]
            self._write_json("data_manifest", "data/manifest.json", payload)
            passed = acquired.passed
            summary = (
                "deterministic synthetic smoke data passed mechanical QA; this campaign is non-promotable"
                if passed and acquired.non_promotable
                else (
                    "full provider-backed data and reconciliation gate passed"
                    if passed
                    else "data gate failed; diagnostics were frozen and promotion stopped"
                )
            )
            return self._record_gate("data", passed, summary, payload)
        except Exception as error:
            evidence = self._diagnostic("data", error)
            return self._record_gate(
                "data",
                False,
                "data acquisition or QA could not complete",
                evidence,
                blocked=True,
            )

    def replicate(self) -> GateResult:
        """Run all six frozen empirical or smoke-fixture replication checks."""

        prior = self._prior("replication")
        if prior is not None:
            return prior
        self.gates.require_previous("replication")
        try:
            bars, factors, manifest = self._load_data()
            fomc_dates = pd.DatetimeIndex(manifest.get("fomc_dates", []))
            if self.config.profile == "smoke":
                from edgestack.pipeline.campaign_data import IngestedCampaignData

                # Rehydrate only the fields used by the deterministic input helper.
                surrogate = IngestedCampaignData(
                    bars,
                    factors,
                    fomc_dates,
                    (),
                    _empty_qa(),
                    str(manifest["snapshot_id"]),
                    {},
                    (),
                    {},
                    True,
                    True,
                    True,
                    True,
                    True,
                )
                inputs = synthetic_replication_inputs(surrogate)
            else:
                inputs = self._full_replication_inputs(bars, factors, fomc_dates)
            suite = run_replication_suite(**inputs)
            empirical_pass_count = sum(check.passed for check in suite.checks)
            diagnostic_policy = (
                self.config.protocol.replication_policy
                == "EXECUTION_WITH_EMPIRICAL_DIAGNOSTICS"
            )
            promotion_gate_pass = suite.passed or (diagnostic_policy and suite.executed)
            evidence = {
                "checks": [asdict(item) for item in suite.checks],
                "failures": list(suite.failures),
                "all_six_pass": suite.passed,
                "empirical_pass_count": empirical_pass_count,
                "empirical_check_count": len(suite.checks),
                "all_checks_executed": suite.executed,
                "promotion_gate_pass": promotion_gate_pass,
                "protocol_version": self.config.protocol.version,
                "replication_policy": self.config.protocol.replication_policy,
                "revision_context": self.config.protocol.revision_context,
                "interpretation": (
                    "Empirical misses remain adverse evidence but do not imply "
                    "that the data/research engine failed to execute."
                    if diagnostic_policy
                    else "Every frozen empirical check is required for promotion."
                ),
                "profile_scope": (
                    "SYNTHETIC_SMOKE_NON_PROMOTABLE"
                    if self.config.profile == "smoke"
                    else "FULL_EMPIRICAL"
                ),
                "disclaimer": DISCLAIMER,
            }
            self._write_json("replication", "replication/evidence.json", evidence)
            return self._record_gate(
                "replication",
                promotion_gate_pass,
                (
                    "all six frozen replication checks passed"
                    if suite.passed
                    else (
                        f"all six checks executed; {empirical_pass_count}/6 matched and "
                        "the miss remains diagnostic under the versioned literature protocol"
                        if diagnostic_policy
                        else "one or more frozen replication checks missed; no retuning was performed"
                    )
                ),
                evidence,
            )
        except Exception as error:
            evidence = self._diagnostic("replication", error)
            return self._record_gate(
                "replication",
                False,
                "replication execution failed and was frozen as a diagnostic",
                evidence,
                blocked=True,
            )

    def discover(self) -> GateResult:
        """Enumerate, cost, test, and persist the real/control hypothesis family."""

        prior = self._prior("discovery")
        if prior is not None:
            return prior
        self.gates.require_previous("discovery")
        try:
            bars, _, manifest = self._load_data()
            prepared = self._prepared(bars, manifest, end=self.holdout_start)
            bundle = run_discovery(prepared, self.config)
            self._write_parquet(
                "discovery_metrics", "discovery/metrics.parquet", bundle.metrics
            )
            self._write_parquet(
                "discovery_returns",
                "discovery/net_returns.parquet",
                bundle.net_returns.rename_axis("session").reset_index(),
            )
            self._write_parquet(
                "discovery_gross_returns",
                "discovery/gross_returns.parquet",
                bundle.gross_returns.rename_axis("session").reset_index(),
            )
            self._write_json(
                "hypothesis_registry",
                "discovery/specs.json",
                canonical_spec_payload(bundle.specs),
            )
            frozen_gate_pass = (
                bundle.survivor_fraction_t_fdr
                <= self.config.stats.survivor_fraction_max
            )
            passed = frozen_gate_pass or self.config.profile == "smoke"
            evidence = {
                "declared_trials": len(bundle.specs),
                "real_trials": sum(spec.placebo_kind is None for spec in bundle.specs),
                "placebo_trials": sum(
                    spec.placebo_kind is not None for spec in bundle.specs
                ),
                "survivors": list(bundle.survivor_ids),
                "survivor_count": len(bundle.survivor_ids),
                "t_plus_fdr_survivor_fraction": bundle.survivor_fraction_t_fdr,
                "survivor_fraction_max": self.config.stats.survivor_fraction_max,
                "frozen_empirical_gate_pass": frozen_gate_pass,
                "smoke_mechanical_override": self.config.profile == "smoke"
                and not frozen_gate_pass,
                "provisional_placebo_fraction": bundle.provisional_placebo_fraction,
                "spa_p_value": bundle.spa_p_value,
                "white_reality_check_p_value": bundle.reality_check_p_value,
                "romano_wolf_rejection_count": bundle.romano_wolf_rejection_count,
                "romano_wolf_method": bundle.romano_wolf_method,
                "romano_wolf_required": self.config.protocol.require_romano_wolf,
                "romano_wolf_alpha": self.config.protocol.romano_wolf_alpha,
                "time_series_t_threshold": (
                    self.config.stats.hard_t
                    if self.config.protocol.version == "FROZEN_V1"
                    else self.config.protocol.time_series_t_threshold
                ),
                "cross_sectional_t_threshold": (
                    self.config.stats.hard_t
                    if self.config.protocol.version == "FROZEN_V1"
                    else self.config.protocol.cross_sectional_t_threshold
                ),
                "evidence_protocol": self.config.protocol.version,
                "empty_survivor_outcome": not bundle.survivor_ids,
                "holdout_excluded_from_research": True,
                "research_end_exclusive": self.holdout_start.isoformat(),
                "disclaimer": DISCLAIMER,
            }
            self._write_json("discovery_summary", "discovery/summary.json", evidence)
            return self._record_gate(
                "discovery",
                passed,
                (
                    "discovery completed within the frozen survivor-fraction guard"
                    if frozen_gate_pass
                    else (
                        "synthetic smoke completed; empirical survivor-fraction result is diagnostic only"
                        if self.config.profile == "smoke"
                        else "more than 5% survived t+FDR; campaign stopped for diagnostic audit"
                    )
                ),
                evidence,
            )
        except Exception as error:
            evidence = self._diagnostic("discovery", error)
            return self._record_gate(
                "discovery",
                False,
                "discovery execution failed and was frozen as a diagnostic",
                evidence,
                blocked=True,
            )

    def reversal_research(
        self, *, run_ml: bool = False, use_gpu: bool = False
    ) -> GateResult:
        """Run the opt-in top-K/reversal-model study without opening holdout data."""

        if run_ml:
            rule_result = self.reversal_research(run_ml=False)
            if rule_result.status is not GateStatus.PASS:
                return rule_result
            return self._reversal_model_study(use_gpu=use_gpu)
        version = self.config.reversal.study_version
        phase = f"reversal_research_{version}"
        artifact_root = f"reversal/{version}"
        prior = self._prior(phase)
        if prior is not None:
            return prior
        self.catalog.require_passed(self.campaign_id, ("data", "replication"))
        if not self.config.reversal.enabled:
            raise RuntimeError("reversal research is not enabled in the frozen config")
        try:
            bars, _, manifest = self._load_data()
            prepared = self._prepared(bars, manifest, end=self.holdout_start)
            grid = run_reversal_grid(
                prepared,
                self.config.reversal,
                cost_model=CostModel(self.config.costs),
                membership=None,
                minimum_observations=self.config.grid.min_observations,
                directed_t_threshold=(
                    3.0
                    if self.config.protocol.version == "FROZEN_V1"
                    else self.config.protocol.cross_sectional_t_threshold
                ),
                fdr_q=self.config.stats.fdr_q,
                dsr_threshold=self.config.stats.dsr_probability,
                cpcv_groups=self.config.validation.cpcv_groups,
                cpcv_test_groups=self.config.validation.cpcv_test_groups,
                purge=self.config.validation.purge_sessions,
                embargo=self.config.validation.embargo_sessions,
                min_train_years=self.config.validation.min_train_years,
                test_years=self.config.validation.test_years,
                step_years=self.config.validation.step_years,
                oos_t_threshold=self.config.validation.oos_t,
                required_positive_fraction=(
                    self.config.validation.oos_positive_fraction
                ),
                rolling_years=self.config.validation.rolling_years,
                stability_min=self.config.validation.stability_min,
                cost_multipliers=self.config.costs.sensitivity_multipliers,
            )
            self._write_parquet(
                f"reversal_grid_metrics_{version}",
                f"{artifact_root}/metrics.parquet",
                grid.metrics,
            )
            self._write_parquet(
                f"reversal_grid_net_returns_{version}",
                f"{artifact_root}/net_returns.parquet",
                grid.net_returns.rename_axis("session").reset_index(),
            )
            self._write_parquet(
                f"reversal_grid_gross_returns_{version}",
                f"{artifact_root}/gross_returns.parquet",
                grid.gross_returns.rename_axis("session").reset_index(),
            )
            self._write_json(
                f"reversal_grid_specs_{version}",
                f"{artifact_root}/specs.json",
                canonical_spec_payload(grid.specs),
            )
            research_years = pd.DatetimeIndex(grid.net_returns.index).year
            annual_returns = (
                grid.net_returns.groupby(research_years)
                .agg(_compounded_return)
                .rename_axis("year")
                .reset_index()
            )
            self._write_parquet(
                f"reversal_grid_annual_returns_{version}",
                f"{artifact_root}/annual_returns.parquet",
                annual_returns,
            )
            model_summary: dict[str, Any] = {"executed": False}
            evidence = {
                "study_version": version,
                "declared_top_k": list(self.config.reversal.top_k),
                "declared_variants": list(self.config.reversal.variants),
                "declared_grid_trials": grid.trial_count,
                "discovery_survivors": int(grid.metrics["passes_discovery"].sum()),
                "rule_validation_survivors": int(
                    grid.metrics["passes_rule_validation"].sum()
                ),
                "candidate_family_pbo": grid.pbo.pbo,
                "pbo_defined": grid.pbo.defined,
                "pbo_by_side": {
                    side: {
                        "pbo": result.pbo,
                        "defined": result.defined,
                        "splits": result.n_splits,
                        "reason": result.reason,
                    }
                    for side, result in grid.pbo_by_side.items()
                },
                "bias_tier": grid.bias_tier,
                "holdout_excluded_from_research": True,
                "research_end_exclusive": self.holdout_start.isoformat(),
                "analysis_provenance": self._reversal_analysis_provenance(manifest),
                "model_study": model_summary,
                "promotion_eligible": False,
                "promotion_blockers": [
                    (
                        "current-membership universe is survivorship biased"
                        if grid.bias_tier == "SURVIVORSHIP_BIASED"
                        else "outer validation and untouched holdout remain required"
                    ),
                    "historical 15:45 quotes, earnings availability, and borrow data are absent",
                ],
                "disclaimer": DISCLAIMER,
            }
            self._write_json(
                f"reversal_research_summary_{version}",
                f"{artifact_root}/summary.json",
                evidence,
            )
            return self._record_gate(
                phase,
                True,
                "selection-aware reversal study executed as a non-promotable diagnostic",
                evidence,
            )
        except Exception as error:
            evidence = self._diagnostic(phase, error)
            return self._record_gate(
                phase,
                False,
                "reversal research execution failed and was frozen as a diagnostic",
                evidence,
                blocked=True,
            )

    def _reversal_model_study(self, *, use_gpu: bool) -> GateResult:
        """Run the separately gated purged ML diagnostics after the rule grid."""

        version = self.config.reversal.study_version
        phase = f"reversal_model_study_{version}"
        artifact_root = f"reversal/{version}"
        prior = self._prior(phase)
        if prior is not None:
            return prior
        self.catalog.require_passed(
            self.campaign_id,
            ("data", "replication", f"reversal_research_{version}"),
        )
        try:
            bars, _, manifest = self._load_data()
            prepared = self._prepared(bars, manifest, end=self.holdout_start)
            dataset = build_cross_sectional_dataset(
                prepared,
                self.config.reversal,
                membership=None,
                short_borrow_annual=self.config.costs.easy_borrow_annual,
            )
            specs = default_model_specs(dataset, seed=self.config.stats.seed)
            study = run_model_study(
                dataset,
                specs,
                catalog=self.catalog,
                study_id=f"{self.campaign_id}-reversal-ml-{version}",
                use_gpu=use_gpu,
                gpu_devices=self.config.reversal.gpu_devices,
                min_train_years=self.config.validation.min_train_years,
                test_years=self.config.validation.test_years,
                step_years=self.config.validation.step_years,
                purge_sessions=max(
                    self.config.reversal.holding_sessions,
                    self.config.validation.purge_sessions,
                ),
                diagnostic_top_k=5,
            )
            persisted_metrics = study.metrics.copy()
            if "folds" in persisted_metrics:
                persisted_metrics["folds"] = persisted_metrics["folds"].map(
                    lambda value: json.dumps(value, sort_keys=True, default=str)
                )
            self._write_parquet(
                f"reversal_model_metrics_{version}",
                f"{artifact_root}/model_metrics.parquet",
                persisted_metrics,
            )
            evidence = {
                "study_version": version,
                "executed": True,
                "device_mode": "GPU" if use_gpu else "CPU",
                "declared_trials": len(study.declared_trial_ids),
                "failed_trials": list(study.failed_trial_ids),
                "bias_tier": study.bias_tier,
                "holdout_excluded_from_research": True,
                "analysis_provenance": self._reversal_analysis_provenance(manifest),
                "promotion_eligible": False,
                "reason": "rank diagnostics require a causal portfolio backtest",
                "disclaimer": DISCLAIMER,
            }
            self._write_json(
                f"reversal_model_summary_{version}",
                f"{artifact_root}/model_summary.json",
                evidence,
            )
            return self._record_gate(
                phase,
                not study.failed_trial_ids,
                "purged reversal model diagnostics completed",
                evidence,
            )
        except Exception as error:
            evidence = self._diagnostic(phase, error)
            return self._record_gate(
                phase,
                False,
                "reversal model study failed and was frozen as a diagnostic",
                evidence,
                blocked=True,
            )

    def _reversal_analysis_provenance(
        self, data_manifest: dict[str, Any]
    ) -> dict[str, str]:
        """Hash the extension code/config/lock and immutable parent data inputs."""

        lock_path = self.workspace / "uv.lock"
        parent = self.catalog.campaign(self.campaign_id) or {}
        return {
            "parent_campaign_config_sha256": str(parent.get("config_sha256", "")),
            "analysis_config_sha256": canonical_sha256(
                self.config.model_dump(mode="json")
            ),
            "reversal_protocol_sha256": canonical_sha256(
                self.config.reversal.model_dump(mode="json")
            ),
            "source_tree_sha256": source_tree_sha256(self.workspace),
            "lock_sha256": (
                sha256_file(lock_path) if lock_path.exists() else "MISSING"
            ),
            "bars_sha256": sha256_file(self.campaign_root / "data/bars.parquet"),
            "universe_sha256": sha256_file(
                self.campaign_root / "data/universe.parquet"
            ),
            "data_manifest_sha256": sha256_file(
                self.campaign_root / "data/manifest.json"
            ),
            "data_snapshot_id": str(data_manifest.get("snapshot_id", "")),
        }

    def validate(self) -> GateResult:
        """Apply OOS, CPCV/PBO, decay, cost, and confirmation requirements."""

        prior = self._prior("validation")
        if prior is not None:
            return prior
        self.gates.require_previous("validation")
        try:
            bars, _, manifest = self._load_data()
            prepared = self._prepared(bars, manifest, end=self.holdout_start)
            specs = self._load_specs()
            metrics = pd.read_parquet(self.campaign_root / "discovery/metrics.parquet")
            net = _read_streams(self.campaign_root / "discovery/net_returns.parquet")
            gross = _read_streams(
                self.campaign_root / "discovery/gross_returns.parquet"
            )
            bundle = run_validation(
                prepared,
                self.config,
                specs,
                metrics,
                net,
                gross,
            )
            self._write_parquet(
                "validation_metrics", "validation/metrics.parquet", bundle.metrics
            )
            self._write_json(
                "provisional_records",
                "validation/records.json",
                serialize_records(bundle.records),
            )
            pbo_payload = {
                "pbo": bundle.pbo.pbo,
                "defined": bundle.pbo.defined,
                "reason": bundle.pbo.reason,
                "n_splits": bundle.pbo.n_splits,
            }
            evidence = {
                "validated_ids": list(bundle.validated_ids),
                "validated_count": len(bundle.validated_ids),
                "candidate_count": int(metrics["discovery_survivor"].sum()),
                "pbo": pbo_payload,
                "placebo_survival_fraction": bundle.placebo_fraction,
                "placebo_survival_max": self.config.stats.placebo_survival_max,
                "reasons": list(bundle.reasons),
                "empty_validated_outcome": not bundle.validated_ids,
                "zipline_reloaded_available": _zipline_available(),
                "profile_scope": (
                    "SYNTHETIC_SMOKE_NON_PROMOTABLE"
                    if self.config.profile == "smoke"
                    else "FULL_EMPIRICAL"
                ),
                "disclaimer": DISCLAIMER,
            }
            self._write_json("validation_summary", "validation/summary.json", evidence)
            passed = bundle.passed or self.config.profile == "smoke"
            return self._record_gate(
                "validation",
                passed,
                (
                    "validation gauntlet completed; individual failures remain visible"
                    if bundle.passed
                    else (
                        "synthetic smoke validation completed with diagnostic-only misses"
                        if self.config.profile == "smoke"
                        else "validation global controls failed; promotion stopped"
                    )
                ),
                evidence,
            )
        except Exception as error:
            evidence = self._diagnostic("validation", error)
            return self._record_gate(
                "validation",
                False,
                "validation execution failed and was frozen as a diagnostic",
                evidence,
                blocked=True,
            )

    def report(self) -> GateResult:
        """Render the exhaustive provisional HTML and CSV evidence."""

        prior = self._prior("report")
        if prior is not None:
            return prior
        self.gates.require_previous("report")
        records = self._load_records("validation/records.json")
        validation = self._read_json("validation/summary.json")
        discovery = self._read_json("discovery/summary.json")
        summary = {
            "campaign_id": self.campaign_id,
            "profile": self.config.profile,
            "as_of": self.as_of.isoformat(),
            "holdout_start": self.holdout_start.isoformat(),
            "tested": len(records),
            "discovery_survivors": discovery["survivor_count"],
            "validated_edges": validation["validated_count"],
            "bias_tier": "SURVIVORSHIP_BIASED",
            "holdout_status": "SEALED",
        }
        preview = self._build_provisional_stack()
        overlays = self._evaluate_overlays(preview.returns)
        self._write_json(
            "provisional_overlay_evidence",
            "reports/provisional/overlay_evidence.json",
            overlays,
        )
        figures = self._report_figures(records, final=False)
        html_path, csv_path = render_verdict_report(
            records,
            summary,
            self.campaign_root / "reports/provisional",
            final=False,
            embedded_figures=figures,
            evidence_sections={
                "Data and replication evidence": {
                    "data": self._read_json("data/manifest.json"),
                    "replication": self._read_json("replication/evidence.json"),
                },
                "Discovery and validation funnel": {
                    "discovery": discovery,
                    "validation": validation,
                },
                "Overlay enable/disable evidence": overlays,
            },
        )
        self._register_file("provisional_html", html_path)
        self._register_file("provisional_csv", csv_path)
        evidence = {
            "html": str(html_path),
            "csv": str(csv_path),
            "rows": len(records),
            "disclaimer_embedded": True,
            "holdout_replay": False,
        }
        return self._record_gate(
            "report", True, "provisional report rendered from frozen evidence", evidence
        )

    def score(self) -> GateResult:
        """Build the shrunk correlation stack and freeze the complete model."""

        prior = self._prior("score")
        if prior is not None:
            return prior
        self.gates.require_previous("score")
        stack = self._build_provisional_stack()
        edge_ids = stack.artifact.edge_ids
        self._write_json("stack", "score/stack.json", asdict(stack.artifact))
        self._write_parquet(
            "composite_returns",
            "score/composite.parquet",
            stack.returns.rename_axis("session").reset_index(),
        )
        overlays = self._read_json("reports/provisional/overlay_evidence.json")
        overlay_path = self._write_json("overlays", "score/overlays.json", overlays)
        stack_path = self.campaign_root / "score/stack.json"
        specs_path = self.campaign_root / "discovery/specs.json"
        bars_path = self.campaign_root / "data/bars.parquet"
        universe_path = self.campaign_root / "data/universe.parquet"
        data_manifest_path = self.campaign_root / "data/manifest.json"
        data_manifest = self._read_json("data/manifest.json")
        cost_hash = canonical_sha256(self.config.costs.model_dump(mode="json"))
        config_hash = canonical_sha256(self.config.model_dump(mode="json"))
        lock_path = self.workspace / "uv.lock"
        decision_thresholds = {
            "confidence_min": self.config.live.minimum_confidence,
            "correlation_cluster": 0.70,
            "overlay_plateau_within": self.config.entrytiming.plateau_within,
            "confidence_formula": (
                "round(100 * composite_DSR_reliability * "
                "direction_specific_forecast_magnitude_percentile)"
            ),
            "forecast_magnitude_reference": _magnitude_reference(stack.returns),
        }
        model_mapping = _canonical_model_mapping(
            edge_ids=edge_ids,
            specs_payload=self._read_json("discovery/specs.json"),
            stack_payload=self._read_json("score/stack.json"),
            overlay_payload=overlays,
            decision_thresholds=decision_thresholds,
        )
        freeze_payload = {
            "campaign_id": self.campaign_id,
            "frozen_at": datetime.now(UTC).isoformat(),
            "edge_ids": list(stack.artifact.edge_ids),
            "specs_sha256": sha256_file(specs_path),
            "stack_sha256": sha256_file(stack_path),
            "overlay_sha256": sha256_file(overlay_path),
            "cost_sha256": cost_hash,
            "config_sha256": config_hash,
            "bars_sha256": sha256_file(bars_path),
            "universe_sha256": sha256_file(universe_path),
            "data_manifest_sha256": sha256_file(data_manifest_path),
            "source_tree_sha256": source_tree_sha256(self.workspace),
            "lock_sha256": (
                sha256_file(lock_path) if lock_path.exists() else "MISSING"
            ),
            "model_mapping_sha256": canonical_sha256(model_mapping),
            "data_snapshot_id": data_manifest["snapshot_id"],
            "decision_thresholds": decision_thresholds,
            "non_promotable": self.config.profile == "smoke",
            "disclaimer": DISCLAIMER,
        }
        freeze_payload["freeze_id"] = "freeze-" + canonical_sha256(freeze_payload)
        self._write_json("holdout_freeze", "score/freeze.json", freeze_payload)
        evidence = {
            "freeze_id": freeze_payload["freeze_id"],
            "edge_ids": list(edge_ids),
            "stack_id": stack.artifact.stack_id,
            "composite_dsr_reliability": stack.artifact.dsr_reliability,
            "cluster_count": len(set(stack.artifact.cluster_by_edge.values())),
            "enabled_overlays": [
                name
                for name, value in overlays["decisions"].items()
                if value["enabled"]
            ],
            "empty_stack_successful_outcome": not edge_ids,
            "non_promotable": self.config.profile == "smoke",
        }
        return self._record_gate(
            "score",
            True,
            (
                "empty model frozen as a valid no-edge result"
                if not edge_ids
                else "complete provisional model frozen before holdout access"
            ),
            evidence,
        )

    def finalize_holdout(self) -> GateResult:
        """Consume the single holdout authorization and render the final report."""

        if self.catalog.gate(self.campaign_id, "holdout") is not None:
            raise RuntimeError("holdout has already been evaluated for this campaign")
        self.gates.require_previous("holdout")
        freeze_payload = self._read_json("score/freeze.json")
        freeze = self._freeze_manifest(freeze_payload)
        guard = HoldoutGuard(self.catalog)

        # Holdout access is consumed before data is exposed. If the analytical
        # result was durably persisted but the process died before sealing or
        # rendering, resume from that result without opening campaign data again.
        access = self.catalog.holdout_access(self.campaign_id)
        result_path = self.campaign_root / "holdout/result.json"
        if access is not None:
            if access.freeze_id != freeze.freeze_id:
                raise RuntimeError("holdout access freeze identity does not match")
            if not result_path.is_file():
                raise RuntimeError(
                    "holdout access was consumed without a persisted result; "
                    "a second analytical evaluation is forbidden"
                )
            self._verify_frozen_model_artifacts(freeze, freeze_payload)
            result, result_sha = self._verified_holdout_result(freeze)
            if access.result_sha256 is not None and access.result_sha256 != result_sha:
                raise RuntimeError(
                    "persisted holdout result differs from sealed identity"
                )
            if access.result_sha256 is None:
                guard.complete(self.campaign_id, result_sha)
            return self._render_final_holdout_result(result, result_sha)
        if result_path.exists():
            raise RuntimeError("holdout result exists without an authorization record")

        # These checks intentionally happen before authorization is consumed.
        self._verify_frozen_artifacts(freeze, freeze_payload)
        with guard.authorize(freeze):
            bars, factors, manifest = self._load_data(include_holdout=True)
            prepared = self._prepared(
                bars, manifest, end=self.as_of, start=self.config.data.start
            )
            specs = {spec.hypothesis_id: spec for spec in self._load_specs()}
            stack_payload = self._read_json("score/stack.json")
            edge_means: dict[str, float] = {}
            edge_cis: dict[str, tuple[float, float]] = {}
            holdout_streams: dict[str, pd.Series] = {}
            selected_dates = prepared.dates >= pd.Timestamp(self.holdout_start)
            for edge_id in freeze.edge_ids:
                trial = run_trial(
                    prepared,
                    specs[edge_id],
                    cost_model=CostModel(self.config.costs),
                )
                stream = pd.Series(
                    trial.result.net_returns, index=prepared.dates, name=edge_id
                ).loc[selected_dates]
                holdout_streams[edge_id] = stream
                edge_means[edge_id] = float(stream.mean())
                if stream.notna().sum() >= 2:
                    interval = stationary_bootstrap_ci(
                        stream.to_numpy(float),
                        statistic="mean",
                        n_resamples=self.config.stats.finalist_bootstrap_reps,
                        seed=self.config.stats.seed,
                    )
                    edge_cis[edge_id] = (interval.lower, interval.upper)
            composite_mean: float | None
            if freeze.edge_ids:
                frame = pd.DataFrame(holdout_streams)
                weights = {
                    str(key): float(value)
                    for key, value in stack_payload["weights"].items()
                }
                composite = frame.mul(pd.Series(weights), axis=1).sum(
                    axis=1, min_count=1
                )
                composite_mean = float(composite.mean())
                composite_interval = stationary_bootstrap_ci(
                    composite.to_numpy(float),
                    statistic="mean",
                    n_resamples=self.config.stats.finalist_bootstrap_reps,
                    seed=self.config.stats.seed,
                )
                composite_ci: tuple[float, float] | None = (
                    composite_interval.lower,
                    composite_interval.upper,
                )
            else:
                composite = pd.Series(dtype=float, name="composite")
                composite_mean = None
                composite_ci = None
            overlays = self._read_json("score/overlays.json")
            enabled = {
                name: value
                for name, value in overlays["decisions"].items()
                if value["enabled"]
            }
            # Overlay holdout increments are recomputed by the exact frozen
            # transformation when one was enabled. Empty is a valid outcome.
            overlay_increments = self._holdout_overlay_increments(
                composite,
                enabled,
                prepared.close["SPY"],
                prepared.close["SPY"]
                .mul(prepared.volume["SPY"])
                .rolling(20, min_periods=1)
                .mean(),
                (
                    factors.set_index("session")["VIXCLS"].astype(float)
                    if {"session", "VIXCLS"}.issubset(factors.columns)
                    else pd.Series(dtype=float)
                ),
            )
            edge_positive = all(value > 0.0 for value in edge_means.values())
            composite_positive = not freeze.edge_ids or (
                composite_mean is not None and composite_mean > 0.0
            )
            overlays_nonnegative = all(
                value >= 0.0 for value in overlay_increments.values()
            )
            research_pass = (
                edge_positive and composite_positive and overlays_nonnegative
            )
            promoted = bool(
                research_pass and freeze.edge_ids and self.config.profile == "full"
            )
            result = {
                "campaign_id": self.campaign_id,
                "freeze_id": freeze.freeze_id,
                "evaluated_at": datetime.now(UTC).isoformat(),
                "holdout_start": self.holdout_start.isoformat(),
                "holdout_end": self.as_of.isoformat(),
                "edge_means": edge_means,
                "edge_mean_cis": edge_cis,
                "evaluated_edge_ids": list(freeze.edge_ids),
                "composite_mean": composite_mean,
                "composite_mean_ci": composite_ci,
                "overlay_incremental_means": overlay_increments,
                "edge_positive": edge_positive,
                "composite_positive": composite_positive,
                "overlays_nonnegative": overlays_nonnegative,
                "research_gate_pass": research_pass,
                "promoted_for_paper_assistant": promoted,
                "empty_model_successful_outcome": not freeze.edge_ids,
                "non_promotable_smoke": self.config.profile == "smoke",
                "disclaimer": DISCLAIMER,
            }
            result_sha = canonical_sha256(result)
            result["result_sha256"] = result_sha
            if not composite.empty:
                self._write_parquet(
                    "holdout_composite",
                    "holdout/composite.parquet",
                    composite.rename_axis("session").reset_index(),
                )
            self._write_json("holdout_result", "holdout/result.json", result)
        guard.complete(self.campaign_id, result_sha)
        return self._render_final_holdout_result(result, result_sha)

    def _freeze_manifest(self, payload: Any) -> HoldoutFreezeManifest:
        """Validate the freeze document's self-identity and score-gate binding."""

        if not isinstance(payload, dict):
            raise RuntimeError("holdout freeze manifest is not a JSON object")
        identity = dict(payload)
        freeze_id = str(identity.pop("freeze_id", ""))
        expected_id = "freeze-" + canonical_sha256(identity)
        if freeze_id != expected_id:
            raise RuntimeError("holdout freeze manifest identity mismatch")
        if str(payload.get("campaign_id")) != self.campaign_id:
            raise RuntimeError("holdout freeze belongs to another campaign")
        score_gate = self.catalog.gate(self.campaign_id, "score")
        if score_gate is None or str(score_gate.evidence.get("freeze_id")) != freeze_id:
            raise RuntimeError("holdout freeze differs from the persisted score gate")
        freeze_path = self.campaign_root / "score/freeze.json"
        if not self.catalog.artifact_registered(
            self.campaign_id, "holdout_freeze", sha256_file(freeze_path)
        ):
            raise RuntimeError("holdout freeze is absent from artifact ledger")
        try:
            return HoldoutFreezeManifest(
                campaign_id=self.campaign_id,
                freeze_id=freeze_id,
                frozen_at=datetime.fromisoformat(str(payload["frozen_at"])),
                edge_ids=tuple(str(value) for value in payload["edge_ids"]),
                specs_sha256=str(payload["specs_sha256"]),
                stack_sha256=str(payload["stack_sha256"]),
                overlay_sha256=str(payload["overlay_sha256"]),
                cost_sha256=str(payload["cost_sha256"]),
                config_sha256=str(payload["config_sha256"]),
                bars_sha256=str(payload["bars_sha256"]),
                universe_sha256=str(payload["universe_sha256"]),
                data_manifest_sha256=str(payload["data_manifest_sha256"]),
                source_tree_sha256=str(payload["source_tree_sha256"]),
                lock_sha256=str(payload["lock_sha256"]),
                model_mapping_sha256=str(payload["model_mapping_sha256"]),
                data_snapshot_id=str(payload["data_snapshot_id"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "holdout freeze manifest is incomplete or invalid"
            ) from error

    def _verify_frozen_artifacts(
        self, freeze: HoldoutFreezeManifest, payload: dict[str, Any]
    ) -> None:
        """Reject any artifact, code, dependency, or model-mapping drift."""

        self._verify_frozen_model_artifacts(freeze, payload)
        actual = {
            "canonical bars": sha256_file(self.campaign_root / "data/bars.parquet"),
            "universe": sha256_file(self.campaign_root / "data/universe.parquet"),
            "data manifest": sha256_file(self.campaign_root / "data/manifest.json"),
        }
        expected = {
            "canonical bars": freeze.bars_sha256,
            "universe": freeze.universe_sha256,
            "data manifest": freeze.data_manifest_sha256,
        }
        for label, expected_hash in expected.items():
            if actual[label] != expected_hash:
                raise RuntimeError(f"frozen {label} hash mismatch")

        data_manifest = self._read_json("data/manifest.json")
        if str(data_manifest.get("snapshot_id")) != freeze.data_snapshot_id:
            raise RuntimeError("frozen data snapshot identity mismatch")

    def _verify_frozen_model_artifacts(
        self, freeze: HoldoutFreezeManifest, payload: dict[str, Any]
    ) -> None:
        """Verify code and model inputs without opening the holdout dataset."""

        lock_path = self.workspace / "uv.lock"
        actual = {
            "specs": sha256_file(self.campaign_root / "discovery/specs.json"),
            "stack": sha256_file(self.campaign_root / "score/stack.json"),
            "overlays": sha256_file(self.campaign_root / "score/overlays.json"),
            "costs": canonical_sha256(self.config.costs.model_dump(mode="json")),
            "config": canonical_sha256(self.config.model_dump(mode="json")),
            "source tree": source_tree_sha256(self.workspace),
            "lockfile": sha256_file(lock_path) if lock_path.exists() else "MISSING",
        }
        expected = {
            "specs": freeze.specs_sha256,
            "stack": freeze.stack_sha256,
            "overlays": freeze.overlay_sha256,
            "costs": freeze.cost_sha256,
            "config": freeze.config_sha256,
            "source tree": freeze.source_tree_sha256,
            "lockfile": freeze.lock_sha256,
        }
        for label, expected_hash in expected.items():
            if actual[label] != expected_hash:
                raise RuntimeError(f"frozen {label} hash mismatch")

        mapping = _canonical_model_mapping(
            edge_ids=freeze.edge_ids,
            specs_payload=self._read_json("discovery/specs.json"),
            stack_payload=self._read_json("score/stack.json"),
            overlay_payload=self._read_json("score/overlays.json"),
            decision_thresholds=payload.get("decision_thresholds"),
        )
        if canonical_sha256(mapping) != freeze.model_mapping_sha256:
            raise RuntimeError("frozen model mapping hash mismatch")

    def _verified_holdout_result(
        self, freeze: HoldoutFreezeManifest
    ) -> tuple[dict[str, Any], str]:
        """Load a durable result and verify its internal and catalog identities."""

        payload = self._read_json("holdout/result.json")
        if not isinstance(payload, dict):
            raise RuntimeError("holdout result is not a JSON object")
        unsigned = dict(payload)
        claimed = str(unsigned.pop("result_sha256", ""))
        computed = canonical_sha256(unsigned)
        if not claimed or claimed != computed:
            raise RuntimeError("persisted holdout result hash mismatch")
        path = self.campaign_root / "holdout/result.json"
        if not self.catalog.artifact_registered(
            self.campaign_id, "holdout_result", sha256_file(path)
        ):
            raise RuntimeError(
                "persisted holdout result is absent from artifact ledger"
            )
        if str(payload.get("campaign_id")) != self.campaign_id:
            raise RuntimeError("persisted holdout result belongs to another campaign")
        if str(payload.get("freeze_id")) != freeze.freeze_id:
            raise RuntimeError("persisted holdout result uses another freeze")
        evaluated = tuple(str(value) for value in payload.get("evaluated_edge_ids", ()))
        if evaluated != freeze.edge_ids:
            raise RuntimeError("persisted holdout result evaluated the wrong edge set")
        edge_means = payload.get("edge_means")
        if not isinstance(edge_means, dict) or set(edge_means) != set(freeze.edge_ids):
            raise RuntimeError("persisted holdout edge evidence is incomplete")
        return payload, computed

    def _render_final_holdout_result(
        self, result: dict[str, Any], result_sha: str
    ) -> GateResult:
        """Render and gate a verified result without reopening holdout inputs."""

        edge_means = {
            str(key): float(value) for key, value in dict(result["edge_means"]).items()
        }
        edge_cis = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in dict(result.get("edge_mean_cis", {})).items()
        }
        evaluated_ids = {str(value) for value in result.get("evaluated_edge_ids", ())}
        provisional = self._load_records("validation/records.json")
        final = final_records(
            provisional,
            edge_means,
            evaluated_ids=evaluated_ids,
            holdout_cis=edge_cis,
        )
        self._write_json(
            "final_records", "holdout/records.json", serialize_records(final)
        )
        figures = self._report_figures(final, final=True)
        composite_mean = result.get("composite_mean")
        promoted = bool(result["promoted_for_paper_assistant"])
        empty = bool(result["empty_model_successful_outcome"])
        html_path, csv_path = render_verdict_report(
            final,
            {
                "campaign_id": self.campaign_id,
                "holdout_start": self.holdout_start,
                "holdout_end": self.as_of,
                "composite_mean": composite_mean,
                "promoted": promoted,
                "empty_model_successful_outcome": empty,
            },
            self.campaign_root / "reports/final",
            final=True,
            embedded_figures=figures,
            evidence_sections={
                "Atomic holdout evidence": result,
                "Frozen stack": self._read_json("score/stack.json"),
                "Frozen overlay evidence": self._read_json("score/overlays.json"),
            },
        )
        self._register_file("final_html", html_path)
        self._register_file("final_csv", csv_path)
        research_pass = bool(result["research_gate_pass"])
        evidence = {
            "result_sha256": result_sha,
            "edge_means": edge_means,
            "edge_mean_cis": edge_cis,
            "composite_mean": composite_mean,
            "composite_mean_ci": result.get("composite_mean_ci"),
            "overlay_incremental_means": result.get("overlay_incremental_means", {}),
            "promoted": promoted,
            "empty_model_successful_outcome": empty,
            "final_html": str(html_path),
            "final_csv": str(csv_path),
        }
        return self._record_gate(
            "holdout",
            research_pass,
            (
                "atomic holdout evaluation passed"
                if research_pass
                else "atomic holdout evaluation failed; model was not rebuilt or promoted"
            ),
            evidence,
        )

    def live(self, *, once: bool = False) -> str:
        """Refuse Phase 6 until a genuine full model and assistant are available."""

        del once
        self.gates.require_previous("live")
        result = self._read_json("holdout/result.json")
        if self.config.profile != "full" or not result["promoted_for_paper_assistant"]:
            raise RuntimeError(
                "DATA_UNAVAILABLE: paper assistant requires a genuinely promoted "
                "non-empty full campaign composite"
            )
        raise RuntimeError(
            "DATA_UNAVAILABLE: Phase 6 paper assistant is not initialized; the "
            "recorded live-demo is an engineering fixture, not a live campaign scan"
        )

    def _full_replication_inputs(
        self,
        bars: pd.DataFrame,
        factors: pd.DataFrame,
        fomc_dates: pd.DatetimeIndex,
    ) -> dict[str, Any]:
        factor_frame = factors.copy()
        factor_frame["session"] = pd.to_datetime(factor_frame["session"])
        market = factor_frame.set_index("session")["market_return"].astype(float)
        by_symbol = {
            str(symbol): group.sort_values("session").set_index("session")
            for symbol, group in bars.groupby("symbol", sort=True)
        }
        spy = by_symbol["SPY"]["adjusted_close"].pct_change(fill_method=None)
        close = bars.pivot(
            index="session", columns="symbol", values="adjusted_close"
        ).sort_index()
        volume = (
            bars.pivot(index="session", columns="symbol", values="volume")
            .sort_index()
            .reindex_like(close)
        )
        adv = (
            close.mul(volume)
            .rolling(20, min_periods=1)
            .mean()
            .fillna(100_000_000.0)
            .to_numpy(float)
        )
        types_by_symbol = (
            bars.sort_values("session", kind="stable")
            .groupby("symbol", sort=False)["asset_type"]
            .last()
            .astype(str)
        )
        asset_types = tuple(
            (
                "etf"
                if types_by_symbol.get(str(symbol), "equity").lower() == "etf"
                else "equity"
            )
            for symbol in close.columns
        )
        returns = close.pct_change(fill_method=None)
        features = canonical_features(close)
        momentum_weights = _monthly_weights(decile_weights(features.momentum))
        reversal_weights = (
            decile_weights(features.reversal).rolling(5, min_periods=1).mean()
        )
        momentum_gross, _, _ = vectorized_backtest(
            momentum_weights.to_numpy(float),
            returns.to_numpy(float),
            cost_model=CostModel(self.config.costs),
            adv_dollars=adv,
            asset_type=asset_types,
        )
        reversal_gross, reversal_net, _ = vectorized_backtest(
            reversal_weights.to_numpy(float),
            returns.to_numpy(float),
            cost_model=CostModel(self.config.costs),
            adv_dollars=adv,
            asset_type=asset_types,
        )
        return {
            "market_returns": market,
            "spy_returns": spy,
            "fomc_dates": fomc_dates,
            "bars_by_symbol": {key: by_symbol[key] for key in ("SPY", "QQQ")},
            "momentum_returns": pd.Series(momentum_gross, index=close.index),
            "reversal_gross": reversal_gross,
            "reversal_net": reversal_net,
        }

    def _prepared(
        self,
        bars: pd.DataFrame,
        manifest: dict[str, Any],
        *,
        end: date,
        start: date | None = None,
    ) -> Any:
        universe = pd.read_parquet(self.campaign_root / "data/universe.parquet")
        sector_by_symbol = {
            str(row.symbol): str(row.sector)
            for row in universe.itertuples()
            if pd.notna(row.sector)
        }
        fomc_dates = pd.DatetimeIndex(manifest.get("fomc_dates", []))
        research_end = (
            pd.Timestamp(end) - pd.Timedelta(days=1)
            if end == self.holdout_start
            else pd.Timestamp(end)
        )
        return prepare_research(
            bars,
            start=pd.Timestamp(start or self.config.data.start),
            end=research_end,
            fomc_dates=fomc_dates,
            sector_by_symbol=sector_by_symbol,
        )

    def _build_provisional_stack(self) -> StackResult:
        """Rebuild the deterministic pre-holdout stack from persisted evidence."""

        validation = self._read_json("validation/summary.json")
        edge_ids = tuple(str(value) for value in validation["validated_ids"])
        streams = _read_streams(self.campaign_root / "discovery/net_returns.parquet")
        metrics = pd.read_parquet(self.campaign_root / "validation/metrics.parquet")
        selected = metrics.set_index("hypothesis_id").reindex(edge_ids)
        net_means = {
            edge: float(
                pd.to_numeric(pd.Series([selected.loc[edge, "net_mean"]])).iloc[0]
            )
            for edge in edge_ids
        }
        sampling_variances = {
            edge: float(
                np.nanvar(streams[edge].to_numpy(dtype=float), ddof=1)
                / max(
                    int(
                        pd.to_numeric(
                            pd.Series([selected.loc[edge, "sample_size"]])
                        ).iloc[0]
                    ),
                    1,
                )
            )
            for edge in edge_ids
        }
        selected_streams = (
            streams.loc[:, list(edge_ids)]
            if edge_ids
            else pd.DataFrame(index=streams.index)
        )
        preliminary = build_stack(
            selected_streams,
            net_means,
            sampling_variances,
            0.0,
            correlation_threshold=0.70,
        )
        values = preliminary.returns.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            reliability = 0.0
        else:
            periodic = _periodic_sharpe(finite)
            centered = finite - float(finite.mean())
            deviation = float(finite.std(ddof=1))
            standardized = centered / deviation if deviation > 0.0 else centered
            trial_sharpes = pd.to_numeric(metrics["sharpe"], errors="coerce").to_numpy(
                float
            ) / math.sqrt(252.0)
            trial_sharpes = trial_sharpes[np.isfinite(trial_sharpes)]
            reliability = deflated_sharpe_ratio(
                periodic,
                n_observations=len(finite),
                n_trials=max(len(metrics), 1),
                skewness=float(np.mean(standardized**3)),
                kurtosis=float(np.mean(standardized**4)),
                trial_sharpes=trial_sharpes if len(trial_sharpes) else None,
            )
            reliability = float(reliability) if math.isfinite(reliability) else 0.0
        return build_stack(
            selected_streams,
            net_means,
            sampling_variances,
            reliability,
            correlation_threshold=0.70,
        )

    def _evaluate_overlays(self, composite: pd.Series) -> dict[str, Any]:
        neighborhoods: dict[str, list[float]] = {
            "rsi2": [float(value) for value in self.config.entrytiming.rsi2_thresholds],
            "bollinger_pct_b": list(self.config.entrytiming.bollinger_thresholds),
            "expiry": [float(value) for value in self.config.entrytiming.expiry_bars],
            "breakout": [
                float(value) for value in self.config.entrytiming.breakout_windows
            ],
            "atr_stop": [
                float(value) for value in self.config.entrytiming.atr_multipliers
            ],
            "ma200": [float(self.config.entrytiming.ma_window)],
            "vix": [self.config.entrytiming.vix_low, self.config.entrytiming.vix_high],
        }
        if composite.empty:
            return {
                "neighborhoods": neighborhoods,
                "evidence": {},
                "decisions": {
                    name: asdict(OverlayDecision(False, None, "no frozen composite"))
                    for name in neighborhoods
                },
                "limited_intraday": {
                    "status": "DATA_UNAVAILABLE",
                    "may_alter_primary_model": False,
                },
            }
        bars, factors, _ = self._load_data()
        spy_bars = bars.loc[bars["symbol"] == "SPY"].set_index("session")
        spy = spy_bars["adjusted_close"].reindex(composite.index).astype(float)
        spy_adv = (
            spy_bars["adjusted_close"]
            .mul(spy_bars["volume"])
            .rolling(20, min_periods=1)
            .mean()
            .reindex(composite.index)
            .fillna(100_000_000.0)
        )
        vix = pd.Series(dtype=float)
        if {"session", "VIXCLS"}.issubset(factors.columns):
            vix = (
                factors.set_index("session")["VIXCLS"]
                .reindex(composite.index)
                .astype(float)
            )
        conditions: dict[str, dict[float, pd.Series]] = {
            "rsi2": {
                threshold: rsi(spy, 2) <= threshold
                for threshold in neighborhoods["rsi2"]
            },
            "bollinger_pct_b": {
                threshold: bollinger_pct_b(spy, 20) <= threshold
                for threshold in neighborhoods["bollinger_pct_b"]
            },
            "breakout": {
                window: spy >= spy.rolling(int(window), min_periods=int(window)).max()
                for window in neighborhoods["breakout"]
            },
            "ma200": {
                neighborhoods["ma200"][0]: spy
                >= spy.rolling(
                    int(neighborhoods["ma200"][0]),
                    min_periods=int(neighborhoods["ma200"][0]),
                ).mean()
            },
        }
        if not vix.empty:
            conditions["vix"] = {
                threshold: vix <= threshold for threshold in neighborhoods["vix"]
            }
        evidence_by_family: dict[str, list[dict[str, Any]]] = {}
        decisions: dict[str, dict[str, Any]] = {}
        raw_items: dict[str, list[OverlayEvidence]] = {}
        p_value_locations: list[tuple[str, int]] = []
        p_values: list[float] = []
        trial_count = sum(len(settings) for settings in conditions.values())
        cost_model = CostModel(self.config.costs)
        for family, settings in conditions.items():
            items: list[OverlayEvidence] = []
            family_streams: list[pd.Series] = []
            for parameter, condition in settings.items():
                sensitivity = {
                    float(scale): _overlay_incremental_stream(
                        composite,
                        condition,
                        cost_model=cost_model,
                        adv_dollars=spy_adv,
                        cost_multiplier=float(scale),
                        order_dollars=self.config.costs.capital
                        * self.config.live.max_position_fraction,
                    )
                    for scale in self.config.costs.sensitivity_multipliers
                }
                incremental = sensitivity[1.0]
                family_streams.append(incremental)
                test = hac_mean_test(incremental.to_numpy(float))
                p_value = float(test.p_value)
                p_values.append(
                    min(max(p_value, 0.0), 1.0) if math.isfinite(p_value) else 1.0
                )
                p_value_locations.append((family, len(items)))
                walk = expanding_walk_forward(
                    incremental.to_numpy(float), pd.DatetimeIndex(incremental.index)
                )
                periodic = _periodic_sharpe(incremental.to_numpy(float))
                dsr = deflated_sharpe_ratio(
                    periodic,
                    n_observations=len(incremental),
                    n_trials=trial_count,
                )
                sensitivity_means = tuple(
                    (scale, float(stream.mean()))
                    for scale, stream in sorted(sensitivity.items())
                )
                item = OverlayEvidence(
                    parameter,
                    test.mean,
                    _annual_sharpe(incremental),
                    test.t_stat,
                    False,
                    dsr,
                    walk.stitched_oos_test.t_stat,
                    walk.positive_fraction,
                    sensitivity_means,
                    all(value >= 0.0 for _, value in sensitivity_means),
                )
                items.append(item)
            pbo = cpcv_pbo(
                np.column_stack(
                    [stream.to_numpy(dtype=float) for stream in family_streams]
                ),
                n_groups=self.config.validation.cpcv_groups,
                n_test_groups=self.config.validation.cpcv_test_groups,
                purge=self.config.validation.purge_sessions,
                embargo=self.config.validation.embargo_sessions,
            )
            items = [
                replace(
                    item,
                    pbo=pbo.pbo,
                    pbo_pass=bool(
                        pbo.defined
                        and pbo.pbo is not None
                        and pbo.pbo < self.config.validation.pbo_max
                    ),
                )
                for item in items
            ]
            raw_items[family] = items
        adjusted = benjamini_hochberg(p_values, q=self.config.stats.fdr_q)
        for index, (family, position) in enumerate(p_value_locations):
            raw_items[family][position] = replace(
                raw_items[family][position],
                fdr_pass=bool(adjusted.reject[index]),
            )
        for family, items in raw_items.items():
            decision = interaction_decision(
                items, plateau_within=self.config.entrytiming.plateau_within
            )
            evidence_by_family[family] = [asdict(item) for item in items]
            decisions[family] = asdict(decision)
        for family in ("expiry", "atr_stop", "vix"):
            reason = (
                "daily data cannot establish causal intraday stop fills"
                if family == "atr_stop"
                else (
                    "VIX history was unavailable in this campaign"
                    if family == "vix" and vix.empty
                    else "daily portfolio streams cannot identify per-entry expiry causally"
                )
            )
            if family not in decisions:
                decisions[family] = asdict(OverlayDecision(False, None, reason))
                evidence_by_family[family] = []
        return {
            "neighborhoods": neighborhoods,
            "evidence": evidence_by_family,
            "decisions": decisions,
            "limited_intraday": {
                "status": "EXPLORATORY_LOW_POWER_DATA_UNAVAILABLE",
                "may_alter_primary_model": False,
                "may_enable_vwap_or_limit_fill_alpha": False,
            },
        }

    def _holdout_overlay_increments(
        self,
        composite: pd.Series,
        enabled: dict[str, Any],
        spy: pd.Series,
        spy_adv: pd.Series,
        vix: pd.Series,
    ) -> dict[str, float]:
        if composite.empty or not enabled:
            return {}
        output: dict[str, float] = {}
        for family, decision in enabled.items():
            parameter = float(decision["selected_parameter"])
            if family == "rsi2":
                condition = rsi(spy, 2) <= parameter
            elif family == "bollinger_pct_b":
                condition = bollinger_pct_b(spy, 20) <= parameter
            elif family == "breakout":
                window = int(parameter)
                condition = spy >= spy.rolling(window, min_periods=window).max()
            elif family == "ma200":
                condition = (
                    spy
                    >= spy.rolling(int(parameter), min_periods=int(parameter)).mean()
                )
            elif family == "vix" and not vix.empty:
                condition = vix.reindex(spy.index) <= parameter
            else:
                output[family] = -math.inf
                continue
            incremental = _overlay_incremental_stream(
                composite,
                condition,
                cost_model=CostModel(self.config.costs),
                adv_dollars=spy_adv,
                order_dollars=self.config.costs.capital
                * self.config.live.max_position_fraction,
            )
            output[family] = float(incremental.mean())
        return output

    def _report_figures(self, records: Any, *, final: bool) -> dict[str, str]:
        """Persist report PNGs and return data URIs for self-contained HTML."""

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        directory = (
            self.campaign_root / "reports" / ("final" if final else "provisional")
        )
        directory.mkdir(parents=True, exist_ok=True)
        output: dict[str, str] = {}

        def save(title: str, stem: str, figure: Any) -> None:
            figure.tight_layout()
            path = directory / f"{stem}.png"
            figure.savefig(path, dpi=140)
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=140)
            plt.close(figure)
            self._register_file(f"{'final' if final else 'provisional'}_{stem}", path)
            output[title] = "data:image/png;base64," + base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")

        counts: dict[str, int] = {}
        for record in records:
            label = (
                record.verdict.value
                if record.verdict
                else record.execution_status.value
            )
            counts[label] = counts.get(label, 0) + 1
        figure, axis = plt.subplots(figsize=(8, 4))
        if counts:
            axis.bar(list(counts), list(counts.values()), color="#315a7d")
        else:
            axis.text(0.5, 0.5, "No evaluated hypotheses", ha="center", va="center")
        axis.set_title("Frozen filter funnel / verdict counts")
        axis.tick_params(axis="x", rotation=25)
        save("Filter funnel", "filter_funnel", figure)

        figure, axis = plt.subplots(figsize=(9, 4.5))
        plotted = False
        if final and (self.campaign_root / "holdout/composite.parquet").exists():
            frame = pd.read_parquet(self.campaign_root / "holdout/composite.parquet")
            if "composite" in frame:
                equity = (1.0 + frame["composite"].fillna(0.0)).cumprod()
                axis.plot(
                    pd.to_datetime(frame["session"]),
                    equity,
                    label="frozen composite",
                )
                plotted = True
        elif not final:
            validation = self._read_json("validation/summary.json")
            identifiers = list(validation["validated_ids"])
            if identifiers:
                streams = _read_streams(
                    self.campaign_root / "discovery/net_returns.parquet"
                )
                for identifier in identifiers[:12]:
                    equity = (1.0 + streams[identifier].fillna(0.0)).cumprod()
                    axis.plot(
                        equity.index,
                        equity,
                        label=str(identifier)[:18],
                        alpha=0.75,
                    )
                    plotted = True
        if plotted:
            axis.legend(fontsize=7, ncol=2)
        else:
            axis.text(
                0.5,
                0.5,
                "No surviving edge/composite — valid empty research outcome",
                ha="center",
                va="center",
            )
        axis.set_title("Net equity curves")
        axis.set_ylabel("Growth of 1.0")
        save("Equity curves", "equity_curves", figure)

        figure, axis = plt.subplots(figsize=(8, 4))
        validation_metrics = pd.read_parquet(
            self.campaign_root / "validation/metrics.parquet"
        )
        cost_rows = validation_metrics.loc[
            validation_metrics["cost_sensitivity"].astype(str) != "{}"
        ].head(8)
        if cost_rows.empty:
            axis.text(
                0.5,
                0.5,
                "No finalist cost-sensitivity curve",
                ha="center",
                va="center",
            )
        else:
            for row in cost_rows.itertuples():
                parsed = ast.literal_eval(str(row.cost_sensitivity))
                values = {str(key): float(value) for key, value in parsed.items()}
                axis.plot(
                    list(values),
                    list(values.values()),
                    marker="o",
                    label=str(row.hypothesis_id)[:16],
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.legend(fontsize=7)
        axis.set_title("0.5x / 1x / 2x / 4x cost sensitivity")
        axis.set_ylabel("Mean net return")
        save("Cost sensitivity", "cost_sensitivity", figure)
        return output

    def _load_data(
        self, *, include_holdout: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Load campaign data while sealing holdout observations by default."""

        filters = None
        if not include_holdout:
            filters = [("session", "<", pd.Timestamp(self.holdout_start))]
        bars = pd.read_parquet(
            self.campaign_root / "data/bars.parquet", filters=filters
        )
        factors = pd.read_parquet(
            self.campaign_root / "data/factors.parquet", filters=filters
        )
        manifest = self._read_json("data/manifest.json")
        if not include_holdout:
            cutoff = pd.Timestamp(self.holdout_start)
            manifest = dict(manifest)
            manifest["fomc_dates"] = [
                value
                for value in manifest.get("fomc_dates", [])
                if pd.Timestamp(value) < cutoff
            ]
            if (not bars.empty and pd.to_datetime(bars["session"]).max() >= cutoff) or (
                not factors.empty and pd.to_datetime(factors["session"]).max() >= cutoff
            ):
                raise RuntimeError("pre-freeze data view crossed the sealed holdout")
        return bars, factors, manifest

    def _load_specs(self) -> tuple[HypothesisSpec, ...]:
        payload = self._read_json("discovery/specs.json")
        return tuple(spec_from_dict(item) for item in payload)

    def _load_records(self, relative: str) -> Any:
        payload = self._read_json(relative)
        return records_from_payload(payload)

    def _prior(self, phase: str) -> GateResult | None:
        return self.catalog.gate(self.campaign_id, phase)

    def _record_gate(
        self,
        phase: str,
        passed: bool,
        summary: str,
        evidence: dict[str, Any] | None = None,
        *,
        blocked: bool = False,
    ) -> GateResult:
        """Persist a gate and emit a self-contained report for every stopped phase."""

        result = self.gates.record(phase, passed, summary, evidence, blocked=blocked)
        if result.status not in {GateStatus.FAIL, GateStatus.BLOCKED}:
            return result
        serialized = json.dumps(
            dict(result.evidence), sort_keys=True, indent=2, default=str
        )
        frame = pd.DataFrame(
            [
                {
                    "campaign_id": self.campaign_id,
                    "phase": phase,
                    "status": result.status.value,
                    "summary": summary,
                    "evidence": json.dumps(
                        dict(result.evidence), sort_keys=True, default=str
                    ),
                    "definitions_changed": False,
                    "disclaimer": DISCLAIMER,
                }
            ]
        )
        csv_path = self.store.write_text(
            f"diagnostics/{phase}_report.csv", frame.to_csv(index=False)
        )
        document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>EdgeStack stopped campaign diagnostic</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem}} .stop{{background:#7b241c;color:white;padding:1rem}}
.disclaimer{{border:2px solid #b03a2e;padding:1rem}} pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem}}
</style></head><body><h1>Stopped campaign diagnostic</h1>
<div class="stop">{html.escape(phase)}: {html.escape(result.status.value)} — {html.escape(summary)}</div>
<p class="disclaimer">{html.escape(DISCLAIMER)}</p><h2>Frozen evidence</h2>
<pre>{html.escape(serialized)}</pre></body></html>"""
        html_path = self.store.write_text(f"diagnostics/{phase}_report.html", document)
        self._register_file(f"{phase}_diagnostic_csv", csv_path)
        self._register_file(f"{phase}_diagnostic_html", html_path)
        return result

    def _write_json(self, kind: str, relative: str, value: Any) -> Path:
        path = self.store.write_json(relative, value)
        self._register_file(kind, path)
        return path

    def _write_parquet(self, kind: str, relative: str, frame: pd.DataFrame) -> Path:
        path = self.store.write_parquet(relative, frame)
        self._register_file(kind, path)
        return path

    def _register_file(self, kind: str, path: Path) -> None:
        self.catalog.record_artifact(self.campaign_id, kind, sha256_file(path), path)

    def _read_json(self, relative: str) -> Any:
        return json.loads((self.campaign_root / relative).read_text(encoding="utf-8"))

    def _diagnostic(self, phase: str, error: Exception) -> dict[str, Any]:
        evidence = {
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "created_at": datetime.now(UTC).isoformat(),
            "definitions_changed": False,
            "disclaimer": DISCLAIMER,
        }
        self._write_json(f"{phase}_diagnostic", f"diagnostics/{phase}.json", evidence)
        return evidence


def _canonical_model_mapping(
    *,
    edge_ids: tuple[str, ...],
    specs_payload: Any,
    stack_payload: Any,
    overlay_payload: Any,
    decision_thresholds: Any,
) -> dict[str, Any]:
    """Return the exact edge-to-spec/weight/overlay mapping used by holdout."""

    if not isinstance(specs_payload, list):
        raise RuntimeError("hypothesis registry is not a JSON array")
    registry: dict[str, dict[str, Any]] = {}
    for raw in specs_payload:
        if not isinstance(raw, dict):
            raise RuntimeError("hypothesis registry contains a non-object entry")
        declared = str(raw.get("hypothesis_id", ""))
        if not declared or spec_from_dict(raw).hypothesis_id != declared:
            raise RuntimeError("hypothesis registry identity mismatch")
        if declared in registry:
            raise RuntimeError("hypothesis registry contains duplicate identities")
        registry[declared] = dict(raw)
    missing = set(edge_ids).difference(registry)
    if missing:
        raise RuntimeError(f"frozen edge specifications are missing: {sorted(missing)}")
    if not isinstance(stack_payload, dict):
        raise RuntimeError("stack artifact is not a JSON object")
    stack_edges = tuple(str(value) for value in stack_payload.get("edge_ids", ()))
    if stack_edges != edge_ids:
        raise RuntimeError("stack edge order differs from frozen edge order")
    for field in ("weights", "cluster_by_edge", "shrunk_means"):
        values = stack_payload.get(field)
        if not isinstance(values, dict) or set(values) != set(edge_ids):
            raise RuntimeError(f"stack {field} does not cover the frozen edge set")
    if not isinstance(overlay_payload, dict) or not isinstance(
        overlay_payload.get("decisions"), dict
    ):
        raise RuntimeError("overlay decision artifact is invalid")
    if not isinstance(decision_thresholds, dict):
        raise RuntimeError("frozen decision thresholds are invalid")
    return {
        "edge_ids": list(edge_ids),
        "specifications": [registry[edge_id] for edge_id in edge_ids],
        "stack_id": stack_payload.get("stack_id"),
        "weights": stack_payload["weights"],
        "cluster_by_edge": stack_payload["cluster_by_edge"],
        "shrunk_means": stack_payload["shrunk_means"],
        "dsr_reliability": stack_payload.get("dsr_reliability"),
        "overlay_decisions": overlay_payload["decisions"],
        "decision_thresholds": decision_thresholds,
    }


def _magnitude_reference(composite: pd.Series) -> dict[str, Any]:
    """Freeze direction-specific empirical magnitudes for ordinal confidence."""

    values = pd.to_numeric(composite, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    percentiles = list(range(101))

    def distribution(sample: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, Any]:
        if not len(sample):
            return {"observations": 0, "percentiles": percentiles, "quantiles": []}
        return {
            "observations": len(sample),
            "percentiles": percentiles,
            "quantiles": np.quantile(sample, np.linspace(0.0, 1.0, 101)).tolist(),
        }

    return {
        "LONG": distribution(values[values > 0.0]),
        "SHORT": distribution(np.abs(values[values < 0.0])),
    }


def _last_completed_session(requested: date | None) -> date:
    calendar = NYSECalendar()
    now = datetime.now(UTC)
    if requested is not None and requested > now.date():
        raise ValueError("as_of cannot be a future calendar date")
    candidate = requested or now.date()
    if calendar.is_session(candidate):
        if requested is not None and candidate < now.date():
            return candidate
        if now >= calendar.close_time(candidate):
            return candidate
    return calendar.previous_session(candidate, inclusive=False).date()


def _resolve(base: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_streams(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["session"] = pd.to_datetime(frame["session"])
    return frame.set_index("session").sort_index()


def _monthly_weights(weights: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(weights.index)
    periods = pd.Series(index.to_period("M"), index=index)
    first = periods.ne(periods.shift(1))
    return weights.where(first, np.nan).ffill().fillna(0.0)


def _empty_qa() -> Any:
    from edgestack.data.quality import QAReport

    return QAReport(datetime.now(UTC), ())


def _zipline_available() -> bool:
    from edgestack.backtest.confirm import zipline_available

    return zipline_available()


def _annual_sharpe(values: pd.Series) -> float:
    finite = values.to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2 or finite.std(ddof=1) == 0:
        return 0.0
    return float(finite.mean() / finite.std(ddof=1) * math.sqrt(252))


def _compounded_return(values: pd.Series) -> float:
    """Compound one annual stream without inventing inactive observations."""

    selected = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    selected = selected[np.isfinite(selected)]
    return float(np.prod(1.0 + selected) - 1.0) if selected.size else math.nan


def _periodic_sharpe(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or finite.std(ddof=1) == 0:
        return 0.0
    return float(finite.mean() / finite.std(ddof=1))


def _overlay_incremental_stream(
    composite: pd.Series,
    condition: pd.Series,
    *,
    cost_model: CostModel,
    adv_dollars: pd.Series,
    cost_multiplier: float = 1.0,
    order_dollars: float = 10_000.0,
) -> pd.Series:
    """Apply one causal daily filter with the campaign's frozen cost model."""

    selected_active = (
        condition.shift(1)
        .fillna(False)
        .astype(float)
        .reindex(composite.index)
        .fillna(0.0)
    )
    liquidity = adv_dollars.reindex(composite.index).fillna(100_000_000.0)
    positions = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        selected_active.to_numpy(dtype=np.float64),
    )
    selected_cost = pd.Series(
        cost_model.portfolio_costs(
            positions,
            asset_type="etf",
            order_dollars=order_dollars,
            adv_dollars=liquidity.to_numpy(dtype=float),
            multiplier=cost_multiplier,
        ),
        index=composite.index,
    )
    overlay = composite.mul(selected_active).sub(selected_cost)
    return overlay.sub(composite).fillna(0.0)
