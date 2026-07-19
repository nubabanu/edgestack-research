"""Preholdout evaluator for the preregistered cross-sectional momentum family.

Implements configs/momentum-xs-study-v1.yaml exactly: 3 real trials on the
sealed (survivorship-biased, stamped) equity panel — Jegadeesh-Titman
winner portfolios ranked by trailing 12-month (skip one) and 6-month (skip
one) returns, monthly equal-weight rebalance via the shared cross-sectional
contract. The forward holdout window is never read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from edgestack.disclaimer import DISCLAIMER
from edgestack.edges._study_common import evaluate_family
from edgestack.edges._xsec_common import XsecTrial, build_xsec_streams
from edgestack.edges.overnight_study import _load_panel


def _momentum_feature(skip: int, lookback: int):
    def build(
        panel: Mapping[str, pd.DataFrame], equities: Sequence[str]
    ) -> pd.DataFrame:
        adjusted = panel["adjusted_close"][list(equities)]
        return adjusted.shift(skip) / adjusted.shift(lookback) - 1.0

    return build


_TRIALS = (
    XsecTrial("momxs|mom_12_1_top_decile", _momentum_feature(21, 252), 10, 30, False),
    XsecTrial("momxs|mom_12_1_top_quintile", _momentum_feature(21, 252), 5, 30, False),
    XsecTrial("momxs|mom_6_1_top_decile", _momentum_feature(21, 126), 10, 30, False),
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("momentum study configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != len(_TRIALS):
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def build_streams(
    config: Mapping[str, Any],
    panel: Mapping[str, pd.DataFrame],
    end_exclusive: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.Series]:
    del config
    return build_xsec_streams(panel, _TRIALS, end_exclusive)


def run_preholdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    base = Path(root).resolve()
    config = _load_config(base / config_path)
    forward_start = date.fromisoformat(
        str(cast(Mapping[str, Any], config["holdout"])["start"])
    )
    panel = _load_panel(base)
    gross, net, definitions, benchmark = build_streams(config, panel, forward_start)

    def rebuild(end_exclusive: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        rebuilt_gross, rebuilt_net, _, _ = build_streams(config, panel, end_exclusive)
        return rebuilt_gross, rebuilt_net

    return evaluate_family(
        campaign_id=str(config["campaign_id"]),
        config_path=base / config_path,
        root=base,
        net=net,
        gross=gross,
        definitions=definitions,
        accounting_family_size=int(
            cast(Mapping[str, Any], config["declared_family"])["accounting_family_size"]
        ),
        forward_start=forward_start,
        rebuild=rebuild,
        benchmark=benchmark,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preholdout",))
    parser.add_argument("--config", default="configs/momentum-xs-study-v1.yaml")
    parser.add_argument("--root", default=".")
    arguments = parser.parse_args(argv)
    path = run_preholdout(arguments.config, root=arguments.root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(DISCLAIMER)
    print(
        json.dumps(
            {
                "preholdout_pass": payload["preholdout_pass"],
                "survivors": payload["survivors"],
                "family_tests": payload["family_tests"],
                "placebos": payload["placebos"],
                "result": str(path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
