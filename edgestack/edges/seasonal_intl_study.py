"""Preholdout evaluator for the international Halloween retest family.

Implements configs/seasonal-intl-v1.yaml exactly: the FROZEN Halloween rule
from the failed US family (long November through April, month-end MOC flips
only) applied, unchanged, to 12 US-listed country ETFs the original test
never touched — the independent-data second chance for the program's
closest miss. Stream construction is reused verbatim from the US module;
only the instrument list and the sealed panel differ. The forward holdout
window is never read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import yaml

from edgestack.data.intl_panel import load_panel
from edgestack.disclaimer import DISCLAIMER
from edgestack.edges._study_common import evaluate_family
from edgestack.edges.seasonal_study import build_streams


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("intl seasonal configuration must be a mapping")
    family = cast(Mapping[str, Any], payload["declared_family"])
    if int(family["real_trial_count"]) != 12:
        raise ValueError("declared trial count does not match the preregistration")
    return cast(dict[str, Any], payload)


def run_preholdout(config_path: str | Path, *, root: str | Path = ".") -> Path:
    base = Path(root).resolve()
    config = _load_config(base / config_path)
    forward_start = date.fromisoformat(
        str(cast(Mapping[str, Any], config["holdout"])["start"])
    )
    panel = load_panel(base)
    gross, net, definitions, benchmark = build_streams(config, panel, forward_start)

    def rebuild(end_exclusive: date) -> Any:
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
    parser.add_argument("--config", default="configs/seasonal-intl-v1.yaml")
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
