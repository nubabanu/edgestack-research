"""Complete self-contained Edge Verdict Report renderer."""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from edgestack.disclaimer import DISCLAIMER
from edgestack.models import VerdictRecord


def verdict_rows(records: Iterable[VerdictRecord]) -> list[dict[str, Any]]:
    """Flatten verdicts and evidence without dropping failed strategies."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "hypothesis_id": record.hypothesis_id,
            "execution_status": record.execution_status.value,
            "verdict": record.verdict.value if record.verdict else "NOT_EVALUATED",
            "decay": record.decay.value,
            "reasons": " | ".join(record.reasons),
            "provisional": record.provisional,
            "bias_tier": record.bias_tier,
            "disclaimer": DISCLAIMER,
        }
        if record.evidence is not None:
            evidence = asdict(record.evidence)
            annotations = evidence.pop("annotations", {})
            row.update(evidence)
            row["annotations"] = json.dumps(annotations, sort_keys=True, default=str)
        rows.append(row)
    return rows


def render_verdict_report(
    records: Iterable[VerdictRecord],
    summary: Mapping[str, Any],
    output_directory: str | Path,
    *,
    final: bool,
    embedded_figures: Mapping[str, str] | None = None,
    evidence_sections: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write exhaustive CSV and standalone HTML reports."""

    record_list = tuple(records)
    if final and any(record.provisional for record in record_list):
        raise ValueError("final report cannot contain provisional verdict records")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = verdict_rows(record_list)
    frame = pd.DataFrame(rows)
    stem = "edge_verdict_final" if final else "edge_verdict_provisional"
    csv_path = directory / f"{stem}.csv"
    html_path = directory / f"{stem}.html"
    export_frame = frame
    if frame.empty:
        export_frame = pd.DataFrame(
            [
                {
                    "hypothesis_id": "",
                    "execution_status": "NO_HYPOTHESES",
                    "verdict": "NOT_EVALUATED",
                    "reasons": "No hypotheses were declared or evaluated.",
                    "disclaimer": DISCLAIMER,
                }
            ]
        )
    export_frame.to_csv(csv_path, index=False)

    counts = frame["verdict"].value_counts().to_dict() if not frame.empty else {}
    merged_summary = dict(summary)
    merged_summary.setdefault("row_count", len(frame))
    merged_summary.setdefault("verdict_counts", counts)
    headers = list(export_frame.columns)
    table_rows = []
    for row in export_frame.fillna("").to_dict(orient="records"):
        cells = "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in headers
        )
        table_rows.append(f"<tr>{cells}</tr>")
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
    summary_html = "".join(
        f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>"
        for key, value in sorted(merged_summary.items(), key=lambda item: str(item[0]))
    )
    warning = (
        "SURVIVORSHIP-BIASED FREE DATA"
        if any(row.get("bias_tier") == "SURVIVORSHIP_BIASED" for row in rows)
        else ""
    )
    figures_html = "".join(
        f"<section><h2>{html.escape(str(title))}</h2>"
        f"<img alt='{html.escape(str(title))}' src='{html.escape(data_uri)}'></section>"
        for title, data_uri in (embedded_figures or {}).items()
    )
    evidence_html = "".join(
        f"<section><h2>{html.escape(str(title))}</h2><pre>"
        f"{html.escape(json.dumps(value, sort_keys=True, indent=2, default=str))}"
        "</pre></section>"
        for title, value in (evidence_sections or {}).items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>EdgeStack Verdict Report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#17202a}} .warning{{background:#7b241c;color:white;padding:1rem;font-weight:700}}
.disclaimer{{border:2px solid #b03a2e;padding:1rem;background:#fdf2e9}} dl{{display:grid;grid-template-columns:max-content 1fr;gap:.3rem 1rem}}
.table-wrap{{overflow:auto;max-height:70vh;border:1px solid #bbb}} table{{border-collapse:collapse;font-size:.75rem;white-space:nowrap}}
th,td{{border:1px solid #ddd;padding:.3rem}} th{{position:sticky;top:0;background:#eee}} tr:nth-child(even){{background:#fafafa}}
img{{max-width:100%;height:auto}} pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem;overflow:auto}}
</style></head><body><h1>EdgeStack {'Final' if final else 'Provisional'} Edge Verdict Report</h1>
<div class="warning">{html.escape(warning)}</div><p class="disclaimer">{html.escape(DISCLAIMER)}</p>
<h2>Campaign summary</h2><dl>{summary_html}</dl>{figures_html}{evidence_html}<h2>All declared/evaluated hypotheses</h2>
<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return html_path, csv_path
