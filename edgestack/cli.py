"""EdgeStack command-line interface with hard campaign-gate ordering."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from edgestack.config import load_config
from edgestack.disclaimer import DISCLAIMER
from edgestack.live.demo import run as run_demo
from edgestack.logging import configure_logging
from edgestack.models import GateStatus

app = typer.Typer(
    name="edgestack",
    no_args_is_help=True,
    help="Reproducible statistical-edge research and paper-assistant engine.",
    epilog=DISCLAIMER,
)
console = Console()


@app.callback()
def main(
    json_logs: Annotated[
        bool, typer.Option("--json-logs", help="Emit structured JSON logs.")
    ] = False,
) -> None:
    """Print the mandatory disclosure and configure structured logging."""

    configure_logging(json_output=json_logs)
    console.print(f"[bold red]{DISCLAIMER}[/bold red]")


@app.command()
def ingest(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            metavar="YYYY-MM-DD",
            help="Last market date considered by the campaign.",
        ),
    ] = None,
    campaign_id: Annotated[str | None, typer.Option("--campaign-id")] = None,
) -> None:
    """Download/cache bars, run data QA, and create a campaign snapshot."""

    from edgestack.pipeline.runner import CampaignRunner

    resolved = load_config(config)
    runner = CampaignRunner.create(
        resolved,
        campaign_id=campaign_id,
        as_of=_parse_iso_date(as_of, "--as-of"),
    )
    result = runner.ingest()
    console.print(
        f"Campaign [bold]{runner.campaign_id}[/bold]: data gate {result.status.value}"
    )
    console.print(result.summary)
    _exit_on_failed_gate(result.status)


@app.command()
def replicate(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
) -> None:
    """Run the frozen known-effects pipeline-correctness suite."""

    _run_phase("replicate", campaign, config)


@app.command()
def discover(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
) -> None:
    """Enumerate and backtest the preregistered real/placebo grid."""

    _run_phase("discover", campaign, config)


@app.command("reversal-study")
def reversal_study(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/reversal-study.yaml"),
    run_ml: Annotated[
        bool,
        typer.Option(
            "--run-ml",
            help="Also run purged ridge/elastic-net/XGBoost rank diagnostics.",
        ),
    ] = False,
    gpu: Annotated[
        bool,
        typer.Option(
            "--gpu",
            help="Assign XGBoost trials across the configured independent CUDA devices.",
        ),
    ] = False,
) -> None:
    """Run the opt-in, non-holdout top-K and residual-reversal study."""

    from edgestack.pipeline.runner import CampaignRunner

    if gpu and not run_ml:
        raise typer.BadParameter("--gpu requires --run-ml")
    runner = CampaignRunner.open(load_config(config), campaign)
    result = runner.reversal_research(run_ml=run_ml, use_gpu=gpu)
    console.print(f"reversal-study: {result.status.value} — {result.summary}")
    _exit_on_failed_gate(result.status)


@app.command()
def validate(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
) -> None:
    """Run walk-forward, CPCV/PBO, stability, decay, and confirmation."""

    _run_phase("validate", campaign, config)


@app.command("report")
def report_command(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
    provisional: Annotated[bool, typer.Option("--provisional")] = False,
    finalize_holdout: Annotated[bool, typer.Option("--finalize-holdout")] = False,
) -> None:
    """Render the provisional report or consume/finalize the one-use holdout."""

    if provisional == finalize_holdout:
        raise typer.BadParameter(
            "choose exactly one of --provisional or --finalize-holdout"
        )
    phase = "finalize_holdout" if finalize_holdout else "report"
    _run_phase(phase, campaign, config)


@app.command()
def score(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
    freeze: Annotated[bool, typer.Option("--freeze")] = False,
) -> None:
    """Build the shrunk, clustered composite and freeze its full definition."""

    if not freeze:
        raise typer.BadParameter(
            "score requires --freeze to protect holdout governance"
        )
    _run_phase("score", campaign, config)


@app.command()
def live(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/full.yaml"),
    once: Annotated[
        bool, typer.Option("--once", help="Run one scan instead of scheduling.")
    ] = False,
) -> None:
    """Start the paper-only scanner/monitor after final holdout promotion."""

    from edgestack.pipeline.runner import CampaignRunner

    runner = CampaignRunner.open(load_config(config), campaign)
    result = runner.live(once=once)
    console.print(result)


@app.command("live-demo")
def live_demo(
    database: Annotated[Path, typer.Option("--database")] = Path(
        "artifacts/live_demo.sqlite"
    ),
) -> None:
    """Run an accelerated recorded day with a forced restart."""

    counts = run_demo(database)
    console.print(counts)
    if counts.get("sent") != counts.get("receiver_unique"):
        raise typer.Exit(1)


@app.command("mobile-api")
def mobile_api(
    host: Annotated[
        str,
        typer.Option(help="Bind address; use 0.0.0.0 only on a trusted network."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    token: Annotated[
        str | None,
        typer.Option(
            envvar="EDGESTACK_MOBILE_TOKEN",
            help="Bearer token. Required unless --demo is used.",
        ),
    ] = None,
    campaign: Annotated[
        str | None,
        typer.Option(
            help="Targeted campaign whose sealed mobile artifacts are served."
        ),
    ] = None,
    artifacts: Annotated[
        Path, typer.Option(file_okay=False, help="EdgeStack artifact directory.")
    ] = Path("artifacts"),
    demo: Annotated[
        bool,
        typer.Option(help="Serve packaged offline demonstration data."),
    ] = False,
) -> None:
    """Serve the read-only Android companion API; never accepts orders."""

    import uvicorn

    from edgestack.mobile.api import create_mobile_app

    if not demo and (token is None or len(token) < 24):
        raise typer.BadParameter(
            "a bearer token of at least 24 characters is required outside demo mode",
            param_hint="--token / EDGESTACK_MOBILE_TOKEN",
        )
    application = create_mobile_app(
        artifact_root=artifacts,
        campaign_id=campaign,
        bearer_token=token,
        demo=demo,
    )
    uvicorn.run(application, host=host, port=port, access_log=False)


def _run_phase(phase: str, campaign: str, config: Path) -> None:
    from edgestack.pipeline.runner import CampaignRunner

    runner = CampaignRunner.open(load_config(config), campaign)
    method = getattr(runner, phase)
    result = method()
    if hasattr(result, "status"):
        console.print(f"{phase}: {result.status.value} — {result.summary}")
        _exit_on_failed_gate(result.status)
    else:
        console.print(result)


def _exit_on_failed_gate(status: GateStatus) -> None:
    """Return a failing process status after a persisted gate has been reported."""

    if status in {GateStatus.FAIL, GateStatus.BLOCKED}:
        raise typer.Exit(code=1)


def _parse_iso_date(value: str | None, option_name: str) -> date | None:
    """Parse an ISO calendar date without relying on Typer date conversion.

    Typer/Click does not expose a native ``datetime.date`` parameter type. Keeping
    the command-line representation as text also makes the accepted format
    explicit and produces a stable user-facing error on every supported Typer
    release.
    """

    if value is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise typer.BadParameter(
            "expected an ISO date in YYYY-MM-DD format",
            param_hint=option_name,
        ) from error
    if parsed.isoformat() != value:
        raise typer.BadParameter(
            "expected an ISO date in YYYY-MM-DD format",
            param_hint=option_name,
        )
    return parsed


if __name__ == "__main__":
    app()
