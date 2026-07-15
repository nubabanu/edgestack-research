"""Backward-compatible executable demo entry point."""

from __future__ import annotations

from pathlib import Path

from edgestack.live.demo import run


def main() -> None:
    """Run the accelerated demo in the local artifacts directory."""

    print(run(Path("artifacts/live_demo.sqlite")))


if __name__ == "__main__":
    main()
