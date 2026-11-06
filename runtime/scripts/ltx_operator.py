"""Command-line entry point for the LTX operator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltx_operator_lib.cli import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli())
