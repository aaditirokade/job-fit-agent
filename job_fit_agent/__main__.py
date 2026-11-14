"""
Command-line entry point.

    python -m job_fit_agent --resume my_resume.md --job https://example.com/job/123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import DEFAULT_MODEL, run_agent
from .tools import Toolbox, load_posting


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a job posting against your resume.")
    p.add_argument("--resume", required=True, help="Path to your resume (.md or .txt)")
    p.add_argument("--job", required=True, help="Job posting: a file path or an http(s) URL")
    p.add_argument("--name", default=None, help="Report file name (default: from the job source)")
    p.add_argument("--out", default="reports", help="Folder for reports (default: reports/)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    args = p.parse_args(argv)

    try:
        import anthropic
    except ImportError:
        print("Missing dependency. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    resume = Path(args.resume).read_text(encoding="utf-8")
    posting = load_posting(args.job)
    name = args.name or Path(args.job.rstrip("/")).stem or "fit-report"
    toolbox = Toolbox(resume, posting, report_dir=Path(args.out), report_name=name)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    result = run_agent(client, toolbox, model=args.model, on_event=print)

    print("\n" + result.final_text)
    print(f"\n[{result.turns} turns, {len(result.tool_calls)} tool calls]")
    if result.report_path:
        print(f"Report: {result.report_path}")
    return 2 if result.stopped_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
