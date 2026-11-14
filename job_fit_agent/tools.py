"""
Tools the agent can call.

Each tool has two halves:
  1. A JSON schema (in TOOL_SCHEMAS) that tells the model the tool exists,
     what it does, and what arguments it takes.
  2. A plain Python function that actually does the work.

Design choice: the model never gets raw file-system or network access.
The resume path and job-posting source are fixed by the person running the
CLI, and the tools only read those. That keeps the agent from wandering
into files it shouldn't see.
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Schemas the model sees
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "get_resume",
        "description": (
            "Return the candidate's resume as plain text, with line numbers. "
            "Call this before judging fit. Cite line numbers as evidence."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_job_posting",
        "description": "Return the job posting being evaluated, as plain text.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_requirements",
        "description": (
            "Deterministically check which requirement terms literally appear "
            "in the resume. Pass the key skills, tools and qualifications you "
            "extracted from the posting. Returns, for each term, whether it "
            "was found and the resume lines that mention it. Use this to "
            "ground your judgement: do not claim a match the check can't "
            "support unless you explain the transferable evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short terms, e.g. 'design systems', 'Figma', 'SQL'.",
                }
            },
            "required": ["terms"],
        },
    },
    {
        "name": "save_report",
        "description": "Save the final fit report as a Markdown file. Call once, at the end.",
        "input_schema": {
            "type": "object",
            "properties": {
                "markdown": {"type": "string", "description": "The full report."}
            },
            "required": ["markdown"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def html_to_text(raw: str) -> str:
    """Very small HTML-to-text converter (no third-party dependencies)."""
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_posting(source: str, timeout: int = 15) -> str:
    """Load a job posting from a local file or an http(s) URL."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "job-fit-agent/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return html_to_text(body)
    return Path(source).read_text(encoding="utf-8")


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len] or "report").rstrip("-")


# ---------------------------------------------------------------------------
# The toolbox: holds the inputs and runs tools by name
# ---------------------------------------------------------------------------

MAX_TOOL_OUTPUT = 20_000  # characters; keeps one huge page from flooding context


@dataclass
class Toolbox:
    resume_text: str
    posting_text: str
    report_dir: Path = Path("reports")
    report_name: str = "fit-report"
    saved_report: Path | None = field(default=None, init=False)

    # -- individual tools ---------------------------------------------------

    def get_resume(self) -> str:
        lines = self.resume_text.splitlines()
        return "\n".join(f"{i:>3}: {line}" for i, line in enumerate(lines, 1))

    def get_job_posting(self) -> str:
        return self.posting_text

    def check_requirements(self, terms: list[str]) -> dict:
        lines = self.resume_text.splitlines()
        results = []
        for term in terms:
            term = term.strip()
            if not term:
                continue
            # Word-boundary match so "UI" doesn't match "build".
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            hits = [i for i, line in enumerate(lines, 1) if pattern.search(line)]
            results.append({"term": term, "found": bool(hits), "resume_lines": hits})
        found = sum(r["found"] for r in results)
        return {"matched": found, "total": len(results), "results": results}

    def save_report(self, markdown: str) -> str:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f"{slugify(self.report_name)}.md"
        path.write_text(markdown, encoding="utf-8")
        self.saved_report = path
        return f"Saved report to {path}"

    # -- dispatcher -----------------------------------------------------------

    def run(self, name: str, args: dict) -> str:
        """Run a tool by name. Returns text for the model; raises on bad calls."""
        if name == "get_resume":
            out = self.get_resume()
        elif name == "get_job_posting":
            out = self.get_job_posting()
        elif name == "check_requirements":
            terms = args.get("terms")
            if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
                raise ValueError("'terms' must be a list of strings")
            out = json.dumps(self.check_requirements(terms), indent=2)
        elif name == "save_report":
            md = args.get("markdown")
            if not isinstance(md, str) or not md.strip():
                raise ValueError("'markdown' must be a non-empty string")
            out = self.save_report(md)
        else:
            raise ValueError(f"Unknown tool: {name}")
        if len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + "\n...[truncated]"
        return out
