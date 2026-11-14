"""
The agent loop.

How it works, in one paragraph:
We send the model a goal, a system prompt and a list of tools. The model
answers either with text (it's done) or with one or more "tool_use" requests.
For each request we run the matching Python function and send the result back
as a "tool_result". We repeat until the model stops asking for tools or we hit
a turn limit. The model decides which tools to call and in what order; our
code only executes them and enforces the rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .tools import TOOL_SCHEMAS, Toolbox

DEFAULT_MODEL = "claude-sonnet-5-5"  # check https://docs.claude.com for current model names
MAX_TURNS = 12  # hard stop so a confused model can't loop forever (or run up a bill)

SYSTEM_PROMPT = """\
You are a careful job-fit analyst helping a candidate decide whether a role is
worth applying to, and how to tailor their resume for it.

Work like this:
1. Read the job posting and the resume with your tools.
2. Pull out the posting's must-have and nice-to-have requirements, then run
   check_requirements on them to see what the resume literally supports.
3. Judge fit. Literal matches are evidence; so is clearly transferable
   experience, but say why it transfers.
4. Write the report and save it with save_report.

Honesty rules (these matter more than sounding positive):
- Never invent experience, employers, numbers or skills. Every tailored bullet
  must be a rewording of something already in the resume, and must cite the
  resume line number(s) it comes from, like (L12).
- If a must-have is missing, say so plainly under Gaps. Don't paper over it.
- If the posting is ambiguous (for example, seniority or location), say so
  rather than guessing.

Report format (Markdown):
# Fit report: <role> at <company>
**Fit score:** <0-100> — <one-line verdict: Strong / Worth a shot / Long shot>
## Why it fits
## Gaps and risks
## Tailored resume bullets
## Questions to ask the recruiter
"""


@dataclass
class AgentResult:
    final_text: str
    turns: int
    tool_calls: list[dict] = field(default_factory=list)
    report_path: str | None = None
    stopped_early: bool = False


def _get(block: Any, key: str) -> Any:
    """Read a field from an SDK object or a plain dict (the tests use dicts)."""
    return block[key] if isinstance(block, dict) else getattr(block, key)


def run_agent(
    client: Any,
    toolbox: Toolbox,
    goal: str = "Evaluate my fit for this job and save a report.",
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TURNS,
    on_event: Callable[[str], None] | None = None,
) -> AgentResult:
    """Run the tool-calling loop until the model finishes or max_turns is hit."""
    log = on_event or (lambda _msg: None)
    messages: list[dict] = [{"role": "user", "content": goal}]
    tool_calls: list[dict] = []

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        content = _get(response, "content")
        # Keep the model's turn in the history so it remembers what it asked for.
        messages.append({"role": "assistant", "content": content})

        tool_uses = [b for b in content if _get(b, "type") == "tool_use"]
        if _get(response, "stop_reason") != "tool_use" or not tool_uses:
            text = "".join(_get(b, "text") for b in content if _get(b, "type") == "text")
            return AgentResult(
                final_text=text.strip(),
                turns=turn,
                tool_calls=tool_calls,
                report_path=str(toolbox.saved_report) if toolbox.saved_report else None,
            )

        # Run every tool the model asked for, and answer each one by id.
        results = []
        for use in tool_uses:
            name, args, use_id = _get(use, "name"), _get(use, "input") or {}, _get(use, "id")
            log(f"→ {name}({', '.join(args)})")
            try:
                output, is_error = toolbox.run(name, args), False
            except Exception as exc:  # report the error to the model so it can recover
                output, is_error = f"Error: {exc}", True
                log(f"  ✗ {output}")
            tool_calls.append({"name": name, "input": args, "is_error": is_error})
            results.append(
                {"type": "tool_result", "tool_use_id": use_id, "content": output, "is_error": is_error}
            )
        messages.append({"role": "user", "content": results})

    return AgentResult(
        final_text="Stopped: reached the turn limit before the agent finished.",
        turns=max_turns,
        tool_calls=tool_calls,
        report_path=str(toolbox.saved_report) if toolbox.saved_report else None,
        stopped_early=True,
    )
