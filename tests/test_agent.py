"""
Offline tests. A scripted FakeClient stands in for the real model, so these
run with no API key and no network, and exercise the real loop and tools.
"""

from pathlib import Path

import pytest

from job_fit_agent import Toolbox, run_agent
from job_fit_agent.tools import html_to_text

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class FakeClient:
    """Returns pre-written responses in order and records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **kwargs):
        # Snapshot the message list, because the agent keeps appending to it.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def tool_use(id_, name, input_=None):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_ or {}}


def reply(*blocks, stop="tool_use"):
    return {"content": list(blocks), "stop_reason": stop}


@pytest.fixture
def toolbox(tmp_path):
    return Toolbox(
        resume_text=(EXAMPLES / "sample_resume.md").read_text(),
        posting_text=(EXAMPLES / "sample_job.md").read_text(),
        report_dir=tmp_path,
        report_name="Contoso Staff Designer",
    )


def test_full_run_calls_tools_and_saves_report(toolbox):
    client = FakeClient([
        reply(tool_use("t1", "get_job_posting"), tool_use("t2", "get_resume")),
        reply(tool_use("t3", "check_requirements", {"terms": ["design system", "SQL", "Figma"]})),
        reply(tool_use("t4", "save_report", {"markdown": "# Fit report\n**Fit score:** 78"})),
        reply({"type": "text", "text": "Done. Fit score 78."}, stop="end_turn"),
    ])
    result = run_agent(client, toolbox)

    assert result.final_text == "Done. Fit score 78."
    assert result.turns == 4
    assert [c["name"] for c in result.tool_calls] == [
        "get_job_posting", "get_resume", "check_requirements", "save_report"
    ]
    assert Path(result.report_path).read_text().startswith("# Fit report")
    assert Path(result.report_path).name == "contoso-staff-designer.md"

    # Both parallel tool calls were answered in ONE user message, matched by id.
    second_request = client.requests[1]["messages"]
    answers = second_request[-1]["content"]
    assert [a["tool_use_id"] for a in answers] == ["t1", "t2"]
    assert "Contoso" in answers[0]["content"]
    assert "  1: # Jordan Lee" in answers[1]["content"]


def test_check_requirements_is_literal_and_cites_lines(toolbox):
    out = toolbox.check_requirements(["design system", "SQL", "UI", "figma"])
    by_term = {r["term"]: r for r in out["results"]}
    assert by_term["design system"]["found"] and by_term["design system"]["resume_lines"]
    assert not by_term["SQL"]["found"]
    assert not by_term["UI"]["found"]          # must not match inside "Built" or "build"
    assert by_term["figma"]["found"]           # case-insensitive
    assert out["matched"] == 2 and out["total"] == 4


def test_bad_tool_call_is_reported_back_not_crashed(toolbox):
    client = FakeClient([
        reply(tool_use("t1", "check_requirements", {"terms": "not a list"})),
        reply(tool_use("t2", "delete_everything")),
        reply({"type": "text", "text": "Recovered."}, stop="end_turn"),
    ])
    result = run_agent(client, toolbox)

    assert result.final_text == "Recovered."
    assert all(c["is_error"] for c in result.tool_calls)
    err = client.requests[1]["messages"][-1]["content"][0]
    assert err["is_error"] is True and "must be a list" in err["content"]


def test_turn_limit_stops_a_looping_model(toolbox):
    client = FakeClient([reply(tool_use(f"t{i}", "get_resume")) for i in range(5)])
    result = run_agent(client, toolbox, max_turns=3)
    assert result.stopped_early
    assert result.turns == 3
    assert len(client.requests) == 3


def test_html_to_text_strips_markup_and_scripts():
    raw = "<html><script>evil()</script><h1>Designer</h1><p>Figma &amp; React</p></html>"
    text = html_to_text(raw)
    assert "evil" not in text
    assert "Designer" in text and "Figma & React" in text
