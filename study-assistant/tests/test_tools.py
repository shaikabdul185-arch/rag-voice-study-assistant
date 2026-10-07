from study_assistant.agent import StudyAgent
from study_assistant.retrieval import Hit
from study_assistant.tools import GradeResult, SolveResult, SolveStep, Toolbox

from conftest import FakeClient, FakeRetriever, text_block, tool_use
from types import SimpleNamespace as NS


def make(hits, **client_kw):
    client = FakeClient(**client_kw)
    retriever = FakeRetriever(hits)
    return Toolbox(retriever, client, "claude-opus-5-5"), client, retriever


def test_tool_schemas(hits):
    tb, _, _ = make(hits)
    defs = {d["name"]: d for d in (t.to_dict() for t in tb.tools)}
    assert set(defs) == {"search_notes", "solve_step_by_step", "check_answer"}
    kind = defs["search_notes"]["input_schema"]["properties"]["kind"]
    assert set(kind["enum"]) == {"any", "notes", "exam", "answer_key"}
    assert defs["check_answer"]["input_schema"]["required"] == ["question", "student_answer"]


def test_search_notes_filters_kind(hits):
    tb, _, retriever = make(hits)
    out = tb.by_name["search_notes"].call({"query": "deadlock", "kind": "answer_key", "k": 50})
    assert "[midterm_2025_key.md p.1]" in out and "data_structures" not in out
    assert retriever.calls == [("deadlock", 10, ["answer_key"])]  # k clamped to 10


def test_check_answer_without_key_does_not_grade(hits):
    tb, client, _ = make([h for h in hits if h.kind != "answer_key"])
    out = tb.by_name["check_answer"].call({"question": "Q1", "student_answer": "idk"})
    assert out.startswith("NO_KEY_FOUND")
    assert client.beta.messages.parse_calls == []


def test_check_answer_low_similarity_key_is_ignored():
    weak = [Hit("k.md", 1, "answer_key", "unrelated", 0.10)]
    tb, client, _ = make(weak)
    assert tb.by_name["check_answer"].call({"question": "Q9", "student_answer": "x"}).startswith("NO_KEY_FOUND")


def test_check_answer_grades_with_key(hits):
    grade = GradeResult(verdict="partially_correct", score_out_of_10=6, feedback="Missed circular wait.",
                        missing_points=["circular wait"], key_source="[midterm_2025_key.md p.1]")
    tb, client, _ = make(hits, parse_outputs=[grade])
    out = tb.by_name["check_answer"].call({"question": "Q1 deadlock", "student_answer": "mutual exclusion..."})
    assert "partially_correct (6/10)" in out and "- circular wait" in out
    call = client.beta.messages.parse_calls[0]
    assert call["output_format"] is GradeResult
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "[midterm_2025_key.md p.1]" in call["messages"][0]["content"]


def test_solve_step_by_step_renders_steps(hits):
    sol = SolveResult(steps=[SolveStep(n=1, explanation="FCFS waits 0, 24, 27"),
                             SolveStep(n=2, explanation="Average = 17")],
                      final_answer="17", sources=["[midterm_2025_key.md p.1]"], used_outside_knowledge=False)
    tb, client, _ = make(hits, parse_outputs=[sol])
    out = tb.by_name["solve_step_by_step"].call({"problem": "FCFS average waiting time for 24,3,3"})
    assert out.splitlines()[:3] == ["Step 1: FCFS waits 0, 24, 27", "Step 2: Average = 17", "Final answer: 17"]


def test_agent_loop_runs_tool_and_keeps_history(hits):
    responses = [
        NS(stop_reason="tool_use", content=[text_block("Let me check."),
                                            tool_use("tu_1", "search_notes", {"query": "B-tree height"})]),
        NS(stop_reason="end_turn", content=[text_block("Height is O(log_t n) [data_structures.md p.2].")]),
        NS(stop_reason="end_turn", content=[text_block("Sure.")]),
    ]
    tb, client, _ = make(hits, create_responses=responses)
    seen = []
    agent = StudyAgent(tb, on_tool=lambda name, args: seen.append(name))

    reply = agent.ask("What is the height of a B-tree?")
    assert reply.text == "Height is O(log_t n) [data_structures.md p.2]."
    assert reply.tools_used == ["search_notes"] and seen == ["search_notes"]

    second = client.beta.messages.create_calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    result = second[2]["content"][0]
    assert result["tool_use_id"] == "tu_1" and "[data_structures.md p.2]" in result["content"]

    agent.ask("Thanks, and in a B+ tree?")  # multi-turn: history carried over
    third = client.beta.messages.create_calls[2]["messages"]
    assert [m["role"] for m in third] == ["user", "assistant", "user", "assistant", "user"]
    req = client.beta.messages.create_calls[0]
    assert req["model"] == "claude-opus-5-5" and req["thinking"] == {"type": "adaptive"}
    assert req["cache_control"] == {"type": "ephemeral"}


def test_agent_reports_tool_errors_to_model(hits):
    responses = [
        NS(stop_reason="tool_use", content=[tool_use("tu_1", "no_such_tool", {})]),
        NS(stop_reason="end_turn", content=[text_block("ok")]),
    ]
    tb, client, _ = make(hits, create_responses=responses)
    StudyAgent(tb).ask("hi")
    result = client.beta.messages.create_calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_agent_handles_refusal(hits):
    tb, _, _ = make(hits, create_responses=[NS(stop_reason="refusal", content=[])])
    agent = StudyAgent(tb)
    reply = agent.ask("something")
    assert reply.stop_reason == "refusal" and "can't help" in reply.text
    assert agent.history[-1]["content"]  # never append an empty assistant turn


def test_agent_answers_pending_tool_calls_when_truncated(hits):
    responses = [NS(stop_reason="max_tokens", content=[tool_use("tu_9", "search_notes", {"query": "x"})])]
    tb, _, _ = make(hits, create_responses=responses)
    agent = StudyAgent(tb)
    reply = agent.ask("q")
    assert "truncated" in reply.text
    assert agent.history[-1]["role"] == "user"
    assert agent.history[-1]["content"][0]["tool_use_id"] == "tu_9"
