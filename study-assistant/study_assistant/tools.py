"""The three tools the agent chooses between.

1. search_notes        - vector search over notes / exams / answer keys, page-cited
2. solve_step_by_step  - grounded, structured step-by-step solution (a focused sub-call)
3. check_answer        - grade a student's answer against the retrieved answer key

Tools are defined with the SDK's `@beta_tool` decorator, so their JSON schemas come from
the type hints and docstrings. They are closures over a Toolbox, which holds the
retriever and the Anthropic client.
"""

from __future__ import annotations

from typing import Any, Literal

from anthropic import beta_tool
from pydantic import BaseModel

from .retrieval import Hit, Retriever, format_hits

FALLBACK_BETA = "server-side-fallback-2026-07-01"


# ---------- structured outputs for the sub-calls ----------


class SolveStep(BaseModel):
    n: int
    explanation: str


class SolveResult(BaseModel):
    steps: list[SolveStep]
    final_answer: str
    sources: list[str]  # citations like "[os_notes.pdf p.3]" that the steps relied on
    used_outside_knowledge: bool


class GradeResult(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    score_out_of_10: int
    feedback: str
    missing_points: list[str]
    key_source: str  # citation of the answer-key passage used, e.g. "[midterm_key.md p.2]"


SOLVER_SYSTEM = """You are a careful CS tutor writing a worked solution.
Solve the problem step by step, one idea per step, showing intermediate results.
Ground your steps in the provided course material when it is relevant and list the
citations (exactly as written in the material headers, e.g. "[file.pdf p.3]") you relied on.
If the material does not cover something, you may use general CS knowledge, but set
used_outside_knowledge to true. Never invent citations."""

GRADER_SYSTEM = """You are a fair, strict exam grader.
Grade the student's answer ONLY against the provided answer key passages.
Give partial credit for partially correct reasoning. List concrete points from the key that
the student missed. key_source must be the citation header (e.g. "[key.md p.2]") of the
passage you graded against. If the passages don't actually contain the answer to this
question, set verdict to "incorrect", score 0, and say in feedback that no matching key was found."""


def _dedupe(hits: list[Hit]) -> list[Hit]:
    seen: set[tuple[str, int, str]] = set()
    out = []
    for h in hits:
        key = (h.source, h.page, h.text)
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


class Toolbox:
    def __init__(
        self,
        retriever: Retriever,
        client: Any,
        model: str,
        effort: str = "medium",
        min_key_score: float = 0.35,
        use_fallbacks: bool = True,
    ):
        self.retriever = retriever
        self.client = client
        self.model = model
        self.effort = effort
        self.min_key_score = min_key_score
        self.use_fallbacks = use_fallbacks
        self.tools = self._build_tools()
        self.by_name = {t.name: t for t in self.tools}

    # Shared request options for sub-calls (and the agent loop).
    def request_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "model": self.model,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if self.use_fallbacks:
            opts["betas"] = [FALLBACK_BETA]
            opts["fallbacks"] = "default"
        return opts

    def _parse(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
        resp = self.client.beta.messages.parse(
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            **self.request_options(),
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return None
        return resp.parsed_output

    def _build_tools(self):
        toolbox = self

        @beta_tool
        def search_notes(
            query: str,
            kind: Literal["any", "notes", "exam", "answer_key"] = "any",
            k: int = 5,
        ) -> str:
            """Search the student's indexed CS notes, past exam papers and answer keys.

            Use this for any factual or conceptual question about course material. Every
            returned passage starts with a citation like [file.pdf p.3]; cite it in your answer.

            Args:
                query: A focused search query (rephrase the question into key terms if helpful).
                kind: Restrict to one collection: notes, exam (past papers), answer_key, or any.
                k: Number of passages to return (1-10).
            """
            k = max(1, min(int(k), 10))
            kinds = None if kind == "any" else [kind]
            return format_hits(toolbox.retriever.search(query, k=k, kinds=kinds))

        @beta_tool
        def solve_step_by_step(problem: str) -> str:
            """Produce a worked, step-by-step solution to a problem (algorithms, complexity,
            proofs, numeric exercises, exam questions), grounded in the student's notes and
            past worked solutions where possible.

            Args:
                problem: The full problem statement, including any numbers or constraints.
            """
            hits = _dedupe(
                toolbox.retriever.search(problem, k=4, kinds=["notes"])
                + toolbox.retriever.search(problem, k=3, kinds=["exam", "answer_key"])
            )
            user = f"Course material:\n\n{format_hits(hits)}\n\n=====\n\nProblem:\n{problem}"
            result = toolbox._parse(SOLVER_SYSTEM, user, SolveResult)
            if result is None:
                return "The solver could not produce a solution for this problem."
            assert isinstance(result, SolveResult)
            lines = [f"Step {s.n}: {s.explanation}" for s in result.steps]
            lines.append(f"Final answer: {result.final_answer}")
            lines.append("Sources: " + (", ".join(result.sources) if result.sources else "none"))
            if result.used_outside_knowledge:
                lines.append("Note: parts of this solution go beyond the indexed notes.")
            return "\n".join(lines)

        @beta_tool
        def check_answer(question: str, student_answer: str) -> str:
            """Check a student's answer against the official answer key / marking scheme
            from past exams, and return a verdict, score out of 10, and feedback.

            Args:
                question: The exam question being answered (as precise as possible, e.g. including the question number).
                student_answer: The student's answer, verbatim.
            """
            hits = [
                h
                for h in toolbox.retriever.search(question, k=4, kinds=["answer_key", "exam"])
                if h.score >= toolbox.min_key_score
            ]
            if not any(h.kind == "answer_key" for h in hits):
                return (
                    "NO_KEY_FOUND: no answer-key passage matched this question, so it can't be "
                    "graded against the key. Ask the student which exam/question this is, or offer "
                    "to solve it step by step instead."
                )
            user = (
                f"Answer key / exam passages:\n\n{format_hits(hits)}\n\n=====\n\n"
                f"Question:\n{question}\n\nStudent answer:\n{student_answer}"
            )
            result = toolbox._parse(GRADER_SYSTEM, user, GradeResult)
            if result is None:
                return "The grader could not grade this answer."
            assert isinstance(result, GradeResult)
            missing = "\n".join(f"- {m}" for m in result.missing_points) or "- none"
            return (
                f"Verdict: {result.verdict} ({result.score_out_of_10}/10)\n"
                f"Feedback: {result.feedback}\n"
                f"Missing points:\n{missing}\n"
                f"Graded against: {result.key_source}"
            )

        return [search_notes, solve_step_by_step, check_answer]
