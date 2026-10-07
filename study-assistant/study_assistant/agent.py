"""The agent loop: Claude picks a tool, we run it, feed the result back, repeat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from .tools import Toolbox

SYSTEM_PROMPT = """You are a study assistant for a computer-science student. You have three tools:

- search_notes: search the student's own notes, past exam papers and answer keys.
- solve_step_by_step: produce a worked solution to a problem or exercise.
- check_answer: grade the student's answer to an exam question against the answer key.

How to work:
- For conceptual or factual questions, search the notes first and answer from what you find.
- When the student asks you to solve, derive, trace or compute something, use solve_step_by_step.
- When the student gives their own answer and wants it checked or marked, use check_answer.
  Never make up a grade: if check_answer reports NO_KEY_FOUND, say so.
- Cite every claim taken from the notes with the citation exactly as it appears in the tool
  output, e.g. [os_notes.pdf p.3]. Do not invent citations or page numbers.
- If the notes don't cover the question, say that clearly, then (if helpful) answer from
  general knowledge and label it as such.
- Be concise and student-friendly. Prefer short paragraphs and lists."""


@dataclass
class AgentReply:
    text: str
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str | None = None


class StudyAgent:
    """Multi-turn agent. Keeps the full conversation (including tool turns) in `history`,
    appending only, so thinking blocks and the prompt cache stay valid across turns."""

    def __init__(
        self,
        toolbox: Toolbox,
        max_tokens: int = 16000,
        max_iterations: int = 8,
        on_tool: Callable[[str, dict], None] | None = None,
    ):
        self.toolbox = toolbox
        self.client = toolbox.client
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.on_tool = on_tool
        self.history: list[dict[str, Any]] = []
        self._tool_defs = [t.to_dict() for t in toolbox.tools]

    def _run_tool(self, block: Any) -> dict[str, Any]:
        tool = self.toolbox.by_name.get(block.name)
        if self.on_tool:
            self.on_tool(block.name, block.input)
        if tool is None:
            return {"type": "tool_result", "tool_use_id": block.id,
                    "content": f"Unknown tool {block.name}", "is_error": True}
        try:
            output = tool.call(block.input)
            return {"type": "tool_result", "tool_use_id": block.id, "content": str(output)}
        except Exception as exc:  # report tool failures back to the model instead of crashing
            return {"type": "tool_result", "tool_use_id": block.id,
                    "content": f"Tool error: {exc}", "is_error": True}

    def ask(self, question: str) -> AgentReply:
        self.history.append({"role": "user", "content": question})
        tools_used: list[str] = []
        response = None

        for _ in range(self.max_iterations):
            response = self.client.beta.messages.create(
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=self._tool_defs,
                messages=self.history,
                cache_control={"type": "ephemeral"},  # caches tools + system + history prefix
                **self.toolbox.request_options(),
            )
            content = list(response.content) or [{"type": "text", "text": "(no response)"}]
            self.history.append({"role": "assistant", "content": content})

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "tool_use" and tool_calls:
                # Run every call, return all results in ONE user message (keeps parallel calls working).
                results = [self._run_tool(b) for b in tool_calls]
                tools_used.extend(b.name for b in tool_calls)
                self.history.append({"role": "user", "content": results})
                continue
            if response.stop_reason == "pause_turn":
                continue  # history ends with the paused assistant turn; resend to resume
            break
        else:
            return self._finish(response, tools_used, note="(stopped after too many tool steps)")

        return self._finish(response, tools_used)

    def _finish(self, response: Any, tools_used: list[str], note: str = "") -> AgentReply:
        # If the turn was cut off with tool calls pending, answer them so the history stays valid.
        pending = [b for b in response.content if b.type == "tool_use"]
        if pending and self.history[-1]["role"] == "assistant":
            self.history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": "Not run: turn ended.",
                 "is_error": True} for b in pending
            ]})

        if response.stop_reason == "refusal":
            text = "Sorry, I can't help with that request."
        else:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "max_tokens":
                text += "\n\n(answer truncated: hit the output token limit)"
        if note:
            text = f"{text}\n\n{note}".strip()
        return AgentReply(text=text, tools_used=tools_used, stop_reason=response.stop_reason)

    def reset(self) -> None:
        self.history.clear()


def friendly_api_error(exc: Exception) -> str:
    """Most-specific-first mapping of SDK errors to a message for the CLI."""
    no_credentials = isinstance(exc, TypeError) and "authentication method" in str(exc)
    if no_credentials:
        return ("No Anthropic API key found. Put ANTHROPIC_API_KEY=sk-ant-... in study-assistant/.env "
                "(copy .env.example) or export it as an environment variable.")
    if isinstance(exc, anthropic.AuthenticationError):
        return ("Your Anthropic API key was rejected (401). Check that ANTHROPIC_API_KEY is copied "
                "correctly and is still active at console.anthropic.com.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"Your API key doesn't have access to this request (403): {exc.message}"
    if isinstance(exc, anthropic.NotFoundError):
        return f"Model not found (404): check CLAUDE_MODEL. {exc.message}"
    if isinstance(exc, anthropic.RateLimitError):
        retry = exc.response.headers.get("retry-after", "a few")
        return f"Rate limited by the API; retry in {retry} seconds."
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API (network error)."
    return f"Unexpected error: {exc}"
