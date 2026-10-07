from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import pytest

from study_assistant.retrieval import Hit


def text_block(text: str) -> NS:
    return NS(type="text", text=text)


def tool_use(id: str, name: str, input: dict) -> NS:
    return NS(type="tool_use", id=id, name=name, input=input)


class FakeMessages:
    """Stands in for client.beta.messages: returns scripted responses, records requests."""

    def __init__(self, create_responses: list[Any] | None = None, parse_outputs: list[Any] | None = None):
        self.create_responses = list(create_responses or [])
        self.parse_outputs = list(parse_outputs or [])
        self.create_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    def create(self, **kwargs):
        # Snapshot messages: the agent keeps appending to the same list.
        self.create_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.create_responses.pop(0)

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        out = self.parse_outputs.pop(0)
        return NS(stop_reason="end_turn", parsed_output=out)


class FakeClient:
    def __init__(self, **kw):
        self.beta = NS(messages=FakeMessages(**kw))


class FakeRetriever:
    def __init__(self, hits: list[Hit]):
        self.hits = hits
        self.calls: list[tuple] = []

    def search(self, query, k=5, kinds=None):
        self.calls.append((query, k, kinds))
        return [h for h in self.hits if not kinds or h.kind in kinds][:k]


@pytest.fixture
def hits() -> list[Hit]:
    return [
        Hit("data_structures.md", 2, "notes", "A B-tree of minimum degree t ...", 0.82),
        Hit("midterm_2025.md", 1, "exam", "Q1. State the four necessary conditions ...", 0.71),
        Hit("midterm_2025_key.md", 1, "answer_key", "Q1 answer key: mutual exclusion ...", 0.66),
    ]
