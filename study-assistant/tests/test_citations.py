from study_assistant.retrieval import Hit, format_hits
from study_assistant.voice import speakable


def test_hit_citation():
    assert Hit("os_notes.pdf", 3, "notes", "x", 0.5).citation == "[os_notes.pdf p.3]"


def test_format_hits_includes_citations(hits):
    out = format_hits(hits)
    assert "[data_structures.md p.2] (notes, similarity 0.82)" in out
    assert "[midterm_2025_key.md p.1]" in out


def test_format_hits_empty():
    assert "No matching passages" in format_hits([])


def test_speakable_rewrites_citations_and_markdown():
    text = "**B-trees** keep height `O(log n)` [data_structures.md p.2].\n- item\n```py\nx=1\n```"
    out = speakable(text)
    assert "(from data structures, page 2)" in out
    assert "*" not in out and "`" not in out and "[" not in out
    assert "see the code on screen" in out
