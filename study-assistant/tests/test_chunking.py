from pathlib import Path

from study_assistant.ingest import chunk_document, chunk_page, infer_kind, load_pages


def test_pages_split_on_markers(tmp_path: Path):
    f = tmp_path / "n.md"
    f.write_text("page one\n\n---\n\npage two\fpage three")
    pages = load_pages(f)
    assert [p.strip() for p in pages] == ["page one", "page two", "page three"]


def test_chunks_never_cross_pages():
    pages = ["alpha " * 300, "beta " * 300]
    chunks = chunk_document(pages, size=400, overlap=50)
    for c in chunks:
        words = set(c.text.split())
        assert words == ({"alpha"} if c.page == 1 else {"beta"})
    assert {c.page for c in chunks} == {1, 2}
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_size_and_overlap():
    paras = [f"Paragraph {i} " + "word " * 40 for i in range(10)]
    chunks = chunk_page("\n\n".join(paras), size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # The start of each later chunk repeats text from the end of the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[0] in prev


def test_long_paragraph_is_split():
    chunks = chunk_page("x" * 2500, size=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_empty_page_gives_no_chunks():
    assert chunk_page("   \n  ") == []


def test_infer_kind():
    assert infer_kind(Path("data/notes/os.pdf")) == "notes"
    assert infer_kind(Path("data/exams/mid.pdf")) == "exam"
    assert infer_kind(Path("data/keys/mid.pdf")) == "answer_key"
    assert infer_kind(Path("misc/final_2024_solutions.pdf")) == "answer_key"
    assert infer_kind(Path("misc/random.pdf")) is None
