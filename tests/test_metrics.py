from ragbench.eval.metrics import (
    parse_citation_range,
    retrieval_recall_at_k,
    citation_hit,
    citation_precision,
    citation_recall,
    numeric_fidelity,
)


def test_parse_citation():
    assert parse_citation_range("file.pdf p3") == ("file.pdf", 3, 3)
    assert parse_citation_range("file.pdf p4-5") == ("file.pdf", 4, 5)


def test_retrieval_recall():
    retrieved = [{"pdf_filename": "file.pdf", "page_start": 2, "page_end": 4}]
    assert retrieval_recall_at_k(retrieved, ["file.pdf p3"])
    assert not retrieval_recall_at_k(retrieved, ["other.pdf p3"])


def test_citation_hit():
    assert citation_hit(["file.pdf p2"], ["file.pdf p2"])
    assert not citation_hit(["file.pdf p2"], ["file.pdf p3"])


def test_citation_precision_recall():
    assert citation_precision(["file.pdf p2"], ["file.pdf p2"]) == 1.0
    assert citation_recall(["file.pdf p2"], ["file.pdf p2", "file.pdf p3"]) == 0.5


def test_numeric_fidelity():
    exact, f1 = numeric_fidelity("Revenue was $10.0 billion.", "Revenue was $10.0 billion.")
    assert exact is True
    assert f1 == 1.0
