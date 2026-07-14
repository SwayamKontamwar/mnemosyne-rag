from pathlib import Path

from mnemosyne.config import Settings
from mnemosyne.ingest import chunk_document
from mnemosyne.models import SearchHit
from mnemosyne.models import SourceDocument
from mnemosyne.service import KnowledgeBase


def test_chunks_keep_line_citations():
    doc = SourceDocument("notes.md", "notes", "one\ntwo\nthree")
    chunks = chunk_document(doc, size=5, overlap=0)
    assert chunks[0].citation == "notes.md#L1-L2"


def test_ingest_and_search(tmp_path: Path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "retrieval.md").write_text("Hybrid retrieval combines semantic vectors and keyword search.")
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db")
    kb = KnowledgeBase(settings)
    assert kb.ingest(notes) == (1, 0)
    assert kb.ingest(notes) == (0, 1)
    hits = kb.search("semantic keyword retrieval")
    assert hits
    assert "retrieval.md" in hits[0].citation


def test_library_stats_and_document_listing(tmp_path: Path):
    note = tmp_path / "ideas.md"
    note.write_text("---\ntags:\n- rag\n- research\n---\nMnemosyne keeps citations attached to every indexed passage. [[Memory Palace]]")
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db")
    kb = KnowledgeBase(settings)
    kb.ingest(note)
    stats = kb.store.stats()
    documents = kb.store.list_documents()

    assert stats["documents"] == 1
    assert stats["chunks"] >= 1
    assert stats["characters"] > 0
    assert stats["tags"] >= 2
    assert documents[0]["path"].endswith("ideas.md")
    assert documents[0]["chunk_count"] >= 1
    assert "rag" in documents[0]["tags"]
    assert documents[0]["folder"] == tmp_path.name


def test_chunk_previews_graph_and_clusters(tmp_path: Path):
    notes = tmp_path / "vault"
    notes.mkdir()
    (notes / "retrieval.md").write_text("#rag Hybrid retrieval improves grounded answers. [[Evaluation]]")
    (notes / "evaluation.md").write_text("#rag Citation validation and answer grading improve trust.")
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db")
    kb = KnowledgeBase(settings)
    kb.ingest(notes)

    hit = kb.search("grounded answers")[0]
    preview = kb.store.chunk_preview(hit.chunk_id)
    assert preview is not None
    assert "grounded answers" in preview.text

    edges = kb.graph()
    assert edges
    clusters = kb.clusters()
    assert clusters


def test_reranking_prefers_term_overlap():
    settings = Settings(Path("/tmp/data"), Path("/tmp/data/knowledge.db"))
    kb = KnowledgeBase(settings)
    hits = [
        SearchHit(1, "generic context", "misc", "a#L1-L2", 0.9, ("other",)),
        SearchHit(2, "hybrid retrieval improves grounded answers", "retrieval", "b#L1-L2", 0.7, ("rag",)),
    ]
    reranked = kb._rerank("hybrid retrieval", hits)
    assert reranked[0].chunk_id == 2


def test_citation_validation_flags_missing_and_weak_support(tmp_path: Path):
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db")
    kb = KnowledgeBase(settings)
    hits = [
        SearchHit(1, "Hybrid retrieval improves grounded answers.", "retrieval", "a#L1-L2", 0.8, ("rag",)),
        SearchHit(2, "Citation validation audits answers.", "validation", "b#L1-L2", 0.7, ("rag",)),
    ]
    validation = kb._validate_citations("Hybrid retrieval helps [1]. Unsupported claim about OCR [2]. Missing source [3].", hits)
    assert validation.answer_has_citations is True
    assert 3 in validation.missing_numbers
    assert 2 in validation.unsupported_numbers
