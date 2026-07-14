from pathlib import Path
from zipfile import ZipFile

from mnemosyne.config import Settings
from mnemosyne.ingest import chunk_document, parse
from mnemosyne.models import SearchHit, SourceDocument
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


def test_saved_searches_watch_folders_and_backup(tmp_path: Path):
    notes = tmp_path / "vault"
    notes.mkdir()
    (notes / "ideas.md").write_text("#rag Local watch folders should reindex changes.")
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", tmp_path / "data" / "settings.json")
    kb = KnowledgeBase(settings)
    kb.register_watch_folder(notes, "obsidian")
    kb.save_search("rag search", "local watch folders", "rag", None, None)
    prefs = kb.save_settings({"privacy_mode": "strict-local", "embed_provider": "ollama"})
    backup = kb.backup()

    assert prefs["privacy_mode"] == "strict-local"
    assert backup["saved_searches"]
    assert backup["watch_folders"]
    assert backup["documents"]
    assert "diagnostics" in backup


def test_reader_and_entities(tmp_path: Path):
    note = tmp_path / "timeline.md"
    note.write_text("OpenAI Research met on July 14, 2026. The plan was not approved at first.")
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", tmp_path / "data" / "settings.json")
    kb = KnowledgeBase(settings)
    kb.ingest(note)
    reader = kb.reader(str(note.resolve()))

    assert reader["chunks"]
    assert reader["entities"]
    assert reader["timeline"]


def test_empty_file_logs_parsing_diagnostic(tmp_path: Path):
    note = tmp_path / "empty.md"
    note.write_text("")
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db")
    kb = KnowledgeBase(settings)

    assert kb.ingest(note) == (0, 0)
    diagnostics = kb.diagnostics()
    assert diagnostics
    assert diagnostics[0]["code"] == "empty_chunks"


def test_missing_watch_folder_is_rejected(tmp_path: Path):
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db")
    kb = KnowledgeBase(settings)
    missing = tmp_path / "does-not-exist"

    try:
        kb.register_watch_folder(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing watch folder should fail")


def test_xlsx_parser_extracts_shared_strings(tmp_path: Path):
    workbook = tmp_path / "sheet.xlsx"
    with ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>Hybrid retrieval</t></si>
              <si><t>Citation audit</t></si>
            </sst>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>
              </sheetData>
            </worksheet>""",
        )

    docs = parse(workbook)
    assert docs
    assert "Hybrid retrieval" in docs[0].text
    assert "Citation audit" in docs[0].text
