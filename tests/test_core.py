import json
import asyncio
import importlib
import re
import time
from pathlib import Path
from urllib.request import Request
from zipfile import ZipFile

import pytest
import httpx

from mnemosyne.config import Settings
from mnemosyne.ingest import chunk_document, parse
from mnemosyne.models import SearchHit, SourceDocument
from mnemosyne.providers import HashingEmbedder, OllamaEmbedder, OllamaGenerator
from mnemosyne.service import KnowledgeBase


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.last_backend = "test"
        self.last_error = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            seed = sum(ord(char) for char in text)
            vectors.append([float(seed % 997), float(len(text) % 251), 1.0])
        return vectors


class ModelSpaceEmbedder:
    def __init__(self, identity: str, dimensions: int, offset: float) -> None:
        self.model_identity = identity
        self.dimensions = dimensions
        self.offset = offset

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [self.offset + (index + 1) / 10_000 for index in range(self.dimensions)]
            for _ in texts
        ]


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


def test_time_travel_search_delete_citation_and_rollback(tmp_path: Path):
    note = tmp_path / "memory.md"
    note.write_text("alpha old project plan\nshared stable line")
    kb = KnowledgeBase(Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", embed_provider="hash"))

    assert kb.ingest(note) == (1, 0)
    v1 = kb.revision_history(str(note.resolve()))[0]
    old_hit = kb.search("alpha old plan")[0]
    old_citation = old_hit.citation

    time.sleep(0.01)
    note.write_text("beta new project plan\nshared stable line")
    assert kb.ingest(note) == (1, 0)
    latest = kb.search("alpha old plan")
    historical = kb.search("alpha old plan", as_of=v1["created_at"])

    assert not any("alpha old" in hit.text for hit in latest)
    assert historical and "alpha old" in historical[0].text

    resolved = kb.store.citation_preview(old_citation)
    assert resolved is not None
    assert "alpha old project plan" in resolved.text
    assert resolved.document_version == 1

    time.sleep(0.01)
    removal = kb.store.remove_document(str(note.resolve()))
    assert removal["closed"]
    assert not kb.search("beta new project")
    assert kb.search("beta new project", as_of=kb.revision_history(str(note.resolve()))[1]["created_at"])

    assert kb.restore_revision(str(note.resolve()), 1) == (1, 0)
    versions = [revision["version"] for revision in kb.revision_history(str(note.resolve()))]
    assert versions[0] == 4
    assert kb.search("alpha old project")
    assert "beta new" in kb.revision_diff(str(note.resolve()), 2, 4)["diff"]


def test_incremental_reindex_reuses_unchanged_chunk_vectors(tmp_path: Path):
    note = tmp_path / "chunks.md"
    note.write_text("first block alpha stays\n\nsecond block beta changes")
    embedder = CountingEmbedder()
    kb = KnowledgeBase(
        Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", embed_provider="hash", chunk_size=24, chunk_overlap=0),
        embedder=embedder,
    )
    kb.ingest(note)
    first_vectors = sum(len(call) for call in embedder.calls)
    assert first_vectors == 2

    time.sleep(0.01)
    note.write_text("first block alpha stays\n\nsecond block gamma changed")
    kb.ingest(note)

    total_vectors = sum(len(call) for call in embedder.calls)
    assert total_vectors == 3
    active = kb.store.document_chunks(str(note.resolve()))
    assert len(active) == 2
    assert any("first block alpha stays" in chunk.text and chunk.document_version == 1 for chunk in active)
    assert any("second block gamma changed" in chunk.text and chunk.document_version == 2 for chunk in active)


def test_unchanged_file_hash_to_ollama_swap_replaces_actual_sqlite_vector_384_to_768(tmp_path: Path, monkeypatch):
    note = tmp_path / "unchanged.md"
    note.write_text("An unchanged note must move into the new embedding space.")
    settings = Settings(
        tmp_path / "data", tmp_path / "data" / "knowledge.db",
        embed_provider="hash", embed_model="offline-hash", vector_provider="sqlite",
    )
    first = KnowledgeBase(settings, embedder=HashingEmbedder(384))
    assert first.ingest(note) == (1, 0)
    before = first.store.connection.execute(
        "SELECT id, vector, embedding_space, embedding_dimensions FROM chunks WHERE valid_to IS NULL"
    ).fetchone()
    before_vector = json.loads(before["vector"])
    assert len(before_vector) == 384
    assert before["embedding_space"] == "hash:blake2b-v1"
    assert before["embedding_dimensions"] == 384

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({"embeddings": [[2.0 + (index + 1) / 10_000 for index in range(768)]]}).encode()

    monkeypatch.setattr("mnemosyne.providers.urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())
    second_settings = Settings(
        settings.home, settings.db_path,
        ollama_url="http://ollama.test", embed_provider="ollama", embed_model="nomic-embed-text", vector_provider="sqlite",
    )
    second = KnowledgeBase(
        second_settings, embedder=OllamaEmbedder("http://ollama.test", "nomic-embed-text", dimensions=768)
    )
    assert second.ingest(note) == (0, 1)
    after = second.store.connection.execute(
        "SELECT id, vector, embedding_space, embedding_dimensions FROM chunks WHERE valid_to IS NULL"
    ).fetchone()
    after_vector = json.loads(after["vector"])
    assert after["id"] == before["id"]
    assert after_vector != before_vector
    assert len(after_vector) == 768
    assert after["embedding_space"] == "ollama:nomic-embed-text"
    assert after["embedding_dimensions"] == 768
    all_stored = [json.loads(row["vector"]) for row in second.store.connection.execute("SELECT vector FROM chunks")]
    assert before_vector not in all_stored
    assert {len(vector) for vector in all_stored} == {768}
    assert second.store.embedding_spaces() == {("ollama:nomic-embed-text", 768)}
    query_vector = second.embedder.embed(["new model query"])[0]
    assert len(query_vector) == 768
    assert [len(vector) for vector in all_stored] == [len(query_vector)] * len(all_stored)
    assert second.search("new model query")


def test_chroma_model_swap_rebuilds_ids_and_removes_old_embeddings(tmp_path: Path):
    note = tmp_path / "chroma-swap.md"
    note.write_text("Chroma must replace this unchanged chunk rather than append beside it.")
    settings = Settings(
        tmp_path / "data", tmp_path / "data" / "knowledge.db",
        embed_provider="test", embed_model="model-a", vector_provider="chroma",
    )
    first = KnowledgeBase(settings, embedder=ModelSpaceEmbedder("test:model-a", 384, 1.0))
    first.ingest(note)
    before = first.vector_store.collection.get(include=["embeddings", "metadatas"])
    before_ids = list(before["ids"])
    before_vector = list(before["embeddings"][0])
    assert len(before_ids) == 1
    assert len(before_vector) == 384

    second = KnowledgeBase(
        Settings(settings.home, settings.db_path, embed_provider="test", embed_model="model-b", vector_provider="chroma"),
        embedder=ModelSpaceEmbedder("test:model-b", 768, 2.0),
    )
    second.ingest(note)
    after = second.vector_store.collection.get(include=["embeddings", "metadatas"])
    after_vector = list(after["embeddings"][0])
    assert list(after["ids"]) == before_ids
    assert second.vector_store.collection.count() == 1
    assert after_vector != before_vector
    assert len(after_vector) == 768
    assert all(metadata["embedding_space"] == "test:model-b" for metadata in after["metadatas"])
    assert all(metadata["embedding_dimensions"] == 768 for metadata in after["metadatas"])
    assert before_vector not in [list(vector) for vector in after["embeddings"]]


def test_search_refuses_mixed_model_or_dimension_before_cosine(tmp_path: Path, monkeypatch):
    note = tmp_path / "mixed.md"
    note.write_text("A plausible score must never be computed across incompatible spaces.")
    settings = Settings(
        tmp_path / "data", tmp_path / "data" / "knowledge.db",
        embed_provider="test", embed_model="model-b", vector_provider="sqlite",
    )
    kb = KnowledgeBase(settings, embedder=ModelSpaceEmbedder("test:model-b", 768, 2.0))
    kb.ingest(note)
    with kb.store.connection:
        kb.store.connection.execute(
            "UPDATE chunks SET vector = ?, embedding_space = ?, embedding_dimensions = ?",
            (json.dumps([1.0] * 384), "test:model-a", 384),
        )

    def scoring_must_not_run(_left, _right):
        raise AssertionError("cosine scoring ran on a mixed embedding space")

    monkeypatch.setattr("mnemosyne.store._cosine", scoring_must_not_run)
    with pytest.raises(RuntimeError, match="mixed embedding spaces") as raised:
        kb.search("plausible score")
    assert "query='test:model-b'" in str(raised.value)
    stored = kb.store.connection.execute(
        "SELECT embedding_space, embedding_dimensions, vector FROM chunks WHERE valid_to IS NULL"
    ).fetchone()
    assert stored["embedding_space"] == "test:model-a"
    assert stored["embedding_dimensions"] == 384
    assert len(json.loads(stored["vector"])) == 384


def test_embedding_model_environment_variable_overrides_saved_preference(tmp_path: Path, monkeypatch):
    home = tmp_path / "data"
    first = KnowledgeBase(
        Settings(home, home / "knowledge.db", embed_provider="hash", vector_provider="sqlite"),
        embedder=HashingEmbedder(),
    )
    first.store.save_setting("embed_model", "older-saved-model")
    monkeypatch.setenv("MNEMO_HOME", str(home))
    monkeypatch.setenv("MNEMO_EMBED_PROVIDER", "ollama")
    monkeypatch.setenv("MNEMO_VECTOR_PROVIDER", "sqlite")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-new")
    restarted = KnowledgeBase(
        Settings.load(), embedder=ModelSpaceEmbedder("ollama:nomic-embed-text-new", 768, 2.0)
    )
    assert restarted.settings.embed_model == "nomic-embed-text-new"
    assert restarted.settings.embed_model != "older-saved-model"


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


def test_reranking_prefers_term_overlap(tmp_path: Path):
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db")
    kb = KnowledgeBase(settings)
    hits = [
        SearchHit(1, "generic context", "misc", "a#L1-L2", 0.9, ("other",)),
        SearchHit(2, "hybrid retrieval improves grounded answers", "retrieval", "b#L1-L2", 0.7, ("rag",)),
    ]
    reranked = kb._rerank("hybrid retrieval", hits)
    assert reranked[0].chunk_id == 2


def test_query_expansion_finds_synonym_matches(tmp_path: Path):
    class ExpansionGenerator:
        def generate(self, prompt: str) -> str:
            if "Rewrite this search query" in prompt:
                return '{"variations":["vehicle maintenance"],"synonyms":["vehicle","automobile"],"hyde":"Vehicle maintenance notes explain repair work."}'
            return '{"ranked_ids":[]}'

    note = tmp_path / "garage.md"
    note.write_text("Vehicle maintenance checklist includes tire pressure and brake inspection.")
    kb = KnowledgeBase(Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", embed_provider="hash"))
    kb.ingest(note)

    hits = kb.search("car repair", generator=ExpansionGenerator())
    assert hits
    assert hits[0].title == "garage"


def test_hyde_embedding_finds_meaning_when_terms_do_not_overlap(tmp_path: Path):
    class HydeGenerator:
        def generate(self, prompt: str) -> str:
            if "Rewrite this search query" in prompt:
                return '{"variations":[],"synonyms":[],"hyde":"Batteries store energy for later use in portable devices."}'
            return '{"ranked_ids":[]}'

    note = tmp_path / "energy.md"
    note.write_text("Batteries store energy for later use in portable devices.")
    kb = KnowledgeBase(Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", embed_provider="hash"))
    kb.ingest(note)

    hits = kb.search("where does saved power live?", generator=HydeGenerator())
    assert hits
    assert hits[0].title == "energy"


def test_qwen_style_reranker_can_put_best_candidate_first(tmp_path: Path):
    class RerankGenerator:
        def __init__(self) -> None:
            self.best_id: int | None = None

        def generate(self, prompt: str) -> str:
            if "Rewrite this search query" in prompt:
                return '{"variations":[],"synonyms":[],"hyde":"Citation validation checks whether answers are supported by sources."}'
            if "Rerank these retrieved note chunks" in prompt:
                match = re.search(r"(\d+): exact", prompt)
                self.best_id = int(match.group(1)) if match else None
                return json.dumps({"ranked_ids": [self.best_id] if self.best_id else []})
            return "{}"

    generic = tmp_path / "generic.md"
    generic.write_text("Citation systems list source numbers and document references.")
    exact = tmp_path / "exact.md"
    exact.write_text("Exact citation validation checks whether generated answers are supported by retrieved sources.")
    kb = KnowledgeBase(Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", embed_provider="hash"))
    kb.ingest(tmp_path)

    hits = kb.search("citation validation support", generator=RerankGenerator())
    assert hits
    assert hits[0].title == "exact"


def test_mmr_diversifies_near_duplicate_chunks_from_same_note(tmp_path: Path):
    repeated = tmp_path / "long.md"
    repeated.write_text(
        "Hybrid retrieval evaluation should compare keyword and vector search.\n\n"
        "Hybrid retrieval evaluation should compare keyword and vector search.\n\n"
        "Hybrid retrieval evaluation should compare keyword and vector search."
    )
    related = tmp_path / "related.md"
    related.write_text("Search evaluation also checks reranking quality and result diversity.")
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", embed_provider="hash", chunk_size=80, chunk_overlap=0)
    kb = KnowledgeBase(settings)
    kb.ingest(tmp_path)

    hits = kb.search("hybrid retrieval evaluation search", limit=2)
    assert len(hits) == 2
    assert len({hit.citation.split("#", 1)[0] for hit in hits}) == 2


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


def test_real_xlsx_upload_indexes_searchable_cell_values(tmp_path: Path, monkeypatch):
    from openpyxl import Workbook

    workbook_path = tmp_path / "budget.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Planning"
    sheet.append(["Project", "Owner", "Signal"])
    sheet.append(["Mnemosyne", "Swayam", "spreadsheet retrieval sentinel"])
    workbook.save(workbook_path)

    monkeypatch.setenv("MNEMO_HOME", str(tmp_path / "web-data"))
    monkeypatch.setenv("MNEMO_EMBED_PROVIDER", "hash")
    monkeypatch.setenv("MNEMO_VECTOR_PROVIDER", "sqlite")
    import mnemosyne.web as web

    web = importlib.reload(web)
    async def run():
        transport = httpx.ASGITransport(app=web.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            uploaded = await client.post(
                "/api/documents",
                files={
                    "files": (
                        "budget.xlsx",
                        workbook_path.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )
            assert uploaded.status_code == 200
            assert uploaded.json()["indexed"][0]["indexed"] is True
            library = await client.get("/api/library")
            library_data = library.json()
            assert library_data["stats"]["documents"] == 1
            assert library_data["documents"][0]["type"] == "xlsx"
            result = await client.post("/api/search", json={"query": "spreadsheet retrieval sentinel"})
            assert result.json()["results"]
            assert "spreadsheet retrieval sentinel" in result.json()["results"][0]["text"]

    asyncio.run(run())


def test_upload_reports_scanned_pdf_ocr_dependency_problem(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
    )
    monkeypatch.setenv("MNEMO_HOME", str(tmp_path / "web-data"))
    monkeypatch.setenv("MNEMO_EMBED_PROVIDER", "hash")
    monkeypatch.setenv("MNEMO_VECTOR_PROVIDER", "sqlite")
    monkeypatch.setattr("mnemosyne.ingest.shutil.which", lambda _name: None)
    import mnemosyne.web as web

    web = importlib.reload(web)
    async def run():
        transport = httpx.ASGITransport(app=web.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            uploaded = await client.post("/api/documents", files={"files": ("scanned.pdf", pdf.read_bytes(), "application/pdf")})
            payload = uploaded.json()
            assert payload["indexed"][0]["indexed"] is False
            assert "missing: pdftoppm, tesseract" in payload["indexed"][0]["diagnostics"][0]["message"]
            library = await client.get("/api/library")
            assert library.json()["stats"]["documents"] == 0

    asyncio.run(run())


def test_watch_scan_indexes_changes_and_removes_deleted_files(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "live.md"
    note.write_text("first version of watch folder knowledge")
    kb = KnowledgeBase(Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db"))
    kb.register_watch_folder(vault)
    note.write_text("second version includes semantic retrieval")
    update = kb.scan_watch_folders()["scanned"][0]
    assert update["indexed"] == 1
    assert kb.search("semantic retrieval")
    note.unlink()
    removed = kb.scan_watch_folders()["scanned"][0]
    assert removed["removed"] == 1
    assert kb.store.stats()["documents"] == 0


def test_notion_zip_import_and_zip_slip_protection(tmp_path: Path):
    archive = tmp_path / "notion.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("Workspace/Research.md", "#rag Notion export retrieval notes")
    kb = KnowledgeBase(Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db"))
    assert kb.import_archive(archive) == (1, 0)
    assert kb.search("Notion export")

    unsafe = tmp_path / "unsafe.zip"
    with ZipFile(unsafe, "w") as bundle:
        bundle.writestr("../escape.md", "unsafe")
    with pytest.raises(ValueError):
        kb.import_archive(unsafe)


def test_backup_restore_round_trip_preserves_searchable_chunks(tmp_path: Path):
    source = tmp_path / "source.md"
    source.write_text("Restorable hybrid retrieval knowledge")
    first = KnowledgeBase(Settings(tmp_path / "one", tmp_path / "one" / "knowledge.db"))
    first.ingest(source)
    payload = first.backup()
    second = KnowledgeBase(Settings(tmp_path / "two", tmp_path / "two" / "knowledge.db"))
    restored = second.restore(payload)
    assert restored["documents"] == 1
    assert restored["chunks"] >= 1
    assert second.search("Restorable retrieval")


def test_pptx_keeps_slide_structure_and_page_citations(tmp_path: Path):
    slides = tmp_path / "deck.pptx"
    with ZipFile(slides, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>First slide</a:t></p:sld>')
        archive.writestr("ppt/slides/slide2.xml", '<p:sld xmlns:p="urn:p" xmlns:a="urn:a"><a:t>Second slide</a:t></p:sld>')
    docs = parse(slides)
    assert [doc.page for doc in docs] == [1, 2]
    chunks = [chunk for doc in docs for chunk in chunk_document(doc, 900, 0)]
    assert chunks[1].citation.endswith("#page=2")


def test_real_ollama_protocol_embedding_search_generation_and_validation(tmp_path: Path, monkeypatch):
    class FakeResponse:
        def __init__(self, body: dict) -> None:
            self.body = json.dumps(body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(request: Request, timeout: int = 120):
        payload = json.loads(request.data or b"{}")
        url = request.full_url
        if url.endswith("/api/embed"):
            vectors = [
                [1.0, 0.0, 0.0, 0.0] if "retrieval" in text.lower() else [0.0, 1.0, 0.0, 0.0]
                for text in payload["input"]
            ]
            return FakeResponse({"embeddings": vectors})
        if url.endswith("/api/generate"):
            assert "NOTES:" in payload["prompt"]
            return FakeResponse({"response": "Hybrid retrieval combines semantic and keyword search [1]."})
        raise AssertionError(f"unexpected Ollama endpoint: {url}")

    monkeypatch.setattr("mnemosyne.providers.urllib.request.urlopen", fake_urlopen)
    base_url = "http://ollama.test"
    settings = Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", ollama_url=base_url)
    embedder = OllamaEmbedder(base_url, "nomic-embed-text", dimensions=4)
    kb = KnowledgeBase(settings, embedder=embedder)
    note = tmp_path / "retrieval.md"
    note.write_text("Hybrid retrieval combines semantic and keyword search.")
    assert kb.ingest(note) == (1, 0)
    answer, hits, validation = kb.ask("How does retrieval work?", OllamaGenerator(base_url, "test-model"))
    assert hits and "retrieval.md" in hits[0].citation
    assert "[1]" in answer
    assert validation.verdict == "grounded"
    assert embedder.last_backend == "ollama"


def test_chroma_adapter_is_persistent_and_searchable(tmp_path: Path):
    from mnemosyne.providers import HashingEmbedder

    settings = Settings(
        tmp_path / "data",
        tmp_path / "data" / "knowledge.db",
        embed_provider="hash",
        vector_provider="chroma",
    )
    kb = KnowledgeBase(settings, embedder=HashingEmbedder())
    note = tmp_path / "chroma.md"
    note.write_text("Persistent Chroma semantic vector retrieval")
    assert kb.ingest(note) == (1, 0)
    assert kb.vector_store is not None
    assert kb.search("Chroma vector retrieval")[0].title == "chroma"


def test_chroma_search_uses_index_instead_of_sqlite_vector_scan(tmp_path: Path, monkeypatch):
    from mnemosyne.providers import HashingEmbedder

    settings = Settings(
        tmp_path / "data",
        tmp_path / "data" / "knowledge.db",
        embed_provider="hash",
        vector_provider="chroma",
    )
    kb = KnowledgeBase(settings, embedder=HashingEmbedder())
    note = tmp_path / "index.md"
    note.write_text("Indexed Chroma retrieval should avoid per-query SQLite vector scans.")
    kb.ingest(note)
    monkeypatch.setattr(kb.store, "vector_search", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sqlite vector scan used")))

    hits = kb.search("Indexed Chroma retrieval")
    assert hits[0].title == "index"


def test_uncited_model_output_is_repaired_or_safely_grounded(tmp_path: Path):
    class UncitedGenerator:
        def generate(self, prompt: str) -> str:
            return "Hybrid retrieval combines vectors and keyword search."

    note = tmp_path / "grounding.md"
    note.write_text("Hybrid retrieval combines vectors and keyword search for trustworthy answers.")
    kb = KnowledgeBase(Settings(tmp_path / "data", tmp_path / "data" / "knowledge.db", embed_provider="hash"))
    kb.ingest(note)
    answer, _, validation = kb.ask("How does hybrid retrieval work?", UncitedGenerator())
    assert "[1]" in answer
    assert validation.verdict == "grounded"


def test_web_upload_library_search_and_citation_preview(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MNEMO_HOME", str(tmp_path / "web-data"))
    monkeypatch.setenv("MNEMO_EMBED_PROVIDER", "hash")
    monkeypatch.setenv("MNEMO_VECTOR_PROVIDER", "sqlite")
    import mnemosyne.web as web

    web = importlib.reload(web)
    transport = httpx.ASGITransport(app=web.app)
    async def run():
        transport = httpx.ASGITransport(app=web.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            home = await client.get("/")
            assert home.status_code == 200
            assert "Upload" in home.text
            uploaded = await client.post(
                "/api/documents",
                files={"files": ("notes.md", b"#rag End to end hybrid knowledge search", "text/markdown")},
            )
            assert uploaded.status_code == 200
            assert uploaded.json()["indexed"][0]["indexed"] is True
            library = await client.get("/api/library")
            assert library.json()["stats"]["documents"] == 1
            result = await client.post("/api/search", json={"query": "hybrid knowledge"})
            assert result.status_code == 200
            hit = result.json()["results"][0]
            assert "#L1-L1" in hit["citation"]
            preview = await client.get(f"/api/chunks/{hit['id']}")
            assert preview.status_code == 200
            assert "End to end" in preview.json()["text"]

    asyncio.run(run())
