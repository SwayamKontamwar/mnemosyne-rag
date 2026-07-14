from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager
import json
import threading
import urllib.error
import urllib.request
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .ingest import SUPPORTED
from .providers import OllamaGenerator
from .service import KnowledgeBase

settings = Settings.load()
knowledge = KnowledgeBase(settings)
static_dir = Path(__file__).parent / "static"
upload_dir = settings.home / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
watch_stop = threading.Event()
watch_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global watch_thread
    if watch_thread is None or not watch_thread.is_alive():
        watch_stop.clear()
        watch_thread = threading.Thread(target=knowledge.watch_forever, args=(watch_stop,), daemon=True, name="mnemosyne-watcher")
        watch_thread.start()
    try:
        yield
    finally:
        watch_stop.set()
        if watch_thread and watch_thread.is_alive():
            watch_thread.join(timeout=3)

app = FastAPI(
    title="Mnemosyne",
    description="Local-first personal knowledge search with grounded citations.",
    version="0.2.0",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=8, ge=1, le=30)
    tag: str | None = None
    folder: str | None = None
    file_type: str | None = None


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    tag: str | None = None
    folder: str | None = None
    file_type: str | None = None


class SavedSearchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=2000)
    tag: str | None = None
    folder: str | None = None
    file_type: str | None = None


class CollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=400)
    tags: list[str] = Field(default_factory=list)
    query: str = Field(default="", max_length=2000)


class SettingsRequest(BaseModel):
    embed_provider: str | None = None
    embed_model: str | None = None
    ollama_model: str | None = None
    privacy_mode: str | None = None
    source_reader_mode: str | None = None
    vector_provider: str | None = None


class WatchFolderRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2000)
    profile: str = Field(default="local", max_length=40)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    ollama = _ollama_health()
    return {
        "status": "ok" if ollama["available"] else "degraded",
        "privacy": "local-first",
        **knowledge.provider_status(),
        "ollama": ollama,
    }


@app.get("/api/settings")
def app_settings() -> dict:
    return {
        "runtime": {
            "ollama_url": knowledge.settings.ollama_url,
            "ollama_model": knowledge.settings.ollama_model,
            "embed_provider": knowledge.settings.embed_provider,
            "embed_model": knowledge.settings.embed_model,
            "vector_provider": knowledge.settings.vector_provider,
        },
        "preferences": knowledge.store.load_settings(),
    }


@app.post("/api/settings")
def save_settings(request: SettingsRequest) -> dict:
    return {"preferences": knowledge.save_settings(_dump_model(request, exclude_none=True))}


@app.get("/api/library")
def library(tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> dict:
    documents = knowledge.store.list_documents(tag=tag, folder=folder, file_type=file_type)
    for document in documents:
        path = Path(document["path"])
        document["name"] = _display_name(path)
        document["type"] = document["file_type"] or path.suffix.lower().lstrip(".") or "text"
        document.pop("digest", None)
    return {
        "documents": documents,
        "stats": knowledge.store.stats(),
        "filters": {
            "tags": knowledge.store.list_tags(),
            "folders": knowledge.store.list_folders(),
            "types": sorted({document["file_type"] for document in documents}),
        },
    }


@app.post("/api/documents")
def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    if len(files) > 50:
        raise HTTPException(400, "Upload at most 50 files at once.")
    indexed: list[dict] = []
    rejected: list[dict] = []
    for upload in files:
        original = Path(upload.filename or "untitled").name
        suffix = Path(original).suffix.lower()
        if suffix not in SUPPORTED and suffix != ".zip":
            rejected.append({"name": original, "reason": f"Unsupported file type: {suffix or 'none'}"})
            continue
        destination = upload_dir / f"{uuid4().hex}-{original}"
        try:
            _save_upload(upload, destination)
        except ValueError as exc:
            rejected.append({"name": original, "reason": str(exc)})
            continue
        try:
            count, _ = knowledge.import_archive(destination) if suffix == ".zip" else knowledge.ingest(destination)
        except (ValueError, OSError) as exc:
            rejected.append({"name": original, "reason": str(exc)})
            destination.unlink(missing_ok=True)
            continue
        indexed.append({"name": original, "indexed": bool(count)})
    return {"indexed": indexed, "rejected": rejected}


@app.post("/api/watch-folders")
def add_watch_folder(request: WatchFolderRequest) -> dict:
    try:
        indexed, skipped = knowledge.register_watch_folder(Path(request.path), request.profile)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"indexed": indexed, "skipped": skipped, "watch_folders": [watch.__dict__ for watch in knowledge.store.list_watch_folders()]}


@app.get("/api/watch-folders")
def watch_folders() -> dict:
    return {"watch_folders": [watch.__dict__ for watch in knowledge.store.list_watch_folders()]}


@app.post("/api/watch-folders/scan")
def scan_watch_folders() -> dict:
    return knowledge.scan_watch_folders()


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    knowledge.store.log_conversation("search", request.query, payload=_dump_model(request))
    hits = knowledge.search(request.query, request.limit, tag=request.tag, folder=request.folder, file_type=request.file_type)
    return {
        "query": request.query,
        "results": [
            {
                "id": hit.chunk_id,
                "title": hit.title,
                "text": hit.text,
                "citation": hit.citation,
                "score": round(hit.score, 4),
                "tags": list(hit.tags),
            }
            for hit in hits
        ],
    }


@app.post("/api/saved-searches")
def create_saved_search(request: SavedSearchRequest) -> dict:
    search_id = knowledge.save_search(request.name, request.query, request.tag, request.folder, request.file_type)
    return {"id": search_id, "saved_searches": [search.__dict__ for search in knowledge.store.list_saved_searches()]}


@app.get("/api/saved-searches")
def saved_searches() -> dict:
    return {"saved_searches": [search.__dict__ for search in knowledge.store.list_saved_searches()]}


@app.get("/api/history")
def history(limit: int = 50) -> dict:
    return {"history": knowledge.history(limit)}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    generator = OllamaGenerator(knowledge.settings.ollama_url, knowledge.settings.ollama_model)
    try:
        answer, hits, validation = knowledge.ask(
            request.query,
            generator,
            tag=request.tag,
            folder=request.folder,
            file_type=request.file_type,
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "answer": answer,
        "sources": [
            {"number": index, "title": hit.title, "citation": hit.citation, "text": hit.text, "chunk_id": hit.chunk_id}
            for index, hit in enumerate(hits, 1)
        ],
        "validation": {
            "cited_numbers": list(validation.cited_numbers),
            "missing_numbers": list(validation.missing_numbers),
            "unsupported_numbers": list(validation.unsupported_numbers),
            "answer_has_citations": validation.answer_has_citations,
            "verdict": validation.verdict,
        },
    }


@app.get("/api/chunks/{chunk_id}")
def chunk_preview(chunk_id: int) -> dict:
    preview = knowledge.store.chunk_preview(chunk_id)
    if not preview:
        raise HTTPException(404, "Chunk not found.")
    return {
        "id": preview.chunk_id,
        "title": preview.title,
        "text": preview.text,
        "citation": preview.citation,
        "page": preview.page,
        "start_line": preview.start_line,
        "end_line": preview.end_line,
        "tags": list(preview.tags),
    }


@app.get("/api/reader")
def reader(path: str) -> dict:
    result = knowledge.reader(path)
    if result["document"] is None:
        raise HTTPException(404, "Document not found.")
    return result


@app.get("/api/graph")
def graph(limit: int = 24) -> dict:
    return {
        "edges": [
            {"source": edge.source, "target": edge.target, "weight": round(edge.weight, 4), "reason": edge.reason}
            for edge in knowledge.graph(limit)
        ]
    }


@app.get("/api/clusters")
def clusters(limit: int = 8) -> dict:
    return {
        "clusters": [
            {"name": cluster.name, "documents": list(cluster.document_paths), "keywords": list(cluster.keywords)}
            for cluster in knowledge.clusters(limit)
        ]
    }


@app.post("/api/collections")
def create_collection(request: CollectionRequest) -> dict:
    collection_id = knowledge.store.save_collection(request.name, request.description, request.tags, request.query)
    return {"id": collection_id, "collections": knowledge.store.list_collections()}


@app.get("/api/collections")
def collections() -> dict:
    return {"collections": knowledge.store.list_collections()}


@app.get("/api/evaluations")
def evaluations() -> dict:
    return knowledge.store.evaluation_summary()


@app.get("/api/diagnostics")
def diagnostics(limit: int = 100) -> dict:
    return {"diagnostics": knowledge.diagnostics(limit)}


@app.get("/api/compare")
def compare(left: str, right: str) -> dict:
    try:
        return knowledge.compare(left, right)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/backup")
def backup() -> JSONResponse:
    return JSONResponse(knowledge.backup())


@app.post("/api/restore")
def restore(payload: dict) -> dict:
    try:
        return knowledge.store.restore_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid backup: {exc}") from exc


def main() -> None:
    import uvicorn

    uvicorn.run("mnemosyne.web:app", host="127.0.0.1", port=8765)


def _display_name(path: Path) -> str:
    name = path.name
    parts = name.split("-", 1)
    if path.parent == upload_dir and len(parts) == 2 and len(parts[0]) == 32:
        return parts[1]
    return name


def _save_upload(upload: UploadFile, destination: Path) -> None:
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("File exceeds the 100 MB upload limit.")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _dump_model(model: BaseModel, **kwargs) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)


def _ollama_health() -> dict:
    try:
        with urllib.request.urlopen(f"{knowledge.settings.ollama_url}/api/tags", timeout=2) as response:
            payload = json.loads(response.read())
        models = [item.get("name", "") for item in payload.get("models", [])]
        return {
            "available": True,
            "models": models,
            "embed_model_ready": any(name.split(":")[0] == knowledge.settings.embed_model.split(":")[0] for name in models),
            "answer_model_ready": any(name.split(":")[0] == knowledge.settings.ollama_model.split(":")[0] for name in models),
        }
    except (urllib.error.URLError, TimeoutError, ValueError):
        return {"available": False, "models": [], "embed_model_ready": False, "answer_model_ready": False}
