from __future__ import annotations

from pathlib import Path
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

app = FastAPI(
    title="Mnemosyne",
    description="Local-first personal knowledge search with grounded citations.",
    version="0.2.0",
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


class WatchFolderRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2000)
    profile: str = Field(default="local", max_length=40)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "privacy": "local-first",
        "model": settings.ollama_model,
        "embed_provider": settings.embed_provider,
        "embed_model": settings.embed_model,
    }


@app.get("/api/settings")
def app_settings() -> dict:
    return {
        "runtime": {
            "ollama_url": settings.ollama_url,
            "ollama_model": settings.ollama_model,
            "embed_provider": settings.embed_provider,
            "embed_model": settings.embed_model,
        },
        "preferences": knowledge.store.load_settings(),
    }


@app.post("/api/settings")
def save_settings(request: SettingsRequest) -> dict:
    return {"preferences": knowledge.save_settings(request.model_dump(exclude_none=True))}


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
        if suffix not in SUPPORTED:
            rejected.append({"name": original, "reason": f"Unsupported file type: {suffix or 'none'}"})
            continue
        destination = upload_dir / f"{uuid4().hex}-{original}"
        try:
            _save_upload(upload, destination)
        except ValueError as exc:
            rejected.append({"name": original, "reason": str(exc)})
            continue
        count, _ = knowledge.ingest(destination)
        indexed.append({"name": original, "indexed": bool(count)})
    return {"indexed": indexed, "rejected": rejected}


@app.post("/api/watch-folders")
def add_watch_folder(request: WatchFolderRequest) -> dict:
    indexed, skipped = knowledge.register_watch_folder(Path(request.path), request.profile)
    return {"indexed": indexed, "skipped": skipped, "watch_folders": [watch.__dict__ for watch in knowledge.store.list_watch_folders()]}


@app.get("/api/watch-folders")
def watch_folders() -> dict:
    return {"watch_folders": [watch.__dict__ for watch in knowledge.store.list_watch_folders()]}


@app.post("/api/watch-folders/scan")
def scan_watch_folders() -> dict:
    return knowledge.scan_watch_folders()


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    knowledge.store.log_conversation("search", request.query, payload=request.model_dump())
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
    generator = OllamaGenerator(settings.ollama_url, settings.ollama_model)
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
    return knowledge.reader(path)


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


@app.get("/api/backup")
def backup() -> JSONResponse:
    return JSONResponse(knowledge.backup())


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
