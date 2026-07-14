from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
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


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "privacy": "local-first", "model": settings.ollama_model}


@app.get("/api/library")
def library() -> dict:
    documents = knowledge.store.list_documents()
    for document in documents:
        path = Path(document["path"])
        document["name"] = _display_name(path)
        document["type"] = path.suffix.lower().lstrip(".") or "text"
        document.pop("digest", None)
    return {"documents": documents, "stats": knowledge.store.stats()}


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


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    hits = knowledge.search(request.query, request.limit)
    return {
        "query": request.query,
        "results": [
            {
                "id": hit.chunk_id,
                "title": hit.title,
                "text": hit.text,
                "citation": hit.citation,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ],
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    generator = OllamaGenerator(settings.ollama_url, settings.ollama_model)
    try:
        answer, hits = knowledge.ask(request.query, generator)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "answer": answer,
        "sources": [
            {"number": index, "title": hit.title, "citation": hit.citation, "text": hit.text}
            for index, hit in enumerate(hits, 1)
        ],
    }


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
