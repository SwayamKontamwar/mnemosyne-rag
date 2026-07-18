from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Protocol, Sequence


class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class VectorStore(Protocol):
    def add(self, ids: Sequence[str], vectors: Sequence[list[float]], metadata: Sequence[dict]) -> None: ...

    def query(self, vector: list[float], limit: int = 20, where: dict | None = None) -> list[tuple[int, float]]: ...


class HashingEmbedder:
    """Dependency-free feature hashing baseline; replaceable with neural embeddings."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_identity(self) -> str:
        return "hash:blake2b-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        features = tokens + [f"{a}:{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += -1.0 if value & 1 else 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, dimensions: int = 768) -> None:
        self.base_url = base_url
        self.model = model
        self._dimensions = dimensions
        self.last_backend = "unavailable"
        self.last_error: str | None = None
        self._fallback = HashingEmbedder(dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_identity(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            self.last_backend = "hash-fallback"
            self.last_error = str(exc)
            return self._fallback.embed(texts)
        embeddings = payload.get("embeddings") or []
        if not embeddings:
            self.last_backend = "hash-fallback"
            self.last_error = "Ollama returned no embeddings"
            return self._fallback.embed(texts)
        self._dimensions = len(embeddings[0])
        self.last_backend = "ollama"
        self.last_error = None
        return embeddings


class OllamaGenerator:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())["response"].strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Run `ollama serve` first."
            ) from exc


class ChromaVectorAdapter:
    """Persistent Chroma adapter used when ``MNEMO_VECTOR_PROVIDER=chroma``."""

    def __init__(self, path: Path, collection: str = "mnemosyne") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Chroma mode requires `pip install -e '.[full]'`.") from exc
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids: Sequence[str], vectors: Sequence[list[float]], metadata: Sequence[dict]) -> None:
        if ids:
            self.collection.upsert(ids=list(ids), embeddings=list(vectors), metadatas=list(metadata))

    def rebuild(
        self, ids: Sequence[str], vectors: Sequence[list[float]], metadata: Sequence[dict],
        embedding_space: str, dimensions: int, checkpoint: Callable[[str], None] | None = None,
        batch_size: int = 100,
    ) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            self.collection.name,
            metadata={"hnsw:space": "cosine", "embedding_space": embedding_space, "embedding_dimensions": dimensions},
        )
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            self.add(ids[start:stop], vectors[start:stop], metadata[start:stop])
            if checkpoint:
                checkpoint("during_embedding_chroma_rebuild")

    def manifest(self) -> dict[str, str]:
        result = self.collection.get(include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = []
        return {
            str(item_id): hashlib.sha256(
                b"".join(struct.pack("<f", float(value)) for value in vector)
            ).hexdigest()
            for item_id, vector in zip(result.get("ids") or [], embeddings)
        }

    def verify_vectors(self, ids: Sequence[str], vectors: Sequence[list[float]]) -> None:
        result = self.collection.get(ids=list(ids), include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = []
        actual = {str(item_id): list(vector) for item_id, vector in zip(result.get("ids") or [], embeddings)}
        expected_ids = {str(item_id) for item_id in ids}
        if set(actual) != expected_ids:
            raise RuntimeError("Chroma rebuild IDs do not match SQLite")
        for item_id, expected in zip(ids, vectors):
            stored = actual[str(item_id)]
            if len(stored) != len(expected):
                raise RuntimeError(f"Chroma dimension mismatch for chunk {item_id}")
            if max((abs(left - right) for left, right in zip(stored, expected)), default=0.0) > 1e-6:
                raise RuntimeError(f"Chroma vector differs from SQLite vector for chunk {item_id}")

    def ensure_space(self, embedding_space: str, dimensions: int) -> None:
        metadata = self.collection.metadata or {}
        count = self.collection.count()
        if count == 0:
            self.collection.modify(metadata={
                "embedding_space": embedding_space, "embedding_dimensions": dimensions,
            })
            return
        if metadata.get("embedding_space") != embedding_space or metadata.get("embedding_dimensions") != dimensions:
            raise RuntimeError("Chroma embedding space does not match SQLite; re-run ingest to rebuild the index")

    def delete_document(self, path: str) -> None:
        self.collection.delete(where={"document_path": path})

    def query(
        self, vector: list[float], limit: int = 20, where: dict | None = None,
        embedding_space: str | None = None,
    ) -> list[tuple[int, float]]:
        if embedding_space is not None:
            self.ensure_space(embedding_space, len(vector))
        kwargs = {"query_embeddings": [vector], "n_results": limit}
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [(int(item_id), 1.0 - float(distance)) for item_id, distance in zip(ids, distances)]
