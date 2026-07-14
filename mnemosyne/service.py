from __future__ import annotations

from pathlib import Path

from .config import Settings
from .ingest import chunk_document, discover, file_digest, parse
from .models import GraphEdge, SearchHit, TopicCluster
from .providers import Embedder, Generator, HashingEmbedder, OllamaEmbedder
from .store import KnowledgeStore


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or self._default_embedder()
        self.store = KnowledgeStore(settings.db_path)
        self.store.initialize()

    def ingest(self, source: Path) -> tuple[int, int]:
        indexed = skipped = 0
        for path in discover(source):
            digest = file_digest(path)
            absolute = str(path.resolve())
            if self.store.digest_for(absolute) == digest:
                skipped += 1
                continue
            documents = parse(path)
            chunks = [
                chunk
                for document in documents
                for chunk in chunk_document(document, self.settings.chunk_size, self.settings.chunk_overlap)
            ]
            if not chunks:
                continue
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            primary = documents[0]
            self.store.replace_document(
                absolute,
                digest,
                zip(chunks, vectors),
                {
                    "title": primary.title,
                    "source_type": primary.source_type,
                    "file_type": primary.file_type,
                    "folder": primary.folder,
                    "tags": list(primary.tags),
                    "links": list(primary.links),
                },
            )
            indexed += 1
        return indexed, skipped

    def search(self, query: str, limit: int = 8, tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> list[SearchHit]:
        return self.store.hybrid_search(query, self.embedder.embed([query])[0], limit, tag=tag, folder=folder, file_type=file_type)

    def ask(self, query: str, generator: Generator) -> tuple[str, list[SearchHit]]:
        hits = self.search(query)
        context = "\n\n".join(f"[{i}] {hit.citation}\n{hit.text}" for i, hit in enumerate(hits, 1))
        prompt = f"""You answer questions using only the supplied personal notes.
If the notes do not support an answer, say so. Cite claims with bracketed source numbers
like [1]. Do not invent sources.

QUESTION:\n{query}\n\nNOTES:\n{context}\n\nANSWER:"""
        return generator.generate(prompt), hits

    def backlinks(self, document_name: str, limit: int = 10) -> list[SearchHit]:
        source = self.store.chunks_for_document(document_name)
        if not source:
            return []
        summary = "\n".join(row["text"] for row in source)[:5000]
        return [hit for hit in self.search(summary, limit + len(source)) if document_name not in hit.citation][:limit]

    def graph(self, limit: int = 24) -> list[GraphEdge]:
        return self.store.graph_edges(limit)

    def clusters(self, limit: int = 8) -> list[TopicCluster]:
        return self.store.clusters(limit)

    def _default_embedder(self) -> Embedder:
        if self.settings.embed_provider == "ollama":
            return OllamaEmbedder(self.settings.ollama_url, self.settings.embed_model)
        return HashingEmbedder()
