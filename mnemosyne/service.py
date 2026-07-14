from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import Settings
from .ingest import chunk_document, discover, file_digest, parse
from .models import CitationValidation, GraphEdge, SearchHit, TopicCluster
from .providers import Embedder, Generator, HashingEmbedder, OllamaEmbedder
from .store import KnowledgeStore

REFERENCE_PATTERN = re.compile(r"\[(\d+)\]")
DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? 20\d{2})\b", re.I)


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
        dense_hits = self.store.hybrid_search(query, self.embedder.embed([query])[0], max(limit * 3, 12), tag=tag, folder=folder, file_type=file_type)
        reranked = self._rerank(query, dense_hits)
        return reranked[:limit]

    def ask(
        self,
        query: str,
        generator: Generator,
        tag: str | None = None,
        folder: str | None = None,
        file_type: str | None = None,
    ) -> tuple[str, list[SearchHit], CitationValidation]:
        hits = self.search(query, tag=tag, folder=folder, file_type=file_type)
        context = "\n\n".join(f"[{i}] {hit.citation}\n{hit.text}" for i, hit in enumerate(hits, 1))
        prompt = f"""You answer questions using only the supplied personal notes.
If the notes do not support an answer, say so. Cite claims with bracketed source numbers
like [1]. Prefer multiple citations when claims combine evidence. Do not invent sources.

QUESTION:\n{query}\n\nNOTES:\n{context}\n\nANSWER:"""
        answer = generator.generate(prompt)
        validation = self._validate_citations(answer, hits)
        self.store.log_conversation(
            "ask",
            query,
            answer,
            {
                "citations": list(validation.cited_numbers),
                "verdict": validation.verdict,
                "filters": {"tag": tag, "folder": folder, "file_type": file_type},
            },
        )
        self.store.log_evaluation(
            query,
            validation.verdict,
            list(validation.cited_numbers),
            list(validation.missing_numbers),
            list(validation.unsupported_numbers),
        )
        return answer, hits, validation

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

    def save_search(self, name: str, query: str, tag: str | None, folder: str | None, file_type: str | None) -> int:
        return self.store.create_saved_search(name, query, tag, folder, file_type)

    def history(self, limit: int = 50) -> list[dict]:
        return self.store.conversation_history(limit)

    def save_settings(self, payload: dict) -> dict:
        for key, value in payload.items():
            self.store.save_setting(key, value)
        return self.store.load_settings()

    def register_watch_folder(self, path: Path, profile: str = "local") -> tuple[int, int]:
        absolute = path.expanduser().resolve()
        self.store.upsert_watch_folder(str(absolute), profile)
        return self.ingest(absolute)

    def scan_watch_folders(self) -> dict:
        results = []
        for watch in self.store.list_watch_folders():
            if not watch.enabled:
                continue
            path = Path(watch.path)
            if path.exists():
                indexed, skipped = self.ingest(path)
                results.append({"path": watch.path, "indexed": indexed, "skipped": skipped, "profile": watch.profile})
        return {"scanned": results}

    def reader(self, path: str) -> dict:
        document = next((doc for doc in self.store.list_documents() if doc["path"] == path), None)
        return {
            "document": document,
            "chunks": [preview.__dict__ for preview in self.store.document_chunks(path)],
            "related": self.related_notes(path),
            "entities": self.entities(path),
            "timeline": self.timeline(path),
            "contradictions": self.contradictions(path),
        }

    def related_notes(self, path: str, limit: int = 6) -> list[dict]:
        edges = self.graph(64)
        related = []
        for edge in edges:
            if edge.source == path:
                related.append({"path": edge.target, "weight": edge.weight, "reason": edge.reason})
            elif edge.target == path:
                related.append({"path": edge.source, "weight": edge.weight, "reason": edge.reason})
        return sorted(related, key=lambda item: item["weight"], reverse=True)[:limit]

    def entities(self, path: str) -> list[str]:
        chunks = self.store.document_chunks(path)
        text = "\n".join(chunk.text for chunk in chunks)
        entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
        return sorted(entity for entity in entities if len(entity) > 2)[:30]

    def timeline(self, path: str) -> list[dict]:
        chunks = self.store.document_chunks(path)
        events = []
        for chunk in chunks:
            for match in DATE_PATTERN.findall(chunk.text):
                events.append({"date": match, "citation": chunk.citation, "text": chunk.text[:180]})
        return events[:20]

    def contradictions(self, path: str) -> list[dict]:
        chunks = self.store.document_chunks(path)
        contradictions = []
        for chunk in chunks:
            lowered = chunk.text.lower()
            if " not " in f" {lowered} " or "never" in lowered or "cannot" in lowered:
                for other in chunks:
                    if other.chunk_id == chunk.chunk_id:
                        continue
                    overlap = set(re.findall(r"[a-z0-9_]+", lowered)) & set(re.findall(r"[a-z0-9_]+", other.text.lower()))
                    if len(overlap) >= 4 and (" not " not in f" {other.text.lower()} "):
                        contradictions.append(
                            {
                                "left": chunk.citation,
                                "right": other.citation,
                                "shared_terms": sorted(list(overlap))[:8],
                            }
                        )
                        break
        return contradictions[:10]

    def backup(self) -> dict:
        return self.store.backup_payload()

    def _default_embedder(self) -> Embedder:
        if self.settings.embed_provider == "ollama":
            return OllamaEmbedder(self.settings.ollama_url, self.settings.embed_model)
        return HashingEmbedder()

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        reranked: list[SearchHit] = []
        for hit in hits:
            text_terms = set(re.findall(r"[a-z0-9_]+", hit.text.lower()))
            title_terms = set(re.findall(r"[a-z0-9_]+", hit.title.lower()))
            overlap = len(terms & text_terms) / max(1, len(terms))
            title_overlap = len(terms & title_terms) / max(1, len(terms))
            tag_overlap = len(terms & set(hit.tags)) / max(1, len(terms))
            citation_bonus = 0.05 if ("#L" in hit.citation or "#page=" in hit.citation) else 0.0
            reranked.append(
                SearchHit(
                    hit.chunk_id,
                    hit.text,
                    hit.title,
                    hit.citation,
                    hit.score + 0.22 * overlap + 0.12 * title_overlap + 0.08 * tag_overlap + citation_bonus,
                    hit.tags,
                )
            )
        return sorted(reranked, key=lambda hit: hit.score, reverse=True)

    def _validate_citations(self, answer: str, hits: list[SearchHit]) -> CitationValidation:
        cited_numbers = tuple(sorted({int(match) for match in REFERENCE_PATTERN.findall(answer)}))
        available = {index for index, _ in enumerate(hits, 1)}
        missing = tuple(number for number in cited_numbers if number not in available)
        unsupported: list[int] = []
        sentences = [sentence.strip().lower() for sentence in re.split(r"(?<=[.!?])\s+", answer) if sentence.strip()]
        for number in cited_numbers:
            if number not in available:
                continue
            source = hits[number - 1].text.lower()
            claiming_sentences = [sentence for sentence in sentences if f"[{number}]" in sentence]
            if claiming_sentences and not any(self._sentence_supported(sentence, source) for sentence in claiming_sentences):
                unsupported.append(number)
        verdict = "grounded"
        if not cited_numbers:
            verdict = "missing-citations"
        elif missing:
            verdict = "invalid-citations"
        elif unsupported:
            verdict = "weak-support"
        return CitationValidation(
            cited_numbers=cited_numbers,
            missing_numbers=missing,
            unsupported_numbers=tuple(unsupported),
            answer_has_citations=bool(cited_numbers),
            verdict=verdict,
        )

    def _sentence_supported(self, sentence: str, source: str) -> bool:
        tokens = [token for token in re.findall(r"[a-z0-9_]+", sentence) if token not in {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "have", "has", "had", "notes", "note"}]
        if not tokens:
            return True
        overlap = sum(1 for token in tokens if token in source)
        return overlap / max(1, len(tokens)) >= 0.35
