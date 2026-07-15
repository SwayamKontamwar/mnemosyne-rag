from __future__ import annotations

import json
import math
import re
import threading
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

from .config import Settings
from .ingest import chunk_document, discover, file_digest, ocr_status, parse
from .models import CitationValidation, GraphEdge, SearchHit, TopicCluster
from .providers import ChromaVectorAdapter, Embedder, Generator, HashingEmbedder, OllamaEmbedder, OllamaGenerator
from .store import KnowledgeStore

REFERENCE_PATTERN = re.compile(r"\[(\d+)\]")
DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? 20\d{2})\b", re.I)


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.store = KnowledgeStore(settings.db_path)
        self.store.initialize()
        preferences = self.store.load_settings()
        if preferences:
            self.settings = replace(
                settings,
                embed_provider=str(preferences.get("embed_provider", settings.embed_provider)),
                embed_model=str(preferences.get("embed_model", settings.embed_model)),
                ollama_model=str(preferences.get("ollama_model", settings.ollama_model)),
                vector_provider=str(preferences.get("vector_provider", settings.vector_provider)),
            )
        self.embedder = embedder or self._default_embedder()
        self.vector_store = self._default_vector_store()

    def ingest(self, source: Path) -> tuple[int, int]:
        indexed = skipped = 0
        for path in discover(source):
            digest = file_digest(path)
            absolute = str(path.resolve())
            if self.store.digest_for(absolute) == digest:
                skipped += 1
                continue
            try:
                documents = parse(path)
            except Exception as exc:
                self.store.log_diagnostic(absolute, "error", "parse_failed", f"{type(exc).__name__}: {exc}")
                continue
            if not documents:
                message = "No searchable text was extracted. This file may need OCR or a richer parser."
                if path.suffix.lower() == ".pdf":
                    status = ocr_status()
                    if not status["available"]:
                        message = (
                            "No searchable text was extracted. Scanned PDF OCR needs Poppler `pdftoppm` "
                            f"and Tesseract installed; missing: {', '.join(status['missing'])}."
                        )
                self.store.log_diagnostic(
                    absolute,
                    "warning",
                    "no_text_extracted",
                    message,
                )
                continue
            chunks = [
                chunk
                for document in documents
                for chunk in chunk_document(document, self.settings.chunk_size, self.settings.chunk_overlap)
            ]
            if not chunks:
                self.store.log_diagnostic(
                    absolute,
                    "warning",
                    "empty_chunks",
                    "The parser found a document but no searchable chunks were produced.",
                )
                continue
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            primary = documents[0]
            chunk_ids = self.store.replace_document(
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
            if self.vector_store:
                self.vector_store.delete_document(absolute)
                self.vector_store.add(
                    [str(item_id) for item_id in chunk_ids],
                    vectors,
                    [
                        {
                            "document_path": absolute,
                            "title": chunk.title,
                            "citation": chunk.citation,
                            "folder": primary.folder,
                            "file_type": primary.file_type,
                            "tags": ",".join(chunk.tags),
                        }
                        for chunk in chunks
                    ],
                )
            indexed += 1
        return indexed, skipped

    def search(
        self,
        query: str,
        limit: int = 8,
        tag: str | None = None,
        folder: str | None = None,
        file_type: str | None = None,
        generator: Generator | None = None,
    ) -> list[SearchHit]:
        generator = generator or self._retrieval_generator()
        plan = self._query_plan(query, generator)
        candidate_limit = max(20, limit * 6)
        rerank_limit = max(20, limit * 3)
        semantic_texts = [query, *plan["expanded_queries"]]
        if plan["hyde"]:
            semantic_texts.append(plan["hyde"])
        query_vectors = self.embedder.embed(semantic_texts)
        weighted_rankings: list[tuple[list[int], float]] = []

        for expanded_query in [query, *plan["expanded_queries"]]:
            ids = self.store.keyword_search(expanded_query, candidate_limit, tag=tag, folder=folder, file_type=file_type)
            if ids:
                weighted_rankings.append((ids, 1.1 if expanded_query == query else 0.8))

        for index, vector in enumerate(query_vectors):
            vector_hits = self._vector_candidates(vector, candidate_limit, tag=tag, folder=folder, file_type=file_type)
            ids = [chunk_id for chunk_id, _ in vector_hits]
            if ids:
                weighted_rankings.append((ids, 1.0 if index == 0 else 0.85))

        fused_scores = self._reciprocal_rank_fusion(weighted_rankings)
        if not fused_scores:
            return []
        fused_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:rerank_limit]
        candidates = self.store.hits_by_ids(fused_ids, fused_scores)
        reranked = self._rerank(query, candidates, generator=generator)
        diverse = self._mmr(query_vectors[0], reranked, limit)
        return diverse[:limit]

    def ask(
        self,
        query: str,
        generator: Generator,
        tag: str | None = None,
        folder: str | None = None,
        file_type: str | None = None,
    ) -> tuple[str, list[SearchHit], CitationValidation]:
        hits = self.search(query, tag=tag, folder=folder, file_type=file_type, generator=generator)
        context = "\n\n".join(f"[{i}] {hit.citation}\n{hit.text}" for i, hit in enumerate(hits, 1))
        prompt = f"""You answer questions using only the supplied personal notes.
If the notes do not support an answer, say so. Cite claims with bracketed source numbers
like [1]. Prefer multiple citations when claims combine evidence. Do not invent sources.

QUESTION:\n{query}\n\nNOTES:\n{context}\n\nANSWER:"""
        answer = generator.generate(prompt)
        validation = self._validate_citations(answer, hits)
        if hits and validation.verdict != "grounded":
            repair_prompt = f"""Revise the draft so every factual paragraph or bullet ends with one or more valid bracketed citations.
Use only source numbers 1 through {len(hits)}. Remove unsupported claims and never invent a source.

QUESTION:\n{query}\n\nSOURCES:\n{context}\n\nDRAFT:\n{answer}\n\nREVISED ANSWER:"""
            repaired = generator.generate(repair_prompt)
            repaired_validation = self._validate_citations(repaired, hits)
            if repaired_validation.verdict == "grounded":
                answer, validation = repaired, repaired_validation
            else:
                excerpt = " ".join(hits[0].text.split())[:600].rstrip(" .!?")
                answer = f"{excerpt} [1]."
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
        preferences = self.store.load_settings()
        self.configure(preferences)
        return preferences

    def configure(self, preferences: dict | None = None) -> None:
        preferences = preferences or self.store.load_settings()
        self.settings = replace(
            self.settings,
            embed_provider=str(preferences.get("embed_provider", self.settings.embed_provider)),
            embed_model=str(preferences.get("embed_model", self.settings.embed_model)),
            ollama_model=str(preferences.get("ollama_model", self.settings.ollama_model)),
            vector_provider=str(preferences.get("vector_provider", self.settings.vector_provider)),
        )
        self.embedder = self._default_embedder()
        self.vector_store = self._default_vector_store()

    def register_watch_folder(self, path: Path, profile: str = "local") -> tuple[int, int]:
        absolute = path.expanduser().resolve()
        if not absolute.exists():
            raise FileNotFoundError(f"Watch folder does not exist: {absolute}")
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
                present = {str(item.resolve()) for item in discover(path)}
                removed = 0
                for stored_path in self.store.document_paths_below(str(path)):
                    if stored_path not in present:
                        self.store.remove_document(stored_path)
                        if self.vector_store:
                            self.vector_store.delete_document(stored_path)
                        removed += 1
                results.append({"path": watch.path, "indexed": indexed, "skipped": skipped, "removed": removed, "profile": watch.profile})
        return {"scanned": results}

    def watch_forever(self, stop_event: threading.Event | None = None, interval: float | None = None) -> None:
        stop_event = stop_event or threading.Event()
        delay = interval if interval is not None else self.settings.watch_interval
        while not stop_event.wait(max(0.25, delay)):
            self.scan_watch_folders()

    def import_archive(self, archive: Path, profile: str = "notion") -> tuple[int, int]:
        destination = self.settings.home / "imports" / archive.stem
        destination.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as bundle:
            root = destination.resolve()
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError("Archive contains an unsafe path.")
            bundle.extractall(destination)
        self.store.upsert_watch_folder(str(destination.resolve()), profile)
        return self.ingest(destination)

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
        other_chunks = self.store.all_chunk_previews(exclude_path=path)
        contradictions = []
        for chunk in chunks:
            lowered = chunk.text.lower()
            left_negative = self._is_negative(lowered)
            for other in other_chunks:
                right_negative = self._is_negative(other.text.lower())
                if left_negative != right_negative:
                    overlap = set(re.findall(r"[a-z0-9_]+", lowered)) & set(re.findall(r"[a-z0-9_]+", other.text.lower()))
                    if len(overlap) >= 4:
                        contradictions.append(
                            {
                                "left": chunk.citation,
                                "right": other.citation,
                                "shared_terms": sorted(list(overlap))[:8],
                            }
                        )
                        break
        return contradictions[:10]

    def compare(self, left: str, right: str) -> dict:
        left_chunks = self.store.document_chunks(left)
        right_chunks = self.store.document_chunks(right)
        if not left_chunks or not right_chunks:
            raise ValueError("Both documents must exist in the library.")
        left_terms = self._meaningful_terms(" ".join(chunk.text for chunk in left_chunks))
        right_terms = self._meaningful_terms(" ".join(chunk.text for chunk in right_chunks))
        shared = sorted(left_terms & right_terms)
        union = left_terms | right_terms
        return {
            "left": left,
            "right": right,
            "similarity": len(shared) / max(1, len(union)),
            "shared_topics": shared[:30],
            "left_only": sorted(left_terms - right_terms)[:30],
            "right_only": sorted(right_terms - left_terms)[:30],
            "contradictions": self.contradictions(left),
        }

    def backup(self) -> dict:
        return self.store.backup_payload()

    def diagnostics(self, limit: int = 100) -> list[dict]:
        return [diagnostic.__dict__ for diagnostic in self.store.list_diagnostics(limit)]

    def _default_embedder(self) -> Embedder:
        if self.settings.embed_provider == "ollama":
            return OllamaEmbedder(self.settings.ollama_url, self.settings.embed_model)
        return HashingEmbedder()

    def _default_vector_store(self) -> ChromaVectorAdapter | None:
        if self.settings.vector_provider == "chroma":
            return ChromaVectorAdapter(self.settings.home / "chroma")
        return None

    def _retrieval_generator(self) -> Generator | None:
        if self.settings.embed_provider == "ollama":
            return OllamaGenerator(self.settings.ollama_url, self.settings.ollama_model)
        return None

    def provider_status(self) -> dict:
        return {
            "embed_provider": self.settings.embed_provider,
            "embed_model": self.settings.embed_model,
            "embed_backend": getattr(self.embedder, "last_backend", "hash"),
            "embed_error": getattr(self.embedder, "last_error", None),
            "vector_provider": self.settings.vector_provider,
            "ollama_model": self.settings.ollama_model,
        }

    @staticmethod
    def _is_negative(text: str) -> bool:
        return bool(re.search(r"\b(no|not|never|cannot|can't|isn't|wasn't|won't|false|incorrect)\b", text))

    @staticmethod
    def _meaningful_terms(text: str) -> set[str]:
        stop = {"the", "and", "that", "this", "with", "from", "have", "were", "was", "for", "are", "but", "not", "into", "your"}
        return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 3 and token not in stop}

    def _query_plan(self, query: str, generator: Generator | None = None) -> dict[str, object]:
        expanded = self._fallback_expansions(query)
        hyde = ""
        if generator:
            try:
                response = generator.generate(
                    "Rewrite this search query for personal note retrieval. "
                    "Return compact JSON with keys variations, synonyms, hyde. "
                    "variations is 2-4 alternate phrasings, synonyms is 3-8 related words, "
                    "and hyde is a short hypothetical answer that would appear in relevant notes.\n\n"
                    f"QUERY: {query}"
                )
                payload = self._json_object(response)
                expanded.extend(str(item) for item in payload.get("variations", []) if str(item).strip())
                synonyms = " ".join(str(item) for item in payload.get("synonyms", []) if str(item).strip())
                if synonyms:
                    expanded.append(f"{query} {synonyms}")
                hyde = str(payload.get("hyde", "")).strip()
            except Exception:
                hyde = ""
        if not hyde:
            hyde = self._fallback_hyde(query, expanded)
        deduped = []
        seen = {query.strip().lower()}
        for item in expanded:
            normalized = item.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                deduped.append(normalized)
                seen.add(key)
        return {"expanded_queries": deduped[:5], "hyde": hyde}

    @staticmethod
    def _fallback_expansions(query: str) -> list[str]:
        synonym_groups = [
            {"car", "cars", "auto", "automobile", "vehicle", "vehicles"},
            {"doc", "docs", "document", "documents", "file", "files", "note", "notes"},
            {"search", "find", "lookup", "retrieve", "retrieval"},
            {"fast", "quick", "speed", "latency", "performance"},
            {"ai", "llm", "model", "agent", "assistant"},
            {"pdf", "scan", "scanned", "ocr", "image"},
        ]
        tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
        additions = sorted({term for group in synonym_groups if tokens & group for term in group})
        return [f"{query} {' '.join(additions)}"] if additions else []

    @staticmethod
    def _fallback_hyde(query: str, expanded: list[str]) -> str:
        terms = " ".join(expanded[:2]) if expanded else query
        return f"Relevant notes may discuss {terms} and directly answer: {query}"

    @staticmethod
    def _json_object(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else {}

    def _vector_candidates(
        self,
        query_vector: list[float],
        limit: int,
        tag: str | None = None,
        folder: str | None = None,
        file_type: str | None = None,
    ) -> list[tuple[int, float]]:
        if self.vector_store and not tag:
            where = self._chroma_where(folder=folder, file_type=file_type)
            try:
                return self.vector_store.query(query_vector, limit, where=where)
            except Exception:
                pass
        return self.store.vector_search(query_vector, limit, tag=tag, folder=folder, file_type=file_type)

    @staticmethod
    def _chroma_where(folder: str | None = None, file_type: str | None = None) -> dict | None:
        clauses = []
        if folder:
            clauses.append({"folder": folder})
        if file_type:
            clauses.append({"file_type": file_type})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _reciprocal_rank_fusion(rankings: list[tuple[list[int], float]], k: int = 60) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for ids, weight in rankings:
            for rank, chunk_id in enumerate(ids, 1):
                scores[chunk_id] += weight / (k + rank)
        return dict(scores)

    def _rerank(self, query: str, hits: list[SearchHit], generator: Generator | None = None) -> list[SearchHit]:
        qwen_order = self._qwen_rerank(query, hits[:20], generator) if generator else []
        order_boost = {chunk_id: (len(qwen_order) - index) / max(1, len(qwen_order)) for index, chunk_id in enumerate(qwen_order)}
        terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        reranked: list[SearchHit] = []
        for hit in hits:
            text_terms = set(re.findall(r"[a-z0-9_]+", hit.text.lower()))
            title_terms = set(re.findall(r"[a-z0-9_]+", hit.title.lower()))
            overlap = len(terms & text_terms) / max(1, len(terms))
            title_overlap = len(terms & title_terms) / max(1, len(terms))
            tag_overlap = len(terms & set(hit.tags)) / max(1, len(terms))
            phrase_bonus = 0.12 if query.lower() in hit.text.lower() else 0.0
            citation_bonus = 0.05 if ("#L" in hit.citation or "#page=" in hit.citation) else 0.0
            reranked.append(
                SearchHit(
                    hit.chunk_id,
                    hit.text,
                    hit.title,
                    hit.citation,
                    hit.score
                    + 0.30 * overlap
                    + 0.14 * title_overlap
                    + 0.08 * tag_overlap
                    + 0.35 * order_boost.get(hit.chunk_id, 0.0)
                    + phrase_bonus
                    + citation_bonus,
                    hit.tags,
                )
            )
        return sorted(reranked, key=lambda hit: hit.score, reverse=True)

    def _qwen_rerank(self, query: str, hits: list[SearchHit], generator: Generator | None) -> list[int]:
        if not generator or not hits:
            return []
        try:
            candidates = "\n".join(f"{hit.chunk_id}: {hit.title}\n{hit.text[:700]}" for hit in hits)
            response = generator.generate(
                "Rerank these retrieved note chunks by relevance to the query. "
                "Return JSON only, with key ranked_ids as the chunk ids in best-first order. "
                "Prefer direct semantic relevance over keyword coincidence.\n\n"
                f"QUERY: {query}\n\nCANDIDATES:\n{candidates}"
            )
            payload = self._json_object(response)
            allowed = {hit.chunk_id for hit in hits}
            return [int(item) for item in payload.get("ranked_ids", []) if int(item) in allowed]
        except Exception:
            return []

    def _mmr(self, query_vector: list[float], hits: list[SearchHit], limit: int, lambda_mult: float = 0.72) -> list[SearchHit]:
        if len(hits) <= limit:
            return hits
        vectors = self.store.vectors_by_ids([hit.chunk_id for hit in hits])
        by_id = {hit.chunk_id: hit for hit in hits}
        selected: list[int] = []
        remaining = [hit.chunk_id for hit in hits]
        max_score = max((hit.score for hit in hits), default=1.0) or 1.0
        while remaining and len(selected) < limit:
            best_id = None
            best_score = -math.inf
            for chunk_id in remaining:
                hit = by_id[chunk_id]
                vector = vectors.get(chunk_id, [])
                relevance = hit.score / max_score
                query_similarity = self._cosine(query_vector, vector) if vector else 0.0
                duplicate_penalty = 0.0
                if selected and vector:
                    duplicate_penalty = max(self._cosine(vector, vectors.get(other_id, [])) for other_id in selected)
                same_document_penalty = 0.12 if any(self._document_key(hit) == self._document_key(by_id[other_id]) for other_id in selected) else 0.0
                score = lambda_mult * (0.75 * relevance + 0.25 * query_similarity) - (1 - lambda_mult) * duplicate_penalty - same_document_penalty
                if score > best_score:
                    best_id = chunk_id
                    best_score = score
            if best_id is None:
                break
            selected.append(best_id)
            remaining.remove(best_id)
        return [by_id[chunk_id] for chunk_id in selected]

    @staticmethod
    def _document_key(hit: SearchHit) -> str:
        return hit.citation.split("#", 1)[0]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
        return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0

    def _validate_citations(self, answer: str, hits: list[SearchHit]) -> CitationValidation:
        cited_numbers = tuple(sorted({int(match) for match in REFERENCE_PATTERN.findall(answer)}))
        available = {index for index, _ in enumerate(hits, 1)}
        missing = tuple(number for number in cited_numbers if number not in available)
        unsupported: list[int] = []
        normalized_answer = re.sub(r"([.!?])\s+(\[\d+\])", r" \2\1", answer)
        sentences = [sentence.strip().lower() for sentence in re.split(r"(?<=[.!?])\s+", normalized_answer) if sentence.strip()]
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
