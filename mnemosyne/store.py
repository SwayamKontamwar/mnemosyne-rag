from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from .models import Chunk, ChunkPreview, ConversationEntry, DocumentRecord, GraphEdge, ParseDiagnostic, SavedSearch, SearchHit, TopicCluster, WatchFolder


class KnowledgeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA foreign_keys=ON")

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY, digest TEXT NOT NULL, indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                title TEXT DEFAULT '', source_type TEXT DEFAULT 'file', file_type TEXT DEFAULT 'txt',
                folder TEXT DEFAULT 'root', tags TEXT DEFAULT '[]', links TEXT DEFAULT '[]',
                document_id TEXT DEFAULT '', deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS document_revisions (
                id INTEGER PRIMARY KEY,
                document_path TEXT NOT NULL,
                version INTEGER NOT NULL,
                digest TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                tombstone INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                UNIQUE(document_path, version),
                FOREIGN KEY(document_path) REFERENCES documents(path)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_path TEXT NOT NULL, title TEXT NOT NULL,
                text TEXT NOT NULL, ordinal INTEGER NOT NULL, citation TEXT NOT NULL,
                vector TEXT NOT NULL, tags TEXT DEFAULT '[]',
                start_line INTEGER, end_line INTEGER, page INTEGER,
                revision_id INTEGER, document_version INTEGER DEFAULT 1,
                content_hash TEXT DEFAULT '', valid_from TEXT, valid_to TEXT,
                embedding_space TEXT DEFAULT '', embedding_dimensions INTEGER DEFAULT 0,
                FOREIGN KEY(document_path) REFERENCES documents(path)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, title, content='chunks', content_rowid='id'
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS saved_searches (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                query TEXT NOT NULL,
                tag TEXT,
                folder TEXT,
                file_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY,
                mode TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT DEFAULT '',
                payload TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS watch_folders (
                path TEXT PRIMARY KEY,
                profile TEXT DEFAULT 'local',
                enabled INTEGER DEFAULT 1,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                query TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                verdict TEXT NOT NULL,
                cited_numbers TEXT DEFAULT '[]',
                missing_numbers TEXT DEFAULT '[]',
                unsupported_numbers TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS parse_diagnostics (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                level TEXT NOT NULL,
                code TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, title) VALUES (new.id, new.text, new.title);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, title)
                VALUES ('delete', old.id, old.text, old.title);
            END;
            """
        )
        self._ensure_columns(
            "documents",
            {
                "title": "TEXT DEFAULT ''",
                "source_type": "TEXT DEFAULT 'file'",
                "file_type": "TEXT DEFAULT 'txt'",
                "folder": "TEXT DEFAULT 'root'",
                "tags": "TEXT DEFAULT '[]'",
                "links": "TEXT DEFAULT '[]'",
                "document_id": "TEXT DEFAULT ''",
                "deleted_at": "TEXT",
            },
        )
        self._ensure_columns(
            "chunks",
            {
                "tags": "TEXT DEFAULT '[]'",
                "start_line": "INTEGER",
                "end_line": "INTEGER",
                "page": "INTEGER",
                "revision_id": "INTEGER",
                "document_version": "INTEGER DEFAULT 1",
                "content_hash": "TEXT DEFAULT ''",
                "valid_from": "TEXT",
                "valid_to": "TEXT",
                "embedding_space": "TEXT DEFAULT ''",
                "embedding_dimensions": "INTEGER DEFAULT 0",
            },
        )
        self._backfill_revisions()
        self.connection.commit()

    def digest_for(self, path: str) -> str | None:
        row = self.connection.execute("SELECT digest FROM documents WHERE path = ? AND deleted_at IS NULL", (path,)).fetchone()
        return row["digest"] if row else None

    def active_vectors_by_hash(self, path: str, embedding_space: str, dimensions: int) -> dict[str, list[list[float]]]:
        rows = self.connection.execute(
            """SELECT content_hash, vector FROM chunks
               WHERE document_path = ? AND valid_to IS NULL
                 AND embedding_space = ? AND embedding_dimensions = ?""",
            (path, embedding_space, dimensions),
        ).fetchall()
        vectors: dict[str, list[list[float]]] = defaultdict(list)
        for row in rows:
            if row["content_hash"]:
                vectors[row["content_hash"]].append(json.loads(row["vector"]))
        return dict(vectors)

    def replace_document(
        self, path: str, digest: str, chunks: Iterable[tuple[Chunk, list[float]]], metadata: dict,
        embedding_space: str, embedding_dimensions: int,
    ) -> dict:
        prepared = list(chunks)
        now = _utc_now()
        content = str(metadata.get("content") or "\n\n".join(chunk.text for chunk, _ in prepared))
        version = self._next_version(path)
        document_id = self._document_id(path)
        revision_metadata = {
            "title": metadata["title"],
            "source_type": metadata["source_type"],
            "file_type": metadata["file_type"],
            "folder": metadata["folder"],
            "tags": list(metadata["tags"]),
            "links": list(metadata["links"]),
        }
        active_rows = self.connection.execute(
            "SELECT * FROM chunks WHERE document_path = ? AND valid_to IS NULL ORDER BY ordinal",
            (path,),
        ).fetchall()
        active_by_hash: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in active_rows:
            if row["embedding_space"] == embedding_space and row["embedding_dimensions"] == embedding_dimensions:
                active_by_hash[row["content_hash"] or _content_hash(row["text"])].append(row)
        reused_ids: list[int] = []
        inserted: list[tuple[int, list[float], dict]] = []
        closed: list[tuple[int, list[float], dict]] = []
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO documents(path, digest, title, source_type, file_type, folder, tags, links, document_id, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(path) DO UPDATE SET
                    digest=excluded.digest,
                    title=excluded.title,
                    source_type=excluded.source_type,
                    file_type=excluded.file_type,
                    folder=excluded.folder,
                    tags=excluded.tags,
                    links=excluded.links,
                    document_id=COALESCE(NULLIF(documents.document_id, ''), excluded.document_id),
                    deleted_at=NULL,
                    indexed_at=CURRENT_TIMESTAMP
                """,
                (
                    path,
                    digest,
                    metadata["title"],
                    metadata["source_type"],
                    metadata["file_type"],
                    metadata["folder"],
                    json.dumps(metadata["tags"]),
                    json.dumps(metadata["links"]),
                    document_id,
                ),
            )
            cursor = self.connection.execute(
                """
                INSERT INTO document_revisions(document_path, version, digest, content, created_at, tombstone, metadata)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    path,
                    version,
                    digest,
                    content,
                    now,
                    json.dumps(revision_metadata),
                ),
            )
            revision_id = int(cursor.lastrowid)
            for chunk, vector in prepared:
                content_hash = _content_hash(chunk.text)
                reusable = active_by_hash.get(content_hash, [])
                if reusable:
                    row = reusable.pop(0)
                    reused_ids.append(row["id"])
                    continue
                citation = self._citation(path, version, chunk.start_line, chunk.end_line, chunk.page)
                cursor = self.connection.execute(
                    """
                    INSERT INTO chunks(document_path,title,text,ordinal,citation,vector,tags,start_line,end_line,page,
                                       revision_id,document_version,content_hash,valid_from,valid_to,
                                       embedding_space,embedding_dimensions)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)
                    """,
                    (
                        chunk.document_path,
                        chunk.title,
                        chunk.text,
                        chunk.ordinal,
                        citation,
                        json.dumps(vector),
                        json.dumps(chunk.tags),
                        chunk.start_line,
                        chunk.end_line,
                        chunk.page,
                        revision_id,
                        version,
                        content_hash,
                        now,
                        embedding_space,
                        embedding_dimensions,
                    )
                )
                chunk_id = int(cursor.lastrowid)
                inserted.append((chunk_id, vector, self._vector_metadata(
                    path, chunk, citation, now, None, content_hash, embedding_space, embedding_dimensions
                )))
            for remaining in active_by_hash.values():
                for row in remaining:
                    self.connection.execute("UPDATE chunks SET valid_to = ? WHERE id = ?", (now, row["id"]))
                    closed.append(
                        (
                            row["id"],
                            json.loads(row["vector"]),
                            self._vector_metadata_from_row(row, valid_to=now),
                        )
                    )
        active_ids = [
            row["id"]
            for row in self.connection.execute(
                "SELECT id FROM chunks WHERE document_path = ? AND valid_to IS NULL ORDER BY ordinal, id",
                (path,),
            )
        ]
        return {
            "revision_id": revision_id,
            "version": version,
            "created_at": now,
            "chunk_ids": active_ids,
            "inserted": inserted,
            "closed": closed,
            "reused": reused_ids,
        }

    def remove_document(self, path: str) -> dict:
        now = _utc_now()
        version = self._next_version(path)
        row = self.connection.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
        if not row or row["deleted_at"]:
            return {"closed": [], "version": version, "created_at": now}
        closed: list[tuple[int, list[float], dict]] = []
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO document_revisions(document_path, version, digest, content, created_at, tombstone, metadata)
                VALUES (?, ?, ?, '', ?, 1, ?)
                """,
                (path, version, row["digest"], now, json.dumps({"deleted": True})),
            )
            revision_id = int(cursor.lastrowid)
            active = self.connection.execute(
                "SELECT * FROM chunks WHERE document_path = ? AND valid_to IS NULL",
                (path,),
            ).fetchall()
            self.connection.execute("UPDATE chunks SET valid_to = ? WHERE document_path = ? AND valid_to IS NULL", (now, path))
            self.connection.execute("UPDATE documents SET deleted_at = ?, indexed_at = CURRENT_TIMESTAMP WHERE path = ?", (now, path))
            for chunk in active:
                closed.append((chunk["id"], json.loads(chunk["vector"]), self._vector_metadata_from_row(chunk, valid_to=now)))
        return {"revision_id": revision_id, "version": version, "created_at": now, "closed": closed}

    def document_paths_below(self, folder: str) -> list[str]:
        prefix = str(Path(folder).resolve())
        return [row["path"] for row in self.connection.execute("SELECT path FROM documents WHERE path LIKE ? AND deleted_at IS NULL", (f"{prefix}%",))]

    def hybrid_search(self, query: str, query_vector: list[float], limit: int = 8, tag: str | None = None, folder: str | None = None, file_type: str | None = None, as_of: str | None = None) -> list[SearchHit]:
        rows = self._chunk_rows(tag=tag, folder=folder, file_type=file_type, as_of=as_of)
        vector_scores = {row["id"]: _cosine(query_vector, json.loads(row["vector"])) for row in rows}
        keyword_scores: dict[int, float] = {}
        terms = [term.replace('"', "") for term in query.split() if term.strip()]
        if terms:
            expression = " OR ".join(f'"{term}"' for term in terms)
            for row in self.connection.execute(
                "SELECT rowid, bm25(chunks_fts) rank FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
                (expression, limit * 5),
            ):
                keyword_scores[row["rowid"]] = 1.0 / (1.0 + max(0.0, row["rank"] + 10.0))
        ranked = sorted(rows, key=lambda row: 0.7 * vector_scores[row["id"]] + 0.3 * keyword_scores.get(row["id"], 0), reverse=True)
        return [
            SearchHit(
                row["id"],
                row["text"],
                row["title"],
                row["citation"],
                0.7 * vector_scores[row["id"]] + 0.3 * keyword_scores.get(row["id"], 0),
                tuple(json.loads(row["tags"] or "[]")),
            )
            for row in ranked[:limit]
        ]

    def keyword_search(self, query: str, limit: int = 20, tag: str | None = None, folder: str | None = None, file_type: str | None = None, as_of: str | None = None) -> list[int]:
        terms = [term.replace('"', "") for term in re.findall(r"[a-z0-9_]+", query.lower()) if term.strip()]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        filters, params = self._filter_sql(tag=tag, folder=folder, file_type=file_type, alias="d")
        time_filter, time_params = self._time_filter_sql(as_of, alias="c")
        filters.append(time_filter)
        params.extend(time_params)
        where = f"AND {' AND '.join(filters)}" if filters else ""
        rows = self.connection.execute(
            f"""
            SELECT c.id, bm25(chunks_fts) rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.path = c.document_path
            WHERE chunks_fts MATCH ? {where}
            ORDER BY rank
            LIMIT ?
            """,
            [expression, *params, limit],
        ).fetchall()
        return [row["id"] for row in rows]

    def vector_search(
        self, query_vector: list[float], limit: int = 20, tag: str | None = None,
        folder: str | None = None, file_type: str | None = None, as_of: str | None = None,
        embedding_space: str | None = None,
    ) -> list[tuple[int, float]]:
        rows = self._chunk_rows(tag=tag, folder=folder, file_type=file_type, as_of=as_of)
        self._validate_vector_rows(rows, embedding_space, len(query_vector))
        scored = [
            (row["id"], _cosine(query_vector, json.loads(row["vector"])))
            for row in rows
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def embedding_spaces(self) -> set[tuple[str, int]]:
        return {
            (str(row["embedding_space"] or ""), int(row["embedding_dimensions"] or 0))
            for row in self.connection.execute(
                "SELECT DISTINCT embedding_space, embedding_dimensions FROM chunks"
            )
        }

    def all_chunk_texts(self) -> list[tuple[int, str]]:
        return [(int(row["id"]), str(row["text"])) for row in self.connection.execute("SELECT id, text FROM chunks ORDER BY id")]

    def replace_all_vectors(self, rows: list[tuple[int, list[float]]], embedding_space: str, dimensions: int) -> None:
        if any(len(vector) != dimensions for _, vector in rows):
            raise RuntimeError("embedding provider returned inconsistent vector dimensions")
        with self.connection:
            self.connection.executemany(
                "UPDATE chunks SET vector = ?, embedding_space = ?, embedding_dimensions = ? WHERE id = ?",
                [(json.dumps(vector), embedding_space, dimensions, chunk_id) for chunk_id, vector in rows],
            )

    def all_vector_rows(self) -> list[tuple[int, list[float], dict]]:
        rows = self.connection.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return [
            (int(row["id"]), json.loads(row["vector"]), self._vector_metadata_from_row(row))
            for row in rows
        ]

    @staticmethod
    def _validate_vector_rows(rows: list[sqlite3.Row], embedding_space: str | None, dimensions: int) -> None:
        for row in rows:
            stored = json.loads(row["vector"])
            if embedding_space is not None and row["embedding_space"] != embedding_space:
                raise RuntimeError(
                    f"mixed embedding spaces: query={embedding_space!r}, chunk {row['id']}={row['embedding_space']!r}; re-run ingest"
                )
            if row["embedding_dimensions"] != dimensions or len(stored) != dimensions:
                raise RuntimeError(
                    f"embedding dimension mismatch: query={dimensions}, chunk {row['id']} metadata={row['embedding_dimensions']} stored={len(stored)}; re-run ingest"
                )

    def chunks_for_document(self, name: str, as_of: str | None = None) -> list[sqlite3.Row]:
        time_filter, params = self._time_filter_sql(as_of)
        return self.connection.execute(
            f"SELECT * FROM chunks WHERE ({time_filter}) AND (document_path LIKE ? OR title = ?)",
            [*params, f"%{name}%", name],
        ).fetchall()

    def list_documents(self, tag: str | None = None, folder: str | None = None, file_type: str | None = None, as_of: str | None = None) -> list[dict]:
        filters = []
        params: list[object] = []
        if tag:
            filters.append("d.tags LIKE ?")
            params.append(f'%"{tag.lower()}"%')
        if folder:
            filters.append("d.folder = ?")
            params.append(folder)
        if file_type:
            filters.append("d.file_type = ?")
            params.append(file_type)
        if as_of is None:
            filters.append("d.deleted_at IS NULL")
        else:
            filters.append("EXISTS (SELECT 1 FROM chunks c2 WHERE c2.document_path = d.path AND c2.valid_from <= ? AND (c2.valid_to IS NULL OR c2.valid_to > ?))")
            params.extend([as_of, as_of])
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self.connection.execute(
            f"""
            SELECT d.path, d.title, d.digest, d.indexed_at, d.source_type, d.file_type, d.folder, d.tags, d.links,
                   COUNT(c.id) chunk_count,
                   COALESCE(SUM(LENGTH(c.text)), 0) character_count
            FROM documents d LEFT JOIN chunks c ON c.document_path = d.path
            {where}
            GROUP BY d.path, d.digest, d.indexed_at
            ORDER BY d.indexed_at DESC
            """,
            params,
        ).fetchall()
        return [
            dict(
                DocumentRecord(
                    path=row["path"],
                    title=row["title"] or Path(row["path"]).stem,
                    digest=row["digest"],
                    indexed_at=row["indexed_at"],
                    source_type=row["source_type"],
                    file_type=row["file_type"],
                    folder=row["folder"],
                    tags=tuple(json.loads(row["tags"] or "[]")),
                    links=tuple(json.loads(row["links"] or "[]")),
                    chunk_count=row["chunk_count"],
                    character_count=row["character_count"],
                ).__dict__
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        documents = self.connection.execute("SELECT COUNT(*) count FROM documents WHERE deleted_at IS NULL").fetchone()["count"]
        chunks = self.connection.execute("SELECT COUNT(*) count FROM chunks WHERE valid_to IS NULL").fetchone()["count"]
        characters = self.connection.execute(
            "SELECT COALESCE(SUM(LENGTH(text)), 0) count FROM chunks WHERE valid_to IS NULL"
        ).fetchone()["count"]
        tags = self.connection.execute(
            "SELECT COALESCE(SUM(json_array_length(tags)), 0) count FROM documents"
        ).fetchone()["count"]
        saved_searches = self.connection.execute("SELECT COUNT(*) count FROM saved_searches").fetchone()["count"]
        conversations = self.connection.execute("SELECT COUNT(*) count FROM conversation_history").fetchone()["count"]
        watch_folders = self.connection.execute("SELECT COUNT(*) count FROM watch_folders WHERE enabled = 1").fetchone()["count"]
        return {
            "documents": documents,
            "chunks": chunks,
            "characters": characters,
            "tags": tags,
            "saved_searches": saved_searches,
            "conversations": conversations,
            "watch_folders": watch_folders,
        }

    def list_tags(self) -> list[str]:
        rows = self.connection.execute("SELECT tags FROM documents").fetchall()
        tags = sorted({tag for row in rows for tag in json.loads(row["tags"] or "[]")})
        return tags

    def list_folders(self) -> list[str]:
        rows = self.connection.execute("SELECT DISTINCT folder FROM documents ORDER BY folder").fetchall()
        return [row["folder"] for row in rows if row["folder"]]

    def chunk_preview(self, chunk_id: int) -> ChunkPreview | None:
        row = self.connection.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not row:
            return None
        return ChunkPreview(
            chunk_id=row["id"],
            title=row["title"],
            text=row["text"],
            citation=row["citation"],
            page=row["page"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            tags=tuple(json.loads(row["tags"] or "[]")),
            document_path=row["document_path"],
            revision_id=row["revision_id"],
            document_version=row["document_version"],
            valid_from=row["valid_from"] or "",
            valid_to=row["valid_to"],
            content_hash=row["content_hash"] or "",
        )

    def hits_by_ids(self, ids: list[int], scores: dict[int, float], as_of: str | None = None) -> list[SearchHit]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        time_filter, time_params = self._time_filter_sql(as_of)
        rows = self.connection.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND {time_filter}", [*ids, *time_params]).fetchall()
        by_id = {row["id"]: row for row in rows}
        return [
            self._hit_from_row(by_id[item_id], scores[item_id])
            for item_id in ids if item_id in by_id
        ]

    def vectors_by_ids(self, ids: list[int]) -> dict[int, list[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(f"SELECT id, vector FROM chunks WHERE id IN ({placeholders})", ids).fetchall()
        return {row["id"]: json.loads(row["vector"]) for row in rows}

    def all_chunk_previews(self, exclude_path: str | None = None, as_of: str | None = None) -> list[ChunkPreview]:
        time_filter, params = self._time_filter_sql(as_of)
        if exclude_path:
            rows = self.connection.execute(
                f"SELECT * FROM chunks WHERE document_path != ? AND {time_filter} ORDER BY id",
                [exclude_path, *params],
            ).fetchall()
        else:
            rows = self.connection.execute(f"SELECT * FROM chunks WHERE {time_filter} ORDER BY id", params).fetchall()
        return [
            self._preview_from_row(row)
            for row in rows
        ]

    def document_chunks(self, path: str, as_of: str | None = None) -> list[ChunkPreview]:
        time_filter, params = self._time_filter_sql(as_of)
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE document_path = ? AND {time_filter} ORDER BY ordinal, id",
            [path, *params],
        ).fetchall()
        return [self._preview_from_row(row) for row in rows]

    def graph_edges(self, limit: int = 24) -> list[GraphEdge]:
        documents = self.list_documents()
        chunk_rows = self.connection.execute("SELECT document_path, vector FROM chunks WHERE valid_to IS NULL").fetchall()
        by_document: dict[str, list[list[float]]] = defaultdict(list)
        for row in chunk_rows:
            by_document[row["document_path"]].append(json.loads(row["vector"]))
        edges: list[GraphEdge] = []
        paths = [doc["path"] for doc in documents]
        for index, left in enumerate(paths):
            left_vector = _average(by_document.get(left, []))
            if not left_vector:
                continue
            left_doc = next(doc for doc in documents if doc["path"] == left)
            left_links = set(left_doc["links"])
            for right in paths[index + 1:]:
                right_vector = _average(by_document.get(right, []))
                if not right_vector:
                    continue
                score = _cosine(left_vector, right_vector)
                right_doc = next(doc for doc in documents if doc["path"] == right)
                reasons: list[str] = []
                if score >= 0.3:
                    reasons.append("semantic similarity")
                if set(left_doc["tags"]) & set(right_doc["tags"]):
                    reasons.append("shared tags")
                if right_doc["title"] in left_links or left_doc["title"] in set(right_doc["links"]):
                    reasons.append("note links")
                if reasons:
                    edges.append(GraphEdge(left, right, score, ", ".join(reasons)))
        return sorted(edges, key=lambda edge: edge.weight, reverse=True)[:limit]

    def clusters(self, limit: int = 8) -> list[TopicCluster]:
        documents = self.list_documents()
        grouped: dict[str, list[dict]] = defaultdict(list)
        for document in documents:
            if document["tags"]:
                grouped[document["tags"][0]].append(document)
            else:
                grouped[document["folder"]].append(document)
        clusters: list[TopicCluster] = []
        for name, members in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:limit]:
            keywords = Counter(token for doc in members for token in doc["tags"]).most_common(4)
            clusters.append(
                TopicCluster(
                    name=name or "uncategorized",
                    document_paths=tuple(doc["path"] for doc in members),
                    keywords=tuple(keyword for keyword, _ in keywords),
                )
            )
        return clusters

    def save_setting(self, key: str, value: dict | list | str | int | bool | None) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO app_settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def load_settings(self) -> dict:
        rows = self.connection.execute("SELECT key, value FROM app_settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def create_saved_search(self, name: str, query: str, tag: str | None, folder: str | None, file_type: str | None) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO saved_searches(name, query, tag, folder, file_type) VALUES (?, ?, ?, ?, ?)",
                (name, query, tag, folder, file_type),
            )
        return int(cursor.lastrowid)

    def list_saved_searches(self) -> list[SavedSearch]:
        rows = self.connection.execute("SELECT * FROM saved_searches ORDER BY created_at DESC").fetchall()
        return [
            SavedSearch(
                id=row["id"],
                name=row["name"],
                query=row["query"],
                tag=row["tag"],
                folder=row["folder"],
                file_type=row["file_type"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def log_conversation(self, mode: str, query: str, answer: str = "", payload: dict | None = None) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO conversation_history(mode, query, answer, payload) VALUES (?, ?, ?, ?)",
                (mode, query, answer, json.dumps(payload or {})),
            )
        return int(cursor.lastrowid)

    def conversation_history(self, limit: int = 50) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM conversation_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            dict(
                ConversationEntry(
                    id=row["id"],
                    mode=row["mode"],
                    query=row["query"],
                    answer=row["answer"],
                    created_at=row["created_at"],
                ).__dict__
            ) | {"payload": json.loads(row["payload"] or "{}")}
            for row in rows
        ]

    def upsert_watch_folder(self, path: str, profile: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO watch_folders(path, profile, enabled) VALUES (?, ?, 1)
                ON CONFLICT(path) DO UPDATE SET profile=excluded.profile, enabled=1, indexed_at=CURRENT_TIMESTAMP
                """,
                (path, profile),
            )

    def list_watch_folders(self) -> list[WatchFolder]:
        rows = self.connection.execute("SELECT * FROM watch_folders ORDER BY path").fetchall()
        return [
            WatchFolder(
                path=row["path"],
                profile=row["profile"],
                enabled=bool(row["enabled"]),
                indexed_at=row["indexed_at"],
            )
            for row in rows
        ]

    def save_collection(self, name: str, description: str, tags: list[str], query: str) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO collections(name, description, tags, query) VALUES (?, ?, ?, ?)",
                (name, description, json.dumps(tags), query),
            )
        return int(cursor.lastrowid)

    def list_collections(self) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM collections ORDER BY created_at DESC").fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "tags": json.loads(row["tags"] or "[]"),
                "query": row["query"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def log_evaluation(self, query: str, verdict: str, cited_numbers: list[int], missing_numbers: list[int], unsupported_numbers: list[int]) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO evaluations(query, verdict, cited_numbers, missing_numbers, unsupported_numbers)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query,
                    verdict,
                    json.dumps(cited_numbers),
                    json.dumps(missing_numbers),
                    json.dumps(unsupported_numbers),
                ),
            )

    def evaluation_summary(self) -> dict:
        rows = self.connection.execute("SELECT verdict, COUNT(*) count FROM evaluations GROUP BY verdict").fetchall()
        counts = {row["verdict"]: row["count"] for row in rows}
        recent = self.connection.execute(
            "SELECT query, verdict, created_at FROM evaluations ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        return {
            "counts": counts,
            "recent": [dict(row) for row in recent],
        }

    def backup_payload(self) -> dict:
        return {
            "documents": self.list_documents(),
            "document_revisions": [dict(row) for row in self.connection.execute("SELECT * FROM document_revisions ORDER BY document_path, version")],
            "saved_searches": [search.__dict__ for search in self.list_saved_searches()],
            "conversation_history": self.conversation_history(100),
            "watch_folders": [watch.__dict__ for watch in self.list_watch_folders()],
            "collections": self.list_collections(),
            "settings": self.load_settings(),
            "evaluations": self.evaluation_summary(),
            "diagnostics": [diagnostic.__dict__ for diagnostic in self.list_diagnostics()],
            "chunks": [dict(row) for row in self.connection.execute("SELECT * FROM chunks ORDER BY id")],
        }

    def restore_payload(self, payload: dict) -> dict[str, int]:
        documents = payload.get("documents") or []
        revisions = payload.get("document_revisions") or []
        chunks = payload.get("chunks") or []
        if not isinstance(documents, list) or not isinstance(chunks, list):
            raise ValueError("Backup must contain document and chunk lists.")
        with self.connection:
            for document in documents:
                self.connection.execute(
                    """INSERT INTO documents(path,digest,title,source_type,file_type,folder,tags,links,indexed_at,document_id,deleted_at)
                       VALUES(?,?,?,?,?,?,?,?,COALESCE(?,CURRENT_TIMESTAMP),?,?)
                       ON CONFLICT(path) DO UPDATE SET digest=excluded.digest,title=excluded.title,source_type=excluded.source_type,file_type=excluded.file_type,folder=excluded.folder,tags=excluded.tags,links=excluded.links,indexed_at=excluded.indexed_at,document_id=excluded.document_id,deleted_at=excluded.deleted_at""",
                    (document["path"], document.get("digest", "restored"), document.get("title", ""), document.get("source_type", "file"), document.get("file_type", "txt"), document.get("folder", "root"), json.dumps(document.get("tags", [])), json.dumps(document.get("links", [])), document.get("indexed_at"), document.get("document_id") or self._document_id(document["path"]), document.get("deleted_at")),
                )
                self.connection.execute("DELETE FROM chunks WHERE document_path = ?", (document["path"],))
                self.connection.execute("DELETE FROM document_revisions WHERE document_path = ?", (document["path"],))
            for revision in revisions:
                self.connection.execute(
                    """INSERT INTO document_revisions(id,document_path,version,digest,content,created_at,tombstone,metadata)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (revision.get("id"), revision["document_path"], revision["version"], revision.get("digest", "restored"), revision.get("content", ""), revision.get("created_at") or _utc_now(), int(bool(revision.get("tombstone"))), revision.get("metadata", "{}")),
                )
            for chunk in chunks:
                self.connection.execute(
                    """INSERT INTO chunks(document_path,title,text,ordinal,citation,vector,tags,start_line,end_line,page,
                                          revision_id,document_version,content_hash,valid_from,valid_to)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (chunk["document_path"], chunk["title"], chunk["text"], chunk["ordinal"], chunk["citation"], chunk["vector"], chunk.get("tags", "[]"), chunk.get("start_line"), chunk.get("end_line"), chunk.get("page"), chunk.get("revision_id"), chunk.get("document_version", 1), chunk.get("content_hash") or _content_hash(chunk["text"]), chunk.get("valid_from") or _utc_now(), chunk.get("valid_to")),
                )
        return {"documents": len(documents), "chunks": len(chunks)}

    def log_diagnostic(self, path: str, level: str, code: str, message: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO parse_diagnostics(path, level, code, message) VALUES (?, ?, ?, ?)",
                (path, level, code, message),
            )

    def list_diagnostics(self, limit: int = 100) -> list[ParseDiagnostic]:
        rows = self.connection.execute(
            "SELECT path, level, code, message, created_at FROM parse_diagnostics ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._diagnostic_from_row(row) for row in rows]

    def diagnostics_for_path(self, path: str, limit: int = 5) -> list[ParseDiagnostic]:
        rows = self.connection.execute(
            "SELECT path, level, code, message, created_at FROM parse_diagnostics WHERE path = ? ORDER BY created_at DESC LIMIT ?",
            (path, limit),
        ).fetchall()
        return [self._diagnostic_from_row(row) for row in rows]

    def revision_history(self, path: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM document_revisions WHERE document_path = ? ORDER BY version DESC",
            (path,),
        ).fetchall()
        return [dict(row) | {"metadata": json.loads(row["metadata"] or "{}")} for row in rows]

    def revision_content(self, path: str, version: int) -> str | None:
        row = self.connection.execute(
            "SELECT content FROM document_revisions WHERE document_path = ? AND version = ?",
            (path, version),
        ).fetchone()
        return row["content"] if row else None

    def revision_diff(self, path: str, left: int, right: int) -> dict:
        left_text = self.revision_content(path, left)
        right_text = self.revision_content(path, right)
        if left_text is None or right_text is None:
            raise ValueError("Both revisions must exist.")
        diff = "\n".join(
            unified_diff(
                left_text.splitlines(),
                right_text.splitlines(),
                fromfile=f"v{left}",
                tofile=f"v{right}",
                lineterm="",
            )
        )
        return {"path": path, "left": left, "right": right, "diff": diff}

    def citation_preview(self, citation: str) -> ChunkPreview | None:
        row = self.connection.execute("SELECT * FROM chunks WHERE citation = ?", (citation,)).fetchone()
        return self._preview_from_row(row) if row else None

    @staticmethod
    def _diagnostic_from_row(row: sqlite3.Row) -> ParseDiagnostic:
        return ParseDiagnostic(
            path=row["path"],
            level=row["level"],
            code=row["code"],
            message=row["message"],
            created_at=row["created_at"],
        )

    def _chunk_rows(self, tag: str | None = None, folder: str | None = None, file_type: str | None = None, as_of: str | None = None) -> list[sqlite3.Row]:
        filters, params = self._filter_sql(tag=tag, folder=folder, file_type=file_type)
        time_filter, time_params = self._time_filter_sql(as_of, alias="c")
        filters.append(time_filter)
        params.extend(time_params)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        return self.connection.execute(
            f"""
            SELECT c.*
            FROM chunks c
            JOIN documents d ON d.path = c.document_path
            {where}
            """,
            params,
        ).fetchall()

    def _next_version(self, path: str) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(version), 0) value FROM document_revisions WHERE document_path = ?", (path,)).fetchone()
        return int(row["value"] or 0) + 1

    def _document_id(self, path: str) -> str:
        row = self.connection.execute("SELECT document_id FROM documents WHERE path = ?", (path,)).fetchone()
        if row and row["document_id"]:
            return row["document_id"]
        return sha256(path.encode()).hexdigest()[:16]

    def _citation(self, path: str, version: int, start_line: int | None, end_line: int | None, page: int | None) -> str:
        if page is not None:
            return f"{path}?rev={version}#page={page}"
        if start_line is not None:
            return f"{path}?rev={version}#L{start_line}-L{end_line}"
        return f"{path}?rev={version}"

    def _vector_metadata(
        self, path: str, chunk: Chunk, citation: str, valid_from: str, valid_to: str | None,
        content_hash: str, embedding_space: str, embedding_dimensions: int,
    ) -> dict:
        return {
            "document_path": path,
            "title": chunk.title,
            "citation": citation,
            "folder": Path(path).parent.name,
            "file_type": Path(path).suffix.lower().lstrip(".") or "txt",
            "tags": ",".join(chunk.tags),
            "valid_from": valid_from,
            "valid_to": valid_to or "__open__",
            "valid_to_sort": valid_to or "9999-12-31T23:59:59.999999Z",
            "content_hash": content_hash,
            "embedding_space": embedding_space,
            "embedding_dimensions": embedding_dimensions,
        }

    def _vector_metadata_from_row(self, row: sqlite3.Row, valid_to: str | None = None) -> dict:
        return {
            "document_path": row["document_path"],
            "title": row["title"],
            "citation": row["citation"],
            "folder": Path(row["document_path"]).parent.name,
            "file_type": Path(row["document_path"]).suffix.lower().lstrip(".") or "txt",
            "tags": ",".join(json.loads(row["tags"] or "[]")),
            "valid_from": row["valid_from"],
            "valid_to": valid_to or row["valid_to"] or "__open__",
            "valid_to_sort": valid_to or row["valid_to"] or "9999-12-31T23:59:59.999999Z",
            "content_hash": row["content_hash"],
            "embedding_space": row["embedding_space"],
            "embedding_dimensions": row["embedding_dimensions"],
        }

    def _hit_from_row(self, row: sqlite3.Row, score: float) -> SearchHit:
        return SearchHit(
            row["id"],
            row["text"],
            row["title"],
            row["citation"],
            score,
            tuple(json.loads(row["tags"] or "[]")),
            document_path=row["document_path"],
            revision_id=row["revision_id"],
            document_version=row["document_version"],
            valid_from=row["valid_from"] or "",
            valid_to=row["valid_to"],
            content_hash=row["content_hash"] or "",
        )

    def _preview_from_row(self, row: sqlite3.Row) -> ChunkPreview:
        return ChunkPreview(
            chunk_id=row["id"],
            title=row["title"],
            text=row["text"],
            citation=row["citation"],
            page=row["page"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            tags=tuple(json.loads(row["tags"] or "[]")),
            document_path=row["document_path"],
            revision_id=row["revision_id"],
            document_version=row["document_version"],
            valid_from=row["valid_from"] or "",
            valid_to=row["valid_to"],
            content_hash=row["content_hash"] or "",
        )

    @staticmethod
    def _time_filter_sql(as_of: str | None, alias: str = "") -> tuple[str, list[object]]:
        prefix = f"{alias}." if alias else ""
        if as_of is None:
            return f"{prefix}valid_to IS NULL", []
        return f"{prefix}valid_from <= ? AND ({prefix}valid_to IS NULL OR {prefix}valid_to > ?)", [as_of, as_of]

    def _backfill_revisions(self) -> None:
        now = _utc_now()
        documents = self.connection.execute("SELECT * FROM documents").fetchall()
        with self.connection:
            for document in documents:
                path = document["path"]
                document_id = document["document_id"] or self._document_id(path)
                self.connection.execute("UPDATE documents SET document_id = ? WHERE path = ?", (document_id, path))
                existing = self.connection.execute(
                    "SELECT id, created_at FROM document_revisions WHERE document_path = ? AND version = 1",
                    (path,),
                ).fetchone()
                chunks = self.connection.execute("SELECT * FROM chunks WHERE document_path = ? ORDER BY ordinal, id", (path,)).fetchall()
                content = "\n\n".join(row["text"] for row in chunks)
                if existing:
                    revision_id = existing["id"]
                    created_at = existing["created_at"]
                else:
                    cursor = self.connection.execute(
                        """
                        INSERT INTO document_revisions(document_path, version, digest, content, created_at, tombstone, metadata)
                        VALUES (?, 1, ?, ?, ?, 0, ?)
                        """,
                        (
                            path,
                            document["digest"],
                            content,
                            document["indexed_at"] or now,
                            json.dumps({
                                "title": document["title"],
                                "source_type": document["source_type"],
                                "file_type": document["file_type"],
                                "folder": document["folder"],
                                "tags": json.loads(document["tags"] or "[]"),
                                "links": json.loads(document["links"] or "[]"),
                                "backfilled": True,
                            }),
                        ),
                    )
                    revision_id = int(cursor.lastrowid)
                    created_at = document["indexed_at"] or now
                for row in chunks:
                    content_hash = row["content_hash"] or _content_hash(row["text"])
                    citation = row["citation"]
                    if "?rev=" not in citation:
                        citation = self._citation(path, 1, row["start_line"], row["end_line"], row["page"])
                    self.connection.execute(
                        """
                        UPDATE chunks
                        SET revision_id = COALESCE(revision_id, ?),
                            document_version = COALESCE(document_version, 1),
                            content_hash = ?,
                            valid_from = COALESCE(valid_from, ?),
                            citation = ?
                        WHERE id = ?
                        """,
                        (revision_id, content_hash, created_at, citation, row["id"]),
                    )

    @staticmethod
    def _filter_sql(tag: str | None = None, folder: str | None = None, file_type: str | None = None, alias: str = "d") -> tuple[list[str], list[object]]:
        filters = []
        params: list[object] = []
        if tag:
            filters.append(f"{alias}.tags LIKE ?")
            params.append(f'%"{tag.lower()}"%')
        if folder:
            filters.append(f"{alias}.folder = ?")
            params.append(folder)
        if file_type:
            filters.append(f"{alias}.file_type = ?")
            params.append(file_type)
        return filters, params

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in columns.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"refusing cosine similarity across dimensions {len(left)} and {len(right)}")
    denominator = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
    return sum(a*b for a, b in zip(left, right)) / denominator if denominator else 0.0


def _average(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dims)]


def _content_hash(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
