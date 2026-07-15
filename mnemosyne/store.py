from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
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
                folder TEXT DEFAULT 'root', tags TEXT DEFAULT '[]', links TEXT DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_path TEXT NOT NULL, title TEXT NOT NULL,
                text TEXT NOT NULL, ordinal INTEGER NOT NULL, citation TEXT NOT NULL,
                vector TEXT NOT NULL, tags TEXT DEFAULT '[]',
                start_line INTEGER, end_line INTEGER, page INTEGER,
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
            },
        )
        self._ensure_columns(
            "chunks",
            {
                "tags": "TEXT DEFAULT '[]'",
                "start_line": "INTEGER",
                "end_line": "INTEGER",
                "page": "INTEGER",
            },
        )
        self.connection.commit()

    def digest_for(self, path: str) -> str | None:
        row = self.connection.execute("SELECT digest FROM documents WHERE path = ?", (path,)).fetchone()
        return row["digest"] if row else None

    def replace_document(self, path: str, digest: str, chunks: Iterable[tuple[Chunk, list[float]]], metadata: dict) -> list[int]:
        prepared = list(chunks)
        with self.connection:
            self.connection.execute("DELETE FROM chunks WHERE document_path = ?", (path,))
            self.connection.execute(
                """
                INSERT INTO documents(path, digest, title, source_type, file_type, folder, tags, links)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    digest=excluded.digest,
                    title=excluded.title,
                    source_type=excluded.source_type,
                    file_type=excluded.file_type,
                    folder=excluded.folder,
                    tags=excluded.tags,
                    links=excluded.links,
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
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO chunks(document_path,title,text,ordinal,citation,vector,tags,start_line,end_line,page)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    (
                        c.document_path,
                        c.title,
                        c.text,
                        c.ordinal,
                        c.citation,
                        json.dumps(v),
                        json.dumps(c.tags),
                        c.start_line,
                        c.end_line,
                        c.page,
                    )
                    for c, v in prepared
                ),
            )
        return [row["id"] for row in self.connection.execute("SELECT id FROM chunks WHERE document_path = ? ORDER BY ordinal", (path,))]

    def remove_document(self, path: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM chunks WHERE document_path = ?", (path,))
            self.connection.execute("DELETE FROM documents WHERE path = ?", (path,))

    def document_paths_below(self, folder: str) -> list[str]:
        prefix = str(Path(folder).resolve())
        return [row["path"] for row in self.connection.execute("SELECT path FROM documents WHERE path LIKE ?", (f"{prefix}%",))]

    def hybrid_search(self, query: str, query_vector: list[float], limit: int = 8, tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> list[SearchHit]:
        rows = self._chunk_rows(tag=tag, folder=folder, file_type=file_type)
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

    def keyword_search(self, query: str, limit: int = 20, tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> list[int]:
        terms = [term.replace('"', "") for term in re.findall(r"[a-z0-9_]+", query.lower()) if term.strip()]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        filters, params = self._filter_sql(tag=tag, folder=folder, file_type=file_type, alias="d")
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

    def vector_search(self, query_vector: list[float], limit: int = 20, tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> list[tuple[int, float]]:
        rows = self._chunk_rows(tag=tag, folder=folder, file_type=file_type)
        scored = [
            (row["id"], _cosine(query_vector, json.loads(row["vector"])))
            for row in rows
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def chunks_for_document(self, name: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM chunks WHERE document_path LIKE ? OR title = ?", (f"%{name}%", name)
        ).fetchall()

    def list_documents(self, tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> list[dict]:
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
        documents = self.connection.execute("SELECT COUNT(*) count FROM documents").fetchone()["count"]
        chunks = self.connection.execute("SELECT COUNT(*) count FROM chunks").fetchone()["count"]
        characters = self.connection.execute(
            "SELECT COALESCE(SUM(LENGTH(text)), 0) count FROM chunks"
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
        )

    def hits_by_ids(self, ids: list[int], scores: dict[int, float]) -> list[SearchHit]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids).fetchall()
        by_id = {row["id"]: row for row in rows}
        return [
            SearchHit(item_id, by_id[item_id]["text"], by_id[item_id]["title"], by_id[item_id]["citation"], scores[item_id], tuple(json.loads(by_id[item_id]["tags"] or "[]")))
            for item_id in ids if item_id in by_id
        ]

    def vectors_by_ids(self, ids: list[int]) -> dict[int, list[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(f"SELECT id, vector FROM chunks WHERE id IN ({placeholders})", ids).fetchall()
        return {row["id"]: json.loads(row["vector"]) for row in rows}

    def all_chunk_previews(self, exclude_path: str | None = None) -> list[ChunkPreview]:
        rows = self.connection.execute("SELECT * FROM chunks WHERE document_path != ? ORDER BY id" if exclude_path else "SELECT * FROM chunks ORDER BY id", (exclude_path,) if exclude_path else ()).fetchall()
        return [
            ChunkPreview(row["id"], row["title"], row["text"], row["citation"], row["page"], row["start_line"], row["end_line"], tuple(json.loads(row["tags"] or "[]")))
            for row in rows
        ]

    def document_chunks(self, path: str) -> list[ChunkPreview]:
        rows = self.connection.execute(
            "SELECT * FROM chunks WHERE document_path = ? ORDER BY ordinal",
            (path,),
        ).fetchall()
        return [
            ChunkPreview(
                chunk_id=row["id"],
                title=row["title"],
                text=row["text"],
                citation=row["citation"],
                page=row["page"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                tags=tuple(json.loads(row["tags"] or "[]")),
            )
            for row in rows
        ]

    def graph_edges(self, limit: int = 24) -> list[GraphEdge]:
        documents = self.list_documents()
        chunk_rows = self.connection.execute("SELECT document_path, vector FROM chunks").fetchall()
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
        chunks = payload.get("chunks") or []
        if not isinstance(documents, list) or not isinstance(chunks, list):
            raise ValueError("Backup must contain document and chunk lists.")
        with self.connection:
            for document in documents:
                self.connection.execute(
                    """INSERT INTO documents(path,digest,title,source_type,file_type,folder,tags,links,indexed_at)
                       VALUES(?,?,?,?,?,?,?,?,COALESCE(?,CURRENT_TIMESTAMP))
                       ON CONFLICT(path) DO UPDATE SET digest=excluded.digest,title=excluded.title,source_type=excluded.source_type,file_type=excluded.file_type,folder=excluded.folder,tags=excluded.tags,links=excluded.links,indexed_at=excluded.indexed_at""",
                    (document["path"], document.get("digest", "restored"), document.get("title", ""), document.get("source_type", "file"), document.get("file_type", "txt"), document.get("folder", "root"), json.dumps(document.get("tags", [])), json.dumps(document.get("links", [])), document.get("indexed_at")),
                )
                self.connection.execute("DELETE FROM chunks WHERE document_path = ?", (document["path"],))
            for chunk in chunks:
                self.connection.execute(
                    "INSERT INTO chunks(document_path,title,text,ordinal,citation,vector,tags,start_line,end_line,page) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (chunk["document_path"], chunk["title"], chunk["text"], chunk["ordinal"], chunk["citation"], chunk["vector"], chunk.get("tags", "[]"), chunk.get("start_line"), chunk.get("end_line"), chunk.get("page")),
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

    @staticmethod
    def _diagnostic_from_row(row: sqlite3.Row) -> ParseDiagnostic:
        return ParseDiagnostic(
            path=row["path"],
            level=row["level"],
            code=row["code"],
            message=row["message"],
            created_at=row["created_at"],
        )

    def _chunk_rows(self, tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> list[sqlite3.Row]:
        filters, params = self._filter_sql(tag=tag, folder=folder, file_type=file_type)
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

    @staticmethod
    def _filter_sql(tag: str | None = None, folder: str | None = None, file_type: str | None = None, alias: str = "d") -> tuple[list[str], list[object]]:
        filters = []
        params: list[object] = []
        if tag:
            filters.append("c.tags LIKE ?")
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
    denominator = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
    return sum(a*b for a, b in zip(left, right)) / denominator if denominator else 0.0


def _average(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dims)]
