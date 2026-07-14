from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .models import Chunk, ChunkPreview, DocumentRecord, GraphEdge, SearchHit, TopicCluster


class KnowledgeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row

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

    def replace_document(self, path: str, digest: str, chunks: Iterable[tuple[Chunk, list[float]]], metadata: dict) -> None:
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
        return {"documents": documents, "chunks": chunks, "characters": characters, "tags": tags}

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

    def _chunk_rows(self, tag: str | None = None, folder: str | None = None, file_type: str | None = None) -> list[sqlite3.Row]:
        filters = []
        params: list[object] = []
        if tag:
            filters.append("c.tags LIKE ?")
            params.append(f'%"{tag.lower()}"%')
        if folder:
            filters.append("d.folder = ?")
            params.append(folder)
        if file_type:
            filters.append("d.file_type = ?")
            params.append(file_type)
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
