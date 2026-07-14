from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import Chunk, SearchHit


class KnowledgeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY, digest TEXT NOT NULL, indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_path TEXT NOT NULL, title TEXT NOT NULL,
                text TEXT NOT NULL, ordinal INTEGER NOT NULL, citation TEXT NOT NULL,
                vector TEXT NOT NULL, FOREIGN KEY(document_path) REFERENCES documents(path)
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
        self.connection.commit()

    def digest_for(self, path: str) -> str | None:
        row = self.connection.execute("SELECT digest FROM documents WHERE path = ?", (path,)).fetchone()
        return row["digest"] if row else None

    def replace_document(self, path: str, digest: str, chunks: Iterable[tuple[Chunk, list[float]]]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM chunks WHERE document_path = ?", (path,))
            self.connection.execute(
                "INSERT INTO documents(path, digest) VALUES (?, ?) "
                "ON CONFLICT(path) DO UPDATE SET digest=excluded.digest, indexed_at=CURRENT_TIMESTAMP",
                (path, digest),
            )
            self.connection.executemany(
                "INSERT INTO chunks(document_path,title,text,ordinal,citation,vector) VALUES(?,?,?,?,?,?)",
                ((c.document_path, c.title, c.text, c.ordinal, c.citation, json.dumps(v)) for c, v in chunks),
            )

    def hybrid_search(self, query: str, query_vector: list[float], limit: int = 8) -> list[SearchHit]:
        rows = self.connection.execute("SELECT * FROM chunks").fetchall()
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
            SearchHit(row["id"], row["text"], row["title"], row["citation"], 0.7 * vector_scores[row["id"]] + 0.3 * keyword_scores.get(row["id"], 0))
            for row in ranked[:limit]
        ]

    def chunks_for_document(self, name: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM chunks WHERE document_path LIKE ? OR title = ?", (f"%{name}%", name)
        ).fetchall()


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
    return sum(a*b for a, b in zip(left, right)) / denominator if denominator else 0.0

