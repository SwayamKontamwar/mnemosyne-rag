from __future__ import annotations

import json
import multiprocessing
import os
import socket
import sqlite3
import tempfile
import argparse
from pathlib import Path

from mnemosyne.config import Settings
from mnemosyne.providers import ChromaVectorAdapter, HashingEmbedder
from mnemosyne.service import KnowledgeBase


class FixedEmbedder:
    model_identity = "test:model-b"
    dimensions = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[2.0 + (index + 1) / 10_000 for index in range(768)] for _ in texts]


def worker(home: str, db_path: str, note: str, port: int) -> None:
    def delete_then_pause(self, ids, vectors, metadata, embedding_space, dimensions):
        self.client.delete_collection(self.collection.name)
        with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
            connection.sendall(f"{os.getpid()} after_embedding_chroma_delete".encode())
            connection.recv(1)

    ChromaVectorAdapter.rebuild = delete_then_pause
    KnowledgeBase(
        Settings(Path(home), Path(db_path), embed_provider="test", embed_model="model-b", vector_provider="chroma"),
        embedder=FixedEmbedder(),
    ).ingest(Path(note))


def main(output: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        home = root / "data"
        db_path = home / "knowledge.db"
        note = root / "note.md"
        note.write_text("Marine navigation relies on instruments that measure position without visible landmarks.")
        initial = KnowledgeBase(
            Settings(home, db_path, embed_provider="hash", embed_model="offline-hash", vector_provider="chroma"),
            embedder=HashingEmbedder(384),
        )
        initial.ingest(note)
        initial.store.connection.close()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            process = multiprocessing.get_context("spawn").Process(
                target=worker, args=(str(home), str(db_path), str(note), listener.getsockname()[1])
            )
            process.start()
            connection, _ = listener.accept()
            with connection:
                signal = connection.recv(200).decode()
                process.kill()
            process.join(30)

        restarted = KnowledgeBase(
            Settings(home, db_path, embed_provider="test", embed_model="model-b", vector_provider="chroma"),
            embedder=FixedEmbedder(),
        )
        query = "celestial orchard"
        chroma_count_after_kill = restarted.vector_store.collection.count()
        before_recovery = restarted.search(query)
        ingest_result = restarted.ingest(note)
        chroma_count_after_recovery = restarted.vector_store.collection.count()
        after_recovery = restarted.search(query)
        sqlite_count = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        result = {
            "source_version": "run this script with PYTHONPATH pointing at mnemosyne-rag commit 9acc47f",
            "process": {"signal": signal, "exit_code": process.exitcode},
            "after_kill": {"sqlite_count": sqlite_count, "chroma_count": chroma_count_after_kill},
            "semantic_query": query,
            "semantic_hits_before_recovery": len(before_recovery),
            "documented_recovery_ingest_result": list(ingest_result),
            "chroma_count_after_recovery": chroma_count_after_recovery,
            "semantic_hits_after_recovery": len(after_recovery),
        }
        assert process.exitcode is not None and process.exitcode < 0
        assert chroma_count_after_kill == 0
        assert len(before_recovery) == 0
        assert ingest_result == (0, 1)
        assert chroma_count_after_recovery == 0
        assert len(after_recovery) == 0
        rendered = json.dumps(result, indent=2)
        if output:
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    main(parser.parse_args().output)
