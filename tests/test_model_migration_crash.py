from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import os
import socket
import sqlite3
from pathlib import Path

import pytest

from mnemosyne.config import Settings
from mnemosyne.providers import HashingEmbedder
from mnemosyne.service import KnowledgeBase


class FixedEmbedder:
    def __init__(self, identity: str, dimensions: int, offset: float) -> None:
        self.model_identity = identity
        self.dimensions = dimensions
        self.offset = offset

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[self.offset + (index + 1) / 10_000 for index in range(self.dimensions)] for _ in texts]


def _worker(home: str, db_path: str, note: str, port: int) -> None:
    def checkpoint(name: str) -> None:
        if name != "during_embedding_chroma_rebuild":
            return
        with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
            connection.sendall(f"{os.getpid()} {name}".encode())
            connection.recv(1)

    settings = Settings(
        Path(home), Path(db_path), embed_provider="test", embed_model="model-b",
        vector_provider="chroma", chunk_size=42, chunk_overlap=0,
    )
    KnowledgeBase(
        settings, embedder=FixedEmbedder("test:model-b", 768, 2.0),
        crash_hook=checkpoint, chroma_rebuild_batch_size=1,
    ).ingest(Path(note))


def _float32_hash(vector) -> str:
    import struct
    return hashlib.sha256(b"".join(struct.pack("<f", float(value)) for value in vector)).hexdigest()


def _snapshot(home: Path, db_path: Path) -> dict:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    sqlite_rows = [
        {
            "id": int(row["id"]),
            "embedding_space": row["embedding_space"],
            "dimensions_metadata": int(row["embedding_dimensions"]),
            "dimensions_actual": len(json.loads(row["vector"])),
            "vector_sha256": _float32_hash(json.loads(row["vector"])),
        }
        for row in connection.execute(
            "SELECT id, embedding_space, embedding_dimensions, vector FROM chunks ORDER BY id"
        )
    ]
    journal = [dict(row) for row in connection.execute("SELECT * FROM embedding_migrations ORDER BY created_at, id")]
    connection.close()

    import chromadb
    client = chromadb.PersistentClient(path=str(home / "chroma"))
    collection = client.get_or_create_collection("mnemosyne", metadata={"hnsw:space": "cosine"})
    result = collection.get(include=["embeddings", "metadatas"])
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []
    chroma_rows = [
        {
            "id": str(item_id),
            "embedding_space": metadata.get("embedding_space"),
            "dimensions_metadata": metadata.get("embedding_dimensions"),
            "dimensions_actual": len(vector),
            "vector_sha256": _float32_hash(vector),
        }
        for item_id, vector, metadata in zip(result.get("ids") or [], embeddings, result.get("metadatas") or [])
    ]
    del collection, client
    gc.collect()
    files = {
        path.relative_to(home).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(home.rglob("*")) if path.is_file()
    }
    pending = [row for row in journal if row["status"] != "completed"]
    return {
        "sqlite": sqlite_rows,
        "chroma": sorted(chroma_rows, key=lambda row: row["id"]),
        "journal": journal,
        "state": {
            "pending_migrations": [row["id"] for row in pending],
            "sqlite_spaces": sorted({(row["embedding_space"], row["dimensions_actual"]) for row in sqlite_rows}),
            "chroma_spaces": sorted({(row["embedding_space"], row["dimensions_actual"]) for row in chroma_rows}),
        },
        "filesystem": files,
    }


def _canonical(snapshot: dict) -> bytes:
    logical = {key: snapshot[key] for key in ("sqlite", "chroma", "journal", "state")}
    return json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()


def _changes(before: dict, after: dict) -> dict:
    left, right = before["filesystem"], after["filesystem"]
    return {
        "added": sorted(set(right) - set(left)),
        "removed": sorted(set(left) - set(right)),
        "modified": sorted(path for path in set(left) & set(right) if left[path] != right[path]),
    }


def run_crash_evidence(root: Path) -> dict:
    home = root / "data"
    db_path = home / "knowledge.db"
    note = root / "unchanged.md"
    note.write_text(
        "First independent chunk exists to make a partial Chroma rebuild observable.\n\n"
        "Second independent chunk must not silently disappear after a process kill."
    )
    initial_settings = Settings(
        home, db_path, embed_provider="hash", embed_model="offline-hash",
        vector_provider="chroma", chunk_size=42, chunk_overlap=0,
    )
    initial = KnowledgeBase(initial_settings, embedder=HashingEmbedder(384))
    initial.ingest(note)
    initial.store.connection.close()
    del initial
    gc.collect()
    before = _snapshot(home, db_path)
    assert len(before["sqlite"]) >= 2

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(30)
        process = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(str(home), str(db_path), str(note), listener.getsockname()[1])
        )
        process.start()
        connection, _ = listener.accept()
        with connection:
            signal = connection.recv(200).decode()
            assert signal == f"{process.pid} during_embedding_chroma_rebuild"
            assert process.is_alive()
            process.kill()
        process.join(30)
    assert not process.is_alive()
    assert process.exitcode is not None and process.exitcode < 0

    after_kill = _snapshot(home, db_path)
    assert after_kill["state"]["pending_migrations"]
    assert after_kill["state"]["sqlite_spaces"] == [("test:model-b", 768)]
    assert 0 < len(after_kill["chroma"]) < len(after_kill["sqlite"])

    blocked = KnowledgeBase(
        Settings(home, db_path, embed_provider="test", embed_model="model-b", vector_provider="chroma"),
        embedder=FixedEmbedder("test:model-b", 768, 2.0),
    )
    with pytest.raises(RuntimeError, match="incomplete; re-run ingest"):
        blocked.search("independent chunk")
    blocked.store.connection.close()
    del blocked
    gc.collect()

    recovery_one_engine = KnowledgeBase(
        Settings(home, db_path, embed_provider="test", embed_model="model-b", vector_provider="chroma"),
        embedder=FixedEmbedder("test:model-b", 768, 2.0),
    )
    recovery_one_engine.ingest(note)
    assert recovery_one_engine.search("independent chunk")
    recovery_one_engine.store.connection.close()
    del recovery_one_engine
    gc.collect()
    recovery_one = _snapshot(home, db_path)

    recovery_two_engine = KnowledgeBase(
        Settings(home, db_path, embed_provider="test", embed_model="model-b", vector_provider="chroma"),
        embedder=FixedEmbedder("test:model-b", 768, 2.0),
    )
    recovery_two_engine.ingest(note)
    assert recovery_two_engine.search("independent chunk")
    recovery_two_engine.store.connection.close()
    del recovery_two_engine
    gc.collect()
    recovery_two = _snapshot(home, db_path)

    assert not recovery_one["state"]["pending_migrations"]
    assert recovery_one["state"]["sqlite_spaces"] == [("test:model-b", 768)]
    assert recovery_one["state"]["chroma_spaces"] == [("test:model-b", 768)]
    assert {row["id"] for row in recovery_one["sqlite"]} == {int(row["id"]) for row in recovery_one["chroma"]}
    assert _canonical(recovery_one) == _canonical(recovery_two)

    return {
        "process": {"signal": signal, "exit_code": process.exitcode, "killed": process.exitcode < 0},
        "stages": {
            "before": before,
            "directly_after_kill": after_kill,
            "after_recovery_one": recovery_one,
            "after_recovery_two": recovery_two,
        },
        "changes": {
            "before_to_kill": _changes(before, after_kill),
            "recovery_one": _changes(after_kill, recovery_one),
            "recovery_two": _changes(recovery_one, recovery_two),
        },
        "proof": {
            "search_refused_while_pending": True,
            "logical_recovery_two_byte_equal": _canonical(recovery_one) == _canonical(recovery_two),
            "sqlite_ids": [row["id"] for row in recovery_two["sqlite"]],
            "chroma_ids": [row["id"] for row in recovery_two["chroma"]],
        },
    }


def test_real_parent_kill_during_partial_chroma_rebuild_recovers_twice(tmp_path: Path):
    run_crash_evidence(tmp_path)


if __name__ == "__main__":
    import argparse
    import tempfile
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    output = parser.parse_args().output
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        rendered = json.dumps(run_crash_evidence(Path(directory)), indent=2) + "\n"
        if output:
            output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
