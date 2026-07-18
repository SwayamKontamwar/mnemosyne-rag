from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mnemosyne.providers as providers
from mnemosyne.config import Settings
from mnemosyne.providers import HashingEmbedder, OllamaEmbedder
from mnemosyne.service import KnowledgeBase


class FakeResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.payload = json.dumps({"embeddings": vectors}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload


def fake_ollama(request, timeout: int = 120):
    payload = json.loads(request.data or b"{}")
    vectors = [
        [2.0 + (index + 1) / 10_000 for index in range(768)]
        for _ in payload["input"]
    ]
    return FakeResponse(vectors)


def stored_row(kb: KnowledgeBase) -> dict:
    row = kb.store.connection.execute(
        """SELECT id, vector, embedding_space, embedding_dimensions
           FROM chunks WHERE valid_to IS NULL"""
    ).fetchone()
    return {
        "id": int(row["id"]),
        "embedding_space": row["embedding_space"],
        "embedding_dimensions": int(row["embedding_dimensions"]),
        "vector": json.loads(row["vector"]),
    }


def main() -> None:
    original_urlopen = providers.urllib.request.urlopen
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            note = root / "unchanged.md"
            note.write_text("An unchanged note must move into the new embedding space.")
            first_settings = Settings(
                root / "data", root / "data" / "knowledge.db",
                embed_provider="hash", embed_model="offline-hash", vector_provider="sqlite",
            )
            first = KnowledgeBase(first_settings, embedder=HashingEmbedder(384))
            first.ingest(note)
            before = stored_row(first)

            providers.urllib.request.urlopen = fake_ollama
            second_settings = Settings(
                first_settings.home, first_settings.db_path, ollama_url="http://ollama.test",
                embed_provider="ollama", embed_model="nomic-embed-text", vector_provider="sqlite",
            )
            second = KnowledgeBase(
                second_settings,
                embedder=OllamaEmbedder("http://ollama.test", "nomic-embed-text", dimensions=768),
            )
            ingest_result = second.ingest(note)
            after = stored_row(second)
            all_vectors = [
                json.loads(row["vector"])
                for row in second.store.connection.execute("SELECT vector FROM chunks ORDER BY id")
            ]
            query_vector = second.embedder.embed(["new model query"])[0]
            evidence = {
                "ingest_after_unchanged_file": {"indexed": ingest_result[0], "skipped": ingest_result[1]},
                "before": before,
                "after": after,
                "proof": {
                    "same_chunk_id": before["id"] == after["id"],
                    "vectors_differ": before["vector"] != after["vector"],
                    "old_vector_exists_anywhere": before["vector"] in all_vectors,
                    "query_vector_length": len(query_vector),
                    "stored_vector_lengths": [len(vector) for vector in all_vectors],
                    "active_embedding_spaces": sorted(second.store.embedding_spaces()),
                },
            }
            print(json.dumps(evidence, indent=2))
            first.store.connection.close()
            second.store.connection.close()
    finally:
        providers.urllib.request.urlopen = original_urlopen


if __name__ == "__main__":
    main()
