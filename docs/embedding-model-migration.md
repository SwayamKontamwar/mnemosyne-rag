# Embedding model migration

Mnemosyne stores an embedding-space identity and vector dimension on every chunk. The identity is the provider and model (for example, `ollama:nomic-embed-text`) or the versioned hash algorithm (`hash:blake2b-v1`).

When `mnemo ingest` sees stored chunks from another embedding space, it re-embeds every stored chunk, including historical revisions, before applying normal file-digest skipping. SQLite vectors are replaced in one transaction and Chroma is rebuilt with the same chunk IDs. Unchanged files can therefore report as skipped after their vectors have been migrated; the skip means parsing was unnecessary, not that the old vector was reused.

Search validates the requested model identity, query dimension, stored dimension metadata, and actual stored vector length before cosine scoring. Chroma validates its collection identity and dimension before querying. A mismatch raises an error instructing the user to run ingest; no cross-space score is returned.

Content-hash reuse remains enabled only inside the same embedding space and dimension.

## Direct evidence

Run:

```powershell
.\.venv\Scripts\python.exe -m tests.model_swap_evidence
```

The command prints the complete vector read directly from SQLite before migration, the complete replacement vector read afterward, and proof fields for row identity, old-vector absence, query length, every stored vector length, and active embedding spaces.

## What the focused tests assert

`test_unchanged_file_hash_to_ollama_swap_replaces_actual_sqlite_vector_384_to_768`:

- Reads the active row directly from SQLite after hash ingestion and asserts the vector has 384 values, identity is `hash:blake2b-v1`, and dimension metadata is 384.
- Reopens the same database with the production `OllamaEmbedder` configured as `nomic-embed-text`, does not touch the source file, and ingests again.
- Reads the row directly again and asserts its ID is unchanged, its complete vector differs, its length and metadata are 768, and its identity is `ollama:nomic-embed-text`.
- Reads every vector in the chunks table and asserts the old complete vector occurs zero times and every stored vector is 768-wide.
- Embeds a query through the new embedder, asserts it is 768-wide, asserts every stored vector has exactly that length, and executes search.
- It does not prove the semantic quality of the real `nomic-embed-text` weights: the clean test fakes Ollama's local HTTP response deterministically so no model download is required. It does exercise the production Ollama client, migration, SQLite storage, and search path.

`test_chroma_model_swap_rebuilds_ids_and_removes_old_embeddings`:

- Pulls IDs and embeddings directly from Chroma before and after a 384-to-768 model swap.
- Asserts the collection still contains exactly one row with the same ID, the complete embedding changed, all new metadata identifies model B/768, and the old complete embedding is absent.
- It proves replacement rather than append-only shadowing. It does not simulate a process crash during Chroma rebuilding.

`test_search_refuses_mixed_model_or_dimension_before_cosine`:

- Corrupts a real SQLite row to model A/384 while the query embedder remains model B/768.
- Replaces the cosine function with an assertion that fails if called.
- Asserts search raises a mixed-space error containing the query identity, then reads the row back to prove the mismatch was real.
- It proves this SQLite search path refuses before scoring. Chroma separately enforces collection identity and dimensions; the test does not corrupt Chroma's internal index files byte by byte.

## Limits

- Migration is transactional in SQLite, but SQLite and Chroma do not share one cross-database transaction. If Chroma rebuilding fails after SQLite commits, Chroma identity remains stale and searches in Chroma mode refuse until ingest successfully rebuilds it.
- A model serving different dimensions under the same unchanged model name is detected when a query or new embedding is produced. Mnemosyne cannot detect changed model weights that retain both the same configured name and dimensions; use a distinct model/version name when changing weights.
- Existing databases without embedding metadata are treated as legacy/mismatched and fully re-embedded on the next ingest.
