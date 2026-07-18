# Embedding model migration

Mnemosyne stores an embedding-space identity and vector dimension on every chunk. The identity is the provider and model (for example, `ollama:nomic-embed-text`) or the versioned hash algorithm (`hash:blake2b-v1`).

When `mnemo ingest` sees stored chunks from another embedding space, it re-embeds every stored chunk, including historical revisions, before applying normal file-digest skipping. SQLite vectors are replaced in one transaction and Chroma is rebuilt with the same chunk IDs. Unchanged files can therefore report as skipped after their vectors have been migrated; the skip means parsing was unnecessary, not that the old vector was reused.

Search validates the requested model identity, query dimension, stored dimension metadata, actual stored vector length, and the complete Chroma ID/vector set before retrieval. Empty, partial, stale, or mismatched Chroma raises an error instructing the user to run ingest; keyword results are not returned as though vector retrieval succeeded. Ingest performs the same audit before the unchanged-file shortcut and rebuilds Chroma from SQLite when needed.

Content-hash reuse remains enabled only inside the same embedding space and dimension.

## Direct evidence

Run:

```powershell
.\.venv\Scripts\python.exe -m tests.model_swap_evidence
```

The command prints the complete vector read directly from SQLite before migration, the complete replacement vector read afterward, and proof fields for row identity, old-vector absence, query length, every stored vector length, and active embedding spaces. **This command uses a deterministic stub at the Ollama HTTP boundary. Its `2.0001, 2.0002, ...` vector is not a real `nomic-embed-text` embedding. It proves migration plumbing only.**

This development machine does not have Ollama installed, so this release has not produced real `nomic-embed-text` weight evidence. Do not treat the committed plumbing JSON as proof of the real model's semantics or normalization. A real-model run requires an installed Ollama service with `nomic-embed-text` pulled and must additionally verify that `OllamaEmbedder.last_backend == "ollama"` so hash fallback cannot masquerade as model evidence.

The committed `model-swap-evidence.json` and `model-migration-crash-evidence.json` files are UTF-8 without a BOM.

## SQLite to Chroma crash recovery

Embedding migration writes the new SQLite vectors and a `sqlite_committed` journal row in one SQLite transaction. Search refuses while any migration journal row is pending. Chroma is rebuilt in batches, verified against SQLite by ID, dimension, metadata, and vector values, and only then is the journal marked `completed`.

Run the real parent-kill evidence:

```powershell
.\.venv\Scripts\python.exe -m tests.test_model_migration_crash
```

The worker signals after SQLite has committed and immediately after deleting the Chroma collection, before recreating it or adding any replacement vector. The parent calls the OS process-kill API, requires a negative exit code, snapshots both stores and the full journal, confirms a semantic-only query refuses instead of quietly returning keyword-only results, recovers twice, and compares the complete canonical logical snapshots byte for byte. It also reports every filesystem path added, removed, or modified between stages.

The suite also deletes and recreates an empty Chroma collection without leaving a pending journal, which reproduces the independent empty-index case. Search must refuse. Re-ingesting an unchanged source must rebuild the index before returning the normal `(0 indexed, 1 skipped)` file result, after which the no-shared-word semantic query must return a hit.

## What the focused tests assert

`test_unchanged_file_hash_to_ollama_swap_replaces_actual_sqlite_vector_384_to_768`:

- Reads the active row directly from SQLite after hash ingestion and asserts the vector has 384 values, identity is `hash:blake2b-v1`, and dimension metadata is 384.
- Reopens the same database with the production `OllamaEmbedder` configured as `nomic-embed-text`, does not touch the source file, and ingests again.
- Reads the row directly again and asserts its ID is unchanged, its complete vector differs, its length and metadata are 768, and its identity is `ollama:nomic-embed-text`.
- Reads every vector in the chunks table and asserts the old complete vector occurs zero times and every stored vector is 768-wide.
- Embeds a query through the new embedder, asserts it is 768-wide, asserts every stored vector has exactly that length, and executes search.
- It does not prove the semantic quality, normalization, or real output values of `nomic-embed-text`: the clean test fakes Ollama's local HTTP response deterministically so no model download is required. It exercises the production Ollama client, migration, SQLite storage, and search path only.

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

- SQLite and Chroma do not share one transaction. SQLite is the recovery source of truth. A durable pending journal prevents search during known migrations. A full pre-search consistency audit also catches empty, old, partial, or externally damaged Chroma state when no pending journal exists. The next ingest rebuilds and verifies Chroma from SQLite before applying file-digest skipping.
- The full ID/vector audit currently reads every active Chroma vector before each search. This favors correctness over large-library query latency and will need a cryptographically bound generation manifest before it can be made constant-time without weakening the check.
- A model serving different dimensions under the same unchanged model name is detected when a query or new embedding is produced. Mnemosyne cannot detect changed model weights that retain both the same configured name and dimensions; use a distinct model/version name when changing weights.
- Existing databases without embedding metadata are treated as legacy/mismatched and fully re-embedded on the next ingest.
