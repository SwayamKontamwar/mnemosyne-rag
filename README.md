# Mnemosyne

Mnemosyne is a local-first personal knowledge base for notes, PDFs, and documents. You can drop in your source material, search it instantly with hybrid retrieval, and ask grounded questions that answer directly from your own library with citations back to the original text.

## What the current version does

- Ingests `.md`, `.markdown`, `.txt`, `.pdf`, `.docx`, `.pptx`, `.csv`, `.tsv`, and `.xlsx` sources
- Tracks content hashes so re-indexing skips unchanged files
- Stores everything locally in SQLite with FTS5 plus vector search
- Keeps append-only per-note revision history with `as_of` time-travel search
- Supports direct search, cited question answering, and semantic backlinks
- Ships with a browser-based local interface for uploads, library browsing, and chat
- Keeps provider boundaries clean so local or cloud models can be swapped later
- Extracts tags, folders, and wiki-style note links for filters and graph features
- Exposes chunk preview, topic clusters, and lightweight note graph APIs

## Product shape

The direction is a private, Notion-like research companion rather than a generic chatbot. The system is meant to feel like a durable knowledge workspace: ingest anything important, retrieve exact passages fast, and then build richer layers on top such as clustering, backlinking, topic maps, and graph exploration.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Spreadsheet ingestion is included in the normal Python install. Scanned PDF OCR also needs native tools on your machine:

```bash
brew install poppler tesseract
```

Without those OCR binaries, text PDFs still index normally, but image-only scanned PDFs will be skipped with a parse diagnostic explaining what is missing. The Docker setup includes the OCR tools for you.

Initialize and index files:

```bash
mnemo init
mnemo ingest ~/Notes
mnemo search "ideas about distributed systems"
mnemo search "old wording" --as-of "2026-07-14T12:00:00Z"
mnemo ask "What have I written about retrieval evaluation?"
mnemo ask "What did I believe then?" --as-of "2026-07-14T12:00:00Z"
mnemo backlinks "meeting-notes.md"
mnemo revisions ~/Notes/meeting-notes.md
mnemo revisions ~/Notes/meeting-notes.md --diff 1 2
mnemo revisions ~/Notes/meeting-notes.md --restore 1
```

Run the local web app:

```bash
mnemo serve
```

Then open `http://127.0.0.1:8765`.

Mnemosyne now treats local Ollama embeddings as the default happy path. For the strongest local setup, run both an embed model and an answer model:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama serve
```

The default runtime assumes:

```bash
export MNEMO_EMBED_PROVIDER=ollama
export OLLAMA_EMBED_MODEL=nomic-embed-text
```

By default Mnemosyne stores data under `.mnemosyne/`. Set `MNEMO_HOME` to move the local database, uploads, and index.

## Product surface

The app now includes:

- Drag-and-drop plus watch-folder ingestion
- Tags, folders, file-type filters, and collections groundwork
- Search history and saved searches
- Source-reader views with chunk previews, entities, and simple timelines
- Citation auditing and an evaluation dashboard
- Provider and privacy preferences stored locally
- Backup/export APIs for local data portability

## One-command local run

You can run the complete local stack in Docker. This starts Mnemosyne, persistent Chroma, Ollama, OCR tooling, and a one-time model downloader:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8765`.

The first launch downloads `nomic-embed-text` and `qwen2.5:3b` (roughly 2.2 GB total), so model-backed chat becomes ready after those downloads finish. The header reports `Ollama ready` when both models are available. Set `MNEMO_PORT=9000` before the command if port 8765 is occupied.

## Architecture

```text
documents -> parser -> chunker -> embedder -> SQLite chunks + FTS + vectors
                                                  |
query -----> embedder -> hybrid retriever --------+
                              |
                         cited context -> generator -> grounded answer
```

Current defaults favor simplicity with a path to a much richer local product:

- Storage: local SQLite for metadata and FTS5, with SQLite vectors by default or persistent Chroma via `MNEMO_VECTOR_PROVIDER=chroma`
- Embeddings: Ollama `nomic-embed-text` by default, with an explicit deterministic hash fallback when Ollama is offline
- Generation: Ollama for private local answering
- UI: FastAPI plus static HTML/CSS/JS for a fast local dashboard
- Metadata: folders, tags, wiki-links, and source types for collections and filters

The UI reports whether real Ollama embeddings are active, whether fallback search is being used, and whether the configured answer model is ready.

## Versioned time travel

Mnemosyne does not overwrite indexed knowledge in place. Every changed file creates a new immutable document revision with a monotonic version number and timestamp. Chunks have `valid_from` and `valid_to` windows, deletes become tombstone revisions, and unchanged chunk text is matched by content hash so its existing vector is reused instead of embedded again.

Search and ask accept an optional `as_of` timestamp. Without it, results mean “latest.” With it, FTS, SQLite vector fallback, and Chroma metadata filters restrict candidates to chunks that were live at that instant. Citations include the document path, revision, and line/page target, so a citation from an old answer resolves to the historical chunk text even after the source file changes later.

You can test it from the UI:

1. Run `mnemo serve` and upload or watch a note.
2. Search for text from revision 1 and copy the result citation.
3. Edit the note, re-index it, and search again with the top-bar `As of` timestamp set to the first revision time.
4. Open the source from the library to view revision history, line diffs, and restore buttons.
5. Restore an older revision; Mnemosyne writes a new revision with the older content instead of deleting history.

API checks:

```bash
curl -X POST http://127.0.0.1:8765/api/search \
  -H 'content-type: application/json' \
  -d '{"query":"old phrase","as_of":"2026-07-14T12:00:00Z"}'

curl 'http://127.0.0.1:8765/api/revisions?path=/absolute/path/to/note.md'
curl 'http://127.0.0.1:8765/api/revisions/diff?path=/absolute/path/to/note.md&left=1&right=2'
curl -X POST http://127.0.0.1:8765/api/revisions/restore \
  -H 'content-type: application/json' \
  -d '{"path":"/absolute/path/to/note.md","version":1}'
curl --get http://127.0.0.1:8765/api/citations/resolve \
  --data-urlencode 'citation=/absolute/path/to/note.md?rev=1#L1-L3'
```

## Completed product layer

1. Strong local Ollama embeddings with observable fallback state
2. Citations that deep-link into exact chunk and page previews
3. Folders, tags, source filters, and collections
4. Semantic connections, clustering, related notes, and graph exploration
5. Continuous watch folders plus safe Notion and Obsidian ZIP imports

## Verification

```bash
pip install -e '.[dev,full]'
pytest -q
docker compose build mnemosyne
```

The suite covers incremental ingestion, time-travel search, tombstone deletes, historical citation resolution, rollback-as-new-revision, chunk-vector reuse by content hash, FTS/vector retrieval, Ollama's real HTTP protocol, Chroma persistence, citation repair/validation, watch-folder updates and deletions, safe ZIP imports, backup restoration, structured Office parsing, real spreadsheet upload/search, OCR dependency diagnostics, and the web upload/search/preview flow. The production image includes Poppler and Tesseract; scanned PDFs are OCRed per page and retain page citations.

## Product milestones

### Milestone 1 — Reliable local library

- Drag-and-drop interface
- Markdown, TXT and PDF ingestion
- Document library
- SQLite metadata
- FTS5 keyword search
- Incremental indexing
- Append-only revision history and rollback
- Exact line and page citations

### Milestone 2 — Real semantic RAG

- Ollama embeddings
- Chroma adapter
- Hybrid retrieval
- Local reranking
- Grounded Ollama answers
- Citation validation
- Search and chat interfaces

### Milestone 3 — Broad document support

- DOCX, PowerPoint and spreadsheets
- OCR for scanned PDFs
- Structure-aware chunking
- Folder imports
- File-system watcher
- Better parsing diagnostics

### Milestone 4 — Knowledge intelligence

- Semantic backlinks
- Entity extraction
- Topic clustering
- Knowledge graph
- Related-note recommendations
- Cross-document comparison
- Timeline and contradiction detection

### Milestone 5 — Polished personal product

- Collections and tags
- Saved searches
- Conversation history
- Provider settings
- Privacy controls
- Import/export and backups
- Evaluation dashboard
- Docker and one-command installation
