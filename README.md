# Mnemosyne

Mnemosyne is a local-first, versioned personal knowledge base for your notes, documents, PDFs, spreadsheets, and exports. It lets you drop files into a private library, search across them with hybrid retrieval, ask questions with grounded citations, explore related notes, and time-travel through old versions of your knowledge.

Think of it as a Notion-like research workspace crossed with a local RAG engine: fast search, cited answers, document history, and knowledge-graph exploration without sending your library to a hosted app by default.

## What it does

- Ingests Markdown, text, PDFs, Word docs, PowerPoint decks, CSV/TSV files, XLSX spreadsheets, and ZIP exports
- Extracts searchable chunks with exact line or page citations
- Stores metadata locally in SQLite with FTS5 keyword search
- Supports vector retrieval through SQLite fallback or persistent Chroma
- Uses Ollama embeddings and local Ollama answering, with deterministic hash fallback for tests/offline mode
- Combines keyword, vector, query expansion, HyDE, reciprocal rank fusion, reranking, and MMR diversity
- Answers questions from retrieved sources and validates citations
- Tracks folders, tags, file types, links, saved searches, history, collections, entities, timelines, and related notes
- Keeps append-only document revisions with `as_of` time-travel search
- Reuses unchanged chunk vectors by content hash only when embedding model identity and dimensions match
- Exposes a local web UI, CLI, backup/export APIs, watch folders, and Docker setup

## Why this exists

Most personal note systems are good at storing information but weak at answering “what do I know about this?” Most chat-with-your-docs demos are good at answering one-off questions but weak at being a durable knowledge workspace.

Mnemosyne is built around the middle ground:

- You own the files.
- The index is local.
- Answers cite the exact source chunks.
- Old answers can still resolve to the historical revision they used.
- Search works even when you phrase a query differently from how the note was written.
- The system is modular enough to swap embedding, vector, storage, and generation providers later.

## Example use cases

### Personal second brain

Drop in notes, meeting logs, PDFs, and research snippets. Search semantically, ask for summaries, and jump back to exact cited passages.

Example:

```bash
mnemo ingest ~/Notes
mnemo ask "What have I written about retrieval evaluation?"
```

### Research assistant

Index papers, slide decks, and project notes. Ask cross-document questions and compare sources while keeping citations visible.

Example:

```bash
mnemo search "citation validation reranking experiments"
mnemo ask "Compare my notes on keyword search vs vector search."
```

### Versioned knowledge archive

Track how your notes changed over time. Query the library as it existed last week, restore an old version without deleting history, or inspect line-level diffs.

Example:

```bash
mnemo search "old project decision" --as-of "2026-07-14T12:00:00Z"
mnemo revisions ~/Notes/project.md
mnemo revisions ~/Notes/project.md --diff 1 3
mnemo revisions ~/Notes/project.md --restore 1
```

### Local document search

Use it as a private desktop search layer for messy folders of PDFs, spreadsheets, Word docs, and exports.

Example:

```bash
mnemo ingest ~/Documents/Research
mnemo search "spreadsheet retrieval sentinel"
```

### Obsidian or Notion companion

Watch an Obsidian vault or import a Notion ZIP export, then use semantic search, backlinks, tags, graph connections, and cited Q&A over the imported library.

Example:

```bash
mnemo watch ~/ObsidianVault --profile obsidian
mnemo import ~/Downloads/notion-export.zip --profile notion
```

## Walkthrough

### 1. Install

Requires Python 3.11+.

Windows (verified with Python 3.12):

```powershell
python -m venv .venv-clean
.\.venv-clean\Scripts\python.exe -m pip install --upgrade pip
.\.venv-clean\Scripts\python.exe -m pip install -e ".[dev,full]"
.\.venv-clean\Scripts\python.exe -m pytest -q
.\.venv-clean\Scripts\python.exe -m tests.model_swap_evidence
.\.venv-clean\Scripts\python.exe -m tests.test_model_migration_crash
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and full parser/test coverage:

```bash
pip install -e '.[dev,full]'
```

Spreadsheet ingestion is included in the normal install. Scanned PDF OCR also needs Poppler and Tesseract installed on your machine:

```bash
brew install poppler tesseract
```

Without those OCR binaries, text PDFs still index normally. Image-only scanned PDFs are skipped with a parse diagnostic explaining which OCR dependency is missing.

### 2. Start local models

Mnemosyne is designed around local Ollama models.

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama serve
```

Default model settings:

```bash
export MNEMO_EMBED_PROVIDER=ollama
export OLLAMA_EMBED_MODEL=nomic-embed-text
```

For a deterministic offline/test mode:

```bash
export MNEMO_EMBED_PROVIDER=hash
export MNEMO_VECTOR_PROVIDER=sqlite
```

### 3. Initialize and ingest

```bash
mnemo init
mnemo ingest ~/Notes
```

You can ingest a single file, a whole folder, or a ZIP export.

Supported source types:

- `.md`, `.markdown`, `.txt`
- `.pdf`
- `.docx`
- `.pptx`
- `.csv`, `.tsv`, `.xlsx`
- `.zip` imports for Notion/Obsidian-style exports

### 4. Search

```bash
mnemo search "ideas about distributed systems"
```

Search combines keyword and semantic retrieval. It can expand queries, use HyDE-style hypothetical answer matching, fuse keyword/vector rankings, rerank candidates, and diversify near-duplicate chunks.

### 5. Ask with citations

```bash
mnemo ask "What have I written about retrieval evaluation?"
```

Answers are built from retrieved chunks and cite source numbers. The citation validator checks whether the answer actually references available sources and can repair or safely ground weak answers.

### 6. Open the web app

```bash
mnemo serve
```

Then open:

```text
http://127.0.0.1:8765
```

The UI includes:

- drag-and-drop uploads
- document library
- search and ask panels
- source reader
- chunk previews
- tags, folders, file-type filters
- saved searches
- conversation history
- collections
- graph and cluster views
- revision history, diffs, and restore controls
- provider/privacy settings
- backup export

### 7. Watch folders

```bash
mnemo watch ~/Notes --profile local
mnemo watch --scan
```

Watch folders support incremental reindexing. Removed files become tombstones so older `as_of` queries can still retrieve their historical chunks.

## Versioned time travel

Mnemosyne does not overwrite indexed knowledge in place. Every changed file creates a new immutable document revision with a monotonic version number and timestamp.

Each chunk stores:

- document path
- revision id
- document version
- content hash
- `valid_from`
- `valid_to`
- line or page citation

When a file changes, Mnemosyne diffs the new chunk set against currently active chunks by content hash. Unchanged chunk text keeps its existing vector only inside the same embedding model and dimension; changed chunks get embedded once; removed chunks receive `valid_to` instead of being hard-deleted. Changing the embedding provider or model causes all stored chunks, including historical revisions, to be re-embedded before normal file skipping. Search refuses mixed model identities or vector dimensions instead of returning a score. See [embedding model migration](docs/embedding-model-migration.md) for direct database evidence, assertion-by-assertion test coverage, and limits.

Deletes create tombstone revisions. Rollback writes a new revision whose content matches an older one, preserving the full history chain.

### Query as of an old date

```bash
mnemo search "old wording" --as-of "2026-07-14T12:00:00Z"
mnemo ask "What did I believe then?" --as-of "2026-07-14T12:00:00Z"
```

Without `--as-of`, search means “latest.” With `--as-of`, candidates are filtered to chunks whose validity window contains that timestamp.

### Inspect and restore history

```bash
mnemo revisions ~/Notes/meeting-notes.md
mnemo revisions ~/Notes/meeting-notes.md --diff 1 2
mnemo revisions ~/Notes/meeting-notes.md --restore 1
```

Historical citations resolve against the old chunk text, so an old answer’s source preview does not drift when the note changes later.

## API examples

Search as of a point in time:

```bash
curl -X POST http://127.0.0.1:8765/api/search \
  -H 'content-type: application/json' \
  -d '{"query":"old phrase","as_of":"2026-07-14T12:00:00Z"}'
```

View revision history:

```bash
curl 'http://127.0.0.1:8765/api/revisions?path=/absolute/path/to/note.md'
```

Diff two revisions:

```bash
curl 'http://127.0.0.1:8765/api/revisions/diff?path=/absolute/path/to/note.md&left=1&right=2'
```

Restore a historical revision:

```bash
curl -X POST http://127.0.0.1:8765/api/revisions/restore \
  -H 'content-type: application/json' \
  -d '{"path":"/absolute/path/to/note.md","version":1}'
```

Resolve a historical citation:

```bash
curl --get http://127.0.0.1:8765/api/citations/resolve \
  --data-urlencode 'citation=/absolute/path/to/note.md?rev=1#L1-L3'
```

## Docker

You can run the complete local stack in Docker. This starts Mnemosyne, persistent Chroma, Ollama, OCR tooling, and a one-time model downloader:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8765
```

The first launch downloads local models, so model-backed chat becomes ready after those downloads finish. Set `MNEMO_PORT=9000` before the command if port `8765` is occupied.

## Architecture

```text
files / folders / uploads
        |
        v
parser -> structure-aware documents -> chunks -> embeddings
        |                                  |
        v                                  v
SQLite metadata + revisions + FTS5     SQLite vectors or Chroma
        |                                  |
        +----------- hybrid retrieval -----+
                         |
                         v
                reranked cited context
                         |
                         v
                  local Ollama answer
```

Core components:

- `mnemosyne/ingest.py`: file discovery, parsing, OCR fallback, chunking
- `mnemosyne/store.py`: SQLite schema, FTS5, revisions, validity windows, backup/restore
- `mnemosyne/service.py`: ingestion orchestration, search, RAG, reranking, graph intelligence
- `mnemosyne/providers.py`: Ollama, Chroma, hash fallback providers
- `mnemosyne/web.py`: FastAPI app and JSON APIs
- `mnemosyne/static/`: local browser UI
- `mnemosyne/cli.py`: command-line interface

## Verification

```bash
pip install -e '.[dev,full]'
pytest -q
python -m tests.model_swap_evidence
python -m tests.test_model_migration_crash
docker compose build mnemosyne
```

The test suite covers:

- incremental ingestion
- time-travel search
- tombstone deletes
- historical citation resolution
- rollback-as-new-revision
- chunk vector reuse by content hash
- full re-embedding and Chroma rebuilding when the embedding model changes
- model-identity and vector-dimension validation before similarity scoring
- journaled refusal and repeatable recovery when a process is killed after Chroma deletion but before replacement vectors are added
- refusal on an empty or mismatched Chroma index, plus repair from SQLite before unchanged-file skipping
- FTS/vector retrieval
- Ollama HTTP protocol behavior
- Chroma persistence and indexed search
- citation repair/validation
- watch-folder updates and deletions
- safe ZIP imports
- backup restoration
- Office parsing
- real spreadsheet upload/search
- OCR dependency diagnostics
- web upload/search/preview flow

## Completed product layer

1. Strong local Ollama embeddings with observable fallback state
2. Citations that deep-link into exact chunk and page previews
3. Folders, tags, source filters, saved searches, history, and collections
4. Semantic connections, clustering, related notes, entities, and graph exploration
5. Watch folders plus safe Notion and Obsidian import flows
6. Versioned time-travel search, historical citations, diffs, and rollback

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
- Query expansion and HyDE
- Reciprocal rank fusion
- MMR diversity
- Local reranking
- Grounded Ollama answers
- Citation validation
- Search and chat interfaces

### Milestone 3 — Broad document support

- DOCX, PowerPoint and spreadsheets
- OCR diagnostics for scanned PDFs
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
