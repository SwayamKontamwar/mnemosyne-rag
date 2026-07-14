# Mnemosyne

Mnemosyne is a local-first personal knowledge base for notes, PDFs, and documents. You can drop in your source material, search it instantly with hybrid retrieval, and ask grounded questions that answer directly from your own library with citations back to the original text.

## What the current version does

- Ingests `.md`, `.markdown`, `.txt`, `.pdf`, `.docx`, `.pptx`, `.csv`, `.tsv`, and `.xlsx` sources
- Tracks content hashes so re-indexing skips unchanged files
- Stores everything locally in SQLite with FTS5 plus vector search
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

Initialize and index files:

```bash
mnemo init
mnemo ingest ~/Notes
mnemo search "ideas about distributed systems"
mnemo ask "What have I written about retrieval evaluation?"
mnemo backlinks "meeting-notes.md"
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

You can run the app in Docker:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8765`.

## Architecture

```text
documents -> parser -> chunker -> embedder -> SQLite chunks + FTS + vectors
                                                  |
query -----> embedder -> hybrid retriever --------+
                              |
                         cited context -> generator -> grounded answer
```

Current defaults favor simplicity with a path to a much richer local product:

- Storage: local SQLite for metadata, chunks, keyword search, and vectors
- Embeddings: hashing baseline today, optional Ollama embeddings via `MNEMO_EMBED_PROVIDER=ollama`
- Generation: Ollama for private local answering
- UI: FastAPI plus static HTML/CSS/JS for a fast local dashboard
- Metadata: folders, tags, wiki-links, and source types for collections and filters

The hashing embedder is intentionally still the safe default for zero-setup development, but the system now has a real local upgrade path through Ollama embeddings.

## Near-term roadmap

1. Upgrade embeddings to a stronger local model
2. Add citations that deep-link into chunk and page previews
3. Introduce folders, tags, and source filters
4. Add semantic clustering and note graph exploration
5. Support watch folders plus Notion and Obsidian import flows

## Product milestones

### Milestone 1 — Reliable local library

- Drag-and-drop interface
- Markdown, TXT and PDF ingestion
- Document library
- SQLite metadata
- FTS5 keyword search
- Incremental indexing
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
