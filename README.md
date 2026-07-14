# Mnemosyne

Mnemosyne is a local-first personal knowledge base for notes, PDFs, and documents. You can drop in your source material, search it instantly with hybrid retrieval, and ask grounded questions that answer directly from your own library with citations back to the original text.

## What the current version does

- Ingests `.md`, `.markdown`, `.txt`, and `.pdf` sources
- Tracks content hashes so re-indexing skips unchanged files
- Stores everything locally in SQLite with FTS5 plus vector search
- Supports direct search, cited question answering, and semantic backlinks
- Ships with a browser-based local interface for uploads, library browsing, and chat
- Keeps provider boundaries clean so local or cloud models can be swapped later

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

`search` works without a generation model. For grounded answers, run Ollama locally:

```bash
ollama pull qwen2.5:7b
ollama serve
```

By default Mnemosyne stores data under `.mnemosyne/`. Set `MNEMO_HOME` to move the local database, uploads, and index.

## Architecture

```text
documents -> parser -> chunker -> embedder -> SQLite chunks + FTS + vectors
                                                  |
query -----> embedder -> hybrid retriever --------+
                              |
                         cited context -> generator -> grounded answer
```

Current defaults favor simplicity:

- Storage: local SQLite for metadata, chunks, keyword search, and vectors
- Embeddings: lightweight hashing embedder for zero-setup development
- Generation: Ollama for private local answering
- UI: FastAPI plus static HTML/CSS/JS for a fast local dashboard

The hashing embedder is intentionally temporary. It keeps the system dependency-light while we shape the product, but the next serious upgrade should be a neural embedding provider through Ollama embeddings or Hugging Face sentence transformers.

## Near-term roadmap

1. Upgrade embeddings to a stronger local model
2. Add citations that deep-link into chunk and page previews
3. Introduce folders, tags, and source filters
4. Add semantic clustering and note graph exploration
5. Support watch folders plus Notion and Obsidian import flows
