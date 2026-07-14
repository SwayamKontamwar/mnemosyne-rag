# Mnemosyne — local-first personal knowledge RAG

Mnemosyne indexes Markdown, text, and PDF files and lets you search or ask questions
with source citations. The MVP is intentionally small, private, and provider-neutral.

## What works

- Recursive ingestion of `.md`, `.txt`, and `.pdf`
- Incremental re-indexing using content hashes
- Hybrid retrieval: SQLite FTS5 keyword search + cosine vector search
- Grounded answers through a local Ollama model
- File-and-line citations for text notes; page citations for PDFs
- Semantic backlink discovery between related chunks
- Pluggable embedding, generation, and future vector-store providers
- No note content leaves the machine with the default configuration

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

mnemo init
mnemo ingest ~/Notes
mnemo search "ideas about distributed systems"
mnemo backlinks "meeting-notes.md"
mnemo ask "What have I written about retrieval evaluation?"
```

`search` works without Ollama. For generated answers, install Ollama and run a model:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Data is stored under `.mnemosyne/` by default. Override it with `MNEMO_HOME`.

## Architecture

```text
files -> parser -> chunker -> embedder -> SQLite chunks + FTS + vectors
                                            |
query -> embedder -> hybrid retriever -------+
                         |
                    cited context -> Ollama -> grounded answer
```

The built-in hashing embedder is dependency-free and useful for getting started. It
is not as semantically capable as a neural embedding model. The next provider should
target Ollama `/api/embed` or `sentence-transformers`; the protocol is already defined
in `mnemosyne/providers.py`.

## Roadmap

1. Neural local embeddings and Chroma/Qdrant adapters
2. Browser UI for reading, querying, and inspecting citations
3. Note graph, clustering, and automatic topic pages
4. Folder watchers and Notion/Obsidian importers
5. Evaluation set for retrieval recall and citation correctness

