# Mnemosyne Product Milestones

This file is the implemented product contract. Each milestone has an executable path in the service, CLI, web UI, or container stack and regression coverage for its critical workflow.

## Milestone 1 — Reliable local library

- Drag-and-drop interface
- Markdown, TXT and PDF ingestion
- Document library
- SQLite metadata
- FTS5 keyword search
- Incremental indexing
- Exact line and page citations

Status:
- Implemented in the current project
- Extended with chunk preview APIs and richer metadata plumbing

## Milestone 2 — Real semantic RAG

- Ollama embeddings
- Chroma adapter
- Hybrid retrieval
- Local reranking
- Grounded Ollama answers
- Citation validation
- Search and chat interfaces

Status: implemented. Compose uses real Ollama embeddings and persistent Chroma; retrieval fuses semantic and FTS5 signals, reranks locally, repairs uncited model output, and validates every returned citation.

## Milestone 3 — Broad document support

- DOCX, PowerPoint and spreadsheets
- OCR for scanned PDFs
- Structure-aware chunking
- Folder imports
- File-system watcher
- Better parsing diagnostics

Status: implemented. Office documents retain paragraph, slide, row, or sheet structure where available. Scanned PDF pages use local Poppler/Tesseract OCR. The app continuously scans registered folders, handles edits and deletions, imports safe ZIP exports, and records parser failures.

## Milestone 4 — Knowledge intelligence

- Semantic backlinks
- Entity extraction
- Topic clustering
- Knowledge graph
- Related-note recommendations
- Cross-document comparison
- Timeline and contradiction detection

Status: implemented as a transparent local intelligence layer. Backlinks and recommendations use document vectors, graph edges include semantic/tag/wiki-link reasons, and reader views expose entities, timelines, cross-document comparison, and explainable contradiction candidates. Entity and contradiction analysis is intentionally heuristic rather than presented as infallible fact extraction.

## Milestone 5 — Polished personal product

- Collections and tags
- Saved searches
- Conversation history
- Provider settings
- Privacy controls
- Import/export and backups
- Evaluation dashboard
- Docker and one-command installation

Status: implemented across the web UI and API, including collection creation, saved searches, history, live provider readiness, strict-local preferences, complete chunk-bearing backups and restore, grounding evaluations, and the tested Compose installation.

## Build principles

- Local-first by default
- Model-provider modularity
- Grounded answers with inspectable citations
- Metadata-rich documents rather than opaque blobs
- Fast retrieval first, richer intelligence layered on top
