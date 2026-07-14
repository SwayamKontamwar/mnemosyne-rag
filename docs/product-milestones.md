# Mnemosyne Product Milestones

This file turns the roadmap into a build contract for the project. The goal is not to claim every milestone is finished today. The goal is to give the repository a concrete product shape so future implementation can stay intentional.

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

Status:
- Hybrid retrieval and grounded answers exist
- Ollama embeddings are now wired as a first-class local provider option
- Chroma adapter, reranking, and citation validation still need focused implementation

## Milestone 3 — Broad document support

- DOCX, PowerPoint and spreadsheets
- OCR for scanned PDFs
- Structure-aware chunking
- Folder imports
- File-system watcher
- Better parsing diagnostics

Status:
- DOCX, PPTX, CSV, TSV, and XLSX ingestion stubs are now recognized
- Folder imports already work through recursive discovery
- OCR, watcher orchestration, and richer parsing diagnostics remain open

## Milestone 4 — Knowledge intelligence

- Semantic backlinks
- Entity extraction
- Topic clustering
- Knowledge graph
- Related-note recommendations
- Cross-document comparison
- Timeline and contradiction detection

Status:
- Semantic backlinks already exist
- Topic clustering and graph endpoints now exist in lightweight form
- Entity extraction, contradiction detection, and deeper cross-document reasoning remain future work

## Milestone 5 — Polished personal product

- Collections and tags
- Saved searches
- Conversation history
- Provider settings
- Privacy controls
- Import/export and backups
- Evaluation dashboard
- Docker and one-command installation

Status:
- Tags and source filters now have backend support
- The remaining items should be treated as the next UX and ops layer

## Build principles

- Local-first by default
- Model-provider modularity
- Grounded answers with inspectable citations
- Metadata-rich documents rather than opaque blobs
- Fast retrieval first, richer intelligence layered on top
