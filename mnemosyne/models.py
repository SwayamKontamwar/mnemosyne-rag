from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceDocument:
    path: str
    title: str
    text: str
    page: int | None = None
    tags: tuple[str, ...] = ()
    source_type: str = "file"
    file_type: str = "txt"
    folder: str = ""
    links: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    document_path: str
    title: str
    text: str
    ordinal: int
    start_line: int | None
    end_line: int | None
    page: int | None = None
    tags: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        if self.page is not None:
            return f"{self.document_path}#page={self.page}"
        if self.start_line is not None:
            return f"{self.document_path}#L{self.start_line}-L{self.end_line}"
        return self.document_path


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    text: str
    title: str
    citation: str
    score: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CitationValidation:
    cited_numbers: tuple[int, ...] = ()
    missing_numbers: tuple[int, ...] = ()
    unsupported_numbers: tuple[int, ...] = ()
    answer_has_citations: bool = False
    verdict: str = "unverified"


@dataclass(frozen=True)
class DocumentRecord:
    path: str
    title: str
    digest: str
    indexed_at: str
    source_type: str
    file_type: str
    folder: str
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    chunk_count: int = 0
    character_count: int = 0


@dataclass(frozen=True)
class ChunkPreview:
    chunk_id: int
    title: str
    text: str
    citation: str
    page: int | None
    start_line: int | None
    end_line: int | None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: float
    reason: str


@dataclass(frozen=True)
class TopicCluster:
    name: str
    document_paths: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)
