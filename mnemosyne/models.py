from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    path: str
    title: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    document_path: str
    title: str
    text: str
    ordinal: int
    start_line: int | None
    end_line: int | None
    page: int | None = None

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

