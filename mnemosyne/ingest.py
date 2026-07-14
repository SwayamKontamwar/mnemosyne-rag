from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from .models import Chunk, SourceDocument

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx", ".pptx", ".csv", ".tsv", ".xlsx"}
LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z0-9_\-/]+)")
FRONTMATTER_TAGS = re.compile(r"(?ms)^---\s*\n(.*?)\n---\s*\n?")


def discover(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED:
            yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED:
            yield candidate


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path: Path) -> list[SourceDocument]:
    absolute = str(path.resolve())
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        documents: list[SourceDocument] = []
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    SourceDocument(
                        absolute,
                        path.stem,
                        text,
                        index + 1,
                        file_type="pdf",
                        folder=_folder_label(path),
                    )
                )
        return documents
    if suffix == ".docx":
        from zipfile import ZipFile

        with ZipFile(path) as archive:
            raw = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        text = " ".join(re.findall(r">([^<]+)<", raw))
        return [_source_document(path, absolute, text, "docx")]
    if suffix == ".pptx":
        from zipfile import ZipFile

        fragments: list[str] = []
        with ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                    raw = archive.read(name).decode("utf-8", errors="ignore")
                    fragments.append(" ".join(re.findall(r">([^<]+)<", raw)))
        return [_source_document(path, absolute, "\n\n".join(fragments), "pptx")]
    if suffix in {".csv", ".tsv"}:
        return [_source_document(path, absolute, path.read_text(encoding="utf-8", errors="ignore"), suffix.lstrip("."))]
    if suffix == ".xlsx":
        return [_source_document(path, absolute, "Spreadsheet workbook imported for future structured parsing.", "xlsx")]
    return [_source_document(path, absolute, path.read_text(encoding="utf-8"), suffix.lstrip(".") or "txt")]


def chunk_document(doc: SourceDocument, size: int, overlap: int) -> list[Chunk]:
    lines = doc.text.splitlines()
    chunks: list[Chunk] = []
    cursor = 0
    ordinal = 0
    while cursor < len(lines):
        selected: list[str] = []
        chars = 0
        end = cursor
        while end < len(lines) and (chars < size or not selected):
            selected.append(lines[end])
            chars += len(lines[end]) + 1
            end += 1
        text = "\n".join(selected).strip()
        if text:
            chunks.append(
                Chunk(
                    document_path=doc.path,
                    title=doc.title,
                    text=text,
                    ordinal=ordinal,
                    start_line=None if doc.page else cursor + 1,
                    end_line=None if doc.page else end,
                    page=doc.page,
                    tags=doc.tags,
                )
            )
            ordinal += 1
        if end >= len(lines):
            break
        overlap_lines = 0
        overlap_chars = 0
        for line in reversed(selected):
            if overlap_chars >= overlap:
                break
            overlap_chars += len(line) + 1
            overlap_lines += 1
        cursor = max(cursor + 1, end - overlap_lines)
    return chunks


def _source_document(path: Path, absolute: str, text: str, file_type: str) -> SourceDocument:
    cleaned = text.strip()
    tags = tuple(sorted(set(_tags_from_text(cleaned))))
    links = tuple(sorted(set(_links_from_text(cleaned))))
    return SourceDocument(
        absolute,
        path.stem,
        cleaned,
        tags=tags,
        file_type=file_type,
        folder=_folder_label(path),
        links=links,
    )


def _folder_label(path: Path) -> str:
    parent = path.parent.name.strip()
    return parent or "root"


def _links_from_text(text: str) -> list[str]:
    return [match.strip() for match in LINK_PATTERN.findall(text) if match.strip()]


def _tags_from_text(text: str) -> list[str]:
    tags = [match.lower() for match in TAG_PATTERN.findall(text)]
    frontmatter = FRONTMATTER_TAGS.search(text)
    if frontmatter:
        tags.extend(re.findall(r"-\s*([a-zA-Z0-9_\-/]+)", frontmatter.group(1)))
        tags.extend(re.findall(r"tags:\s*\[([^\]]+)\]", frontmatter.group(1)))
    expanded: list[str] = []
    for tag in tags:
        if "," in tag:
            expanded.extend(part.strip().lower() for part in tag.split(",") if part.strip())
        else:
            expanded.append(tag.strip().lower())
    return [tag for tag in expanded if tag]
