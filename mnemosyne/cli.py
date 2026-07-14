from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .providers import OllamaGenerator
from .service import KnowledgeBase


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mnemo", description="Local personal knowledge search")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create the local knowledge database")
    ingest = commands.add_parser("ingest", help="Index a file or directory")
    ingest.add_argument("path", type=Path)
    search = commands.add_parser("search", help="Run hybrid search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    ask = commands.add_parser("ask", help="Answer from notes through Ollama")
    ask.add_argument("query")
    links = commands.add_parser("backlinks", help="Find semantically related notes")
    links.add_argument("document")
    links.add_argument("--limit", type=int, default=10)
    return root


def main() -> None:
    args = parser().parse_args()
    settings = Settings.load()
    kb = KnowledgeBase(settings)
    if args.command == "init":
        print(f"Knowledge base ready at {settings.db_path}")
    elif args.command == "ingest":
        indexed, skipped = kb.ingest(args.path.expanduser())
        print(f"Indexed {indexed} file(s); skipped {skipped} unchanged file(s).")
    elif args.command == "search":
        _print_hits(kb.search(args.query, args.limit))
    elif args.command == "backlinks":
        _print_hits(kb.backlinks(args.document, args.limit))
    elif args.command == "ask":
        answer, hits = kb.ask(args.query, OllamaGenerator(settings.ollama_url, settings.ollama_model))
        print(answer)
        print("\nSources:")
        for index, hit in enumerate(hits, 1):
            print(f"[{index}] {hit.citation}")


def _print_hits(hits: list) -> None:
    if not hits:
        print("No matches found.")
    for hit in hits:
        preview = " ".join(hit.text.split())[:220]
        print(f"{hit.score:.3f}  {hit.citation}\n       {preview}\n")


if __name__ == "__main__":
    main()

