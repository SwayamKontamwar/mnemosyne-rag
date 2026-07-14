from __future__ import annotations

import argparse
import json
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
    search.add_argument("--tag")
    search.add_argument("--folder")
    search.add_argument("--type", dest="file_type")
    ask = commands.add_parser("ask", help="Answer from notes through Ollama")
    ask.add_argument("query")
    ask.add_argument("--tag")
    ask.add_argument("--folder")
    ask.add_argument("--type", dest="file_type")
    links = commands.add_parser("backlinks", help="Find semantically related notes")
    links.add_argument("document")
    links.add_argument("--limit", type=int, default=10)
    serve = commands.add_parser("serve", help="Launch the local web application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    watch = commands.add_parser("watch", help="Register or scan watch folders")
    watch.add_argument("path", nargs="?")
    watch.add_argument("--profile", default="local")
    watch.add_argument("--scan", action="store_true")
    watch.add_argument("--daemon", action="store_true", help="Continuously reindex registered folders")
    watch.add_argument("--interval", type=float, default=2.0)
    commands.add_parser("backup", help="Export a local JSON backup")
    restore = commands.add_parser("restore", help="Restore a Mnemosyne JSON backup")
    restore.add_argument("path", type=Path)
    archive = commands.add_parser("import", help="Import a Notion or Obsidian ZIP export")
    archive.add_argument("path", type=Path)
    archive.add_argument("--profile", default="notion")
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
        _print_hits(kb.search(args.query, args.limit, tag=args.tag, folder=args.folder, file_type=args.file_type))
    elif args.command == "backlinks":
        _print_hits(kb.backlinks(args.document, args.limit))
    elif args.command == "ask":
        answer, hits, validation = kb.ask(
            args.query,
            OllamaGenerator(kb.settings.ollama_url, kb.settings.ollama_model),
            tag=args.tag,
            folder=args.folder,
            file_type=args.file_type,
        )
        print(answer)
        print(f"\nAudit: {validation.verdict}")
        print("\nSources:")
        for index, hit in enumerate(hits, 1):
            print(f"[{index}] {hit.citation}")
    elif args.command == "watch":
        if args.daemon:
            print(f"Watching registered folders every {args.interval:g}s. Press Ctrl+C to stop.")
            try:
                kb.watch_forever(interval=args.interval)
            except KeyboardInterrupt:
                pass
        elif args.scan:
            print(json.dumps(kb.scan_watch_folders(), indent=2))
        elif args.path:
            indexed, skipped = kb.register_watch_folder(Path(args.path), args.profile)
            print(f"Watching {Path(args.path).expanduser()} — indexed {indexed}, skipped {skipped}.")
        else:
            for watch in kb.store.list_watch_folders():
                print(f"{watch.path} [{watch.profile}]")
    elif args.command == "backup":
        print(json.dumps(kb.backup(), indent=2))
    elif args.command == "restore":
        print(json.dumps(kb.store.restore_payload(json.loads(args.path.read_text())), indent=2))
    elif args.command == "import":
        indexed, skipped = kb.import_archive(args.path.expanduser(), args.profile)
        print(f"Imported {indexed} file(s); skipped {skipped} unchanged file(s).")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("mnemosyne.web:app", host=args.host, port=args.port, reload=False)


def _print_hits(hits: list) -> None:
    if not hits:
        print("No matches found.")
    for hit in hits:
        preview = " ".join(hit.text.split())[:220]
        print(f"{hit.score:.3f}  {hit.citation}\n       {preview}\n")


if __name__ == "__main__":
    main()
