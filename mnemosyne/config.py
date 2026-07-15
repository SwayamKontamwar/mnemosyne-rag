from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path
    db_path: Path
    settings_path: Path | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    embed_provider: str = "ollama"
    vector_provider: str = "chroma"
    watch_interval: float = 2.0
    chunk_size: int = 900
    chunk_overlap: int = 150

    @classmethod
    def load(cls) -> "Settings":
        home = Path(os.getenv("MNEMO_HOME", ".mnemosyne")).expanduser().resolve()
        return cls(
            home=home,
            db_path=home / "knowledge.db",
            settings_path=home / "settings.json",
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            embed_provider=os.getenv("MNEMO_EMBED_PROVIDER", "ollama"),
            vector_provider=os.getenv("MNEMO_VECTOR_PROVIDER", "chroma"),
            watch_interval=float(os.getenv("MNEMO_WATCH_INTERVAL", "2")),
        )

    def __post_init__(self) -> None:
        if self.settings_path is None:
            object.__setattr__(self, "settings_path", self.home / "settings.json")
