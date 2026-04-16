from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ExperimentLogger:
    def __init__(self, log_dir: Path) -> None:
        self.request_writer = JsonlWriter(log_dir / "requests.jsonl")
        self.token_writer = JsonlWriter(log_dir / "tokens.jsonl")
        self.layer_writer = JsonlWriter(log_dir / "layers.jsonl")
        self.memory_writer = JsonlWriter(log_dir / "memory.jsonl")

    @staticmethod
    def now_ts() -> float:
        return time.time()

    def log_request(self, **kwargs: Any) -> None:
        kwargs.setdefault("ts", self.now_ts())
        self.request_writer.write(kwargs)

    def log_token(self, **kwargs: Any) -> None:
        kwargs.setdefault("ts", self.now_ts())
        self.token_writer.write(kwargs)

    def log_layer(self, **kwargs: Any) -> None:
        kwargs.setdefault("ts", self.now_ts())
        self.layer_writer.write(kwargs)

    def log_memory(self, **kwargs: Any) -> None:
        kwargs.setdefault("ts", self.now_ts())
        self.memory_writer.write(kwargs)