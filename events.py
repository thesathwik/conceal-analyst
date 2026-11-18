"""Structured JSON event log: runs/<run_id>/events.jsonl.

Every record carries `t` (seconds since run start on the monotonic clock) and,
where relevant, `audio_s` (audio-clock offset), so benchmarks can be computed
from the log alone.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings

# Environment descriptors recorded with every run (set them on the GPU box; see deploy/README.md)
ENV_DESCRIPTORS = ("GPU_MODEL", "ROCM_VERSION", "IMAGE_TAG", "VLLM_VERSION", "VLLM_FLAGS", "ASR_BACKEND")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def run_metadata(settings: Settings, **extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "env": settings.env,
        "asr_model": settings.asr_model,
        "asr_precision": settings.asr_precision,
        "llm_model": settings.llm_model,
        "llm_precision": settings.llm_precision,
        "chunk_seconds": settings.chunk_seconds,
        "chunk_overlap_seconds": settings.chunk_overlap_seconds,
        "window_seconds": settings.window_seconds,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    for name in ENV_DESCRIPTORS:
        meta[name.lower()] = os.environ.get(name, "")
    meta.update(extra)
    return meta


class EventLog:
    def __init__(self, run_dir: Path, run_id: str, t0: float | None = None) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.t0 = t0 if t0 is not None else time.monotonic()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "events.jsonl"
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)  # line-buffered: UI tails it

    def t_of(self, mono: float) -> float:
        return round(mono - self.t0, 4)

    def now(self) -> float:
        return self.t_of(time.monotonic())

    def emit(self, event: str, call_id: str, audio_s: float | None = None, **payload: Any) -> dict[str, Any]:
        rec = {
            "event": event,
            "run_id": self.run_id,
            "call_id": call_id,
            "t": self.now(),
            "wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "audio_s": None if audio_s is None else round(audio_s, 3),
            **payload,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def write_meta(self, meta: dict[str, Any]) -> None:
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._fh.close()


def read_events(path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Read complete lines from byte `offset`; returns (events, new_offset). Safe on a growing file."""
    if not path.exists():
        return [], offset
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    end = data.rfind(b"\n") + 1
    events = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return events, offset + end
