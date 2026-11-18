"""Orchestration: chunks -> ASR -> rolling window -> LLM extraction -> compare -> alerts.

ASR runs in the chunk loop; extraction runs in a separate worker task, so ASR
of chunk N+1 overlaps extraction of window N. If extraction falls behind, the
worker skips stale windows and always takes the newest one ("latest wins"),
which keeps alert latency bounded instead of building a queue.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from chunker import chunk_audio
from clients import ASRClient, EndpointError, LLMClient
from compare import GuidanceTracker, Thresholds, load_prior
from config import Settings, get_settings
from events import EventLog, new_run_id, run_metadata
from extract import extract_window, load_prompt
from schema import Alert, PriorGuidance

log = logging.getLogger("pipeline")
_WORD_RE = re.compile(r"[^\w%₹.]+")


def _norm_word(w: str) -> str:
    return _WORD_RE.sub("", w.lower()).rstrip(".")


def dedupe_overlap(prev: str, new: str, max_words: int = 30, min_words: int = 2) -> str:
    """Strip from `new` the longest prefix that repeats the end of `prev` (chunk-boundary duplicates)."""
    pw, nw = prev.split(), new.split()
    pn = [_norm_word(w) for w in pw[-max_words:]]
    nn = [_norm_word(w) for w in nw[:max_words]]
    for k in range(min(len(pn), len(nn)), min_words - 1, -1):
        if pn[-k:] == nn[:k]:
            return " ".join(nw[k:])
    return new


class RollingWindow:
    """Last `seconds` of transcript on the audio clock."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.segments: deque[tuple[float, float, str]] = deque()

    def add(self, start_s: float, end_s: float, text: str) -> str:
        if self.segments:
            text = dedupe_overlap(self.segments[-1][2], text)
        text = text.strip()
        if text:
            self.segments.append((start_s, end_s, text))
        while self.segments and self.segments[0][1] <= end_s - self.seconds:
            self.segments.popleft()
        return text

    def text(self) -> str:
        return " ".join(s[2] for s in self.segments)

    def tail(self, n_chars: int = 200) -> str:
        return self.text()[-n_chars:]


@dataclass
class Snapshot:
    text: str
    chunk: int
    audio_s: float
    source_t: float  # when the newest chunk in this window was released (run clock)


@dataclass
class CallResult:
    call_id: str
    alerts: list[Alert]
    chunks: int
    extractions: int
    skipped_windows: int


async def run_call(
    audio: str | Path,
    settings: Settings,
    events: EventLog,
    asr: ASRClient,
    llm: LLMClient,
    prior: PriorGuidance,
    call_id: str = "call-0",
    speed: float | None = None,
    max_seconds: float | None = None,
) -> CallResult:
    speed = settings.playback_speed if speed is None else speed
    prompt = load_prompt(settings.prompt_path)
    window = RollingWindow(settings.window_seconds)
    tracker = GuidanceTracker(
        prior.metrics, Thresholds(settings.cut_raise_relative_pct, settings.margin_threshold_bps)
    )
    alerts: list[Alert] = []
    state = {"pending": None, "done": False, "skipped": 0, "extractions": 0}
    wake = asyncio.Event()

    def emit_alerts(new_alerts: list[Alert], snap: Snapshot | None) -> None:
        for a in new_alerts:
            name = "alert" if a.kind.notifies else "reiterated"
            extra = {}
            if snap is not None:
                extra = {"source_chunk": snap.chunk, "source_t": snap.source_t,
                         "latency_s": round(events.now() - snap.source_t, 4)}
            events.emit(name, call_id, audio_s=snap.audio_s if snap else None, alert=a.model_dump(mode="json"), **extra)
            if a.kind.notifies:
                alerts.append(a)
                log.info("[%s] %s %s %s: %s", call_id, a.kind.value, a.metric, a.period, a.quote[:80])

    async def llm_worker() -> None:
        while True:
            await wake.wait()
            wake.clear()
            snap: Snapshot | None = state["pending"]
            state["pending"] = None
            if snap is None:
                if state["done"]:
                    return
                continue
            try:
                outcome = await extract_window(llm, prompt, snap.text)
            except EndpointError as exc:
                events.emit("extract_failed", call_id, audio_s=snap.audio_s, source_chunk=snap.chunk, error=str(exc))
                wake.set()
                continue
            state["extractions"] += 1
            calls = [{"ttft_s": c.ttft_s, "total_s": round(c.total_s, 4), "prompt_tokens": c.prompt_tokens,
                      "completion_tokens": c.completion_tokens, "output_tok_per_s": c.output_tok_per_s}
                     for c in outcome.calls]
            if outcome.extraction is None:
                events.emit("extract_failed", call_id, audio_s=snap.audio_s, source_chunk=snap.chunk,
                            error=outcome.error, calls=calls)
            else:
                events.emit(
                    "llm_done", call_id, audio_s=snap.audio_s, source_chunk=snap.chunk, source_t=snap.source_t,
                    calls=calls, repaired=outcome.repaired, window_chars=len(snap.text),
                    metrics=[m.model_dump() for m in outcome.extraction.metrics],
                    tone_flags=[f.model_dump() for f in outcome.extraction.tone_flags],
                    rejected=outcome.rejected,
                )
                emit_alerts(tracker.update(outcome.extraction.metrics), snap)
            wake.set()  # re-check: a newer window may have arrived meanwhile

    worker = asyncio.create_task(llm_worker())
    chunks = 0
    try:
        async for chunk in chunk_audio(audio, settings.chunk_seconds, speed, settings.chunk_overlap_seconds,
                                       max_seconds, start_at=time.monotonic()):
            chunks += 1
            released_t = events.t_of(chunk.released_at)
            events.emit("chunk_start", call_id, audio_s=chunk.end_s, chunk=chunk.index,
                        audio_start_s=chunk.start_s, released_t=released_t)
            try:
                res = await asr.transcribe(chunk.wav, prompt=window.tail(200) or None)
            except EndpointError as exc:
                events.emit("asr_failed", call_id, audio_s=chunk.end_s, chunk=chunk.index, error=str(exc))
                continue
            added = window.add(chunk.start_s, chunk.end_s, res.text)
            events.emit("asr_done", call_id, audio_s=chunk.end_s, chunk=chunk.index,
                        latency_s=round(res.latency_s, 4), text=added, raw_text=res.text)
            if worker.done():
                worker.result()  # surface unexpected worker crashes
            if added:
                if state["pending"] is not None:
                    state["skipped"] += 1
                state["pending"] = Snapshot(window.text(), chunk.index, chunk.end_s, released_t)
                wake.set()
    finally:
        state["done"] = True
        wake.set()
        await worker

    emit_alerts(tracker.withdrawn(), None)
    events.emit("call_done", call_id, chunks=chunks, extractions=state["extractions"],
                skipped_windows=state["skipped"], alerts=len(alerts))
    return CallResult(call_id, alerts, chunks, state["extractions"], state["skipped"])


async def _main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description="Replay a recorded concall through the live pipeline.")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--speed", type=float, default=settings.playback_speed, help="playback speed; 0 = unpaced")
    ap.add_argument("--max-seconds", type=float, help="only process the first N seconds of audio")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--prior", default=settings.last_quarter_path)
    args = ap.parse_args()

    run_id = args.run_id or new_run_id()
    events = EventLog(Path(settings.runs_dir) / run_id, run_id)
    events.write_meta(run_metadata(settings, audio=Path(args.audio).name, speed=args.speed, kind="pipeline"))
    print(f"run_id={run_id}  events -> {events.path}")
    prior = load_prior(args.prior)
    try:
        async with ASRClient(settings) as asr, LLMClient(settings) as llm:
            result = await run_call(args.audio, settings, events, asr, llm, prior, speed=args.speed,
                                    max_seconds=args.max_seconds)
    finally:
        events.close()
    print(f"done: {result.chunks} chunks, {result.extractions} extractions, "
          f"{result.skipped_windows} skipped windows, {len(result.alerts)} alerts")
    for a in result.alerts:
        print(f"  {a.kind.value:9s} {a.metric:15s} {a.period:8s} {a.prior_low}-{a.prior_high} -> {a.new_low}-{a.new_high}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main())
