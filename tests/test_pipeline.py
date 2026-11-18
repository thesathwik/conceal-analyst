"""Pipeline pieces that need no network or audio: window, de-dupe, and run_call with fake clients."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clients import ASRResult, LLMResult
from config import Settings
from events import EventLog, read_events
from pipeline import RollingWindow, dedupe_overlap, run_call
from schema import PriorGuidance


def test_dedupe_overlap_strips_repeated_boundary_words():
    assert dedupe_overlap("we expect margins of", "margins of 16 percent") == "16 percent"
    assert dedupe_overlap("hello world", "something else") == "something else"
    assert dedupe_overlap("the", "the end") == "the end"  # single-word overlap is kept


def test_rolling_window_trims_by_audio_clock():
    w = RollingWindow(seconds=20)
    w.add(0, 10, "one")
    w.add(10, 20, "two")
    w.add(20, 30, "three")
    assert w.text() == "two three"


class FakeASR:
    def __init__(self, texts):
        self.texts = list(texts)

    async def transcribe(self, wav, prompt=None):
        return ASRResult(text=self.texts.pop(0) if self.texts else "", latency_s=0.01)


class FakeLLM:
    async def chat(self, messages):
        window = messages[-1]["content"]
        metrics = []
        if "EBITDA margins of 15 percent" in window:
            metrics.append({"metric": "ebitda_margin", "value_low": 15, "value_high": 15, "unit": "percent",
                            "period": "FY27", "type": "guidance", "quote": "EBITDA margins of 15 percent"})
        return LLMResult(json.dumps({"metrics": metrics, "tone_flags": []}), 0.01, 0.02, 100, 20)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_run_call_end_to_end_with_fake_endpoints(tmp_path: Path):
    audio = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=30", str(audio)], check=True)
    settings = Settings(_env_file=None, chunk_seconds=10, window_seconds=150)
    prior = PriorGuidance.model_validate({"metrics": [
        {"metric": "ebitda_margin", "value_low": 16, "value_high": 17, "unit": "percent", "period": "FY27",
         "type": "guidance", "quote": "q"}]})
    events = EventLog(tmp_path / "run", "test")
    asr = FakeASR(["good afternoon everyone", "for FY27 we now see EBITDA margins of 15 percent", "thank you"])
    result = await run_call(audio, settings, events, asr, FakeLLM(), prior, speed=0)
    events.close()

    assert result.chunks == 3
    assert [(a.kind.value, a.metric) for a in result.alerts] == [("CUT", "ebitda_margin")]
    evs, _ = read_events(events.path)
    names = [e["event"] for e in evs]
    for required in ("chunk_start", "asr_done", "llm_done", "alert", "call_done"):
        assert required in names
    alert = next(e for e in evs if e["event"] == "alert")
    # unpaced replay may coalesce windows, so the alert comes from chunk 1's or chunk 2's window
    assert alert["latency_s"] >= 0 and alert["audio_s"] in (20.0, 30.0)
