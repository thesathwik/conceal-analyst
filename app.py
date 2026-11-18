"""Streamlit dashboard: transcript | extracted metrics | alerts, updating live.

The UI starts `pipeline.py` as a subprocess and tails runs/<run_id>/events.jsonl,
so the dashboard shows exactly what the benchmarked pipeline produced.
Run: streamlit run app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from config import get_settings
from events import new_run_id, read_events

settings = get_settings()
RUNS = Path(settings.runs_dir)
AUDIO_DIR = Path("data/audio")
KIND_STYLE = {"CUT": "🔻", "RAISE": "🔺", "NEW": "🆕", "WITHDRAWN": "⚠️", "REITERATED": "＝"}

st.set_page_config(page_title="Concall Analyst", layout="wide")


def _state() -> dict:
    s = st.session_state
    s.setdefault("run_id", None)
    s.setdefault("proc", None)
    s.setdefault("offset", 0)
    s.setdefault("events", [])
    return s


def start_run(audio: Path, speed: float, max_seconds: float | None) -> None:
    s = _state()
    run_id = new_run_id("ui")
    cmd = [sys.executable, "pipeline.py", "--audio", str(audio), "--speed", str(speed), "--run-id", run_id]
    if max_seconds:
        cmd += ["--max-seconds", str(max_seconds)]
    (RUNS / run_id).mkdir(parents=True, exist_ok=True)
    log = (RUNS / run_id / "pipeline.log").open("w")
    s.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    s.run_id, s.offset, s.events = run_id, 0, []


def stop_run() -> None:
    s = _state()
    if s.proc is not None and s.proc.poll() is None:
        s.proc.terminate()


def sidebar() -> None:
    s = _state()
    st.sidebar.header("Replay a recorded call")
    files = sorted(p for p in AUDIO_DIR.glob("*") if p.suffix.lower() in {".mp3", ".wav", ".m4a"})
    if not files:
        st.sidebar.warning(f"Put a recorded call in {AUDIO_DIR}/ (not committed).")
    audio = st.sidebar.selectbox("Audio", files, format_func=lambda p: p.name) if files else None
    speed = st.sidebar.select_slider("Playback speed", [1.0, 2.0, 4.0, 8.0], value=settings.playback_speed)
    max_s = st.sidebar.number_input("Only first N seconds (0 = all)", min_value=0, value=0, step=60)
    running = s.proc is not None and s.proc.poll() is None
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Start", disabled=running or audio is None, type="primary"):
        start_run(audio, speed, max_s or None)
        st.rerun()
    if c2.button("Stop", disabled=not running):
        stop_run()

    st.sidebar.divider()
    past = sorted((p.name for p in RUNS.glob("*") if (p / "events.jsonl").exists()), reverse=True)
    if past:
        pick = st.sidebar.selectbox("…or open a past run", ["—"] + past)
        if pick != "—" and pick != s.run_id and not running:
            s.run_id, s.offset, s.events = pick, 0, []
    st.sidebar.caption(f"ENV={settings.env} · ASR={settings.asr_model or '?'} · LLM={settings.llm_model or '?'}")


def _fmt_range(lo, hi) -> str:
    if lo is None:
        return "—"
    return f"{lo:,.2f}" if lo == hi else f"{lo:,.2f} – {hi:,.2f}"


def render(events: list[dict]) -> None:
    transcript = [e for e in events if e["event"] == "asr_done" and e.get("text")]
    llm_done = [e for e in events if e["event"] == "llm_done"]
    alerts = [e for e in events if e["event"] == "alert"]
    status = {}
    for e in events:
        if e["event"] in ("alert", "reiterated"):
            a = e["alert"]
            status[(a["metric"], a["period"])] = a["kind"]

    audio_s = max((e.get("audio_s") or 0 for e in events), default=0)
    done = any(e["event"] == "call_done" for e in events)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Audio clock", f"{int(audio_s // 60)}:{int(audio_s % 60):02d}")
    m2.metric("Chunks transcribed", len(transcript))
    m3.metric("Extractions", len(llm_done))
    m4.metric("Alerts", len(alerts))
    if done:
        st.success("Call finished.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Transcript")
        lines = [f"`{int(e['audio_s'] // 60):02d}:{int(e['audio_s'] % 60):02d}` {e['text']}" for e in transcript[-40:]]
        with st.container(height=520):
            st.markdown("\n\n".join(reversed(lines)) or "_waiting for audio…_")

    with right:
        st.subheader("Alerts")
        with st.container(height=520):
            if not alerts:
                st.caption("No alerts yet.")
            for e in reversed(alerts):
                a = e["alert"]
                prior = _fmt_range(a.get("prior_low"), a.get("prior_high"))
                new = _fmt_range(a.get("new_low"), a.get("new_high"))
                delta = f" ({a['change_bps']:+.0f} bps)" if a.get("change_bps") is not None else (
                    f" ({a['change_pct']:+.1f}%)" if a.get("change_pct") is not None else "")
                st.markdown(f"**{KIND_STYLE.get(a['kind'], '')} {a['kind']} · {a['metric']} {a['period']}** "
                            f"— {prior} → {new} {a['unit']}{delta}")
                if a.get("quote"):
                    st.caption(f"“{a['quote']}”  ·  latency {e.get('latency_s', 0):.1f}s")

    st.subheader("Extracted metrics (latest per metric & period)")
    latest: dict = {}
    for e in llm_done:
        for m in e["metrics"]:
            latest[(m["metric"], m["period"], m["type"])] = {**m, "audio_s": e["audio_s"]}
    if latest:
        df = pd.DataFrame(latest.values())
        df["status"] = [status.get((r.metric, r.period), "") if r.type == "guidance" else "actual"
                        for r in df.itertuples()]
        st.dataframe(df[["metric", "period", "type", "value_low", "value_high", "unit", "status", "quote"]],
                     width="stretch", hide_index=True)
    else:
        st.caption("Nothing extracted yet.")

    failed = [e for e in events if e["event"] in ("extract_failed", "asr_failed")]
    if failed:
        with st.expander(f"{len(failed)} failed requests"):
            st.json(failed[-10:])


sidebar()
st.title("Concall Analyst")


@st.fragment(run_every=1.0)
def live() -> None:
    s = _state()
    if s.run_id is None:
        st.info("Choose a recorded call in the sidebar and press Start.")
        return
    new, s.offset = read_events(RUNS / s.run_id / "events.jsonl", s.offset)
    s.events.extend(new)
    st.caption(f"run `{s.run_id}`")
    render(s.events)
    if s.proc is not None and s.proc.poll() not in (None, 0):
        st.error(f"Pipeline exited with code {s.proc.returncode}; see runs/{s.run_id}/pipeline.log")


live()
