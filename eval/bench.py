"""Latency / throughput / concurrency benchmark: replay 1, 5, 20 calls simultaneously at --speed 1.

Refuses to write bench.csv unless ENV=rocm (dev-provider latency is not a result).
All numbers are computed from runs/<run_id>/events.jsonl.

  python eval/bench.py --audio data/audio/call.mp3 --concurrency 1 5 20 --max-seconds 600
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clients import ASRClient, LLMClient  # noqa: E402
from compare import load_prior  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from eval.gpu import GpuSampler  # noqa: E402
from eval.metrics import summarize  # noqa: E402
from events import EventLog, new_run_id, read_events, run_metadata  # noqa: E402
from pipeline import run_call  # noqa: E402

SLO_P95_S = 10.0
SLO_CONCURRENCY = 5
ENV_COLS = ["gpu_model", "rocm_version", "image_tag", "vllm_version", "vllm_flags", "asr_backend",
            "asr_model", "asr_precision", "llm_model", "llm_precision"]


async def server_info(base_url: str) -> dict:
    """vLLM exposes /version at the server root and /v1/models; record both."""
    root = base_url.rstrip("/").removesuffix("/v1")
    info: dict = {}
    async with httpx.AsyncClient(timeout=10) as http:
        for key, url in (("version", f"{root}/version"), ("models", f"{base_url.rstrip('/')}/models")):
            try:
                r = await http.get(url)
                if r.status_code == 200:
                    info[key] = r.json()
            except httpx.HTTPError:
                pass
    return info


def summarize_level(events: list[dict], level: int) -> list[dict]:
    prefix = f"n{level}-"
    ev = [e for e in events if str(e.get("call_id", "")).startswith(prefix)]
    llm_calls = [c for e in ev if e["event"] == "llm_done" for c in e["calls"]]
    series = {
        "alert_latency_s": [e["latency_s"] for e in ev if e["event"] == "alert" and "latency_s" in e],
        "insight_latency_s": [e["t"] - e["source_t"] for e in ev if e["event"] == "llm_done"],
        "asr_latency_s": [e["latency_s"] for e in ev if e["event"] == "asr_done"],
        "llm_ttft_s": [c["ttft_s"] for c in llm_calls],
        "llm_total_s": [c["total_s"] for c in llm_calls],
        "llm_output_tok_per_s": [c["output_tok_per_s"] for c in llm_calls],
    }
    rows = [{"concurrency": level, "measure": name, **summarize(vals)} for name, vals in series.items()]
    done = [e for e in ev if e["event"] == "call_done"]
    failures = sum(1 for e in ev if e["event"] in ("asr_failed", "extract_failed"))
    span = (max(e["t"] for e in ev) - min(e["t"] for e in ev)) if ev else 0
    tokens = sum(c.get("completion_tokens") or 0 for c in llm_calls)
    n_chunks = sum(1 for e in ev if e["event"] == "asr_done")
    empty = {"p50": None, "p95": None, "p99": None}
    rows.append({"concurrency": level, "measure": "throughput_output_tok_per_s", "count": len(done),
                 "mean": tokens / span if span else None, **empty})
    rows.append({"concurrency": level, "measure": "throughput_chunks_per_s", "count": n_chunks,
                 "mean": n_chunks / span if span else None, **empty})
    for r in rows:
        r["failures"] = failures
        r["skipped_windows"] = sum(e.get("skipped_windows", 0) for e in done)
    return rows


async def run_level(level: int, args, s: Settings, events: EventLog, prior, sampler: GpuSampler | None) -> dict:
    gpu_on = await sampler.start() if sampler else False
    async with ASRClient(s) as asr, LLMClient(s) as llm:
        results = await asyncio.gather(
            *(run_call(args.audio, s, events, asr, llm, prior, call_id=f"n{level}-{i}",
                       speed=args.speed, max_seconds=args.max_seconds) for i in range(level)),
            return_exceptions=True,
        )
    gpu = await sampler.stop() if sampler and gpu_on else {"gpu_util_mean": None, "gpu_util_max": None}
    errors = [r for r in results if isinstance(r, BaseException)]
    for err in errors:
        print(f"  call failed: {err!r}")
    return {**gpu, "call_errors": len(errors)}


def _fmt(v):
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)


async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, help="replay only the first N seconds of each call")
    ap.add_argument("--gpu-ssh", help="user@gpu-host, if bench runs off-box")
    ap.add_argument("--run-id")
    ap.add_argument("--prior", default=str(ROOT / s.last_quarter_path))
    args = ap.parse_args()

    if s.env != "rocm":
        sys.exit("bench.py only runs with ENV=rocm: dev-provider latency numbers are not results.")
    if args.speed != 1.0:
        print("warning: concurrency results are only comparable at --speed 1")

    run_id = args.run_id or new_run_id("bench")
    run_dir = ROOT / s.runs_dir / run_id
    events = EventLog(run_dir, run_id)
    meta = run_metadata(s, kind="bench", audio=Path(args.audio).name, speed=args.speed,
                        max_seconds=args.max_seconds, concurrency=args.concurrency,
                        llm_server=await server_info(s.llm_base_url), asr_server=await server_info(s.asr_base_url))
    if not meta["vllm_version"]:
        meta["vllm_version"] = str(meta["llm_server"].get("version", {}).get("version", ""))
    events.write_meta(meta)
    prior = load_prior(args.prior)
    sampler = GpuSampler(args.gpu_ssh)
    if not await sampler.detect():
        print("warning: amd-smi/rocm-smi not reachable; GPU utilisation will be empty (use --gpu-ssh)")
        sampler = None

    level_info = {}
    for level in args.concurrency:
        print(f"== {level} concurrent call(s) ==")
        level_info[level] = await run_level(level, args, s, events, prior, sampler)
    events.close()

    all_events, _ = read_events(events.path)
    rows = []
    for level in args.concurrency:
        for r in summarize_level(all_events, level):
            rows.append({**r, **level_info[level], **{c: meta.get(c, "") for c in ENV_COLS}})

    out = run_dir / "bench.csv"
    cols = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({k: _fmt(v) for k, v in r.items()} for r in rows)

    for r in rows:
        if r["measure"] in ("alert_latency_s", "insight_latency_s", "asr_latency_s", "llm_total_s"):
            print(f"n={r['concurrency']:>2} {r['measure']:20s} count={r['count']:>5} "
                  f"p50={_fmt(r['p50'])} p95={_fmt(r['p95'])} p99={_fmt(r['p99'])} gpu={_fmt(r['gpu_util_mean'])}")
    slo = next((r for r in rows if r["concurrency"] == SLO_CONCURRENCY and r["measure"] == "alert_latency_s"), None)
    if slo and slo["p95"] is not None:
        verdict = "PASS" if slo["p95"] < SLO_P95_S else "FAIL"
        print(f"SLO p95 alert latency < {SLO_P95_S}s at {SLO_CONCURRENCY} calls: {verdict} ({slo['p95']:.2f}s)")
    print(f"-> {out}\n   meta: {json.dumps({k: meta.get(k) for k in ENV_COLS})}")


if __name__ == "__main__":
    asyncio.run(main())
