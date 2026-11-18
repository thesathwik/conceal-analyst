"""Bring-up smoke checks for the ROCm deployment (reads endpoints from .env, ENV=rocm).

1. amd-smi / rocm-smi lists the GPU (locally, or via --gpu-ssh user@host)
2. /v1/models responds on both services
3. one transcription succeeds
4. one extraction succeeds and returns schema-valid JSON

  python deploy/smoke.py --audio data/audio/<call>.mp3 [--gpu-ssh user@gpu-host]
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chunker import chunk_audio  # noqa: E402
from clients import ASRClient, LLMClient  # noqa: E402
from config import get_settings  # noqa: E402
from extract import extract_window, load_prompt  # noqa: E402

SAMPLE = ("For FY27 we expect EBITDA margins of 16 to 17 percent, and capex of about 1,200 crore. "
          "Our order book stands at 25,000 crores.")


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


def gpu_check(ssh: str | None) -> bool:
    for cmd in (["amd-smi", "list"], ["rocm-smi", "--showproductname"]):
        full = ["ssh", ssh, " ".join(map(shlex.quote, cmd))] if ssh else cmd
        try:
            out = subprocess.run(full, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and ("GPU" in out.stdout or "MI300" in out.stdout):
            first = next((ln.strip() for ln in out.stdout.splitlines() if "MI3" in ln or "GPU" in ln), "")
            return check(f"{cmd[0]} lists the GPU", True, first[:100])
    return check("amd-smi / rocm-smi lists the GPU", False, "not found or no GPU (try --gpu-ssh)")


async def models_check(name: str, base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(base_url.rstrip("/") + "/models")
        ids = [m["id"] for m in r.json().get("data", [])]
        return check(f"{name} /v1/models", r.status_code == 200 and bool(ids), ", ".join(ids))
    except (httpx.HTTPError, ValueError) as exc:
        return check(f"{name} /v1/models", False, repr(exc))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", help="any recorded call; the first chunk is transcribed")
    ap.add_argument("--gpu-ssh")
    args = ap.parse_args()
    s = get_settings()
    if s.env != "rocm":
        print(f"warning: ENV={s.env}; smoke checks are meant for the rocm environment")

    results = [gpu_check(args.gpu_ssh),
               await models_check("ASR", s.asr_base_url),
               await models_check("LLM", s.llm_base_url)]

    if args.audio:
        async with ASRClient(s) as asr:
            async for chunk in chunk_audio(args.audio, s.chunk_seconds, 0, max_seconds=s.chunk_seconds):
                try:
                    res = await asr.transcribe(chunk.wav)
                    results.append(check("transcription", bool(res.text), f"{res.latency_s:.2f}s '{res.text[:80]}'"))
                except Exception as exc:  # noqa: BLE001 - report any failure
                    results.append(check("transcription", False, repr(exc)))
                break
    else:
        print("[SKIP] transcription (pass --audio)")

    async with LLMClient(s) as llm:
        try:
            out = await extract_window(llm, load_prompt(ROOT / s.prompt_path), SAMPLE)
            ok = out.extraction is not None and len(out.extraction.metrics) >= 2
            got = [(m.metric, m.value_low, m.value_high, m.period) for m in out.extraction.metrics] if out.extraction else out.error
            results.append(check("extraction", ok, f"{out.llm_latency_s:.2f}s {got}"))
        except Exception as exc:  # noqa: BLE001
            results.append(check("extraction", False, repr(exc)))

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
