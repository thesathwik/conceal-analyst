"""Validation per model x precision: ASR WER, finance-term error, numeric extraction accuracy, grounding.

One invocation validates the models currently served by the endpoints in `.env`
(on the ROCm box you restart vLLM with another model / --quantization fp8 and
run again with the same --run-id). Rows accumulate in runs/<run_id>/validation.csv
and the FP16-baseline gate is recomputed over the whole file each time.

  python eval/validate.py --audio data/audio/call.mp3 --reference data/reference/call.txt --run-id val-1
  python eval/validate.py --stage llm --text-source reference --reference data/reference/call.txt --run-id val-1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402

from chunker import chunk_audio  # noqa: E402
from clients import ASRClient, EndpointError, LLMClient  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from eval.metrics import (  # noqa: E402
    Figure, normalize_for_wer, score_numbers, strip_speaker_labels, term_error_rate,
)
from events import run_metadata  # noqa: E402
from extract import extract_window, load_prompt  # noqa: E402
from pipeline import RollingWindow  # noqa: E402
from schema import Metric  # noqa: E402

Segment = tuple[float, float, str]
RECALL_GATE_POINTS = 2.0
COLUMNS = [
    "timestamp", "env", "stage", "text_source", "asr_model", "asr_precision", "llm_model", "llm_precision",
    "wer", "finance_term_er", "finance_terms_in_ref", "n_gold", "n_extracted", "numeric_recall",
    "numeric_precision", "grounding_rate", "wrong_inr_figures", "windows", "json_failures", "repairs",
    "recall_drop_vs_fp16", "gate", "gpu_model", "rocm_version", "image_tag", "vllm_version", "vllm_flags",
    "asr_backend", "notes",
]


async def transcribe_all(audio: str, s: Settings, max_seconds: float | None) -> list[Segment]:
    window = RollingWindow(float("inf"))
    async with ASRClient(s) as asr:
        async for c in chunk_audio(audio, s.chunk_seconds, 0, s.chunk_overlap_seconds, max_seconds):
            res = await asr.transcribe(c.wav, prompt=window.tail(200) or None)
            window.add(c.start_s, c.end_s, res.text)
            print(f"\rASR {c.end_s / 60:6.1f} min", end="", flush=True)
    print()
    return list(window.segments)


def reference_segments(text: str, chunk_seconds: float, wpm: float = 150.0) -> list[Segment]:
    """Cut the reference into pseudo-chunks (~speaking rate) so LLM-only runs use the same windowing."""
    words = strip_speaker_labels(text).split()
    per = max(1, int(chunk_seconds * wpm / 60))
    return [(i / per * chunk_seconds, (i / per + 1) * chunk_seconds, " ".join(words[i : i + per]))
            for i in range(0, len(words), per)]


async def extract_all(segments: list[Segment], s: Settings, stride: int, parallel: int) -> tuple[list[Figure], dict]:
    prompt = load_prompt(ROOT / s.prompt_path)
    stats = {"windows": 0, "json_failures": 0, "repairs": 0, "kept": 0, "quote_fail": 0, "unit_fail": 0}
    figures: list[Figure] = []
    sem = asyncio.Semaphore(parallel)

    async def one(i: int, llm: LLMClient) -> None:
        end = segments[i][1]
        text = " ".join(t for (_, e, t) in segments[: i + 1] if e > end - s.window_seconds)
        async with sem:
            try:
                out = await extract_window(llm, prompt, text)
            except EndpointError as exc:
                print(f"\nwindow {i}: {exc}")
                stats["json_failures"] += 1
                return
        stats["windows"] += 1
        stats["repairs"] += int(out.repaired)
        if out.extraction is None:
            stats["json_failures"] += 1
            return
        stats["kept"] += len(out.extraction.metrics)
        for r in out.rejected:
            if r["error"] == "quote_not_in_window":
                stats["quote_fail"] += 1
            elif r["error"].startswith(("no_", "ambiguous")):
                stats["unit_fail"] += 1
        figures.extend(Figure.of(m) for m in out.extraction.metrics)
        print(f"\rLLM windows {stats['windows']}/{len(range(0, len(segments), stride))}", end="", flush=True)

    async with LLMClient(s) as llm:
        await asyncio.gather(*(one(i, llm) for i in range(0, len(segments), stride)))
    print()
    return figures, stats


def load_gold(path: Path) -> list[Figure]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Figure.of(Metric.model_validate({**f, "quote": "gold"})) for f in data["figures"]]


def _f(x: float | None, nd: int = 4) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def apply_gate(rows: list[dict]) -> None:
    """Recall drop > 2 points vs the same model's FP16 row fails; any wrong ₹ figure fails."""
    baselines = {}
    for r in rows:
        if r["llm_precision"].lower() in ("fp16", "bf16") and r["numeric_recall"]:
            baselines[r["llm_model"]] = float(r["numeric_recall"])
    for r in rows:
        if not r["numeric_recall"]:
            r["recall_drop_vs_fp16"], r["gate"] = "", "n/a"
            continue
        base = baselines.get(r["llm_model"])
        drop = None if base is None else (base - float(r["numeric_recall"])) * 100
        r["recall_drop_vs_fp16"] = _f(drop, 2)
        fail = (drop is not None and drop > RECALL_GATE_POINTS) or int(r["wrong_inr_figures"] or 0) > 0
        r["gate"] = "FAIL" if fail else ("PASS" if base is not None else "no_fp16_baseline")


def write_rows(path: Path, new_row: dict) -> list[dict]:
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    rows.append({k: new_row.get(k, "") for k in COLUMNS})
    apply_gate(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio")
    ap.add_argument("--reference", required=True, help="official exchange-filed transcript (plain text)")
    ap.add_argument("--gold", default=str(ROOT / "eval" / "gold_numbers.json"))
    ap.add_argument("--terms", default=str(ROOT / "eval" / "finance_terms.txt"))
    ap.add_argument("--stage", choices=["both", "asr", "llm"], default="both")
    ap.add_argument("--text-source", choices=["asr", "reference"], default="asr",
                    help="LLM input: our ASR output, or the reference text (isolates LLM accuracy)")
    ap.add_argument("--stride", type=int, default=1, help="extract every N-th chunk's window")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, help="debug only: WER on a partial call is meaningless")
    ap.add_argument("--run-id", default="validation")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    needs_asr = args.stage in ("both", "asr") or args.text_source == "asr"
    if needs_asr and not args.audio:
        ap.error("--audio is required unless --stage llm --text-source reference")
    reference = Path(args.reference).read_text(encoding="utf-8")
    meta = run_metadata(s)
    row: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": s.env, "stage": args.stage,
        "text_source": args.text_source if args.stage != "asr" else "", "notes": args.notes,
        **{k: meta.get(k, "") for k in ("gpu_model", "rocm_version", "image_tag", "vllm_version", "vllm_flags",
                                        "asr_backend")},
    }

    segments: list[Segment] = []
    if needs_asr:
        segments = await transcribe_all(args.audio, s, args.max_seconds)
    if args.stage in ("both", "asr"):
        ref_n = normalize_for_wer(strip_speaker_labels(reference))
        hyp_n = normalize_for_wer(" ".join(t for *_, t in segments))
        terms = [ln.strip() for ln in Path(args.terms).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        ter, n_terms = term_error_rate(ref_n, hyp_n, terms)
        row.update(asr_model=s.asr_model, asr_precision=s.asr_precision, wer=_f(jiwer.wer(ref_n, hyp_n)),
                   finance_term_er=_f(ter), finance_terms_in_ref=n_terms)

    if args.stage in ("both", "llm"):
        if args.text_source == "reference":
            segments = reference_segments(reference, s.chunk_seconds)
        figures, st = await extract_all(segments, s, args.stride, args.parallel)
        score = score_numbers(figures, load_gold(Path(args.gold)))
        checked = st["kept"] + st["quote_fail"] + st["unit_fail"]
        row.update(
            llm_model=s.llm_model, llm_precision=s.llm_precision, n_gold=score.n_gold,
            n_extracted=score.n_extracted, numeric_recall=_f(score.recall), numeric_precision=_f(score.precision),
            grounding_rate=_f(1 - st["quote_fail"] / checked if checked else None),
            wrong_inr_figures=len(score.wrong_inr), windows=st["windows"], json_failures=st["json_failures"],
            repairs=st["repairs"],
        )
        for e, g in score.wrong_inr:
            print(f"HARD FAILURE wrong ₹ figure: {e.metric} {e.period} extracted {e.low}-{e.high} cr, gold {g.low}-{g.high} cr")

    name = "validation.csv" if s.env == "rocm" else "validation_dev.csv"
    if s.env != "rocm":
        print("ENV != rocm: writing validation_dev.csv. Only ROCm results go in the README.")
    out = ROOT / s.runs_dir / args.run_id / name
    rows = write_rows(out, row)
    print(json.dumps(rows[-1], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
