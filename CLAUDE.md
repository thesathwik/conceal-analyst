# CLAUDE.md — Concall Analyst

Guidance for Claude Code (and any contributor) working in this repository.

## 1. What this project is

**Concall Analyst** is a near-real-time analyzer for Indian listed companies' earnings calls ("concalls").
While a call plays, it:

1. Transcribes the audio in ~10-second chunks (speech-to-text).
2. Extracts financial guidance from a rolling transcript window — revenue, EBITDA margin, capex, order book, debt — in Indian units (₹ crore, lakh, bps).
3. Compares extracted figures against last quarter's guidance and raises alerts (cut / raise / new / withdrawn).
4. Shows a live dashboard: transcript, extracted metrics table, and alerts.

**Why it exists:** it is a portfolio project for an **AI/MLOps engineer (inference layer)** role whose key skills are:
**validate models, build ROCm-compatible images, tune inference, and test performance — using open-source models.**
The app is the vehicle; the validation, ROCm deployment, tuning, and benchmark results are the main deliverables.

## 2. Hard constraints (do not violate)

- **Open-weight models only.** Never use proprietary models (GPT, Claude, Gemini, etc.) anywhere in the code path, tests, or evals.
- **No local model downloads or local inference.** The developer's laptop runs only the app code (Python, ffmpeg, Streamlit). Do not add Ollama, faster-whisper, `transformers` model loading, or any weight download to the local setup.
- **All model access goes through OpenAI-*compatible* HTTP endpoints.** Two environments:
  - `dev`: a hosted provider that serves the *same open-weight models* (e.g. Groq, DeepInfra, Together, Hugging Face Inference Providers). For building and debugging only.
  - `rocm`: our own vLLM on AMD MI300X. **All validation and benchmark results reported in the README must come from this environment.**
  Switching environments = changing `.env`, never code changes. Never hard-code a provider.
- **Recorded audio replayed as live** for the MVP. Do not build live webcast capture yet.
- **Do not commit** audio files, exchange transcripts, model weights, or `.env`. Do not republish companies' call content; only our extracted outputs and metrics go in the README.
- Keep the MVP small. If a feature isn't in Section 4 "In scope", don't build it without being asked.

## 3. Model stack

| Stage | Default | Comparison candidates | License |
|---|---|---|---|
| Speech-to-text | `openai/whisper-large-v3` | `openai/whisper-large-v3-turbo`, `distil-whisper/distil-large-v3` | MIT |
| Extraction LLM | Qwen Instruct 7B (Qwen2.5-7B-Instruct or newer Qwen release) | `mistralai/Mistral-7B-Instruct-v0.3`, `meta-llama/Llama-3.1-8B-Instruct` | Apache 2.0 / Apache 2.0 / Llama Community License (not OSI) |
| Embeddings (post-MVP) | `BAAI/bge-m3` | `nomic-embed-text` | MIT / Apache 2.0 |

Record each model's license in the README results section.

## 4. MVP scope

**In scope**
- Chunked playback of one recorded concall MP3 at real-time speed.
- Transcription per chunk → rolling 2–3 minute transcript window.
- LLM extraction to strict JSON (schema in Section 7).
- Comparison against `data/last_quarter.json` → alerts.
- Streamlit dashboard that updates live.
- Validation: WER vs. the official exchange-filed transcript + numeric extraction accuracy.
- Benchmarks: latency, throughput, and concurrency (1 / 5 / 20 simultaneous replayed calls).
- ROCm deployment via docker-compose with vLLM on AMD MI300X.

**Out of scope (for now)**
Live webcast capture, speaker diarization, Telegram/WhatsApp alerts, RAG over past transcripts, Kubernetes, auth, multi-user.

## 5. Architecture

```
data/audio/<call>.mp3
        │
        ▼
chunker.py ── 10s chunks, real-time pacing (or --speed N for faster replay)
        │
        ▼
POST {ASR_BASE_URL}/audio/transcriptions           (hosted open-weight provider in dev | vLLM on ROCm)
        │
        ▼
pipeline.py ── rolling window (default 150s), dedupe overlapping text
        │
        ▼
POST {LLM_BASE_URL}/chat/completions                (hosted open-weight provider in dev | vLLM on ROCm)
        │      prompt: prompts/extract.txt, JSON output only
        ▼
compare.py ── diff vs data/last_quarter.json → alerts
        │
        ▼
app.py (Streamlit) ── transcript | metrics table | alerts
        │
        └── every event also appended to runs/<run_id>/events.jsonl (timestamps for benchmarking)
```

## 6. Repository layout

```
concall-analyst/
├── CLAUDE.md
├── README.md                 # demo GIF, architecture, results tables, lessons learned
├── pyproject.toml            # or requirements.txt
├── .env.example
├── app.py                    # Streamlit UI
├── pipeline.py               # orchestration: chunks → ASR → window → LLM → compare
├── chunker.py                # ffmpeg-based chunking + real-time playback
├── compare.py                # guidance diffing + alert rules
├── clients.py                # thin OpenAI-compatible clients for ASR and LLM (httpx)
├── schema.py                 # pydantic models for extraction output and alerts
├── normalize.py              # Indian number/unit normalization (crore, lakh, bps, %)
├── asr_server.py             # ROCm FALLBACK ONLY (runs on the GPU box, never locally): HF transformers Whisper behind /v1/audio/transcriptions
├── prompts/
│   └── extract.txt
├── data/
│   ├── last_quarter.json     # hand-written prior guidance for the chosen company
│   ├── audio/                # gitignored
│   └── reference/            # gitignored: official transcript text for WER + gold numbers
├── eval/
│   ├── validate.py           # WER + numeric accuracy, per model × precision
│   ├── bench.py              # latency/throughput/concurrency
│   └── gold_numbers.json     # hand-labeled figures from the reference transcript
├── deploy/
│   ├── docker-compose.rocm.yml
│   └── README.md             # GPU rental + bring-up steps
├── runs/                     # gitignored: per-run events.jsonl and result CSVs
└── tests/
    ├── test_normalize.py
    ├── test_compare.py
    └── test_schema.py
```

## 7. Data contracts

### Extraction output (one object per LLM call)
```json
{
  "metrics": [
    {
      "metric": "ebitda_margin",
      "value_low": 16.0,
      "value_high": 17.0,
      "unit": "percent",
      "period": "FY27",
      "type": "guidance",
      "quote": "we expect EBITDA margins of 16 to 17 percent for FY27"
    }
  ],
  "tone_flags": [
    {"flag": "hedging", "quote": "temporary headwinds that we believe will normalize"}
  ]
}
```
- `metric` ∈ `revenue_growth`, `revenue`, `ebitda_margin`, `pat`, `capex`, `order_book`, `net_debt`, `other`.
- `unit` ∈ `percent`, `inr_crore`, `bps`, `count`. Normalize lakh/crore/billion to `inr_crore` in `normalize.py` — never trust the LLM's arithmetic.
- `type` ∈ `guidance`, `actual`.
- `quote` must be a verbatim span from the window (used to verify and to reject hallucinated figures).
- Single values: set `value_low == value_high`.
- Invalid JSON → retry once with a repair prompt, then drop and log.

### `data/last_quarter.json`
Same `metrics` shape, `type: "guidance"` only.

### Alert rules (`compare.py`)
- **CUT**: new midpoint < prior midpoint by > 5% relative (or > 50 bps for margins).
- **RAISE**: the inverse.
- **NEW**: metric/period not in prior guidance.
- **REITERATED**: within thresholds (shown in UI, not alerted).
- Make thresholds config values, not literals.

## 8. Environments

### Dev (Day 1) — laptop runs app code only; models are hosted
- Laptop needs only: Python 3.11+, ffmpeg, the repo's Python deps. No model weights, no local inference servers.
- ASR and LLM: a hosted provider serving open-weight models through an OpenAI-compatible API. Pick one that lists Whisper large-v3 (or turbo) and a Qwen / Llama / Mistral instruct model; the ASR and LLM providers may differ.
- Model IDs differ per provider (e.g. `whisper-large-v3` vs `openai/whisper-large-v3`) — keep them in `.env` only.
- Respect provider rate limits: add retry with exponential backoff on 429s in `clients.py`.
- Dev-environment latency numbers are **not** results; `bench.py` must refuse to write `bench.csv` unless `ENV=rocm`.
- Use `--speed 4` playback to iterate quickly.

### ROCm (Day 2) — AMD MI300X, vLLM
- Base image: official `rocm/vllm` (pin an exact tag in the compose file; record it in results).
- Containers need: `--device /dev/kfd --device /dev/dri --group-add video --ipc=host --security-opt seccomp=unconfined`.
- One service for Whisper, one for the LLM. Confirm the pinned vLLM version supports Whisper via `/v1/audio/transcriptions` before relying on it; if it doesn't, fall back to a ROCm PyTorch image running Hugging Face `transformers` Whisper behind `asr_server.py`.
- FP8 runs: `--quantization fp8`, optionally `--kv-cache-dtype fp8`.
- Tuning knobs to sweep and record: `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, chunked prefill on/off, `PYTORCH_TUNABLEOP_ENABLED=1`.
- Bring-up smoke checks (script them): `rocm-smi` or `amd-smi` lists the GPU, `/v1/models` responds on both services, one transcription + one extraction succeed.

### `.env.example`
```
ENV=dev                                   # dev | rocm
# dev: hosted open-weight provider (OpenAI-compatible)
ASR_BASE_URL=https://<provider>/openai/v1
ASR_API_KEY=
ASR_MODEL=whisper-large-v3
LLM_BASE_URL=https://<provider>/v1
LLM_API_KEY=
LLM_MODEL=<provider's Qwen 7B/8B instruct ID>
# rocm: our vLLM on MI300X (uncomment on Day 2)
# ASR_BASE_URL=http://<gpu-host>:8001/v1
# LLM_BASE_URL=http://<gpu-host>:8000/v1
# ASR_API_KEY=EMPTY
# LLM_API_KEY=EMPTY
CHUNK_SECONDS=10
WINDOW_SECONDS=150
PLAYBACK_SPEED=1
```

## 9. Validation (`eval/validate.py`)

Run per **model × precision** (FP16, FP8) and write `runs/<run_id>/validation.csv`.

- **ASR WER** vs. the reference transcript using `jiwer`, after normalization (lowercase, strip punctuation, unify number formats). Also report WER on a finance-term list (EBITDA, crore, bps, company/product names).
- **Numeric extraction accuracy** vs. `eval/gold_numbers.json`:
  - recall = gold figures found; precision = extracted figures that match gold;
  - a match = same metric + period, value within 0.5% (or exact for bps).
- **Quote grounding rate** — share of extracted items whose `quote` appears in the transcript window.
- **Gate:** flag any configuration whose numeric recall drops > 2 points vs. the FP16 baseline of the same model. A wrong ₹ figure is a hard failure; call it out in the README.

## 10. Benchmarks (`eval/bench.py`)

Write `runs/<run_id>/bench.csv`. Timestamps come from `events.jsonl`.

- **Alert latency**: chunk end time (audio clock) → alert emitted. Report p50 / p95 / p99.
- **ASR latency per chunk** and **LLM latency per window**: TTFT, total, output tokens/s.
- **Concurrency**: replay 1, 5, 20 calls simultaneously (`--speed 1`); record p95 alert latency and GPU utilization (sample `amd-smi`/`rocm-smi` every second).
- Record environment with every run: GPU model, ROCm version, image tag, vLLM version, model ID, precision, flags.
- **Target SLO for the README:** p95 alert latency < 10 s at 5 concurrent calls.

## 11. Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# fill in provider URLs, keys and model IDs in .env (no local model downloads)

# run
python pipeline.py --audio data/audio/<call>.mp3 --speed 4
streamlit run app.py

# eval
python eval/validate.py --audio data/audio/<call>.mp3 --reference data/reference/<call>.txt
python eval/bench.py --audio data/audio/<call>.mp3 --concurrency 1 5 20

# ROCm
docker compose -f deploy/docker-compose.rocm.yml up -d

# tests
pytest -q
```

## 12. Coding conventions

- Python 3.11+, type hints everywhere, `pydantic` for all model I/O, `httpx` for HTTP.
- Async in the pipeline (`asyncio`) so ASR of chunk N+1 overlaps LLM extraction of window N.
- No framework sprawl: no LangChain/LlamaIndex for the MVP.
- Config via env vars (`pydantic-settings`); no hard-coded URLs or model names.
- Log structured JSON events to `runs/<run_id>/events.jsonl` (`chunk_start`, `asr_done`, `llm_done`, `alert`), each with a monotonic timestamp and audio-clock offset.
- Unit tests for `normalize.py` and `compare.py` are required (e.g. "₹1,250 crore", "12.5 billion rupees", "about 450 crores", "150 basis points").
- Keep functions small; no file over ~300 lines.

## 13. Build plan

**Day 1 (dev, hosted open-weight endpoints):** chunker → OpenAI-compatible clients with retry → pipeline with rolling window → extraction prompt + schema + normalization → compare + alerts → Streamlit UI → tests. Done when one recorded call runs end to end at `--speed 4` and produces sensible alerts.

**Day 2 (ROCm):** compose bring-up + smoke checks → validation across Qwen / Mistral / Llama × FP16 / FP8 and Whisper large-v3 vs. turbo → benchmarks at 1 / 5 / 20 concurrency → README with results tables, demo GIF, licenses, and 3 lessons learned.

## 14. Definition of done (MVP)

- [ ] One real concall runs end to end and the dashboard updates live.
- [ ] Only open-weight models, all behind OpenAI-compatible endpoints; nothing downloaded or run locally.
- [ ] Runs unchanged against the hosted dev provider and against our MI300X (vLLM + ROCm) by swapping `.env`.
- [ ] Every reported validation/benchmark number comes from the `rocm` environment.
- [ ] `validation.csv` covers ≥ 3 LLMs × 2 precisions and 2 Whisper variants.
- [ ] `bench.csv` covers 1 / 5 / 20 concurrent calls with p50/p95/p99.
- [ ] README states the environment (GPU, ROCm, image tag, vLLM version), results, licenses, and lessons learned.
- [ ] `pytest` passes.