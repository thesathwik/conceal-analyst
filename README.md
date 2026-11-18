# Concall Analyst

Near-real-time analysis of Indian listed companies' earnings calls. While a recorded call replays at live
speed, it transcribes 10-second chunks, extracts financial guidance (revenue, EBITDA margin, capex, order book,
debt in ₹ crore / % / bps) from a rolling 150-second window, compares it with last quarter's guidance, and raises
**CUT / RAISE / NEW / WITHDRAWN** alerts on a live dashboard.

The app is the vehicle. The main deliverables are **model validation, a ROCm deployment on AMD MI300X, inference
tuning, and performance benchmarks**, all with open-weight models served through vLLM.

<!-- demo GIF: docs/demo.gif -->

## Architecture

```
data/audio/<call>.mp3
   │  chunker.py: ffmpeg → 16 kHz mono, 10 s chunks, released on the audio clock (--speed N)
   ▼
POST {ASR_BASE_URL}/audio/transcriptions        Whisper large-v3 (vLLM on MI300X | hosted in dev)
   │  pipeline.py: rolling 150 s window, boundary de-dupe
   ▼
POST {LLM_BASE_URL}/chat/completions            Qwen2.5-7B-Instruct, JSON mode, streamed (TTFT)
   │  extract.py: schema validation → repair once → quote grounding → normalize.py re-derives ₹ figures from the quote
   ▼
compare.py: diff vs data/last_quarter.json → alerts (de-duplicated across overlapping windows)
   ▼
app.py (Streamlit) tails runs/<run_id>/events.jsonl   ← every stage logs a timestamped event
```

Design choices:

- **ASR of chunk N+1 overlaps extraction of window N** (asyncio). When extraction falls behind, the worker skips stale
  windows and takes the newest one, so latency stays bounded instead of queueing.
- **The LLM's arithmetic is never trusted.** Every figure must come with a verbatim quote; the quote must appear in
  the window, and the number is re-parsed from the quote (crore / lakh / lakh crore / billion / million, Indian digit
  grouping, ranges). Ungrounded figures are dropped and logged.
- **One code path, two environments.** `ENV=dev` uses a hosted provider serving the same open-weight models;
  `ENV=rocm` uses our vLLM on MI300X. Switching is a `.env` change.

## Quick start (laptop: app code only, no model downloads)

```bash
brew install ffmpeg            # or apt install ffmpeg
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # provider URLs, keys and model IDs
# put a recorded call in data/audio/ and prior guidance in data/last_quarter.json

python pipeline.py --audio data/audio/<call>.mp3 --speed 4
streamlit run app.py
pytest -q
```

ROCm bring-up, the validation matrix, and tuning sweeps: [deploy/README.md](deploy/README.md).

## Results

> All numbers below must come from `ENV=rocm` (vLLM on AMD MI300X). `bench.py` refuses to run otherwise,
> and `validate.py` writes dev runs to a separate `validation_dev.csv`. **Pending the ROCm runs.**

### Environment

| | |
|---|---|
| GPU | AMD Instinct MI300X (192 GB) |
| ROCm | _tbd_ |
| Image | `rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0` |
| vLLM | _tbd_ |
| Flags | _tbd_ |

### ASR validation (WER vs. exchange-filed transcript)

| Model | Precision | WER | Finance-term error | License |
|---|---|---|---|---|
| openai/whisper-large-v3 | FP16 | | | MIT |
| openai/whisper-large-v3-turbo | FP16 | | | MIT |

### Extraction validation (vs. hand-labelled gold figures)

| Model | Precision | Recall | Precision | Grounding | Wrong ₹ figures | Gate | License |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | FP16 | | | | | | Apache 2.0 |
| Qwen2.5-7B-Instruct | FP8 | | | | | | Apache 2.0 |
| Mistral-7B-Instruct-v0.3 | FP16 | | | | | | Apache 2.0 |
| Mistral-7B-Instruct-v0.3 | FP8 | | | | | | Apache 2.0 |
| Llama-3.1-8B-Instruct | FP16 | | | | | | Llama 3.1 Community License (not OSI) |
| Llama-3.1-8B-Instruct | FP8 | | | | | | Llama 3.1 Community License (not OSI) |

Gate: a configuration fails if its numeric recall is more than 2 points below the same model's FP16 run, or if
it extracts **any wrong ₹ figure** (hard failure).

### Latency and concurrency (`--speed 1`)

| Concurrent calls | Alert latency p50 | p95 | p99 | ASR p95 / chunk | LLM TTFT p95 | GPU util (mean) |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 5 | | | | | | |
| 20 | | | | | | |

SLO: p95 alert latency < 10 s at 5 concurrent calls. Alert latency = time from the end of the audio chunk
(audio clock) to the alert being emitted.

### Tuning sweep

| Setting | p95 alert latency @ 20 | Output tok/s | Notes |
|---|---|---|---|
| baseline | | | |
| `--max-num-seqs` 32 / 128 | | | |
| `--max-num-batched-tokens` 4k / 16k | | | |
| chunked prefill off | | | |
| `PYTORCH_TUNABLEOP_ENABLED=1` | | | |
| FP8 weights + KV cache | | | |

## Lessons learned

_To be written after the ROCm runs._

## Data policy

Audio, exchange transcripts, model weights and `.env` are never committed. Only extracted outputs and metrics
appear here.
