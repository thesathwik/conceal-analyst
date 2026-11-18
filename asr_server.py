"""ROCm FALLBACK ONLY. Runs on the GPU box inside the rocm/pytorch container, never on the laptop.

Serves Hugging Face transformers Whisper behind the OpenAI-compatible
/v1/audio/transcriptions route, for use if the pinned vLLM image cannot serve Whisper.
The app's clients.py talks to it exactly as it talks to vLLM.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

MODEL_ID = os.environ.get("ASR_MODEL_ID", "openai/whisper-large-v3")
DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
    os.environ.get("ASR_TORCH_DTYPE", "float16")
]
SAMPLE_RATE = 16_000

app = FastAPI(title="whisper-asr-fallback")
_asr = None
_lock = asyncio.Lock()  # one GPU, one model: serialize requests


def get_pipeline():
    global _asr
    if _asr is None:
        if not torch.cuda.is_available():  # ROCm builds of PyTorch expose HIP devices as "cuda"
            raise RuntimeError("no ROCm GPU visible to PyTorch")
        _asr = pipeline("automatic-speech-recognition", model=MODEL_ID, torch_dtype=DTYPE, device="cuda:0")
    return _asr


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health() -> dict:
    return {"ok": _asr is not None}


def _transcribe(audio: bytes, language: str | None, prompt: str | None, temperature: float) -> str:
    asr = get_pipeline()
    samples = ffmpeg_read(audio, SAMPLE_RATE)
    generate_kwargs: dict = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    if temperature > 0:
        generate_kwargs.update(do_sample=True, temperature=temperature)
    if prompt:
        generate_kwargs["prompt_ids"] = asr.tokenizer.get_prompt_ids(prompt, return_tensors="pt").to("cuda:0")
    out = asr({"raw": samples, "sampling_rate": SAMPLE_RATE}, generate_kwargs=generate_kwargs)
    text = out["text"].strip()
    if prompt and text.startswith(prompt.strip()):  # some versions echo the prompt
        text = text[len(prompt.strip()):].strip()
    return text


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
) -> dict:
    if response_format not in ("json", "text", "verbose_json"):
        raise HTTPException(400, f"unsupported response_format {response_format}")
    audio = await file.read()
    t0 = time.perf_counter()
    async with _lock:
        text = await asyncio.to_thread(_transcribe, audio, language, prompt, temperature)
    return {"text": text, "model": model, "duration_s": round(time.perf_counter() - t0, 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    get_pipeline()  # load weights before accepting traffic
    uvicorn.run(app, host=args.host, port=args.port)
