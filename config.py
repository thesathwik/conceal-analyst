"""Runtime configuration, loaded from environment variables / `.env`.

Switching between the hosted dev provider and our ROCm vLLM box is a `.env`
change only: nothing in the code names a provider, URL or model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "rocm"] = "dev"

    # Speech-to-text endpoint (OpenAI-compatible /audio/transcriptions)
    asr_base_url: str = ""
    asr_api_key: str = "EMPTY"
    asr_model: str = ""
    asr_language: str = "en"

    # Extraction LLM endpoint (OpenAI-compatible /chat/completions)
    llm_base_url: str = ""
    llm_api_key: str = "EMPTY"
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_json_mode: bool = True  # send response_format={"type": "json_object"}

    # Precision labels for validation/bench bookkeeping (what the server runs; not sent anywhere)
    asr_precision: str = "fp16"
    llm_precision: str = "fp16"

    # Playback / windowing
    chunk_seconds: float = 10.0
    chunk_overlap_seconds: float = 0.0
    window_seconds: float = 150.0
    playback_speed: float = 1.0

    # Alert thresholds (Section 7 of CLAUDE.md)
    cut_raise_relative_pct: float = 5.0
    margin_threshold_bps: float = 50.0

    # HTTP behaviour
    http_timeout_s: float = 120.0
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0

    # Paths
    runs_dir: str = "runs"
    prompt_path: str = "prompts/extract.txt"
    last_quarter_path: str = "data/last_quarter.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
