"""Thin OpenAI-compatible clients (httpx) for ASR and the extraction LLM.

Works unchanged against a hosted open-weight provider (dev) and our own vLLM
on MI300X (rocm): only base URL, key and model ID differ, and they come from `.env`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config import Settings

log = logging.getLogger(__name__)

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ASRResult:
    text: str
    latency_s: float


@dataclass
class LLMResult:
    text: str
    ttft_s: float | None
    total_s: float
    prompt_tokens: int | None
    completion_tokens: int | None

    @property
    def output_tok_per_s(self) -> float | None:
        if not self.completion_tokens or self.ttft_s is None:
            return None
        gen_time = self.total_s - self.ttft_s
        return self.completion_tokens / gen_time if gen_time > 0 else None


class EndpointError(RuntimeError):
    pass


def _backoff_delay(attempt: int, settings: Settings, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), settings.backoff_max_s)
        except ValueError:
            pass
    delay = settings.backoff_base_s * (2**attempt)
    return min(delay, settings.backoff_max_s) * (0.5 + random.random() / 2)  # jitter


def _make_http(settings: Settings, base_url: str, api_key: str) -> httpx.AsyncClient:
    if not base_url:
        raise EndpointError("base URL is empty: fill ASR_BASE_URL / LLM_BASE_URL in .env")
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
        timeout=httpx.Timeout(settings.http_timeout_s, connect=10.0),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )


class _Base:
    def __init__(self, settings: Settings, base_url: str, api_key: str) -> None:
        self.settings = settings
        self.http = _make_http(settings, base_url, api_key)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _sleep_before_retry(self, attempt: int, resp: httpx.Response | None, err: str) -> None:
        if attempt >= self.settings.max_retries:
            raise EndpointError(f"giving up after {attempt + 1} attempts: {err}")
        retry_after = resp.headers.get("retry-after") if resp is not None else None
        delay = _backoff_delay(attempt, self.settings, retry_after)
        log.warning("retrying in %.1fs (attempt %d): %s", delay, attempt + 1, err)
        await asyncio.sleep(delay)


class ASRClient(_Base):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.asr_base_url, settings.asr_api_key)

    async def transcribe(self, wav: bytes, prompt: str | None = None) -> ASRResult:
        """POST /audio/transcriptions with a WAV chunk. `prompt` = tail of previous text for context."""
        data = {"model": self.settings.asr_model, "response_format": "json", "temperature": "0"}
        if self.settings.asr_language:
            data["language"] = self.settings.asr_language
        if prompt:
            data["prompt"] = prompt
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                resp = await self.http.post(
                    "audio/transcriptions", data=data, files={"file": ("chunk.wav", wav, "audio/wav")}
                )
            except httpx.TransportError as exc:
                await self._sleep_before_retry(attempt, None, repr(exc))
                attempt += 1
                continue
            if resp.status_code in RETRYABLE:
                await self._sleep_before_retry(attempt, resp, f"HTTP {resp.status_code}: {resp.text[:200]}")
                attempt += 1
                continue
            if resp.status_code >= 400:
                raise EndpointError(f"ASR HTTP {resp.status_code}: {resp.text[:500]}")
            text = resp.json().get("text", "")
            return ASRResult(text=text.strip(), latency_s=time.perf_counter() - t0)


class LLMClient(_Base):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.llm_base_url, settings.llm_api_key)

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.settings.llm_json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    async def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        """Streaming chat completion so we can measure TTFT. Retries 429/5xx before the stream starts."""
        body = self._body(messages)
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                async with self.http.stream("POST", "chat/completions", json=body) as resp:
                    if resp.status_code in RETRYABLE:
                        err = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                        await self._sleep_before_retry(attempt, resp, err)
                        attempt += 1
                        continue
                    if resp.status_code >= 400:
                        raise EndpointError(f"LLM HTTP {resp.status_code}: {(await resp.aread())[:500]!r}")
                    return await self._read_stream(resp, t0)
            except httpx.TransportError as exc:
                await self._sleep_before_retry(attempt, None, repr(exc))
                attempt += 1

    @staticmethod
    async def _read_stream(resp: httpx.Response, t0: float) -> LLMResult:
        parts: list[str] = []
        ttft: float | None = None
        usage: dict[str, Any] = {}
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
        return LLMResult(
            text="".join(parts),
            ttft_s=ttft,
            total_s=time.perf_counter() - t0,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
