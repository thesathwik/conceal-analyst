"""ffmpeg-based chunking of a recorded call, replayed at real-time pace (or N× faster).

ffmpeg decodes the file to 16 kHz mono PCM on a pipe; we cut it into
CHUNK_SECONDS WAV chunks and release each one when the audio clock reaches the
chunk's end (divided by --speed), as a live stream would.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import math
import shutil
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # s16le
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH


@dataclass
class Chunk:
    index: int
    start_s: float  # audio clock, including any overlap
    end_s: float  # audio clock
    wav: bytes
    released_at: float  # time.monotonic() when the chunk became "available"


def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH (macOS: `brew install ffmpeg`)")
    return exe


async def _read_exactly(stream: asyncio.StreamReader, n: int) -> bytes:
    try:
        return await stream.readexactly(n)
    except asyncio.IncompleteReadError as exc:
        return exc.partial


async def chunk_audio(
    path: str | Path,
    chunk_seconds: float = 10.0,
    speed: float = 1.0,
    overlap_seconds: float = 0.0,
    max_seconds: float | None = None,
    start_at: float | None = None,
) -> AsyncIterator[Chunk]:
    """Yield chunks paced to the audio clock. speed <= 0 (or inf) means no pacing."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    cmd = [_require_ffmpeg(), "-nostdin", "-loglevel", "error", "-i", str(path)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    assert proc.stdout is not None
    chunk_bytes = int(chunk_seconds * BYTES_PER_SECOND) // SAMPLE_WIDTH * SAMPLE_WIDTH
    overlap_bytes = int(overlap_seconds * BYTES_PER_SECOND) // SAMPLE_WIDTH * SAMPLE_WIDTH
    paced = speed > 0 and math.isfinite(speed)
    t0 = start_at if start_at is not None else time.monotonic()
    tail = b""
    index = 0
    audio_pos = 0.0
    try:
        while True:
            pcm = await _read_exactly(proc.stdout, chunk_bytes)
            if not pcm:
                break
            end_s = audio_pos + len(pcm) / BYTES_PER_SECOND
            if paced:
                delay = t0 + end_s / speed - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            start_s = audio_pos - len(tail) / BYTES_PER_SECOND
            yield Chunk(index, start_s, end_s, pcm_to_wav(tail + pcm), time.monotonic())
            tail = pcm[-overlap_bytes:] if overlap_bytes else b""
            audio_pos = end_s
            index += 1
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        if proc.returncode not in (0, -9) and proc.stderr is not None:
            err = (await proc.stderr.read()).decode(errors="replace").strip()
            if err:
                raise RuntimeError(f"ffmpeg failed: {err}")


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Dry-run the chunker: print chunk timings.")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--chunk-seconds", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=0, help="0 = as fast as possible")
    ap.add_argument("--max-seconds", type=float)
    args = ap.parse_args()
    t0 = time.monotonic()
    async for c in chunk_audio(args.audio, args.chunk_seconds, args.speed, max_seconds=args.max_seconds, start_at=t0):
        print(f"chunk {c.index:4d}  audio {c.start_s:7.1f}-{c.end_s:7.1f}s  "
              f"released +{c.released_at - t0:6.2f}s  {len(c.wav) / 1024:.0f} KiB")


if __name__ == "__main__":
    asyncio.run(_main())
