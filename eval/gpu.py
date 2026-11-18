"""Sample AMD GPU utilisation once a second via amd-smi (or rocm-smi) during a benchmark.

Run bench.py on the GPU host, or pass --gpu-ssh user@host so samples are taken remotely.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import shutil
import time
from typing import Any

COMMANDS = [
    ["amd-smi", "metric", "--usage", "--json"],
    ["rocm-smi", "--showuse", "--json"],
]
_UTIL_KEY = re.compile(r"gfx_activity|gpu use \(%\)|gfx_busy|gpu_util", re.IGNORECASE)


def extract_utilisation(data: Any) -> list[float]:
    """Walk amd-smi / rocm-smi JSON and return one utilisation % per GPU found."""
    found: list[float] = []

    def as_number(v: Any) -> float | None:
        if isinstance(v, dict):
            v = v.get("value")
        try:
            return float(str(v).strip().rstrip("%"))
        except (TypeError, ValueError):
            return None

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if _UTIL_KEY.search(str(k)) and (n := as_number(v)) is not None:
                    found.append(n)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return found


class GpuSampler:
    def __init__(self, ssh: str | None = None, interval_s: float = 1.0) -> None:
        self.ssh = ssh
        self.interval_s = interval_s
        self.samples: list[tuple[float, list[float]]] = []
        self.command: list[str] | None = None
        self._task: asyncio.Task | None = None

    def _wrap(self, cmd: list[str]) -> list[str]:
        return ["ssh", self.ssh, " ".join(map(shlex.quote, cmd))] if self.ssh else cmd

    async def _run(self, cmd: list[str]) -> list[float] | None:
        proc = await asyncio.create_subprocess_exec(
            *self._wrap(cmd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        try:
            return extract_utilisation(json.loads(out)) or None
        except json.JSONDecodeError:
            return None

    async def detect(self) -> bool:
        for cmd in COMMANDS:
            if self.ssh is None and shutil.which(cmd[0]) is None:
                continue
            if await self._run(cmd) is not None:
                self.command = cmd
                return True
        return False

    async def _loop(self) -> None:
        assert self.command is not None
        while True:
            t = time.monotonic()
            util = await self._run(self.command)
            if util is not None:
                self.samples.append((t, util))
            await asyncio.sleep(max(0.0, self.interval_s - (time.monotonic() - t)))

    async def start(self) -> bool:
        self.samples = []
        if self.command is None and not await self.detect():
            return False
        self._task = asyncio.create_task(self._loop())
        return True

    async def stop(self) -> dict[str, float | None]:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        per_sample = [sum(u) / len(u) for _, u in self.samples if u]  # mean across GPUs
        peak = [max(u) for _, u in self.samples if u]
        return {
            "gpu_util_mean": sum(per_sample) / len(per_sample) if per_sample else None,
            "gpu_util_max": max(peak) if peak else None,
            "gpu_samples": len(per_sample),
        }
