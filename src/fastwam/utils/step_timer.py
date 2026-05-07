"""Lightweight CUDA-event step timer for measuring VAE encode share of training time.

Design notes:
- All `Event(enable_timing=True).record()` calls are host-side and async — no per-step
  `cuda.synchronize()` is performed, so steady-state training is not slowed down.
- `flush()` synchronizes only on the *last* end-event of each list; in steady-state
  training the GPU is at most a few steps behind the CPU, so this rarely blocks.
- On non-main ranks the timer is a no-op so other ranks pay zero overhead.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch


_EventPair = Tuple[torch.cuda.Event, torch.cuda.Event]


class StepTimer:
    def __init__(self, *, skip_warmup_steps: int = 0, enabled: bool = True) -> None:
        self.enabled = enabled
        self._skip = max(int(skip_warmup_steps), 0)
        self._step_count = 0

        self._step_pairs: list[_EventPair] = []
        self._step_wall_starts: list[float] = []
        self._step_wall_ends: list[float] = []
        self._step_vae_groups: list[list[_EventPair]] = []

        self._step_pending: Optional[_EventPair] = None
        self._step_wall_start: Optional[float] = None
        self._vae_pending: Optional[_EventPair] = None
        self._current_step_vae_pairs: list[_EventPair] = []

    @staticmethod
    def _make_pair() -> _EventPair:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        return s, e

    def begin_step(self) -> None:
        # Always reset transient state, even if disabled, so leftovers can't cross steps.
        self._vae_pending = None
        self._current_step_vae_pairs = []
        self._step_pending = None
        self._step_wall_start = None
        if not self.enabled or self._step_count < self._skip:
            return
        self._step_pending = self._make_pair()
        self._step_wall_start = time.perf_counter()

    def end_step(self) -> None:
        self._step_count += 1
        if self._step_pending is None:
            return
        s, e = self._step_pending
        e.record()
        self._step_pairs.append((s, e))
        self._step_wall_starts.append(self._step_wall_start)  # type: ignore[arg-type]
        self._step_wall_ends.append(time.perf_counter())
        self._step_vae_groups.append(self._current_step_vae_pairs)
        self._step_pending = None
        self._step_wall_start = None
        self._current_step_vae_pairs = []
        self._vae_pending = None

    def begin_vae(self) -> None:
        # Tie VAE recording to step recording: if no step is active, skip — this
        # cleanly excludes VAE calls that happen during evaluate().
        if not self.enabled or self._step_pending is None:
            return
        self._vae_pending = self._make_pair()

    def end_vae(self) -> None:
        if self._vae_pending is None:
            return
        s, e = self._vae_pending
        e.record()
        self._current_step_vae_pairs.append((s, e))
        self._vae_pending = None

    def flush(self) -> Optional[dict]:
        if not self.enabled or not self._step_pairs:
            return None
        # Synchronize on last end-events; in steady-state the GPU is already past these.
        self._step_pairs[-1][1].synchronize()
        for group in reversed(self._step_vae_groups):
            if group:
                group[-1][1].synchronize()
                break

        step_gpu_ms = [s.elapsed_time(e) for s, e in self._step_pairs]
        step_wall_ms = [
            (b - a) * 1000.0 for a, b in zip(self._step_wall_starts, self._step_wall_ends)
        ]
        vae_per_step_ms = [
            sum(s.elapsed_time(e) for s, e in group) for group in self._step_vae_groups
        ]

        n = len(step_gpu_ms)
        sum_step_gpu = sum(step_gpu_ms)
        sum_step_wall = sum(step_wall_ms)
        sum_vae = sum(vae_per_step_ms)
        n_vae = sum(1 for v in vae_per_step_ms if v > 0)

        out = {
            "vae_encode/gpu_ms_mean": sum_vae / max(n_vae, 1),
            "vae_encode/step_gpu_ms_mean": sum_step_gpu / n,
            "vae_encode/step_wall_ms_mean": sum_step_wall / n,
            "vae_encode/ratio_gpu": (sum_vae / sum_step_gpu) if sum_step_gpu > 0 else 0.0,
            "vae_encode/ratio_wall": (sum_vae / sum_step_wall) if sum_step_wall > 0 else 0.0,
            "vae_encode/n_samples": n,
        }

        self._step_pairs.clear()
        self._step_wall_starts.clear()
        self._step_wall_ends.clear()
        self._step_vae_groups.clear()
        return out
