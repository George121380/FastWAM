"""Lightweight CUDA-event step timer for per-phase training profiling.

Design notes:
- All `Event(enable_timing=True).record()` calls are host-side and async — no per-step
  `cuda.synchronize()` is performed, so steady-state training is not slowed down.
- `flush()` synchronizes only on the *last* end-event of each window; in steady-state
  training the GPU is at most a few steps behind the CPU, so this rarely blocks.
- On non-main ranks the timer is a no-op so other ranks pay zero overhead.

Two-mode operation:
- Legacy mode (`enabled_phases` is empty): only `vae_encode` is recorded (via the
  `begin_vae` / `end_vae` wrappers); `flush()` returns the original 6-key flat dict
  (`vae_encode/...`) — bit-for-bit today's behavior, drop-in compatible with the
  trainer's existing wandb payload merge.
- Profiling mode (`enabled_phases` non-empty): listed phases plus `vae_encode` are
  recorded with paired GPU events (`gpu_ms`) and host wall times (`wall_ms`).
  `flush()` returns the legacy keys plus a richer dict (`step_*`, `phases: {...}`,
  `step_unattributed_*`) intended for JSONL output. The richer dict is also flat-
  enough that the trainer can pick prefixed `profile/{phase}/{metric}` keys for
  wandb without leaking the nested `phases` blob into dashboards.

Note on `gpu_ms` semantics: it is the default-stream elapsed time between the
begin and end events bracketing a phase, NOT pure kernel-busy time. Host gaps
inside the bracketed block (e.g. `.item()` syncs) are included. The `wall_ms`
companion captures wall-clock for the same block; comparing the two reveals
host-side sync overhead.

TODO: when `gradient_accumulation_steps > 1`, every micro-step before the
sync_gradients boundary calls `begin_step` without a matching `end_step`.
The orphan-reset at `begin_step` silently overwrites `_step_pending`, so events
do not pile up indefinitely, but the CUDA Events allocated for orphan begins
become unreachable and rely on GC + driver event-pool recycling. Trainer
defaults to `gradient_accumulation_steps=1` so this is not currently exercised.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import torch


_EventPair = Tuple[torch.cuda.Event, torch.cuda.Event]

# vae_encode is always enabled regardless of enabled_phases — preserves legacy timing.
_ALWAYS_ENABLED = frozenset({"vae_encode"})


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


class StepTimer:
    def __init__(
        self,
        *,
        skip_warmup_steps: int = 0,
        enabled: bool = True,
        enabled_phases: Optional[Iterable[str]] = None,
    ) -> None:
        self.enabled = enabled
        self._skip = max(int(skip_warmup_steps), 0)
        self._step_count = 0
        self._enabled_phases = frozenset(enabled_phases or ()) | _ALWAYS_ENABLED
        # Profiling mode = caller asked for at least one non-vae phase.
        self.profiling_mode = bool(self._enabled_phases - _ALWAYS_ENABLED)

        # Whole-step accounting
        self._step_pairs: list[_EventPair] = []
        self._step_wall_starts: list[float] = []
        self._step_wall_ends: list[float] = []
        # Per-step phase containers: dicts keyed by phase name.
        self._step_phase_groups: list[dict[str, list[_EventPair]]] = []
        self._step_wall_phase_groups: list[dict[str, list[float]]] = []

        # In-flight state for the current step.
        self._step_pending: Optional[_EventPair] = None
        self._step_wall_start: Optional[float] = None
        self._phase_pending: dict[str, _EventPair] = {}
        self._wall_pending: dict[str, float] = {}
        self._current_step_phase_pairs: dict[str, list[_EventPair]] = {}
        self._current_step_wall_phases: dict[str, list[float]] = {}

    @staticmethod
    def _make_pair() -> _EventPair:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        return s, e

    def _phase_enabled(self, name: str) -> bool:
        return name in self._enabled_phases

    # ------------------------------------------------------------------
    # Public API: step boundaries
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        # Always reset transient state so leftovers can't cross steps.
        self._phase_pending = {}
        self._wall_pending = {}
        self._current_step_phase_pairs = {}
        self._current_step_wall_phases = {}
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
        self._step_phase_groups.append(self._current_step_phase_pairs)
        self._step_wall_phase_groups.append(self._current_step_wall_phases)
        self._step_pending = None
        self._step_wall_start = None
        self._current_step_phase_pairs = {}
        self._current_step_wall_phases = {}
        self._phase_pending = {}
        self._wall_pending = {}

    # ------------------------------------------------------------------
    # Public API: phases
    # ------------------------------------------------------------------

    def begin_phase(self, name: str) -> None:
        # Tie phase recording to step recording: no active step means we drop
        # the call (e.g. VAE during evaluate() never has a step around it).
        if not self.enabled or self._step_pending is None:
            return
        if not self._phase_enabled(name):
            return
        # Silent overwrite if an earlier begin without matching end exists,
        # mirroring the legacy VAE behavior.
        self._phase_pending[name] = self._make_pair()
        self._wall_pending[name] = time.perf_counter()

    def end_phase(self, name: str) -> None:
        pair = self._phase_pending.pop(name, None)
        wall_start = self._wall_pending.pop(name, None)
        if pair is None:
            return
        s, e = pair
        e.record()
        self._current_step_phase_pairs.setdefault(name, []).append((s, e))
        if wall_start is not None:
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            self._current_step_wall_phases.setdefault(name, []).append(wall_ms)

    def record_phase_wall(self, name: str, t_start: float) -> None:
        """Wall-only phase (e.g. data_load). No CUDA event."""
        if not self.enabled or self._step_pending is None:
            return
        if not self._phase_enabled(name):
            return
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        self._current_step_wall_phases.setdefault(name, []).append(wall_ms)

    # ------------------------------------------------------------------
    # Backward-compat wrappers
    # ------------------------------------------------------------------

    def begin_vae(self) -> None:
        self.begin_phase("vae_encode")

    def end_vae(self) -> None:
        self.end_phase("vae_encode")

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(
        self,
        *,
        batch_size: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> Optional[dict]:
        if not self.enabled or not self._step_pairs:
            return None
        # CUDA events on the default stream are FIFO-ordered, so syncing the
        # final step's end event guarantees every prior phase event is also
        # complete. No need to walk per-phase groups.
        self._step_pairs[-1][1].synchronize()

        step_gpu_ms = [s.elapsed_time(e) for s, e in self._step_pairs]
        step_wall_ms = [
            (b - a) * 1000.0 for a, b in zip(self._step_wall_starts, self._step_wall_ends)
        ]
        n = len(step_gpu_ms)

        # Per-phase aggregates (flattened across the window).
        # phase_gpu_per_step[name]: list[float], one entry per step (sum across
        # multiple begin/end pairs of the same phase within a step).
        phase_gpu_per_step: dict[str, list[float]] = {}
        for group in self._step_phase_groups:
            for name, pairs in group.items():
                phase_gpu_per_step.setdefault(name, []).append(
                    sum(s.elapsed_time(e) for s, e in pairs)
                )
        phase_wall_per_step: dict[str, list[float]] = {}
        for group in self._step_wall_phase_groups:
            for name, walls in group.items():
                phase_wall_per_step.setdefault(name, []).append(sum(walls))

        # Per-step sums for ratio_step_gpu / ratio_step_wall.
        sum_step_gpu = sum(step_gpu_ms)
        sum_step_wall = sum(step_wall_ms)

        # ---- Legacy keys (always emitted) ----
        vae_gpu_per_step = phase_gpu_per_step.get("vae_encode", [])
        sum_vae = sum(vae_gpu_per_step)
        n_vae = sum(1 for v in vae_gpu_per_step if v > 0)
        legacy = {
            "vae_encode/gpu_ms_mean": sum_vae / max(n_vae, 1),
            "vae_encode/step_gpu_ms_mean": sum_step_gpu / n,
            "vae_encode/step_wall_ms_mean": sum_step_wall / n,
            "vae_encode/ratio_gpu": (sum_vae / sum_step_gpu) if sum_step_gpu > 0 else 0.0,
            "vae_encode/ratio_wall": (sum_vae / sum_step_wall) if sum_step_wall > 0 else 0.0,
            "vae_encode/n_samples": n,
        }

        if not self.profiling_mode:
            self._reset_window()
            return legacy

        # ---- Profiling mode: rich dict for JSONL (legacy keys included) ----
        out: dict = dict(legacy)

        out["n_steps"] = n
        out["step_gpu_ms_mean"] = sum_step_gpu / n
        out["step_gpu_ms_p50"] = _percentile(step_gpu_ms, 0.5)
        out["step_gpu_ms_p95"] = _percentile(step_gpu_ms, 0.95)
        out["step_wall_ms_mean"] = sum_step_wall / n
        out["step_wall_ms_p50"] = _percentile(step_wall_ms, 0.5)
        out["step_wall_ms_p95"] = _percentile(step_wall_ms, 0.95)

        # Window-local throughput (wall-based).
        if sum_step_wall > 0:
            steps_per_sec_window = n / (sum_step_wall / 1000.0)
        else:
            steps_per_sec_window = 0.0
        out["steps_per_sec_window"] = steps_per_sec_window
        if batch_size is not None and world_size is not None:
            out["samples_per_sec_window"] = (
                steps_per_sec_window * float(batch_size) * float(world_size)
            )

        # Phase block.
        phases: dict[str, dict] = {}
        # Per-phase summary helper.
        def phase_summary(values: list[float]) -> dict:
            if not values:
                return {
                    "ms_mean": 0.0, "ms_p50": 0.0, "ms_p95": 0.0,
                    "n_samples": 0,
                }
            return {
                "ms_mean": sum(values) / len(values),
                "ms_p50": _percentile(values, 0.5),
                "ms_p95": _percentile(values, 0.95),
                "n_samples": len(values),
            }

        all_phase_names = set(phase_gpu_per_step) | set(phase_wall_per_step)
        for name in all_phase_names:
            entry: dict = {}
            gpu_values = phase_gpu_per_step.get(name, [])
            wall_values = phase_wall_per_step.get(name, [])
            sum_gpu = sum(gpu_values)
            sum_wall = sum(wall_values)
            if gpu_values:
                gs = phase_summary(gpu_values)
                entry["gpu_ms_mean"] = gs["ms_mean"]
                entry["gpu_ms_p50"] = gs["ms_p50"]
                entry["gpu_ms_p95"] = gs["ms_p95"]
                entry["ratio_step_gpu"] = (sum_gpu / sum_step_gpu) if sum_step_gpu > 0 else 0.0
                entry["ratio_step_wall_gpu"] = (sum_gpu / sum_step_wall) if sum_step_wall > 0 else 0.0
            if wall_values:
                ws = phase_summary(wall_values)
                entry["wall_ms_mean"] = ws["ms_mean"]
                entry["wall_ms_p50"] = ws["ms_p50"]
                entry["wall_ms_p95"] = ws["ms_p95"]
                entry["ratio_step_wall"] = (sum_wall / sum_step_wall) if sum_step_wall > 0 else 0.0
            entry["n_samples"] = max(len(gpu_values), len(wall_values))
            phases[name] = entry
        out["phases"] = phases

        # Unattributed.
        sum_attributed_gpu = sum(sum(v) for v in phase_gpu_per_step.values())
        # Wall sum should exclude data_load for unattributed_wall, since data_load
        # is accounted alongside step (it happens before step_wall_start).
        # Actually: step_wall_start is captured AFTER record_phase_wall completes
        # because data_load is recorded after begin_step. Wait, let's check the
        # call order in trainer:
        #   _step_timer.begin_step()           # captures step_wall_start
        #   t_data = perf_counter()
        #   sample = next(...)
        #   _step_timer.record_phase_wall("data_load", t_data)  # records into the active step
        # So data_load IS within step_wall (between begin_step and end_step).
        # All wall-recorded phases sum into sum_attributed_wall.
        sum_attributed_wall = sum(sum(v) for v in phase_wall_per_step.values())
        out["step_unattributed_gpu_ms_mean"] = max(0.0, (sum_step_gpu - sum_attributed_gpu) / n)
        # Wall unattributed = step_wall - sum(non-overlapping wall phases). Since
        # GPU phases are nested inside step time and wall phases bracket the same
        # blocks, simply use the wall contributions for the subtraction.
        out["step_unattributed_wall_ms_mean"] = (sum_step_wall - sum_attributed_wall) / n

        self._reset_window()
        return out

    def _reset_window(self) -> None:
        self._step_pairs.clear()
        self._step_wall_starts.clear()
        self._step_wall_ends.clear()
        self._step_phase_groups.clear()
        self._step_wall_phase_groups.clear()
