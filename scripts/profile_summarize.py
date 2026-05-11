#!/usr/bin/env python
"""scripts/profile_summarize.py

Aggregate timing.jsonl files produced by the profiling sweep into a comparison
table. Drops records with is_warmup=true. Outputs markdown or CSV.

Usage:
    python scripts/profile_summarize.py [--root runs/profile] [--format md|csv]

The sanity flags are warnings, not gates — the user reads them and decides.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import statistics
import sys


PHASES = ["data_load", "vae_encode", "dit_forward", "loss_compute", "backward", "optimizer"]


def load_run(jsonl_path: str, step_range: tuple[int, int] | None = (21, 30)) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records = [r for r in records if not r.get("is_warmup", False)]
    if step_range is None:
        return records
    selected = [r for r in records if tuple(r.get("step_range", ())) == step_range]
    return selected


def aggregate_run(records: list[dict]) -> dict:
    if not records:
        return {}
    keys_scalar = [
        "step_gpu_ms_mean", "step_wall_ms_mean",
        "steps_per_sec_window", "samples_per_sec_window",
        "step_unattributed_gpu_ms_mean", "step_unattributed_wall_ms_mean",
    ]
    out = {}
    for k in keys_scalar:
        vals = [r.get(k, 0.0) for r in records]
        out[k] = statistics.fmean(vals) if vals else 0.0

    # Phase-level
    phase_metrics: dict[str, dict[str, list[float]]] = {p: {} for p in PHASES}
    for rec in records:
        phases = rec.get("phases", {}) or {}
        for p in PHASES:
            entry = phases.get(p, {}) or {}
            for mname, mval in entry.items():
                if isinstance(mval, (int, float)):
                    phase_metrics[p].setdefault(mname, []).append(float(mval))

    out["phases"] = {
        p: {k: statistics.fmean(v) for k, v in metrics.items() if v}
        for p, metrics in phase_metrics.items()
    }

    # Carry metadata from the first record
    first = records[0]
    for k in (
        "batch_size", "world_size", "n_nodes", "num_workers",
        "vae_encode_cuda_graph", "vae_encode_batch_parallel", "run_id",
        "startup_seconds", "startup_after_steps",
    ):
        if k in first:
            out[k] = first[k]
    out["n_records_used"] = len(records)
    out["step_ranges_used"] = [
        "-".join(str(x) for x in r.get("step_range", []))
        for r in records
        if r.get("step_range")
    ]
    return out


def fmt_pct(num: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * num / denom


def build_table_row(agg: dict) -> dict:
    bs = agg.get("batch_size", "?")
    n_gpu = agg.get("world_size", "?")
    n_nodes = agg.get("n_nodes", 1)
    n_workers = agg.get("num_workers", "?")
    graph = "1" if agg.get("vae_encode_cuda_graph") else "0"
    bp = "1" if agg.get("vae_encode_batch_parallel") else "0"
    sps = agg.get("steps_per_sec_window", 0.0)
    samps = agg.get("samples_per_sec_window", 0.0)
    step_wall = agg.get("step_wall_ms_mean", 0.0)
    step_unattrib_wall = agg.get("step_unattributed_wall_ms_mean", 0.0)

    phases = agg.get("phases", {})

    def gpu_pct(name):
        gpu_mean = phases.get(name, {}).get("gpu_ms_mean", 0.0)
        return fmt_pct(gpu_mean, step_wall) if step_wall > 0 else 0.0

    def wall_pct(name):
        wall_mean = phases.get(name, {}).get("wall_ms_mean", 0.0)
        return fmt_pct(wall_mean, step_wall) if step_wall > 0 else 0.0

    data_load_wall = phases.get("data_load", {}).get("wall_ms_mean", 0.0)
    data_load_max_across = phases.get("data_load", {}).get("wall_ms_max_across_ranks", float("nan"))
    vae_gap = (
        phases.get("vae_encode", {}).get("wall_ms_mean", 0.0)
        - phases.get("vae_encode", {}).get("gpu_ms_mean", 0.0)
    )
    loss_gap = (
        phases.get("loss_compute", {}).get("wall_ms_mean", 0.0)
        - phases.get("loss_compute", {}).get("gpu_ms_mean", 0.0)
    )

    # Sanity flags
    flags = []
    sum_phase_wall_ratio = sum(wall_pct(p) for p in PHASES)
    if sum_phase_wall_ratio > 105.0:
        flags.append("OVERSUM")
    if step_wall > 0 and (step_unattrib_wall / step_wall) > 0.15:
        flags.append("UNATTRIB>15%")

    startup = agg.get("startup_seconds")
    startup_after = agg.get("startup_after_steps", 10)
    return {
        "bs": bs,
        "graph": graph,
        "bp": bp,
        "n_workers": n_workers,
        "n_nodes": n_nodes,
        "n_gpu": n_gpu,
        "startup_s": "n/a" if startup is None else f"{float(startup):.1f}",
        "steps/s": f"{sps:.3f}",
        "samples/s": f"{samps:.2f}",
        "step_wall_ms": f"{step_wall:.2f}",
        "data_load%": f"{wall_pct('data_load'):.1f}",
        "data_load_max_rank_ms": (
            "n/a" if data_load_max_across != data_load_max_across
            else f"{data_load_max_across:.2f}"
        ),
        "vae_gpu%": f"{gpu_pct('vae_encode'):.1f}",
        "dit_gpu%": f"{gpu_pct('dit_forward'):.1f}",
        "loss_gpu%": f"{gpu_pct('loss_compute'):.1f}",
        "bwd_gpu%": f"{gpu_pct('backward'):.1f}",
        "opt_gpu%": f"{gpu_pct('optimizer'):.1f}",
        "unattrib_wall%": f"{fmt_pct(step_unattrib_wall, step_wall):.1f}",
        "vae_wall-gpu_ms": f"{vae_gap:.2f}",
        "loss_wall-gpu_ms": f"{loss_gap:.2f}",
        "flags": ",".join(flags) or "-",
        "run_id": agg.get("run_id", "?"),
        "n_records": agg.get("n_records_used", 0),
        "step_range": ",".join(agg.get("step_ranges_used", [])) or "?",
        "startup_after_steps": startup_after,
    }


def render_md(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    out = io.StringIO()
    out.write("Profiling sweep summary\n")
    out.write("=======================\n\n")
    out.write("Notes:\n")
    out.write("- `startup_s` = wall seconds from process start to end of step `startup_after_steps` (default 10). Includes Python imports, model load, DeepSpeed/NCCL init, dataloader spin-up, plus the first 10 training steps (which include any cuda_graph capture and JIT warmup).\n")
    out.write("- Steady-state per-step columns (`steps/s`, `step_wall_ms`, all phase ratios) are computed from the last 10-step window only (default 91-100; configurable via --step-range); no cross-window averaging is applied.\n")
    out.write("- Timing from rank 0 only (data_load_max_rank_ms is the cross-rank max).\n")
    out.write("- `wall_ms - gpu_ms` per phase (vae_wall-gpu, loss_wall-gpu) reveals host-side `.item()` syncs.\n")
    out.write("- `n_workers` differs across cells (graph=1 implies num_workers=16, per user's spec) — confound noted.\n")
    out.write("- Flags: OVERSUM = sum(phase_wall_ratio) > 105% (kernel overlap). UNATTRIB>15% = `step_wall - sum(phase_wall) > 15%` (consider adding forward_prepare phase).\n\n")
    out.write("| " + " | ".join(cols) + " |\n")
    out.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for r in rows:
        out.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    return out.getvalue()


def render_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/profile",
                    help="Glob root containing profile_bs*/timing.jsonl")
    ap.add_argument("--format", choices=("md", "csv"), default="md")
    ap.add_argument("--pattern", default="profile_*/timing.jsonl",
                    help="Glob pattern relative to --root")
    ap.add_argument("--step-range", default="91-100",
                    help="Measured step range to summarize, e.g. 91-100. Use 'all' to average all non-warmup records.")
    args = ap.parse_args()
    if args.step_range == "all":
        step_range = None
    else:
        try:
            start_s, end_s = args.step_range.split("-", 1)
            step_range = (int(start_s), int(end_s))
        except ValueError:
            print(f"Invalid --step-range: {args.step_range}", file=sys.stderr)
            sys.exit(2)

    pattern = os.path.join(args.root, args.pattern)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No timing.jsonl matches: {pattern}", file=sys.stderr)
        sys.exit(2)

    rows: list[dict] = []
    for p in paths:
        records = load_run(p, step_range=step_range)
        if not records:
            print(f"WARN: no matching non-warmup records in {p}", file=sys.stderr)
            continue
        agg = aggregate_run(records)
        rows.append(build_table_row(agg))

    # Sort by (n_nodes, n_gpu, graph, bp, bs) for readability.
    # zfill so 8 sorts after 1 numerically, not lexically.
    def sort_key(r):
        return (
            int(r.get("n_nodes", 1)),
            int(r["n_gpu"]) if str(r["n_gpu"]).isdigit() else 0,
            str(r["graph"]),
            str(r["bp"]),
            int(r["bs"]) if str(r["bs"]).isdigit() else 0,
        )
    rows.sort(key=sort_key)

    if args.format == "md":
        print(render_md(rows))
    else:
        print(render_csv(rows))


if __name__ == "__main__":
    main()
