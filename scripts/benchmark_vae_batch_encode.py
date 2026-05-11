#!/usr/bin/env python3
"""Benchmark serial, batch-parallel, and batch-graph VAE encode paths."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging
from fastwam.utils.pytorch_utils import set_global_seed


MODES = ("serial_eager", "batch_eager", "batch_graph")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark WanVideoVAE.encode on saved batches. The script first "
            "checks batch-parallel CUDA graph compatibility, then times each mode."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--sample-root", default=str(REPO_ROOT / "test_samples"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--compat-batches", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compat-only", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides. Defaults to task=robotwin_uncond_3cam_384_1e-4.",
    )
    args = parser.parse_args()
    overrides = args.overrides or ["task=robotwin_uncond_3cam_384_1e-4"]
    return args, overrides


def _load_cfg(config_name: str, overrides: list[str]):
    register_default_resolvers()
    config_dir = REPO_ROOT / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _load_vae(cfg, device: torch.device, dtype: torch.dtype):
    _, _, vae_config, _ = _resolve_configs(
        model_id=str(cfg.model.model_id),
        tokenizer_model_id=str(cfg.model.tokenizer_model_id),
        redirect_common_files=bool(cfg.model.get("redirect_common_files", True)),
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=dtype,
        device=str(device),
    )
    vae.eval()
    return vae


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    if dtype not in {torch.float16, torch.bfloat16}:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_sample(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sample_paths(sample_root: Path, batch_size: int, max_batches: int) -> list[Path]:
    paths = sorted(sample_root.glob(f"batch_size_{batch_size:03d}_batch_*.pt"))
    return paths[:max_batches]


def _configure_vae_mode(vae, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    setattr(vae, "_encode_cuda_graph_enabled", mode == "batch_graph")
    setattr(vae, "_encode_batch_parallel_enabled", mode in {"batch_eager", "batch_graph"})
    if mode == "batch_graph":
        setattr(vae, "_encode_cuda_graph_failed", False)


def _clear_graph_cache(vae, device: torch.device) -> None:
    setattr(vae, "_encode_cuda_graph_cache", {})
    setattr(vae, "_encode_cuda_graph_failed", False)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}
    ordered = sorted(values)
    p90_idx = min(round((len(ordered) - 1) * 0.90), len(ordered) - 1)
    return {
        "mean_ms": float(statistics.mean(values)),
        "p50_ms": float(statistics.median(values)),
        "p90_ms": float(ordered[p90_idx]),
    }


def _set_rng(device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.cuda.get_rng_state(device)


@torch.no_grad()
def _run_encode(
    *,
    vae,
    video_cpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    rng_state: torch.Tensor | None = None,
    include_noise: bool = False,
) -> dict:
    _configure_vae_mode(vae, mode)
    video = video_cpu.to(device=device, dtype=dtype, non_blocking=True)
    if rng_state is not None:
        torch.cuda.set_rng_state(rng_state, device)

    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    with _autocast_context(device, dtype):
        latent = vae.encode(video, device=device, tiled=False)
        noise = torch.randn_like(latent) if include_noise else None
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    gpu_ms = float(start_event.elapsed_time(end_event))
    rng_after = torch.cuda.get_rng_state(device)

    result = {
        "latent": latent.detach().clone(),
        "noise": None if noise is None else noise.detach().clone(),
        "rng_after": rng_after.detach().clone(),
        "gpu_ms": gpu_ms,
        "wall_ms": wall_ms,
        "graph_failed": bool(getattr(vae, "_encode_cuda_graph_failed", False)),
    }
    del video, latent, noise
    return result


def _compatibility_check(
    *,
    vae,
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    sample = _load_sample(path)
    video_cpu = sample["video"]
    rng_state = _set_rng(device, seed)

    serial = _run_encode(
        vae=vae,
        video_cpu=video_cpu,
        device=device,
        dtype=dtype,
        mode="serial_eager",
        rng_state=rng_state,
        include_noise=True,
    )
    batch = _run_encode(
        vae=vae,
        video_cpu=video_cpu,
        device=device,
        dtype=dtype,
        mode="batch_eager",
        rng_state=rng_state,
        include_noise=True,
    )
    graph = _run_encode(
        vae=vae,
        video_cpu=video_cpu,
        device=device,
        dtype=dtype,
        mode="batch_graph",
        rng_state=rng_state,
        include_noise=True,
    )

    serial_shape = list(serial["latent"].shape)
    batch_shape = list(batch["latent"].shape)
    graph_shape = list(graph["latent"].shape)
    batch_graph_latent_equal = bool(torch.equal(batch["latent"], graph["latent"]))
    batch_graph_noise_equal = bool(torch.equal(batch["noise"], graph["noise"]))
    batch_graph_rng_equal = bool(torch.equal(batch["rng_after"], graph["rng_after"]))
    ok = (
        batch_shape == graph_shape
        and not graph["graph_failed"]
        and batch_graph_latent_equal
        and batch_graph_noise_equal
        and batch_graph_rng_equal
    )

    record = {
        "path": str(path),
        "video_shape": list(video_cpu.shape),
        "serial_shape": serial_shape,
        "batch_shape": batch_shape,
        "graph_shape": graph_shape,
        "serial_batch_shape_equal": serial_shape == batch_shape,
        "batch_graph_shape_equal": batch_shape == graph_shape,
        "serial_batch_max_abs": _max_abs(serial["latent"], batch["latent"]),
        "batch_graph_max_abs": _max_abs(batch["latent"], graph["latent"]),
        "batch_graph_latent_equal": batch_graph_latent_equal,
        "batch_graph_noise_equal": batch_graph_noise_equal,
        "batch_graph_rng_equal": batch_graph_rng_equal,
        "graph_failed": bool(graph["graph_failed"]),
        "ok": bool(ok),
    }
    del sample, video_cpu, serial, batch, graph
    torch.cuda.empty_cache()
    return record


def _benchmark_batch_size(
    *,
    vae,
    paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    warmup_batches: int,
    graph_ok: bool,
) -> dict:
    modes = ["serial_eager", "batch_eager"] + (["batch_graph"] if graph_ok else [])
    timings: dict[str, list[float]] = {mode: [] for mode in modes}
    batch_records = []

    for batch_idx, path in enumerate(paths):
        sample = _load_sample(path)
        video_cpu = sample["video"]
        per_batch = {"path": str(path), "batch_idx": batch_idx, "modes": {}}
        for mode in modes:
            rng_state = _set_rng(device, seed + batch_idx * 1009)
            result = _run_encode(
                vae=vae,
                video_cpu=video_cpu,
                device=device,
                dtype=dtype,
                mode=mode,
                rng_state=rng_state,
                include_noise=False,
            )
            per_batch["modes"][mode] = {
                "gpu_ms": float(result["gpu_ms"]),
                "wall_ms": float(result["wall_ms"]),
                "graph_failed": bool(result["graph_failed"]),
            }
            if batch_idx >= warmup_batches:
                timings[mode].append(float(result["gpu_ms"]))
            del result
        batch_records.append(per_batch)
        del sample, video_cpu
        torch.cuda.empty_cache()

    summary = {mode: _stats(values) for mode, values in timings.items()}
    serial_mean = summary["serial_eager"]["mean_ms"]
    for mode, mode_summary in summary.items():
        mode_mean = mode_summary["mean_ms"]
        mode_summary["speedup_vs_serial"] = (
            float(serial_mean / mode_mean) if mode_mean > 0 else 0.0
        )

    return {
        "num_batches": len(paths),
        "warmup_batches": warmup_batches,
        "summary": summary,
        "records": batch_records,
    }


def main() -> int:
    args, overrides = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VAE encode benchmarking.")

    setup_logging()
    set_global_seed(args.seed)
    cfg = _load_cfg(args.config_name, overrides)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    dtype = _mixed_precision_to_model_dtype(mixed_precision)
    vae = _load_vae(cfg, device=device, dtype=dtype)

    sample_root = Path(args.sample_root)
    print(f"[vae-bench] device={device} dtype={dtype} sample_root={sample_root}")
    print(f"[vae-bench] overrides={overrides}")

    output = {
        "config_name": args.config_name,
        "overrides": overrides,
        "device": str(device),
        "dtype": str(dtype),
        "sample_root": str(sample_root),
        "batch_sizes": args.batch_sizes,
        "compatibility": {},
        "benchmarks": {},
    }

    all_compat_ok = True
    for batch_size in args.batch_sizes:
        paths = _sample_paths(sample_root, batch_size, max(args.max_batches, args.compat_batches))
        if not paths:
            raise FileNotFoundError(f"No saved batches found for batch size {batch_size}.")
        compat_paths = paths[: max(int(args.compat_batches), 1)]
        compat_records = []
        graph_ok = True
        for compat_idx, path in enumerate(compat_paths):
            record = _compatibility_check(
                vae=vae,
                path=path,
                device=device,
                dtype=dtype,
                seed=args.seed + batch_size * 100 + compat_idx,
            )
            compat_records.append(record)
            graph_ok = graph_ok and bool(record["ok"])
            print(
                "[vae-bench][compat] "
                f"bs={batch_size} file={path.name} ok={record['ok']} "
                f"serial_batch_max_abs={record['serial_batch_max_abs']:.6g} "
                f"batch_graph_max_abs={record['batch_graph_max_abs']:.6g} "
                f"graph_failed={record['graph_failed']}"
            )
        output["compatibility"][str(batch_size)] = {
            "ok": bool(graph_ok),
            "records": compat_records,
        }
        all_compat_ok = all_compat_ok and graph_ok

        if args.compat_only:
            _clear_graph_cache(vae, device)
            continue

        bench_paths = paths[: int(args.max_batches)]
        bench = _benchmark_batch_size(
            vae=vae,
            paths=bench_paths,
            device=device,
            dtype=dtype,
            seed=args.seed + batch_size * 1000,
            warmup_batches=int(args.warmup_batches),
            graph_ok=graph_ok,
        )
        output["benchmarks"][str(batch_size)] = bench
        for mode, summary in bench["summary"].items():
            print(
                "[vae-bench][timing] "
                f"bs={batch_size} mode={mode} "
                f"mean_gpu_ms={summary['mean_ms']:.3f} "
                f"p50_gpu_ms={summary['p50_ms']:.3f} "
                f"p90_gpu_ms={summary['p90_ms']:.3f} "
                f"speedup_vs_serial={summary['speedup_vs_serial']:.3f}x"
            )
        _clear_graph_cache(vae, device)

    output["ok"] = bool(all_compat_ok)
    if args.output_json is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "compat" if args.compat_only else "benchmark"
        output_json = REPO_ROOT / "bench_logs" / f"vae_batch_encode_{suffix}_{stamp}.json"
    else:
        output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[vae-bench] wrote {output_json}")
    if args.compat_only and not all_compat_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
