#!/usr/bin/env python3
"""Check VAE CUDA graph encode against the original encode path on real data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging
from fastwam.utils.pytorch_utils import set_global_seed


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    view = cpu.view(torch.uint8)
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(view.numpy().tobytes())
    return h.hexdigest()


def _rng_sha256(state: torch.Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * pct)), 0), len(ordered) - 1)
    return float(ordered[index])


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline WanVideoVAE.encode with vae_encode_cuda_graph=true "
            "using the real Hydra task config and real dataloader samples."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed-warmup-batches", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    return args, args.overrides


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


def _run_encode(
    *,
    vae,
    video: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    use_graph: bool,
    rng_state: torch.Tensor,
) -> dict:
    torch.cuda.set_rng_state(rng_state, device)
    setattr(vae, "_encode_cuda_graph_enabled", bool(use_graph))
    if use_graph:
        setattr(vae, "_encode_cuda_graph_failed", False)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad(), _autocast_context(device, dtype):
        latent = vae.encode(video, device=device, tiled=False)
        noise = torch.randn_like(latent)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    rng_after = torch.cuda.get_rng_state(device)

    return {
        "latent": latent.detach().clone(),
        "noise": noise.detach().clone(),
        "elapsed_ms": elapsed_ms,
        "rng_after": rng_after.detach().clone(),
        "graph_failed": bool(getattr(vae, "_encode_cuda_graph_failed", False)),
    }


def main() -> int:
    args, overrides = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA graph correctness testing.")

    setup_logging()
    set_global_seed(args.seed)
    cfg = _load_cfg(args.config_name, overrides)
    work_dir = Path(cfg.get("output_dir", "./runs/vae_graph_correctness")).resolve()
    misc.register_work_dir(str(work_dir))
    os.makedirs(work_dir, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    dtype = _mixed_precision_to_model_dtype(mixed_precision)

    print(f"[graph-test] device={device} dtype={dtype} max_batches={args.max_batches}")
    print(f"[graph-test] overrides={overrides}")

    dataset = instantiate(cfg.data.train)
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    vae = _load_vae(cfg, device=device, dtype=dtype)

    records = []
    baseline_times = []
    graph_times = []
    total_samples = 0
    for batch_idx, sample in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        video = sample["video"].to(device=device, dtype=dtype, non_blocking=True)
        torch.cuda.synchronize(device)
        rng_seed = int(args.seed + batch_idx * 1009)
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed_all(rng_seed)
        rng_before = torch.cuda.get_rng_state(device)

        base = _run_encode(
            vae=vae,
            video=video,
            device=device,
            dtype=dtype,
            use_graph=False,
            rng_state=rng_before,
        )
        graph = _run_encode(
            vae=vae,
            video=video,
            device=device,
            dtype=dtype,
            use_graph=True,
            rng_state=rng_before,
        )

        latent_equal = bool(torch.equal(base["latent"], graph["latent"]))
        noise_equal = bool(torch.equal(base["noise"], graph["noise"]))
        rng_equal = bool(torch.equal(base["rng_after"], graph["rng_after"]))
        record = {
            "batch_idx": batch_idx,
            "batch_size": int(video.shape[0]),
            "video_shape": list(video.shape),
            "latent_shape": list(base["latent"].shape),
            "latent_equal": latent_equal,
            "noise_equal": noise_equal,
            "rng_equal": rng_equal,
            "graph_failed": bool(graph["graph_failed"]),
            "latent_max_abs": _max_abs(base["latent"], graph["latent"]),
            "noise_max_abs": _max_abs(base["noise"], graph["noise"]),
            "baseline_elapsed_ms": float(base["elapsed_ms"]),
            "graph_elapsed_ms": float(graph["elapsed_ms"]),
            "baseline_latent_sha256": _tensor_sha256(base["latent"]),
            "graph_latent_sha256": _tensor_sha256(graph["latent"]),
            "baseline_noise_sha256": _tensor_sha256(base["noise"]),
            "graph_noise_sha256": _tensor_sha256(graph["noise"]),
            "baseline_rng_after_sha256": _rng_sha256(base["rng_after"]),
            "graph_rng_after_sha256": _rng_sha256(graph["rng_after"]),
        }
        records.append(record)
        total_samples += int(video.shape[0])
        if batch_idx >= int(args.speed_warmup_batches):
            baseline_times.append(float(base["elapsed_ms"]))
            graph_times.append(float(graph["elapsed_ms"]))

        print(
            "[graph-test] "
            f"batch={batch_idx} size={video.shape[0]} "
            f"latent_equal={latent_equal} noise_equal={noise_equal} rng_equal={rng_equal} "
            f"graph_failed={record['graph_failed']} "
            f"base_ms={record['baseline_elapsed_ms']:.2f} graph_ms={record['graph_elapsed_ms']:.2f}"
        )

        del video, base, graph
        torch.cuda.empty_cache()

    failures = [
        r for r in records
        if (not r["latent_equal"]) or (not r["noise_equal"]) or (not r["rng_equal"]) or r["graph_failed"]
    ]
    summary = {
        "ok": len(failures) == 0 and len(records) == int(args.max_batches),
        "num_batches": len(records),
        "expected_batches": int(args.max_batches),
        "total_samples": total_samples,
        "num_failures": len(failures),
        "baseline_ms_mean": _mean(baseline_times),
        "baseline_ms_p50": _median(baseline_times),
        "baseline_ms_p90": _percentile(baseline_times, 0.90),
        "graph_ms_mean": _mean(graph_times),
        "graph_ms_p50": _median(graph_times),
        "graph_ms_p90": _percentile(graph_times, 0.90),
        "speedup_mean": (_mean(baseline_times) / _mean(graph_times)) if graph_times and _mean(graph_times) > 0 else 0.0,
        "records": records,
    }

    output_json = args.output_json
    if output_json is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_json = str(REPO_ROOT / "bench_logs" / f"vae_graph_correctness_{stamp}.json")
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[graph-test] wrote {output_path}")
    print(
        "[graph-test] summary "
        f"ok={summary['ok']} batches={summary['num_batches']}/{summary['expected_batches']} "
        f"samples={summary['total_samples']} failures={summary['num_failures']} "
        f"baseline_mean_ms={summary['baseline_ms_mean']:.2f} "
        f"graph_mean_ms={summary['graph_ms_mean']:.2f} "
        f"speedup={summary['speedup_mean']:.3f}x"
    )
    if not summary["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
