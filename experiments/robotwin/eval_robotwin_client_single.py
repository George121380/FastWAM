"""
RoboTwin single-task client evaluation entrypoint.

This wrapper mirrors eval_robotwin_single.py, but it never resolves or checks a
checkpoint. The simulator process talks to a remote model server through
fastwam_client_policy; ckpt loading belongs entirely to the server process.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = os.environ.get("ROBOTWIN_POLICY_NAME", "fastwam_client_policy")
CLIENT_CKPT_SETTING = "client"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _resolve_client_port(cfg: DictConfig) -> int:
    port = cfg.EVALUATION.get("client_port")
    if port is not None and str(port).strip().lower() not in {"", "none", "null"}:
        return int(port)
    return int(cfg.EVALUATION.client_base_port) + int(cfg.gpu_id)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")

    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / CLIENT_CKPT_SETTING / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    task_name = str(cfg.EVALUATION.task_name)
    log_file = run_output_dir / f"eval_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    robotwin_eval_base = run_output_dir / task_name

    client_port = _resolve_client_port(cfg)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", CLIENT_CKPT_SETTING)
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )
    _append_override(overrides, "client_host", cfg.EVALUATION.client_host)
    _append_override(overrides, "client_port", client_port)
    _append_override(overrides, "client_timeout_sec", cfg.EVALUATION.client_timeout_sec)
    _append_override(overrides, "client_request_retries", cfg.EVALUATION.client_request_retries)
    _append_override(overrides, "client_retry_sleep_sec", cfg.EVALUATION.client_retry_sleep_sec)

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        print(
            f"[client-wrapper] inner eval_policy.py exited with code {return_code}. "
            f"Log: {log_file}",
            flush=True,
        )
        sys.exit(return_code)

    print(f"Client evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{task_name}.yaml"),
    )


if __name__ == "__main__":
    main()
