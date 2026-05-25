import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
SINGLE_ENTRY_CLIENT = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_client_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2

WATCHDOG_RUNNING_SEC = int(os.environ.get("ROBOTWIN_WATCHDOG_RUNNING_SEC", "900"))
WATCHDOG_INIT_SEC = int(os.environ.get("ROBOTWIN_WATCHDOG_INIT_SEC", "1800"))
WATCHDOG_GRACE_SEC = int(os.environ.get("ROBOTWIN_WATCHDOG_GRACE_SEC", "120"))
MAX_RESTARTS = int(os.environ.get("ROBOTWIN_MAX_RESTARTS", "20"))


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
        "EVALUATION.client_port",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _phase_state_filename(phase: str) -> str:
    if phase == "clean":
        return "_state_clean.json"
    if phase == "random":
        return "_state_random.json"
    raise ValueError(f"Unsupported phase: {phase}")


def _phase_skipped_filename(phase: str) -> str:
    if phase == "clean":
        return "_skipped_clean.json"
    if phase == "random":
        return "_skipped_random.json"
    raise ValueError(f"Unsupported phase: {phase}")


def _read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _bump_restart_and_maybe_force_skip(path: Path, seed_or_none: int | None) -> int:
    """Always bumps restart_count. Appends to force_skip_seeds only when a seed
    is known (i.e. the worker was past init_done and had a current now_seed).
    Returns the new restart_count.
    """
    data = _read_state(path) or {}
    data["restart_count"] = int(data.get("restart_count", 0)) + 1
    if seed_or_none is not None:
        fss = list(data.get("force_skip_seeds", []))
        if int(seed_or_none) not in fss:
            fss.append(int(seed_or_none))
            data["force_skip_seeds"] = sorted(set(fss))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return int(data["restart_count"])


_EMPTY_SKIPPED: dict[str, Any] = {
    "num_skipped": 0,
    "skipped_seeds_signal": [],
    "skipped_seeds_watchdog": [],
    "skipped_seeds_union": [],
    "evaluated_seeds": [],
}


def _parse_skipped(skipped_file: Path) -> dict[str, Any]:
    if not skipped_file.exists():
        return dict(_EMPTY_SKIPPED)
    try:
        return json.loads(skipped_file.read_text(encoding="utf-8"))
    except Exception:
        return dict(_EMPTY_SKIPPED)


def _synthesize_skipped_from_state(state_file: Path) -> dict[str, Any]:
    """Build a skipped-record from _state_<phase>.json when _skipped_<phase>.json
    is missing — typical case is `watchdog_gave_up` where the worker was killed
    before the post-loop write of _skipped happened. Returns the same schema as
    _parse_skipped so write_outputs() can treat both paths uniformly.
    """
    if not state_file.exists():
        return dict(_EMPTY_SKIPPED, source="missing")
    try:
        d = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return dict(_EMPTY_SKIPPED, source="state_unparseable")

    sig = [int(s) for s in d.get("skipped_seeds", [])]
    wd = [int(s) for s in d.get("force_skip_seeds", [])]
    union = sorted(set(sig) | set(wd))
    return {
        "num_skipped": len(union),
        "skipped_seeds_signal": sig,
        "skipped_seeds_watchdog": wd,
        "skipped_seeds_union": union,
        "evaluated_seeds": [
            {"seed": int(e["seed"]), "success": bool(e["success"])}
            for e in d.get("evaluated_seeds", [])
        ],
        "st_seed": int(d.get("st_seed", -1)),
        "final_seed": int(d.get("now_seed", -1)),
        "test_num": int(d.get("test_num", 0)),
        "episode_timeout_sec": int(d.get("episode_timeout_sec", 0)),
        "source": "synthesized_from_state",
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]
    launch_ts: float = 0.0
    restart_count: int = 0


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    client_mode = bool(cfg.EVALUATION.get("client_mode", False))
    single_entry = SINGLE_ENTRY_CLIENT if client_mode else SINGLE_ENTRY
    if not single_entry.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {single_entry}")

    if client_mode:
        ckpt_path: Path | None = None
        ckpt_tag = "client"
    else:
        if cfg.ckpt is None:
            raise ValueError("`ckpt` must not be None.")
        ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks()
    else:
        tasks = [str(task_name_cfg)]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    task_skipped: dict[str, dict[str, dict[str, Any] | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    task_incomplete: dict[str, dict[str, dict[str, Any] | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }
    client_base_port = int(cfg.EVALUATION.get("client_base_port", 29556))

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(single_entry),
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        if client_mode:
            cmd.append(f"EVALUATION.client_port={client_base_port + gpu_id}")
        else:
            assert ckpt_path is not None
            cmd.append(f"ckpt={str(ckpt_path)}")
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        # start_new_session=True puts the worker in its own process group
        # so the manager can SIGTERM/SIGKILL the whole chain (wrapper +
        # eval_policy.py + any helpers like ffmpeg) atomically via killpg.
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            start_new_session=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
            launch_ts=time.time(),
        )

    def _killpg(state: RunningState, sig: int) -> None:
        try:
            os.killpg(os.getpgid(state.process.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            _killpg(state, signal.SIGTERM)
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                _killpg(state, signal.SIGKILL)
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase="clean"))

    def _state_file_for(state: RunningState) -> Path:
        return run_output_dir / state.task_name / _phase_state_filename(state.phase)

    def _status_str(task: str, phase: str) -> str:
        inc = task_incomplete[task][phase]
        if inc is None:
            return "ok"
        return f"incomplete:{inc.get('reason', 'unknown')}"

    def write_outputs() -> None:
        # Read each task's _skipped_<phase>.json for skip counts and per-seed log.
        # If the file is missing — typical for watchdog_gave_up phases where the
        # worker was killed before writing the post-loop summary — fall back to
        # the last-written _state_<phase>.json so the skip counts and per-seed
        # evaluations still reach the summary.
        for task in tasks:
            for phase in ("clean", "random"):
                if task_skipped[task][phase] is None:
                    skipped_file = run_output_dir / task / _phase_skipped_filename(phase)
                    if skipped_file.exists():
                        task_skipped[task][phase] = _parse_skipped(skipped_file)
                    else:
                        state_file = run_output_dir / task / _phase_state_filename(phase)
                        task_skipped[task][phase] = _synthesize_skipped_from_state(state_file)

        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        num_clean_completed = sum(1 for t in tasks if task_incomplete[t]["clean"] is None and task_rates[t]["clean"] is not None)
        num_clean_incomplete = sum(1 for t in tasks if task_incomplete[t]["clean"] is not None)
        num_random_completed = sum(1 for t in tasks if task_incomplete[t]["random"] is None and task_rates[t]["random"] is not None)
        num_random_incomplete = sum(1 for t in tasks if task_incomplete[t]["random"] is not None)

        total_clean_skipped = sum(int((task_skipped[t]["clean"] or {}).get("num_skipped", 0)) for t in tasks)
        total_random_skipped = sum(int((task_skipped[t]["random"] or {}).get("num_skipped", 0)) for t in tasks)
        total_watchdog_kills = sum(
            len((task_skipped[t][p] or {}).get("skipped_seeds_watchdog", []))
            for t in tasks for p in ("clean", "random")
        )

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Backward-compatible columns first; new columns appended at the end.
            writer.writerow([
                "task_name", "clean_success_rate", "random_success_rate",
                "clean_skipped", "random_skipped",
                "clean_status", "random_status",
            ])
            for task in tasks:
                writer.writerow([
                    task,
                    task_rates[task]["clean"],
                    task_rates[task]["random"],
                    (task_skipped[task]["clean"] or {}).get("num_skipped"),
                    (task_skipped[task]["random"] or {}).get("num_skipped"),
                    _status_str(task, "clean"),
                    _status_str(task, "random"),
                ])
            writer.writerow([
                "__overall__", clean_mean, random_mean,
                total_clean_skipped, total_random_skipped, "", "",
            ])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                    "clean_skipped": task_skipped[task]["clean"],
                    "random_skipped": task_skipped[task]["random"],
                    "clean_status": _status_str(task, "clean"),
                    "random_status": _status_str(task, "random"),
                    "incomplete": {
                        "clean": task_incomplete[task]["clean"],
                        "random": task_incomplete[task]["random"],
                    } if (task_incomplete[task]["clean"] or task_incomplete[task]["random"]) else None,
                }
                for task in tasks
            ],
            "overall": {
                # Kept as backward-compatible aliases of the *_completed_only fields.
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
                "clean_mean_success_rate_completed_only": _to_jsonable(clean_mean),
                "random_mean_success_rate_completed_only": _to_jsonable(random_mean),
                "num_clean_completed": num_clean_completed,
                "num_clean_incomplete": num_clean_incomplete,
                "num_random_completed": num_random_completed,
                "num_random_incomplete": num_random_incomplete,
                "total_clean_skipped": total_clean_skipped,
                "total_random_skipped": total_random_skipped,
                "total_watchdog_kills": total_watchdog_kills,
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} client_mode={client_mode} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir}"
    )

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while len(running_states) > 0:
        progressed = False

        # ------------------------------------------------------------------
        # Watchdog scan — kills workers whose state-file heartbeat has gone
        # stale beyond the appropriate timeout (longer during model-load /
        # init_done=false, shorter once init_done=true and the eval loop is
        # running). Worker process group is killed via killpg and the
        # in-flight seed (if known) is blacklisted in force_skip_seeds for
        # the restarted worker to honour.
        # ------------------------------------------------------------------
        watchdog_now = time.time()
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            sf = _state_file_for(state)
            if not sf.exists():
                # No heartbeat yet — only kill once the wrapper-startup grace
                # window has lapsed.
                if watchdog_now - state.launch_ts <= WATCHDOG_GRACE_SEC:
                    continue
                last = state.launch_ts
                effective_timeout = WATCHDOG_GRACE_SEC
                reason = "no_state_file"
            else:
                last = sf.stat().st_mtime
                st_for_timeout = _read_state(sf) or {}
                if bool(st_for_timeout.get("init_done", False)):
                    effective_timeout = WATCHDOG_RUNNING_SEC
                    reason = "running_silent"
                else:
                    effective_timeout = WATCHDOG_INIT_SEC
                    reason = "init_silent"
            if watchdog_now - last <= effective_timeout:
                continue

            # Hard hang — kill the process group, blacklist the in-flight
            # seed (only if we have one), and either re-launch or mark the
            # (task,phase) incomplete depending on restart_count.
            st = _read_state(sf) or {}
            hung_seed = st.get("now_seed") if bool(st.get("init_done", False)) else None
            log(
                f"[WATCHDOG] task={state.task_name} phase={state.phase} "
                f"gpu={state.gpu_id} silent for {int(watchdog_now - last)}s "
                f"reason={reason} hung_seed={hung_seed}"
            )
            _killpg(state, signal.SIGTERM)
            try:
                state.process.wait(timeout=TERMINATE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                _killpg(state, signal.SIGKILL)
                state.process.wait()
            running_states.remove(state)
            progressed = True

            new_restart_count = _bump_restart_and_maybe_force_skip(sf, hung_seed)

            if new_restart_count > MAX_RESTARTS:
                log(
                    f"[WATCHDOG] giving up: restart_count={new_restart_count} "
                    f"> MAX_RESTARTS reason={reason}"
                )
                task_rates[state.task_name][state.phase] = None
                task_incomplete[state.task_name][state.phase] = {
                    "reason": "watchdog_gave_up",
                    "restart_count": new_restart_count,
                    "last_watchdog_reason": reason,
                }
                failed_records.append({
                    "task_name": state.task_name,
                    "phase": state.phase,
                    "gpu_id": state.gpu_id,
                    "return_code": -1,
                    "reason": "watchdog_gave_up",
                })
                if state.phase == "clean":
                    log(
                        f"[WATCHDOG] also marking random phase incomplete for "
                        f"task={state.task_name}"
                    )
                    task_rates[state.task_name]["random"] = None
                    task_incomplete[state.task_name]["random"] = {
                        "reason": "skipped_due_to_clean_phase_gave_up",
                        "restart_count": 0,
                        "last_watchdog_reason": "n/a",
                    }
                try_launch_pending(state.gpu_id)
                continue

            log(
                f"[WATCHDOG] restarting task={state.task_name} phase={state.phase} "
                f"gpu={state.gpu_id} restart_count={new_restart_count}"
            )
            new_state = launch_phase(
                task_name=state.task_name, gpu_id=state.gpu_id, phase=state.phase,
            )
            new_state.restart_count = new_restart_count
            running_states.append(new_state)

        # ------------------------------------------------------------------
        # Reap workers that exited on their own.
        # ------------------------------------------------------------------
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                # Special case: return code 42 = the worker proactively bailed
                # after ROBOTWIN_MAX_CONSEC_CLOSE_FAILURES consecutive close_env
                # failures in the EpisodeTimeoutError handler. The worker asked
                # the manager for a fresh-process-group restart rather than the
                # rest of the sweep being globally aborted. Treat it exactly
                # like a watchdog kill: blacklist the in-flight seed, bump
                # restart_count, respect MAX_RESTARTS, then relaunch.
                if return_code == 42:
                    sf = run_output_dir / state.task_name / _phase_state_filename(state.phase)
                    st = _read_state(sf) or {}
                    # sys.exit(42) only fires inside the eval loop where
                    # init_done is already true; still defensive-check.
                    hung_seed = st.get("now_seed") if bool(st.get("init_done", False)) else None
                    log(
                        f"[CLOSE-ENV-FATAL] task={state.task_name} phase={state.phase} "
                        f"gpu={gpu_id} worker exited 42 hung_seed={hung_seed}"
                    )
                    new_restart_count = _bump_restart_and_maybe_force_skip(sf, hung_seed)
                    progressed = True

                    if new_restart_count > MAX_RESTARTS:
                        log(
                            f"[CLOSE-ENV-FATAL] giving up: restart_count={new_restart_count} "
                            f"> MAX_RESTARTS"
                        )
                        task_rates[state.task_name][state.phase] = None
                        task_incomplete[state.task_name][state.phase] = {
                            "reason": "close_env_fatal_gave_up",
                            "restart_count": new_restart_count,
                            "last_watchdog_reason": "close_env_fatal",
                        }
                        failed_records.append({
                            "task_name": state.task_name,
                            "phase": state.phase,
                            "gpu_id": gpu_id,
                            "return_code": 42,
                            "reason": "close_env_fatal_gave_up",
                        })
                        if state.phase == "clean":
                            log(
                                f"[CLOSE-ENV-FATAL] also marking random phase incomplete "
                                f"for task={state.task_name}"
                            )
                            task_rates[state.task_name]["random"] = None
                            task_incomplete[state.task_name]["random"] = {
                                "reason": "skipped_due_to_clean_phase_gave_up",
                                "restart_count": 0,
                                "last_watchdog_reason": "n/a",
                            }
                        try_launch_pending(state.gpu_id)
                        continue

                    log(
                        f"[CLOSE-ENV-FATAL] restarting task={state.task_name} "
                        f"phase={state.phase} gpu={state.gpu_id} "
                        f"restart_count={new_restart_count}"
                    )
                    new_state = launch_phase(
                        task_name=state.task_name,
                        gpu_id=state.gpu_id,
                        phase=state.phase,
                    )
                    new_state.restart_count = new_restart_count
                    running_states.append(new_state)
                    continue

                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            skipped_file = run_output_dir / state.task_name / _phase_skipped_filename(state.phase)
            task_skipped[state.task_name][state.phase] = _parse_skipped(skipped_file)
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f} "
                f"skipped={(task_skipped[state.task_name][state.phase] or {}).get('num_skipped', 0)}"
            )

            if state.phase == "clean":
                running_states.append(launch_phase(
                    task_name=state.task_name,
                    gpu_id=gpu_id,
                    phase="random",
                ))
                continue

            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
