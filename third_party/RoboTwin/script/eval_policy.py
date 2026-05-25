import sys
import os
import subprocess
import signal
import json
import time
import threading

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


# --------------------------------------------------------------------------
# Per-episode timeout + skip-and-retry plumbing.
#
# - ROBOTWIN_EPISODE_TIMEOUT_SEC: SIGALRM threshold for one (expert_check + rollout)
#   loop iteration (default 600, 0 disables).
# - ROBOTWIN_CLOSE_ENV_TIMEOUT_SEC: secondary alarm wrapped around close_env() after
#   a timeout (default 15).
# - ROBOTWIN_DEBUG_HANG_OFFSETS / ROBOTWIN_DEBUG_HANG_SEC: deterministic fault
#   injector — sleeps in `eval_policy` for the seeds matching st_seed+offset, so
#   the SIGALRM + watchdog paths can be exercised end-to-end. Both empty in prod.
# --------------------------------------------------------------------------
def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_EPISODE_TIMEOUT_SEC = _env_int("ROBOTWIN_EPISODE_TIMEOUT_SEC", 600)
_CLOSE_ENV_TIMEOUT_SEC = _env_int("ROBOTWIN_CLOSE_ENV_TIMEOUT_SEC", 15)
# Bail out of the worker after this many consecutive close_env failures
# during EpisodeTimeoutError handling — at that point the env is too broken
# to recover and we'd rather exit cleanly so the manager-side restart path
# fires than spin in an inner exception loop.
_MAX_CONSEC_CLOSE_FAILURES = _env_int("ROBOTWIN_MAX_CONSEC_CLOSE_FAILURES", 3)


class EpisodeTimeoutError(TimeoutError):
    pass


class _CloseEnvTimeoutError(TimeoutError):
    pass


def _episode_alarm_handler(signum, frame):
    raise EpisodeTimeoutError(f"episode > {_EPISODE_TIMEOUT_SEC}s")


def _close_env_alarm_handler(signum, frame):
    raise _CloseEnvTimeoutError(f"close_env > {_CLOSE_ENV_TIMEOUT_SEC}s")


def _atomic_write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write _state_<phase>.json with init_done=false BEFORE the heavy model
    # load / sapien init begins. Two invariants:
    #   (a) Preserve any progress fields from a prior watchdog-killed worker
    #       so the new worker resumes from where the old one left off.
    #   (b) Always set init_done=false on boot — the manager applies a longer
    #       WATCHDOG_INIT_SEC timeout while init_done=false so model reload is
    #       never killed.
    # ------------------------------------------------------------------
    _phase_name = _result_suffix_from_task_config(task_config)
    _state_path_init = os.path.join(str(save_dir), f"_state_{_phase_name}.json")
    _existing = {}
    if os.path.exists(_state_path_init):
        try:
            with open(_state_path_init, "r", encoding="utf-8") as _ef:
                _existing = json.load(_ef)
        except Exception:
            _existing = {}

    _st_seed_val = int(100000 * (1 + int(usr_args["seed"])))
    _initial_state = {
        "task_name": task_name,
        "task_config": task_config,
        "phase": _phase_name,
        "init_done": False,
        "test_num": int(eval_num_episodes),
        "st_seed": int(_existing.get("st_seed", _st_seed_val)),
        "now_seed": int(_existing.get("now_seed", _st_seed_val)),
        "succ_seed": int(_existing.get("succ_seed", 0)),
        "now_id": int(_existing.get("now_id", 0)),
        "suc": int(_existing.get("suc", 0)),
        "test_num_done": int(_existing.get("test_num_done", 0)),
        "suc_test_seed_list": list(_existing.get("suc_test_seed_list", [])),
        "skipped_seeds": list(_existing.get("skipped_seeds", [])),
        "force_skip_seeds": list(_existing.get("force_skip_seeds", [])),
        "evaluated_seeds": list(_existing.get("evaluated_seeds", [])),
        "restart_count": int(_existing.get("restart_count", 0)),
        "last_heartbeat_ts": time.time(),
        "episode_timeout_sec": _EPISODE_TIMEOUT_SEC,
    }
    _atomic_write_json(_state_path_init, _initial_state)

    # Now that the state file exists with init_done=false, the manager applies
    # WATCHDOG_INIT_SEC (1800s default) to this worker. Sapien_TEST is the
    # first heavy GPU/Vulkan touch and historically the slowest path on
    # hi-quality (OIDN init). Running it AFTER the state write lets the
    # init grace cover any preflight stall.
    from test_render import Sapien_TEST
    Sapien_TEST()

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num, skipped_seeds, force_skipped_seeds, evaluated_seeds = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        skip_get_obs_within_replan=skip_get_obs_within_replan,
        save_dir=str(save_dir),
        phase=_phase_name,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")

    # Write _skipped_<phase>.json with the full per-seed accounting.
    # Union of (signal-timed-out, watchdog-killed) seeds avoids double-counting
    # under rare races where the same seed lands in both lists.
    _all_skipped = sorted({int(s) for s in skipped_seeds} | {int(s) for s in force_skipped_seeds})
    _skipped_payload = {
        "task_name": task_name,
        "task_config": task_config,
        "phase": result_suffix,
        "test_num": test_num,
        "episode_timeout_sec": _EPISODE_TIMEOUT_SEC,
        "num_skipped": len(_all_skipped),
        "skipped_seeds_signal": [int(s) for s in skipped_seeds],
        "skipped_seeds_watchdog": [int(s) for s in force_skipped_seeds],
        "skipped_seeds_union": _all_skipped,
        "evaluated_seeds": [
            {"seed": int(e["seed"]), "success": bool(e["success"])} for e in evaluated_seeds
        ],
        "final_seed": int(st_seed),
        "st_seed": _st_seed_val,
    }
    _skipped_path = os.path.join(str(save_dir), f"_skipped_{result_suffix}.json")
    _atomic_write_json(_skipped_path, _skipped_payload)
    print(f"Skipped-seed log saved to {_skipped_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                save_dir=None,
                phase=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # ------------------------------------------------------------------
    # Resume + state persistence + SIGALRM handler install. main() has
    # already written an _state_<phase>.json with init_done=false before
    # entering this function; we restore any prior progress, then flip
    # init_done=true (which tells the manager to use the running-phase
    # watchdog timeout).
    # ------------------------------------------------------------------
    assert save_dir is not None and phase is not None, (
        "eval_policy() now requires save_dir and phase; main() must pass them."
    )
    assert threading.current_thread() is threading.main_thread(), (
        "eval_policy() must run on the main thread (signal handlers require it)."
    )

    state_path = os.path.join(str(save_dir), f"_state_{phase}.json")
    state = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as _sf:
                state = json.load(_sf)
        except Exception:
            state = {}

    if state:
        now_seed = int(state.get("now_seed", now_seed))
        succ_seed = int(state.get("succ_seed", 0))
        now_id = int(state.get("now_id", 0))
        TASK_ENV.suc = int(state.get("suc", 0))
        TASK_ENV.test_num = int(state.get("test_num_done", 0))
        suc_test_seed_list = list(state.get("suc_test_seed_list", []))
        skipped_seeds = list(state.get("skipped_seeds", []))
        force_skip_seeds_set = set(int(s) for s in state.get("force_skip_seeds", []))
        evaluated_seeds = list(state.get("evaluated_seeds", []))
        restart_count = int(state.get("restart_count", 0))
        _has_progress = (
            succ_seed > 0
            or len(skipped_seeds) > 0
            or len(force_skip_seeds_set) > 0
            or len(evaluated_seeds) > 0
            or restart_count > 0
        )
        if _has_progress:
            print(
                f"[RESUME] succ_seed={succ_seed} now_seed={now_seed} "
                f"skipped={len(skipped_seeds)} force_skipped={len(force_skip_seeds_set)} "
                f"evaluated={len(evaluated_seeds)} restart_count={restart_count}",
                flush=True,
            )
    else:
        skipped_seeds = []
        force_skip_seeds_set = set()
        evaluated_seeds = []
        restart_count = 0

    # Counter for close_env failures in the EpisodeTimeoutError handler. Reset
    # on any successful close_env; triggers a clean worker exit (sys.exit(42))
    # once we hit _MAX_CONSEC_CLOSE_FAILURES so the manager can restart with
    # a fresh process group instead of letting the worker loop on a broken env.
    _consec_close_failures = 0

    # `usr_args` is NOT in scope here. Use `st_seed` parameter, but trust a
    # persisted st_seed from a prior restart if present.
    _st_seed_const = int(state.get("st_seed", st_seed)) if state else int(st_seed)

    _prev_alarm_handler = None
    if _EPISODE_TIMEOUT_SEC > 0:
        _prev_alarm_handler = signal.signal(signal.SIGALRM, _episode_alarm_handler)

    def _save_state(init_done=True):
        payload = {
            "task_name": args["task_name"],
            "task_config": args["task_config"],
            "phase": phase,
            "init_done": bool(init_done),
            "test_num": int(test_num),
            "st_seed": _st_seed_const,
            "now_seed": int(now_seed),
            "succ_seed": int(succ_seed),
            "now_id": int(now_id),
            "suc": int(TASK_ENV.suc),
            "test_num_done": int(TASK_ENV.test_num),
            "suc_test_seed_list": [int(s) for s in suc_test_seed_list],
            "skipped_seeds": [int(s) for s in skipped_seeds],
            "force_skip_seeds": sorted(int(s) for s in force_skip_seeds_set),
            "evaluated_seeds": [
                {"seed": int(e["seed"]), "success": bool(e["success"])} for e in evaluated_seeds
            ],
            "restart_count": int(restart_count),
            "last_heartbeat_ts": time.time(),
            "episode_timeout_sec": _EPISODE_TIMEOUT_SEC,
        }
        _atomic_write_json(state_path, payload)

    # Flip init_done=true now that model load + resume are done. From this
    # point on the manager applies WATCHDOG_RUNNING_SEC.
    _save_state(init_done=True)

    try:
        while succ_seed < test_num:
            # Re-read force_skip_seeds in case the manager appended one
            # asynchronously between iterations (cheap stat + json read).
            try:
                with open(state_path, "r", encoding="utf-8") as _rf:
                    _fss = json.load(_rf).get("force_skip_seeds", [])
                    force_skip_seeds_set |= set(int(s) for s in _fss)
            except Exception:
                pass

            # Honour seeds the manager has flagged as hung (hard hang from
            # a previous restart).
            if now_seed in force_skip_seeds_set:
                print(f"[FORCE-SKIP] seed={now_seed} (manager watchdog)", flush=True)
                now_seed += 1
                _save_state()
                continue

            # Heartbeat at iteration start — mtime is the watchdog signal.
            _save_state()

            # Snapshot pre-iteration state for rollback on timeout.
            _snap_now_seed = now_seed
            _snap_succ_seed = succ_seed
            _snap_now_id = now_id
            _snap_suc = TASK_ENV.suc
            _snap_test_num = TASK_ENV.test_num
            _snap_suc_list_len = len(suc_test_seed_list)
            _snap_eval_len = len(evaluated_seeds)
            _snap_render_freq = args["render_freq"]

            if _EPISODE_TIMEOUT_SEC > 0:
                signal.alarm(_EPISODE_TIMEOUT_SEC)

            try:
                render_freq = args["render_freq"]
                args["render_freq"] = 0

                # ------------------------------------------------------------------
                # Debug fault injector (off by default). ROBOTWIN_DEBUG_HANG_OFFSETS
                # is comma-separated integer offsets from st_seed; sleeping seeds
                # exercise the SIGALRM / watchdog paths deterministically.
                # ------------------------------------------------------------------
                _dbg_offsets = {
                    int(s) for s in os.environ.get("ROBOTWIN_DEBUG_HANG_OFFSETS", "").split(",")
                    if s.strip()
                }
                if _dbg_offsets and (now_seed - _st_seed_const) in _dbg_offsets:
                    _default_hang_sec = (
                        max(_EPISODE_TIMEOUT_SEC + 10, 30) if _EPISODE_TIMEOUT_SEC > 0 else 30
                    )
                    _dbg_hang_sec = int(
                        os.environ.get("ROBOTWIN_DEBUG_HANG_SEC", str(_default_hang_sec))
                    )
                    print(
                        f"[DEBUG-HANG] sleeping {_dbg_hang_sec}s in seed={now_seed} "
                        f"(offset={now_seed - _st_seed_const}) to trigger timeout path",
                        flush=True,
                    )
                    time.sleep(_dbg_hang_sec)

                if expert_check:
                    try:
                        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                        episode_info = TASK_ENV.play_once()
                        TASK_ENV.close_env()
                    except UnStableError as e:
                        TASK_ENV.close_env()
                        now_seed += 1
                        args["render_freq"] = render_freq
                        continue
                    except EpisodeTimeoutError:
                        # Let the outer handler catch it — don't swallow into Exception.
                        raise
                    except Exception as e:
                        print(" -------------")
                        print("Error: ", e)
                        print("Stack Trace: ", traceback.format_exc())
                        print(" -------------")
                        TASK_ENV.close_env()
                        now_seed += 1
                        args["render_freq"] = render_freq
                        print("error occurs !")
                        continue

                if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
                    succ_seed += 1
                    suc_test_seed_list.append(now_seed)
                else:
                    now_seed += 1
                    args["render_freq"] = render_freq
                    continue

                args["render_freq"] = render_freq

                try:
                    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                except UnStableError as e:
                    succ_seed -= 1
                    if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                        suc_test_seed_list.pop()
                    TASK_ENV.close_env()
                    now_seed += 1
                    continue
                except EpisodeTimeoutError:
                    raise
                except Exception as e:
                    succ_seed -= 1
                    if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                        suc_test_seed_list.pop()
                    print(" -------------")
                    print("Error: ", e)
                    print("Stack Trace: ", traceback.format_exc())
                    print(" -------------")
                    TASK_ENV.close_env()
                    now_seed += 1
                    print("error occurs !")
                    continue
                episode_info_list = [episode_info["info"]]
                results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
                instruction = np.random.choice(results[0][instruction_type])
                TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

                current_video_path = None
                if TASK_ENV.eval_video_path is not None:
                    episode_idx = TASK_ENV.test_num
                    current_video_path = Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}.mp4"
                    ffmpeg = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            video_size,
                            "-framerate",
                            "10",
                            "-i",
                            "-",
                            "-pix_fmt",
                            "yuv420p",
                            "-vcodec",
                            "libx264",
                            "-crf",
                            "23",
                            str(current_video_path),
                        ],
                        stdin=subprocess.PIPE,
                    )
                    TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

                succ = False
                reset_func(model)
                while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                    need_obs = True
                    if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                        need_obs = bool(model.should_request_observation())

                    observation = None
                    if need_obs:
                        observation = TASK_ENV.get_obs()
                    eval_func(TASK_ENV, model, observation)
                    if TASK_ENV.eval_success:
                        succ = True
                        break
                # task_total_reward += TASK_ENV.episode_score
                if TASK_ENV.eval_video_path is not None:
                    TASK_ENV._del_eval_video_ffmpeg()
                    if current_video_path is None or not current_video_path.exists():
                        raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
                    is_randomized = "randomized" in str(args["task_config"]).lower()
                    renamed_video_path = (
                        Path(TASK_ENV.eval_video_path)
                        / f"episode{episode_idx}_randomized-{str(is_randomized).lower()}_success-{str(succ).lower()}.mp4"
                    )
                    current_video_path.rename(renamed_video_path)

                if succ:
                    TASK_ENV.suc += 1
                    print("\033[92mSuccess!\033[0m")
                else:
                    print("\033[91mFail!\033[0m")

                now_id += 1
                TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

                if TASK_ENV.render_freq:
                    TASK_ENV.viewer.close()

                TASK_ENV.test_num += 1
                evaluated_seeds.append({"seed": int(now_seed), "success": bool(succ)})

                print(
                    f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
                    f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
                )
                # TASK_ENV._take_picture()
                now_seed += 1
                _save_state()
            except EpisodeTimeoutError:
                if _EPISODE_TIMEOUT_SEC > 0:
                    signal.alarm(0)
                # Roll back scalar state. (TASK_ENV.suc / test_num set after
                # the potential rebuild below.)
                now_seed = _snap_now_seed
                succ_seed = _snap_succ_seed
                now_id = _snap_now_id
                while len(suc_test_seed_list) > _snap_suc_list_len:
                    suc_test_seed_list.pop()
                while len(evaluated_seeds) > _snap_eval_len:
                    evaluated_seeds.pop()
                args["render_freq"] = _snap_render_freq
                skipped_seeds.append(int(_snap_now_seed))
                print(
                    f"[TIMEOUT] seed={_snap_now_seed} > {_EPISODE_TIMEOUT_SEC}s; "
                    f"total_skipped={len(skipped_seeds)}",
                    flush=True,
                )

                # Best-effort close_env under a secondary alarm. If close_env
                # itself hangs or raises, the env may be in an undefined state.
                # Track this so we can bail out of the worker after too many
                # consecutive close_env failures rather than spinning forever
                # against a broken Robot/scene.
                #
                # We do NOT rebuild TASK_ENV via class_decorator() here: that
                # function only constructs the env shell; first-time field init
                # (e.g. Robot.left_planner) actually happens inside setup_demo,
                # so a fresh class_decorator() instance crashes the very next
                # setup_demo with AttributeError — see commit history.
                # Instead we let the existing `except Exception` in the loop
                # body catch any cascading setup_demo failure and skip the
                # seed; the manager's watchdog will eventually kill the
                # worker if recovery never converges.
                _close_failed = False
                signal.signal(signal.SIGALRM, _close_env_alarm_handler)
                try:
                    if _CLOSE_ENV_TIMEOUT_SEC > 0:
                        signal.alarm(_CLOSE_ENV_TIMEOUT_SEC)
                    try:
                        # Fault-injection knob for testing the consec-close-fail
                        # → sys.exit(42) path. Off by default. Only affects the
                        # in-timeout close_env call, NOT the success-path one
                        # (line 661), so the rest of the eval loop is untouched.
                        if os.environ.get("ROBOTWIN_DEBUG_FORCE_CLOSE_ENV_FAIL", "0") == "1":
                            raise RuntimeError(
                                "[DEBUG] ROBOTWIN_DEBUG_FORCE_CLOSE_ENV_FAIL=1 — "
                                "forcing in-timeout close_env to raise"
                            )
                        TASK_ENV.close_env()
                    except _CloseEnvTimeoutError:
                        print(
                            f"[TIMEOUT] close_env() also hung; abandoning env "
                            f"state for this iter (rely on next setup_demo to "
                            f"reinit scene, Exception-handler to skip if not)",
                            flush=True,
                        )
                        _close_failed = True
                    except Exception as _close_exc:
                        print(
                            f"close_env after timeout raised: {_close_exc!r}",
                            flush=True,
                        )
                        _close_failed = True
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, _episode_alarm_handler)

                # Best-effort ffmpeg cleanup so a slow leak of orphans
                # doesn't accumulate across many timeouts.
                try:
                    TASK_ENV._del_eval_video_ffmpeg()
                except Exception:
                    pass

                if _close_failed:
                    _consec_close_failures += 1
                    if _consec_close_failures >= _MAX_CONSEC_CLOSE_FAILURES:
                        print(
                            f"[FATAL] {_consec_close_failures} consecutive "
                            f"close_env failures — exiting worker so manager "
                            f"can restart with a fresh process group.",
                            flush=True,
                        )
                        _save_state()
                        # Non-zero exit triggers the manager's failure path,
                        # which is also the watchdog-style restart trigger
                        # when run_robotwin_manager is extended to allow it;
                        # at minimum it keeps the bad worker from looping.
                        sys.exit(42)
                else:
                    _consec_close_failures = 0

                # Restore counters (TASK_ENV is reused; close_env may have
                # mutated suc/test_num to zero, restore from snapshot).
                TASK_ENV.suc = _snap_suc
                TASK_ENV.test_num = _snap_test_num

                now_seed += 1
                _save_state()
                continue
            finally:
                if _EPISODE_TIMEOUT_SEC > 0:
                    signal.alarm(0)
    finally:
        if _EPISODE_TIMEOUT_SEC > 0:
            signal.alarm(0)
            if _prev_alarm_handler is not None:
                signal.signal(signal.SIGALRM, _prev_alarm_handler)

    return (
        now_seed,
        TASK_ENV.suc,
        skipped_seeds,
        sorted(force_skip_seeds_set),
        evaluated_seeds,
    )


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    # NOTE: Sapien_TEST() is now invoked inside main() AFTER the initial
    # _state_<phase>.json is written, so a hung RT preflight is covered by
    # the manager's WATCHDOG_INIT_SEC (1800s default) rather than the much
    # shorter WATCHDOG_GRACE_SEC (120s default) that applies before the
    # state file exists.
    usr_args = parse_args_and_config()

    main(usr_args)
