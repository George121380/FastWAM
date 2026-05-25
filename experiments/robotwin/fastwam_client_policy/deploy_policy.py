import json
import logging
import os
import socket
import time
import uuid
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from ._msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _is_timeout_error(exc: BaseException) -> bool:
    return isinstance(exc, (socket.timeout, TimeoutError))


def _emit_event(event: Dict[str, Any]) -> None:
    print("[fastwam_client_event] " + json.dumps(event, sort_keys=True), flush=True)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _parse_int(value: Any, default: int) -> int:
    if _is_none_like(value):
        return int(default)
    return int(value)


def _extract_robotwin_inputs(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    try:
        obs_data = observation["observation"]
        head_rgb = obs_data["head_camera"]["rgb"]
        left_rgb = obs_data["left_camera"]["rgb"]
        right_rgb = obs_data["right_camera"]["rgb"]
        joint_action_vector = observation["joint_action"]["vector"]
    except KeyError as exc:
        raise KeyError(
            "RoboTwin observation is missing a required client field. Expected "
            "observation.{head_camera,left_camera,right_camera}.rgb and "
            "joint_action.vector."
        ) from exc

    return {
        "head_camera_rgb": np.ascontiguousarray(head_rgb),
        "left_camera_rgb": np.ascontiguousarray(left_rgb),
        "right_camera_rgb": np.ascontiguousarray(right_rgb),
        "joint_action_vector": np.ascontiguousarray(
            np.asarray(joint_action_vector, dtype=np.float32)
        ),
    }


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError(f"Connection closed while receiving {size} bytes")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class TcpActionClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout_sec: int,
        request_retries: int,
        retry_sleep_sec: int,
    ) -> None:
        self.host = "127.0.0.1" if host == "0.0.0.0" else str(host)
        self.port = int(port)
        self.timeout_sec = int(timeout_sec)
        self.request_retries = max(0, int(request_retries))
        self.retry_sleep_sec = max(0, int(retry_sleep_sec))
        self.addr = (self.host, self.port)
        self._packer = Packer()
        self._sock: Optional[socket.socket] = None
        self._connect_forever()

    def _connect_once(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_sec)
        try:
            sock.connect(self.addr)
        except Exception:
            sock.close()
            raise
        return sock

    def _connect_forever(self) -> None:
        while True:
            try:
                self._sock = self._connect_once()
                logger.info("Connected to action server at %s:%d", self.host, self.port)
                return
            except Exception as exc:
                logger.info(
                    "Waiting for action server at %s:%d; retry in %ss. Last error: %r",
                    self.host,
                    self.port,
                    self.retry_sleep_sec,
                    exc,
                )
                time.sleep(self.retry_sleep_sec)

    def _close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except Exception:
            pass
        finally:
            self._sock = None

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        max_attempts = self.request_retries + 1
        last_exc: Optional[BaseException] = None
        meta = {
            "request_id": payload.get("request_id"),
            "client_id": payload.get("client_id"),
            "cmd": payload.get("cmd"),
            "server": f"{self.host}:{self.port}",
        }
        for attempt in range(1, max_attempts + 1):
            try:
                if self._sock is None:
                    self._connect_forever()
                assert self._sock is not None
                request_bytes = self._packer.pack(payload)
                self._sock.sendall(len(request_bytes).to_bytes(4, "big"))
                self._sock.sendall(request_bytes)
                response_len = int.from_bytes(_recv_exact(self._sock, 4), "big")
                response = _recv_exact(self._sock, response_len)
                unpacked = unpackb(response)
                if not isinstance(unpacked, dict):
                    raise TypeError(f"Action server response must be a dict, got {type(unpacked)}")
                return unpacked
            except Exception as exc:
                last_exc = exc
                self._close()
                if _is_timeout_error(exc):
                    _emit_event({
                        "event": "action_request_timeout",
                        "time": _now_iso(),
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "timeout_sec": self.timeout_sec,
                        "error": repr(exc),
                        **meta,
                    })
                if attempt >= max_attempts:
                    break
                logger.warning(
                    "Action request failed on attempt %d/%d; retry in %ss: %r",
                    attempt,
                    max_attempts,
                    self.retry_sleep_sec,
                    exc,
                )
                time.sleep(self.retry_sleep_sec)

        if last_exc is not None:
            _emit_event({
                "event": "action_request_failed",
                "time": _now_iso(),
                "attempts": max_attempts,
                "timeout": _is_timeout_error(last_exc),
                "error": repr(last_exc),
                **meta,
            })
        raise RuntimeError(
            f"Action request failed after {max_attempts} attempts to "
            f"{self.host}:{self.port}: {last_exc!r}"
        ) from last_exc


class RobotWinClientPolicy:
    def __init__(
        self,
        host: str,
        port: int,
        replan_steps: int,
        timeout_sec: int,
        request_retries: int,
        retry_sleep_sec: int,
        timing_enabled: bool,
    ) -> None:
        self.client_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._request_idx = 0
        self._client = TcpActionClient(
            host=host,
            port=port,
            timeout_sec=timeout_sec,
            request_retries=request_retries,
            retry_sleep_sec=retry_sleep_sec,
        )
        self.replan_steps = max(1, int(replan_steps))
        self.timing_enabled = bool(timing_enabled)
        self.pending_actions: deque[np.ndarray] = deque()
        self.step_count = 0
        self.episode_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized RobotWinClientPolicy | client_id=%s | server=%s:%d | replan=%d",
            self.client_id,
            host,
            port,
            self.replan_steps,
        )

    def _next_request_id(self) -> str:
        self._request_idx += 1
        return f"{self.client_id}:{self._request_idx}"

    def _request_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        request_id = self._next_request_id()
        obs_payload = _extract_robotwin_inputs(observation)
        expected_action_dim = int(np.asarray(obs_payload["joint_action_vector"]).reshape(-1).shape[0])
        payload = {
            "cmd": "infer_action_chunk",
            "request_id": request_id,
            "client_id": self.client_id,
            "observation": obs_payload,
            "instruction": str(instruction),
        }

        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        response = self._client.request(payload)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        if response.get("request_id") not in {None, request_id}:
            raise RuntimeError(
                f"Action server response request_id mismatch: expected {request_id}, "
                f"got {response.get('request_id')}"
            )
        if "error" in response:
            raise RuntimeError(f"Action server error for request {request_id}: {response['error']}")
        if "action" not in response:
            raise KeyError(f"Action server response missing `action` for request {request_id}")

        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim != 2:
            raise ValueError(
                f"Action server must return a [T,D] action array, got shape {action.shape}"
            )
        if action.shape[0] <= 0 or action.shape[1] <= 0:
            raise ValueError(f"Action server returned an empty action array: shape {action.shape}")
        if action.shape[1] != expected_action_dim:
            raise ValueError(
                f"Action server returned action dim {action.shape[1]}, expected "
                f"{expected_action_dim} from joint_action_vector."
            )
        return action

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._request_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required when the client action queue is empty.")
            self._fill_action_queue(
                observation=observation,
                instruction=task_env.get_instruction(),
            )

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        # Keep reset local. Multiple simulator clients may share one server port.
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def get_model(usr_args: Dict[str, Any]):
    host = str(usr_args.get("client_host") or os.environ.get("ROBOTWIN_CLIENT_HOST", "127.0.0.1"))
    port = _parse_int(usr_args.get("client_port"), int(os.environ.get("ROBOTWIN_CLIENT_BASE_PORT", "29556")))
    replan_steps = _parse_int(usr_args.get("replan_steps"), 24)
    timeout_sec = _parse_int(
        usr_args.get("client_timeout_sec"),
        int(os.environ.get("ROBOTWIN_CLIENT_TIMEOUT_SEC", "600")),
    )
    request_retries = _parse_int(
        usr_args.get("client_request_retries"),
        int(os.environ.get("ROBOTWIN_CLIENT_REQUEST_RETRIES", "3")),
    )
    retry_sleep_sec = _parse_int(
        usr_args.get("client_retry_sleep_sec"),
        int(os.environ.get("ROBOTWIN_CLIENT_RETRY_SLEEP_SEC", "5")),
    )
    timing_enabled = _parse_bool(usr_args.get("timing_enabled", False))

    return RobotWinClientPolicy(
        host=host,
        port=port,
        replan_steps=replan_steps,
        timeout_sec=timeout_sec,
        request_retries=request_retries,
        retry_sleep_sec=retry_sleep_sec,
        timing_enabled=timing_enabled,
    )


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    model.step(TASK_ENV, observation)


def reset_model(model):
    model.reset()
