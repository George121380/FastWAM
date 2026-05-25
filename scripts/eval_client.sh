#!/usr/bin/env bash
# ==============================================================================
# RoboTwin simulator-client evaluation launcher.
# ==============================================================================
# This launcher does not take a checkpoint. Start model servers separately; each
# simulator worker sends observations to ROBOTWIN_CLIENT_HOST:
#   ROBOTWIN_CLIENT_BASE_PORT + gpu_id
#
# Usage:
#   QUALITY=mid EPISODES=50 ./scripts/eval_client.sh [HYDRA_OVERRIDES...]
#
# Env overrides:
#   NUM_GPUS=8 TASKS_PER_GPU=2 EPISODES=100 QUALITY=mid
#   ROBOTWIN_CLIENT_HOST=127.0.0.1
#   ROBOTWIN_CLIENT_BASE_PORT=29556
#   ROBOTWIN_CLIENT_TIMEOUT_SEC=600
#   ROBOTWIN_CLIENT_REQUEST_RETRIES=3
#   ROBOTWIN_CLIENT_RETRY_SLEEP_SEC=5
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="/share-2/code/fanqilin/peiqi/FastWAM"
DEFAULT_TASK="robotwin_uncond_3cam_384_1e-4"

TASK="${TASK:-$DEFAULT_TASK}"
NUM_GPUS="${NUM_GPUS:-8}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
EPISODES="${EPISODES:-100}"

if [[ $# -ge 1 && "${1:-}" =~ ^[1-9][0-9]*$ ]]; then
  EPISODES="$1"
  shift
fi
if ! [[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[eval_client.sh] ERROR: EPISODES must be a positive integer, got '${EPISODES}'" >&2
  exit 2
fi

export ROBOTWIN_POLICY_NAME="fastwam_client_policy"
export ROBOTWIN_CLIENT_HOST="${ROBOTWIN_CLIENT_HOST:-127.0.0.1}"
export ROBOTWIN_CLIENT_BASE_PORT="${ROBOTWIN_CLIENT_BASE_PORT:-29556}"
export ROBOTWIN_CLIENT_TIMEOUT_SEC="${ROBOTWIN_CLIENT_TIMEOUT_SEC:-600}"
export ROBOTWIN_CLIENT_REQUEST_RETRIES="${ROBOTWIN_CLIENT_REQUEST_RETRIES:-3}"
export ROBOTWIN_CLIENT_RETRY_SLEEP_SEC="${ROBOTWIN_CLIENT_RETRY_SLEEP_SEC:-5}"

# ----- render quality preset -> RT env vars --------------------------
_orig_QUALITY="${QUALITY-}"
_orig_Quality="${Quality-}"
_orig_quality="${quality-}"
_orig_Qualiaty="${Qualiaty-}"
_orig_qualiaty="${qualiaty-}"
_orig_QUALIATY="${QUALIATY-}"
QUALITY="${QUALITY:-${Quality:-${quality:-${Qualiaty:-${qualiaty:-${QUALIATY:-mid}}}}}}"
if [[ -n "$_orig_Qualiaty" || -n "$_orig_qualiaty" || -n "$_orig_QUALIATY" ]] && \
   [[ -z "$_orig_QUALITY" && -z "$_orig_Quality" && -z "$_orig_quality" ]]; then
  echo "[eval_client.sh] note: 'Qualiaty=' is a typo alias; prefer QUALITY=${QUALITY} next time." >&2
fi
case "${QUALITY}" in
  hi|high|paper)
    DEFAULT_RT_SPP=32 ; DEFAULT_RT_PATH_DEPTH=8 ; DEFAULT_RT_DENOISER=oidn
    ;;
  mid|medium)
    DEFAULT_RT_SPP=4  ; DEFAULT_RT_PATH_DEPTH=2 ; DEFAULT_RT_DENOISER=optix
    ;;
  lo|low|fast)
    DEFAULT_RT_SPP=2  ; DEFAULT_RT_PATH_DEPTH=1 ; DEFAULT_RT_DENOISER=optix
    ;;
  *)
    echo "[eval_client.sh] ERROR: unknown QUALITY='${QUALITY}' (expected hi|mid|lo)" >&2
    exit 2
    ;;
esac
export ROBOTWIN_RT_SHADER="${ROBOTWIN_RT_SHADER:-rt}"
export ROBOTWIN_RT_SPP="${ROBOTWIN_RT_SPP:-$DEFAULT_RT_SPP}"
export ROBOTWIN_RT_PATH_DEPTH="${ROBOTWIN_RT_PATH_DEPTH:-$DEFAULT_RT_PATH_DEPTH}"
export ROBOTWIN_RT_DENOISER="${ROBOTWIN_RT_DENOISER:-$DEFAULT_RT_DENOISER}"

# ----- env setup -----------------------------------------------------
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source /share-2/home/vla/miniforge3/etc/profile.d/conda.sh
conda activate fastwam_eval_cu13

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if ! python -c "import msgpack, numpy" 2>/dev/null; then
  echo "[eval_client.sh] ERROR: fastwam_eval_cu13 must provide msgpack and numpy." >&2
  exit 1
fi

# Pre-create the RoboTwin policy symlink so parallel workers do not race.
CLIENT_POLICY_DIR="${PROJECT_ROOT}/experiments/robotwin/fastwam_client_policy"
CLIENT_POLICY_TARGET="${PROJECT_ROOT}/third_party/RoboTwin/policy/fastwam_client_policy"
if [[ ! -d "${CLIENT_POLICY_DIR}" ]]; then
  echo "[eval_client.sh] ERROR: missing policy directory: ${CLIENT_POLICY_DIR}" >&2
  exit 1
fi
if [[ -e "${CLIENT_POLICY_TARGET}" && ! -L "${CLIENT_POLICY_TARGET}" ]]; then
  echo "[eval_client.sh] ERROR: policy target exists and is not a symlink: ${CLIENT_POLICY_TARGET}" >&2
  exit 1
fi
CLIENT_POLICY_SOURCE="$(readlink -f "${CLIENT_POLICY_DIR}")"
mkdir -p "$(dirname "${CLIENT_POLICY_TARGET}")"
ln -sfn "${CLIENT_POLICY_SOURCE}" "${CLIENT_POLICY_TARGET}"
echo "[eval_client.sh] policy symlink: ${CLIENT_POLICY_TARGET} -> ${CLIENT_POLICY_SOURCE}"

PROBE_PORT=$((ROBOTWIN_CLIENT_BASE_PORT + 0))
if ! python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('${ROBOTWIN_CLIENT_HOST}', ${PROBE_PORT})); s.close()" 2>/dev/null; then
  echo "[eval_client.sh] WARN: cannot reach ${ROBOTWIN_CLIENT_HOST}:${PROBE_PORT}" >&2
  echo "  Start model servers separately. Continuing; workers will wait/retry." >&2
fi

# ----- run tag -------------------------------------------------------
HOSTNAME_TAG="$(hostname | sed 's/[^A-Za-z0-9._-]/_/g')"
SWEEP_TAG_SAFE=""
if [[ -n "${SWEEP_TAG:-}" ]]; then
  SWEEP_TAG_SAFE="$(echo -n "${SWEEP_TAG}" | sed 's/[^A-Za-z0-9._-]/_/g')"
fi
CROP_TAG="$(printf 'crop%.2f' "${ROBOTWIN_EVAL_CROP_RATIO:-1.00}")"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_${HOSTNAME_TAG}_client_tpg${TASKS_PER_GPU}_q${QUALITY}_${CROP_TAG}${SWEEP_TAG_SAFE:+_sweep-${SWEEP_TAG_SAFE}}"
OUTPUT_DIR="./evaluate_results/robotwin/client/${RUN_TAG}"

echo "==============================================================================="
echo " RoboTwin simulator-client eval"
echo "==============================================================================="
echo "  PROJECT_ROOT      : ${PROJECT_ROOT}"
echo "  POLICY_NAME       : ${ROBOTWIN_POLICY_NAME}"
echo "  CKPT              : <none on client>"
echo "  TASK              : ${TASK}"
echo "  NUM_GPUS          : ${NUM_GPUS}"
echo "  TASKS_PER_GPU     : ${TASKS_PER_GPU}"
echo "  EPISODES          : ${EPISODES} per task per setting"
echo "  QUALITY           : ${QUALITY}  (spp=${ROBOTWIN_RT_SPP} path_depth=${ROBOTWIN_RT_PATH_DEPTH} denoiser=${ROBOTWIN_RT_DENOISER})"
echo "  CLIENT RPC        : ${ROBOTWIN_CLIENT_HOST}:${ROBOTWIN_CLIENT_BASE_PORT}+gpu_id"
echo "  CLIENT TIMEOUT    : ${ROBOTWIN_CLIENT_TIMEOUT_SEC}s"
echo "  OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "  HOST              : $(hostname)"
echo "  ENV               : $(which python)"
echo "==============================================================================="
echo ""

exec python experiments/robotwin/run_robotwin_manager.py \
  task="${TASK}" \
  EVALUATION.client_mode=true \
  EVALUATION.policy_name=fastwam_client_policy \
  EVALUATION.output_dir="${OUTPUT_DIR}" \
  EVALUATION.eval_num_episodes="${EPISODES}" \
  EVALUATION.client_host="${ROBOTWIN_CLIENT_HOST}" \
  EVALUATION.client_base_port="${ROBOTWIN_CLIENT_BASE_PORT}" \
  EVALUATION.client_timeout_sec="${ROBOTWIN_CLIENT_TIMEOUT_SEC}" \
  EVALUATION.client_request_retries="${ROBOTWIN_CLIENT_REQUEST_RETRIES}" \
  EVALUATION.client_retry_sleep_sec="${ROBOTWIN_CLIENT_RETRY_SLEEP_SEC}" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${TASKS_PER_GPU}" \
  "$@"
