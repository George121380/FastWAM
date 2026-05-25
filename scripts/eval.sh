#!/usr/bin/env bash
# ==============================================================================
# RoboTwin 8-GPU evaluation — one-shot launcher
# ==============================================================================
# Usage:
#   ./scripts/eval.sh [CKPT] [TASK] [DATASET_STATS] [EPISODES]
#
# Examples:
#   # Use defaults (released checkpoint + "mid" quality):
#   ./scripts/eval.sh
#
#   # Custom checkpoint:
#   ./scripts/eval.sh /path/to/my_ckpt.pt
#
#   # Full override:
#   ./scripts/eval.sh /path/ckpt.pt robotwin_uncond_3cam_384_1e-4 /path/stats.json
#
#   # Run 50 valid episodes per task per setting (clean/random):
#   EPISODES=50 ./scripts/eval.sh
#   ./scripts/eval.sh /path/ckpt.pt robotwin_uncond_3cam_384_1e-4 /path/stats.json 50
#
# Env overrides:
#   NUM_GPUS=4 TASKS_PER_GPU=1 ./scripts/eval.sh ...
#   EPISODES=50 ./scripts/eval.sh ...
#
#   # Render quality preset (changes sapien RT settings):
#   QUALITY=hi  ./scripts/eval.sh ...   # paper-spec: spp=32 path=8 OIDN  (may hang on B300, ~3-4× slower)
#   QUALITY=mid ./scripts/eval.sh ...   # default:    spp=4  path=2 OptiX (no hang, paper-comparable quality)
#   QUALITY=lo  ./scripts/eval.sh ...   # fastest:    spp=2  path=1 OptiX (visually ≈ mid, ~10% faster)
#
#   # Or set individual RT knobs directly:
#   ROBOTWIN_RT_SPP=8 ROBOTWIN_RT_PATH_DEPTH=4 ROBOTWIN_RT_DENOISER=optix ./scripts/eval.sh ...
#
# Tip: run inside `tmux` so the manager survives ssh disconnect.
#   tmux new -s eval     # then run this script inside
# ==============================================================================
set -euo pipefail

# ----- 默认参数（可被命令行参数覆盖）---------------------------------
PROJECT_ROOT="/share-2/code/fanqilin/peiqi/FastWAM"
DEFAULT_CKPT="${PROJECT_ROOT}/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
DEFAULT_TASK="robotwin_uncond_3cam_384_1e-4"
DEFAULT_DATASET_STATS="${PROJECT_ROOT}/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"

CKPT="${1:-$DEFAULT_CKPT}"
TASK="${2:-$DEFAULT_TASK}"
DATASET_STATS="${3:-$DEFAULT_DATASET_STATS}"

NUM_GPUS="${NUM_GPUS:-8}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
EPISODES="${EPISODES:-100}"

# Backward-compatible positional parsing:
#   arg4 = positive integer  -> episode count
#   arg4 = anything else     -> first Hydra override, preserving the older
#                               "args 4+ are forwarded" behavior.
EXTRA_OVERRIDE_START=4
if [[ $# -ge 4 ]]; then
  if [[ "${4}" =~ ^[0-9]+$ ]]; then
    EPISODES="${4}"
    EXTRA_OVERRIDE_START=5
  fi
fi
if ! [[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[eval.sh] ERROR: EPISODES must be a positive integer, got '${EPISODES}'" >&2
  exit 2
fi

# ----- render quality preset → RT env vars ----------------------------
# QUALITY can be "hi" / "mid" / "lo", or unset (== "mid" default).
# Accept the common `Quality=...` / `quality=...` aliases too.
# Individual env vars (ROBOTWIN_RT_SPP / _PATH_DEPTH / _DENOISER) take
# precedence over the preset if explicitly set.
# Snapshot original env vars BEFORE the fallback resolves, so the typo-alias
# warning below can tell apart "user spelled QUALITY correctly" vs "user spelled
# Qualiaty and fell through to typo-alias".
_orig_QUALITY="${QUALITY-}"
_orig_Quality="${Quality-}"
_orig_quality="${quality-}"
_orig_Qualiaty="${Qualiaty-}"
_orig_qualiaty="${qualiaty-}"
_orig_QUALIATY="${QUALIATY-}"
QUALITY="${QUALITY:-${Quality:-${quality:-${Qualiaty:-${qualiaty:-${QUALIATY:-mid}}}}}}"
if [[ -n "$_orig_Qualiaty" || -n "$_orig_qualiaty" || -n "$_orig_QUALIATY" ]] && \
   [[ -z "$_orig_QUALITY" && -z "$_orig_Quality" && -z "$_orig_quality" ]]; then
  echo "[eval.sh] note: 'Qualiaty=' is a typo alias; prefer QUALITY=${QUALITY} next time." >&2
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
    echo "[eval.sh] ERROR: unknown QUALITY='${QUALITY}' (expected hi|mid|lo)" >&2
    exit 2
    ;;
esac
export ROBOTWIN_RT_SHADER="${ROBOTWIN_RT_SHADER:-rt}"
export ROBOTWIN_RT_SPP="${ROBOTWIN_RT_SPP:-$DEFAULT_RT_SPP}"
export ROBOTWIN_RT_PATH_DEPTH="${ROBOTWIN_RT_PATH_DEPTH:-$DEFAULT_RT_PATH_DEPTH}"
export ROBOTWIN_RT_DENOISER="${ROBOTWIN_RT_DENOISER:-$DEFAULT_RT_DENOISER}"

# ----- env setup ------------------------------------------------------
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source /share-2/home/vla/miniforge3/etc/profile.d/conda.sh
conda activate fastwam_eval_cu13

# SAPIEN looks for nvidia Vulkan ICD only under /usr/share/vulkan/icd.d/ by default;
# on this host the nvidia ICD lives at /etc/vulkan/icd.d/. Point sapien at it
# explicitly to avoid loading a stale bundled ICD (api 1.2.140).
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# Required for torch.use_deterministic_algorithms(True) on CUDA (set inside
# third_party/RoboTwin/envs/_base_task.py). Must be exported BEFORE the first
# CUDA op in the worker process — env inherits through subprocess.Popen, so
# setting it here is sufficient. Without it, the expert-check / curobo path
# can pick non-deterministic CUBLAS kernels, which makes "two runs of the
# same quality must agree" unreliable.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ----- sanity checks --------------------------------------------------
if [[ ! -f "${CKPT}" ]]; then
  echo "[eval.sh] ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_STATS}" ]]; then
  echo "[eval.sh] ERROR: dataset stats not found: ${DATASET_STATS}" >&2
  exit 1
fi

# ----- build a unique run tag -----------------------------------------
HOSTNAME_TAG="$(hostname | sed 's/[^A-Za-z0-9._-]/_/g')"
CKPT_TAG="$(basename "${CKPT}" .pt)"
# Optional SWEEP_TAG suffix — set by scripts/launch_quality_sweep.sh so the
# Phase 2 report builder can locate this exact run dir without disambiguating
# by mtime against historical/concurrent runs on the same host. Allowed
# characters mirror HOSTNAME_TAG.
SWEEP_TAG_SAFE=""
if [[ -n "${SWEEP_TAG:-}" ]]; then
  SWEEP_TAG_SAFE="$(echo -n "${SWEEP_TAG}" | sed 's/[^A-Za-z0-9._-]/_/g')"
fi
# ROBOTWIN_EVAL_CROP_RATIO is consumed by deploy_policy.py (CenterCrop + Resize
# at eval to match training-time RandomResizedCrop centre). Embed it in the run
# tag so A/B comparisons (crop1.00 vs crop0.95) on the same ckpt don't collide.
CROP_TAG="$(printf 'crop%.2f' "${ROBOTWIN_EVAL_CROP_RATIO:-1.00}")"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_${HOSTNAME_TAG}_tpg${TASKS_PER_GPU}_q${QUALITY}_${CROP_TAG}_${CKPT_TAG}${SWEEP_TAG_SAFE:+_sweep-${SWEEP_TAG_SAFE}}"
OUTPUT_DIR="./evaluate_results/robotwin/robotwin_uncond_3cam_384_1e-4/${RUN_TAG}"

# ----- banner ---------------------------------------------------------
echo "==============================================================================="
echo " RoboTwin 8-GPU eval"
echo "==============================================================================="
echo "  PROJECT_ROOT  : ${PROJECT_ROOT}"
echo "  CKPT          : ${CKPT}"
echo "  TASK          : ${TASK}"
echo "  DATASET_STATS : ${DATASET_STATS}"
echo "  NUM_GPUS      : ${NUM_GPUS}"
echo "  TASKS_PER_GPU : ${TASKS_PER_GPU}"
echo "  EPISODES      : ${EPISODES} per task per setting"
echo "  QUALITY       : ${QUALITY}  (spp=${ROBOTWIN_RT_SPP} path_depth=${ROBOTWIN_RT_PATH_DEPTH} denoiser=${ROBOTWIN_RT_DENOISER})"
echo "  EVAL_CROP     : ${ROBOTWIN_EVAL_CROP_RATIO:-1.0}  (1.0 = off; 0.95 mirrors crop_jitter train-time RRC centre)"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  HOST          : $(hostname)"
echo "  ENV           : $(which python)"
echo "==============================================================================="
echo ""

# ----- launch ---------------------------------------------------------
# Remaining args are forwarded verbatim as Hydra overrides so callers can pass
# through e.g. EVALUATION.task_name=... while still using the QUALITY-preset RT
# env wiring above. If arg4 was numeric it was consumed as EPISODES above.
EXTRA_OVERRIDES=( "${@:${EXTRA_OVERRIDE_START}}" )
exec python experiments/robotwin/run_robotwin_manager.py \
  task="${TASK}" \
  ckpt="${CKPT}" \
  EVALUATION.dataset_stats_path="${DATASET_STATS}" \
  EVALUATION.output_dir="${OUTPUT_DIR}" \
  EVALUATION.eval_num_episodes="${EPISODES}" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${TASKS_PER_GPU}" \
  "${EXTRA_OVERRIDES[@]}"
