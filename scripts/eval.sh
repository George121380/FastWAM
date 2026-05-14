#!/usr/bin/env bash
# ==============================================================================
# RoboTwin 8-GPU evaluation — one-shot launcher
# ==============================================================================
# Usage:
#   ./scripts/eval.sh [CKPT] [TASK] [DATASET_STATS]
#
# Examples:
#   # Use defaults (released checkpoint):
#   ./scripts/eval.sh
#
#   # Custom checkpoint:
#   ./scripts/eval.sh /path/to/my_ckpt.pt
#
#   # Full override:
#   ./scripts/eval.sh /path/ckpt.pt robotwin_uncond_3cam_384_1e-4 /path/stats.json
#
# Env overrides:
#   NUM_GPUS=4 TASKS_PER_GPU=1 ./scripts/eval.sh ...
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

# ----- env setup ------------------------------------------------------
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source /share-2/home/vla/miniforge3/etc/profile.d/conda.sh
conda activate fastwam_eval_cu13

# SAPIEN looks for nvidia Vulkan ICD only under /usr/share/vulkan/icd.d/ by default;
# on this host the nvidia ICD lives at /etc/vulkan/icd.d/. Point sapien at it
# explicitly to avoid loading a stale bundled ICD (api 1.2.140).
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

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
RUN_TAG="$(date +%Y%m%d_%H%M%S)_${HOSTNAME_TAG}_tpg${TASKS_PER_GPU}_${CKPT_TAG}"
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
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  HOST          : $(hostname)"
echo "  ENV           : $(which python)"
echo "==============================================================================="
echo ""

# ----- launch ---------------------------------------------------------
exec python experiments/robotwin/run_robotwin_manager.py \
  task="${TASK}" \
  ckpt="${CKPT}" \
  EVALUATION.dataset_stats_path="${DATASET_STATS}" \
  EVALUATION.output_dir="${OUTPUT_DIR}" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${TASKS_PER_GPU}"
