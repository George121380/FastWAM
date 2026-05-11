#!/usr/bin/env bash
# scripts/profile_sweep.sh
#
# Sequential 8-cell profiling sweep:
#   bs ∈ {16, 32}, vae_encode_cuda_graph ∈ {0, 1}, n_gpu ∈ {1, 8}
# Each cell: 220 steps single-task, drops first 2 flushes (20 steps) as warmup,
# leaves 200 measured steps over 20 records in timing.jsonl.
#
# Flags:
#   --skip-overhead-check : skip Step 4 (not recommended)
#   --smoke               : run only the 30-step Step 0 smoke and exit
#   --dry-run             : print commands, do not execute

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Strip inherited proxy env — the cluster proxy 10.25.7.231:8888 is dead and
# huggingface_hub / network probes hang on it during init.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
# Belt-and-braces: prevent huggingface_hub from any network probe during init.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SKIP_OVERHEAD=0
SMOKE_ONLY=0
DRY=0

for arg in "$@"; do
  case "${arg}" in
    --skip-overhead-check) SKIP_OVERHEAD=1 ;;
    --smoke) SMOKE_ONLY=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

log() { echo "[sweep] $*"; }

run() {
  if [[ "${DRY}" == "1" ]]; then
    echo "+ $*"
  else
    eval "$@"
  fi
}

# ---------- Env preamble ----------
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "fastwam-py312-cu13" ]]; then
  log "activating conda env fastwam-py312-cu13"
  source /share-2/home/vla/miniforge3/etc/profile.d/conda.sh
  conda activate fastwam-py312-cu13
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/13.0.1 || true
fi

# Single-node sweep — IB is known-flaky on this cluster (per user memory) and
# segfaults during init. NVLink P2P is fine; leave it on for 8-GPU NCCL.
export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA NCCL_IB_ROCE_VERSION_NUM NCCL_IB_RETRY_CNT NCCL_IB_TIMEOUT
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ---------- Smoke (Step 0) ----------
run_smoke() {
  local rid="smoke_$(date +%Y%m%d_%H%M%S)"
  local out="runs/profile/${rid}"
  log "smoke: 30-step single-GPU bs=16, output=${out}"
  run "RUN_ID=${rid} bash scripts/train_zero1.sh 1 \
    task=robotwin_uncond_3cam_384_1e-4 \
    output_dir=${out} \
    batch_size=16 \
    max_steps=30 log_every=10 \
    save_every=0 eval_every=0 save_final=false \
    wandb.enabled=false \
    'profile_phases=[data_load,vae_encode,dit_forward,loss_compute,backward,optimizer]'"

  log "smoke: validating ${out}/timing.jsonl ..."
  python - <<EOF
import json, os, sys
p = "${out}/timing.jsonl"
if not os.path.exists(p):
    print(f"FAIL: {p} not found"); sys.exit(1)
records = [json.loads(l) for l in open(p) if l.strip()]
if len(records) != 3:
    print(f"FAIL: expected 3 records, got {len(records)}"); sys.exit(1)
warm = [r for r in records if r.get('is_warmup')]
if len(warm) != 2:
    print(f"FAIL: expected 2 warmup records, got {len(warm)}"); sys.exit(1)
required_phases = {'data_load','vae_encode','dit_forward','loss_compute','backward','optimizer'}
final = records[-1]
have = set(final.get('phases', {}).keys())
if not required_phases.issubset(have):
    print(f"FAIL: missing phases: {required_phases - have}"); sys.exit(1)
print("smoke validation: PASS")
EOF
}

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  run_smoke
  exit 0
fi

# ---------- Step 4 overhead check ----------
if [[ "${SKIP_OVERHEAD}" == "0" ]]; then
  log "running overhead check (Step 4)..."
  run "bash scripts/profile_overhead_check.sh"
fi

# ---------- Step 5: 16-cell sweep (2x2x2x2) ----------
# Variables: bs ∈ {16, 32}, vae_encode_cuda_graph ∈ {0, 1},
#            n_gpu ∈ {1, 8}, vae_encode_batch_parallel ∈ {0, 1}
# Order: cheapest -> most expensive (single-GPU first; bp=0 then bp=1)
declare -a CELLS=(
  # bs graph n_gpu bp
  "16 0 1 0"
  "32 0 1 0"
  "16 1 1 0"
  "32 1 1 0"
  "16 0 1 1"
  "32 0 1 1"
  "16 1 1 1"
  "32 1 1 1"
  "16 0 8 0"
  "32 0 8 0"
  "16 1 8 0"
  "32 1 8 0"
  "16 0 8 1"
  "32 0 8 1"
  "16 1 8 1"
  "32 1 8 1"
)

for cell in "${CELLS[@]}"; do
  read -r BS GRAPH N BP <<< "${cell}"
  RID="profile_bs${BS}_graph${GRAPH}_n${N}_bp${BP}_$(date +%Y%m%d_%H%M%S)"
  OUT="runs/profile/${RID}"
  EXTRAS=""
  if [[ "${GRAPH}" == "1" ]]; then
    EXTRAS+="vae_encode_cuda_graph=true num_workers=16 "
  fi
  if [[ "${BP}" == "1" ]]; then
    EXTRAS+="vae_encode_batch_parallel=true "
  fi
  log "cell bs=${BS} graph=${GRAPH} n=${N} bp=${BP} -> ${RID}"
  # 30 steps total: steps 1-10 are warmup (is_warmup=true; covers cuda_graph
  # capture + JIT). Steps 21-30 are the primary measurement window for
  # summary outputs; steps 11-20 remain available for anomaly comparison.
  run "RUN_ID=${RID} bash scripts/train_zero1.sh ${N} \
    task=robotwin_uncond_3cam_384_1e-4 \
    output_dir=${OUT} \
    batch_size=${BS} \
    max_steps=30 log_every=10 profile_warmup_flushes=1 \
    save_every=0 eval_every=0 save_final=false \
    wandb.enabled=false \
    'profile_phases=[data_load,vae_encode,dit_forward,loss_compute,backward,optimizer]' \
    ${EXTRAS}"
  sleep 1
done

log "sweep complete. summarizing..."
run "python scripts/profile_summarize.py --format md > runs/profile/sweep.md"
run "python scripts/profile_summarize.py --format csv > runs/profile/sweep.csv"
log "wrote runs/profile/sweep.md and runs/profile/sweep.csv"
