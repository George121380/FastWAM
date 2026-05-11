#!/bin/bash
#SBATCH --job-name=fastwam-mn
#SBATCH --nodes=8                      # 节点数；要换数只动这一行
#SBATCH --ntasks-per-node=1            # 每节点一个 launcher（accelerate 内部 spawn 8 个进程）
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128            # 留够给 num_workers=16 × 8 GPU
#SBATCH --output=slurm-%j.out
#SBATCH --exclusive
#SBATCH --exclude=gpu03,gpu05    # gpu05: slurm DOWN+NOT_RESPONDING; gpu03,gpu06: 用户排除
#SBATCH --time=96:00:00                # 4 天 wall-time 上限，5 epoch 实测 ~80h，留点余量
set -euo pipefail

# ---- 头节点地址（用 NodeList 第一个节点）----
# 注意：scontrol show hostnames 返回的是 *hostname*。多数集群上 hostname 直接可解析到训练网络。
# 但如果 hostname 解析到管理网 (mgmt) 而 NCCL/torch rendezvous 必须走 bond0，
# 这里需要换成 rank-0 节点在 bond0 上的 IP（例如 `getent hosts ${headnode}-bond0` 或预先映射）。
# 总之：MASTER_ADDR 必须是所有节点都能访问的 rank-0 地址，且最好和训练网络（bond0/RoCE）一致。
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# 注意：默认 29500 在 pod11 的部分 GPU 节点上常被其他 dev session 占用（例如本机 vla 的 debugpy 训练）。
# 换用 29555（RUN_ID 同步端口会自动 = MASTER_PORT + 11 = 29566）。两者必须在节点间放行。
export MASTER_PORT=29555
export NNODES="$SLURM_NNODES"

# ---- module + NCCL/IB 设置（每个 srun step 都会继承这些环境变量）----
module load cuda/13.0.1
export NCCL_IB_HCA=roce_vf_rail0,roce_vf_rail1,roce_vf_rail2,roce_vf_rail3,roce_vf_rail4,roce_vf_rail5,roce_vf_rail6,roce_vf_rail7
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=23
export NCCL_DEBUG=INFO

source activate fastwam-py312-cu13

# ---- 切到 repo 根目录（否则相对路径 scripts/train_zero1.sh 找不到）----
# 用 SLURM_SUBMIT_DIR 假定从 repo 根 sbatch 提交；如果不是，换成绝对路径，例如：
# cd /share/path/to/FastWAM
cd "$SLURM_SUBMIT_DIR"

# ---- 每个节点起一份 launcher（关键：必须用 srun 在所有节点执行，sbatch 主体默认只在头节点跑一次）----
# 显式给 --nodes/--ntasks/--ntasks-per-node，避免不同集群 srun 默认行为不一致。
# NODE_RANK 来自 SLURM_NODEID（0-indexed 节点序号；不要用 SLURM_PROCID，那是任务全局序号）。
#
# 如果实际日志里 torch.cuda.device_count() 不是 8，说明本集群在 --ntasks-per-node=1 下没把 8 张
# GPU 都暴露给单个 launcher。可在下面 srun 后面加 --gpus-per-task=8 或按本集群规范加 --gres=gpu:8。
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export NODE_RANK="$SLURM_NODEID"
  bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 vae_encode_cuda_graph=true vae_encode_cuda_graph_lanes=1 num_workers=16 batch_size=16 wandb.enabled=true
'

# ---- 用法 ----
# 正式训练：
#   sbatch scripts/slurm/train_multinode.sbatch
#
# 多节点最小验证（推荐先跑一次，几分钟内就能确认链路通畅）：把上面 srun 块里那条命令的尾部加上：
#   max_steps=30 save_every=20 eval_every=0 log_every=1
# 即：
#   bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 vae_encode_cuda_graph=true \
#     num_workers=16 max_steps=30 save_every=20 eval_every=0 log_every=1
#
# 验证通过的标志：
#   - 头节点 stdout 出现 [run_id_sync] mode=tcpstore ... run_id=<X>，两节点 run_id 必须相同
#   - 头节点 [launch] 行 num_machines=2 machine_rank=0；副节点 machine_rank=1
#   - DeepSpeed 启动日志 world_size=16
#   - runs/.../checkpoints/state/step_xxxxxx/ 下有 16 个 shard
