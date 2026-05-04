# VAE CUDA Graph Encode 合并前测试报告

日期：2026-05-02  
仓库：`/share-2/code/fanqilin/peiqi/FastWAM`  
环境：`fastwam-py312-cu13`，PyTorch 2.9.0+cu130，CUDA 13.0.1，8x NVIDIA B300 SXM6 AC

## 总结

我的结论是：建议将 VAE encode CUDA Graph 优化合并，但保持默认关闭。

当前 graph 路径已经通过关键正确性检查：

- 在 64 个真实 batch / 4096 个视频上，VAE latent 输出与 baseline 字节级一致。
- graph encode 之后的 CUDA RNG 状态，以及紧接着执行的 `torch.randn_like(...)` 结果，都与 baseline 字节级一致。
- 从同一个 checkpoint 恢复后，8 卡 baseline 训练和 graph 训练在检查的 step 上产生了完全一致的日志 loss。
- baseline 和 graph 分别保存的 `step_000012.pt` checkpoint SHA256 完全一致。
- 8 卡 200-step benchmark 从 baseline 的 26.82 samples/s 提升到 graph 的 42.44 samples/s。
- 8 卡 1000-step graph soak 完整跑完，最终累计速度 46.81 samples/s，没有 CUDA/NCCL/runtime 错误，也没有观察到长期 GPU 空载。

这份报告只支持 `vae_encode_cuda_graph=true` 这个优化。它不推荐 `vae_encode_channels_last=true` 用于数值严格一致的训练，因为之前已经观察到 channels-last 路径会改变 BF16 输出。

## Graph 实现细节

graph 功能由以下配置控制：

```yaml
vae_encode_cuda_graph: false
vae_encode_channels_last: false
```

Trainer 侧行为：

- `src/fastwam/trainer.py` 读取 `cfg.vae_encode_cuda_graph`。
- 训练启动时设置 `model.vae._encode_cuda_graph_enabled`。
- 默认值是 `false`，所以除非显式开启，否则仍然走原始训练路径。

VAE 侧行为：

- 实现在 `src/fastwam/models/wan22/wan_video_vae.py`。
- `WanVideoVAE.encode(...)` 只在以下条件同时满足时尝试 CUDA Graph 路径：
  - `tiled=False`
  - 输入是 5D tensor
  - CUDA 可用
  - `_encode_cuda_graph_enabled=True`
- graph cache key 是：

```python
(str(video.device), str(video.dtype), tuple(video.shape[1:]))
```

这意味着不同的 `C/T/H/W` 形状会分别 capture，不会复用错误形状的 graph。

graph 路径 capture 的是原始单样本 VAE encode：

```python
static_output = self.model.encode(static_input, scale)
```

replay 时仍然逐样本循环：

```python
for idx in range(video.shape[0]):
    static_input.copy_(video[idx:idx + 1])
    graph.replay()
    output[idx].copy_(static_output[0])
```

关键正确性设计：

- 不把 VAE convolution 改成 batch 维度上的整体计算。
- 不改变 memory format。
- 不开启 channels-last。
- 不改变 dtype/device。
- 保留原始 encoder 的 causal/cache 行为。
- 在 graph capture/replay 前后保存并恢复 CUDA RNG state。
- 使用 `capture_error_mode="thread_local"`，避免干扰前面分布式训练中观察到的 RNG 行为。

这个优化的加速来源是减少重复单样本 VAE encode 过程中的 Python/CUDA launch 调度开销，而不是改变数学计算。

## 正确性测试

### 1. 真实数据 VAE 字节级一致性测试

新增脚本：

```bash
scripts/test_vae_graph_correctness.py
```

测试命令：

```bash
python -u scripts/test_vae_graph_correctness.py \
  --max-batches 64 \
  --batch-size 64 \
  --num-workers 16 \
  --speed-warmup-batches 1 \
  --output-json bench_logs/vae_graph_correctness_64b_nw16.json \
  task=robotwin_uncond_3cam_384_1e-4 \
  output_dir=./runs/graph_premerge/vae_64b_nw16 \
  wandb.enabled=false
```

结果：

```text
ok: true
batches: 64 / 64
videos: 4096
failures: 0
latent_equal: all true
noise_equal: all true
rng_equal: all true
graph_failed: all false
```

同一个测试里的计时结果，排除第一个 warmup batch：

```text
baseline VAE mean: 11754.21 ms / batch
baseline VAE p50 : 11702.83 ms / batch
baseline VAE p90 : 11731.15 ms / batch
graph VAE mean   : 5255.38 ms / batch
graph VAE p50    : 5256.29 ms / batch
graph VAE p90    : 5256.94 ms / batch
encode-only speedup: 2.237x
```

这是目前最强的证据：对于当前真实训练输入形状，graph 路径在数值上是严格一致的。

### 2. 默认关闭 smoke test

使用正常 8 卡训练路径，并显式关闭 graph：

```bash
bash scripts/train_zero1.sh 8 \
  task=robotwin_uncond_3cam_384_1e-4 \
  max_steps=10 log_every=1 save_every=0 eval_every=0 \
  wandb.enabled=false profiling.enabled=false \
  vae_encode_cuda_graph=false vae_encode_channels_last=false \
  num_workers=16
```

在与正式训练一致的 CUDA/NCCL 环境下，该 run 正常完成：

```text
step=10/10
loss=1.2399
samples/s=11.74
graph_enabled: false
errors: false
```

日志中没有出现 graph enabled 相关信息。

说明：之前有一次没有带完整训练 CUDA/NCCL 环境的尝试，在进入训练前的 distributed init 阶段 segfault。我将它视为环境不一致导致的问题，而不是 graph 结果。使用正式环境重跑后测试通过。

### 3. 8 卡 resume 后的 loss 和 checkpoint 一致性

baseline 和 graph 都从同一个状态恢复：

```text
runs/robotwin_uncond_3cam_384_1e-4/2026-05-01_18-31-40/checkpoints/state/step_000010
```

Baseline：

```text
step=11/12 loss=1.2602 loss_action=0.7034 loss_video=0.5568
step=12/12 loss=1.1479 loss_action=0.6064 loss_video=0.5415
```

Graph：

```text
step=11/12 loss=1.2602 loss_action=0.7034 loss_video=0.5568
step=12/12 loss=1.1479 loss_action=0.6064 loss_video=0.5415
```

最终保存的 weights checkpoint 字节级一致：

```text
a59e8b174160ce1b439d0a0cd044ea7e7b71dc890f4d265e4d4f381b53a15105  baseline step_000012.pt
a59e8b174160ce1b439d0a0cd044ea7e7b71dc890f4d265e4d4f381b53a15105  graph    step_000012.pt
```

这说明从同一个训练状态出发，graph 路径在被检查的训练 step 内没有改变模型更新。

## 性能和稳定性测试

所有 8 卡训练测试都使用：

```bash
num_workers=16
save_every=0
eval_every=0
wandb.enabled=false
profiling.enabled=false
vae_encode_channels_last=false
NCCL_DEBUG=WARN
```

### 1. 200-step baseline vs graph

Baseline 命令：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  max_steps=200 log_every=10 save_every=0 eval_every=0 \
  wandb.enabled=false profiling.enabled=false \
  vae_encode_cuda_graph=false vae_encode_channels_last=false \
  num_workers=16
```

Graph 命令：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  max_steps=200 log_every=10 save_every=0 eval_every=0 \
  wandb.enabled=false profiling.enabled=false \
  vae_encode_cuda_graph=true vae_encode_channels_last=false \
  num_workers=16
```

结果：

| Run | 最终累计 samples/s | 日志区间 mean samples/s | 日志区间 p50 samples/s | 错误 |
| --- | ---: | ---: | ---: | --- |
| Baseline, 200 steps | 26.82 | 28.97 | 29.09 | 无 |
| Graph, 200 steps | 42.44 | 48.97 | 43.76 | 无 |

step 200 时的累计速度提升：

```text
42.44 / 26.82 = 1.58x
```

graph run 期间，`nvidia-smi` 采样显示 8 张 GPU 都在工作。之前 `num_workers=32` 时出现的“只有 GPU0 长期繁忙”的现象，在 `num_workers=16` 下没有复现。

### 2. 1000-step graph soak

命令：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  max_steps=1000 log_every=50 save_every=0 eval_every=0 \
  wandb.enabled=false profiling.enabled=false \
  vae_encode_cuda_graph=true vae_encode_channels_last=false \
  num_workers=16
```

结果：

```text
completed: yes
final step: 1000/1000
final loss: 0.1442
final loss_action: 0.0165
final loss_video: 0.1276
final cumulative samples/s: 46.81
log-interval mean samples/s: 48.19
log-interval p50 samples/s: 48.39
last interval samples/s: 48.85
errors: none
```

观察到的稳定性：

- 没有 CUDA Graph replay error。
- 没有 CUDA runtime error。
- 没有 NCCL error。
- 没有 Python traceback 或子进程失败。
- 没有持续性的 GPU rank 空载。
- 采样时每张 GPU 显存稳定在约 236 GB。
- soak 过程中多次 `nvidia-smi` 采样显示 8 张 GPU 都活跃，很多时候接近 100% util。

## 实验产物

关键日志和输出：

```text
bench_logs/vae_graph_correctness_64b_nw16.json
bench_logs/graph_disabled_smoke_env_20260501_183140.log
bench_logs/graph_loss_resume_base_20260501_184443.log
bench_logs/graph_loss_resume_graph_20260501_185359.log
bench_logs/graph_bench_base_200_20260501_190340.log
bench_logs/graph_bench_graph_200_20260501_201202.log
bench_logs/graph_soak_1000_20260501_205740.log
```

重要 run 目录：

```text
runs/robotwin_uncond_3cam_384_1e-4/2026-05-01_18-31-40
runs/robotwin_uncond_3cam_384_1e-4/2026-05-01_18-44-43
runs/robotwin_uncond_3cam_384_1e-4/2026-05-01_18-53-59
runs/robotwin_uncond_3cam_384_1e-4/2026-05-01_19-03-40
runs/robotwin_uncond_3cam_384_1e-4/2026-05-01_20-12-02
runs/robotwin_uncond_3cam_384_1e-4/2026-05-01_20-57-40
```

## 残余风险和建议

残余风险：

- graph 是 shape-specific 的。如果未来出现新的 `C/T/H/W` 形状，会触发新的 graph capture。这是预期行为，但如果后续配置改变视频形状，需要重新观察。
- 当前代码在 graph 异常后会回退到 baseline。测试中没有发生 fallback，但公开合并前最好在 fallback 时打印 warning，避免速度回退不可见。
- CUDA Graph capture 会额外占用静态 input/output/cache 显存。B300 上不是约束，但这仍然是真实资源开销。
- 当前验证覆盖的是 `robotwin_uncond_3cam_384_1e-4` 形状和 ZeRO1 8 卡训练路径。其他 dataset shape 或 tiled encode 路径需要重新测试。
- `vae_encode_channels_last=true` 对严格数值一致仍然不安全，不应该在正式严格训练中开启。

推荐 merge 策略：

- 合并 `vae_encode_cuda_graph`，但默认保持 `false`。
- 严格数值训练中按以下方式开启：

```bash
vae_encode_cuda_graph=true vae_encode_channels_last=false num_workers=16
```

- 长训时保持 `NCCL_DEBUG=WARN`，除非正在主动调试 NCCL。
- 使用修正后的 HCA export 写法：

```bash
export NCCL_IB_HCA=roce_vf_rail0,roce_vf_rail1,roce_vf_rail2,roce_vf_rail3,roce_vf_rail4,roce_vf_rail5,roce_vf_rail6,roce_vf_rail7
```

总体结论：graph 优化已经有足够证据支持合并，并可以在上述限制下用于大规模训练。
