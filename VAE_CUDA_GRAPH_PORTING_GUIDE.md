# VAE CUDA Graph 优化迁移指南

本文档总结 FastWAM 中 VAE encode CUDA Graph 优化的核心改动、实现思路，以及迁移到其他项目时应该如何改。

## 核心改动位置

FastWAM 当前 graph 优化主要改在 3 个地方：

1. VAE 内部 CUDA Graph 实现

```text
src/fastwam/models/wan22/wan_video_vae.py
```

核心函数：

- `_get_encode_cuda_graph_state()`：创建并缓存 CUDA Graph。
- `_try_cuda_graph_encode()`：用 graph replay 替代原来的逐 sample encode。
- `encode()`：在原始 encode loop 前尝试走 graph 路径。

2. Trainer 里打开开关

```text
src/fastwam/trainer.py
```

读取配置：

```python
self.vae_encode_cuda_graph = bool(getattr(cfg, "vae_encode_cuda_graph", False))
```

训练开始时设置 VAE flag：

```python
setattr(vae, "_encode_cuda_graph_enabled", self.vae_encode_cuda_graph)
```

3. 配置默认关闭

```text
configs/train.yaml
```

```yaml
vae_encode_cuda_graph: false
```

## 修改思路

原始 VAE encode 逻辑大致是：

```python
hidden_states = []
for video in videos:
    video = video.unsqueeze(0)
    hidden_state = self.single_encode(video, device)
    hidden_states.append(hidden_state.squeeze(0))
hidden_states = torch.stack(hidden_states)
```

也就是说，一个 batch 里有多个视频，但 VAE 实际是逐个 sample encode。每个 sample encode 都会触发大量 Python 调度和 CUDA kernel launch。

CUDA Graph 的改法是：

1. 准备一个固定 shape 的 `static_input`。
2. capture 一次原始的单样本 VAE encode。
3. 后续每个 sample 只执行：
   - 把当前 sample copy 到 `static_input`
   - `graph.replay()`
   - 把 `static_output` copy 到 batch output

核心 replay 逻辑：

```python
for idx in range(video.shape[0]):
    static_input.copy_(video[idx:idx + 1])
    graph.replay()
    output[idx].copy_(static_output[0])
```

这个方案不改变 VAE 数学计算，也没有把 VAE 改成 batch convolution，所以可以保持和原始逐 sample encode 字节级一致。

## 关键实现细节

### 1. graph cache 按 shape/dtype/device 区分

FastWAM 当前 cache key：

```python
key = (str(video.device), str(video.dtype), tuple(video.shape[1:]))
```

迁移到其他项目时，也应该至少按以下内容缓存：

```python
(device, dtype, input_shape_without_batch)
```

这样不同输入形状不会误用同一个 graph。

### 2. capture 前需要 warmup

FastWAM 当前实现：

```python
warmup_stream = torch.cuda.Stream(device=video.device)
current_stream = torch.cuda.current_stream(device=video.device)
warmup_stream.wait_stream(current_stream)

with torch.cuda.stream(warmup_stream):
    for _ in range(3):
        static_input.copy_(video[:1])
        warmup_output = self.model.encode(static_input, scale)

current_stream.wait_stream(warmup_stream)
```

原因是 CUDA Graph capture 前最好先跑几次，让 CUDA kernel、cuDNN plan、内存分配等准备好。否则 capture 期间可能遇到不允许的动态行为。

### 3. capture 的必须是原始单样本 encode

FastWAM 当前 capture：

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, capture_error_mode="thread_local"):
    static_output = self.model.encode(static_input, scale)
```

注意：这里 capture 的是原始 `self.model.encode(...)`，没有改 dtype、device、layout、batch 维度。

### 4. 必须保存和恢复 CUDA RNG

这是这次改动里最关键的安全点之一：

```python
rng_state = torch.cuda.get_rng_state(video.device)
try:
    ...
finally:
    torch.cuda.set_rng_state(rng_state, video.device)
```

原因：VAE encode 本身不应该消耗训练里的随机数。CUDA Graph capture/replay 可能影响 CUDA RNG 状态，所以必须恢复，否则后面的 `torch.randn_like(input_latents)` 会变，训练数值就不一致。

### 5. graph 异常时回退 baseline

FastWAM 当前逻辑：

```python
except Exception:
    setattr(self, "_encode_cuda_graph_failed", True)
    return None
```

然后 `encode()` 会继续走原来的 baseline 路径。

迁移到其他项目时，建议在 fallback 时打印 warning，避免 graph 静默失败后你误以为还在加速。

## 迁移到其他项目的改法

先找目标项目里类似这样的瓶颈：

```python
outputs = []
for x in batch:
    y = expensive_model(x.unsqueeze(0))
    outputs.append(y.squeeze(0))
outputs = torch.stack(outputs)
```

然后按下面步骤改。

### 1. 加一个默认关闭开关

```yaml
vae_encode_cuda_graph: false
```

### 2. 在模块里加 graph cache 和 enable flag

```python
self._encode_cuda_graph_cache = {}
self._encode_cuda_graph_enabled = False
```

### 3. 实现 `_get_cuda_graph_state(x)`

```python
def _get_cuda_graph_state(self, x):
    cache = getattr(self, "_encode_cuda_graph_cache", None)
    if cache is None:
        cache = {}
        setattr(self, "_encode_cuda_graph_cache", cache)

    key = (str(x.device), str(x.dtype), tuple(x.shape[1:]))
    if key in cache:
        return cache[key]

    static_input = torch.empty((1, *x.shape[1:]), device=x.device, dtype=x.dtype)

    warmup_stream = torch.cuda.Stream(device=x.device)
    current_stream = torch.cuda.current_stream(device=x.device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_input.copy_(x[:1])
            warmup_output = original_single_sample_fn(static_input)
    current_stream.wait_stream(warmup_stream)
    del warmup_output

    static_input.copy_(x[:1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        static_output = original_single_sample_fn(static_input)

    cache[key] = {
        "graph": graph,
        "input": static_input,
        "output": static_output,
    }
    return cache[key]
```

这里的 `original_single_sample_fn(static_input)` 要替换成目标项目里原始的单样本 encode / forward 函数。

### 4. 实现 graph replay

```python
def _try_cuda_graph_encode(self, x):
    if not getattr(self, "_encode_cuda_graph_enabled", False):
        return None
    if getattr(self, "_encode_cuda_graph_failed", False):
        return None
    if not isinstance(x, torch.Tensor) or x.ndim < 2:
        return None
    if not torch.cuda.is_available():
        return None

    x = x.to(self.device)
    if x.device.type != "cuda" or x.shape[0] == 0:
        return None

    rng_state = torch.cuda.get_rng_state(x.device)
    try:
        state = self._get_cuda_graph_state(x)
        static_input = state["input"]
        static_output = state["output"]

        output = torch.empty(
            (x.shape[0], *static_output.shape[1:]),
            device=x.device,
            dtype=static_output.dtype,
        )

        for i in range(x.shape[0]):
            static_input.copy_(x[i:i + 1])
            state["graph"].replay()
            output[i].copy_(static_output[0])

        return output
    except Exception:
        setattr(self, "_encode_cuda_graph_failed", True)
        return None
    finally:
        torch.cuda.set_rng_state(rng_state, x.device)
```

### 5. 在原始 encode 前插入 graph 尝试

```python
if self._encode_cuda_graph_enabled:
    graph_out = self._try_cuda_graph_encode(videos)
    if graph_out is not None:
        return graph_out

# fallback: original encode
```

## 迁移时必须满足的条件

这个方法适合：

- 模型权重在 graph replay 期间不变，或者至少参数 storage 不变。
- 输入 shape 固定或种类很少。
- 原始逻辑是逐 sample 调同一个重模型。
- 目标函数没有依赖 Python side effect。
- 目标函数不应该消耗训练 RNG；如果会影响 RNG，必须像这里一样保存/恢复。
- 你要求数值严格一致，所以不能顺手加 channels-last、batch conv、TF32、不同 autocast scope。

不适合：

- 输入 shape 每步都变很多。
- encode 内部有大量 CPU 控制流、文件 IO、print、同步 `.item()`。
- 模型本身在训练且参数结构或 storage 会变。
- 你想把逐样本计算改成真正 batch 计算。那可能更快，但不一定数值一致。

## 必做验证

迁移后至少做下面几类测试。

### 1. 真实数据输出字节级一致

```python
torch.equal(baseline_output, graph_output)
max_abs == 0
```

### 2. graph 后的随机数一致

```python
baseline_noise = torch.randn_like(latents_after_baseline)
graph_noise = torch.randn_like(latents_after_graph)
torch.equal(baseline_noise, graph_noise)
```

### 3. 从同一个 checkpoint resume 后比较 loss

分别跑 baseline 和 graph，检查每个 step 的 loss 是否一致。

### 4. 比较最终 checkpoint

保存最终 checkpoint，比较 SHA256。

### 5. 多卡真实训练稳定性

多卡真实跑 100-1000 steps，检查：

- 是否有 NCCL/CUDA error。
- 是否发生 graph fallback。
- 是否出现 GPU 长期空载。
- 速度是否稳定提升。

## 一句话总结

这次 graph 优化的本质是：不改变 VAE 的单样本 encode 数学逻辑，只把这段固定 GPU 操作录下来反复 replay，从而减少大量 CUDA kernel launch 调度开销；为了保证训练数值不变，必须按 shape 缓存 graph，并保存/恢复 CUDA RNG。
