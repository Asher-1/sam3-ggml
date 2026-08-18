# SAM3.cpp 性能优化计划

## 目标

先将 SAM3 F16 的 PVS 热推理压到 500-600 ms，并保持自然图像语义精度；
随后再以同一算子路径推进其他精度和 PCS。

## 当前状态

### SAM3 PVS 热推理基线（RTX 3060，2026-08-17）

统一口径：`sam3-f16.gguf`、`tests/cat.jpg`、点提示 `(315, 250)`、CUDA、
1008 输入分辨率，先 warmup 再重复采样；模型加载时间不计入推理时间。

| 实现 | encode p50 | segment p50 | 合计 p50 | 说明 |
|------|-----------:|------------:|---------:|------|
| sam3.cpp（原始 F16 热路径） | 829.338 ms | 32.179 ms | **861.517 ms** | 优化前同口径 |
| sam3.cpp（优化后） | 566.167 ms | 31.925 ms | **598.092 ms** | 2 warmup + 7 repeat |
| 官方 PyTorch 计算图 | 394.2 ms | 8.2 ms | **402.4 ms** | BF16 autocast；随机权重，仅测性能 |

本机已登录 Hugging Face，但 `facebook/sam3` 门控申请仍在审核；checkpoint
请求返回 HTTP 403。因此
PyTorch 行只能代表官方代码、相同张量形状和算子路径的耗时，不能用于精度对比。
获得权限后运行 `benchmarks/bench_pytorch_sam3.py` 即可测官方权重。

本轮已落地：

- PVS 的高分辨率 tracker feature 改为 GPU 内直接拷贝，避免约 106 MB 的
  GPU -> CPU -> GPU pageable host staging。
- detector/tracker 双 neck 共享同一个 20 MiB ViT 布局转换结果。
- 约 111 MB 的四层 neck PE 改为首次 PCS 时延迟初始化；首次 PVS encode
  从约 1.20 s 降到约 1.00 s，首次 encode + point decode 约 1.07 s。
- 新增 `sam3_encode_image_pvs()`，完整 SAM 3 做点/框分割时不再构建 detector
  neck；原 `sam3_encode_image()` 保持 PCS 所需的完整语义。
- ViT matmul 中间结果使用 F16，CUDA 上融合自定义频率 RoPE、F16->F32
  RoPE、window partition 以及 QKV layout + Q/K RoPE，减少中间显存流量和
  kernel launch。
- 基准程序同时测 encode/segment 的 warmup、repeat、平均值和 p50。

最终热推理 p50 从 861.517 ms 降到 598.092 ms，降低 30.6%。RTX 3060 会随
P-state 出现约 590-620 ms 波动，因此性能门槛按热态多次采样的 p50 判断，
不使用单次最小值。

精度边界不是逐 bit 一致，而是语义输出一致：检测数和 box 相同、score 差值
不超过 0.001、真实图像 mask IoU 不低于 0.999。

| 输入 | mask IoU（优化前 vs 优化后） | score | box | 结论 |
|------|------------------------------:|-------|-----|------|
| `cat.jpg` | 0.999033 | 0.9525 / 0.9527 | 相同 | 通过 |
| `llama.jpg` | 1.000000 | 0.8100 / 0.8103 | 相同 | 通过 |
| 随机噪声 | 0.987616 | 0.5075 / 0.4998 | 阈值附近变化 | 仅作数值敏感性记录 |

随机噪声恰好落在决策阈值附近，不代表自然图像精度回退，也不能被写成通过。
当前剩余差距主要来自 MLP 的独立 bias/activation kernel 与通用 ggml 图调度；
继续追赶官方 PyTorch 的 402 ms 需要进一步融合 `matmul + bias + activation`。

### 已完成的工作

#### 1. Vulkan conv_transpose_2d 快速路径修复 (2026-08-16)

**问题定位：**
- Bug 1: Reorder shader 调试探针覆盖前 4 个权重元素
- Bug 2: Conv2D push constants 中 `nb02` 和 `nb03` 计算错误
  - 错误值：`nb02=1, nb03=Cin`
  - 正确值：`nb02=Cin, nb03=Cin*Cout`
- Bug 3: 残留调试代码（环境变量检查、dump 输出）

**修复内容：**
- 移除 `conv_transpose_2d_reorder.comp` 中的调试探针
- 修正 `ggml-vulkan.cpp` 中的 push constants 计算
- 清除所有 `GGML_VK_FAST_DEBUG` 和 `GGML_VK_DISABLE_FAST` 代码

**交付物：**
- 干净的 patch 文件：`ggml-patches/0001-sam3-ggml-combined.patch`
- 编译验证通过：`build-vulkan/libsam3.so` (909K)

**技术细节：**
```cpp
// 修正后的权重布局理解
// Reorder 后：[Cin, Cout, 4] 即逻辑形状 [1, 1, Cin, K]，K=4*Cout
// Conv_transpose_2d shader 交换 Cin↔Cout，看到 [1, 1, Cout, K]
// 正确的 stride：nb01=1, nb02=Cin, nb03=Cin*Cout
p.nb01 = 1;          // ci
p.nb02 = Cin;        // co
p.nb03 = Cin * Cout; // phase
```

### 当前性能基线 (RTX 3060, Linux, 2026-08-17 更新)

**单图分割（cat.jpg 单点 prompt，2 次取 min，CUDA 后端）：**

| 模型 | load ms | encode ms | seg ms | 总计 ms | score |
|------|--------:|----------:|-------:|--------:|------:|
| sam2_hiera_tiny_f16 | 155 | 419 | 105 | **679** | 0.959 |
| sam2.1_hiera_tiny_q8_0 | 129 | 447 | 108 | **684** | 0.945 |
| sam2.1_hiera_base_plus_q8_0 | 195 | 541 | 95 | **831** | 0.953 |
| sam3-visual-f16 | 322 | 921 | 107 | **1350** | 0.953 |
| sam3-f16（当前 PVS 路径） | 807 | 566 | 32 | **1405** | 0.953 |
| sam3-f32 | 1285 | 1494 | 105 | **2884** | 0.953 |

**视频跟踪（test_video.mp4 10 帧，Track/fr 均值，CUDA 后端）：**

| 模型 | Track/fr 优化前 (ms) | Track/fr 优化后 (ms) | 加速比 |
|------|---------------------|---------------------|--------|
| sam2.1_hiera_tiny_q8_0 | ~697 | **260** | 2.7x |
| sam2.1_hiera_tiny_f16 | ~712 | **261** | 2.7x |
| sam2.1_hiera_base_plus_q8_0 | ~1475 | **387** | 3.8x |
| sam2_hiera_base_plus_f16 | ~1240 | **415** | 3.0x |
| sam3-f16 | ~7.7s (Metal) | **1573** | — |

**Metal 后端 (Apple M4 Pro)：**
- SAM 3 f16: 7.7s track/frame
- SAM 2.1 Tiny f16: 0.8s track/frame

## 性能优化路线图

### 阶段 1: 性能分析与基线建立 (1-2 周)

#### 1.1 PyTorch 基线测量
```bash
# 运行 PyTorch 官方 SAM2 基准测试
cd benchmarks
python bench_pytorch_sam2.py --model sam2.1_hiera_tiny --frames 10
python bench_pytorch_sam2.py --model sam2.1_hiera_small --frames 10
python bench_pytorch_sam2.py --model sam2.1_hiera_base_plus --frames 10
python bench_pytorch_sam2.py --model sam2.1_hiera_large --frames 10
```

**目标指标：**
- Frame 0 延迟 (encode + prompt + propagate)
- Track/frame 延迟 (稳态传播)
- 内存占用
- GPU 利用率

#### 1.2 SAM3.cpp 详细性能分析
```bash
# 使用内置基准测试工具
./build-vulkan/examples/sam3_benchmark --gpu-only --filter tiny
./build-cuda/examples/sam3_benchmark --gpu-only --filter tiny

# 使用 profiling 工具
./build-vulkan/examples/sam3_profile_sam3 --model ../models/sam3-f16.gguf
```

**分析维度：**
- 各阶段耗时分布 (image encoding, prompt encoding, mask decoding, propagation)
- 各算子耗时 (conv2d, conv_transpose_2d, matmul, flash_attention, etc.)
- 内存带宽利用率
- GPU 计算单元利用率

#### 1.3 瓶颈识别
**预期瓶颈区域：**
1. **Conv_transpose_2d** - 上采样操作（已通过快速路径优化）
2. **Flash Attention** - 长序列注意力计算
3. **Matrix Multiplication** - 大规模矩阵乘法
4. **Memory Copy** - CPU-GPU 数据传输
5. **Kernel Launch Overhead** - 小算子频繁启动

### 阶段 2: 关键算子优化 (2-4 周)

#### 2.1 Conv_transpose_2d 优化 (已完成)
- ✅ CUDA: cuBLAS 加速的 4-phase 快速路径
- ✅ Vulkan: 1x1 GEMM + reorder/interleave shader
- **预期加速**: 10-50x (相比朴素实现)

#### 2.2 Flash Attention 优化
**当前状态：**
- 已支持 head_dim=16 和 head_dim=56 (通过 patch 0001/0002)
- 使用 ggml 的 flash_attn_ext 实现

**优化方向：**
1. **Tile-based 计算** - 减少 HBM 访问
2. **Async copy** - 重叠计算与内存传输
3. **Warp specialization** - 不同 warp 负责不同阶段
4. **Quantized KV cache** - 减少内存带宽需求

**参考实现：**
- NVIDIA cutlass flash attention
- PyTorch's xformers
- ggml 的 CUDA/Vulkan flash attention 实现

#### 2.3 Matrix Multiplication 优化
**当前状态：**
- CUDA: 使用 cuBLAS
- Vulkan: 使用 ggml 的 matmul 实现

**优化方向：**
1. **Batched GEMM** - 合并多个小矩阵乘法
2. **Quantized weights** - Q4/Q8 量化减少带宽
3. **Tiling strategies** - 针对特定 GPU 架构优化
4. **Async execution** - 重叠计算与数据传输

#### 2.4 Memory Layout 优化
**问题：**
- 频繁的 tensor reshape/transpose
- 非连续内存访问模式
- CPU-GPU 数据传输瓶颈

**优化方向：**
1. **Fused operations** - 合并 reshape + compute
2. **Pinned memory** - 使用 CUDA pinned memory
3. **Async transfer** - 重叠传输与计算
4. **Memory pool** - 减少内存分配开销

### 阶段 3: 系统级优化 (2-3 周)

#### 3.1 Pipeline Parallelism
**思路：**
- 将模型分成多个阶段
- 不同阶段在不同 GPU stream 上并行执行
- 使用 double buffering 重叠计算与传输

**适用场景：**
- 大模型 (SAM 2.1 Large, SAM 3)
- 多 GPU 环境

#### 3.2 Operator Fusion
**思路：**
- 将多个连续算子融合为单个 kernel
- 减少 kernel launch overhead
- 减少中间结果内存访问

**候选融合：**
1. Conv2D + BatchNorm + ReLU
2. MatMul + Add + Activation
3. Attention QKV projection + split

#### 3.3 Dynamic Batching
**思路：**
- 合并多个请求的推理
- 提高 GPU 利用率
- 减少 per-request overhead

**适用场景：**
- 视频处理 (多帧批量处理)
- 服务器部署 (多用户并发)

### 阶段 4: 架构特定优化 (持续)

#### 4.1 NVIDIA GPU 优化
- Tensor Core 利用 (WMMA/MMA 指令)
- CUDA Graph 减少 launch overhead
- cuBLASLt 高级配置

#### 4.2 Vulkan 后端优化
- Subgroup operations 优化
- Cooperative matrix (VK_KHR_cooperative_matrix)
- Async compute queues

#### 4.3 Apple Metal 优化
- Memoryless render targets
- Tile shaders
- Argument buffers 减少 descriptor updates

## 性能目标

### 短期目标 (1-2 个月)
- SAM 2.1 Tiny: CUDA < 200ms/frame (当前 626ms)
- SAM 2.1 Small: CUDA < 300ms/frame
- SAM 2.1 Base+: CUDA < 500ms/frame

### 中期目标 (3-6 个月)
- 所有 SAM 2.1 模型: 超过 PyTorch-CUDA 速度
- SAM 3 PVS: CUDA 稳定 < 500 ms（当前热态 p50 约 598 ms）

### 长期目标 (6-12 个月)
- SAM 2.1 Tiny: < 100ms/frame
- SAM 3: 达到或超过同机官方 PyTorch-CUDA
- 多 GPU 支持：线性加速比

## 测试与验证

### 性能测试
```bash
# 自动化性能回归测试
python scripts/benchmark_regression.py \
    --models tiny,small,base_plus \
    --backends cuda,vulkan \
    --threshold 1.05  # 允许 5% 波动
```

### 正确性验证
```bash
# 与 PyTorch 输出对比
python tests/compare_pytorch_outputs.py \
    --model sam2.1_hiera_tiny \
    --tolerance 1e-4
```

## 资源需求

### 硬件
- NVIDIA GPU (RTX 3060 或更好)
- 16GB+ 内存
- 支持 Vulkan 的 GPU (用于 Vulkan 后端测试)

### 软件
- CUDA Toolkit 11.8+
- Vulkan SDK
- PyTorch (用于基线对比)
- Nsight Systems (性能分析)

### 人力
- 1-2 名高性能计算工程师
- 3-6 个月开发周期

## 风险与挑战

### 技术风险
1. **数值精度** - 优化可能引入数值误差
   - 缓解：每步都进行正确性验证
   
2. **后端一致性** - 不同后端性能差异大
   - 缓解：为每个后端制定独立的优化策略

3. **维护成本** - 优化代码可能难以维护
   - 缓解：良好的代码文档和测试覆盖

### 进度风险
1. **性能目标可能过于激进**
   - 缓解：分阶段设定目标，优先实现最容易的优化

2. **依赖 ggml 上游更新**
   - 缓解：积极参与 ggml 社区，提交优化补丁

## 下一步行动

### 立即执行 (本周)
1. ✅ 完成 Vulkan conv_transpose_2d 快速路径修复
2. ⏳ 运行 PyTorch 基线测试
3. ⏳ 使用 nsight systems 分析当前性能瓶颈

### 短期 (1-2 周)
1. 实现 Flash Attention 优化原型
2. 优化内存布局减少数据传输
3. 实现 operator fusion 原型

### 中期 (1-2 个月)
1. 完成所有关键算子优化
2. 实现 pipeline parallelism
3. 达到短期性能目标

## 参考资源

### 论文
- FlashAttention: Fast and Memory-Efficient Exact Attention
- FlashAttention-2: Faster Attention with Better Parallelism
- Memory-Efficient Attention with Hierarchical Context

### 开源项目
- [xformers](https://github.com/facebookresearch/xformers)
- [cutlass](https://github.com/NVIDIA/cutlass)
- [triton](https://github.com/openai/triton)

### 工具
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- Vulkan Profiling Layers
- Metal System Trace

---

**文档版本**: 1.0  
**最后更新**: 2026-08-16  
**维护者**: sam3.cpp 团队
