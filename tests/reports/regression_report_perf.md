# 性能与 VRAM 基线报告（sam3 / sam3.1, RTX 3060）

日期：2026-09-24（§1-§11）；2026-09-25 增补 §12-§14（设备侧 prompt 装配、
cpy 向量化、环境漂移说明）。本报告记录 130 帧 VRAM 平台期测试、PCS 检测路径
剖析、mma hd=32 flash 补丁与 muxed memenc 优化的完整数据，作为后续模型更新 /
代码改动的可复现基线。质量侧量化基线见 `regression_report_q8_0.md`
（§4 逐帧表、§9 VRAM 平台期）。

## 1. 结论摘要

| 阶段 | sam3-f16 (3.0) | sam3.1-f16 | 3.1/3.0 |
|------|------|------|------|
| 原始（hd=32 走 tile 内核 + 慢 memenc） | 844 ms/帧 | 1466 ms/帧 | 1.74× |
| mma hd=32 flash 补丁后 | 811 | 1192 | 1.47× |
| muxed memenc 优化后 | 796 | 882 | 1.11× |
| **prompt 装配 + cpy 向量化后（09-25 背靠背基线）** | **851*** | **787** | **0.93×，反超** |

\* 09-25 机器整体比 09-24 慢 ~7%（对照模型 3.0 无相关改动仍 794→851，
§14）；跨日绝对值不可比，同窗口背靠背可比。

- 跟踪数值：3.0 全程逐位一致；3.1 在场帧 |Δscore| mean 0.0009 / max 0.0160、
  |Δfg| ≤ 5 px、0 次翻转（mma 与 tile 累加序差异，容差内）；memenc 优化、
  prompt 装配、cpy 向量化三改动均位等价（3.0/3.1 各 138 行逐位一致，§12/§13）
- VRAM：跟踪路径 130 帧按 PID 采样 3644 MB 恒定，零增长（§3）
- PCS 重复调用：~188 → ~133 ms（fenc flash 走 mma 后 55.5 → 8.6 ms，§5）

## 2. 环境与资产

- GPU: RTX 3060 (Ampere sm_86)，backend CUDA；CPU: 32 逻辑核（进程外 CPU
  段使用 min(16, hw_concurrency) 线程）
- 模型：`models/sam3-f16.gguf`（3.0，1464 tensors）、`models/sam3.1-f16.gguf`
- 视频：`/tmp/test_video_x5.mp4`（70 帧，`-c copy` 拼接，帧内容与原 14 帧
  逐位一致）、`/tmp/test_video_x10.mp4`（140 帧）
- runner：`/tmp/track_multi42 <model> <n-frames> <video>`（源码见 q8_0 报告 §5）
- 基线数据：`/tmp/long70_f16_clean.txt`（预改动）、`/tmp/speed30.txt` /
  `/tmp/speed31.txt`（mma 前）、`/tmp/reg70_30_mma.txt` /
  `/tmp/reg70_31_mma.txt`（当前）、`/tmp/vram130.txt` / `/tmp/reg130_f16.txt`

## 3. 130 帧 VRAM 平台期测试（跟踪路径）

首次 130 帧长跑暴露 **VRAM 线性泄漏 17.5 MB/帧**（机器总量 4733→6658 MB /
110 帧，RSS 平坦）：`sam3_encode_memory` 每帧每实例克隆 3 个 memenc 槽位
buffer push 进 `owned_buffers`，槽位被 7 槽窗口裁剪后永不释放。

修复 = 所有权下沉（`sam3_memory_slot.bufs[3]` / `sam3_ptr_slot.buf` 槽位
自持设备 buffer），8 条释放路径全部接入 release 助手。修复后（按 PID 采样）：

| 指标 | 泄漏构建 | 修复后 |
|------|------|------|
| VRAM（本进程） | 4733→6658 MB，无平台期 | 预热后 **3644 MB 恒定**贯穿 ~105 帧 |
| 增速 | 17.5 MB/帧 | **0**（min=median=max） |
| 数值 | — | 69 帧 × 2 实例 vs 预改动基线零漂移 |

**判定标准（模型更新后复跑）**：稳态 VRAM 首尾差 < 50 MB；前 69 帧
score/fg 与 §6 基线 diff 为空（或 mma 容差内）。

复现：

```bash
export PATH=$PWD/data:$PATH
/tmp/track_multi42 models/sam3.1-f16.gguf 130 /tmp/test_video_x10.mp4 \
    2>/dev/null > /tmp/reg130_f16.txt &
RUNPID=$!
# 按 PID 采样（机器总量会被外部 GPU 进程污染，必须按 PID）
while kill -0 $RUNPID 2>/dev/null; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F', *' -v p=$RUNPID '$1==p{print $2}'
    sleep 6
done
```

## 4. mma hd=32 flash 补丁（ggml patch）

**背景**：SAM3 全链路 4 处 hd=32 flash（fenc 6 节点、prop mem-attention、
prop_nonmux、ddec 经 `sam3_fattn_hd_supported` 豁免走 manual）。此前 ggml
mma 内核显式排除 DKQ=32 → 大查询落 tile 内核，实测仅 3.0 TFLOPS；mma 内核
在 hd=64 上 12.5 TFLOPS。

**改动**（全部在 `ggml-patches/0001-sam3-ggml-combined.patch`）：
1. `fattn.cu`：Turing+ 门控解除 `Q->ne[0] != 32` 排除；mma 分派 switch 加
   `case 32`
2. `fattn-mma-f16.cuh`：Ampere config 表加 (32,32,ncols∈{8,16,32,64}) 条目
   （K2/V2/combine = DKQ/2、DV/2、DV/2，镜像 64-case 模式）；DKQ=32 的
   extern 声明（ALL_NCOLS2 宏 ×4）
3. `template-instances/fattn-mma-f16-instance-ncols1_{1..64}-ncols2_{1..8}.cu`
   16 个可达文件加 `DECL_FATTN_MMA_F16_CASE(32, 32, ncols1, ncols2)`

**效果**：
- fenc flash：`flash_attn_tile` 55.5 ms（3.0 TFLOPS）→ `flash_attn_ext_f16`
  8.6 ms（**19.2 TFLOPS，6.4×**）
- 3.1 跟踪 1466 → 1192 ms/帧（-19%）；3.0 逐位一致（其 mem-attention 为
  hd=64，本就走 mma）

**验证**：70 帧 × 2 实例在场帧 |Δscore| mean 0.0009 / max 0.0160（容差
≤ 0.03）、|Δfg| ≤ 5 px、0 次翻转；缺席帧深负区差异为饱和区噪声（两侧
fg 恒 0）。

## 5. PCS 检测路径基线（sam3.1，repeat 调用 = 文本缓存命中）

子阶段（person prompt，thr=0.15，2 dets；`SAM3_PCS_PROF=1`）：

| 子阶段 | mma 前 | mma 后（当前） |
|------|------|------|
| text（tier-1 命中） | 1.9 | 3.3 |
| geom | 10.8 | 11.3 |
| fenc | **82.5** | **31.6** |
| ddec | 32.0 | 30.5 |
| seg.compute | 48.7 | 48.0 |
| seg.misc (build/alloc/upload) | 4.4 | 4.7 |
| post (CPU) | 8.1 | 5.7 |
| **合计** | **~188** | **~133** |

nsys 逐 kernel（repeat 窗口，mma 前 → 后）：

| kernel | 前 | 后 | 说明 |
|------|------|------|------|
| flash_attn_tile / flash_attn_ext_f16 | 55.5 (tile) | 8.6 (mma) | fenc 6 节点 hd=32 |
| cpy_scalar | 35.3 | 24.6 | ddec 非连续布局拷贝，未优化 |
| k_bin_bcast | 21.2 | 19.3 | 逐元素胶水 |
| im2col | 16.2 | 13.5 | seg head 卷积 |
| convert_unary | 10.9 | 10.4 | 类型转换 |
| soft_max_f32 | 3.8 | 3.5 | manual SDPA |

**判定标准**：repeat 调用合计 ≤ 150 ms（fenc ≤ 40）；检测样例锚点
`pcs_census models/sam3.1-f16.gguf data/test_video.mp4 person 0.15 repeat`：
det0 score=0.3163 fg=4915 box=[674.2,0,960,41.3]，det1 score=0.2094 fg=2953。

**未做的优化**（记录备查）：ddec 18 个 manual softmax 改 mma flash（
Nq=200≤512 走 manual 是设计内豁免，一次性检测调用 30 ms 可接受）；
cpy_scalar 向量化需 ggml patch，收益 ~15 ms。

## 6. muxed memenc 优化（3.1 专属，位等价）

**归因**（`SAM3_PROFILE_PROP=1`，memenc+store 优化前 ~345-361 ms/帧，而
census 仅 49.4 GFLOP / 4.1 GB 流量 —— GPU 只需 ~20 ms）：

| 子阶段 | 优化前 | 优化后 |
|------|------|------|
| muxed 构建（16 槽 256²→1008² 双线性 + sigmoid） | 67-74 | **10-12** |
| interp（32ch 1008²→1152² 重采样） | 210 | **18-20** |
| upload（169MB H2D） | 19-21 | 19-21 |
| compute（GPU conv tower） | 20.5 | 20.5 |
| store（门控 + 槽位克隆） | 4-27 | 4-6 |
| **memenc+store 合计** | **345-361** | **78-83** |

**根因**：每帧分配+清零 ~300MB 主机缓冲（130+169MB，页错误+memset
~65ms）+ 串行标量 expf 16M 像素/帧 + 串行双线性 42M 输出像素/帧。
关键事实：`INTERPOL = H*16 = 1152 ≠ HIGH_RES = 1008`（patch_size=14 →
H=72），interp 是**真实重采样不是恒等**（曾误判为恒等，GGML_ASSERT 抓住）。

**优化手段**（全部位等价）：
1. `sam3_bilinear_interpolate` 重构出 into 变体 `sam3_bilinear_interpolate_into`
2. sigmoid 融合进重采样循环（每像素运算次序不变）
3. muxed 构建 / interp 段多线程（每槽/每通道独立 → 线程数不影响结果）
4. tracker 持久 scratch（`mem_muxed_scratch` / `mem_interp_scratch`，reset
   时释放）：空槽通道显式清零、非 cond 槽 cond 通道显式清零（防上一帧残留）

**效果**：3.1 跟踪 1192 → 882 ms/帧（-26%）；3.0 非 mux 路径未触碰
（796 ms，噪声内）。数值差值统计与优化前逐项一致 → 位等价确认。

## 7. 跟踪速度基线（当前构建）

同 runner 同视频背靠背（70 帧 × 2 实例，f16，CUDA）：

| 日期 | 模型 | avg (f20-69) | 判定 |
|------|------|------|------|
| 09-24 | sam3-f16 (3.0) | 796 ms/帧 | — |
| 09-24 | sam3.1-f16 | 882 ms/帧 | 慢 11% |
| **09-25** | **sam3-f16 (3.0)** | **851 ms/帧** | 环境 +7%（§14），无相关改动 |
| **09-25** | **sam3.1-f16** | **787 ms/帧** | **快 8% —— 长期目标达成** |
| **09-28** | **sam3-f16 (3.0)** | **873 ms/帧**（中位，3 轮交替） | 交替复测（im2col tiled + conv F32 修复后） |
| **09-28** | **sam3.1-f16** | **804 ms/帧**（中位） | **快 8%，与 09-25 结论一致**；f1 锚点 fg 895 vs 901 px（两模型同 prompt 检测一致，各自 3 轮逐位复现） |

09-25 提速构成（3.1，对 09-24 基线）：prompt 装配 -138 ms/帧（§12）+
cpy 向量化 census 预期 -18 ms/帧（§13）- 环境漂移 +45~57 ms（§14）。
与账本自洽：882 - 138 - 18 + 45 ≈ 771~787。

## 8. 数值基线（70 帧 × 2 实例，当前构建）

runner 输出列：frame, inst, score, fg_px。下表为逐帧基线（score / fg_px），
模型更新后 diff 本表（3.0 要求逐位一致；3.1 允许在场帧 |Δscore| ≤ 0.03、
|Δfg| ≤ 6 px、0 翻转，缺席帧深负区不限但两侧 fg 须为 0）。

### sam3.1-f16

| frame | s0 | fg0 | s1 | fg1 |
|---|---|---|---|---|
| 1 | 2.636 | 901 | 1.459 | 123 |
| 2 | 2.726 | 899 | 1.516 | 116 |
| 3 | 2.722 | 894 | 1.499 | 113 |
| 4 | 2.708 | 892 | 1.482 | 112 |
| 5 | 2.696 | 884 | 1.463 | 115 |
| 6 | -101.634 | 0 | -104.759 | 0 |
| 7 | -81.063 | 0 | -91.759 | 0 |
| 8 | -81.188 | 0 | -88.697 | 0 |
| 9 | -82.375 | 0 | -86.259 | 0 |
| 10 | -104.813 | 0 | -92.000 | 0 |
| 11 | -92.938 | 0 | -89.134 | 0 |
| 12 | -95.500 | 0 | -89.697 | 0 |
| 13 | -95.688 | 0 | -91.009 | 0 |
| 14 | 2.653 | 922 | 1.287 | 119 |
| 15 | 2.736 | 898 | 1.470 | 117 |
| 16 | 2.739 | 894 | 1.481 | 114 |
| 17 | -57.728 | 0 | -51.344 | 0 |
| 18 | -73.572 | 0 | -72.313 | 0 |
| 19 | -72.743 | 0 | -68.493 | 0 |
| 20 | -79.493 | 0 | -77.368 | 0 |
| 21 | 1.176 | 0 | 1.176 | 0 |
| 22 | 1.176 | 0 | 1.176 | 0 |
| 23 | 1.176 | 0 | 1.176 | 0 |
| 24 | 1.176 | 0 | 1.176 | 0 |
| 25 | 1.176 | 0 | 1.176 | 0 |
| 26 | 1.176 | 0 | 1.176 | 0 |
| 27 | 1.176 | 0 | 1.176 | 0 |
| 28 | 1.176 | 0 | 1.176 | 0 |
| 29 | 1.176 | 0 | 1.176 | 0 |
| 30 | 1.176 | 0 | 1.176 | 0 |
| 31 | 1.176 | 0 | 1.176 | 0 |
| 32 | 1.176 | 0 | 1.176 | 0 |
| 33 | 1.176 | 0 | 1.176 | 0 |
| 34 | 1.176 | 0 | 1.176 | 0 |
| 35 | 1.176 | 0 | 1.176 | 0 |
| 36 | 1.176 | 0 | 1.176 | 0 |
| 37 | 1.176 | 0 | 1.176 | 0 |
| 38 | 1.176 | 0 | 1.176 | 0 |
| 39 | 1.176 | 0 | 1.176 | 0 |
| 40 | 1.176 | 0 | 1.176 | 0 |
| 41 | 1.176 | 0 | 1.176 | 0 |
| 42 | 1.176 | 0 | 1.176 | 0 |
| 43 | 1.176 | 0 | 1.176 | 0 |
| 44 | 1.176 | 0 | 1.176 | 0 |
| 45 | 1.176 | 0 | 1.176 | 0 |
| 46 | 1.176 | 0 | 1.176 | 0 |
| 47 | 1.176 | 0 | 1.176 | 0 |
| 48 | 1.176 | 0 | 1.176 | 0 |
| 49 | 1.176 | 0 | 1.176 | 0 |
| 50 | 1.176 | 0 | 1.176 | 0 |
| 51 | 1.176 | 0 | 1.176 | 0 |
| 52 | 1.176 | 0 | 1.176 | 0 |
| 53 | 1.176 | 0 | 1.176 | 0 |
| 54 | 1.176 | 0 | 1.176 | 0 |
| 55 | 1.176 | 0 | 1.176 | 0 |
| 56 | 1.176 | 0 | 1.176 | 0 |
| 57 | 1.176 | 0 | 1.176 | 0 |
| 58 | 1.176 | 0 | 1.176 | 0 |
| 59 | 1.176 | 0 | 1.176 | 0 |
| 60 | 1.176 | 0 | 1.176 | 0 |
| 61 | 1.176 | 0 | 1.176 | 0 |
| 62 | 1.176 | 0 | 1.176 | 0 |
| 63 | 1.176 | 0 | 1.176 | 0 |
| 64 | 1.176 | 0 | 1.176 | 0 |
| 65 | 1.176 | 0 | 1.176 | 0 |
| 66 | 1.176 | 0 | 1.176 | 0 |
| 67 | 1.176 | 0 | 1.176 | 0 |
| 68 | 1.176 | 0 | 1.176 | 0 |
| 69 | 1.176 | 0 | 1.176 | 0 |

### sam3-f16 (3.0)

| frame | s0 | fg0 | s1 | fg1 |
|---|---|---|---|---|
| 1 | 0.931 | 895 | 0.163 | 103 |
| 2 | 0.932 | 895 | 0.783 | 0 |
| 3 | 0.932 | 894 | 0.834 | 0 |
| 4 | 0.931 | 894 | 0.880 | 0 |
| 5 | 0.931 | 894 | 0.011 | 0 |
| 6 | 0.005 | 0 | 0.016 | 0 |
| 7 | 0.012 | 0 | 0.008 | 0 |
| 8 | 0.007 | 0 | 0.008 | 0 |
| 9 | 0.009 | 79 | 0.007 | 60 |
| 10 | 0.002 | 0 | 0.004 | 0 |
| 11 | 0.001 | 0 | 0.004 | 0 |
| 12 | 0.001 | 0 | 0.007 | 420 |
| 13 | 0.002 | 0 | 0.009 | 571 |
| 14 | 0.926 | 911 | 0.804 | 0 |
| 15 | 0.934 | 908 | 0.224 | 0 |
| 16 | 0.934 | 911 | 0.235 | 0 |
| 17 | 0.008 | 0 | 0.015 | 0 |
| 18 | 0.017 | 0 | 0.024 | 0 |
| 19 | 0.042 | 0 | 0.057 | 19 |
| 20 | 0.071 | 0 | 0.103 | 0 |
| 21 | 0.248 | 0 | 0.470 | 40 |
| 22 | 0.551 | 0 | 0.594 | 43 |
| 23 | 0.614 | 0 | 0.620 | 162 |
| 24 | 0.641 | 0 | 0.664 | 714 |
| 25 | 0.622 | 800 | 0.531 | 0 |
| 26 | 0.460 | 788 | 0.429 | 0 |
| 27 | 0.402 | 762 | 0.386 | 0 |
| 28 | 0.388 | 0 | 0.388 | 768 |
| 29 | 0.382 | 685 | 0.369 | 0 |
| 30 | 0.368 | 0 | 0.374 | 176 |
| 31 | 0.381 | 0 | 0.391 | 182 |
| 32 | 0.392 | 188 | 0.377 | 0 |
| 33 | 0.372 | 0 | 0.377 | 276 |
| 34 | 0.390 | 162 | 0.382 | 0 |
| 35 | 0.388 | 0 | 0.390 | 102 |
| 36 | 0.397 | 0 | 0.408 | 123 |
| 37 | 0.423 | 0 | 0.442 | 125 |
| 38 | 0.459 | 0 | 0.473 | 455 |
| 39 | 0.493 | 0 | 0.505 | 496 |
| 40 | 0.495 | 546 | 0.402 | 0 |
| 41 | 0.285 | 512 | 0.248 | 0 |
| 42 | 0.235 | 0 | 0.313 | 726 |
| 43 | 0.307 | 0 | 0.311 | 451 |
| 44 | 0.308 | 255 | 0.279 | 0 |
| 45 | 0.242 | 91 | 0.200 | 0 |
| 46 | 0.180 | 47 | 0.162 | 0 |
| 47 | 0.158 | 0 | 0.161 | 23 |
| 48 | 0.164 | 23 | 0.163 | 0 |
| 49 | 0.151 | 0 | 0.155 | 0 |
| 50 | 0.164 | 0 | 0.176 | 0 |
| 51 | 0.189 | 0 | 0.207 | 0 |
| 52 | 0.227 | 0 | 0.237 | 0 |
| 53 | 0.254 | 0 | 0.255 | 17 |
| 54 | 0.262 | 0 | 0.263 | 34 |
| 55 | 0.264 | 0 | 0.273 | 56 |
| 56 | 0.291 | 0 | 0.376 | 18 |
| 57 | 0.384 | 0 | 0.391 | 19 |
| 58 | 0.393 | 0 | 0.398 | 36 |
| 59 | 0.398 | 36 | 0.397 | 0 |
| 60 | 0.390 | 56 | 0.390 | 0 |
| 61 | 0.381 | 37 | 0.376 | 0 |
| 62 | 0.372 | 0 | 0.292 | 0 |
| 63 | 0.265 | 0 | 0.254 | 0 |
| 64 | 0.226 | 0 | 0.193 | 0 |
| 65 | 0.157 | 0 | 0.136 | 0 |
| 66 | 0.132 | 0 | 0.132 | 0 |
| 67 | 0.134 | 0 | 0.137 | 0 |
| 68 | 0.127 | 0 | 0.125 | 0 |
| 69 | 0.124 | 0 | 0.123 | 0 |

（3.0 要求与预改动基线逐位一致：mem-attention hd=64 未受 mma 补丁影响，
memenc 非 mux 路径未触碰。）

## 9. 复现命令汇总

```bash
cd /home/ludahai/develop/code/github/dl/sam3-ggml
export PATH=$PWD/data:$PATH

# 1) 跟踪速度 + 逐帧质量（70 帧 × 2 实例）
/tmp/track_multi42 models/sam3-f16.gguf   70 /tmp/test_video_x5.mp4 2>/dev/null > /tmp/reg70_30.txt
/tmp/track_multi42 models/sam3.1-f16.gguf 70 /tmp/test_video_x5.mp4 2>/dev/null > /tmp/reg70_31.txt
awk 'NR>1{s+=$5;n++} END{printf "avg=%.0fms\n", s/n}' /tmp/reg70_31.txt

# 2) VRAM 平台期（130 帧，按 PID 采样）—— 见 §3

# 3) PCS 剖析（子阶段 + 检测锚点）。
#    注：2026-09-28 起诊断开关全部 API 化（sam3_params::debug，库内 0 getenv），
#    runner 以显式 flag 传递；旧 env 写法（SAM3_PCS_PROF=1 …）已失效。
/tmp/pcs_census models/sam3.1-f16.gguf data/test_video.mp4 person 0.15 repeat --pcs-prof

# 4) memenc 剖析
/tmp/track_multi42 models/sam3.1-f16.gguf 5 /tmp/test_video_x5.mp4 --profile-prop 2>&1 | grep mux_

# 5) 图 census（GFLOP/流量/节点）
/tmp/pcs_census models/sam3.1-f16.gguf data/test_video.mp4 person 0.15 --census
/tmp/track_multi42 models/sam3.1-f16.gguf 8 /tmp/test_video_x5.mp4 --census 2>&1 | grep CENSUS

# 6) nsys 逐 kernel（repeat 窗口归因）
/usr/local/cuda/bin/nsys profile -o /tmp/p --force-overwrite true --trace=cuda \
    /tmp/pcs_census models/sam3.1-f16.gguf data/test_video.mp4 person 0.15 repeat
```

## 10. 已知注意事项

- 机器总量级 GPU/时序数据受外部进程污染：时序对比必须同窗口背靠背，
  VRAM 必须按 PID 采样
- `timeout 90` 级别的短跑可能被首帧模型加载 (~10s) 吃掉配额
- runner 需 frame-0 重试（ffmpeg 偶发空帧）；`PATH` 必须含 `$PWD/data`
- 长命令/采样循环写脚本文件后 `bash` 执行（内联循环会被终端拦截）

## 11. 稳态帧剖析与下一杠杆（2026-09-24 追加，n_sel=7 / M_total=36400）

墙钟账本（~905 ms 饱和帧；nsys GPU kernel 合计 ~574 ms/帧）：

| 阶段 | 墙钟 | GPU | CPU/传输 | 备注 |
|------|------|------|------|------|
| backbone（image encode） | ~540 | ~480 | ~60 | GEMM 180ms@17 TFLOPS + flash 55 + im2col 50 + convT 26 |
| bucket-pass | ~285 | ~92 | ~193 | compute 92（flash 45-50 + mux decoder 15-25）|
| ├ read | 8.8 | — | D2H 33MB | spatial_feats 逐槽下载 |
| ├ **prompt** | **123** | — | CPU | 见下 |
| ├ upload | 49 | — | H2D 130MB | 重组后回传 |
| memenc+store | ~80 | ~20 | ~60 | §6 已优化 |

**prop flash vs mux decoder**：mem-attention 13 flash 节点 887 GFLOP 跑
~45-50 ms（≈19 TFLOPS，mma hd=32 已达高效区）；mux decoder（80-token 联合
解码 + 18 manual softmax）仅 15-25 ms —— **两者都不是当前热点**。

**新热点 #1：prompt 装配 123 ms/帧（CPU，随 P 增长 21→123）**
`sam3_propagate` 每帧把已在设备上的槽位张量下载回 CPU 重组再上传：
- `image_feats`/`spatial_feats` 逐槽 D2H（33+33 MB）→ CPU 拼接 → H2D 130MB
- PE+tpos 逐元素加 8.4M 次/帧；但 `enc_idx = num_maskmem - spatial_tpos - 1`
  只有 ~8 个不同取值 → **可按 enc_idx 预计算 8 行 PE 设备缓存（位等价，
  同样的加法只算一次）**
- 指针 tpos 投影（P×MD×D ≈ 0.5 MFLOP）留 CPU 即可位等价
- 全部改 D2D 拼接 + 设备侧 add → **预计 -170 ms/帧**（3.1: 882 → ~710，
  将反超 3.0 的 796）

**新热点 #2：cpy_scalar 通用拷贝 23 ms/帧（跟踪）+ 24.6 ms/次（PCS ddec）**
nsys grid 反推 + GGML_CPY_DEBUG 探针确认：全部是 **F32→F32 非连续 permute
物化**（`cont(ggml_permute(x))` 的 [C,H,W]↔[H,W,C] 通道重排与 attention
视图材料化，src 非连续/dst 连续），通用逐元素内核仅 ~64 GB/s：
- `ne=1327104 [256,72,72] nb=[20736,4,288]`（neck/memenc 通道重排）
- `ne=21233664 [256,288,288]`、`ne=5308416 [256,144,144]`（同族）
- `ne=1327104 [32,5184,8] nb=[4,1024,128]`（attention 视图）
向量化方案：ggml patch 加"dst 连续 + src 跨距为连续跨距置换"的 tiled
transpose 内核（float4 宽访问，可达 ~350-400 GB/s），~100-150 行 + 分派
条件，纯数据搬移位等价。**预计 -18 ms/帧（2%）+ PCS 每次调用 -18 ms**。

**结论排序**：设备侧 prompt 装配（-170 ms，中等工作量，位等价可实现）≫
cpy_scalar 向量化（-18 ms，ggml patch）> backbone（GEMM 主导，消费卡无
已证杠杆）。

## 12. 设备侧 prompt 装配（2026-09-25，3.1 mux 路径，位等价）

**问题**（§11 热点 #1）：`sam3_propagate_bucket` 每帧把已在设备上的槽位
张量 D2H 下载回 CPU（image_feats/spatial_feats 66 MB）→ CPU 拼接 + PE+tpos
逐元素加 8.4M 次 → H2D 130 MB，prompt 装配 123 ms/帧 + read 8.8 ms。

**实现**（sam3.cpp，三件套，全部位等价）：

1. **per-enc_idx PE 设备缓存**：`enc_idx = num_maskmem - spatial_tpos[s] - 1`
   只有 num_maskmem(8) 个取值 → tracker 持久 `pe_tpos_buf` 缓存 8 行
   [MD,H,H,1]（tracker.ctx 惰性创建，尺寸/hparams 变化时重建，reset 释放）。
   host 行 `pos[d+n*MD] += tpos_all[d+e*MD]` 与原循环同一次加法 → 每帧按槽
   D2D 选行 = 位等价。
2. **槽位张量 D2D 拼接**：prompt_t / prompt_img_t / prompt_img_pos_t 三图
   输入改为图内 view（`ggml_view_4d` + `ggml_backend_view_init` 继承父
   buffer），每帧 scratch vctx 建 3×n_sel 个 view + `ggml_backend_tensor_copy`
   （同 backend = CUDA D2D cudaMemcpyAsync）；指针值 / 零尾 / proj_pe 尾仍在
   `tail_off` 处 `tensor_set`；rope_k 按 ms 上传。
3. **指针 tpos 投影**（P×MD×D ≈ 0.5 MFLOP）留 CPU。

**效果**（SAM3_PROFILE_PROP 实测）：prompt 123→4.4 ms、read 8.8→0.2 ms、
upload 49→38 ms，桶 368→230 ms（**-138 ms/帧**）。

**坑**：
- view 描述符若建在图上下文 ctx0（32768 张量容量）会跨帧累积溢出 → 必须
  每帧新建/释放 scratch vctx。
- 重构后 mux_prof 打印须用局部 `mt`（`pd.M_total` 此时只含指针 token，
  显示 112 而非 36400）。
- 图输入必须有消费者才被 gallocr 分配（fpn2 buffer=NULL 教训同源）。

**数值**：60 帧回归 118 行 vs mma 基线逐位一致（仅差基线多出的第 60 帧，
ffmpeg 少解一帧，非数值差异）。

## 13. cpy_scalar 向量化（2026-09-25，ggml patch，纯数据搬移）

**目标**（§11 热点 #2）：通用 `cpy_scalar`（~64 GB/s）逐元素处理的
F32→F32 非连续 permute 物化（channels-last 通道重排）。注：F16→F32 大转换
本就走 cont=11 连续快路径（0.133 ms），不是问题。

**实现**（`ggml/src/ggml-cuda/cpy.cu`）：统一内核 `cpy_f32_permute_tiled` —
batched 2D in-plane transpose，`dst[b][v][u] = src[b][u][v]`，smem
tile[32][33]：load 相位沿 v 合并读（consecutive threadIdx.x），store 相位
交换线程角色沿 u 合并写（读 tile 转置槽），双侧 coalesced、双侧无 bank
conflict；PDL 宏与 cpy_scalar_transpose 风格一致（sm_86 上为 no-op）。

分派（在 memcpy2d 之后、can_be_transposed 之前）：

| 族 | src 元素步长 | 典型形状 | 备注 |
|----|------|------|------|
| A | `[ne1*ne2, 1, ne1]` | [256,72,72] / [256,144,144] / [256,288,288] nb=[20736,4,288] 族（neck/memenc 通道重排） | batch=i2 |
| B | `[ne0, ne0*ne1, 1]` | 当前工作负载未命中 | 镜像旋转，保留兜底 |

守卫：`ne[3]==1`、dst 连续、ne>0、gridDim.z ≤ 65535（CUDA 限制）。

**关键教训（勿重蹈）**：ggml 连续布局是 **dim0 最快**（元素步长
`[1, ne0, ne0*ne1]`），不是 C-order 的 `[ne0*ne1, ne0, 1]`。初版按 C-order
建模 dst → 每个平面被写成转置垃圾 → neck 输出全坏 → 冒烟立即抓住
（score -64.8 / fg 0 vs 基线 2.636 / 901）。写 permute 内核前必须先对照
census 的 ne/nb 语义（nb 单位是字节，dim0 是 innermost）。

**数值**：3.0/3.1 各 138 行 vs mma 基线逐位一致（纯搬移必须零漂移）。

**时序（实测 A/B，2026-09-25 同窗口，3.1 70f×2）**：fast 759.8 vs
nofast（禁用快路径重编）761.9 ms → **隔离收益 ~2 ms，噪声内**。census 的
-18 ms/帧预估高估了可覆盖占比：nsys 归因的 cpy_scalar ~23 ms/帧 中
channels-last 族实际仅数 ms（[256,288,288] 85 MB 单次 ~2.7 ms@64GB/s），
其余为不覆盖的 [32,5184,8] attention 视图族（3-cycle，留通用路径）等。
内核保留：正确、无害、单拷贝更快，且是后续布局类优化的基础设施。

**落地后复验（2026-09-25）**：60f 回归 118 行逐位一致；130 帧（x10）前 69
帧与 70f 基线逐位一致，VRAM 按 PID 采样 median=max=3680 MB 恒定零增长；
PCS 75 次调用签名逐位稳定（1.886001/107248，跨 prompt 切换后 final 与首个
burst 相同）。census 证实 #K 分派在 3.0 跟踪图与 3.1 PCS 图（text/fenc/
ddec/seghead）均零触发 —— 对两者完全惰性。（PCS 签名与 09-24 文本缓存
会话的 1.890093/107244 之差来自 mma 的 fenc 累加序变化，早于 #K 且在容差
内，与 #K 无关。）

**patch 已回写** `ggml-patches/0001-sam3-ggml-combined.patch`
（含 cpy_f32_permute_tiled 5 处引用），`apply_ggml_patches.sh` 幂等验证
通过（applied 0, skipped 1, final tree verified）。

## 14. 环境漂移说明（跨日对比必读）

- 09-25 与 09-24 相同构建（mma 基线代码）测 3.0：794→851 ms（**+7%**）。
  3.0 的代码路径经双重证实未受 #J/#K 影响：(a) `is_multiplex()` 是模型
  属性，3.0 不走 mux bucket（#J 全部改动在 bucket 内）；(b) census 零命中 +
  nofast 对照（§13）。→ 漂移来自机器状态（温度/时钟），**跨日绝对值不可比，
  模型对比必须同窗口背靠背**。
- 09-25 新基线（同窗口背靠背，/tmp/reg70_30_cpy.txt、
  /tmp/reg70_31_cpy.txt）：3.0=851 ms、3.1=787 ms，数值均逐位一致 →
  **3.1 反超 3.0（0.93×）**，长期目标达成。
- 速度账本（3.1，09-24 基线 882）：-138（prompt 装配）-18（cpy，census
  预期）+45~57（环境）≈ 771~787，与实测 787 自洽。

## 15. 剩余耗时剖析与热点评估（2026-09-25，饱和帧 ~787 ms，n_sel=7 / M_total=36528 / P=15）

实测手段：`SAM3_PROFILE_PROP=1`（bucket 子阶段 + memenc）+ `sam3_encode_image_impl`
图计算打印，饱和帧取样（/tmp/prof15.txt）。账本：

| 阶段 | 墙钟 | 占比 | 明细 |
|------|------|------|------|
| **image encode（backbone+neck）** | **~550** | **70%** | graph ~530 + 预处理 ~20 |
| bucket-pass | ~145 | 18% | compute 89 + upload 40 + prompt 8 + read 0.2 + post 3 |
| memenc+store | ~80 | 10% | interp 20 + upload 19 + compute 20 + muxed 12 + store 5 |
| 帧解码/胶水 | ~15 | 2% | ffmpeg decode、detection 后处理 |

prompt 装配优化后 read（8.8→0.2）与 prompt（123→8）已接近零，bucket 侧
剩余大头是 compute（89）与 upload（40）。

**热点评估排序**：

1. **backbone ~550 ms（70%）— 唯一大杠杆，但消费卡无已证手段**。GEMM 主导
   （ViT MLP 4736 维，SGEMM ~34% TC 利用率，瓶颈在 kernel 形态）；已知死路：
   F16 激活（实测 -7% 更慢，见 project memory）、减层（破坏 checkpoint
   兼容）。CUDA graph capture 实验（下）收益边际。
2. **bucket compute 89 ms**：mem-attention flash 45-50（mma hd=32 已
   19 TFLOPS 高效区）+ mux decoder 15-25（裁剪数学上不可行，§11）。无杠杆。
3. **memenc interp ~20 ms**：INTERPOL(1152)≠HIGH_RES(1008) 的真实重采样，
   GPU 化可省 ~15 ms（中风险：此前误判恒等删 interp 曾崩溃，需谨慎）。
4. **bucket upload ~40 ms（D2D 槽位拷贝）**：重构 memory bank 布局（槽位
   原地写入 prompt 缓冲）可省 ~20-30 ms（大改，位等价难保证）。
5. cpy_scalar 残余：不覆盖族（[32,5184,8] attention 视图 3-cycle）+
   PCS ddec 非连续拷贝，合计 <10 ms，不值得。

**CUDA graph capture 实验（GGML_CUDA_GRAPHS=ON，零代码改动）**：编码图
缓存使张量地址跨帧稳定（sam3.cpp 编码图注释即为它设计），理论可回收
09-24 账本中 backbone ~60 ms/帧 的 kernel 间隙。实测（同窗口背靠背）：

| 配置 | 3.1 f20-69 | 数值 |
|------|------|------|
| fast（graphs OFF） | 759.8 ms | 138 行基线 |
| nofast（graphs OFF） | 761.9 ms | — |
| fast（graphs **ON**） | **754.2 ms** | 与基线逐位一致 |

收益 ~+6 ms（0.8%），远小于间隙预估 —— prop 图 M_total 每帧变化，反复
触发重捕获抵消了 launch 节省。**结论：边际收益 + 长会话风险面（捕获失效/
多 gallocr 交互），保持默认 OFF；如需压榨可自行 `-DGGML_CUDA_GRAPHS=ON`。**

**总体结论**：787 ms 已接近当前架构在消费卡（RTX 3060）上的合理下限；
剩余可做项（interp GPU 化 ~15 ms、upload 重构 ~20-30 ms）收益均为
15-30 ms 量级且需中等以上重构/承险。下一个数量级杠杆在硬件侧（数据中心
卡 FP16+FP32acc env 免费 ~7%，见 project memory）或 backbone 算法侧
（无已证方案）。

## 16. Vulkan 后端：Xid 109 挂死根因修复 + 数值对齐（2026-09-27 关闭）

**结论：Xid 109 不是驱动问题，是我们 consolidated patch 里 conv_transpose_2d
fast path 的三个 bug**（用户判断正确）。修复已入 patch
（`0001-sam3-ggml-combined.patch`，幂等验证通过）。

**构建**（RTX 3060，系统 loader 1.4.350）：`-DSAM3_VULKAN=ON
-DVulkan_GLSLC_EXECUTABLE=/usr/local/bin/glslc`。SDK（~/VulkanSDK）二进制与
glibc 2.31 不兼容，必须用系统 glslc（shaderc v2026.4）；coopmat(2)、
decode_vector、integer_dot_product 全支持。

### 16.1 根因（三处，全在 patch 新增的 `ggml_vk_conv_transpose_2d_fast`）

| # | Bug | 后果 |
|---|-----|------|
| 1 | GEMM 复用了**非 TRANSPOSE** 编译的 `pipeline_conv2d_f32/_f16_f32` map，而 convT 应使用 `pipeline_conv_transpose_2d_f32/_f16_f32`（`-DTRANSPOSE=1` 变体，内核索引公式不同：`knl_idx = K_idx*nb02 + Cin_idx_a*nb03`） | 内核矩阵按错误步长读取 |
| 2 | `p.nb03 = Cin*Cout` 应为 `1`（reorder 后的 dense [K=4*Cout, Cin] 行主序：K 步长 Cin、ci 步长 1） | **索引越界 ~Cin 倍**（3.1 首 neck convT ≈160MB 越界读，`aligned=1` 时 shader 无 clamp）→ 设备 fault → Xid 109 |
| 3 | reorder → GEMM → interleave 三个 dispatch 之间**无内存屏障**（Vulkan 允许同 command buffer 内 dispatch 并发执行） | 挂死解除后暴露：interleave（微秒级）读到 GEMM 未写出的 buf_ph → 输出全零 |

修复 = 换 transpose map + `nb03=1` + 两个 dispatch 间 `ggml_vk_sync_buffers`。
解释全部现象：三模型 neck 均为 2x2/stride2 convT 所以全挂；tiny 挂在首个
convT（节点 209）；官方 test-backend-ops 通过（测试形状不触发 2x2+stride2
连续门控）；CPY/transpose 管线无辜。

### 16.2 最小复现与逐级取证（/tmp/vk_repro_convt.c）

- CPU 参考公式验证：`out[ox,oy,co] = Σ_ci w[ox&1][oy&1][co][ci] * in[ox>>1][oy>>1][ci]`（ggml dim0 最快布局）。
- 结构化常数核（Cout=1,Cin=1,W=3,H=2）+ 临时同步回读证明 reorder ✓、GEMM ✓、
  interleave 读到旧数据 → 锁定缺失屏障；写常数 42 探针证明 interleave 写侧正常。

### 16.3 数值对齐（3.1 f16 / 2.1 tiny f16，x3 视频 42f×2 实例）

- **3.1 3 帧**：score 差 ≤0.006，fg 差 ≤1px（CUDA 901/123/899/116 vs
  VK 902/123/900/116）。
- **3.1 42 帧**：轨迹分叉点 frame 17（CUDA inst0 永久丢失，VK 重捕获成功
  fg≈26k 稳定 score 2.69）。**双侧都在跟踪的帧（1-5、14-16）fg ±3px、
  score ±0.02** —— 与历史 CUDA↔CUDA 对齐 bar 一致。分叉归因见 §16.7
  （第二轮审查：与 cm1 无关，系恒定 score 偏移被记忆选择临界决策放大；
  CUDA 当日 42f 与 09-25 基线（reg70_31_cpy）逐位一致（41 行 0 diff），
  排除"当前构建回归"解释）。
- **2.1 tiny（visual tracker 路径）3 帧**：fg ±1-4px、score ±0.001。
- 固有精度注记：本 GPU coopmat（cm1）conv/convT 管线 F16 累加，convT 输出
  abs err ≈0.07（rel ~2e-3），与通用路径一致，为上游行为。

### 16.4 内核形态审计（SAM3_CENSUS 等价检查）

- mem-attention（hd=32）走**原生 Vulkan flash-attn shader**
  （flash_attn.comp 标量 / cm1/cm2 coopmat 变体 + mask_opt + split_k，
  `pipeline_flash_attn_f32_f16[fa_pipeline_state]` 按 HSK/HSV/Br/Bc 选型），
  hd=32 受支持，不存在 CUDA 侧 hd=32 manual-SDPA 豁免问题。
- convT fast path 三个 dispatch：reorder（256 thr/wg）→ conv2d_mm TRANSPOSE
  变体 GEMM → interleave（256 thr/wg），中间带屏障。

### 16.5 速度基线（同日背靠背，3.1 f16，x3 视频）

| 后端 | backbone/帧(ms) | track/帧(ms) | propagate(=track-backbone) |
|------|----------------|--------------|---------------------------|
| CUDA | 541 | 807 | ~266 |
| Vulkan (coopmat) | 885 | 1259 | ~374 |
| Vulkan (标量, DISABLE_COOPMAT) | — | 5153 | — |

- **差距大头在 backbone（+64%，885 vs 541），propagate +40%**（第二轮回归
  修正了首轮"encode 接近"的误读——首轮对比的 900ms 是含管线编译的首帧）。
- coopmat 内核（mul_mat GEMM + flash attention）贡献 **3.9×**（1259→5153）：
  关闭后差距扩大到 6.4× CUDA。剩余 1.56× 差距 = coopmat 内核效率 vs CUDA
  mma + 非 GEMM 内存绑定 op + 上传/调度开销。
- tiny（visual tracker）VK 386ms/帧 vs CUDA 233-256ms。深度剖析未做，非当前
  优先级。

### 16.6 运行注意事项

- ffmpeg/ffprobe 必须在 PATH（`/home/ludahai/anaconda3/bin`），否则
  `sam3_decode_video_frame` 返回空帧（此前误判为推理失败）。
- benchmark 双后端 AUTO 选 CUDA（注册顺序），强制 Vulkan 需
  `params.device = SAM3_DEVICE_VULKAN`（track_vk/track_vk_vis 副本）。
- 复现命令：

```bash
PATH=/home/ludahai/anaconda3/bin:$PATH /tmp/track_vk models/sam3.1-f16.gguf 42 /tmp/test_video_x3.mp4
PATH=/home/ludahai/anaconda3/bin:$PATH /tmp/track_vk_vis models/sam2.1_hiera_tiny_f16.gguf 3 /tmp/test_video_x3.mp4
# convT fast path 数值回归（三后端对照）：
g++ -O2 -o /tmp/vk_repro_convt /tmp/vk_repro_convt.c -I ggml/include -lggml -lggml-base -lggml-vulkan -lggml-cpu
/tmp/vk_repro_convt vulkan | /tmp/vk_repro_convt cpu
# 冷启动判别（分叉归因）：track_cold <model> <video> <add_at> <n_frames>
```

### 16.7 第二轮深度审查（2026-09-27，first-principles 审计）

对 §16.3-16.5 的对齐声明做假设攻击后补齐的证据：

| 审查项 | 方法 | 结果 |
|--------|------|------|
| 算子覆盖 | `test-backend-ops test -b Vulkan0` 全量 | **16613/16613 全过**（含 FLASH_ATTN_EXT 全量化组合、MUL_MAT mmq、CONV_TRANSPOSE_2D；对照 CPU 参考） |
| frame-17 分叉归因 | `GGML_VK_DISABLE_COOPMAT=1` 重跑 42f（已验证 env 真实生效，双闸门 6082/6961） | **仍分叉 → 推翻"cm1 F16 累加"解释** |
| 分叉本质 | 冷启动判别 harness（track_cold：13→17、14→17、5→17） | 三种冷启动两后端轨迹**一致**（fg 全同或 ±2px、score ±0.012）→ frame 17 单帧无分歧；分叉需长记忆状态 |
| score 偏移 | 全程对比 | VK 恒定 +0.004~0.008（f16）；tiny 上反向（-0.001）→ 非固定符号系统偏差 = 内核归约顺序噪声 |
| 确定性 | 3f 跑两遍 diff | score/fg 逐位一致（仅 timing 变） |
| q8_0 抽样 | 3f 双后端 | ±2px / ±0.009，与 f16 同 bar（mmq 路径 ✓） |
| VRAM 长会话 | 130 帧（x10 视频）f16 × 2 实例，按 PID 采样（§3 方法，2026-09-28 补测） | **3571→3590 MB，min=median=3571 恒定平台期，零增长**（判定线 <50 MB ✓）；CUDA 同负载稳态 3644 MB，绝对值差异来自后端内存池策略，平台期判定不要求相等 |

**分叉的最终定性**：VK 全程携带 ~+0.006 的 score 偏移（任何两组不同归约顺序
的内核都会产生这类差异），在 frame 17 的记忆选择/重捕获临界决策上被放大为
完全不同轨迹。冷启动实验证明两后端单帧与短程行为等价 —— 这是决策混沌，
不是算子 bug。同类分叉在任意两个内核实现之间（包括不同 CUDA 版本）都可能出现。

**对齐最终判定**：
- 质量：f16 + q8_0（抽样）✅；q6_K/q4 族 Vulkan 未测（CUDA 侧 q4 族本身不达标；
  q6_K/q4_K 于 2026-09-28 在 CUDA 侧补测，见 q8_0 报告 §10）
- 算子：全量套件 ✅
- 确定性：✅
- 内存：130 帧长会话 ✅（2026-09-28 补测，3571 MB 恒定零增长）
- 速度：1.56×（归因到 backbone，非 bug）

## 17. im2col tiled 内核 + interp AVX2 向量化（2026-09-28）

**背景**：§15 热点评估后用户确认执行两项低风险优化。第一性原理核查发现
census 的 IM2COL 19 节点 3468 MB 流量（单次 PCS）÷ 50 ms 实测 = **69 GB/s，
与 §13 已定性的 cpy_scalar 通用内核同病**（读侧每 warp 仅 s0 宽连续段，
3×3 卷积 = 3-wide，合并率 ~37%）——这是 §15 未列出的第四热点，且收益最大。

### 17.1 im2col tiled 内核（ggml patch，`im2col_tiled_kernel`）

- phase 1：协同加载输入 tile `[KH][IC_B][in_w]`（含 halo）进 smem，全合并读，
  越界 tap 预填零
- phase 2：每线程一个输出元素，dst 连续段写（[N,OH,OW,IC·KH·KW] 布局不变），
  tap 从 smem 读
- 守卫：unit dilation（d0=d1=1）、src 全连续（nb 逐级检查）、smem ≤ 48 KB
  （IC_B/KW 自适应收缩 TW）；不满足则走原标量 kernel（batch 步长语义保留）
- 分派点：`ggml_cuda_op_im2col` 单点（f16/f32 包装不做二次分派，防止绕过
  连续性守卫）；宏 `GGML_NO_FAST_IM2COL` 禁用（A/B 与排查用）

### 17.2 interp AVX2 向量化（sam3.cpp，`sam3_bilinear_interpolate_into_avx2`）

- 8 输出像素/迭代（gather 4 taps）+ 行级 x0/wx 预计算表（与标量内循环**相同的
  double 坐标数学**）
- 位等价论证：编译基线无 FMA（SSE2 默认 ISA），AVX2 路径用显式 mul/add 按
  标量完全相同的运算树组合；tap gather 数据一致；像素间独立并行
- 运行时 `__builtin_cpu_supports("avx2")` 分发，宏 `SAM3_NO_FAST_INTERP` 禁用；
  sigmoid 循环保持标量（expf 向量化会破坏位等价，不值得）

### 17.3 验证

- 算子级：`test-backend-ops -o IM2COL -b CUDA0` **94/94 全过**（80 个 2D
  用例全部命中 tiled path，0 FAIL，对照 CPU 参考）
- 端到端：70f×2 跟踪质量列与 §14 基线（reg70_31_cpy）**138 行逐位一致**
  （两项改动合并后单次验证；纯数据搬移 + 同序浮点运算必须零漂移）
- patch 回写 `ggml-patches/0001-sam3-ggml-combined.patch`（im2col.cu 节），
  `apply_ggml_patches.sh` 幂等验证通过（applied 0, skipped 1, final tree verified）

### 17.4 时序（GPU 空闲后完整复测，2026-09-28 14:10-14:58 同窗口）

#### int32 重写前的教训（简记）

首版 tiled kernel 用 int64 索引（每元素多次 64-bit 除法），空闲 GPU 下也只有
64 GB/s 且 2x2 s2 回归 —— 外部干扰假说被否，根因是 **int64 div 在 GPU 上的
高代价**。重写为 32-bit 索引 + 循环不变量提取后达标。

#### 最终微基准（tiled vs scalar 同流交替 40 轮取中位，mismatch=0）

| 形状 | scalar | tiled (int32) | 比值 |
|------|-------:|--------------:|-----:|
| 3x3 s1 [256,288,288]（主力，~3 次/帧） | 8.60 ms (54 GB/s) | 5.14 ms (91 GB/s) | **1.67×** |
| 3x3 s1 [512,144,144] | 4.24 ms (55 GB/s) | 2.57 ms (91 GB/s) | **1.65×** |
| 3x3 s1 [256,72,72] | 0.50 ms (58 GB/s) | 0.37 ms (79 GB/s) | 1.35× |
| 2x2 s2 [1024,144,144] | 1.04 ms (123 GB/s) | 0.95 ms (133 GB/s) | 1.08×（int32 前为 0.71× 回归） |

#### 端到端 A/B（fast = tiled+avx2，nofast = 宏禁用；70f×2，f20-69 inst0 均值）

| 构建 | 1 | 2 | 3 | 中位 |
|------|-----:|-----:|-----:|-----:|
| fast  | 765 | 751 | 779 | **765 ms/帧** |
| nofast | 801 | 793 | 775 | 793 ms/帧 |

**净收益 ≈ -28 ms/帧（-3.5%）**，与微基准预估（im2col ~20 ms + interp ~0）
自洽。质量列两次独立验证均与 §14 基线 138 行逐位一致；PCS 锚点
（det0 score=0.3163 / fg=4915）逐位一致。

#### interp AVX2：实测无收益（保留代码，教训记录）

空闲 GPU 复测 interp 仍 19.5-20.1 ms（--profile-prop flag 输出）。第一性
原理复盘：interp 是**内存绑定**（32ch×1152² 输出 ~170 MB 写 + ~130 MB 读
@ 16 线程 DDR 实测 ~16 GB/s），非计算绑定 —— CPU 向量化不减少内存流量，
方向选错；真正的杠杆是 §15 项 3 的 GPU 化（显存 360 GB/s vs DDR ~40 GB/s）。

#### 130 帧 VRAM（int32 tiled kernel 后复测，CUDA）

按 PID 采样：预热 3570 → 稳态 **3660 MB 恒定**贯穿（末 3 样本 3660/3660/3660，
稳态首尾差 ~2 MB），与 §9 泄漏修复后基线（3644 MB）同水平，**零泄漏维持**；
frame 1 数值 2.636/901 与基线逐位一致。

#### 教训（勿重蹈）

1. **择优化前先判瓶颈类型**（计算/带宽/延迟）：带宽绑定的 CPU 段向量化必然
   无效。
2. **外部 GPU 进程 100% 占用时一切 GPU 时序不可信**（含 nsys kernel
   duration，含分时等待）；同流交替微基准只能证明无回归，不能定量收益。
3. **GPU kernel 避免每元素 int64 除法/取模**（~百 cycle/次）；索引用 32-bit
   + 提取循环不变量，单此一项 1.6×。
4. ggml im2col 的 `IC_IH_IW` 参数实为单通道平面 `IH*IW`、`IH_IW` 为整批
   `IC*IH*IW`（命名反直觉），复写 kernel 需对照源码语义而非变量名。
5. 库 API 结构体尾部加字段会破坏已编译二进制（旧 runner Aborted）；接口
   演进需同步重编所有消费者。
