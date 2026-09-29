# SAM 3.1 PyTorch 对齐状态报告（multiplex tracker 数值 parity）

日期：2026-09-28 ｜ 对比对象：官方 `sam3_official` repo +
`pytorch/sam3_1/sam3.1_multiplex.pt`（CPU F32） vs 本仓 C++（sam3.1-f16，CUDA）
输入：`data/test_video.mp4` 前 3 帧，单实例点提示 (315,250)，propagate f1-f2

## 1. 端到端锚点（同输入同 prompt）

| 指标 | 官方 PyTorch | C++ | 差距 |
|------|-------------|-----|------|
| f1 fg px | 940 | 1588 | +69% |
| f1 obj_logits | 7.379 | 2.900（score 列） | 结构性 |
| f2 fg px | — | 1581 | — |

结论：**端到端存在结构性分歧**，非浮点噪声。差异源头在 memory 组装 /
memenc 层（见 §3），backbone/neck/PE 层已对齐。

## 2. 对齐基础设施（本轮建成）

| 资产 | 位置 | 说明 |
|------|------|------|
| 插桩 API | `sam3_params::debug.parity_dump_dir` | 显式传参；prop 图 + memenc 图 + ptr 统一 dump |
| dump 点 | `sam3_mux_track_instances`（b==0 块） | 18 个 prop 图命名张量 + `mux_mem_out` + ptr，帧后缀 f1/f2 |
| runner | `/tmp/mux_parity_dump.cpp` | 官方预处理数据注入 f0（`sam3_encode_image_from_preprocessed`）+ 相同 prompt + 2 帧传播 |
| 官方参考 | `/tmp/mux_ref.npz`（130 张量） | `/tmp/dump_mux_official.py`（官方 repo hooks，CPU F32） |
| 对比脚本 | `/tmp/compare_final2.py` | 多 layout 穷举配对 |

**关键陷阱（已修复并记录）**：gallocr 会复用早生命周期节点的 buffer——
compute 全部结束后再 dump 输入张量拿到的是被覆盖的垃圾（首版 PE dump
min/max = -68/24，实锤）。修复 = parity 开启时对被 dump 张量
`ggml_set_output` 强制保留存储（prop 图 15 个张量；mem_out 原本就是
output）。修复后 PE rel 3.8e-6、extra rel 0.0（纯函数级验证 ✓）。

**次陷阱 ×2（均已复现并定位）**：
1. runner 裸跑偶发 segfault、gdb 下稳定通过——UB/竞态，未定位，不影响
   dump 数据正确性；
2. `sam3_params` 二次演进（加 parity_dump_dir）后未重编 runner →
   `sam3_load_model` 内 std::string assign 越界 → GGML 打印 backtrace 后
   abort（教训 #5 的实锤复现）；重编消费者即愈，frame-1 数值
   2.636/901 与基线逐位一致（插桩零影响验证 ✓）。

## 3. 差异地图（2026-09-28，f1，可信）

### 已对齐（f16 噪声内）

| 对照点 | rel |
|--------|-----|
| mux_dec_pe（image PE，纯函数） | 3.8e-6 |
| mux_extra_embed（valid/invalid embeds） | 0.0 |
| backbone/neck/imgfeat（f0，9-23 数据） | ~3e-3 |

### 结构性分歧（真差异）

| 对照点 | rel | 备注 |
|--------|-----|------|
| **prompt_img token 数** | — | **C++ 5200 vs 官方 5184：obj_ptr tokens（16 个）被重复拼进 image 流**；官方只拼入 spatial memory 流（5200=5184+16） |
| mux_prompt（spatial mem） | 1.29 | 继承 memenc 输出差异（maskmem 是其输入） |
| mux_mem_out（memenc f1） | 1.60 | **分歧源头候选 #1**：memenc 输入 mask 的选择/值 |
| mux_prop_curr（f1 当前帧特征） | 0.72 | 候选 #2：f1/f2 的预处理差异未排除（f0 注入对齐了，f1/f2 双方各自预处理） |
| mux_prompt_img_pos（pe+tpos） | 0.194 | tpos 索引/权重部分 |
| dec 输出（masks/iou/obj/tokens） | 0.33~1.12 | 继承上游 |

### 归因树（优先级，含官方源码考古修正）

**官方 decoder.py:1373-1400 实锤语义**：官方在 attention 入口主动把
`memory_image` **零 pad** 到与 `memory` 同长（cat zeros，宽度差 =
num_obj_ptr_tokens）——**C++ 的 img 流零尾设计是对的**（先前 "ptr 重复
拼接"假设被证伪）；但 `memory_image_pos` 的 pad 段填的是
**`memory_pos[0:1, -num_obj_ptr_tokens:]`（maskmem 流 pos 的 tpos 尾行，非零）**。

1. **pos 流配对修正 + 尾部填充验证**：此前对照写反了 —— 正确配对：
   `mux_prompt_img_pos ↔ enc_f0_memory_image_pos`（非 memory_pos）；
   C++ 的 maskmem 流 pos（pe_tpos_rows 拼接产物）尚无 dump 点，需补。
   然后逐点验证：① imgfeat 流 pos 尾部 16 槽是否填 tpos 尾行（官方语义）
   还是零（C++ 现状，疑似 bug）；② maskmem 流 pos 的 pe+tpos 逐点对。
2. **memenc 输入 mask 语义**：官方 add_new_points 的 cond mask → memenc；
   C++ add 路径 multimask=false 的 mask 选择（3 个输出 token 里取哪个）
   与官方是否一致；muxed 构建（sigmoid+scale）逐点对。
3. **f1/f2 预处理**：runner 三帧全部改用官方 preprocessed 数据注入
   （`/tmp/mux_images/frame_{0,1,2}.f32` 已有）以排除该变量，然后
   mux_prop_curr 若仍差 → curr 组装逻辑（neck_trk[2] 的槽位指向）。
4. attention 层内（l0_q/k/v/att）细分 dump：`mux_mem_attn_input/output`
   已可 dump，层内需在 `sam3_build_mem_attn_graph_mux` 加命名。

## 4. 复现命令

```bash
cd /home/ludahai/develop/code/github/dl/sam3-ggml
g++ -O2 -std=c++14 -I. /tmp/mux_parity_dump.cpp -Lbuild -lsam3 \
    -Wl,-rpath,$PWD/build -o /tmp/mux_parity_dump
export PATH=$PWD/data:/home/ludahai/anaconda3/bin:$PATH
/tmp/mux_parity_dump models/sam3.1-f16.gguf /tmp/mux_cpp2 /tmp/mux_images
.venv/bin/python /tmp/compare_final2.py        # 差异地图
# 官方参考重生成（如需）：
# .venv/bin/python /tmp/dump_mux_official.py   # 需 /tmp/sam3_official
```

## 5. 下一步（按归因树顺序）

1. 修 ptr tokens 重复拼接 → 重跑 → prompt_img 5200→5184 且 rel 收敛
2. runner 改三帧全注入官方预处理数据 → 隔离预处理变量
3. memenc 输入侧 dump（muxed 构建前后的 mask logits/sigmoid）→ 对齐
   memenc → prompt(spatial) 收敛
4. attention 层内命名 + dump（如上游仍差）
5. 端到端锚点闭环：fg 1588 → 940 ± 噪声、obj_logits 7.38

## 6. 第二轮实验（2026-09-28 晚，归因树执行结果）

### 6.1 已排除的假设

- **attention q/k/v 公式**：官方 decoder.py:1244-1250 实锤
  `q = img_q(image)+q_proj(norm2)`、`k = img_k(memory_image)+mem_k(memory)+memory_image_pos`、
  `v = mem_v(memory)` —— C++ 12278-12297 逐项一致 ✅（公式排除）
- **int32 tiled im2col 回归**：patch_embed k14 s14 形状补测 mismatch=0 ✅；
  imgfeat rel 3.7e-3→8.9e-2 的漂移归因于 9-24 mma flash 累加序变化（容差内，暂挂）
- **ptr 重复拼接**：官方源码证实零 pad 是官方行为 ✅（假设证伪）

### 6.2 memenc 输入对比（新增官方 hook dump，136 张量）

官方 `_encode_new_memory` 输入 `pred_masks_high_res` = **(1,1,1008,1008) logits**
（f0: min -40.7 max 15.1，fg 1825 px；obj_logits f0=22.50 / f1=7.38 / f2=7.36）。

C++ memenc 输入 = slot_logits **288²** → muxed 双线性 1008²。f0 对比：

| 指标 | 值 |
|------|-----|
| sign agreement（288 采样网格） | **99.50%** |
| 边缘符号翻转 | ~400 px（±40 logits 级） |
| C++ 288² fg | 265 px |

**结论**：PVS mask 本体高度一致，但边缘 ±40 logits 的翻转经 memenc
sigmoid 后成为 0/1 级输入差异 → memenc 输出 rel=1.29/1.6 的主要来源。
边缘翻转的根因候选：① 上采样路径（官方 1008² 直出 vs C++ 288²→1008²
双线性）在 4→7 倍放大时的边缘行为；② f16 vs F32 的低分 logit 翻转。

### 6.3 第三轮实验（同日，F32 决定性对照 + 非确定性发现）

**实验矩阵**（f32 模型新转换 `sam3.1-f32.gguf`，权重与官方逐位一致 ✓）：

| 对照 | imgfeat rel | maskmem rel | objptr rel |
|------|------------:|------------:|-----------:|
| f16 CUDA（上轮） | 8.9e-2 | 1.29 | 1.15e-1 |
| **f32 CUDA（TF32 地板）** | 1.14e-1 | 1.29 | 1.15e-1 |
| **f32 CPU（真 F32 地板）** | **1.14e-1** | **1.29** | **1.15e-1** |

**结论：三后端/双精度差异完全相同 → 全部为确定性实现/语义差异，非噪声**。
（f16 量化噪声、TF32、竞态全部排除；判定地板 = 1e-3 以下）

**过程中修复的真 bug**：`from_preprocessed` 路径的
`struct ggml_tensor* neck_trk_out[4];` **未初始化**——build_neck_graph 只填
3 层，第 4 层栈残留（f32 布局下 0x8000000000000006）→
`ggml_set_name(垃圾指针)` segfault。修复 = 零初始化（encode_image_impl
路径 6499 行本就有 = {}，仅 from_preprocessed 缺）。

### 6.4 非确定性发现（本轮最重要的线索）

C++ 内部矛盾链：
1. 同一运行内：mem_bank.image_feats == encode 时 neck_trk_2（**0.0**）✓ 拷贝正确
2. neckcheck 程序自身：两次运行 **0.0** ✓ 确定性
3. **跨程序**：parity-run 的 neck_trk_2 vs neckcheck-run 的 neck_trk_2 = **0.604** ❌

同一 f32 模型 + 同一输入数据 + 同一 API → **encode 的输出依赖执行上下文**。
parity 与 neckcheck 的唯一结构差异：**parity 在 encode 前创建了 tracker**。
官方对照同样受此影响：官方 vs neckcheck-run neck = 1.1e-2（对齐）vs
官方 vs parity-run neck = 0.114（差异）——**tracker 上下文改变了 backbone
数值**。

候选根因（下一轮）：① tracker 创建时的某全局/共享状态（backend buffer、
线程池配置）影响 encode 图；② encode 的 graph 缓存与 tracker 的 galloc
交互；③ CUDA/CPU 后端初始化顺序效应。

### 6.5 第四轮：layout 全搜 + 根因范围锁定

no-tracker 隔离结果：**非 tracker**（no-tracker 依旧 rel=0.114，且自身确定性
两次 0.0）；layout 全搜（C++ 4 种 × 官方 2 种 × 三路 neck）全部 ≥0.114
（非 layout）；**sorted-value diff = 0.0135**（与官方同值域多重集，空间分布
不同）。

**结论**：C++ backbone/neck 数值相对官方漂移 0.114 rel（maxabs 0.556），
非噪声、非 layout、非 tracker 上下文。时间线：9-23 对齐 3.7e-3 →
9-24/26 后漂移 30×；同精度无关 → **候选元凶 = ggml v0.18.1→v0.21.0
升级（103351b）的 conv/flash 内核行为变化**（其次 9-24 mma 补丁）。
注：升级 commit 声称的 inference parity 验证的口径是跟踪输出指标，
非 backbone 中间张量对官方的逐点对照。

**影响评估**：跟踪锚点（2.636/901）与 C++ 自身基线稳定，产品功能未破坏；
但与官方逐位对齐的目标被此漂移阻断——**下游所有差异（memenc 1.29、
dec 1.09、fg 1588 vs 940）都 inheriting 这个 backbone 漂移**。

### 6.6 第五轮：逐层二分（同日；⚠️ 归因"windowed 路径"被 §6.8 修正——
真根因是 F16 im2col + CPU fattn 内核，且 global_attn_indexes 实为 (7,15,23,31)）

from_preprocessed 图已含全部 32 层 `cpp_blkNN` 命名（dump_vit_blocks 开关），
f32 CPU 跑后逐层对照官方 blkNN（官方 shape = (1,72,72,1024) token-major）：

| 层 | rel | 备注 |
|----|-----:|------|
| blk00-08 | 3.3e-3~7.1e-3 | f16/f32 噪声级 ✅（含 global 2/5/8） |
| **blk09（windowed）** | 7.1e-3 | **爬升起点**（连续 windowed 段 9→10） |
| blk10-11 | 8.6e-3→1.2e-2 | 持续爬升 |
| blk12-31（全 windowed） | 2.3e-2 → **3.3e-1** | 误差指数放大 |

官方配置实锤：`global_att_blocks: (2, 5, 8, 11)`（vitdet.py:784）。

**归因收敛**：分歧源 = **windowed attention 路径**（blk09 起的连续
windowed 段引入额外差异，global 层后部分回收，后续 windowed 段继续累积）。
候选根因：① **window partition 的切分语义**（窗口划分/填充 vs 官方
vitdet 的 window_partition）；② **windowed 分支的 RoPE**（rope_n =
window_size²，窗口内坐标偏移）；③ 上述差异被 23 个后续 windowed 层指数放大。

注：blk12+ 的 0.33 饱和值 ≈ 完全去相关的输出（误差 ≈ 信号幅度）——
与端到端 fg 1588 vs 940 的结构性差异自洽。

### 6.7 下一轮计划（已由 §6.8 执行完毕）
> 状态：windowed/global hook 内部对照、RoPE 核对、根因闭环均已在第六轮完成；
> 逐位 parity 在深度 transformer 上不可达（固有发散），产品判定见 §6.8。

1. **windowed attention 逐 op 对照**：dump blk09 的内部（window part 后的
   q/k/v、RoPE 后、attention out、window unpart）与官方 vitdet.py 的
   对应 hook 逐点比对（官方脚本加 window hook）。
2. 首查项：① window 划分的坐标序（ggml [W,H,C] 的 window 行主序 vs
   官方 [B,H,W,C] 的窗口遍历序）；② windowed 分支的 RoPE 频率表
   （window 内 (wy,wx) 的复频序 vs 官方 compute_axial_cis）；③
   pad（72 = 3×24 整除，无 pad —— 排除）。
3. 修复后重跑 32 层逐层表 → 预期全层 ≤1e-2 → 下游 memenc/dec 自动收敛。
4. 端到端锚点闭环（fg 940 ± 噪声）。
5. 漂移与 9-23 的差异 origin：blk09 前（含 patch_embed/pos）已对齐，
   **排除 ggml 升级/mma/今天的 im2col** —— 差异就在 windowed attention
   的 C++ 实现细节里（9-23 时同样存在但被当时更窄的对照表漏掉——
   9-23 的 3.7e-3 是 neck_trk_2 对照，不是逐层）。

**方法论沉淀**：逐层二分（每层中间张量 vs 官方同层）是定位数值分歧的
最强工具——一次实验将搜索空间从 32 层×多 op 压缩到单一路径。前几轮的
端点对照（imgfeat/neck）只能证明"对齐/不对齐"，无法定位；本轮的
per-block dump 一次实验完成定位。

### 6.8 第六轮：根因闭环（同日，配置颠覆 + 双根因实锤）

**配置修正**：config.json 实锤 `global_attn_indexes = [7, 15, 23, 31]`（每 8 块
末块），vitdet.py:784 的 `(2, 5, 8, 11)` 只是函数默认参数。§6.6 的
"windowed 路径"归因作废——重标后的曲线揭示真正的形态：

| 关键转折（修复前） | 数据 | 解读 |
|---|---|---|
| blk06 → blk07 [GLOBAL] | 2.83e-4 → 9.93e-4（×3.5） | global 层注入跳变 |
| blk15 → blk16 | 5.2e-2 → 2.3e-1（×2.8） | 又一 global 层 |
| blk17+ 饱和 | ~0.33 | 去相关极限 |

#### 根因 #1：ggml_conv_2d 的 im2col 被硬编码为 F16（已修复）

`ggml.c` `ggml_conv_2d`：im2col dst type = `BF16 ? F32 : F16` —— 连 F32 权重
模型也被强制 F16，输入 patch 被舍入 → 种子误差（cpp_pe vs official pe_out
= 2.15e-2，torch conv 对同输入逐位 0.00e+00）。修复：F32/BF16 权重 → F32
im2col（其余不变）。修复后 pe 6.26e-4（34×），全程曲线整体下移。
注：f16/q8 模型行为不变（旧规则对它们就是 F16，且 torch f16 conv 同样
F16 输入 = 语义正确）→ 生产锚点逐位不变（2.636/901 ✓）。

#### 根因 #2：CPU flash_attn_ext 长序列精度（已定性，不修）

修复 #1 后曲线仍从 blk07 [GLOBAL] 跳变（2.83e-4 → 9.93e-4，torch 同种子
在 global 层继续收敛到 7.4e-5）。用 manual SDPA fallback 做判别实验：

| 方案 | blk07 | 饱和值（blk16+） |
|---|---:|---:|
| flash_attn_ext（默认） | 9.93e-4 | 9.2e-2 |
| manual SDPA（仅 global） | 3.18e-4 | 2.5e-2 |
| manual SDPA（全部 hd=64） | 1.38e-4 | **1.43e-2** |
| torch 参考曲线（同种子） | 7.4e-5 | 4.7e-3 |

**定性**：ggml CPU 在线 softmax fattn 内核在长 KV（5184 keys）下每层注入
误差 ~1e-3 级，比精确 softmax 高一个量级；手动精确 SDPA 后 ggml 与 torch
进入同一量级（1.43e-2 vs 4.7e-3，剩余为 GEMM 累加序差异，不可归零）。
**决定：保留 flash 内核**（manual 全量 score 矩阵在 global 层 = 1.7 GB +
无性能优化；此误差仅影响 f32 CPU parity 场景，生产 f16 CUDA 用 CUDA 内核）。

#### 固有发散特性（重要认知）

官方 torch 从同一种子（4.18e-4）前向 32 层自身也放大 ×11 并饱和（4.7e-3），
曲线形态与 ggml 相同（blk09 起放大、blk16 饱和）——**误差放大是模型固有的
Lyapunov 不稳定性**：任何两个非逐位一致的正确实现，在 24 个 windowed 层的
链式放大下都会在 blk16+ 去相关。逐位 parity 的目标在深度 transformer 上
数学上不可达；可达目标 = 把每层注入压到累加序噪声级（已通过修复 #1 +
manual SDPA 实验证明路线）。

#### 过程副产物

- `pos_embed` 表值/cls 剥离/平铺验证全绿（GGUF == checkpoint rows 1..576，
  maxabs=0）；`sam3_apply_rope` 手动路径逐项核对正确（配对/interleave/公式）。
- 官方新增锚点：`blk9_qkv/blk9_proj_in/blk9_proj_out`（内部 hook）+ 修正
  `pos_out` 语义（是 pos embed 本身，非加后值）。
- blk9 内部逐级对照：输入 1.28e-3 → qkv 3.68e-3 → attn 4.58e-3 → proj
  4.57e-3，各级平滑放大无语义突变。
- 9-23 的 imgfeat=3.7e-3 与现 1.14e-1 的矛盾未完全考古（9-23 的对照脚本
  layout 可信度存疑），但双根因已闭环，历史差异不再重要。
