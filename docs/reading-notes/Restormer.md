# Restormer: Efficient Transformer for High-Resolution Image Restoration 精读笔记

> [📄 arXiv](https://arxiv.org/abs/2111.09881) | 🎯 CVPR 2022 (Oral) | [💻 代码](https://github.com/swz30/Restormer)

---

## 一、基本信息

| 属性 | 内容 |
|------|------|
| **论文标题** | Restormer: Efficient Transformer for High-Resolution Image Restoration |
| **发表会议** | CVPR 2022 (Oral) |
| **作者** | Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang |
| **单位** | IIAI (Inception Institute of Artificial Intelligence), MBZUAI (Mohamed bin Zayed University of Artificial Intelligence), Monash University, UC Merced / Yonsei / Google Research |
| **核心创新** | MDTA (跨通道线性注意力) + GDFN (门控深度卷积前馈网络) + 渐进式训练策略 |
| **适用任务** | 去雨、去噪、去模糊 (运动模糊 + 散焦模糊)、超分辨率 |
| **参数量** | ~25.6M (Classical), ~4.2M (Lightweight) |
| **关键指标** | Rain100H PSNR 32.09dB, GoPro PSNR 33.55dB, SIDD PSNR 40.15dB (首次破40) |
| **数据集** | Rain100L/H, Test100/1200/2800, SPA-Data, GoPro, HIDE, RealBlur-J/R, SIDD, BSD68, Set12 |

### 论文意义

Restormer 是首个在图像恢复领域实现**线性复杂度全局注意力**的 Transformer 架构，也是首个在 SIDD 真实噪声去噪 benchmark 上 PSNR 突破 40dB 的方法。作为 CVPR 2022 Oral，它重新定义了图像恢复任务的 Transformer 设计范式，影响了后续大量工作 (如 DRSformer, SwinIR 改进版等)。

---

## 二、痛点分析

| 痛点编号 | 痛点描述 | 深层原因 | 现有方案的不足 | Restormer 的解决方案 |
|----------|----------|----------|----------------|----------------------|
| **P1** | **标准 Transformer 自注意力 O(N²) 不可行** | 空间自注意力在所有像素对之间计算相似度，复杂度为 O(H²W²C)，其中 N = HW。高分辨率图像 (如 256×256, N=65536) 需要计算 65536² ≈ 4.3×10⁹ 个注意力权重 | SwinIR 等窗口注意力方法将图像分割为不重叠的局部窗口，牺牲了全局上下文建模能力；窗口之间的信息交互依赖移位窗口，效率较低 | **MDTA (跨通道注意力)**：将注意力的计算维度从空间 (HW × HW) 转置到通道 (C × C)，复杂度降为 O(C²HW)。因为 C ≪ HW，计算量大幅降低 |
| **P2** | **CNN 感受野有限，长距离依赖建模不足** | 卷积操作是局部的，感受野随层数线性增长，需要极深的网络才能覆盖全局。图像恢复 (尤其是去模糊、去雨) 需要全局上下文来区分退化模式和真实纹理 | 深层 CNN 参数量大、训练困难；多尺度堆叠增加计算开销但全局建模效果仍有限 | **Transformer 全局建模**：MDTA 通过通道注意力实现全局感受野，单层即可捕获长距离依赖 |
| **P3** | **窗口注意力 (Window SA) 牺牲全局上下文** | 为降低复杂度，SwinIR 等方法使用固定大小窗口 (如 8×8)，窗口内计算自注意力。跨窗口信息交换依赖 Shifted Window，但每次仅能交换相邻窗口信息 | 对于大面积退化 (如大面积雨纹、长运动模糊)，局部窗口无法捕获足够上下文来恢复退化区域 | **无需窗口分割**：MDTA 在通道维度计算注意力，天然具备全局感受野，无需将图像切块 |
| **P4** | **静态权重限制特征自适应能力** | 标准 FFN 对所有空间位置使用相同的线性变换，无法根据输入内容自适应调整特征传播路径 | 去雨/去模糊等任务中，退化程度因区域而异 (如局部浓雨 vs 薄雾)，静态变换无法差异化处理 | **GDFN (门控 FFN)**：通过门控机制和 DWConv，让 FFN 具备空间自适应能力，抑制冗余特征 |

---

## 三、核心方法

### 3.1 整体架构

Restormer 采用 **4 级编码器-解码器架构** (4-level Encoder-Decoder)，并附带一个精细细化阶段 (Refinement Stage)。

```
输入退化图像 I (3×H×W)
    │
    ▼
┌──────────────────────────────────────────────────┐
│            浅层特征提取 (Shallow Feature)         │
│  3×3 Conv (3 → 48 channels) + LayerNorm          │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              Level 1 (最高分辨率)                  │
│  Transformer Blocks ×4  (48 channels, H×W)       │
│  ┌─────────────────────────────────────────────┐│
│  │  [LN → MDTA → 残差 → LN → GDFN → 残差] ×4   ││
│  └─────────────────────────────────────────────┘│
│                      │                            │
│              下采样 (Downsample)                  │
│              48ch → 96ch, H×W → H/2×W/2          │
└──────────────────────┼───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              Level 2                              │
│  Transformer Blocks ×6  (96 channels, H/2×W/2)  │
│                      │                            │
│              下采样 (Downsample)                  │
│              96ch → 192ch, H/2×W/2 → H/4×W/4     │
└──────────────────────┼───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              Level 3 (Bottleneck)                  │
│  Transformer Blocks ×6  (192 channels, H/4×W/4) │
│                      │                            │
│              下采样 (Downsample)                  │
│              192ch → 384ch, H/4×W/4 → H/8×W/8     │
└──────────────────────┼───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              Level 4 (最低分辨率)                  │
│  Transformer Blocks ×8  (384 channels, H/8×W/8)  │
└──────────────────────┼───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              Decoder (上采样 + Skip Connection)    │
│  Level 4 → Level 3: 384→192, 上采样+concat+×3     │
│  Level 3 → Level 2: 192→96,  上采样+concat+×6     │
│  Level 2 → Level 1: 96→48,   上采样+concat+×4     │
└──────────────────────┼───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│            Refinement Stage (细化阶段)              │
│  3×3 Conv (48→48) + LN + Transformer Block ×4    │
│  (在原始分辨率 H×W 上进行精细调整)                  │
└──────────────────────┼───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              3×3 Conv (48 → 3) + Sigmoid          │
│              残差图像 R ∈ R^(3×H×W)                │
└──────────────────────┼───────────────────────────┘
                       │
                       ▼
               Î = I + R (全局残差连接)
                       │
                       ▼
                   恢复图像 Î
```

**各级别配置汇总**:

| Level | 分辨率 | 通道数 | Transformer Block 数量 |
|-------|--------|--------|----------------------|
| Level 1 | H × W | 48 | 4 |
| Level 2 | H/2 × W/2 | 96 | 6 |
| Level 3 | H/4 × W/4 | 192 | 6 |
| Level 4 | H/8 × W/8 | 384 | 8 |
| Refinement | H × W | 48 | 4 |

### 3.2 MDTA (Multi-Dconv Head Transposed Attention) -- 核心创新

MDTA 是 Restormer 最关键的创新。其核心洞察是：**将注意力的计算维度从空间 (N×N, N=HW) 转置到通道 (C×C)**，从而将复杂度从 O(H²W²C) 降为 O(C²HW)。

#### 3.2.1 详细步骤

**Step 1: 生成 Q, K, V**

输入特征 X ∈ R^(C×H×W)，通过 1×1 卷积生成 Query, Key, Value:

\[
Q = \text{Linear}_q(X) = \text{Conv}_{1\times1}(X) \in \mathbb{R}^{C\times H\times W}
\]
\[
K = \text{Linear}_k(X) = \text{Conv}_{1\times1}(X) \in \mathbb{R}^{C\times H\times W}
\]
\[
V = \text{Linear}_v(X) = \text{Conv}_{1\times1}(X) \in \mathbb{R}^{C\times H\times W}
\]

**Step 2: 深度卷积编码局部上下文**

对 Q, K, V 分别应用 **多头深度卷积** (Multi-Dconv Head)，编码局部空间上下文:

\[
Q' = \text{DWConv}_k(Q), \quad K' = \text{DWConv}_k(K), \quad V' = \text{DWConv}_k(V)
\]

其中 DWConv_k 表示 kernel size = k 的深度可分离卷积 (论文中 k=3, head数=C/k²)。

> **为什么要加 DWConv？** 纯 1×1 卷积生成的 Q/K/V 缺乏局部空间上下文，DWConv 在计算注意力之前先编码局部邻域信息，使得注意力可以同时利用局部和全局信息。

**Step 3: 通道转置**

将 Q', K', V' 从形状 (C, H, W) 转置为 (H×W, C):

\[
\hat{Q} = Q'^T \in \mathbb{R}^{HW \times C}, \quad \hat{K} = K'^T \in \mathbb{R}^{HW \times C}, \quad \hat{V} = V'^T \in \mathbb{R}^{HW \times C}
\]

**Step 4: L2 归一化**

对 Q' 和 K' 的每个通道维度进行 L2 归一化，稳定训练:

\[
\hat{Q} = \frac{Q'}{\|Q'\|_2}, \quad \hat{K} = \frac{K'}{\|K'\|_2}
\]

**Step 5: 可学习温度参数**

引入可学习参数 α ∈ R^(1×1×C)，控制注意力分布的锐度:

\[
\alpha = \text{ReLU}(\text{Conv}_{1\times1}(\text{GAP}(X)))
\]

**Step 6: 计算跨通道注意力**

\[
\text{Attn}(\hat{Q}, \hat{K}) = \text{Softmax}\left(\frac{\hat{K}^T \cdot \hat{Q}}{\alpha}\right) \in \mathbb{R}^{C \times C}
\]

\[
\text{Output} = \hat{V} \cdot \text{Attn}(\hat{Q}, \hat{K}) \in \mathbb{R}^{HW \times C}
\]

**Step 7: 转置回原始形状**

\[
\text{Output} \in \mathbb{R}^{C \times H \times W}
\]

**关键公式**:

\[
\text{MDTA}(X) = V' \cdot \text{Softmax}\left(\frac{K'^T \cdot Q'}{\alpha}\right)
\]

其中注意力矩阵大小为 C×C (而非 N×N)，复杂度从 O(N²C) 变为 O(C²HW)。

#### 3.2.2 MDTA 的直觉理解

| 维度 | 标准空间注意力 | MDTA 通道注意力 |
|------|---------------|-----------------|
| 注意力矩阵大小 | HW × HW (如 256×256=65536²) | C × C (如 48×48=2304) |
| 含义 | "空间位置 i 应该关注哪些空间位置 j" | "通道 i 应该从哪些通道 j 获取信息" |
| 全局性 | 天然全局 | 天然全局 |
| 计算瓶颈 | N² 太大 (高分辨率不可行) | C² 很小 (通常 C < 400) |

### 3.3 GDFN (Gated-Dconv Feed-Forward Network)

GDFN 是对标准 FFN 的改进，引入了**门控机制**和**深度卷积**，使 FFN 具备空间自适应能力。

#### 3.3.1 详细步骤

**输入**: X ∈ R^(C×H×W)

**Step 1: 通道扩展 (Project In)**

\[
X_1 = \text{Conv}_{1\times1}(X) \in \mathbb{R}^{C' \times H \times W}
\]

其中 C' = C × ffn_expansion_factor = C × 2.66

**Step 2: 深度卷积编码局部空间信息**

\[
X_2 = \text{DWConv}_{3\times3}(X_1) \in \mathbb{R}^{C' \times H \times W}
\]

**Step 3: 通道分割 + 门控**

将 C' 通道平均分为两半，分别记为 x₁ 和 x₂:

\[
x_1, x_2 = \text{Chunk}(X_2, 2, \text{dim}=1)
\]

门控输出:

\[
X_{\text{gate}} = \text{GELU}(x_1) \otimes x_2 \in \mathbb{R}^{C'/2 \times H \times W}
\]

> **门控的物理意义**: GELU(x₁) 作为门控信号，逐元素乘以 x₂，可以自适应地选择性地传递有用特征、抑制冗余特征。不同空间位置的退化程度不同，门控机制允许网络根据局部内容动态调整信息流。

**Step 4: 通道压缩 (Project Out)**

\[
\text{Output} = \text{Conv}_{1\times1}(X_{\text{gate}}) \in \mathbb{R}^{C \times H \times W}
\]

#### 3.3.2 GDFN 与标准 FFN 对比

| 维度 | 标准 FFN | GDFN |
|------|---------|------|
| 通道变换 | 1×1 Conv 扩展 → GELU → 1×1 Conv 压缩 | 1×1 Conv → DWConv → Chunk → GELU门控 → 1×1 Conv |
| 空间建模 | 无 (逐位置相同) | DWConv 提供局部空间上下文 |
| 特征选择 | 全部通道都激活 | 门控机制自适应抑制冗余 |
| 参数效率 | expansion=4 (MLP) | expansion=2.66 |

### 3.4 Transformer Block

每个 Transformer Block 由 MDTA 和 GDFN 串联组成，配合残差连接和 LayerNorm:

```
输入 X
    │
    ▼
┌──────────────────┐
│   LayerNorm      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      MDTA        │  ← 跨通道注意力 (全局建模)
└────────┬─────────┘
         │
         ▼
   + 残差连接 (X + MDTA)
         │
         ▼
┌──────────────────┐
│   LayerNorm      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      GDFN        │  ← 门控前馈网络 (局部 + 自适应)
└────────┬─────────┘
         │
         ▼
   + 残差连接 (X + GDFN)
         │
         ▼
   输出 (送入下一个 Block)
```

数学表达:

\[
X' = X + \text{MDTA}(\text{LayerNorm}(X))
\]
\[
X'' = X' + \text{GDFN}(\text{LayerNorm}(X'))
\]

### 3.5 下采样 / 上采样模块

**下采样 (Downsample)**: 利用 PixelUnshuffle (空间重排) 降低分辨率、增加通道数:

\[
\text{Downsample}(X) = \text{Conv}_{3\times3}\left(\text{PixelUnshuffle}_{2\times2}(X)\right)
\]

其中 PixelUnshuffle 将每个 2×2 像素块展平为通道维度 (空间减半，通道变 4 倍)，然后 3×3 Conv 将通道压缩到目标维度。

> 输入: (C_in, H, W) → PixelUnshuffle → (4·C_in, H/2, W/2) → 3×3 Conv → (C_out, H/2, W/2)

**上采样 (Upsample)**: 利用 PixelShuffle (空间展开) 提高分辨率:

\[
\text{Upsample}(X) = \text{PixelShuffle}_{2\times2}\left(\text{Conv}_{3\times3}(X)\right)
\]

其中 3×3 Conv 先将通道扩展为 4 倍，然后 PixelShuffle 将通道维度重新排列为空间维度 (通道减 4 倍，空间加倍)。

> 输入: (C_in, H/2, W/2) → 3×3 Conv → (4·C_out, H/2, W/2) → PixelShuffle → (C_out, H, W)

> **为什么用 PixelUnshuffle/PixelShuffle 而非 Interpolation + Conv?**
> 插值 + 卷积会引入模糊效应 (插值是一种平滑操作)，而 PixelUnshuffle/PixelShuffle 是纯粹的空间重排，不引入额外参数和模糊，信息保留更完整。

### 3.6 渐进式训练策略 (Progressive Training)

Restormer 采用渐进式训练策略来处理高分辨率图像训练时的显存瓶颈，本质上是**课程学习** (Curriculum Learning) 的一种变体。

| 阶段 | Patch 尺寸 | Batch Size | 训练迭代数 | 说明 |
|------|-----------|------------|-----------|------|
| Stage 1 | 128 × 128 | 8 | 92,000 | 从低分辨率开始，快速学习基础特征 |
| Stage 2 | 160 × 160 | 5 | 64,000 | 逐步提高分辨率 |
| Stage 3 | 192 × 192 | 4 | 48,000 | 中等分辨率 |
| Stage 4 | 256 × 256 | 2 | 36,000 | 高分辨率 |
| Stage 5 | 320 × 320 | 1 | 36,000 | 更高分辨率 |
| Stage 6 | 384 × 384 | 1 | 24,000 | 接近原始分辨率 |

> **总训练迭代**: 约 300,000 次。学习率在每个阶段内使用余弦退火，阶段间自动重启 (CosineAnnealingRestartCyclicLR)。
>
> **设计动机**: 高分辨率 patch 需要大量显存，无法从开始就使用大 patch。渐进式策略让模型先在低分辨率上学习全局退化模式，再逐步在高分辨率上学习细节恢复。

---

## 三.5 数学推导过程详解 (Mathematical Walkthrough)

### MDTA 复杂度分析

**标准空间自注意力 (Standard Self-Attention)**:

\[
\text{Complexity} = O(N^2 \cdot C + N \cdot C^2) \approx O(N^2 \cdot C)
\]

其中 N = H × W。对于 256×256 图像，N = 65536，注意力矩阵大小为 65536² ≈ 4.3×10⁹，完全不可行。

**MDTA 跨通道注意力**:

\[
\text{Complexity} = O(C^2 \cdot HW + C \cdot HW \cdot k^2) \approx O(C^2 \cdot HW)
\]

其中 k 是 DWConv 的核大小。注意力矩阵大小为 C × C (如 Level 1: 48×48=2304, Level 4: 384×384=147456)。

**关键对比**:

| 分辨率 | N = HW | 标准注意力 N² | MDTA 注意力 C² (C=48) | 加速比 |
|--------|--------|-------------|----------------------|--------|
| 128×128 | 16,384 | 2.68×10⁸ | 2,304 | ~116,000× |
| 256×256 | 65,536 | 4.29×10⁹ | 2,304 | ~1,860,000× |
| 384×384 | 147,456 | 2.17×10¹⁰ | 2,304 | ~9,400,000× |

> 由于 C ≪ HW (通道数通常在 48~384 之间，而空间像素数在数万以上)，MDTA 的复杂度远低于标准空间注意力。

### 损失函数

Restormer 仅使用 **L1 Loss**:

\[
\mathcal{L} = \|R - (I_{gt} - I)\|_1
\]

其中:
- I: 输入退化图像
- I_gt: Ground Truth 清晰图像
- R: 网络预测的残差图像

**全局残差连接**:

\[
\hat{I} = I + R
\]

即恢复图像 = 输入退化图像 + 网络预测的残差。残差学习的优势在于退化通常只影响图像的一部分信息，残差的量级远小于完整像素值，更容易学习。

> **为什么只用 L1？** L1 Loss 相比 L2 Loss (MSE) 能产生更锐利的恢复结果。L2 会对大误差施加二次惩罚，倾向于输出均值化的模糊结果；L1 对所有误差均匀惩罚，鼓励保留边缘和纹理细节。

---

## 为什么这样做 (Why This Design)

| 设计选择 | 原因 | 不这样做的后果 |
|----------|------|---------------|
| **通道注意力而非空间注意力** | 通道数 C 通常远小于空间像素数 HW (48 vs 65536)，C×C 注意力矩阵远小于 HW×HW。通道维度天然编码了语义信息，通道间的关系同样重要 | 使用标准空间注意力，256×256 图像需要 65536² 内存，完全不可行；使用窗口注意力则牺牲全局上下文 |
| **DWConv 编码局部上下文** | 纯 1×1 Conv 生成的 Q/K/V 缺乏空间位置信息，DWConv 在计算注意力前先聚合局部邻域信息，使注意力计算同时利用局部细节和全局关系 | 不加 DWConv，注意力无法区分不同空间位置的退化模式，退化区域边缘处理能力下降 |
| **L2 归一化 + 可学习温度** | L2 归一化防止 Q/K 值过大导致 Softmax 饱和；可学习温度 α 让网络自适应控制注意力分布的锐度 (类似大模型中的 RoPE 缩放因子) | 不归一化可能导致训练不稳定、梯度消失/爆炸；固定温度无法适应不同层级的注意力需求 |
| **GDFN 门控机制** | 不同空间位置的退化程度不同，门控机制允许 FFN 逐位置自适应地传递有用特征、抑制冗余信息。GELU 作为门控函数提供非线性 | 标准 FFN 对所有位置使用相同变换，无法区分"需要恢复"和"已经清晰"的区域，浪费计算量 |
| **GDFN expansion=2.66** | 比 MLP 的 expansion=4 更紧凑，因为 DWConv + 门控已经提供了足够的非线性表达能力，不需要过大的通道扩展 | expansion=4 会增加参数量和计算量，但不会带来等比例的性能提升 |
| **PixelUnshuffle 下采样** | 纯空间重排操作，不引入额外可学习参数，不产生插值模糊效应，信息保留最完整 | 使用双线性插值/转置卷积下采样会引入模糊，丢失高频细节信息 |
| **渐进式训练策略** | 高分辨率 patch 在训练初期需要大量显存，且大 patch 在训练早期容易过拟合局部模式。从小 patch 开始训练是自然的课程学习 | 直接用 384×384 训练，显存不足且训练不稳定，收敛困难 |
| **Refinement 细化阶段** | 编码器-解码器经历多次下采样-上采样，可能在恢复过程中引入微小的伪影或不一致。Refinement 在原始分辨率上进一步修正 | 没有细化阶段，编码器-解码器的上采样伪影会直接影响最终输出质量 |

---

## 四、实验与效果

### 4.1 训练配置

| 配置项 | 值 |
|--------|-----|
| **优化器** | AdamW (β₁=0.9, β₂=0.999) |
| **学习率** | 初始 3×10⁻⁴ |
| **权重衰减** | 1×10⁻⁴ |
| **总训练迭代** | 300,000 |
| **学习率策略** | CosineAnnealingRestartCyclicLR (余弦退火 + 自动重启) |
| **Batch Size** | 8 GPU，总有效 batch 随阶段变化 (8→5→4→2→1→1) |
| **损失函数** | L1 Loss |
| **数据增强** | 随机水平翻转、随机旋转 (90°, 180°, 270°) |
| **评估指标** | PSNR (dB), SSIM |

### 4.2 去雨结果

| 数据集 | PSNR (dB) ↑ | SSIM ↑ | 对比 MPRNet (PSNR) |
|--------|-------------|--------|---------------------|
| **Rain100L** | **41.24** | 0.9858 | 38.67 |
| **Rain100H** | **32.09** | 0.9428 | 30.79 |
| **Test100** | **39.68** | 0.9787 | - |
| **Test1200** | **36.47** | 0.9673 | - |
| **Test2800** | **35.82** | 0.9609 | - |
| **SPA-Data** | **37.15** | 0.9756 | - |

**关键发现**：
- Rain100L 上 41.24dB，比 MPRNet 高 2.57dB，提升显著
- Rain100H 上 32.09dB，比 MPRNet 高 1.30dB
- 在所有去雨 benchmark 上均为 SOTA

### 4.3 去模糊结果

| 数据集 | PSNR (dB) ↑ | SSIM ↑ | 对比 MPRNet (PSNR) |
|--------|-------------|--------|---------------------|
| **GoPro** | **33.55** | 0.9647 | 32.66 |
| **HIDE** | **33.66** | 0.9688 | 32.98 |
| **RealBlur-J** | **32.52** | 0.9456 | 30.82 |
| **RealBlur-R** | **33.78** | 0.9611 | 31.76 |

**关键发现**：
- GoPro 上 33.55dB，首次突破 33dB
- 真实模糊数据集 RealBlur 上优势更明显 (比 MPRNet 高 1.7~2.0dB)
- 证明了 Restormer 在真实场景下的强泛化能力

### 4.4 去噪结果

| 数据集 | PSNR (dB) ↑ | SSIM ↑ | 备注 |
|--------|-------------|--------|------|
| **SIDD** | **40.15** | 0.9637 | 首次突破 40dB! |
| **BSD68 (gray, σ=25)** | **30.61** | 0.8789 | 灰度高斯去噪 |
| **Urban100 (σ=50)** | **29.66** | 0.8768 | 彩色高斯去噪 (大噪声) |

**关键发现**：
- SIDD 上 40.15dB，是**首个在该 benchmark 上突破 40dB 的方法**
- 比 MPRNet (39.52dB) 高 0.63dB
- 在灰度和彩色高斯去噪上也达到或超越 SOTA

### 4.5 消融实验

#### 4.5.1 注意力机制消融 (SIDD 去噪)

| 注意力机制 | PSNR (dB) |
|-----------|-----------|
| 无注意力 (纯 CNN) | 39.31 |
| 窗口注意力 (Window SA, 类似 SwinIR) | 39.72 |
| **MDTA (跨通道注意力)** | **40.15** |

**结论**: MDTA 比窗口注意力高 0.43dB，证明跨通道全局注意力的有效性。无注意力时性能最低，说明长距离依赖建模的重要性。

#### 4.5.2 FFN 消融

| FFN 类型 | PSNR (dB) |
|----------|-----------|
| 标准 FFN (无 DWConv, 无门控) | 39.86 |
| + DWConv (无门控) | 39.98 |
| **GDFN (DWConv + 门控)** | **40.15** |

**结论**: DWConv 贡献 +0.12dB，门控机制额外贡献 +0.17dB。两者都有用，门控机制的贡献更大。

#### 4.5.3 多尺度 vs 单尺度

| 架构 | PSNR (dB) |
|------|-----------|
| 单尺度 (仅 Level 1, 48ch) | 39.22 |
| 两级 (48 + 96) | 39.61 |
| 三级 (48 + 96 + 192) | 39.89 |
| **四级 (48 + 96 + 192 + 384)** | **40.15** |

**结论**: 更多的尺度级别带来持续提升，四级编码器-解码器效果最优。

#### 4.5.4 渐进式训练 vs 固定 patch

| 训练策略 | PSNR (dB) |
|----------|-----------|
| 固定 128×128 patch | 39.52 |
| 固定 256×256 patch | 39.87 |
| **渐进式训练 (128→384)** | **40.15** |

**结论**: 渐进式训练比固定任何单一 patch 大小都更好，课程学习策略有效。

---

## 五、对比总结

### 5.1 Restormer 与主流方法多维度对比

| 对比维度 | MPRNet | HINet | SwinIR | DRSformer | Uformer | **Restormer** |
|----------|--------|-------|--------|-----------|---------|--------------|
| **发表会议** | CVPR 2021 | CVPRW 2021 | ICCVW 2021 | ECCV 2022 | CVPR 2022 | CVPR 2022 Oral |
| **架构类型** | 多阶段 CNN | 多阶段 CNN+HIN | 窗口 Transformer | CNN+Transformer | 窗口 Transformer | 通道 Transformer |
| **注意力机制** | 无 (纯 CNN) | 无 | 窗口自注意力 | 局部-全局混合 | 窗口自注意力 | 跨通道注意力 (MDTA) |
| **注意力复杂度** | - | - | O(N²) (窗口内) | O(N²) (窗口内) | O(N²) (窗口内) | O(C²HW) 线性 |
| **全局感受野** | 多层堆叠 | 多层堆叠 | 受窗口限制 | 受窗口限制 | 受窗口限制 | 天然全局 |
| **FFN 类型** | 标准 | 标准 (无BN) | 标准 | 标准 | 标准 | 门控 (GDFN) |
| **参数量** | ~3.6M | ~3.5M / 88K | ~11.8M | ~22.1M | ~27.5M | ~25.6M / ~4.2M |
| **SIDD PSNR** | 39.52 | 39.99 | 39.71 | 40.44 | - | 40.15 |
| **GoPro PSNR** | 32.66 | 32.77 | 33.08 | 33.15 | 33.59 | 33.55 |
| **Rain100H PSNR** | 30.79 | - | - | 32.49 | - | 32.09 |
| **适用任务** | 多任务 | 多任务 | 超分/去噪/去JPEG | 去噪/去JPEG | 去噪/去模糊 | 去雨/去噪/去模糊/超分 |
| **训练策略** | 固定 256 | 固定 256 | 固定 128/64 | 固定 128 | 固定 128 | 渐进式 128→384 |

### 5.2 核心优势

1. **线性复杂度全局注意力**: MDTA 是首个在图像恢复中实现 O(C²HW) 复杂度的全局注意力，无需窗口分割即可建模长距离依赖
2. **统一架构多任务通用**: 同一架构在去雨、去噪、去模糊、超分四大任务上均达到 SOTA，无需为每个任务设计专门架构
3. **门控 FFN 增强特征选择**: GDFN 的门控机制使 FFN 具备空间自适应能力，优于标准 MLP FFN
4. **渐进式训练降低显存需求**: 课程学习式的训练策略让高分辨率训练成为可能
5. **SIDD 首次破 40dB**: 真实噪声去噪的重要里程碑

### 5.3 核心劣势

1. **通道注意力的语义局限**: 通道注意力在通道间建模关系，而通道的语义含义不如空间位置直观，可解释性较弱
2. **参数量偏大**: Classical 版本 25.6M 参数，相比 HINet (3.5M) 参数量较大
3. **训练成本高**: 300K 迭代 + 8 GPU + 渐进式 6 阶段，训练成本远高于 CNN 方法

---

## 六、不足与局限

| 序号 | 不足与局限 | 详细说明 |
|------|-----------|----------|
| 1 | **Pixel Tokenization 的显存瓶颈** | 虽然注意力复杂度为线性，但 Transformer Block 仍需对每个像素进行 Q/K/V 变换。384×384 patch 的中间特征仍占用大量显存，训练时需要 8 GPU，部署成本高 |
| 2 | **通道注意力的维度瓶颈** | MDTA 的注意力矩阵大小为 C×C。当 C 较大时 (如 384)，C²=147456 已经不小；且通道数限制了注意力建模的细粒度——通道间的关系远不如像素间丰富 |
| 3 | **单阶段推理** | 推理时仅使用编码器-解码器 + Refinement，没有利用多阶段渐进推理的机会。相比 MPRNet 的多阶段级联推理，单次推理的信息处理能力有限 |
| 4 | **训练成本过高** | 渐进式训练需要 6 个阶段共 300K 迭代，每个阶段需要手动调整 patch 大小和 batch size。总训练时间长，复现成本高 |
| 5 | **缺乏感知损失和对抗损失** | 仅使用 L1 Loss，可能导致感知质量 (如纹理细节、色彩鲜艳度) 不如使用感知损失/GAN 损失的方法。PSNR 高不一定意味着视觉质量最好 |
| 6 | **对极端退化泛化性待验证** | 论文主要在合成数据集上评估，对于极端退化 (如超大运动模糊核、极端降雨密度) 的鲁棒性缺乏充分分析 |
| 7 | **轻量版本与重量版本差距** | Lightweight (4.2M) 版本性能明显低于 Classical (25.6M) 版本，说明模型容量对性能影响显著，小模型设计空间仍有探索空间 |

---

## 七、一句话总结

**Restormer 通过"跨通道线性注意力 (MDTA) 将自注意力复杂度从 O(N²) 降至 O(C²HW) + 门控深度卷积前馈网络 (GDFN) 实现空间自适应特征选择 + 渐进式课程学习训练策略"三位一体设计，首次在 SIDD 真实噪声去噪上突破 40dB，并在去雨、去噪、去模糊、超分四大任务上统一达到 SOTA，是图像恢复领域 Transformer 架构的里程碑工作。**

---

## 八、生活化例子：小明的"旧影修复工作室"

> **场景四：50寸大家族全家福**

一个大家族找到小明，要修复一张**50寸的高清全家福**——照片里有三十多口人，从百岁太奶奶到襁褓中的小婴儿，每个人都要看清楚。但照片又模糊又有噪点，文件还特别大。

"这么大的照片，如果每个像素都仔细算，电脑要算到明年了！"小明挠头。这时他想起了 Restormer 的"聪明注意力"：

"就像老师批改作文，不会一个字一个字地抠，而是先看整体结构，再抓重点段落。**把注意力放在'通道'上而不是'每个像素'上**——看整体色调对不对、整体轮廓清不清晰，而不是死磕每一个小点。"

小明还用了"门控"的思想——就像审稿时用红笔圈出需要修改的地方，只在这些地方下功夫，其他地方快速通过。再加上"渐进式"的修复策略：先修小图练手，再慢慢放大修大图，像学生上课一样由浅入深。

三天后，太奶奶看着照片里自己清晰的皱纹和曾孙甜甜的笑容，笑得合不拢嘴。

小明感叹：**聪明地工作，比努力地工作更重要。**

---

> 工作室的生意蒸蒸日上，各种稀奇古怪的"问题照片"接踵而至……

## 附录

### 附录 A: MDTA 的 PyTorch 风格伪代码

```python
class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention
    跨通道线性注意力 - Restormer 的核心创新
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # QKV 投影 (1×1 Conv)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        # QKV 深度卷积 (多头深度可分离卷积, 编码局部上下文)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3,
            kernel_size=3, padding=1, groups=channels * 3
        )
        # 输出投影
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)  # 各自为 (B, C, H, W)

        # 多头分割
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)

        # L2 归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 跨通道注意力: (C, HW) × (HW, C) → (C, C)
        attn = (k @ q.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)  # Softmax over channels

        # 输出: (C, HW) × (C, C) → (C, HW)
        out = (v @ attn.transpose(-2, -1))

        # 重组回 (B, C, H, W)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Restormer Transformer Block: LN → MDTA → 残差 → LN → GDFN → 残差"""
    def __init__(self, channels, num_heads=8, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, ffn_expansion)

    def forward(self, x):
        # LayerNorm 需要在 (B, C, H, W) → (B, H, W, C) → norm → back
        x = x + self.attn(self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2))
        x = x + self.ffn(self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2))
        return x
```

### 附录 B: GDFN 的 PyTorch 风格伪代码

```python
class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network
    门控深度卷积前馈网络 - 带空间自适应能力的 FFN
    """
    def __init__(self, channels, ffn_expansion=2.66):
        super().__init__()
        # 计算扩展后的通道数 (确保可被 2 整除, 用于 chunk)
        hidden_dim = int(channels * ffn_expansion)
        if hidden_dim % 2 != 0:
            hidden_dim += 1

        self.project_in = nn.Conv2d(channels, hidden_dim, kernel_size=1)

        # 深度卷积编码局部空间信息
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim
        )

        # 门控激活函数
        self.act = nn.GELU()

        # 通道压缩 (输出)
        self.project_out = nn.Conv2d(hidden_dim // 2, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)       # (B, C) → (B, C'×2.66)
        x = self.dwconv(x)           # 局部空间上下文编码

        x1, x2 = x.chunk(2, dim=1)  # 通道均分两半

        # 门控: GELU(x1) * x2
        x = self.act(x1) * x2       # 逐元素乘法 (空间自适应)

        x = self.project_out(x)      # (B, C'/2) → (B, C)
        return x
```
