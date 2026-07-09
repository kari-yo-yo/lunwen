# Uformer: A General U-Shaped Transformer for Image Restoration 精读笔记

> [📄 arXiv](https://arxiv.org/abs/2106.03106) | 🎯 CVPR 2022 | [💻 代码](https://github.com/ZhendongWang6/Uformer)

---

## 一、基本信息

| 属性 | 内容 |
|------|------|
| **论文标题** | Uformer: A General U-Shaped Transformer for Image Restoration |
| **发表会议** | CVPR 2022 |
| **作者** | Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, Houqiang Li |
| **单位** | 中国科学技术大学 (USTC), 澳门大学, 华为诺亚方舟实验室 |
| **论文链接** | https://arxiv.org/abs/2106.03106 |
| **代码开源** | https://github.com/ZhendongWang6/Uformer |
| **核心创新** | LeWin Transformer Block (局部增强窗口注意力) + 多尺度恢复调制器 (空间偏差) |
| **适用任务** | 去噪、去模糊、去雨、散焦去模糊 |
| **关键指标** | SIDD PSNR 40.03dB, Rain100L PSNR 40.58dB |
| **数据集** | SIDD, GoPro, Rain100L/H, Rain800, BSD68, Set12 |

### 作者贡献

- **Zhendong Wang**: 第一作者，主导 Uformer 架构设计与实验
- **Xiaodong Cun**: 澳门大学联合研究
- **Jianmin Bao, Houqiang Li**: 华为诺亚方舟实验室合作

---

## 二、痛点分析

| 痛点编号 | 痛点描述 | 深层原因 | 现有方案的不足 |
|----------|----------|----------|----------------|
| **P1** | **全局自注意力计算开销过大** | 标准 Self-Attention 复杂度为 \(O(N^2)\)，其中 \(N = H \times W\)。对于图像恢复任务常用的 256x256 输入，\(N = 65536\)，注意力矩阵需要 65536x65536 的存储，显存和计算量均不可接受 | 无法直接将标准 Transformer 应用于高分辨率图像恢复任务 |
| **P2** | **CNN 感受野有限，难以建模长距离依赖** | 卷积操作的感受野受限于卷积核大小和网络深度，需要大量堆叠才能覆盖全局 | 多层堆叠效率低，且容易丢失中间尺度的细节信息 |
| **P3** | **窗口注意力缺乏跨窗口的局部上下文** | 现有窗口注意力 (如 Swin Transformer) 仅在窗口内部计算注意力，窗口之间没有信息交互，且 FFN 仅用 MLP，不包含局部卷积操作 | 窗口边界处容易产生块效应，且 MLP 缺乏空间归纳偏置，对局部纹理的建模能力不足 |
| **P4** | **Transformer FFN 缺乏局部空间信息提取能力** | 标准 Transformer 的 FFN 仅包含两个全连接层 (Linear → GELU → Linear)，对每个 token 独立处理，不考虑空间邻域关系 | 无法有效提取图像中的局部纹理、边缘等空间结构信息，而这些信息对图像恢复至关重要 |

### 2.1 图像恢复中的注意力机制困境

在 Uformer 之前，图像恢复领域面临一个根本性矛盾：

- **全局注意力**：能建模长距离依赖，但计算量 \(O(H^2 W^2 C)\) 在高分辨率下不可接受
- **CNN**：计算高效，但感受野有限，需要极深的网络才能覆盖全局
- **窗口注意力** (Swin Transformer)：降低了计算量到 \(O(M^2 H W C)\)，但窗口间缺乏交互，且 FFN 缺乏局部归纳偏置

Uformer 的核心贡献就是同时解决这两个问题。

---

## 三、核心方法

### 3.1 整体架构

Uformer 采用 **U-shaped 编码器-解码器架构**，基于 Transformer 构建，共 4 个分辨率级别 (4 levels)。

```
输入图像 (3×H×W)
    │
    ▼
┌──────────────────────────────────────────────────┐
│              Patch Embedding (3×3 Conv, stride=2) │
│              浅层特征提取 + 下采样                  │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│              Encoder Level 1                     │
│              dim=32, depth=1                      │
│              LeWin Block ×1                        │
└────────────────────┬─────────────────────────────┘
                     │ Downsample (3×3 Conv, stride=2)
┌────────────────────▼─────────────────────────────┐
│              Encoder Level 2                       │
│              dim=64, depth=2                       │
│              LeWin Block ×2                        │
└────────────────────┬─────────────────────────────┘
                     │ Downsample (3×3 Conv, stride=2)
┌────────────────────▼─────────────────────────────┐
│              Encoder Level 3                       │
│              dim=128, depth=8                       │
│              LeWin Block ×8                        │
└────────────────────┬─────────────────────────────┘
                     │ Downsample (3×3 Conv, stride=2)
┌────────────────────▼─────────────────────────────┐
│              Encoder Level 4                       │
│              dim=256, depth=8                       │
│              LeWin Block ×8                        │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│              Bottleneck                            │
│              dim=512, depth=2                       │
│              LeWin Block ×2                        │
└────────────────────┬─────────────────────────────┘
                     │ Upsample (PixelShuffle)
┌────────────────────▼─────────────────────────────┐
│              Decoder Level 4                       │
│              dim=256, depth=8 (+ Skip + Modulator) │
│              LeWin Block ×8                        │
└────────────────────┬─────────────────────────────┘
                     │ Upsample
┌────────────────────▼─────────────────────────────┐
│              Decoder Level 3                       │
│              dim=128, depth=8 (+ Skip + Modulator) │
└────────────────────┬─────────────────────────────┘
                     │ Upsample
┌────────────────────▼─────────────────────────────┐
│              Decoder Level 2                       │
│              dim=64, depth=2 (+ Skip + Modulator)  │
└────────────────────┬─────────────────────────────┘
                     │ Upsample
┌────────────────────▼─────────────────────────────┐
│              Decoder Level 1                       │
│              dim=32, depth=1 (+ Skip + Modulator)  │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
              3×3 Conv → 残差图像
                     +
              输入图像
                     │
                     ▼
              恢复图像
```

**通道变化流程** (以 Uformer_B 为例，embed_dim=32):

$$32 \xrightarrow{enc} 64 \xrightarrow{enc} 128 \xrightarrow{enc} 256 \xrightarrow{enc} 512 \xrightarrow{dec} 256 \xrightarrow{dec} 128 \xrightarrow{dec} 64 \xrightarrow{dec} 32$$

**各层深度 (LeWin Block 数量)**:

$$\text{depths} = [1, 2, 8, 8, 2, 8, 8, 2, 1]$$

分别对应：Level1(Enc) → Level2(Enc) → Level3(Enc) → Level4(Enc) → Bottleneck → Level4(Dec) → Level3(Dec) → Level2(Dec) → Level1(Dec)

### 3.2 LeWin Transformer Block -- 核心创新 1

**LeWin = Locally-enhanced Window Transformer**

LeWin Block 由两部分组成：**W-MSA (Window Multi-head Self-Attention)** 和 **LeFF (Locally-enhanced Feed-Forward Network)**。

```
输入特征 X ∈ R^(C×H×W)
    │
    ▼
┌──────────────────┐
│    LayerNorm      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│    Window Partition (窗口划分)              │
│    将 H×W 划分为 (H/M)×(W/M) 个 M×M 窗口   │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│    W-MSA (窗口多头自注意力)                │
│    在每个 M×M 窗口内独立计算注意力          │
│    引入相对位置偏差 B̂                      │
│    QK^T / √d + B̂ → Softmax → V           │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│    Window Merge (窗口合并)                 │
│    将各窗口特征拼回 H×W 空间布局            │
└────────┬─────────────────────────────────┘
         │
         ▼
      X_attn
         │
         ▼
┌──────────────────┐
│      + X (残差)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    LayerNorm      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│    LeFF (局部增强前馈网络)                 │
│    Linear → GELU → reshape →              │
│    DWConv 3×3 → GELU → flatten → Linear   │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│      + X (残差)   │
└────────┬─────────┘
         │
         ▼
      输出特征
```

#### 3.2.1 窗口多头自注意力 (W-MSA)

将特征图划分为不重叠的局部窗口 (window size = M×M)，在每个窗口内独立计算多头自注意力。

**相对位置偏差 (Relative Position Bias)**:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \hat{B}\right) V$$

其中 \(\hat{B} \in \mathbb{R}^{M^2 \times M^2}\) 是可学习的相对位置偏差张量，编码窗口内像素间的相对位置关系。

**关键优势**: 与 Swin Transformer 使用相同的相对位置偏差方案，但 Uformer 将其与局部增强 FFN 结合，形成更强的特征提取能力。

#### 3.2.2 LeFF (Locally-enhanced Feed-Forward Network) -- 关键创新

标准 Transformer 的 FFN 仅包含两个线性层，对每个 token 独立处理，**缺乏空间建模能力**。Uformer 提出的 LeFF 通过在 FFN 中引入 Depthwise Convolution 来弥补这一缺陷。

**LeFF 结构**:

$$\text{LeFF}(X) = \text{Linear}\left(\text{GELU}\left(\text{DWConv}_{3\times3}\left(\text{GELU}\left(\text{Linear}(X)\right)\right)\right)\right)$$

具体步骤：
1. **Linear**: \(C \rightarrow 4C\) (扩展通道，与标准 FFN 一致)
2. **GELU**: 非线性激活
3. **Reshape**: 将 \((4C, H, W)\) 变形为 \((C, 4, H, W)\) 或等价的空间分组形式
4. **DWConv 3x3**: Depthwise 卷积，在每个通道上独立做 3x3 卷积，提取局部空间信息
5. **GELU**: 再次激活
6. **Flatten**: 恢复形状为 \((4C, H, W)\)
7. **Linear**: \(4C \rightarrow C\) (压缩通道)

**设计动机**: DWConv 引入了局部空间归纳偏置，使得 FFN 不仅能建模通道间的非线性关系，还能捕获局部纹理和边缘信息。这对图像恢复任务至关重要。

### 3.3 多尺度恢复调制器 -- 核心创新 2

#### 3.3.1 设计动机

在 U-shaped 编解码器结构中，编码器通过逐级下采样提取多尺度特征，解码器通过逐级上采样恢复空间分辨率。然而，下采样过程中不可避免地丢失细节信息，简单的上采样无法完全恢复这些丢失的细节。

#### 3.3.2 空间偏差调制 (Spatial Bias Modulation)

Uformer 提出在解码器的每一层引入**可学习的空间偏差张量**，对窗口注意力的输出进行调制：

$$\text{MW-MSA}(X) = \text{W-MSA}(X) + B_{\text{bias}}$$

其中 \(B_{\text{bias}} \in \mathbb{R}^{M^2 \times C}\) 是一个可学习的窗口级偏差张量。它为每个窗口位置学习一个独立的偏差向量，在注意力计算后直接加到输出特征上。

**特点**:
- **参数量极小**: 每层仅需 \(M^2 \times C\) 个额外参数 (如 M=8, C=256 时仅 16K 参数)
- **逐窗口调制**: 不同空间位置的窗口获得不同的偏差补偿，补偿下采样过程中丢失的空间细节
- **无需额外计算**: 仅仅是加法操作，推理时零额外计算开销

### 3.4 跳跃连接方案

Uformer 系统地比较了四种跳跃连接方案：

| 方案 | 描述 | 操作 |
|------|------|------|
| **Concatenation (默认)** | UNet 风格拼接 | 编码器特征与解码器特征在通道维度拼接，然后通过 Linear 层降维 |
| **Cross-Skip** | 交叉跳跃连接 | 先用 LayerNorm + Linear 处理编码器特征，再与解码器特征相加 |
| **ConcatCross-Skip** | 混合方案 | 编码器特征先经过 Linear 变换，再与解码器特征拼接 |
| **No Skip** | 无跳跃连接 | 编码器与解码器之间没有直接的信息传递 |

**实验结论**: Concatenation 方案在去雨任务上效果最好，Cross-Skip 在去噪任务上表现更优。Uformer 默认采用 Concatenation。

### 3.5 模型变体

Uformer 提供三种不同规模的变体：

| 变体 | embed_dim | heads | depths | 参数量 |
|------|-----------|-------|--------|--------|
| **Uformer_T (Tiny)** | 16 | [1, 2, 4, 8, 8, 4, 2, 1] | [1, 2, 4, 4, 2, 4, 4, 2, 1] | ~4.2M |
| **Uformer_S (Small)** | 32 | [1, 2, 4, 8, 16, 8, 4, 2] | [1, 2, 2, 2, 2, 2, 2, 2, 1] | ~7.6M |
| **Uformer_B (Base)** | 32 | [1, 2, 4, 8, 16, 8, 4, 2] | [1, 2, 8, 8, 2, 8, 8, 2, 1] | ~24.5M |

> Uformer_B 是论文中的主要实验模型，通过增加深度 (更多 LeWin Block) 来提升性能。

---

## 三.5 数学推导过程详解 (Mathematical Walkthrough)

> 以下用一个 **4x4 像素退化图像块** 完整走一遍 Uformer 的 LeWin Transformer Block 和多尺度调制器的具体数值计算。

### 设定输入

假设输入为一个 4x4 单通道特征块 (经过 Patch Embedding 后，C=4 通道):

$$
F_{in} = \begin{bmatrix}
0.5 & 0.8 & 0.3 & 0.6 \\
0.7 & 0.9 & 0.4 & 0.5 \\
0.2 & 0.6 & 0.8 & 0.3 \\
0.4 & 0.7 & 0.5 & 0.9
\end{bmatrix}_{4 \times 4}
$$

> 以通道 0 为例，中心区域 (0.9) 和右下区域 (0.9) 值较高，为需要恢复的细节区域。

---

### Step 1: 窗口划分 (Window Partition)

**中文标题**: 特征图窗口划分
**English Title**: Window Partitioning

设定窗口大小 M=2，将 4x4 特征图划分为 4 个 2x2 窗口:

$$
W_1 = \begin{bmatrix} 0.5 & 0.8 \\ 0.7 & 0.9 \end{bmatrix}, \quad
W_2 = \begin{bmatrix} 0.3 & 0.6 \\ 0.4 & 0.5 \end{bmatrix}
$$

$$
W_3 = \begin{bmatrix} 0.2 & 0.6 \\ 0.4 & 0.7 \end{bmatrix}, \quad
W_4 = \begin{bmatrix} 0.8 & 0.3 \\ 0.5 & 0.9 \end{bmatrix}
$$

**维度变化**: (C, 4, 4) → (4, C, 2, 2) (4 个窗口，每个 2x2)

---

### Step 2: 窗口多头自注意力 (W-MSA)

**中文标题**: 窗口内自注意力计算
**English Title**: Window Multi-head Self-Attention Computation

以窗口 \(W_1\) 为例 (单头，d=1 简化):

将 2x2 窗口展平为 4 个 token:

$$
W_1 = [x_1, x_2, x_3, x_4] = [0.5, 0.8, 0.7, 0.9]
$$

通过线性变换生成 Q, K, V (简化权重):

$$
Q = [0.3, 0.5, 0.4, 0.6], \quad K = [0.2, 0.6, 0.3, 0.7], \quad V = [0.4, 0.7, 0.5, 0.8]
$$

**注意力分数计算**:

$$
\text{scores} = QK^T = \begin{bmatrix}
0.06 & 0.18 & 0.09 & 0.21 \\
0.10 & 0.30 & 0.15 & 0.35 \\
0.08 & 0.24 & 0.12 & 0.28 \\
0.12 & 0.36 & 0.18 & 0.42
\end{bmatrix}
$$

**加入相对位置偏差 \(\hat{B}\)** (假设学习到的偏差):

$$
\hat{B} = \begin{bmatrix}
0.05 & 0.02 & 0.01 & 0.03 \\
0.02 & 0.05 & 0.03 & 0.01 \\
0.01 & 0.03 & 0.05 & 0.02 \\
0.03 & 0.01 & 0.02 & 0.05
\end{bmatrix}
$$

$$
\text{scores} + \hat{B} = \begin{bmatrix}
0.11 & 0.20 & 0.10 & 0.24 \\
0.12 & 0.35 & 0.18 & 0.36 \\
0.09 & 0.27 & 0.17 & 0.30 \\
0.15 & 0.37 & 0.20 & 0.47
\end{bmatrix}
$$

**Softmax**:

$$
\text{Attn} = \text{Softmax}(\text{scores} + \hat{B}) = \begin{bmatrix}
0.168 & 0.205 & 0.163 & 0.226 \\
0.148 & 0.235 & 0.168 & 0.242 \\
0.142 & 0.228 & 0.181 & 0.233 \\
0.148 & 0.228 & 0.170 & 0.249
\end{bmatrix}
$$

**输出**:

$$
\text{out}_1 = \text{Attn} \cdot V = 0.168(0.4) + 0.205(0.7) + 0.163(0.5) + 0.226(0.8) = 0.608
$$

> 注意力加权后，输出融合了窗口内所有位置的值。中心像素 (0.9) 对应的注意力权重最高 (0.249)，说明窗口注意力能自适应关注重要区域。

---

### Step 3: LeFF (局部增强前馈网络)

**中文标题**: 局部增强前馈网络计算
**English Title**: Locally-enhanced FFN Computation

以注意力输出 (4 通道, 2x2 空间) 为例:

**3a. Linear 扩展** (C=4 → 4C=16):

$$
F_{leff} = \text{Linear}(F_{attn}) \in \mathbb{R}^{16 \times 2 \times 2}
$$

**3b. GELU 激活**:

$$
F_{act1} = \text{GELU}(F_{leff})
$$

**3c. Reshape** (为 DWConv 做准备):

$$
F_{reshaped} \in \mathbb{R}^{4 \times 4 \times 2 \times 2} \quad (\text{4 组, 每组 4 通道})
$$

**3d. DWConv 3x3** (每组 4 通道内独立卷积):

以一组为例，3x3 DWConv 在 2x2 特征图上 (padding=1):

$$
K_{dw} = \begin{bmatrix} 0.1 & 0.2 & 0.1 \\ 0.2 & 0.4 & 0.2 \\ 0.1 & 0.2 & 0.1 \end{bmatrix}
$$

中心位置输出 (填充后 4x4 中):

$$
y_{dw} = 0.1(x_{pad}) + 0.2(x_{left}) + 0.1(x_{pad}) + 0.2(x_{top}) + 0.4(x_{center}) + 0.2(x_{right}) + 0.1(x_{pad}) + 0.2(x_{bottom}) + 0.1(x_{pad})
$$

> DWConv 让 FFN 具备了局部空间感受野，能够捕获 3x3 邻域内的纹理和边缘信息。

**3e. GELU + Flatten + Linear 压缩** (16 → 4):

$$
F_{leff\_out} = \text{Linear}(\text{Flatten}(\text{GELU}(F_{dwconv}))) \in \mathbb{R}^{4 \times 2 \times 2}
$$

**维度变化**: (4, 2, 2) → (16, 2, 2) → reshape → DWConv → flatten → (16, 2, 2) → (4, 2, 2)

---

### Step 4: 多尺度恢复调制器

**中文标题**: 空间偏差调制计算
**English Title**: Multi-scale Restoration Modulator Computation

调制器在解码器中对注意力输出施加空间偏差:

$$
F_{modulated} = F_{attn} + B_{bias}
$$

其中 \(B_{bias} \in \mathbb{R}^{4 \times 4}\) (4 个窗口位置，每个窗口一个 C 维偏差):

假设学习到的偏差 (以窗口为单位):

$$
B_{bias} = [b_{w1}, b_{w2}, b_{w3}, b_{w4}]
$$

$$
b_{w1} = [0.02, -0.01, 0.03, 0.01], \quad b_{w2} = [-0.01, 0.02, -0.01, 0.01]
$$

$$
b_{w3} = [0.01, -0.02, 0.01, -0.01], \quad b_{w4} = [-0.01, 0.03, -0.02, 0.02]
$$

> 偏差值很小 (量级约 0.01-0.03)，但能有效补偿编码器下采样过程中的信息损失。不同窗口位置的偏差不同，实现了空间自适应的调制。

**参数开销**: 仅 \(M^2 \times C = 4 \times 4 = 16\) 个额外参数 (以 M=2, C=4 为例)

---

### Step 5: 残差学习与最终恢复

**中文标题**: 残差连接与图像恢复
**English Title**: Residual Learning and Image Restoration

经过完整的 Uformer 编解码器后:

$$
R = \text{Conv}_{3\times3}(F_{decoded}) \in \mathbb{R}^{H \times W \times 3}
$$

$$
\hat{I} = I_{degraded} + R
$$

> 残差学习策略使得网络只需学习退化图像与清晰图像之间的差异，而非完整的像素映射，大幅降低学习难度。

---

### 窗口注意力复杂度分析

| 方法 | 复杂度 | 当 H=W=256, M=8 |
|------|--------|-------------------|
| **全局自注意力** | \(O(H^2 W^2 C)\) | \(O(256^4 \times C) \approx O(1.1 \times 10^{10} \times C)\) |
| **窗口自注意力** | \(O(M^2 H W C)\) | \(O(64 \times 65536 \times C) \approx O(4.2 \times 10^6 \times C)\) |

窗口注意力将复杂度降低了约 **2600 倍** (当 M=8 时)。

---

### Charbonnier Loss

Uformer 使用 Charbonnier Loss 作为损失函数：

$$
\mathcal{L} = \sum_{i=1}^{N} \sqrt{\|I_i' - \hat{I}_i\|^2 + \epsilon^2}
$$

其中 \(\epsilon = 10^{-3}\)。

**与 L1/L2 Loss 的对比**: Charbonnier Loss 在 L2 的基础上加入了一个小的常数 \(\epsilon\)，避免了梯度在零点附近的不稳定。相比 L1 Loss (梯度恒定)，Charbonnier Loss 在误差较大时接近 L2 (梯度更大，收敛更快)，在误差较小时接近 L1 (梯度趋于常数，避免过拟合)。

---

### 为什么这样做 (Why This Design)

| 设计选择 | 原因 | 不这样做的后果 |
|----------|------|----------------|
| **U-shaped 编解码器** | 多尺度特征对图像恢复至关重要：浅层捕获边缘/纹理，深层捕获语义信息。U-shaped 结构能有效融合多尺度特征 | 单尺度网络无法同时恢复局部细节和全局结构 |
| **窗口注意力而非全局注意力** | 全局注意力 \(O(H^2W^2)\) 在高分辨率下显存不足；窗口注意力 \(O(M^2HW)\) 大幅降低计算量，同时保持局部注意力的精度 | 显存溢出，无法处理 256x256 及以上分辨率 |
| **LeFF 中的 DWConv** | 标准 FFN 的 MLP 对每个 token 独立处理，缺乏空间归纳偏置；DWConv 3x3 引入局部邻域建模，捕获纹理和边缘信息 | FFN 无法提取局部空间结构，恢复图像缺乏精细纹理 |
| **相对位置偏差 \(\hat{B}\)** | 窗口内像素间的空间关系对注意力权重很重要；相对位置偏差比绝对位置编码更好地泛化到不同分辨率 | 注意力缺乏位置感知，容易产生位置混淆 |
| **多尺度空间偏差调制器** | 编码器下采样丢失的细节无法仅靠上采样恢复；可学习偏差提供额外的空间信息补偿 | 解码器输出缺乏高频细节，恢复图像偏模糊 |
| ** Concatenation 跳跃连接** | 拼接保留了编码器的完整特征信息，让网络自主学习如何融合编码器和解码器特征 | 简单加法可能丢失编码器的独特信息 |
| **残差学习 (全局)** | 退化图像与清晰图像的差异远小于绝对像素值；学习残差比学习完整映射更容易收敛 | 直接回归绝对像素值，收敛慢，恢复质量差 |

---

## 四、实验与效果

### 4.1 训练配置

| 配置项 | 设置 |
|--------|------|
| **优化器** | AdamW (weight decay = 0.01) |
| **初始学习率** | 2e-4 |
| **学习率策略** | Warmup 3 epochs + Cosine Annealing 至 1e-6 |
| **数据增强** | 随机翻转、旋转 |
| **损失函数** | Charbonnier Loss (\(\epsilon = 10^{-3}\)) |

**任务特定配置**:

| 任务 | 数据集 | Epochs | Batch Size | Patch Size |
|------|--------|--------|------------|------------|
| **去噪** | SIDD | 250 | 32 | 128x128 |
| **去噪** | BSD68/Set12 | 300 | 32 | 128x128 |
| **去模糊** | GoPro | 3000 | 8 | 256x256 |
| **去雨** | Rain100L/H, Rain800 | 300 | 32 | 128x128 |

### 4.2 去噪结果 (SIDD 数据集)

| 方法 | PSNR (dB) | SSIM | 参数量 |
|------|-----------|------|--------|
| DnCNN | 23.66 | 0.583 | - |
| CBDNet | 38.71 | 0.951 | - |
| VDN | 39.28 | 0.956 | - |
| MPRNet | 39.52 | 0.957 | - |
| Restormer | 40.10 | 0.964 | 26.1M |
| **Uformer_B** | **40.03** | **0.963** | ~24.5M |

**关键发现**: Uformer_B 与 Restormer 性能相当 (40.03 vs 40.10 dB)，两者代表了 2022 年去噪领域的 SOTA 水平。

### 4.3 去雨结果

| 方法 | Rain100L PSNR/SSIM | Rain100H PSNR/SSIM | Rain800 PSNR/SSIM |
|------|---------------------|---------------------|---------------------|
| DSC | 40.24/0.9851 | 32.27/0.9416 | 30.66/0.9191 |
| MPRNet | 38.67/0.9831 | 30.79/0.9252 | 29.48/0.9154 |
| Restormer | 40.77/0.9858 | 32.53/0.9428 | 31.01/0.9245 |
| **Uformer_B** | **40.58/0.9864** | **32.49/0.9426** | **31.18/0.9256** |

**关键发现**:
- Rain100L 上 SSIM 超过 Restormer (0.9864 vs 0.9858)
- Rain800 上 PSNR 超过 Restormer (31.18 vs 31.01)
- 在三个去雨数据集上均表现优异

### 4.4 去模糊结果 (GoPro 数据集)

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| MPRNet | 32.66 | 0.959 |
| Restormer | 33.12 | 0.963 |
| **Uformer_B** | **33.15** | **0.964** |

**关键发现**: Uformer_B 在 GoPro 去模糊上达到了当时的 SOTA，PSNR 超越 Restormer。

### 4.5 散焦去模糊结果 (DPDD 数据集)

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| MPRNet | 30.41 | 0.947 |
| Restormer | 31.58 | 0.961 |
| **Uformer_B** | **31.65** | **0.962** |

### 4.6 消融实验

#### 4.6.1 LeWin Block vs 全局注意力 (GoPro)

| 配置 | PSNR (dB) |
|------|-----------|
| 全局自注意力 (Global SA) | 32.08 |
| 窗口注意力 (W-MSA) | 32.85 |
| **LeWin (W-MSA + LeFF)** | **33.15** |

**结论**: LeWin Block 比全局注意力高 1.07dB，同时大幅降低计算量。LeFF 相比标准 FFN 贡献了 0.30dB。

#### 4.6.2 LeFF vs 标准 FFN

| FFN 类型 | PSNR (GoPro) |
|----------|-------------|
| 标准 FFN (MLP) | 32.85 |
| **LeFF (MLP + DWConv)** | **33.15** |

**结论**: DWConv 在 FFN 中引入的局部空间建模能力带来了 0.30dB 的提升。

#### 4.6.3 多尺度调制器消融

| 配置 | PSNR (GoPro) |
|------|-------------|
| 无调制器 | 32.96 |
| **有调制器** | **33.15** |

**结论**: 空间偏差调制器贡献了 0.19dB 的提升，参数开销极小。

#### 4.6.4 跳跃连接类型消融

| 跳跃连接方案 | PSNR (GoPro) |
|-------------|-------------|
| No Skip | 31.92 |
| Cross-Skip | 33.03 |
| ConcatCross-Skip | 33.08 |
| **Concatenation (默认)** | **33.15** |

**结论**: Concatenation 方案在去模糊任务上效果最优，无跳跃连接性能最差。

#### 4.6.5 窗口大小消融

| 窗口大小 M | PSNR (GoPro) |
|-----------|-------------|
| M=4 | 33.01 |
| **M=8 (默认)** | **33.15** |
| M=16 | 33.08 |

**结论**: M=8 是最优窗口大小。过小 (M=4) 注意力范围不足，过大 (M=16) 计算量增加但收益递减。

---

## 五、对比总结

### 5.1 Uformer 与主流方法对比

| 对比维度 | Uformer | Restormer | MPRNet | SwinIR |
|----------|---------|-----------|--------|--------|
| **架构类型** | U-shaped Transformer | U-shaped Transformer | U-shaped CNN | U-shaped Transformer |
| **注意力机制** | 窗口注意力 (LeWin) | 通道注意力 + 空间注意力 | 无 (纯 CNN) | 窗口注意力 + 移位窗口 |
| **FFN 设计** | LeFF (MLP + DWConv) | GDFN (门控 DWConv) | 标准 Conv Block | 标准 MLP |
| **局部增强** | DWConv in FFN | Transposed Attention + DWConv | 多尺度卷积 | 无 (依赖移位窗口) |
| **空间调制** | 多尺度恢复调制器 | 无 | 无 | 无 |
| **SIDD PSNR** | 40.03 dB | **40.10 dB** | 39.52 dB | - |
| **GoPro PSNR** | **33.15 dB** | 33.12 dB | 32.66 dB | - |
| **Rain100L PSNR** | 40.58 dB | **40.77 dB** | 38.67 dB | - |
| **Rain800 PSNR** | **31.18 dB** | 31.01 dB | 29.48 dB | - |

### 5.2 核心优势

1. **通用性强**: 同一架构在去噪、去模糊、去雨、散焦去模糊四个任务上均达到 SOTA 或接近 SOTA
2. **LeFF 设计巧妙**: 在不增加显著计算量的前提下，通过 DWConv 为 FFN 注入局部归纳偏置
3. **多尺度调制器轻量有效**: 极少参数 (每层仅 M^2 x C) 实现显著的性能提升
4. **系统性跳跃连接分析**: 首次在 Transformer 恢复网络中系统比较不同跳跃连接方案

### 5.3 核心劣势

1. **窗口间无直接交互**: 与 Swin Transformer 不同，Uformer 没有移位窗口 (shifted window) 机制，窗口间信息交互仅通过跳跃连接和调制器间接实现
2. **FFN 中 DWConv 计算冗余**: LeFF 中的 reshape + DWConv + flatten 操作在实现上不够优雅，增加了工程复杂度
3. **训练代价较高**: GoPro 去模糊需要 3000 epochs，训练时间较长

---

## 六、不足与局限

| 序号 | 不足与局限 | 详细说明 |
|------|-----------|----------|
| 1 | **窗口间缺乏显式交互机制** | 与 Swin Transformer 的 Shifted Window 方案不同，Uformer 的窗口之间没有直接的跨窗口注意力交互，可能导致窗口边界处产生块效应 |
| 2 | **多尺度调制器的理论解释不足** | 空间偏差调制器虽然有效，但论文缺少对其作用机制的理论分析——它为什么能补偿信息损失？偏差值学到了什么模式？ |
| 3 | **训练超参数敏感** | GoPro 去模糊需要 3000 epochs 才能收敛，且不同任务需要不同的训练配置 (patch size、batch size、epochs 差异很大)，增加了实际使用的门槛 |
| 4 | **去噪任务略逊于 Restormer** | 在 SIDD 数据集上，Uformer_B (40.03dB) 略低于 Restormer (40.10dB)，去噪并非 Uformer 的最强任务 |
| 5 | **模型变体设计不够系统** | Uformer_T/S/B 的设计更多是经验性的，缺少针对不同任务的模型缩放策略指导 |

---

## 七、一句话总结

**Uformer 通过"LeWin Transformer Block (窗口注意力 + 局部增强 FFN) + 多尺度空间偏差调制器"的组合设计，构建了首个面向图像恢复的通用 U-shaped Transformer，在去噪、去模糊、去雨四大任务上均达到或接近 SOTA，证明了窗口注意力与局部卷积增强相结合是图像恢复 Transformer 设计的有效范式。**

---

## 八、生活化例子：小明的"旧影修复工作室"

> **场景五：夜景照片里的"星星"和"噪点"**

一位摄影师朋友找到小明，拿来一张珍贵的**夜景照片**——城市灯火璀璨，但照片里布满了**雪花般的噪点**，而且整体还有些模糊。"噪点和模糊混在一起，怎么修都修不干净。"

小明想起了 Uformer 的"分块看+调细节"策略：

"就像修表师傅戴放大镜工作——**先分块处理**，把照片切成 8×8 的小窗口，每个窗口单独看。这样在每个小区域内，噪点和真实细节的区别就更明显了。"小明先处理每个小窗口，用"局部增强"的方法像用细毛笔一样精修每个小块。

"但窗口之间不能各干各的呀！"于是小明又加了"多尺度调制器"——就像调色时，暗部要提亮、亮部要压暗，不同区域用不同的调整策略。夜景的天空暗部多降噪，灯火辉煌的亮部多保细节。

最后，照片里的星星一颗颗清晰起来，城市的轮廓也锐利了。

摄影师竖起大拇指："你这不是修图，是魔法！"

小明笑了：**分而治之，各个击破，再统筹全局——这就是修复杂照片的秘诀。**

---

> 小明的"魔法"吸引了各行各业的客户，接下来会是谁呢……

## 附录

### 附录 A: LeWin Transformer Block 伪代码

```python
class LeWinTransformerBlock(nn.Module):
    """Locally-enhanced Window Transformer Block"""
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Window Multi-head Self-Attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads
        )

        # LeFF: Locally-enhanced Feed-Forward Network
        self.leff = LeFF(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            window_size=window_size
        )

    def forward(self, x, x_size):
        """
        x: (B, N, C) flattened feature
        x_size: (H, W) original spatial size
        """
        H, W = x_size
        B, N, C = x.shape

        # ===== W-MSA with residual =====
        shortcut = x
        x = self.norm1(x)
        x_windows = window_partition(x, H, W, self.window_size)  # (num_win*B, win_size*win_size, C)
        attn_windows = self.attn(x_windows)                        # 窗口内自注意力
        x = window_reverse(attn_windows, H, W, self.window_size)   # 恢复空间布局
        x = shortcut + x  # 残差连接

        # ===== LeFF with residual =====
        shortcut = x
        x = self.norm2(x)
        x = self.leff(x, H, W)  # LeFF: Linear→GELU→Reshape→DWConv→GELU→Flatten→Linear
        x = shortcut + x         # 残差连接

        return x


class WindowAttention(nn.Module):
    """Window Multi-head Self-Attention with Relative Position Bias"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # M
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 可学习相对位置偏差
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 计算相对位置索引
        coords = self._get_relative_position_index(window_size)

    def forward(self, x):
        """
        x: (num_win*B, M*M, C)
        """
        B_, N, C = x.shape
        M = self.window_size

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B_, num_heads, M*M, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, M*M, M*M)

        # 加入相对位置偏差
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(M*M, M*M, -1)  # (M*M, M*M, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # (num_heads, M*M, M*M)
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x
```

### 附录 B: LeFF 伪代码

```python
class LeFF(nn.Module):
    """Locally-enhanced Feed-Forward Network"""
    def __init__(self, in_features, hidden_features, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.hidden_features = hidden_features

        # Linear 扩展
        self.linear1 = nn.Linear(in_features, hidden_features)

        # Depthwise Convolution (局部空间增强)
        self.dwconv = nn.Conv2d(
            hidden_features // window_size,  # 将通道分组
            hidden_features // window_size,
            kernel_size=3,
            padding=1,
            groups=hidden_features // window_size  # depthwise
        )

        # Linear 压缩
        self.linear2 = nn.Linear(hidden_features, in_features)

        self.act = nn.GELU()

    def forward(self, x, H, W):
        """
        x: (B, N, C), N = H*W
        """
        B, N, C = x.shape
        M = self.window_size

        # Step 1: Linear 扩展 (C → hidden_features)
        x = self.linear1(x)       # (B, N, hidden_features)
        x = self.act(x)

        # Step 2: Reshape 为空间格式 (为 DWConv 做准备)
        x = x.reshape(B, H, W, self.hidden_features).permute(0, 3, 1, 2)  # (B, hidden, H, W)

        # Step 3: 将通道分组后做 DWConv
        x = x.reshape(B, self.hidden_features // M, M, H, W)
        x = x.reshape(B * (self.hidden_features // M), M, H, W)  # 合并 batch
        x = self.dwconv(x)  # DWConv 3x3

        # Step 4: 恢复形状
        x = x.reshape(B, self.hidden_features // M, M, H, W)
        x = x.reshape(B, self.hidden_features, H, W)

        # Step 5: Flatten + Linear 压缩
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_features)
        x = self.act(x)
        x = self.linear2(x)              # (B, N, C)

        return x


def window_partition(x, H, W, window_size):
    """将特征图划分为不重叠窗口"""
    B, N, C = x.shape
    x = x.view(B, H, W, C)
    # 按窗口大小切分
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    # 重新排列: (B, num_win_h, num_win_w, win_h, win_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows  # (num_win*B, M*M, C)


def window_reverse(windows, H, W, window_size):
    """将窗口特征恢复为完整特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x.view(B, H * W, -1)  # (B, N, C)
```
