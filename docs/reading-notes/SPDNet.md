# SPDNet: Structure-Preserving Deraining with Residue Channel Prior Guidance 精读笔记

> [📄 arXiv](https://arxiv.org/abs/2108.09079) | 🎯 ICCV 2021 | [💻 代码](https://github.com/Joyies/SPDNet)

---

## 一、基本信息

| 属性 | 内容 |
|------|------|
| **论文标题** | Structure-Preserving Deraining with Residue Channel Prior Guidance |
| **发表会议** | ICCV 2021 |
| **作者** | Qiaosi Yi, Juncheng Li, Qinyan Dai, Faming Fang, Guixu Zhang, Tieyong Zeng |
| **单位** | 华东师范大学 (上海市多维信息处理重点实验室), 香港中文大学 (数学系) |
| **通讯作者** | Faming Fang (华东师范大学) |
| **核心创新** | 残差通道先验 (RCP) 引导结构保持去雨 + 小波多级模块 (WMLM) + 交互融合模块 (IFM) + 迭代引导策略 |
| **适用任务** | 单图像去雨 (Single Image Deraining) |
| **关键指标** | Rain100H PSNR 32.69dB / SSIM 0.9433, Rain100L PSNR 40.73dB, Rain800 PSNR 31.22dB |
| **数据集** | Rain100L, Rain100H, Rain800, SPA-Data |
| **训练配置** | Patch 128x128, Batch 16, Adam, lr=5e-4, 300 epochs, MSE Loss |
| **基础通道数** | n_feats=32, n_resblocks=3, scale=2 |

### 作者贡献分工

- **Qiaosi Yi, Juncheng Li**: 主要实验设计与实现
- **Faming Fang**: 通讯作者，研究方向为图像恢复与去雨
- **Tieyong Zeng**: 香港中文大学数学系，提供数学理论支撑
- **Guixu Zhang**: 华东师范大学，多维信息处理方向

---

## 二、痛点分析

| 痛点编号 | 痛点描述 | 深层原因 | 现有方案的不足 | SPDNet 的解决方案 |
|----------|----------|----------|----------------|-------------------|
| **P1** | **去雨过程破坏图像结构** | 现有方法直接学习雨纹层并从有雨图像中减去，但雨纹密度多变，导致过度去除或去除不足，恢复图像的结构信息不完整 | CNN 方法 (DDN, RESCAN, PReNet 等) 关注雨纹结构而非背景物体结构，缺乏对图像先验的利用 | 引入残差通道先验 (RCP) 显式保护结构信息，引导网络关注背景物体而非雨纹 |
| **P2** | **依赖雨生成模型假设** | 传统方法基于雨滴的物理模型 (折射、反射系数等)，对真实场景泛化差；CNN 方法虽不依赖物理模型，但仍隐式假设雨纹可分离 | 物理模型方法需要复杂迭代优化，CNN 方法在雨纹密度变化大时效果下降 | SPDNet 不依赖任何雨生成假设，仅利用从图像本身提取的 RCP 先验 |
| **P3** | **高频雨纹与高频边缘信息纠缠** | 雨纹和图像边缘/纹理都分布在高频域，直接在空间域或频域处理都难以有效分离 | 传统 TV 先验会平滑纹理细节，稀疏先验需要额外域知识，边缘检测器对雨纹敏感 | RCP 本身不含雨纹且保留结构，天然地将结构信息与雨纹解耦 |
| **P4** | **缺乏结构保持的显式引导** | 现有方法大多只有端到端的监督信号 (GT 图像)，缺少中间过程的显式结构约束 | 网络自由度太高，可能在去除雨纹的同时也破坏了纹理和边缘 | IFM 将 RCP 结构信息与骨干特征交互融合，迭代引导逐步提升结构精度 |
| **P5** | **合成数据与真实场景的域差距** | 合成雨数据 (Rain100H/L) 的雨纹模式单一，与真实雨景差异大 | 在合成数据上训练的模型在真实雨景上泛化能力差 | RCP 是从图像本身提取的先验，不依赖合成雨模型，对真实场景更鲁棒 |

### 2.1 现有先验方法的局限

SPDNet 在引言中系统分析了多种先验用于去雨的可行性：

| 先验类型 | 用于去雨的问题 | RCP 的优势 |
|----------|----------------|-----------|
| **Total Variation (TV)** | 会平滑纹理细节，导致恢复图像过度光滑 | RCP 保留纹理和边缘，不引入过度平滑 |
| **稀疏先验 (Sparse Prior)** | 难以建模，需要其他域知识 | RCP 计算简单，无需额外参数 |
| **边缘先验 (Edge Prior)** | 现有边缘检测器对雨纹敏感，提取的边缘含雨纹噪声 | RCP 天然不含雨纹，结构信息干净 |
| **层先验 (Layer Prior)** | 需要额外参数，如 GMM 等 | RCP 仅需 RGB 通道的 max-min 运算，零参数 |

---

## 三、核心方法

### 3.1 残差通道先验 (RCP) -- 核心创新

#### 3.1.1 关键发现

**RCP 包含比有雨图像更准确的结构信息**。这是 SPDNet 最核心的洞察。即使 RCP 是从有雨图像中提取的，它仍然不包含雨纹，只包含背景细节的变换版本。

直观理解：
- 有雨图像：既有雨纹也有背景结构，两者混合
- 有雨图像的 RCP：只有背景结构，**不含雨纹**
- 无雨图像的 RCP：干净的背景结构
- 因此，RCP 的差异比原始图像的差异更能指导结构恢复

#### 3.1.2 RCP 的数学推导

根据 Li et al. 的工作，彩色图像的雨纹强度可表示为：

\[
\tilde{\mathcal{O}}(x) = t\beta_{rs}(x)\mathcal{B}\alpha + (T - t)\mathcal{R}\pi
\]

其中：
- $\tilde{O}(x)$: 有雨图像的彩色强度向量
- $\beta_{rs}$: 雨滴的折射、镜面反射和内部反射系数
- $\mathcal{B} = (B_r, B_g, B_b)^T$: 光源亮度，$\mathcal{B}' = B_r + B_g + B_b$
- $\mathcal{R} = (R_r, R_g, R_b)^T$: 背景反射，$\mathcal{R}' = R_r + R_g + R_b$
- $\alpha = \mathcal{B}/\mathcal{B}'$: 光源色度
- $\pi = \mathcal{R}/\mathcal{R}'$: 背景色度
- $T$: 曝光时间，$t$: 雨滴通过像素 $x$ 的时间

第一项为**雨纹项**，第二项为**背景项**。通过颜色恒常性算法估计 $\alpha$ 后进行归一化：

\[
\mathcal{O}(x) = \frac{\tilde{\mathcal{O}}(x)}{\alpha} = O_{rs}(x)\mathbf{i} + O_{bg}(x)
\]

其中 $\mathbf{i} = (1, 1, 1)^T$。此时，对 $\mathcal{O}(x)$ 求各通道之间的差值：

\[
\mathcal{O}_r(x) - \mathcal{O}_g(x) = O_{bg,r}(x) - O_{bg,g}(x) = R_{rcp}(x)
\]

**关键结论**: 雨纹项 $O_{rs}(x)\mathbf{i}$ 在通道相减时被完全消除，RCP 中**只保留背景结构信息**，不含任何雨纹。

#### 3.1.3 RCP 的实际计算

在实际实现中，RCP 通过 RGB 通道的最大值减最小值提取：

\[
\text{RCP}(x) = \max(R(x), G(x), B(x)) - \min(R(x), G(x), B(x))
\]

这等价于计算 RGB 三通道之间的极差，无需颜色恒常性估计，计算极其简洁。

```
输入: 有雨图像 I (H x W x 3)
    │
    ▼
RCP = max(R, G, B) - min(R, G, B)  (逐像素)
    │
    ▼
输出: RCP 图 (H x W x 1)
    │
    特性: ✗ 不含雨纹
          ✓ 保留背景结构 (边缘、纹理)
          ✓ 零参数，计算高效
```

### 3.2 整体架构

SPDNet 的整体架构采用 **RCP 引导的迭代渐进去雨框架**：

```
有雨图像 O
    │
    ├──→ [RCP 提取模块] → RCP_0
    │         │
    ▼         ▼
┌─────────────────────────────────────────┐
│          迭代阶段 (×N, 默认 N=2)         │
│                                          │
│  ┌───────────────────────────────────┐    │
│  │  Wavelet-based Feature Extraction │    │
│  │  Backbone (WMLM × 多级)          │    │
│  │                                    │    │
│  │  Level 0: SRiR(Ỹ_0)              │    │
│  │    ↓ DWT + Conv                   │    │
│  │  Level 1: SRiR(Ỹ_1)              │    │
│  │    ↓ DWT + Conv                   │    │
│  │  Level 2: SRiR(Ỹ_2)              │    │
│  │    ↑ IWT + Conv (逐级上采样融合)    │    │
│  └──────────┬────────────────────────┘    │
│             │                             │
│             ▼                             │
│  ┌───────────────────────────────────┐    │
│  │  Interactive Fusion Module (IFM) │    │
│  │                                    │    │
│  │  RCP 特征 ──→ [交互融合] ←── 骨干特征 │    │
│  │                │                   │    │
│  │                ▼                   │    │
│  │          增强后的特征                │    │
│  └──────────┬────────────────────────┘    │
│             │                             │
│             ▼                             │
│        中间去雨结果 Ỹ                      │
│             │                             │
│             ▼                             │
│    [RCP 提取模块] → RCP_1 (更准确)         │
│             │                             │
│             ▼                             │
│       进入下一次迭代...                     │
└─────────────────────────────────────────┘
    │
    ▼
  最终去雨结果
```

**核心数据流**:
1. 从有雨图像提取初始 RCP
2. 骨干网络 (WMLM) 学习背景信息
3. IFM 将 RCP 结构信息与骨干特征交互融合
4. 输出中间去雨结果
5. 从中间结果提取更准确的 RCP
6. 重复迭代，逐步精炼

### 3.3 小波多级模块 (WMLM) -- 骨干网络

#### 3.3.1 设计动机

雨纹的大小和密度任意变化，被遮挡区域和遮挡程度未知。直接使用下采样或反卷积会造成大量信息丢失。DWT (离散小波变换) 能同时捕获特征图的**频率和位置信息**，有助于保留细节纹理。

#### 3.3.2 WMLM 的具体结构

WMLM 采用 3 级小波分解与重构，每级包含 SE-ResBlock：

```
输入特征 Ỹ (C x H x W)
    │
    ├──→ Level 0: SRiR(Ỹ_0)          ──→ Ỹ_0^s
    │         │
    │         ▼ DWT (离散小波变换)
    │    LL_0, LH_0, HL_0, HH_0 (4C x H/2 x W/2)
    │         │
    │    Conv (通道调整) → Ỹ_1
    │         │
    ├──→ Level 1: SRiR(Ỹ_1)          ──→ Ỹ_1^s
    │         │
    │         ▼ DWT
    │    LL_1, LH_1, HL_1, HH_1 (8C x H/4 x W/4)
    │         │
    │    Conv (通道调整) → Ỹ_2
    │         │
    └──→ Level 2: SRiR(Ỹ_2)          ──→ Ỹ_2^s
              │
              ▼
         Conv + IWT (逆小波变换) → Ỹ_1^s = IWT(Conv(Ỹ_2^s)) + Ỹ_1^s
              │
              ▼
         Conv + IWT → Ỹ_0^s = IWT(Conv(Ỹ_1^s)) + Ỹ_0^s
```

#### 3.3.3 数学表达

**下采样路径** (公式1):

\[
\tilde{\mathcal{F}}_{i} = \begin{cases}
\tilde{\mathcal{F}}, & \text{if } i = 0 \\
\text{Conv}(\text{DWT}(\tilde{\mathcal{F}}_{i-1})), & \text{if } i > 0
\end{cases}
\]

**SRiR 处理** (公式2):

\[
\tilde{\mathcal{F}}_{i}^{s} = \text{SRiR}(\tilde{\mathcal{F}}_{i}), \quad i = 0, 1, 2
\]

**上采样融合** (公式3):

\[
\tilde{\mathcal{F}}_{i-1}^{s} = \text{IWT}(\text{Conv}(\tilde{\mathcal{F}}_{i}^{s})) + \tilde{\mathcal{F}}_{i-1}^{s}, \quad i = 2, 1
\]

#### 3.3.4 Haar 小波分解详解

SPDNet 使用 Haar 小波进行 DWT，将输入分解为 4 个子带：

| 子带 | 含义 | 捕捉的信息 |
|------|------|-----------|
| **LL** (Low-Low) | 近似分量 | 低频背景结构、大面积平滑区域 |
| **LH** (Low-High) | 水平细节 | 垂直边缘、水平方向的高频变化 |
| **HL** (High-Low) | 垂直细节 | 水平边缘、垂直方向的高频变化 |
| **HH** (High-High) | 对角细节 | 对角线边缘、对角纹理 |

Haar 小波变换的核心公式 (以一维为例):

\[
LL = \frac{a + b}{2}, \quad HH = \frac{a - b}{2}
\]

其中 $a, b$ 为相邻像素值。LL 保留平均信息 (低频)，HH 保留差分信息 (高频)。

### 3.4 交互融合模块 (IFM)

#### 3.4.1 设计动机

仅有骨干网络 (WMLM) 不足以保持重建图像的清晰结构。需要将 RCP 的结构信息与骨干网络的背景特征进行有效交互。

#### 3.4.2 IFM 的具体设计

IFM 的核心是让 RCP 特征与骨干特征进行**交互式注意力融合**：

```
RCP 特征 F_rcp ──→ Conv ──→ F_rcp'
                         │
骨干特征 F_backbone ──→ Conv ──→ F_bb'
                         │
                         ▼
                  Concat([F_rcp', F_bb'])
                         │
                         ▼
                    Conv + Sigmoid
                         │
                         ▼
                  注意力掩码 M ∈ (0, 1)
                         │
                         ▼
    F_fused = F_bb ⊙ M + F_rcp ⊙ (1 - M)
```

**核心思想**: RCP 特征指导骨干网络"哪些区域需要重点保持结构"。注意力掩码 M 决定了每个空间位置更依赖骨干特征还是 RCP 特征。

### 3.5 迭代引导策略

#### 3.5.1 设计动机

论文发现：**RCP 的质量随着图像质量的提升而提升**。有雨图像的 RCP 虽不含雨纹，但仍受雨纹间接影响 (如雨纹导致的颜色偏移)，精度有限；而初步去雨后图像的 RCP 更接近无雨图像的 RCP，结构信息更准确。

#### 3.5.2 迭代流程

```
第 1 次迭代:
  输入: 有雨图像 O
  RCP 提取: RCP_0 = RCP(O)           ← 从有雨图像提取，精度较低
  骨干网络: 学习背景信息
  IFM 融合: RCP_0 指导结构保持
  输出: 中间去雨结果 Ỹ_1              ← 初步去雨

第 2 次迭代:
  输入: 中间去雨结果 Ỹ_1
  RCP 提取: RCP_1 = RCP(Ỹ_1)         ← 从初步去雨结果提取，精度提高
  骨干网络: 进一步学习背景信息
  IFM 融合: RCP_1 (更准确) 指导精炼
  输出: 最终去雨结果 Ỹ_2              ← 高质量去雨

(可继续迭代 N 次，论文默认 N=2)
```

**关键洞察**: 每次迭代产生的中间去雨结果质量更好，从中提取的 RCP 也更准确，形成**正向反馈循环**。

---

## 三.5 数学推导过程详解 (Mathematical Walkthrough)

> 以下用一个 **4x4 像素有雨 RGB 图像块** 完整走一遍 SPDNet 的 RCP 提取、WMLM 处理和 IFM 融合的具体数值计算。

### 设定输入

假设一个 4x4 有雨图像的 R 通道 (简化分析):

$$
O_R = \begin{bmatrix}
120 & 135 & 125 & 140 \\
130 & 145 & 132 & 138 \\
128 & 142 & 136 & 122 \\
134 & 130 & 118 & 126
\end{bmatrix}_{4 \times 4}
$$

G 通道:

$$
O_G = \begin{bmatrix}
115 & 130 & 120 & 135 \\
125 & 140 & 127 & 133 \\
123 & 137 & 131 & 117 \\
129 & 125 & 113 & 121
\end{bmatrix}_{4 \times 4}
$$

B 通道:

$$
O_B = \begin{bmatrix}
110 & 128 & 118 & 132 \\
122 & 138 & 125 & 130 \\
120 & 134 & 128 & 114 \\
126 & 122 & 110 & 118
\end{bmatrix}_{4 \times 4}
$$

> 雨纹叠加在背景上，导致各通道值偏高且通道间关系发生变化。

---

### Step 1: RCP 提取

**中文标题**: 残差通道先验提取
**English Title**: Residue Channel Prior Extraction

RCP = max(R, G, B) - min(R, G, B) (逐像素):

以位置 (0,0) 为例:
$$
\text{RCP}(0,0) = \max(120, 115, 110) - \min(120, 115, 110) = 120 - 110 = 10
$$

以位置 (1,1) 为例:
$$
\text{RCP}(1,1) = \max(145, 140, 138) - \min(145, 140, 138) = 145 - 138 = 7
$$

完整 RCP 图:

$$
\text{RCP} = \begin{bmatrix}
10 & 7 & 7 & 8 \\
8 & 7 & 7 & 8 \\
8 & 8 & 8 & 8 \\
8 & 8 & 8 & 8
\end{bmatrix}_{4 \times 4}
$$

> RCP 图中不包含雨纹的高频噪声，只保留背景的平滑结构信息。雨纹在 RGB 三通道中近似均匀叠加 (因为雨滴接近白色)，因此 max-min 差值将雨纹项消除。

**维度变化**: 4x4x3 → 4x4x1 (RCP 提取)

---

### Step 2: Haar 小波分解 (DWT)

**中文标题**: Haar 小波分解
**English Title**: Haar Wavelet Decomposition

对 RCP 图进行一级 Haar DWT:

**LL 子带** (低频近似):

$$
LL = \begin{bmatrix}
\frac{10+7+8+7}{4} & \frac{7+8+7+8}{4} \\
\frac{8+8+8+8}{4} & \frac{8+8+8+8}{4}
\end{bmatrix}
= \begin{bmatrix}
8.0 & 7.5 \\
8.0 & 8.0
\end{bmatrix}_{2 \times 2}
$$

**LH 子带** (水平细节):

$$
LH = \begin{bmatrix}
\frac{10+7-8-7}{4} & \frac{7+8-7-8}{4} \\
\frac{8+8-8-8}{4} & \frac{8+8-8-8}{4}
\end{bmatrix}
= \begin{bmatrix}
0.5 & 0.0 \\
0.0 & 0.0
\end{bmatrix}_{2 \times 2}
$$

**HL 子带** (垂直细节):

$$
HL = \begin{bmatrix}
\frac{10-7+8-7}{4} & \frac{7-8+7-8}{4} \\
\frac{8-8+8-8}{4} & \frac{8-8+8-8}{4}
\end{bmatrix}
= \begin{bmatrix}
1.0 & -0.5 \\
0.0 & 0.0
\end{bmatrix}_{2 \times 2}
$$

**HH 子带** (对角细节):

$$
HH = \begin{bmatrix}
\frac{10-7-8+7}{4} & \frac{7-8-7+8}{4} \\
\frac{8-8-8+8}{4} & \frac{8-8-8+8}{4}
\end{bmatrix}
= \begin{bmatrix}
0.5 & 0.0 \\
0.0 & 0.0
\end{bmatrix}_{2 \times 2}
$$

> LL 保留了 RCP 的整体结构 (背景亮度), LH/HL/HH 捕捉水平和垂直边缘。RCP 的高频子带值很小，说明 RCP 本身确实是一个结构平滑的先验。

**维度变化**: 4x4 → 4 个 2x2 子带 (DWT)

---

### Step 3: SRiR (SE-ResBlock in Residual Block) 处理

**中文标题**: SE-ResBlock 残差块处理
**English Title**: SE-ResBlock in Residual Block Processing

SRiR 对每个尺度的特征进行学习:

$$
\tilde{\mathcal{F}}_{i}^{s} = \text{SRiR}(\tilde{\mathcal{F}}_{i})
$$

以 Level 0 为例 (4x4 空间, C=32 通道):

$$
F_{in} \in \mathbb{R}^{32 \times 4 \times 4}
$$

经过 3 个 SE-ResBlock:
- 每个 SE-ResBlock: 3x3 Conv → SE Attention → 3x3 Conv + Shortcut
- SE Attention 对通道重要性进行加权

$$
F_{SE} = \sigma(\text{FC}_2(\text{ReLU}(\text{FC}_1(\text{GAP}(F))))) \in \mathbb{R}^{C}
$$

$$
F_{out} = F_{SE} \odot F_{mid}
$$

> SE 注意力让网络自适应地关注重要通道 (如捕捉背景结构的通道)，抑制次要通道 (如含噪声的通道)。

**维度变化**: 32x4x4 → 32x4x4 (SRiR 不改变空间维度和通道数)

---

### Step 4: IFM 交互融合

**中文标题**: 交互融合模块
**English Title**: Interactive Fusion Module

IFM 将 RCP 特征与骨干特征进行交叉注意力融合:

**RCP 分支**:

$$
F_{rcp} = \text{Conv}(\text{RCP\_feat}) \in \mathbb{R}^{C \times H \times W}
$$

**骨干分支**:

$$
F_{bb} = \text{Conv}(\text{Backbone\_feat}) \in \mathbb{R}^{C \times H \times W}
$$

**注意力掩码生成**:

$$
M = \sigma(\text{Conv}(\text{Concat}([F_{rcp}, F_{bb}]))) \in \mathbb{R}^{1 \times H \times W}
$$

以 2x2 空间为例:

$$
M = \begin{bmatrix} 0.7 & 0.6 \\ 0.8 & 0.5 \end{bmatrix}
$$

> 注意力值高 (如 0.8) 表示该位置更依赖骨干特征 (背景信息丰富), 注意力值低 (如 0.5) 表示更依赖 RCP 特征 (结构需要保护)。

**融合输出**:

$$
F_{fused} = F_{bb} \odot M + F_{rcp} \odot (1 - M)
$$

以位置 (0,0) 为例:

$$
F_{fused}(0,0) = F_{bb}(0,0) \times 0.7 + F_{rcp}(0,0) \times 0.3
$$

> 融合结果综合了骨干网络的背景学习能力和 RCP 的结构保护能力。

---

### Step 5: 迭代精炼

**中文标题**: 迭代引导策略
**English Title**: Iterative Guidance Strategy

第一次迭代后的中间结果:

$$
\hat{O}_1 = \text{SPDNet}_1(O, \text{RCP}_0)
$$

从中提取更准确的 RCP:

$$
\text{RCP}_1 = \max(\hat{O}_1^R, \hat{O}_1^G, \hat{O}_1^B) - \min(\hat{O}_1^R, \hat{O}_1^G, \hat{O}_1^B)
$$

由于 $\hat{O}_1$ 已经去除了大部分雨纹，$\text{RCP}_1$ 比 $\text{RCP}_0$ 更接近无雨图像的 RCP:

$$
\text{RCP}_0 \xrightarrow{\text{去雨改善}} \text{RCP}_1 \xrightarrow{\text{进一步改善}} \text{RCP}_2 \rightarrow \dots
$$

第二次迭代:

$$
\hat{O}_2 = \text{SPDNet}_2(\hat{O}_1, \text{RCP}_1)
$$

> 每次迭代，RCP 的结构信息更准确，去雨结果更清晰。

**维度变化**: 每次迭代不改变空间维度，但输出质量逐步提升

---

### Step 6: 损失函数计算

**中文标题**: 损失函数
**English Title**: Loss Function

SPDNet 使用 **MSE Loss**:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{O}_i - O_{gt} \|_2^2
$$

其中 $\hat{O}_i$ 是去雨结果，$O_{gt}$ 是 Ground Truth 无雨图像。

以 4x4 单通道为例 (简化):

$$
\text{MSE} = \frac{1}{16} \sum_{h,w} (\hat{O}(h,w) - O_{gt}(h,w))^2
$$

假设中心位置 (1,1) 的预测值为 143.2, GT 为 143:

$$
(\hat{O}(1,1) - O_{gt}(1,1))^2 = (143.2 - 143)^2 = 0.04
$$

> 训练命令中使用 `--loss 1*MSE`，即标准均方误差损失。

---

### 为什么这样做 (Why This Design)

| 设计选择 | 原因 | 不这样做的后果 |
|----------|------|---------------|
| **使用 RCP 而非其他先验** | RCP 天然不含雨纹 (通道相减消除均匀雨纹)，且计算零参数；TV 先验会平滑纹理，边缘先验受雨纹干扰 | 使用 TV 先验会导致恢复图像过度平滑，使用边缘先验会引入雨纹伪边缘 |
| **用 max-min 而非逐通道差分提取 RCP** | max-min 计算更鲁棒，不受通道顺序影响，对所有颜色组合都适用 | 逐通道差分 (R-G, G-B) 可能丢失某些颜色区域的结构信息 |
| **WMLM 用 DWT 替代 Pooling/Deconv** | DWT 同时保留频率和位置信息，无信息丢失；Pooling 丢失高频细节，Deconv 引入棋盘效应 | 直接下采样会丢失雨纹区域的背景细节，导致恢复后在雨纹区域出现模糊 |
| **IFM 交叉注意力融合** | 让网络自适应学习每个位置更依赖 RCP 还是骨干特征；RCP 在结构丰富区域更重要 | 简单相加无法区分哪些区域需要结构保护，可能导致结构过度修改 |
| **迭代引导而非单次推理** | RCP 质量随去雨质量提升而提升，迭代形成正向反馈；单次推理的 RCP 精度有限 | 单次推理只能使用不精确的初始 RCP，结构保持效果受限 |
| **SRiR 中的 SE 注意力** | SE 注意力自动选择重要特征通道，抑制噪声通道；去雨需要区分雨纹通道和背景通道 | 所有通道等权重处理，雨纹相关通道会干扰背景恢复 |
| **MSE 损失函数** | MSE 在图像恢复中广泛使用，对大误差惩罚更重，有利于整体 PSNR 提升 | 使用 L1 可能导致恢复结果偏粗糙，使用感知损失可能偏离像素级精度 |

---

## 四、实验与效果

### 4.1 训练配置

| 配置项 | 设置 |
|--------|------|
| **框架** | PyTorch 0.4.1 |
| **Patch Size** | 128 x 128 |
| **Batch Size** | 16 |
| **优化器** | Adam |
| **学习率** | 5e-4 |
| **训练轮数** | 300 epochs |
| **损失函数** | MSE Loss |
| **基础通道数** | n_feats = 32 |
| **ResBlock 数量** | n_resblocks = 3 |
| **多级尺度数** | scale = 2 (3 级) |
| **数据增强** | 随机裁剪 patch |
| **评估指标** | PSNR, SSIM (基于 YCbCr 的 Y 通道计算) |
| **小波工具** | pytorch_wavelets |

### 4.2 去雨结果对比

#### 4.2.1 Rain100H (重度雨)

| 方法 | PSNR (dB) | SSIM | 年份 |
|------|-----------|------|------|
| DID-MDN | 28.14 | 0.8532 | 2020 |
| JORDER | 28.65 | 0.8695 | 2018 |
| PReNet | 29.45 | 0.8822 | 2019 |
| SPA-Net | 31.99 | 0.9390 | 2020 |
| MPRNet | 30.79 | 0.9252 | 2021 |
| **SPDNet** | **32.69** | **0.9433** | 2021 |

#### 4.2.2 Rain100L (轻度雨)

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| JORDER | 36.47 | 0.9743 |
| PReNet | 37.76 | 0.9830 |
| MPRNet | 38.67 | 0.9875 |
| SPA-Net | 39.21 | 0.9878 |
| **SPDNet** | **40.73** | **0.9894** |

#### 4.2.3 Rain800

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| PReNet | 27.94 | 0.8776 |
| MPRNet | 29.48 | 0.9154 |
| SPA-Net | 30.22 | 0.9208 |
| **SPDNet** | **31.22** | **0.9277** |

#### 4.2.4 SPA-Data (大规模数据集)

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| RCDNet | 37.78 | 0.9739 |
| SPA-Net | 39.45 | 0.9826 |
| MPRNet | 38.61 | 0.9782 |
| **SPDNet** | **40.27** | **0.9870** |

**关键发现**:
- SPDNet 在所有合成数据集上均达到 SOTA
- Rain100H 上 PSNR 超出 SPA-Net 0.70dB，超出 MPRNet 1.90dB
- Rain100L 上 PSNR 超出 SPA-Net 1.52dB
- SPA-Data 大规模数据集上表现突出，证明了 RCP 引导的泛化能力

### 4.3 消融实验

#### 4.3.1 各模块消融 (Rain100H)

| 配置 | PSNR (dB) | SSIM |
|------|-----------|------|
| 完整 SPDNet | **32.69** | **0.9433** |
| 去掉 RCP 引导 | 30.86 | 0.9258 |
| 去掉 WMLM (用普通下采样) | 31.52 | 0.9342 |
| 去掉 IFM (简单拼接) | 31.78 | 0.9367 |
| 去掉迭代引导 (单次推理) | 32.01 | 0.9385 |

**结论**:
- **RCP 引导贡献最大** (+1.83dB): 证明 RCP 对结构保持的关键作用
- WMLM 贡献 +1.17dB: 小波变换比普通下采样更有效地保留细节
- IFM 贡献 +0.91dB: 交互融合比简单拼接更有效
- 迭代引导贡献 +0.68dB: 渐进精炼比单次推理更精确

#### 4.3.2 迭代次数的影响

| 迭代次数 | PSNR (dB) | SSIM |
|----------|-----------|------|
| 1 次 | 32.01 | 0.9385 |
| **2 次 (默认)** | **32.69** | **0.9433** |
| 3 次 | 32.72 | 0.9435 |

**结论**: 2 次迭代已接近收敛，3 次迭代仅带来边际提升 (+0.03dB)，考虑到推理效率，默认使用 2 次。

#### 4.3.3 WMLM 级数的影响

| 级数 | PSNR (dB) |
|------|-----------|
| 1 级 | 31.45 |
| **2 级 (默认)** | **32.69** |
| 3 级 | 32.55 |

**结论**: 2 级小波分解是最优平衡点，更多级数可能因过度下采样而丢失空间信息。

### 4.4 下游任务验证: 目标检测

为验证去雨对下游任务的实际价值，论文在去雨后的图像上进行目标检测：

| 数据集 | 方法 | mAP |
|--------|------|-----|
| Rainy COD | 有雨图像 (直接检测) | 较低 |
| Rainy COD | SPA-Net 去雨后检测 | 中等 |
| Rainy COD | **SPDNet 去雨后检测** | **最高** |

**结论**: SPDNet 去雨后图像的结构保持更好，为目标检测器提供了更高质量的输入。

### 4.5 真实雨景效果

论文展示了在真实雨天图像上的去雨效果：
- SPDNet 能有效去除真实雨纹，同时保持建筑、行人等物体的边缘清晰
- 相比 SPA-Net 和 MPRNet，SPDNet 在纹理保持方面表现更好
- 不依赖合成雨模型使得 SPDNet 在真实场景中更具鲁棒性

---

## 五、对比总结

### 5.1 SPDNet 与主流方法详细对比

| 对比维度 | SPDNet | SPA-Net | MPRNet | PReNet |
|----------|--------|---------|--------|--------|
| **核心机制** | RCP 引导 + 小波 + 迭代 | 自注意力先验 | 多阶段渐进恢复 | 递归展开网络 |
| **先验利用** | RCP (显式, 从图像提取) | 层先验 (图像分解) | 无显式先验 | 无显式先验 |
| **结构保持** | 强 (RCP 显式保护) | 中等 | 较弱 | 较弱 |
| **Rain100H PSNR** | **32.69** | 31.99 | 30.79 | 29.45 |
| **Rain100L PSNR** | **40.73** | 39.21 | 38.67 | 37.76 |
| **Rain800 PSNR** | **31.22** | 30.22 | 29.48 | 27.94 |
| **迭代/多阶段** | 是 (2 次迭代) | 否 | 是 (3 阶段) | 是 (递归) |
| **下采样方式** | DWT (小波) | Pooling | Pooling + Conv | 无下采样 |
| **雨模型假设** | 不依赖 | 不依赖 | 不依赖 | 不依赖 |

### 5.2 核心优势

1. **RCP 先验的理论创新**: 首次将残差通道先验引入去雨任务，提供了零参数、高效的结构保护机制
2. **结构保持能力突出**: 在所有数据集上，SPDNet 的 SSIM 提升幅度均大于 PSNR，说明结构保持是核心优势
3. **不依赖雨模型假设**: 适用于合成和真实雨景，泛化能力强
4. **迭代引导的正向反馈**: RCP 质量和去雨质量互相促进，形成优雅的渐进精炼机制
5. **小波变换与深度学习的有效结合**: DWT 替代下采样，保留频率和位置信息

### 5.3 核心劣势

1. **推理阶段需要多次前向传播**: 迭代引导策略 (2 次) 增加了推理时间
2. **小波变换的硬件部署挑战**: DWT/IWT 在移动端优化不如标准卷积成熟
3. **PyTorch 版本较旧**: 代码基于 PyTorch 0.4.1，与现代 PyTorch 版本兼容性较差
4. **仅验证了去雨任务**: RCP 先验在其他图像恢复任务 (去雾、去雪、去噪) 上的泛化性未验证

---

## 六、不足与局限

| 序号 | 不足与局限 | 详细说明 |
|------|-----------|----------|
| 1 | **迭代推理增加延迟** | 2 次迭代需要 2 次完整的前向传播，推理速度约为单次推理的 2 倍，不利于实时应用 |
| 2 | **RCP 对极端雨情的鲁棒性未充分验证** | 在暴雨、雨雾混合等极端条件下，RCP 是否仍能有效提取结构信息需要更多实验验证 |
| 3 | **小波基的选择缺乏讨论** | 论文使用 Haar 小波，但未系统对比 Db2、Sym 等其他小波基的效果差异 |
| 4 | **代码库维护不足** | 代码基于 PyTorch 0.4.1 (2018 年版本)，与现代深度学习生态不兼容，复现成本高 |
| 5 | **通道数和级数的理论指导缺失** | n_feats=32、scale=2 的选择基于实验，缺少理论或自适应机制 |
| 6 | **对彩色图像的 RCP 利用不充分** | RCP 被简化为单通道 (max-min)，未充分利用 RGB 通道间的方向性差异信息 |
| 7 | **与最新 Transformer 方法的对比不足** | 未与 Restormer、SwinIR 等基于 Transformer 的图像恢复方法进行对比 |

---

## 七、一句话总结

**SPDNet 通过发现"残差通道先验 (RCP) 比有雨图像本身包含更准确的结构信息"这一关键洞察，设计了一套 RCP 引导的迭代渐进去雨框架，在零雨模型假设下实现了结构保持的单图像去雨 SOTA，是"从数据自身挖掘先验"思想在图像恢复任务中的经典范例。**

---

## 附录：关键模块伪代码

```python
class RCPExtraction(nn.Module):
    """残差通道先验提取模块"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: B x 3 x H x W (RGB 图像)
        # RCP = max(R,G,B) - min(R,G,B)
        rcp = torch.max(x, dim=1)[0] - torch.min(x, dim=1)[0]
        return rcp  # B x 1 x H x W


class WMLM(nn.Module):
    """小波多级模块 (Wavelet-based Multi-Level Module)"""
    def __init__(self, n_feats=32, n_resblocks=3, scale=2):
        super().__init__()
        self.scale = scale

        # 每级包含 SE-ResBlock
        self.levels = nn.ModuleList()
        for i in range(scale + 1):
            in_ch = n_feats * (2 ** i)
            self.levels.append(
                nn.Sequential(*[SEResBlock(in_ch) for _ in range(n_resblocks)])
            )

        # 下采样: DWT + Conv
        self.downsample = nn.ModuleList()
        for i in range(scale):
            in_ch = n_feats * (2 ** i)
            out_ch = n_feats * (2 ** (i + 1))
            self.downsample.append(nn.Sequential(
                DWTForward(wave='db1', J=1),  # Haar 小波
                nn.Conv2d(in_ch * 4, out_ch, 1)  # DWT 产生 4 倍通道, 1x1 压缩
            ))

        # 上采样: Conv + IWT
        self.upsample = nn.ModuleList()
        for i in range(scale):
            in_ch = n_feats * (2 ** (scale - i))
            out_ch = n_feats * (2 ** (scale - i - 1))
            self.upsample.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 1),
                DWTInverse(wave='db1', J=1)  # 逆小波变换
            ))

    def forward(self, x):
        # 编码路径 (下采样)
        feats = []
        for i in range(self.scale + 1):
            if i > 0:
                x = self.downsample[i-1](x)
            x = self.levels[i](x)
            feats.append(x)

        # 解码路径 (上采样 + 跳跃连接)
        for i in range(self.scale - 1, -1, -1):
            x = self.upsample[i](feats[i + 1]) + feats[i]

        return x


class IFM(nn.Module):
    """交互融合模块 (Interactive Fusion Module)"""
    def __init__(self, n_feats):
        super().__init__()
        self.rcp_conv = nn.Conv2d(1, n_feats, 3, padding=1)
        self.bb_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feats, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, rcp_feat, backbone_feat):
        # rcp_feat: B x 1 x H x W
        # backbone_feat: B x C x H x W
        rcp = self.rcp_conv(rcp_feat)        # B x C x H x W
        bb = self.bb_conv(backbone_feat)     # B x C x H x W

        # 生成注意力掩码
        mask = self.fusion_conv(
            torch.cat([rcp, bb], dim=1)
        )  # B x 1 x H x W

        # 交叉注意力融合
        fused = bb * mask + rcp * (1 - mask)
        return fused


class SPDNet(nn.Module):
    """结构保持去雨网络 (完整架构)"""
    def __init__(self, n_feats=32, n_resblocks=3, scale=2, n_iterations=2):
        super().__init__()
        self.rcp_extract = RCPExtraction()
        self.backbone = WMLM(n_feats, n_resblocks, scale)
        self.ifm = IFM(n_feats)
        self.recon = nn.Conv2d(n_feats, 3, 3, padding=1)
        self.n_iterations = n_iterations

    def forward(self, x):
        # 初始 RCP 提取
        rcp = self.rcp_extract(x)  # B x 1 x H x W

        # 迭代引导
        for i in range(self.n_iterations):
            # 骨干网络学习背景
            bg_feat = self.backbone(x)  # B x C x H x W

            # IFM: RCP 引导结构保持融合
            fused_feat = self.ifm(rcp, bg_feat)  # B x C x H x W

            # 重建去雨图像
            x = x + self.recon(fused_feat)  # 残差学习

            # 更新 RCP (从更干净的图像中提取更准确的 RCP)
            rcp = self.rcp_extract(x)

        return x  # 最终去雨结果
```
