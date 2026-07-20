# Multi-Stage Progressive Image Restoration (MPRNet) 精读笔记

> [📄 arXiv](https://arxiv.org/abs/2102.02808) | 🎯 CVPR 2021 | [💻 代码](https://github.com/swz30/MPRNet)

## 一、基本信息

| 属性 | 内容 |
|------|------|
| **论文标题** | Multi-Stage Progressive Image Restoration |
| **发表会议** | CVPR 2021 |
| **作者** | Syed Waqas Zamir*, Aditya Arora*, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao |
| **单位** | Inception Institute of AI (IIAI), Mohamed bin Zayed University of AI (MBZUAI), Monash University, UC Merced |
| **核心创新** | 多阶段渐进式恢复架构：Encoder-Decoder 学上下文 + ORSNet 保细节 + SAM 监督注意力 + CSFF 跨阶段特征融合 |
| **适用任务** | 图像去模糊、去雨、去噪 |
| **参数量** | ~20M（去模糊版本） |
| **关键指标** | GoPro 去模糊 PSNR 32.66 dB, Rain100L 去雨 PSNR ~40.68 dB, SIDD 去噪 PSNR 39.71 dB |
| **数据集** | GoPro, HIDE, RealBlur, Rain100L/H, Test1200/2800, SIDD, DND, Set12, BSD68 |

## 二、痛点分析

| 痛点 | 深层原因 | 现有方法的局限 | MPRNet 的解决方案 |
|------|---------|---------------|------------------|
| 单阶段设计难以兼顾细节与上下文 | 图像恢复需要同时保留空间细节和高级上下文，单阶段网络难以平衡 | Encoder-Decoder 下采样损失细节；单尺度流水线感受野有限 | 多阶段架构：前两阶段学上下文，最后阶段原分辨率保细节 |
| 朴素级联效果差 | 直接将上一阶段输出送入下一阶段，误差累积、信息丢失 | 简单串联缺乏特征传播和监督机制 | SAM（监督注意力）+ CSFF（跨阶段特征融合）实现阶段间信息交换 |
| 缺乏逐阶段监督 | 渐进恢复需要每阶段都有明确目标 | 单一损失只在最终输出上监督 | 每阶段输出都参与损失计算（Charbonnier + Edge） |
| 编码器-解码器与原分辨率分支割裂 | U-Net 类反复下采样牺牲空间精度 | 只能在"空间准确"或"上下文可靠"中二选一 | ORSNet 在原始分辨率上处理，通过 CSFF 接收前阶段的上下文特征 |

## 三、核心方法

### 3.1 整体架构

![MPRNet 架构图](assets/arch-MPRNet.jpg)

MPRNet 采用**三阶段渐进式架构**：

```
输入图像 I
  ├── Stage 1: 4 patches → Encoder-Decoder → SAM1 → 输出 stage1_img
  ├── Stage 2: 2 patches → Encoder-Decoder(+CSFF) → SAM2 → 输出 stage2_img
  └── Stage 3: 1 patch → ORSNet(+CSFF) → 输出 stage3_img
最终输出: I + R_S3 (残差预测)
```

**Multi-Patch Hierarchy（多分块层级）**：
- Stage 1：将输入切成 4 块（左上/右上/左下/右下），每块独立处理
- Stage 2：将输入切成 2 块（上/下），扩大感受野
- Stage 3：使用完整原图，恢复全局一致性

这种设计使每个阶段都能直接访问输入，但感受野逐阶段扩大，从局部到全局渐进恢复。

### 3.2 Encoder-Decoder 子网络（Stage 1 & 2）

- **Encoder**：三级，每级 2 个 CAB 块；下采样使用 reshape 拆通道（避免 strided conv 的信息损失）
- **Decoder**：三级，每级 2 个 CAB 块；上采样使用 bilinear + skip connection（避免转置卷积的棋盘效应）
- **Skip Connection**：额外加 CAB 做特征重标定

通道数逐级变化：`n_feat → n_feat+s → n_feat+2s`（s = scale_unetfeats）

### 3.3 CAB（Channel Attention Block）

CAB 是基本特征提取单元，结合卷积和 SE 式通道注意力：

```python
class CAB(nn.Module):
    def forward(self, x):
        res = self.body(x)        # Conv → PReLU → Conv
        res = self.CA(res)        # SE 通道注意力
        res += x                  # 残差连接
        return res
```

**通道注意力（CALayer）**：
- GAP → Conv(C→C/r) → ReLU → Conv(C/r→C) → Sigmoid
- reduction 默认为 4

### 3.4 SAM（Supervised Attention Module）

SAM 插在阶段之间，在真值监督下生成本阶段复原图，并用注意力掩码控制特征传递：

```python
class SAM(nn.Module):
    def forward(self, x, x_img):
        x1 = self.conv1(x)                    # Conv1×1(F_in)
        img = self.conv2(x) + x_img           # X̂ = I + R (残差预测)
        x2 = torch.sigmoid(self.conv3(img))   # M = σ(Conv3×3(X̂))
        x1 = x1 * x2                          # M ⊙ Conv1(F_in)
        x1 = x1 + x                           # + F_in (残差)
        return x1, img                        # (传给下阶段的特征, 本阶段复原图)
```

**公式**：
- 残差：R_S = Conv2(F_in)
- 本阶段复原图：X̂_S = I + R_S
- 注意力掩码：M_S = σ(Conv3(X̂_S))
- 传给下一阶段的特征：F_out = M_S ⊙ Conv1(F_in) + F_in

> **设计思想**：恢复好的区域（掩码值高）的特征被传递到下一阶段，恢复差的区域被抑制，避免错误信息传播。

### 3.5 CSFF（Cross-Stage Feature Fusion）

CSFF 通过 1×1 卷积将前一阶段的 Encoder/Decoder 多尺度特征传递到后一阶段：

```python
# Stage 2 Encoder 中
enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])
enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])
enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
```

**公式**：F_enc_i^(S) = CAB_i^(S)(·) + W_enc_i · F_enc_i^(S-1) + W_dec_i · F_dec_i^(S-1)

ORSNet 阶段则先把前阶段特征上采样到原分辨率再融合。

### 3.6 ORSNet（Original Resolution Subnetwork）

Stage 3，**无任何下采样**，由 3 个 ORB（Original Resolution Block）串联：

```python
class ORB(nn.Module):
    def forward(self, x):
        res = self.body(x)    # num_cab 个 CAB + Conv
        res += x              # 残差
        return res
```

ORSNet 内部通过 CSFF 接收 Stage 2 的多尺度特征（上采样到原分辨率），在保持空间精度的同时利用上下文信息。

## 三.5 数学推导过程详解

### 损失函数

每个阶段都有输出并参与监督，总损失为三阶段之和：

$$\mathcal{L}=\sum_{S=1}^{3}\left[\mathcal{L}_{char}(\mathbf{X}_{S},\mathbf{Y})+\lambda\,\mathcal{L}_{edge}(\mathbf{X}_{S},\mathbf{Y})\right]$$

**Charbonnier 损失**（L1 的平滑近似，ε=10⁻³）：

$$\mathcal{L}_{char}=\sqrt{\|\mathbf{X}_{S}-\mathbf{Y}\|^{2}+\varepsilon^{2}}$$

**边缘损失**（Δ 为 Laplacian 算子）：

$$\mathcal{L}_{edge}=\sqrt{\|\Delta(\mathbf{X}_{S})-\Delta(\mathbf{Y})\|^{2}+\varepsilon^{2}}$$

- λ = 0.05（去模糊、去雨使用）
- 去噪任务仅使用 Charbonnier 损失（噪声不引起剧烈边缘差异）

### 残差预测

每个阶段不直接预测复原图，而是预测残差：

$$\mathbf{X}_S = \mathbf{I} + \mathbf{R}_S$$

其中 I 为退化输入，R_S 为第 S 阶段预测的残差。这种设计使网络只需学习退化模式，降低学习难度。

## 为什么这样做

| 设计选择 | 原因 | 不这样做的后果 |
|---------|------|---------------|
| 多阶段而非单阶段 | 将困难恢复分解为子任务，逐步精炼 | 单阶段难以同时兼顾细节和上下文 |
| 前两阶段用 Encoder-Decoder | 需要大感受野学习上下文 | 单尺度流水线感受野有限 |
| 最后阶段用 ORSNet（无下采样） | 保留空间细节和精细纹理 | 反复下采样会丢失高频信息 |
| 多分块层级 | 逐阶段扩大感受野，从局部到全局 | 全图处理在早期阶段计算量大 |
| SAM 监督注意力 | 控制阶段间信息流，抑制错误特征 | 朴素级联导致误差累积 |
| CSFF 跨阶段融合 | 避免上下文信息在阶段间丢失 | 后阶段无法利用前阶段的中间特征 |
| 残差预测 | 网络只需学习退化模式 | 直接预测复原图学习难度更大 |
| 避免转置卷积 | 防止棋盘效应 | 转置卷积的上采样会产生伪影 |
| 逐阶段监督 | 每阶段有明确目标 | 仅最终输出监督导致中间阶段学不好 |

## 四、实验与效果

### 训练配置

| 配置项 | 去模糊 | 去雨 | 去噪 |
|--------|--------|------|------|
| GPU | 4×GPU | 4×GPU | 4×GPU |
| Batch size | 16 | 16 | 16 |
| Epochs | 3000 | 250 | 80 |
| 初始学习率 | 2e-4 | 2e-4 | 2e-4 |
| 训练 Patch | 256×256 | 256×256 | 128×128 |
| 优化器 | Adam | Adam | Adam |
| 学习率调度 | Warmup + Cosine Annealing | 同左 | 同左 |
| 损失函数 | Charbonnier + Edge (λ=0.05) | 同左 | 仅 Charbonnier |

### 去模糊结果（GoPro）

| 方法 | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| DeblurGAN-v2 | 29.55 | 0.934 |
| SRN-Deblur | 30.26 | 0.934 |
| DMPHN | 31.20 | 0.940 |
| HINet | 32.71 | 0.959 |
| **MPRNet** | **32.66** | **0.959** |

### 去雨结果

| 数据集 | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|
| Rain100L | ~40.68 | ~0.977 |
| Rain100H | ~30.41 | ~0.890 |
| Test1200 | ~32.91 | ~0.916 |
| Test2800 | ~33.16 | ~0.926 |

### 去噪结果

| 数据集 | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|
| SIDD | 39.71 | 0.958 |
| DND | ~39.94 | - |

### 消融实验（GoPro 去模糊）

| 配置 | 结论 |
|------|------|
| 1→2→3 阶段 | PSNR 递增，3 阶段最佳 |
| +ORSNet | 显著提升空间细节 |
| +SAM | PSNR 提升约 0.2~0.3 dB，验证监督注意力有效 |
| +CSFF | 进一步小幅提升，稳定多阶段优化 |
| 逐阶段监督 | 对渐进恢复至关重要 |

## 五、对比总结

| 维度 | MPRNet | HINet | Restormer | Uformer |
|------|--------|-------|-----------|---------|
| 架构类型 | 多阶段 Encoder-Decoder + ORSNet | 多阶段 + 半实例归一化 | Transformer（通道注意力） | U-Net Transformer |
| 参数量 | ~20M | ~17M | ~26M | ~21M |
| 去模糊 GoPro | 32.66 | 32.71 | 32.92 | 31.04 |
| 去噪 SIDD | 39.71 | 39.99 | 40.02 | 39.77 |
| 核心优势 | 多阶段渐进、细节+上下文兼顾 | 半实例归一化 | 高效通道注意力 | 窗口注意力 |
| 计算效率 | 中等 | 较高 | 较高 | 中等 |

> MPRNet 是多阶段恢复的标杆方法，启发了 NTIRE 2021 多项冠军方案。虽然后续 HINet、Restormer 等在部分指标上略有超越，但 MPRNet 的多阶段设计思想影响深远。

## 六、深度分析：为什么这样设计？每一步代表什么？

### 6.1 研究背景：图像恢复的共同挑战

MPRNet 的独特之处在于它**不是为单一任务设计，而是为图像恢复的通用问题设计**。去模糊、去雨、去噪看似不同，但共享一个核心矛盾：

> **空间精度 vs. 上下文信息之间的根本性矛盾**

- **空间精度**要求网络处理原始分辨率，不下采样，保留每一个像素的信息
- **上下文信息**要求网络看到"全貌"——哪些区域退化严重、退化模式是什么——这需要大感受野

传统单阶段网络被迫在两者之间做取舍：
- 纯 CNN（如 SRCNN）：感受野太小，看不到全局退化模式
- U-Net：下采样扩大感受野，但丢失空间精度
- Transformer：全局感受野，但 O(L²) 计算太贵

MPRNet 的天才之处在于**用"时间（多阶段）换空间"**——把一个复杂的恢复任务拆成三个子任务，每个阶段专注于不同的目标。

### 6.2 逐模块动机深度解析

#### 6.2.1 三阶段渐进：为什么是 3 个阶段？不是 2 个或 5 个？

| 阶段 | 目标 | 输入分块 | 网络结构 | 处理分辨率 |
|------|------|---------|---------|-----------|
| **Stage 1** | 粗略恢复 + 上下文学习 | 4 块 (H/2 × W/2) | Encoder-Decoder | 原始分辨率的 1/4 |
| **Stage 2** | 中等恢复 + 更大上下文 | 2 块 (H × W/2) | Encoder-Decoder + CSFF | 原始分辨率的 1/2 |
| **Stage 3** | 精细恢复 + 细节保留 | 1 块 (H × W) | ORSNet (无下采样) | 原始分辨率 |

**为什么 3 阶段是最优？**

消融实验显示：2 阶段 PSNR 低于 3 阶段，而增加到 4~5 阶段时性能饱和甚至下降（过拟合 + 计算浪费）。

**深层原因**：图像恢复的"信息量"决定了阶段数。Stage 1 学习"退化模式是什么"（低分辨率，信息量大，模式简单）；Stage 2 在更大上下文中精炼恢复结果；Stage 3 在原始分辨率上做最后的细节修正。三个阶段正好覆盖了"模式识别 → 上下文精炼 → 细节恢复"的完整链条。

**为什么 Stage 1 用 4 分块？**

4 分块将每个块的分辨率降到原图的 1/4，使 Stage 1 的 Encoder-Decoder 在**较低计算量下就能获得与原图同等的感受野覆盖**。这像是在用"放大镜逐块扫描"而非"站在远处看全景"——对于学习退化模式来说，局部扫描 + 上下文聚合是更高效的方式。

**为什么 Stage 3 不下采样？**

Stage 3 的目标是**像素级精细修复**——如果再做下采样，之前所有阶段精心保留的空间细节就会丢失。ORSNet（Original Resolution Subnetwork）在原始分辨率上工作，确保最终输出的每一根发丝、每一条雨纹都被精确处理。

> **类比**：Stage 1 像先用 4K 缩略图看全景，Stage 2 像 2K 预览图看细节，Stage 3 像 4K 原图做最终修饰。

#### 6.2.2 SAM（监督注意力）：为什么需要"真值"参与？推理时怎么办？

SAM 的核心公式：
```
F_out = M_S ⊙ Conv1(F_in) + F_in
其中 M_S = σ(Conv3(X̂_S)), X̂_S = I + R_S, R_S = Conv2(F_in)
```

**训练时**：X̂_S 可以和真值 Y 计算 loss，所以 M_S 学到了"本阶段恢复得好不好"的判别能力。

**推理时**：没有真值 Y，但 SAM 的 M_S 仍然可以工作——因为 M_S 是基于当前阶段的恢复结果 X̂_S 计算的，而 X̂_S 在推理时也有。网络在训练时已经学会了"什么样的恢复结果意味着这个区域已经修好了"。

**但这带来一个问题**：SAM 在推理时的行为可能与训练时不完全一致（training-inference discrepancy）。论文没有深入讨论这一点，但这是 MPRNet 的一个已知弱点。

**SAM 的注意力掩码 M_S 到底起什么作用？**

- **恢复好的区域**：M_S 值高 → Conv1(F_in) 被传递 → 告诉下一阶段"这里已经修好了，保持不变"
- **恢复差的区域**：M_S 值低 → Conv1(F_in) 被抑制 → 告诉下一阶段"这里还没修好，需要重新处理"

这是一种**基于恢复质量的自适应门控**，比简单的特征拼接更聪明。

#### 6.2.3 CSFF（跨阶段特征融合）：为什么不直接把前阶段的输出拼到后阶段？

如果只是简单拼接，后阶段只能看到前阶段的**最终输出**，而丢失了前阶段中间层学到的多尺度特征。

CSFF 将前阶段的**每个 Encoder 和 Decoder 层**的特征都传递给后阶段：

```
F_enc_i^(S) = CAB_i^(S)(·) + W_enc_i · F_enc_i^(S-1) + W_dec_i · F_dec_i^(S-1)
```

**为什么同时传 Encoder 和 Decoder 的特征？**

- **Encoder 特征**：包含多尺度上下文信息（高层语义、退化模式）
- **Decoder 特征**：包含恢复后的空间细节（边缘、纹理）

两者互补：Decoder 特征告诉后阶段"这里恢复到了什么程度"，Encoder 特征告诉后阶段"全局退化模式是什么"。1×1 卷积 W 做通道维度对齐。

**为什么不直接传 feature map 而是要 1×1 卷积？**

因为不同阶段的特征维度可能不同（Stage 1 的 n_feat 和 Stage 2 的 n_feat+s 不一样），1×1 卷积做线性投影对齐维度，同时提供可学习的特征变换。

#### 6.2.4 CAB（通道注意力块）：为什么用 SE 式注意力而不是更复杂的注意力？

CAB = Conv → PReLU → Conv + SE(通道注意力) + 残差

**为什么不用空间注意力？**

MPRNet 的设计哲学是：**Encoder-Decoder 负责空间感受野的扩大**（通过下采样），CAB 只需要在通道维度上做特征调制。这种分工避免了在每个 CAB 中都做昂贵的空间注意力计算。

SE 注意力的计算代价极低（GAP + 两个 FC 层），但能有效让网络学会"在当前上下文中，哪些特征通道更重要"。对于去雨来说，可能"检测雨纹的通道"在雨区重要性高，"检测背景颜色的通道"在无雨区重要性高——SE 注意力可以自适应调整。

#### 6.2.5 多分块层级的深层含义

**为什么不是统一分辨率处理？**

统一分辨率处理意味着 Stage 1 就要在原图分辨率上操作——这对计算量的要求极高，而且 Stage 1 的目标是学习退化模式，不需要像素级精度。

**4→2→1 的分块策略**本质上是一个**从粗到细的分辨率金字塔**：
- Stage 1 (4 块)：每块 H/2×W/2，总共处理的信息量与 H×W 相同，但每个块的感受野覆盖了整块
- Stage 2 (2 块)：每块 H×W/2，感受野更大
- Stage 3 (1 块)：全图，感受野最大

**为什么每块之间没有信息交互？**

论文中的 4 块/2 块是**独立处理**的（no communication between patches）。这意味着 Stage 1 的 4 个块各自独立学习退化模式，然后在 Stage 2 的 CSFF 中通过特征融合"汇总"前阶段的信息。

这个设计的优点是简单、并行度高（4 块可以在 GPU 上并行处理）。缺点是 Stage 1 无法学习跨块的全局信息——但 CSFF 在 Stage 2 弥补了这一点。

### 6.3 损失函数的深层解读

**为什么用 Charbonnier 损失而不是 L1？**

Charbonnier 损失 `L(x,y) = √((x-y)² + ε²)` 是 L1 的平滑近似：
- 在误差大时（|x-y| >> ε）：近似 L1，鲁棒性强（不像 L2 对大误差过度惩罚）
- 在误差小时（|x-y| << ε）：近似 L2，梯度更平滑（不像 L1 在 0 点梯度不连续）

**为什么去雨/去模糊用边缘损失，去噪不用？**

- 去雨/去模糊的退化主要影响**边缘和结构**（雨纹是高频线条，模糊是边缘扩散），边缘损失直接约束这些关键结构
- 去噪的退化是**随机噪声**，不影响边缘的统计特性（噪声在空间上是均匀分布的），边缘损失没有额外收益

**λ = 0.05 为什么这么小？**

边缘损失是辅助损失，权重不能太大。如果 λ 过大（比如 0.5），网络会过度关注边缘匹配而忽略像素级准确性，导致整体颜色偏移或纹理失真。

### 6.4 消融实验的深层解读

1. **去掉 SAM → PSNR 下降 0.2~0.3 dB**：看似不大，但 SAM 的真正价值不在于 PSNR 提升，而在于**防止误差在阶段间传播**。没有 SAM，Stage 1 的错误恢复会被 Stage 2 继承和放大，导致视觉质量显著下降（即使 PSNR 降幅不大）。

2. **去掉 CSFF → PSNR 小幅下降**：CSFF 提供的是**多尺度上下文传递**。没有它，后阶段只能看到前阶段的最终输出，丢失了中间层的有用信息。

3. **去掉逐阶段监督 → 性能显著下降**：这说明"渐进式恢复"需要每一步都有明确目标。没有中间监督，Stage 1 可能什么都没学好，Stage 2 和 Stage 3 就在"错误的基础上修修补补"。

---

## 七、如果以 MPRNet 作为 Baseline：改善点与可添加模块

### 7.1 现有架构的明确改善点

| 改善点 | 现状 | 问题 | 改善方向 |
|-------|------|------|---------|
| **三阶段串行推理慢** | Stage 1→2→3 串行执行 | 推理延迟是单阶段的约 3 倍 | 用**知识蒸馏**将 3 阶段蒸馏为单阶段，保持精度但加速推理 |
| **SAM 的训练-推理差异** | 训练时用真值监督，推理时无真值 | 推理时掩码质量可能下降 | 设计**自监督的 SAM**（用恢复结果自身生成掩码，而非依赖真值） |
| **分块无交互** | 4 块/2 块独立处理 | 块边界可能出现不一致 | 加入**跨块注意力**或**边界平滑层** |
| **通道注意力太简单** | SE 式通道注意力，无空间感知 | 无法区分同一通道中不同位置的退化差异 | 用**CBAM**或**ECA**替代 SE，加入轻量空间维度 |
| **无频率域建模** | 纯空间域处理 | 去模糊的高频丢失、去雨的方向性雨纹无法在频域约束 | 加入**频域损失**或**DCT 分支** |
| **固定架构** | 三阶段结构固定 | 不同任务可能不需要完全相同的阶段数 | 设计**动态阶段数**的网络（简单任务用 2 阶段，复杂任务用 3 阶段） |
| **CAB 参数效率低** | 每 CAB 都有独立的卷积和注意力参数 | 3 阶段 × 多层 CAB 的参数量约 20M | 用**权重共享**或**深度可分离卷积**减少参数 |

### 7.2 可添加的推荐模块

**模块 1：自适应阶段选择器 (Adaptive Stage Selector)**
- 在输入前加一个轻量级退化分类器（估计退化严重程度）
- 轻度退化：跳过 Stage 1，直接从 Stage 2 开始 → 推理速度 +33%
- 重度退化：保留完整 3 阶段
- 预期提升：平均推理速度 +20~40%，PSNR 损失 <0.1 dB

**模块 2：频域正则化分支 (Frequency Domain Regularizer)**
- 对每个阶段的输出做 FFT，约束恢复图像与真值在频域的差异
- 对去模糊特别有效（运动模糊在频域有明确的方向性频谱衰减）
- 预期提升：PSNR +0.2~0.4 dB，视觉清晰度显著提升

**模块 3：跨块特征交互 (Cross-Patch Interaction)**
- 在 Stage 1 的 4 个 patch 之间加轻量级信息交换
- 用 Shifted Window Attention（类似 Swin Transformer）在 patch 边界做特征交互
- 预期提升：消除分块边界伪影，PSNR +0.1~0.2 dB

**模块 4：增强型 SAM (Enhanced SAM with Self-Supervision)**
- 用当前阶段的恢复结果 X̂_S 和原始输入 I 的差异来生成掩码（不需要真值）
- 加入一个辅助损失约束掩码的"置信度"（高置信度区域应与低残差区域对齐）
- 预期提升：缩小训练-推理差距，真实场景泛化 +0.2~0.3 dB

**模块 5：知识蒸馏的单阶段加速 (Distilled Single-Stage)**
- 用训练好的 3 阶段 MPRNet 作为教师
- 训练一个单阶段学生网络（如轻量 U-Net + 注意力）
- 学生网络的中间层特征对齐教师的 Stage 2/3 特征
- 预期提升：推理速度 2~3×，PSNR 损失 <0.5 dB

### 7.3 实验验证建议

1. **第一步（最容易）**：在损失函数中加入 FFT 频域损失，在 GoPro 和 Rain100L 上验证
2. **第二步（性价比最高）**：实现自适应阶段选择，在混合退化数据集上测量延迟-精度权衡
3. **第三步（创新性最强）**：设计跨块注意力交互机制，消除分块边界伪影
4. **第四步（应用导向）**：知识蒸馏到单阶段模型，测量移动端部署的推理速度

---

## 八、不足与局限

1. **推理时延**：三阶段串行 + 多分块处理，推理时延高于单阶段轻量模型
2. **SAM 的训练-推理差异**：训练时利用真值进行"原位监督"，推理时用简化版掩码
3. **大分辨率处理**：对超大分辨率图像需分块处理，可能引入边界伪影
4. **去噪任务适配**：去噪仅用 Charbonnier 损失（未用边缘损失），未针对噪声特性优化
5. **无独立 Limitations 章节**：论文未明确讨论方法局限

## 九、一句话总结

MPRNet 通过三阶段渐进式架构（Encoder-Decoder 学上下文 + ORSNet 保细节 + SAM 监督注意力 + CSFF 跨阶段融合），在去模糊、去雨、去噪三大任务上同时达到 SOTA，是多阶段图像恢复的里程碑工作。

---

## 十、生活化例子：小明开了一间"旧影修复工作室"

> **场景一：老奶奶的传家宝老照片**

小明的工作室刚开业，一位白发苍苍的老奶奶颤巍巍地捧来一张泛黄的老照片——那是她年轻时和丈夫的合影，但照片已经**模糊、褪色、布满划痕**，几乎看不清人脸。

小明没有急着一步到位修复，而是想起了 MPRNet 的"三阶段渐进"思想：
- **第一阶段（粗修）**：先把照片分成四小块，像拼图一样分别处理，恢复大致轮廓——"嗯，这是两个人的轮廓，左边是女士，右边是男士。"
- **第二阶段（细修）**：把照片分成上下两半，进一步恢复五官和衣服纹理，同时把第一阶段的修复经验传过来——"原来这就是奶奶年轻时的样子！"
- **第三阶段（精修）**：用原始分辨率精细打磨，每一根发丝、每一条皱纹都尽力还原——"太像了！奶奶您看，您丈夫当年的领带花纹都出来了！"

老奶奶看着修复好的照片，眼眶湿润了。小明也明白了：**复杂的修复不能一步到位，分阶段、由粗到细，每一步都检查、都修正，才能还原最真实的记忆。**

---

> 小明的工作室从此名声传开，来找他的客户越来越多，每一张照片背后都有一个故事，而每个故事都教会他一种"修复智慧"……

## 附录

### 关键组件伪代码

```python
class MPRNet(nn.Module):
    def forward(self, x):
        # Multi-Patch Hierarchy
        x1 = self.split_4(x)     # 4 patches
        x2 = self.split_2(x)     # 2 patches
        x3 = x                   # full image

        # Stage 1: Encoder-Decoder
        feat1 = self.shallow_feat1(x1)
        enc1 = self.encoder1(feat1)
        dec1 = self.decoder1(enc1)
        sam1_feat, stage1_img = self.sam1(dec1, x1_img)

        # Stage 2: Encoder-Decoder + CSFF
        feat2 = self.shallow_feat2(x2)
        feat2 = torch.cat([feat2, sam1_feat], dim=1)
        enc2 = self.encoder2(feat2, encoder_outs=enc1_outs, decoder_outs=dec1_outs)
        dec2 = self.decoder2(enc2)
        sam2_feat, stage2_img = self.sam2(dec2, x2_img)

        # Stage 3: ORSNet + CSFF
        feat3 = self.shallow_feat3(x3)
        feat3 = torch.cat([feat3, sam2_feat], dim=1)
        ors_out = self.orsnet(feat3, encoder_outs=enc2_outs, decoder_outs=dec2_outs)
        stage3_img = self.tail(ors_out)

        return [stage3_img + x3_img, stage2_img, stage1_img]

# 损失函数
class CharbonnierLoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt((pred - target)**2 + 1e-6)

class EdgeLoss(nn.Module):
    def __init__(self):
        self.laplacian = torch.tensor([[.05, .25, .4, .25, .05]]).T @ \
                         torch.tensor([[.05, .25, .4, .25, .05]])
    def forward(self, pred, target):
        return torch.sqrt(
            (F.conv2d(pred, self.laplacian) - F.conv2d(target, self.laplacian))**2 + 1e-6
        )
```
