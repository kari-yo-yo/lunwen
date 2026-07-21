# MFFDNet 精读笔记

> **论文全称**: MFFDNet: Single Image Deraining via Dual-Channel Mixed Feature Fusion
> **期刊**: IEEE Transactions on Instrumentation and Measurement, Vol. 73, 2024
> **作者**: Wenyin Tao, Xuefeng Yan, Yongzhen Wang, Mingqiang Wei
> **机构**: 南京航空航天大学
> **代码**: https://github.com/taowenyin/MFFDNet

---

## 一、论文核心问题与动机

图像去雨面临三个关键瓶颈：
1. **CNN** 擅长提取局部特征，但受限于卷积核大小，**感受野有限**，无法捕获全局信息。
2. **Transformer** 擅长捕获全局特征，但对**局部特征提取能力不足**，冗余信息会影响局部细节恢复。
3. 现有特征融合模块**未能充分探索**如何将局部特征与全局特征有效整合用于图像去雨任务。

**核心动机**: 图像的高频部分包含雨线信息，低频部分包含去雨后的背景语义信息。分离高低频并针对性建模，同时融合 CNN 局部特征和 Transformer 全局特征，是提升去雨效果的关键。

---

## 二、整体架构（Figure 2 描述）

MFFDNet 采用**三段式顺序结构 + 跳跃连接**：

```
输入图像 I (R^{3xHxW})
    |
    v
[浅层特征提取网络] --- 3x3 卷积提取浅层特征 X_0 (R^{CxHxW})
    |
    v
[深层特征提取网络] --- N 个阶段 (Stage), 每个阶段包含:
    |                 ├── CSAB (局部特征提取)
    |                 ├── WTB  (全局特征提取)
    |                 └── MFFU (混合特征融合)
    v
[去雨网络] --- 3x3 卷积 + PixelShuffle 得到残差图像 R (R^{3xHxW})
    |
    v
输出: I' = I + R
```

**数据流详细描述**:
- 每个阶段输入特征 X_l 经 CSAB 得到局部特征 X_l^C，经 WTB 得到全局特征 X_l^T
- 两者沿通道维度拼接为 X_l (R^{2CxHxW})
- 经 MFFU 融合得到 X_l^F (R^{2CxHxW})
- X_l^F 沿通道维度拆分为局部特征 X_l^{CF} (R^{CxHxW}) 和全局特征 X_l^{TF} (R^{CxHxW})，分别送入下一阶段的 CSAB 和 WTB
- 经 N 个阶段后，通道维度用 3x3 卷积减半，再通过跳跃连接与输入特征相加
- 最终经 PixelShuffle 上采样得到残差图 R，与原图相加得到去雨结果

---

## 三、CSAB（Channel-Spatial Attention Block）详解

### 3.1 设计动机

CSAB 基于 CBAM 和 FA 模块设计，解决两个问题：
1. **高低频分离**: 雨线只存在于图像高频部分，需要分离高低频并单独建模高频
2. **局部上下文建模**: Transformer 局部信息提取能力弱，需通过 CNN 建模局部空间区域间的依赖关系

### 3.2 整体公式

```
X_l'' = SAB(CAB(Conv(X_l)))
X_{l+1} = X_l'' + X_l    (残差连接)
```

### 3.3 CAB（Channel Attention Block）— 通道注意力模块

**功能**: 分离高频和低频信息，仅对高频信息进行建模。

**操作流程**:
1. 输入特征 X_l (R^{CxHxW}) 先经过一组卷积+激活操作 Conv(X_l)
2. 对卷积结果进行**全局平均池化 (GAP)** 获取每个通道的全局信息:

   X_c^{HxW} = (1/HW) * Σ Σ x_c(i,j)

3. 通过卷积为通道信息赋予非线性特征
4. 经 **Sigmoid 激活函数** 得到通道权重 W_l^C (R^{Cx1x1})
5. 将权重与输入特征**逐元素相乘**:

   X_l'^C = W_l^C ⊙ Conv(X_l)

**关键效果**: 高频信息（雨线、边缘、细节）获得更高响应权重被增强，低频信息（语义）被抑制并保留。

### 3.4 SAB（Spatial Attention Block）— 空间注意力模块

**功能**: 对高频特征去噪，扩大感受野，重建图像细节，最终完成去雨。

**五步操作流程**:

| 步骤 | 操作 | 公式 | 作用 |
|------|------|------|------|
| 1 | 通道压缩 | X_A^{l-C} = Conv_C↓(X_l'), 输出 R^{C/4 x H x W} | 1x1 卷积降维，减少计算量 |
| 2 | 下采样 + 扩大感受野 | X_A^{l-M} = MaxPooling(Conv_{s=2}(X_A^{l-C})) | stride=2 卷积扩大感受野，MaxPooling 保留更精细特征 |
| 3 | 细节恢复 | X_A^{l-U} = Upsampling(Conv(Conv(Conv(X_A^{l-M})))) | 三层卷积 + 上采样恢复图像细节 |
| 4 | 细节叠加 | — | 将恢复的图像细节加到 X_A^{l-C} 上 |
| 5 | 空间注意力输出 | X_l'^S = X_l' ⊙ Conv_C↑(X_A^{l-U} + Conv(X_A^{l-C})) | 1x1 卷积恢复通道数 + 激活后与输入逐元素相乘 |

**SAB 的核心思想**: 先去噪（去除雨线噪声），再扩大感受野学习更多周围特征点关系，然后重建细节并叠加到输入上，最终完成去雨。

---

## 四、WTB（Window Transformer Block）详解

### 4.1 设计动机

- 标准 Transformer 自注意力复杂度为 O(n^2)，计算负担过重
- 顺序堆叠的 Transformer 块缺乏跨阶段特征聚合能力

### 4.2 基于滑动窗口的全局特征提取

采用 **Swin Transformer** 作为骨干网络：

```
X_l^T'  = W-MSA(LN(X_l^T)) + X_l^T     (窗口内自注意力)
X_l^T'' = SW-MSA(LN(X_l^T')) + X_l^T'   (滑动窗口自注意力)
X_{l+1}^T = X_l^T'' + X_l^T              (残差连接)
```

其中 LN(·) 为 Layer Normalization。

### 4.3 W-MSA（Window-Based Multi-Head Self-Attention）

- 输入特征 X_0^T (R^{CxHxW}) 被分割为 N = (H/M) x (W/M) 个**不重叠窗口**，每个窗口大小为 **M x M**
- 每个窗口展平为 X_i^T (R^{NxC})，N = M^2 为每个窗口的 token 数
- 假设有 k 个注意力头，每个头维度 d_k = C/k

**自注意力计算**:
```
A_h = (XW_h_Q)(XW_h_K)^T / sqrt(d_k)
Z_h = Σ_{p∈G} softmax(A_h) XW_h_V
Z = Concat(Z_1, ..., Z_k) W_O
```

其中 W_h_Q, W_h_K, W_h_V ∈ R^{C x d_k} 为第 h 个头的 Q/K/V 投影矩阵。

### 4.4 SW-MSA（Sliding Window MSA）

- 滑动窗口每次移动距离为 **M/2**，实现跨窗口信息交互
- 弥补 W-MSA 只计算窗口内自注意力、忽略跨窗口关系的不足

### 4.5 残差式 Transformer 模块

堆叠 B 个 Transformer 模块后，经卷积层并与输入特征做残差连接：
```
X_l^T,i = T_B^i(X_l^T,i-1),  i = 1,2,...,B
X_{l,out}^T = Conv(X_l^T,B) + X_l^T,0
```

在 Transformer 堆叠后引入**卷积层**，增强局部特征提取能力，并通过残差连接聚合不同 Transformer 的特征。

---

## 五、MFFU（Mixed Feature Fusion Unit）详解

### 5.1 设计动机

- CNN 特征: 包含更多**边缘和结构信息**，但缺少精细纹理细节
- Transformer 特征: 包含更丰富的**纹理模式和语义信息**，但局部特征不足
- 目标: 产生**具有全局特征的局部特征**和**具有局部特征的全局特征**

### 5.2 特征融合网络（Feature Fusion Network）

```
X_T' = Reshape_{T2C}(X_T)           # Transformer 特征转为 CNN 格式
X_F^i = Concat(X_T', X_C),  i=0     # 沿通道维度拼接 (R^{2CxHxW})
X_F^{i+1} = Conv_i(X_F^i) + X_F^i   # 1x1 残差卷积逐步融合, i ∈ [0, L-1]
```

- L 个串联的 1x1 残差卷积逐步融合每个特征点
- 融合特征维度为 2C（沿通道拼接）

### 5.3 特征分解网络（Feature Decomposition Network）

```
X_FT, X_FC = Split(X_F^{i+1})               # 拆分为两个 R^{CxHxW}
X_FT' = Reshape_{C2T}(X_FT)                 # CNN 格式转回 Transformer 格式
X_FT'' = MLP_T(X_FT') + X_FT'               # MLP 增强全局特征 (残差)
X_FC' = Conv_C(X_FC) + X_FC                 # 3x3 卷积增强局部特征 (残差)
```

**核心输出**: X_FT'' 是**融合了局部信息的全局特征**（用于下一阶段 WTB），X_FC' 是**融合了全局信息的局部特征**（用于下一阶段 CSAB）。

---

## 六、损失函数

采用 **L1 损失 + 感知损失 (Perceptual Loss)** 的组合:

```
L_1(I', I_GT) = (1/N) Σ ||I' - I_GT||_1

L_perceptual = Σ_j (1/C_j H_j W_j) ||φ_j(I') - φ_j(I_GT)||_2^2

L_total(I', I_GT) = L_1 + λ × L_perceptual    (公式1)
```

- I_GT: 真实清晰图像
- φ_j(·): 感知网络（VGG）第 j 层的输出
- C_j, H_j, W_j: 第 j 层特征的通道数、高度、宽度
- **λ = 0.04** (平衡超参数)

---

## 七、实验设置

| 项目 | 设置 |
|------|------|
| 网络阶段数 | N = 4 个阶段 (默认配置) |
| 每阶段 CSAB 数 | N = 10 |
| 每阶段 WTB 数 | T = 6 |
| 训练 patch 大小 | 128 x 128 |
| 优化器 | Adam |
| Batch size | 6 |
| 初始学习率 | 1e-3 |
| Epoch 数 | 200 |
| 学习率策略 | Cosine Annealing |
| 硬件 | Intel Xeon Platinum 8260 + NVIDIA RTX 4090 |
| 框架 | PyTorch 1.7 |

---

## 八、实验结果

### 8.1 合成数据集结果（Table I）

在四个合成数据集上与 8 种 SOTA 方法对比：

| 数据集 | 指标 | MFFDNet 表现 |
|--------|------|-------------|
| Rain200L (单类型雨线, 200测试图) | PSNR/SSIM | SOTA (最佳) |
| Rain200H (5种雨线, 200测试图) | PSNR/SSIM | 第二名 |
| Rain800 (100测试图) | PSNR/SSIM | SOTA (最佳) |
| Rain1400 (14种雨线, 1400测试图) | PSNR/SSIM | SOTA (最佳) |

对比方法: LPNet, HINet, MPRNet, DSMNet, ECNet (CNN类); TransWeather, Uformer (Transformer类); HCT-FFN (混合类)

论文 Fig.1 展示的 Rain200L 示例: MFFDNet 达到 **PSNR=49.04, SSIM=0.996**，远超 ECNet (46.64/0.996)、HINet (45.91/0.995) 等。

**模型效率**: 虽然参数量不是最小，但在相似参数量下推理速度满足实时要求，且去雨质量最优。

### 8.2 真实数据集结果（Table II）

在 MPID (185张) 和 SPA (146张) 真实雨图上使用**无参考指标**:

| 指标 | 说明 | MFFDNet 表现 |
|------|------|-------------|
| NIQE | 越低越好 | SOTA (最佳) |
| BRISQUE | 越低越好 | SOTA (最佳) |

### 8.3 下游任务验证（YOLOv5 目标检测）

在 RID 数据集上，经 MFFDNet 去雨后，YOLOv5 对 100 张真实雨图的检测**置信度和精度显著提升**，验证了去雨结果对下游视觉任务的增益。

---

## 九、消融实验（Table III, Fig.12）

以四阶段 Swin Transformer 为基础模型，逐步添加模块验证有效性:

| 变体 | 描述 | PSNR/SSIM (Rain200L) | 增益 |
|------|------|---------------------|------|
| V0 | 基础模型 (四阶段 Swin Transformer) | 36.90/0.970 | baseline |
| V1 | + CAB 模块 | 37.23/0.976 | +0.33/+0.006 |
| V2 | + SAB 模块 (即完整 CSAB) | 37.45/0.979 | +0.22/+0.003 |
| V3 | + W-MSA 模块 | 37.73/0.982 | +0.28/+0.003 |
| V4 | + MFFU 模块 (即完整 MFFDNet) | 38.76/0.985 | +1.03/+0.003 |
| V5 | 最终完整模型 | 39.44/0.987 | +0.68/+0.002 |
| GT | 清晰图像 | ∞/1.000 | — |

**关键消融发现**:
1. **CAB 独立有效** (+0.33 dB): 通道注意力对去除雨线有直接贡献
2. **SAB 补充 CAB** (+0.22 dB): 空间注意力专注于背景纹理重建，两者组合后 CSAB 能更好地建模高频信号
3. **W-MSA 提升显著** (+0.28 dB): 全局特征提取生成复杂图像细节
4. **MFFU 贡献最大** (+1.03 dB): 混合特征融合是提升最大的单一模块，验证了 CNN-Transformer 互补融合的有效性
5. **每个模块都有正向贡献**，V0 到 V5 持续提升，总计提升 **+2.54 dB PSNR / +0.017 SSIM**

---

## 十、框架图描述汇总

| 图号 | 内容 |
|------|------|
| Fig.1 | 不同算法在 Rain200L 上的去雨结果对比，MFFDNet (49.04/0.996) 远超对比方法 |
| Fig.2 | **整体架构图**: 浅层提取 -> 深层提取 (CSAB + WTB + MFFU x N阶段) -> 去雨网络 |
| Fig.3 | **CSAB 结构图**: (a) CAB 高低频分离并建模高频; (b) SAB 探索局部空间区域间依赖 |
| Fig.4 | 通道注意力权重可视化: 输出特征的结构信息比输入更显著（高频增强） |
| Fig.5 | **SAB 五步流程图**: 通道压缩 -> stride=2 卷积+MaxPool -> 三层卷积+上采样 -> 细节叠加 -> 1x1卷积+激活+逐元素相乘 |
| Fig.6 | **滑动窗口 Transformer 结构图**: W-MSA (窗口内) -> SW-MSA (滑动跨窗口) -> 残差连接 |
| Fig.7 | CNN vs Transformer 特征可视化: CNN 提取边缘结构，Transformer 提取全局纹理，融合后两者兼得 |
| Fig.8 | **MFFN 结构图**: 特征融合网络 (拼接+残差卷积) -> 特征分解网络 (拆分+MLP/Conv增强) |
| Fig.9 | 数据集样例展示: Rain200L/H/800/1400 (合成) + MPID/SPA (真实) |
| Fig.10 | Rain200L 定性对比: MFFDNet 去雨最干净自然，无色彩畸变 |
| Fig.11 | 真实雨图定性对比: MFFDNet 色彩真实、雨线残留最少 |
| Fig.12 | **视觉消融实验**: V0-V5 逐步提升，MFFDNet 最终结果最接近 GT |
| Fig.13 | YOLOv5 下游检测验证: 经 MFFDNet 去雨后目标检测置信度和精度显著提升 |

---

## 十一、核心贡献总结

1. **双通道混合特征融合网络 MFFDNet**: 巧妙利用 CNN 和 Transformer 各自优势，通过 CSAB 提取局部特征、WTB 提取全局特征、MFFU 融合两者，产生更具判别力的特征表示。
2. **CSAB 模块**: 通过通道注意力分离并建模高频信息（雨线所在），通过空间注意力扩大感受野重建图像细节，有效补充 Transformer 局部信息提取不足的缺陷。
3. **MFFU 模块**: 不仅简单融合，而是通过"融合-分解"策略产生**带全局信息的局部特征**和**带局部信息的全局特征**，实现特征的双向增强。

---

## 十二、个人思考与评价

**优势**:
- 双分支设计思路清晰，CSAB/WTB/MFFU 各司其职
- MFFU 的"融合后分解"设计比简单拼接/相加更精细，实现了特征的双向增强
- 消融实验充分，每个模块贡献量化清晰
- 在合成和真实数据集上均达到 SOTA

**可改进/疑问点**:
- MFFU 中 L 个 1x1 残差卷积的具体值 L 未在正文中明确给出
- 窗口大小 M 的具体数值未在正文中说明（SwinIR 默认为 8x8）
- 参数量与推理速度的具体数值未在正文中列表对比（仅文字描述）
- 感知损失使用的 VGG 网络具体层数未详述