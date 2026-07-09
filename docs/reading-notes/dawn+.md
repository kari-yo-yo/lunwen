# DAWN+: Wavelet-Based Image Deraining with Direction-Aware Attention and Mutual Representation 精读笔记

> [📄 DOI](https://doi.org/10.1109/TNNLS.2025.3587248) | 📰 IEEE TNNLS 2025 | [💻 代码](https://github.com/kuijiang94/DAWN-plus)

## 一、基本信息

| 属性 | 内容 |
|------|------|
| **论文标题** | DAWN+: Wavelet-Based Image Deraining Meets Direction-Aware Attention and Mutual Representation |
| **发表期刊** | IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2025, Vol. 36, No. 10, pp. 18244-18258 |
| **作者** | Kui Jiang, Junjun Jiang, Zheng Wang, Zihan Geng, Xianming Liu（通讯作者） |
| **单位** | 哈尔滨工业大学计算机科学与技术学院，武汉大学多媒体软件国家工程研究中心，清华大学深圳国际研究生院 |
| **核心创新** | 解耦对角系数学习 + 多阶段向量分解 + 跨频率互表示（NRN）+ Top-K 稀疏注意力 + 混合尺度前馈网络 |
| **适用任务** | 图像去雨、雨滴去除、雨雾去除、去雾、低光增强、水下增强（6 个任务） |
| **参数量** | ~1.9M（DRSformer 约 26.7M，节省 94.4%） |
| **推理时间** | ~0.067s（DRSformer 约 1.34s，节省 95%） |
| **关键指标** | 相比 DAWN 平均 PSNR +1.17 dB；相比 DRSformer PSNR +0.15 dB，参数仅 5.6% |

## 二、痛点分析

| 痛点 | 深层原因 | 现有方法的局限 | DAWN+ 的解决方案 |
|------|---------|---------------|------------------|
| 雨条纹方向性被忽视 | 雨条纹具有特定方向性（水平/垂直），现有方法忽略这一关键特征 | 导致异构退化，尤其在雨纹方向对齐的纹理区域 | 小波分解天然分离方向（HL/LH/HH），方向感知注意力精炼雨水信息 |
| 小波域频率混叠 | 低频和对角分量合并处理，对角系数的方向特性被淹没 | 不同频率系数因雨纹方向特性产生异构退化，频率间冲突 | 解耦对角系数学习：专用投影参数表征对角分量 |
| 单阶段分解误差累积 | 向量分解和参数拟合在一次完成，误差不可控 | 单阶段拟合精度有限，残余误差影响恢复质量 | 多阶段向量分解和参数拟合，逐阶段减少误差 |
| 跨频率信息交互不足 | 小波子带间存在相关性，但现有方法独立处理各子带 | 缺乏频率间信息共享，限制恢复性能 | 跨频率互表示（NRN）：基于坐标的隐式神经表示促进子带交互 |
| Transformer 计算冗余 | 标准自注意力对所有 token 密集计算 | 去雨任务中大量注意力值冗余，计算开销大 | TKSA：4 路 Top-K 稀疏注意力自适应融合 |

## 三、核心方法

### 3.1 整体架构

![DAWN+ 架构图](assets/arch-DAWN-plus(1).jpg)

```
输入雨图 I_rain
  → DWT (Haar 小波分解) → [LL, HL, LH, HH] 四个子带
  → 初始嵌入层（3 个并行卷积）
  → 多阶段方向感知注意力模块
    ├── DAB: 主特征提取(CAB) + 垂直/水平方向注意力(CoordAtt)
    ├── TKSA: Top-K 稀疏注意力（4 路融合）
    ├── MSFN: 混合尺度前馈网络（3×3 + 5×5）
    └── NRN: 神经表示网络（跨频率互表示）
  → 小波重建模块
  → IWT (Haar 小波逆变换) → 去雨图像
```

### 3.2 Haar 小波变换

DAWN+ 使用 **Haar 小波**将图像分解为四个子带，不可训练：

```python
class DWT(nn.Module):
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2    # 偶数行
        x02 = x[:, :, 1::2, :] / 2    # 奇数行
        x1 = x01[:, :, :, 0::2]       # 偶数行偶数列
        x2 = x02[:, :, :, 0::2]       # 奇数行偶数列
        x3 = x01[:, :, :, 1::2]       # 偶数行奇数列
        x4 = x02[:, :, :, 1::2]       # 奇数行奇数列

        x_LL = x1 + x2 + x3 + x4     # 低频（背景）
        x_HL = -x1 - x2 + x3 + x4    # 水平方向
        x_LH = -x1 + x2 - x3 + x4    # 垂直方向
        x_HH = x1 - x2 - x3 + x4     # 对角方向
        return x_LL, x_HL, x_LH, x_HH
```

**方向含义**：
- **LL**：低频近似，保留背景结构
- **HL**：水平高频，捕获水平方向边缘（对应垂直雨条纹）
- **LH**：垂直高频，捕获垂直方向边缘（对应水平雨条纹）
- **HH**：对角高频，捕获对角方向信息

### 3.3 方向感知注意力（DAB — Direction-Aware Attention Block）

DAB 是 DAWN+ 的核心模块，通过**向量分解**将雨水分量从背景中分离：

```python
class DAB(nn.Module):
    def forward(self, x):
        # x = [混合低频+对角特征, 垂直特征, 水平特征]
        main_fea = self.CAB(x[0])                    # 主特征提取
        v_ATT_fea = self.CoordAtt_V(main_fea)        # 垂直方向雨水特征
        h_ATT_fea = self.CoordAtt_H(main_fea)        # 水平方向雨水特征
        v_feas = self.CAB_dsc(x[1]) - v_ATT_fea      # 减去垂直雨水 → 干净垂直背景
        h_feas = self.CAB_dsc(x[2]) - h_ATT_fea      # 减去水平雨水 → 干净水平背景
        return main_fea, v_feas, h_feas
```

**关键思想**：从混合特征中分别提取垂直和水平方向的雨水分量，然后从原始特征中减去这些雨水分量，得到干净背景特征。

### 3.4 坐标注意力（CoordAtt）

CoordAtt 分别沿水平和垂直方向池化，学习方向特定的注意力权重：

```python
class CoordAtt(nn.Module):
    def forward(self, x):
        x_h = self.pool_h(x)        # (N,C,H,1) 沿宽度池化
        x_w = self.pool_w(x)        # (N,C,1,W) 沿高度池化
        y = torch.cat([x_h, x_w], dim=2)
        y = self.h_swish(self.bn(self.conv1(y)))
        y_h, y_w = torch.split(y, [H, W], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))   # 垂直注意力
        a_w = torch.sigmoid(self.conv_w(y_w))   # 水平注意力
        out = x * a_w * a_h
        return out
```

其中 `h_swish(x) = x · ReLU6(x+3) / 6`。

### 3.5 DAWN+ 三大核心改进（相比 DAWN）

#### 改进 1：解耦对角系数学习

DAWN 将 LL + HH 合并处理，导致对角分量的方向特性被淹没。DAWN+ 使用**专用投影参数**独立表征对角分量，消除频率混叠。

#### 改进 2：多阶段向量分解

DAWN 单阶段完成向量分解和参数拟合。DAWN+ 将其分为**多个阶段**，逐阶段减少误差累积。

#### 改进 3：跨频率互表示（NRN）

NRN（Neural Representation Network）是基于坐标的隐式神经表示，促进不同频率子带间的信息交互：

```python
class NRN(nn.Module):
    def __init__(self):
        self.imnet = MLP(612, 3, [256, 256, 256])
        # 输入维度: 特征展开(64×9=576) + 坐标(2) + 位置编码(4×8=32) + 单元(2) = 612
```

**NRN 工作流程**：
1. **特征展开**：将 3×3 邻域特征展开为 576 维
2. **位置编码**：L=8 的 sin/cos 编码，生成 32 维
3. **局部集成**：使用 4 个邻居（vx, vy ∈ {-1, 1}）的加权预测
4. **MLP 解码**：612 → [256, 256, 256] → 3

### 3.6 TKSA 与 MSFN（与 DRSformer 共享设计）

DAWN+ 引入了与 DRSformer 类似的 TKSA 和 MSFN：

**TKSA**：4 路 Top-K 稀疏注意力（C/2, 2C/3, 3C/4, 4C/5），可学习权重融合。

**MSFN**：3×3 + 5×5 双分支混合尺度前馈网络。

> 详细机制参见 DRSformer 精读笔记。

## 三.5 数学推导过程详解

### Haar 小波变换

**正向变换（DWT）**：

$$x_{LL} = x_1 + x_2 + x_3 + x_4$$
$$x_{HL} = -x_1 - x_2 + x_3 + x_4$$
$$x_{LH} = -x_1 + x_2 - x_3 + x_4$$
$$x_{HH} = x_1 - x_2 - x_3 + x_4$$

其中 $x_1 = x[0::2, 0::2]/2$, $x_2 = x[1::2, 0::2]/2$, $x_3 = x[0::2, 1::2]/2$, $x_4 = x[1::2, 1::2]/2$

**逆向变换（IWT）**为上述操作的逆过程。

### 方向感知向量分解

雨图经 Haar 分解后：

$$I_{rain} \xrightarrow{Haar} \{A_{rain}, H_{rain}, V_{rain}, D_{rain}\}$$

主特征提取（DAWN+ 解耦对角系数）：

$$f_{main} = \text{CAB}(\text{Conv}([A_{rain}; D_{rain}]))$$

方向感知雨水提取：

$$f_{v,rain} = \text{CoordAtt}_V(f_{main}), \quad f_{h,rain} = \text{CoordAtt}_H(f_{main})$$

背景恢复（减去雨水分量）：

$$f_{v,bg} = \text{Conv}(V_{rain}) - f_{v,rain}$$
$$f_{h,bg} = \text{Conv}(H_{rain}) - f_{h,rain}$$

### Top-K 稀疏注意力

$$A = \frac{Q K^\top}{\tau} \cdot \text{temperature}$$

4 路 Top-K 掩码：

$$M_i = \text{TopK}(A, k_i), \quad k_i \in \left\{\frac{C}{2},\, \frac{2C}{3},\, \frac{3C}{4},\, \frac{4C}{5}\right\}$$

$$\text{out} = \sum_{i=1}^{4} \alpha_i \cdot \text{softmax}(\text{Mask}(A, M_i)) \cdot V$$

### 跨频率互表示（NRN）

位置编码（L=8）：

$$\gamma(p) = [\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)]$$

局部集成预测：

$$\hat{I}(p) = \sum_{(v_x, v_y) \in \{-1,1\}^2} \frac{S_{v_x,v_y}}{\sum S} \cdot \text{MLP}(\text{feat}_{v_x,v_y}, \Delta p_{v_x,v_y}, \text{cell})$$

### 损失函数

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{Charb} + 0.2 \cdot \mathcal{L}_{Edge} - 0.15 \cdot \mathcal{L}_{SSIM}$$

$$\mathcal{L}_{Charb} = \sqrt{(x-y)^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

**三项损失的设计意图**：
- Charbonnier（权重 0.5）：L1 的平滑近似，鲁棒的像素级恢复
- Edge（权重 0.2）：基于 Laplacian 核，关注高频纹理结构
- SSIM（权重 -0.15）：取负号以最大化结构相似性

## 为什么这样做

| 设计选择 | 原因 | 不这样做的后果 |
|---------|------|---------------|
| 小波域而非空间域去雨 | 小波分解天然分离方向（HL/LH/HH），与雨条纹方向性匹配 | 空间域难以区分不同方向雨条纹 |
| 向量分解（减法） | 从混合特征中显式减去雨水分量，得到干净背景 | 直接预测干净图难以分离雨纹和背景 |
| 解耦对角系数（DAWN+ 新增） | 对角分量有独立方向特性，不应与低频合并 | 频率混叠导致对角信息丢失 |
| 多阶段分解（DAWN+ 新增） | 逐阶段拟合减少误差累积 | 单阶段拟合精度不足 |
| 跨频率互表示 NRN（DAWN+ 新增） | 不同子带间存在相关性，需信息交互 | 独立处理子带限制恢复性能 |
| 坐标注意力 | 分别沿 H/W 方向池化，捕捉方向特定信息 | 标准注意力无法区分方向 |
| 深度可分离卷积（CAB_dsc） | 降低计算量，保持方向特征提取能力 | 普通卷积参数量大、计算慢 |
| TKSA 4 路融合 | 不同稀疏度适应不同密度雨纹 | 单一稀疏度无法处理多变雨纹 |
| MSFN 3×3+5×5 | 多尺度雨纹需要多尺度感受野 | 单尺度遗漏尺度间信息 |
| 复合损失 | Charbonnier 保像素 + Edge 保纹理 + SSIM 保结构 | 单一损失无法全面优化 |

## 四、实验与效果

### 训练配置

| 配置项 | 值 |
|--------|-----|
| 框架 | PyTorch 1.1.0 |
| 环境 | Ubuntu 16.04, Python 3.7, CUDA 9.0, cuDNN 7.5 |
| 训练数据集 | Rain13K |
| 测试数据集 | Rain100L, Rain100H, Test100, Test1200, Test2800 |
| Batch Size | 8 |
| 训练轮数 | 1000 |
| 初始学习率 | 5e-4 |
| 最小学习率 | 1e-7 |
| 优化器 | Adam (betas=(0.9, 0.999)) |
| 学习率调度 | StepLR (step_size=80, gamma=0.8) |
| 训练 Patch | 224×224 |
| 损失函数 | 0.5×Charbonnier + 0.2×Edge - 0.15×SSIM |

### 性能对比

| 对比对象 | DAWN+ 优势 |
|---------|-----------|
| vs DAWN (ACM MM 2023) | 平均 PSNR +1.17 dB |
| vs DRSformer (CVPR 2023) | PSNR +0.15 dB，参数节省 94.4%，推理快 95% |
| vs MPRNet (CVPR 2021) | DAWN 已超 0.88 dB（Test1200），DAWN+ 进一步提升 |

### 效率对比

| 指标 | DAWN+ | DRSformer | 对比 |
|------|-------|-----------|------|
| 参数量 | ~1.9M | ~26.7M | 节省 94.4% |
| 推理时间 | ~0.067s | ~1.34s | 节省 95% |
| PSNR | +0.15 dB | 基准 | 性能更优 |

### 任务可移植性

DAWN+ 在 **6 个任务**上验证了策略的可移植性：
1. 图像去雨（Deraining）
2. 雨滴去除（Raindrop Removal）
3. 雨雾去除（Rainhaze Removal）
4. 图像去雾（Dehazing）
5. 低光增强（Low-light Enhancement）
6. 水下增强（Underwater Enhancement）

### DAWN vs DAWN+ 对比

| 方面 | DAWN (ACM MM 2023) | DAWN+ (TNNLS 2025) |
|------|--------------------|--------------------|
| 对角系数处理 | LL+HH 合并 | 解耦对角系数学习 |
| 分解阶段 | 单阶段 | 多阶段 |
| 互表示 | 无 | NRN 跨频率互表示 |
| 注意力 | CoordAtt | 新增 TKSA |
| FFN | 标准卷积 | MSFN（3×3+5×5） |
| 归一化 | 无 | LayerNorm |
| 任务范围 | 去雨+检测 | 6 个任务 |
| 性能提升 | 基准 | 平均 +1.17 dB |

## 五、对比总结

| 维度 | DAWN+ | DRSformer | DAWN | MPRNet | Restormer |
|------|-------|-----------|------|--------|-----------|
| 领域 | 小波域 | 空间域 | 小波域 | 空间域 | 空间域 |
| 注意力 | CoordAtt + TKSA | TKSA | CoordAtt | SE | MDTA |
| 参数量 | ~1.9M | ~26.7M | ~1.5M | ~20M | ~26M |
| 推理时间 | ~0.067s | ~1.34s | - | - | - |
| 去雨 SOTA | ✅ 最优 | 次优 | 第三 | 中等 | 中等 |
| 核心优势 | 超轻量+方向感知 | 稀疏注意力 | 方向感知 | 多阶段 | 通道注意力 |
| 效率 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

> DAWN+ 的核心贡献在于：以仅 1.9M 参数（DRSformer 的 5.6%）实现了更优的去雨性能，同时将方向感知注意力与小波域处理相结合，在 6 个低层视觉任务上展现了出色的可移植性。

## 六、不足与局限

1. **小波变换固定**：使用 Haar 小波（不可学习），可能不是所有场景的最优选择
2. **NRN 计算开销**：隐式神经表示的 MLP 推理有一定开销，虽然总体参数量小
3. **训练轮数多**：1000 epochs 训练时间较长
4. **复合损失调参**：三个损失项的权重（0.5, 0.2, -0.15）需要经验调参
5. **轻雨场景简化**：简单雨纹场景可能不需要全部模块，存在冗余
6. **IEEE 全文限制**：部分精确 PSNR/SSIM 数值需查阅论文原文表格

## 七、一句话总结

DAWN+ 在小波域中通过解耦对角系数学习、多阶段向量分解和跨频率互表示，结合方向感知注意力和 Top-K 稀疏注意力，以仅 1.9M 参数实现了超越 DRSformer 的去雨 SOTA 性能，是轻量高效去雨的标杆工作。

---

## 八、生活化例子：小明的"旧影修复工作室"

> **场景二：大学毕业照上的斜雨丝**

一位大学生拿着毕业照来找小明，照片里同学们笑容灿烂，但那天偏偏下起了**绵绵小雨**，照片上布满了**斜斜的雨条纹**，像一道道银线划过每个人的脸。

小明仔细观察——这些雨纹不是乱糟糟的，而是**有方向的**：风往左吹，雨丝就往右斜。这让他想起了 DAWN+ 的"方向感知注意力"：

"就像用不同方向的梳子梳头发，垂直的梳子对付水平的雨丝，水平的梳子对付垂直的雨丝。"小明把照片"分解"成不同方向的信息——水平方向的细节、垂直方向的细节、还有对角线方向的小纹理，然后**分别处理、再组合**。

更妙的是，DAWN+ 教他的"跨频率互表示"——就像调色时，知道了红色通道的信息，就能推断绿色通道该怎么调。不同方向的频率信息互相帮忙，照片很快就清爽了。

"太神奇了！雨没了，但我的表情一点没变！"大学生惊喜地说。

小明心里得意：**雨有方向，修复也要有方向。对症下药，才能药到病除。**

---

> 小明的口碑越来越好，接下来的客户会带来什么样的难题呢……

## 附录

### 关键组件伪代码

```python
class DAWN_PLUS(nn.Module):
    def forward(self, x):
        # 1. Haar 小波分解
        LL, HL, LH, HH = self.dwt(x)

        # 2. 初始嵌入
        feat_main = self.conv_main(torch.cat([LL, HH], dim=1))  # 解耦对角
        feat_v = self.conv_v(LH)
        feat_h = self.conv_h(HL)

        # 3. 多阶段方向感知注意力
        for stage in self.stages:
            feat_main, feat_v, feat_h = stage.dab(
                [feat_main, feat_v, feat_h]
            )
            # TKSA + MSFN 精炼
            feat_main = stage.stb(feat_main)
            # NRN 跨频率互表示
            feat_main = stage.nrn(feat_main, feat_v, feat_h)

        # 4. 小波重建
        restored = self.reconstruct(feat_main, feat_v, feat_h)

        # 5. Haar 逆变换
        out = self.iwt(restored)
        return out

class DAB(nn.Module):
    """方向感知注意力块"""
    def forward(self, x):
        main_fea = self.cab(x[0])                     # 主特征
        v_att = self.coordatt_v(main_fea)              # 垂直雨水特征
        h_att = self.coordatt_h(main_fea)              # 水平雨水特征
        v_clean = self.cab_dsc(x[1]) - v_att           # 干净垂直背景
        h_clean = self.cab_dsc(x[2]) - h_att           # 干净水平背景
        return main_fea, v_clean, h_clean

class NRN(nn.Module):
    """跨频率互表示网络"""
    def forward(self, feat, coord, cell):
        feat_unfold = F.unfold(feat, 3)                # 3×3 邻域展开
        coord_enc = positional_encoding(coord, L=8)    # 位置编码
        inp = torch.cat([feat_unfold, coord, coord_enc, cell], dim=-1)
        # 4 邻居局部集成
        outputs = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                pred = self.imnet(inp_neighbor)
                weight = area_weight(vx, vy, coord)
                outputs.append(weight * pred)
        return sum(outputs) / sum(weights)

# 损失函数
loss = 0.5 * CharbonnierLoss(pred, target) \
     + 0.2 * EdgeLoss(pred, target) \
     - 0.15 * SSIMLoss(pred, target)
```
