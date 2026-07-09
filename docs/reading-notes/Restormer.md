# Restormer: Efficient Transformer for High-Resolution Image Restoration 精读笔记

> **论文标题**: Restormer: Efficient Transformer for High-Resolution Image Restoration
> **发表会议**: CVPR 2022 Workshop (NTIRE)
> **作者**: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang
> **代码地址**: [https://github.com/swz30/Restormer](https://github.com/swz30/Restormer)
> **核心标签**: `图像恢复` `高效Transformer` `线性复杂度注意力` `跨通道注意力` `去雨` `去模糊`

---

## 一、基本信息

| 属性 | 内容 |
|------|------|
| **简称** | Restormer |
| **发表平台** | CVPRW 2022 (NTIRE Challenge) |
| **年份** | 2022 |
| **核心创新** | MDTA (跨通道线性注意力) + GDFN (门控前馈网络) |
| **应用任务** | 去雨、去噪、去模糊、超分辨率 |
| **关键指标** | Rain100L PSNR 40.77dB / SSIM 0.9858 |

---

## 二、痛点分析

| 痛点编号 | 痛点描述 | 深层原因 | 现有方案不足 |
|----------|----------|----------|-------------|
| **P1** | 标准Transformer无法处理高分辨率图像 | 自注意力复杂度 $O(W^2 H^2)$ 随分辨率二次增长 | 窗口注意力(Swin)限制感受野 |
| **P2** | CNN感受野有限，难以建模长距离依赖 | 卷积核尺寸固定，局部操作 | 多层堆叠导致计算量大 |
| **P3** | 通用恢复模型对特定任务效果次优 | 单一架构处理所有退化类型 | 缺乏任务适应性 |

---

## 三、核心方法

### 3.1 总体架构

多尺度编码器-解码器结构，4个分辨率级别，无需将图像分解为局部窗口。

### 3.2 MDTA (Multi-Dconv Head Transposed Attention)

核心创新：将注意力从空间维度转置到通道维度。

```
标准注意力: 在空间位置间计算相似度 → O(H²W²)
Restormer:  在通道间计算互协方差     → O(HW × C)
```

**步骤**:
1. 输入 X (C×H×W) 通过 1×1 Conv 得到 Q, K, V
2. Q, K, V 经过 Depth-wise Conv 编码局部上下文
3. **转置注意力**: 在通道维度计算 Q 和 K 的互协方差矩阵
4. 乘以转置的 V，得到全局上下文编码的输出

### 3.3 GDFN (Gated-Dconv Feed-Forward Network)

```
输入 → 1×1Conv → Depth-wise Conv → (并行) → 逐元素乘积(门控) → 1×1Conv → 输出
         ↓                                      ↑
       Gating                        Linear path
```

门控机制抑制冗余特征，仅传递有用信息。

### 3.4 渐进式训练策略

从低分辨率区块开始训练，逐步增大分辨率，类似课程学习。

---

## 四、实验结果

| 数据集 | 指标 | MPRNet | MaxIM | Restormer |
|--------|------|--------|-------|-----------|
| **Rain100L** | PSNR/SSIM | 38.67/0.9831 | - | **40.77/0.9858** |
| **Rain100H** | PSNR/SSIM | 30.79/0.9252 | - | **32.53/0.9428** |
| **Rain800** | PSNR/SSIM | 29.48/0.9154 | - | **31.01/0.9245** |

---

## 五、总结与启发

1. **跨通道注意力**是高效Transformer设计的关键洞察
2. **门控FFN**有效抑制冗余特征
3. **统一架构**可处理多种退化任务
4. 渐进式训练策略对高分辨率图像恢复任务有效
