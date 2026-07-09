# Uformer: A General U-Shaped Transformer for Image Restoration 精读笔记

> **论文标题**: Uformer: A General U-Shaped Transformer for Image Restoration
> **发表会议**: CVPR 2022
> **作者**: Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, Houqiang Li
> **作者单位**: 中国科学院大学 & 澳门大学
> **代码地址**: [https://github.com/ZhendongWang6/Uformer](https://github.com/ZhendongWang6/Uformer)
> **核心标签**: `图像恢复` `U-shaped Transformer` `窗口注意力` `多尺度空间偏差` `去雨` `去噪`

---

## 一、基本信息

| 属性 | 内容 |
|------|------|
| **简称** | Uformer |
| **发表平台** | CVPR 2022 |
| **年份** | 2022 |
| **核心创新** | LeWin Transformer Block (局部增强窗口注意力) + 多尺度空间偏差调制器 |
| **应用任务** | 去噪、去模糊、去雨、超分辨率 |
| **关键指标** | Rain100L PSNR 40.58dB / SSIM 0.9864 |

---

## 二、痛点分析

| 痛点编号 | 痛点描述 | 深层原因 | 现有方案不足 |
|----------|----------|----------|-------------|
| **P1** | 全局自注意力计算开销过大 | 标准注意力 $O(N^2)$，N = H×W | 无法直接应用于高分辨率图像 |
| **P2** | 纯CNN架构缺乏长距离依赖建模 | 卷积感受野有限 | 多层堆叠效率低 |
| **P3** | U-Net跳接方案对Transformer有效性的影响未知 | 编码器-解码器信息传递方式多样 | 缺乏系统性的跳接方案比较 |

---

## 三、核心方法

### 3.1 总体架构

基于U-shaped编码器-解码器的Transformer网络，4个分辨率级别。

### 3.2 LeWin Transformer Block

**LeWin = Locally-enhanced Window Transformer**

1. 将特征图划分为不重叠的局部窗口
2. 在每个窗口内计算多头自注意力（复杂度仅与窗口大小相关）
3. 在FFN中使用Depth-wise Conv进一步捕获局部上下文

```
输入 → LayerNorm → 窗口划分 → 窗口多头自注意力 → 窗口合并 →
      → LayerNorm → Depth-wise Conv FFN → 输出
```

### 3.3 多尺度恢复调制器

- **空间偏差 (Spatial Bias)**: 以多尺度形式调整解码器各层特征
- 引入少量额外参数即可有效恢复细节
- 补偿编码器-解码器之间的信息损失

### 3.4 跳接方案比较

Uformer系统比较了三种跳接方案：
1. **Addition**: 编码器与解码器特征相加（推荐）
2. **Concatenation**: 拼接后降维
3. **No Skip**: 无跳接（效果最差）

---

## 四、实验结果

| 数据集 | 指标 | MPRNet | Restormer | Uformer |
|--------|------|--------|-----------|---------|
| **Rain100L** | PSNR/SSIM | 38.67/0.9831 | **40.77/0.9858** | 40.58/0.9864 |
| **Rain100H** | PSNR/SSIM | 30.79/0.9252 | **32.53/0.9428** | 32.49/0.9426 |
| **Rain800** | PSNR/SSIM | 29.48/0.9154 | 31.01/0.9245 | **31.18/0.9256** |

---

## 五、总结与启发

1. **窗口注意力**是高效Transformer在图像恢复中的关键设计范式
2. **LeWin Block** 在窗口注意力的基础上用 Depth-wise Conv 补充局部信息
3. **空间偏差调制器** 是轻量但有效的细节恢复机制
4. 与 Restormer 是 2022 年图像恢复领域两篇标志性 Transformer 工作
