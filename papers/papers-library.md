# 📚 深度学习论文库

> 创建时间: 2026-07-01
> 分类体系: 注意力机制 / 通道增强 / 骨干网络 / 目标检测 / 语义分割 / 生成模型 / 多模态 / 模型压缩 / 优化器 / 其他

<!--
## 🔍 部分未确认的论文（待补充）
- DAWN (已确认: IEEE TNNLS 2025, ISSN 2162-237X, 作者/DOI待补充)
- DAWN+ (DAWN改进版, 待补充)
- ELFormer / IDT / MFDNet / FMRNet (名称可能是缩写，需精确全称)
如果你有这些论文的 arXiv ID 或完整标题，我会立即更新到库中。
-->

---

## 📊 论文统计

| 分类 | 数量 | 已读 | 待读 | 精读中 | 已复现 |
|------|------|------|------|--------|--------|
| 🔥 注意力机制 | 5 | 0 | 5 | 0 | 0 |
| 🌊 通道增强/特征增强 | 3 | 0 | 3 | 0 | 0 |
| 🏗️ 骨干网络 | 0 | 0 | 0 | 0 | 0 |
| 🎯 目标检测 | 0 | 0 | 0 | 0 | 0 |
| 🖼️ 语义分割 | 0 | 0 | 0 | 0 | 0 |
| 📝 生成模型 | 0 | 0 | 0 | 0 | 0 |
| 🔗 多模态 | 0 | 0 | 0 | 0 | 0 |
| ⚡ 模型压缩/加速 | 0 | 0 | 0 | 0 | 0 |
| 📊 优化器/训练技巧 | 0 | 0 | 0 | 0 | 0 |
| 🔄 其他 | 3 | 0 | 3 | 0 | 0 |
| **总计** | **11** | **0** | **11** | **0** | **0** |

---

## 🌊 通道增强/特征增强

### Multi-Stage Progressive Image Restoration (MPRNet)
- **分类**: 🌊 通道增强/特征增强
- **标签**: image-restoration, multi-stage, attention, CVPR2021
- **状态**: 待读
- **一句话总结**: 提出了多阶段渐进式图像恢复网络，通过编码器-解码器学习上下文特征，结合高分辨率分支保留局部信息，在图像去模糊、去雨、去噪任务上达到SOTA。
- **核心痛点**: 图像恢复需在空间细节和上下文特征间取得平衡；单阶段恢复过于复杂；下采样导致信息丢失。
- **核心方法**: 三阶段渐进架构（编码器-解码器 + ORSNet）+ CAB通道注意力 + CSFF交叉阶段特征融合 + SAM监督注意力模块。
- **适用场景**: 图像去模糊、去雨、去噪；复杂退化场景；多阶段网络设计参考。
- **arXiv/DOI**: [arXiv:2102.02808](https://arxiv.org/abs/2102.02808)
- **代码链接**: [GitHub - swz30/MPRNet](https://github.com/swz30/MPRNet)
- **笔记链接**: [MPRNet 精读笔记](reading-notes/mprnet.md)
- **架构图**: reading-notes/assets/arch-MPRNet.jpg
- **添加时间**: 2026-07-01

### PHENet: Physics-Guided Height-Enhanced Network for Fog-Resilient Change Detection
- **分类**: 🔄 其他
- **标签**: remote-sensing, change-detection, fog-resilient, physics-guided, Sciencedirect2026
- **状态**: 待读
- **一句话总结**: 基于物理引导的高程增强网络，解决遥感影像中雾霾干扰下的变化检测问题。
- **核心痛点**: 遥感影像受雾霾影响严重；传统变化检测难抗雾
- **核心方法**: Physics-guided 物理引导 + Height-enhanced 高程增强机制
- **适用场景**: 遥感影像变化检测、抗雾场景分析
- **arXiv/DOI**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231226010684)
- **代码链接**: [GitHub - sssy3/PHENet](https://github.com/sssy3/PHENet)
- **笔记链接**: [PHENet 精读笔记](reading-notes/PHENet.md)
- **架构图**: reading-notes/assets/arch-PHENet.jpg
- **添加时间**: 2026-07-01

### SSFA-Net: Spatial-Spectral Fusion Attention Network
- **分类**: 🔄 其他
- **标签**: hyperspectral, spatial-spectral-fusion, attention, classification
- **状态**: 待读
- **一句话总结**: 空间-光谱融合注意力网络，用于高光谱图像分类等遥感任务。
- **核心痛点**: 高光谱数据空间和光谱维度信息提取分离，融合效率低
- **核心方法**: 空间-光谱融合注意力机制 (SSFA)
- **适用场景**: 高光谱图像分类；遥感图像分析
- **arXiv/DOI**: 待补充
- **代码链接**: [GitHub - hafeezbabar/SSFA-Net](https://github.com/hafeezbabar/SSFA-Net)
- **笔记链接**: [SSFA-Net 精读笔记](reading-notes/SSFA-Net.md)
- **架构图**: reading-notes/assets/arch-SSFA-Net.jpg
- **添加时间**: 2026-07-01

### Comprehensive and Delicate: An Efficient Transformer for Image Restoration (CDT)
- **分类**: 🔥 注意力机制
- **标签**: image-restoration, efficient-transformer, superpixel-attention, lightweight, CVPR2023
- **状态**: 待读
- **一句话总结**: 提出"特征聚合→全局依赖建模→特征恢复"三步范式，在超像素低维空间中高效计算全局注意力再转回像素级，仅需SwinIR 6%的参数量即可达到同等性能。
- **核心痛点**: ViT在图像恢复中计算开销大；全局感知与计算效率难以兼得
- **核心方法**: CA凝聚注意力模块（超像素级全局依赖）+ DA双路自适应模块（像素级恢复）+ 三步范式
- **适用场景**: 图像去噪、JPEG去伪影、运动去模糊
- **arXiv/DOI**: CVPR 2023
- **代码链接**: [GitHub - XLearning-SCU/2023-CVPR-CODE](https://github.com/XLearning-SCU/2023-CVPR-CODE)
- **笔记链接**: [CDT 精读笔记](reading-notes/CDT.md)
- **架构图**: reading-notes/assets/arch-CDT.jpg
- **添加时间**: 2026-07-01

### Adaptive Sparse Self-Attention for Efficient Image Super-Resolution and Beyond (ASSANet)
- **分类**: 🔥 注意力机制
- **标签**: image-super-resolution, sparse-attention, adaptive-sparsity, plug-and-play, TPAMI2026
- **状态**: 待读
- **一句话总结**: 提出6种可学习的稀疏策略（Top-K硬稀疏、MaskedSoftmax掩码稀疏、GELU柔性稀疏等），让自注意力自适应选择最相关的token相似度，在超分、去噪等多任务上实现高效轻量的SOTA。
- **核心痛点**: 标准全注意力计算复杂度O(N²)，大量token间交互存在冗余
- **核心方法**: 自适应稀疏自注意力(ASSA) + 局部空间变化特征估计 + 双分支并行（全局KV压缩 + 稀疏分支）
- **适用场景**: 图像超分辨率、去噪、JPEG去伪影；可即插即用替换标准自注意力
- **arXiv/DOI**: IEEE TPAMI 2026
- **代码链接**: [GitHub - sunny2109/ASSANet](https://github.com/sunny2109/ASSANet)
- **笔记链接**: [ASSANet 精读笔记](reading-notes/ASSANet.md)
- **架构图**: reading-notes/assets/arch-ASSANet.jpg
- **添加时间**: 2026-07-01

### HINet: Half Instance Normalization Network for Image Restoration
- **分类**: 🌊 通道增强/特征增强
- **标签**: image-restoration, half-instance-normalization, multi-stage, lightweight, CVPRW2021
- **状态**: 待读
- **一句话总结**: 提出半实例归一化块(HIN Block)来提升图像恢复网络性能，基于HIN Block设计了简洁高效的多阶段网络HINet，在去噪/去模糊/去雨任务上以极低计算量实现SOTA。
- **核心痛点**: BatchNorm消除范围灵活性不适合恢复任务；InstanceNorm虽有帮助但会丢弃部分有用信息
- **核心方法**: HIN Block（半实例半批归一化）+ 多阶段编解码子网络 + 跨阶段特征融合 + 监督注意力模块
- **适用场景**: 图像去噪(SIDD)、去模糊(GoPro/REDS)、去雨
- **arXiv/DOI**: [arXiv:2105.06086](https://arxiv.org/abs/2105.06086)
- **代码链接**: [GitHub - megvii-model/HINet](https://github.com/megvii-model/HINet)
- **笔记链接**: [HINet 精读笔记](reading-notes/HINet.md)
- **架构图**: reading-notes/assets/arch-HINet.jpg
- **添加时间**: 2026-07-01

### DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention
- **分类**: 🌊 通道增强/特征增强
- **标签**: image-dehazing, detail-enhanced-convolution, content-guided-attention, lightweight, arXiv2023
- **状态**: 待读
- **一句话总结**: 通过细节增强卷积（DEConv）和内容引导注意力（CGA）组成的DEAB模块，以仅3.653M参数在去雾任务上突破41dB PSNR，DEConv通过重参数化技术推理时不增加额外开销。
- **核心痛点**: 普通卷积缺乏细节先验信息；现有注意力机制对内容感知不足
- **核心方法**: DEConv（细节增强卷积）+ CGA（内容引导注意力）+ CGA-based mixup融合 + 重参数化
- **适用场景**: 图像去雾(SOTS/OTS/HazeRD)；特征增强模块可迁移至其他2D视觉任务
- **arXiv/DOI**: [arXiv:2301.04805](https://arxiv.org/abs/2301.04805)
- **代码链接**: [GitHub - cecret3350/DEA-Net](https://github.com/cecret3350/DEA-Net)
- **笔记链接**: [DEA-Net 精读笔记](reading-notes/DEA-Net.md)
- **架构图**: reading-notes/assets/arch-DEA-Net.jpg
- **添加时间**: 2026-07-01

### DAWN+: Wavelet-Based Image Deraining with Direction-Aware Attention and Mutual Representation
- **分类**: 🔥 注意力机制
- **标签**: image-deraining, wavelet, direction-aware-attention, mutual-representation, lightweight, TNNLS2025
- **状态**: 待读
- **一句话总结**: 在小波域中结合方向感知注意力和互表示学习进行图像去雨，以1.9M超轻量参数在Test1200上达到33.14dB PSNR，推理仅0.067秒。
- **核心痛点**: 雨纹具有方向性特征但传统注意力无法有效捕捉；去雨模型难以在性能和效率间取得平衡
- **核心方法**: 小波域向量分解 + 方向感知注意力 (Direction-Aware Attention) + 互表示学习 (Mutual Representation)
- **适用场景**: 图像去雨；资源受限/实时部署场景
- **arXiv/DOI**: IEEE TNNLS 2025 (ISSN 2162-237X)
- **代码链接**: （待补充）
- **笔记链接**: [DAWN+ 精读笔记](reading-notes/dawn+.md)
- **架构图**: reading-notes/assets/arch-dawn-plus.jpg
- **添加时间**: 2026-07-01

---

## 🔥 注意力机制

### Learning A Sparse Transformer Network for Effective Image Deraining (DRSformer)
- **分类**: 🔥 注意力机制
- **标签**: image-deraining, sparse-transformer, attention, top-k, CVPR2023
- **状态**: 待读
- **一句话总结**: 提出稀疏Transformer网络，通过Top-K稀疏注意力只关注最重要的特征交互，大幅降低Transformer在图像去雨中的计算开销同时保持高性能。
- **核心痛点**: 标准Transformer在图像恢复中计算量过大；全局注意力存在大量冗余
- **核心方法**: 稀疏Top-K注意力机制 + U型编码器-解码器 + 多尺度特征融合
- **适用场景**: 图像去雨
- **arXiv/DOI**: [arXiv:2303.11950](https://arxiv.org/abs/2303.11950)
- **代码链接**: [GitHub - cschenxiang/DRSformer](https://github.com/cschenxiang/DRSformer)
- **笔记链接**: [DRSformer 精读笔记](reading-notes/drsformer.md)
- **架构图**: reading-notes/assets/arch-DRSformer.jpg
- **添加时间**: 2026-07-01

---

## 🏗️ 骨干网络

> 暂无论文

---

## 🎯 目标检测

> 暂无论文

---

## 🖼️ 语义分割

> 暂无论文

---

## 📝 生成模型

> 暂无论文

---

## 🔗 多模态

> 暂无论文

---

## ⚡ 模型压缩/加速

> 暂无论文

---

## 📊 优化器/训练技巧

> 暂无论文

---

## 🔄 其他

### Deep Generalized Unfolding Networks for Image Restoration (DGUNet)
- **分类**: 🔄 其他
- **标签**: image-restoration, deep-unfolding, model-based, interpretable, CVPR2022
- **状态**: 待读
- **一句话总结**: 将梯度估计策略融入近端梯度下降算法的展开过程，设计多尺度和空间自适应的跨阶段信息通路，在多种恢复任务上实现可解释的SOTA性能。
- **核心痛点**: 纯学习网络缺乏可解释性；传统优化展开方法泛化性不足
- **核心方法**: PGD算法深度展开 + 梯度估计策略 + 多尺度空间自适应跨阶段连接
- **适用场景**: 图像去噪、去模糊、超分辨率
- **arXiv/DOI**: [arXiv:2204.13348](https://arxiv.org/abs/2204.13348)
- **代码链接**: [GitHub - MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration](https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration)
- **笔记链接**: [DGUNet 精读笔记](reading-notes/DGUNet.md)
- **架构图**: reading-notes/assets/arch-DGUNet.jpg
- **添加时间**: 2026-07-01

---

## 📝 待读清单

- [ ] MPRNet - Multi-Stage Progressive Image Restoration (CVPR 2021)
- [ ] HINet - Half Instance Normalization Network (CVPRW 2021, NTIRE 1st)
- [ ] DRSformer - Sparse Transformer for Image Deraining (CVPR 2023)
- [ ] DGUNet - Deep Generalized Unfolding Networks (CVPR 2022)
- [ ] DEA-Net - Detail-Enhanced Convolution for Dehazing (arXiv 2023)
- [ ] CDT - Efficient Transformer for Image Restoration (CVPR 2023)
- [ ] ASSANet - Adaptive Sparse Self-Attention for Super-Resolution (TPAMI 2026)
- [ ] PHENet - Physics-Guided Height-Enhanced Change Detection (2026)
- [ ] SSFA-Net - Spatial-Spectral Fusion Attention Network
- [ ] DAWN+ - Wavelet-Based Direction-Aware Deraining (TNNLS 2025)

---

## 📊 论文对比总结

| 维度 | MPRNet | HINet | DGUNet | DRSformer | DEA-Net | CDT | ASSANet |
|------|--------|-------|--------|-----------|---------|-----|---------|
| **核心思想** | 多阶段渐进恢复 | 半实例归一化 | 优化算法深度展开 | 稀疏Top-K注意力 | 细节增强卷积 | 超像素注意力三步范式 | 自适应稀疏自注意力 |
| **网络架构** | 三阶段编解码+ORSNet | 多阶段编解码子网络 | PGD展开+跨阶段融合 | U型编解码+Transformer | U-Net+DEAB堆叠 | 聚合→全局建模→恢复 | 双分支并行+ASSA即插即用 |
| **注意力机制** | CAB + SAM | 监督注意力(SAM) | 多尺度自适应连接 | 稀疏Top-K自注意 | CGA内容引导注意力 | CA凝聚+DA双路 | 6种可学习稀疏策略 |
| **归一化方式** | InstanceNorm | HIN (半IN半BN) | 标准归一化层 | LayerNorm | 标准归一化 | LayerNorm | LayerNorm |
| **多阶段** | ✓ (3阶段) | ✓ | ✓ (跨阶段) | ✗ | ✗ (单阶段U-Net) | ✗ (单阶段) | ✗ (单阶段) |
| **主要任务** | 去模糊/去雨/去噪 | 去噪/去模糊/去雨 | 去噪/去模糊/超分 | 图像去雨 | 图像去雾 | 去噪/去模糊/去伪影 | 超分/去噪/去伪影 |
| **参数量级** | ~20M | ~88K~3.5M | ~4M~8M | ~2M | **3.653M** | **~1M (SwinIR的6%)** | 轻量级 |
| **可解释性** | 中等 | 低 | **高** (优化展开) | 中等 | 中等 (重参数化) | 中等 | 中等 |
| **年份/会议** | CVPR 2021 | CVPRW 2021 | CVPR 2022 | CVPR 2023 | arXiv 2023 | CVPR 2023 | **TPAMI 2026** |
