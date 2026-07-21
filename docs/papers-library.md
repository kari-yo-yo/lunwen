# 📚 深度学习论文库

> 创建时间: 2026-07-01
> 分类体系: 注意力机制 / 通道增强/特征增强 / 去雨 / 去雾 / 红外小目标检测 / 骨干网络 / 目标检测 / 语义分割 / 生成模型 / 多模态 / 模型压缩 / 优化器 / 其他

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
| 🔥 注意力机制 | 3 | 0 | 3 | 0 | 0 |
| 🌊 通道增强/特征增强 | 2 | 0 | 2 | 0 | 0 |
| 🌧️ 去雨 | 5 | 0 | 5 | 0 | 0 |
| 🌫️ 去雾 | 2 | 0 | 2 | 0 | 0 |
| 🌧️+🌫️ 多任务恢复 | 1 | 0 | 1 | 0 | 0 |
| ⚡ 模型压缩/加速 | 1 | 0 | 1 | 0 | 0 |
| 🏗️ 骨干网络 | 0 | 0 | 0 | 0 | 0 |
| 🎯 目标检测 | 0 | 0 | 0 | 0 | 0 |
| 🖼️ 语义分割 | 0 | 0 | 0 | 0 | 0 |
| 📝 生成模型 | 0 | 0 | 0 | 0 | 0 |
| 🔗 多模态 | 0 | 0 | 0 | 0 | 0 |
| 📊 优化器/训练技巧 | 0 | 0 | 0 | 0 | 0 |
| 🔬 科学成像 | 1 | 0 | 1 | 0 | 0 |
| 🔄 其他 | 1 | 0 | 1 | 0 | 0 |
| **总计** | **16** | **0** | **16** | **0** | **0** |

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

---

## 🔥 注意力机制

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

---

## 🌧️ 去雨

### DAWN+: Wavelet-Based Image Deraining with Direction-Aware Attention and Mutual Representation
- **分类**: 🌧️ 去雨
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

### Learning A Sparse Transformer Network for Effective Image Deraining (DRSformer)
- **分类**: 🌧️ 去雨
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

### MFFDNet: Single Image Deraining via Dual-Channel Mixed Feature Fusion
- **分类**: 🌧️ 去雨
- **标签**: image-deraining, CNN-Transformer, channel-spatial-attention, mixed-feature-fusion, TIM2024
- **状态**: 待读
- **一句话总结**: CNN+Transformer双分支混合去雨网络，CSAB通过通道注意力分离高低频并建模雨线特征，WTB用窗口Transformer捕获全局依赖，MFFU互补融合局部与全局特征。
- **核心痛点**: CNN感受野有限无法捕获全局信息；Transformer局部特征提取不足；现有融合模块未充分整合局部与全局特征
- **核心方法**: CSAB(通道-空间注意力模块) + WTB(窗口Transformer块) + MFFU(混合特征融合单元) + PixelShuffle上采样
- **适用场景**: 图像去雨；CNN-Transformer混合架构参考；多尺度特征融合
- **arXiv/DOI**: IEEE TIM 2024
- **代码链接**: [GitHub - taowenyin/MFFDNet](https://github.com/taowenyin/MFFDNet)
- **笔记链接**: [MFFDNet 精读笔记](reading-notes/MFFDNet.md)
- **架构图**: reading-notes/assets/arch-MFFDNet.png
- **添加时间**: 2026-07-21

### TransMamb: A Hybrid Transformer-Mamba Network for Single Image Deraining
- **分类**: 🌧️ 去雨
- **标签**: image-deraining, transformer, mamba, SSM, spectral-attention, hybrid, TIP2024
- **状态**: 待读
- **一句话总结**: Transformer-Mamba双分支混合去雨网络，频谱分组Transformer块在频域通道维度执行自注意力捕获全局雨相关依赖，Mamba分支用双向SSM捕获局部与全局序列一致性，频谱一致性损失约束重建信号关系。
- **核心痛点**: Transformer固定窗口限制非局部感受野；纯Transformer计算量大；单一范式难以全面建模雨纹特征
- **核心方法**: SDTB(频谱分组Transformer-SBSA+SEFF) + CBSM(级联双向SSM) + 通道级联融合 + 频谱一致性损失
- **适用场景**: 单图像去雨；Transformer-SSM混合架构参考；频域去雨方法
- **arXiv/DOI**: [arXiv:2409.00410](https://arxiv.org/abs/2409.00410) (IEEE TIP)
- **代码链接**: [GitHub - sunshangquan/TransMamb](https://github.com/sunshangquan/TransMamb)
- **笔记链接**: [TransMamb 精读笔记](reading-notes/TransMamb.md)
- **架构图**: reading-notes/assets/arch-TransMamb.png
- **添加时间**: 2026-07-21

### CPRAformer: Cross Paradigm Representation and Alignment Transformer for Image Deraining
- **分类**: 🌧️ 去雨
- **标签**: image-deraining, cross-paradigm, sparse-attention, alignment, frequency-module, MM2025
- **状态**: 待读
- **一句话总结**: 跨范式表征与对齐Transformer，SPC-SA用稀疏提示过滤通道注意力增强全局依赖，SPR-SA用CNN近似空间注意力建模局部细节，AAFM两阶段渐进式对齐融合两种范式特征，在8个基准数据集达SOTA。
- **核心痛点**: 不规则雨纹和复杂几何重叠对单范式架构构成挑战；通道注意力和空间注意力特征不对齐
- **核心方法**: SPC-SA(稀疏提示通道自注意力) + SPR-SA(空间像素细化自注意力) + AAFM(自适应对齐频率模块) + MSGN(多尺度流门控网络)
- **适用场景**: 图像去雨；跨范式特征融合参考；稀疏注意力机制
- **arXiv/DOI**: [arXiv:2504.16455](https://arxiv.org/abs/2504.16455) (ACM MM 2025)
- **代码链接**: [GitHub - zs1314/CPRAformer](https://github.com/zs1314/CPRAformer)
- **笔记链接**: [CPRAformer 精读笔记](reading-notes/CPRAformer.md)
- **架构图**: reading-notes/assets/arch-CPRAformer.png
- **添加时间**: 2026-07-21

---

## 🌫️ 去雾

### DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention
- **分类**: 🌫️ 去雾
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

### SSFA-Net: Sparse Strip and Dual-Domain Spatial-Frequency Attention for Efficient Image Dehazing
- **分类**: 🌫️ 去雾
- **标签**: image-dehazing, sparse-strip-attention, spatial-frequency-attention, LSH, efficient-attention, Neurocomputing2026
- **状态**: 待读
- **一句话总结**: SSA稀疏条带注意力(O(L log L)) + DDSFA空间-频率双域注意力，3.86M参数/30ms延迟在9个去雾基准达SOTA。
- **核心痛点**: ViT二次方注意力计算开销大；空间域与频率域割裂处理；非均匀雾局部差异被忽略
- **核心方法**: LSH哈希+能量选择+动态条带滤波(SSA); Avg/Max/Var三路池化+DC/残差分解(DDSFA); 多尺度残差块(MSRBlock)
- **适用场景**: 图像去雾、低光增强、遥感图像去雾、自动驾驶视觉预处理
- **arXiv/DOI**: [10.1016/j.neucom.2026.133671](https://doi.org/10.1016/j.neucom.2026.133671)
- **代码链接**: [GitHub - hafeezbabar/SSFA-Net](https://github.com/hafeezbabar/SSFA-Net)
- **笔记链接**: [SSFA-Net 精读笔记](reading-notes/SSFA-Net.md)
- **架构图**: reading-notes/assets/arch-SSFA-Net.jpg
- **添加时间**: 2026-07-01

---

## 🌧️+🌫️ 多任务恢复

### DPPD: Learning Dynamic Prompts for All-in-One Image Restoration
- **分类**: 🌧️+🌫️ 多任务恢复
- **标签**: all-in-one, image-restoration, dynamic-prompts, degradation-prototype, TIP2025
- **状态**: 待读
- **一句话总结**: 将退化先验提取解耦为退化原型分配(DPA)和提示分布学习(PDL)两个组件，为每种退化类型学习高斯分布提示并通过重参数化采样实现动态提示生成，在all-in-one恢复任务上平均提升2.94dB。
- **核心痛点**: 现有all-in-one方法使用静态任务提示判别性不足；推理时提示退化为固定参数限制表达能力
- **核心方法**: DPA(退化原型分配-K-means聚类+高斯分布) + PDL(提示分布学习-重参数化采样) + CrossFusion(提示引导交叉注意力) + 复合退化分布建模
- **适用场景**: All-in-one图像恢复(去噪/去雨/去雾/去雪/低光等)；任务感知的提示学习参考
- **arXiv/DOI**: IEEE TIP 2025
- **代码链接**: [GitHub - Aitical/DPPD](https://github.com/Aitical/DPPD)
- **笔记链接**: [DPPD 精读笔记](reading-notes/DPPD.md)
- **架构图**: reading-notes/assets/arch-DPPD.png
- **添加时间**: 2026-07-21

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

### DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables
- **分类**: ⚡ 模型压缩/加速
- **标签**: image-denoising, lookup-table, channel-aware, ultra-efficient, lightweight, CVPR2025
- **状态**: 待读
- **一句话总结**: 通过PCM成对通道混合器捕获通道相关性和L形卷积消除旋转冗余，训练后转LUT实现仅500KB存储、能耗为DnCNN 0.1%的超高效去噪，PSNR超越所有现有LUT方法超1dB。
- **核心痛点**: DNN去噪计算和内存需求大，边缘设备部署困难；现有LUT方法忽略通道相关性或空间关系
- **核心方法**: PCM成对通道混合器(RG/GB/BR三对并行) + L形卷积(无重叠感受野) + 训练后3D/4D LUT转换 + 级联查表推理
- **适用场景**: 极端资源受限的边缘设备去噪；嵌入式/IoT设备图像处理；实时视频去噪
- **arXiv/DOI**: [arXiv:2503.15931](https://arxiv.org/abs/2503.15931) (CVPR 2025)
- **代码链接**: [GitHub - Stephen0808/DnLUT](https://github.com/Stephen0808/DnLUT)
- **笔记链接**: [DnLUT 精读笔记](reading-notes/DnLUT.md)
- **架构图**: reading-notes/assets/arch-DnLUT.png
- **添加时间**: 2026-07-21

---

## 🔬 科学成像

### SCGN: Statistical Characteristic-Guided Denoising for Rapid High-Resolution TEM Imaging
- **分类**: 🔬 科学成像
- **标签**: electron-microscopy, denoising, statistical-guidance, spatial-frequency, CVPR2026
- **状态**: 待读
- **一句话总结**: 利用统计特性在空间域(标准差偏差引导加权)和频率域(FFT频带引导加权)同时引导去噪，并提出HRTEM专用噪声标定方法，有效恢复快速成像中的原子级结构信息。
- **核心痛点**: HRTEM快速成像需短曝光导致严重噪声；现有去噪方法缺乏无序结构数据且噪声分布不匹配
- **核心方法**: 空间偏差引导加权(3x3窗口标准差) + 频率带引导加权(FFT) + HRTEM噪声标定(列噪声建模) + 空间-频率增强块级联
- **适用场景**: HRTEM原子级成像去噪；材料科学成核动力学观测；电子显微镜快速成像
- **arXiv/DOI**: [arXiv:2603.18834](https://arxiv.org/abs/2603.18834) (CVPR 2026)
- **代码链接**: [GitHub - HeasonLee/SCGN](https://github.com/HeasonLee/SCGN)
- **笔记链接**: [SCGN 精读笔记](reading-notes/SCGN.md)
- **架构图**: reading-notes/assets/arch-SCGN.png
- **添加时间**: 2026-07-21

---

## 📊 优化器/训练技巧

> 暂无论文

---

## 🔄 其他

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
- [ ] SSFA-Net - Sparse Strip and Dual-Domain Spatial-Frequency Attention for Efficient Image Dehazing (Neurocomputing 2026)
- [ ] DAWN+ - Wavelet-Based Direction-Aware Deraining (TNNLS 2025)
- [ ] DnLUT - Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables (CVPR 2025)
- [ ] SCGN - Statistical Characteristic-Guided Denoising for Rapid HRTEM Imaging (CVPR 2026)
- [ ] MFFDNet - Single Image Deraining via Dual-Channel Mixed Feature Fusion (IEEE TIM 2024)
- [ ] TransMamb - Hybrid Transformer-Mamba Network for Single Image Deraining (IEEE TIP)
- [ ] CPRAformer - Cross Paradigm Representation and Alignment Transformer for Image Deraining (ACM MM 2025)
- [ ] DPPD - Learning Dynamic Prompts for All-in-One Image Restoration (IEEE TIP 2025)

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
