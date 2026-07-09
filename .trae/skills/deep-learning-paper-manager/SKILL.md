---
name: "deep-learning-paper-manager"
description: "Manages deep learning papers: search, explain (pain points & methods), and categorize (e.g., attention, channel enhancement). Invoke when user needs to find, read, or organize DL papers."
---

# Deep Learning Paper Manager

## Purpose
帮助用户完成深度学习论文的**搜索、精读讲解、分类归档**全流程管理。

## When to Invoke
- 用户想要搜索某篇深度学习论文或某个方向的论文
- 用户上传/提供了论文内容，希望深入讲解
- 用户读完论文后，希望按类别归档整理
- 用户希望查看已整理的论文库或某个分类下的论文

---

## Capability 1: Search Papers (论文搜索)

### Supported Sources
- arXiv (优先)
- Google Scholar
- Semantic Scholar
- Papers With Code

### Search Workflow
1. 接收用户的论文标题、arXiv ID、DOI 或关键词
2. 使用 WebSearch 查找论文的官方链接与信息
3. 使用 WebFetch 获取论文摘要、引言、方法等关键内容
4. 整理为如下格式的论文卡片：

```markdown
### 📄 论文卡片: [标题]
- **作者**: 
- **发表时间/会议**: 
- **arXiv/DOI**: [链接]
- **关键词**: 
- **一句话总结**: 
- **摘要核心**: 
```

---

## Capability 2: Explain Paper (论文讲解)

### 讲解维度 (必须覆盖)
讲解论文时必须按以下结构输出：

```markdown
## 📖 论文详解: [标题]

### 1. 研究背景与动机 (Motivation)
- 该方向已有的方法是什么？
- 现有方法存在什么**痛点/问题**？
- 本文要解决的核心问题是什么？

### 2. 核心方法 (Method)
- 方法的整体框架图/流程描述
- 关键模块的详细解释（配合公式、伪代码）
- 与之前方法的关键差异

### 3. 痛点分析 (Pain Points Addressed)
| 痛点 | 现有方法的问题 | 本文的解决方案 |
|------|--------------|--------------|
| 痛点1 | ... | ... |
| 痛点2 | ... | ... |

### 4. 实验与效果 (Experiments)
- 数据集、评价指标
- 与 SOTA 的对比结果
- 消融实验 (Ablation Study) 结论

### 5. 个人总结与适用场景
- 该方法的优点与局限
- 适合用在什么任务/场景
- 是否值得精读/复现
```

### 讲解风格
- 语言通俗易懂，避免堆砌公式而不解释
- 对关键公式要解释每个符号的物理意义
- 多用比喻和直观解释帮助理解

---

## Capability 3: Categorize & Store Papers (论文分类归档)

### 分类体系 (可根据用户需要扩展)
默认提供以下分类，用户可自定义新增：

| 分类名 | 说明 | 典型论文 |
|--------|------|----------|
| 🔥 注意力机制 (Attention Mechanisms) | SE-Net, CBAM, Self-Attention, Non-local 等 | 通道/空间注意力相关 |
| 🌊 通道增强/特征增强 (Channel/Feature Enhancement) | GhostNet, EfficientNet, 特征融合等 | 提升特征表达能力 |
| 🏗️ 骨干网络 (Backbones) | ResNet, Transformer, ConvNeXt, Mamba 等 | 基础特征提取网络 |
| 🎯 目标检测 (Object Detection) | YOLO, DETR, RCNN 系列等 | 检测任务专用 |
| 🖼️ 语义分割 (Semantic Segmentation) | U-Net, DeepLab, SAM 等 | 分割任务专用 |
| 📝 生成模型 (Generative Models) | GAN, VAE, Diffusion 等 | 图像/文本生成 |
| 🔗 多模态 (Multi-modal) | CLIP, BLIP, LLaVA 等 | 跨模态学习 |
| ⚡ 模型压缩/加速 (Model Compression) | 剪枝、量化、知识蒸馏、NAS 等 | 轻量化部署 |
| 📊 优化器/训练技巧 (Optimizers & Training) | AdamW, Warmup, Label Smoothing 等 | 训练策略 |
| 🔄 其他 (Others) | 不属于以上分类的论文 | 待进一步细分 |

### 归档格式
每篇归档论文保存为如下结构化信息：

```markdown
### [论文标题]
- **分类**: [分类名]
- **标签**: [可添加多个标签，如 transformer, lightweight, real-time]
- **状态**: [已读/待读/精读中/已复现]
- **一句话总结**: 
- **核心痛点**: 
- **核心方法**: 
- **适用场景**: 
- **笔记链接**: [如适用]
- **代码链接**: [Papers With Code / GitHub]
- **添加时间**: YYYY-MM-DD
```

### 归档操作
- 当用户说"帮我归档"、"分类整理"、"存到XX分类"时触发
- 询问用户确认分类，或根据论文内容自动推荐分类
- 更新到用户指定的论文库文件（如 `papers-library.md`）

---

## Capability 4: Show Library (展示论文库)

### 展示方式
1. **按分类展示**: 列出某分类下的所有论文
2. **按标签展示**: 筛选特定标签的论文
3. **按状态展示**: 如只看"已读"或"待读"
4. **全文检索**: 根据关键词在论文库中搜索

### 输出格式
```markdown
## 📚 论文库概览

### 🔥 注意力机制 (共 X 篇)
| 序号 | 标题 | 状态 | 一句话总结 |
|------|------|------|------------|
| 1 | SE-Net | 已读 | 通道注意力开山之作 |
| 2 | CBAM | 已读 | 通道+空间双重注意力 |

### 🏗️ 骨干网络 (共 X 篇)
...
```

---

## File Management

### 推荐文件结构
```
papers/
├── papers-library.md          # 主论文库，记录所有论文的分类归档信息
├── reading-notes/             # 精读笔记目录
│   ├── se-net.md
│   └── cbam.md
├── search-history.md          # 搜索历史记录 (可选)
└── todo-read.md               # 待读清单 (可选)
```

### 文件位置
- 所有论文相关文件默认保存在当前工作目录下的 `papers/` 文件夹中
- 如果用户有指定位置，优先使用用户指定的路径

---

## Interaction Rules

1. **搜索时**: 尽量找到论文的 arXiv 版本和 PDF 链接，方便用户下载
2. **讲解时**: 必须覆盖"痛点"和"方法"两大维度，不能只列标题和摘要
3. **归档时**: 主动推荐合适的分类，等待用户确认后再写入文件
4. **展示时**: 用表格形式呈现，清晰美观
5. **学习追踪**: 可询问用户"是否已读完"、"是否需要归档"，主动推进学习流程
