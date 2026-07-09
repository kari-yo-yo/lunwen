# 论文卡片添加神经网络架构图

## Summary
为论文库网站的每张卡片添加 CVHub 风格的神经网络架构图缩略图，并在详情弹窗中展示完整大图。采用 GenerateImage 生成图片 + 数据驱动的方式，前端自动解析展示。

## Current State Analysis
- 前端为单文件 `index.html`，卡片由 JS `renderPapers()` 动态渲染
- 论文数据从 `papers-library.md` 解析，通过 `parseLibrary()` 生成 `allPapers` 数组
- 现有 10 篇论文，每篇有 method/tags/notesUrl 等字段，**无架构图字段**
- `reading-notes/assets/` 下已有 17 张图（7 张旧架构图 + 10 张 process 流程图）
- mediakit-cli 不可用，但可用 GenerateImage 工具生成图像

## Proposed Changes

### 1. 为 10 篇论文生成架构图 PNG

使用 GenerateImage 为每篇论文生成 CVHub 风格的神经网络架构图（3D tensor blocks、彩色编码模块、数据流箭头、数学符号标注）。

| 文件名 | 论文 | 架构核心 |
|--------|------|---------|
| arch-MPRNet.png | MPRNet | 三阶段渐进: Stage1(1/4) -> SAM -> Stage2(1/2) -> SAM -> Stage3(ORSNet) |
| arch-DRSformer.png | DRSformer | U型编码解码 + Top-K 稀疏注意力(TKSA) + MEFC专家混合 |
| arch-dawn-plus.png | DAWN+ | 小波分解 -> 向量分解(V/H方向) -> DAM方向感知 -> 逆小波 |
| arch-HINet.png | HINet | 两阶段U-Net + HIN Block(半实例归一化) + CSFF |
| arch-DEA-Net.png | DEA-Net | U-Net + DEAB(DEConv 5路差分卷积 + CGA注意力) |
| arch-CDT.png | CDT | 聚合(CA) -> 全局建模(超像素注意力) -> 恢复(DA双路自适应) |
| arch-ASSANet.png | ASSANet | 双分支: 全局KV压缩 + 稀疏注意力(6种策略) |
| arch-DGUNet.png | DGUNet | PGD展开: FGDM(梯度估计) -> IPMM(近端映射) 多阶段 |
| arch-PHENet.png | PHENet | RGB+DEM双输入 -> 物理引导 -> HEM高程增强 -> 变化检测 |
| arch-SSFA-Net.png | SSFA-Net | 空间注意力 + 光谱注意力 -> 门控融合 -> 分类 |

**保存路径**: `papers/reading-notes/assets/arch-{Name}.png`
**生成方式**: GenerateImage 工具，prompt 中描述具体模块名称、连接关系、维度变化

### 2. 修改 `papers-library.md` — 每篇论文添加架构图字段

在每篇论文的 bullet 列表中，`笔记链接` 之后添加：

```markdown
- **架构图**: reading-notes/assets/arch-MPRNet.png
```

### 3. 修改 `index.html` — parseLibrary 解析新字段

**位置**: `parseLibrary()` 函数（约第 1469 行）
- paper 对象初始化（第 1511 行）添加 `archImage: ''`
- switch-case（约第 1544 行前）添加 `case '架构图': paper.archImage = val.trim(); break;`

### 4. 修改 `index.html` — 卡片模板展示架构图

**位置**: `renderPapers()` 中的卡片 HTML 模板（约第 1784 行）

在 `<div class="paper-card-header">` 之前插入：
```html
${p.archImage ? `<div class="paper-card-arch"><img src="${p.archImage}" ...></div>` : ''}
```

### 5. 添加 CSS 样式

- `.paper-card-arch`: 卡片顶部架构图区域（高度 180px，渐变遮罩，hover 缩放）
- `.detail-arch`: 详情弹窗架构图（完整尺寸，点击 lightbox）

### 6. 修改 `openDetail` — 详情弹窗展示架构图

**位置**: `openDetail()` 函数（约第 1819 行）

在 `<h2>` 之后插入架构图展示区域，支持点击放大。

## Assumptions & Decisions

| 决策 | 选择 | 理由 |
|------|------|------|
| 图片生成方式 | GenerateImage | 最快且效果好，prompt 可精确描述架构 |
| 容错 | onerror 隐藏 | 图片不存在时不留空白 |
| 卡片位置 | 顶部 | 架构图是最直观的论文标识 |
| 以后新论文 | 加架构图字段即可 | 数据驱动，无需改代码 |

## Verification

1. 部署后检查每张卡片是否展示架构图缩略图
2. 点击卡片后检查详情弹窗是否展示完整架构图
3. 点击架构图检查 lightbox 放大功能
4. 检查没有架构图的论文（如果新增但未指定）是否正常降级不显示图片
5. 部署到 IGA Pages 验证线上效果
