# 计划：在论文库中新增12篇硬编码论文

## 摘要
在 `docs/index.html` 的 JavaScript 中硬编码12篇新论文数据，追加到现有论文列表末尾。在侧边栏分类中新增"去雨"分类。保留原有论文和 localStorage 阅读状态，通过去重逻辑避免重复添加。

## 当前状态分析
- 网站为单文件 `docs/index.html`，论文数据通过 `fetch('papers-library.md')` 加载后由 `parseLibrary()` 解析为 `allPapers` 数组
- 全局 `categories` 数组（line ~2219）定义了侧边栏分类，共10个分类
- `init()` 函数（line ~3541）的加载流程：`fetch md` → `parseLibrary` → `applySavedStatus()` → `migrateMethodData()` → 渲染
- `applySavedStatus()` 只从 localStorage 恢复已有论文的阅读状态，不会覆盖新论文的默认"待读"状态
- `papers-library.md` 位于 `docs/` 目录下，与 `index.html` 同级，GitHub Pages 部署时可正常访问

## 修改内容

### 1. 新增"去雨"分类
**文件**: `docs/index.html`
**位置**: 全局 `categories` 数组（line ~2219）
**操作**: 在数组中插入 `{ emoji: '🌧️', name: '去雨' }`
**原因**: 用户提供的12篇论文中有3篇属于"去雨"分类，需要侧边栏支持筛选

### 2. 定义12篇新论文的硬编码数据
**文件**: `docs/index.html`
**位置**: 在 `categories` 数组之后、`detectVenue` 函数之前（line ~2231 附近）
**操作**: 定义常量 `DEFAULT_NEW_PAPERS`，包含12个论文对象
**论文对象字段**: `title, category, categoryEmoji, tags, status, summary, painPoints, method, scenarios, arxiv, arxivUrl, code, notes, archImage, date`
**字段默认值策略**:
- `status`: 固定为 `'待读'`
- `category` / `categoryEmoji`: 按用户提供的分类映射
- `tags`: 按用户提供的标签拆分
- `arxivUrl`: 用户提供的链接（"待补充"时为空）
- `date`: 用户提供的年份
- `summary` / `painPoints` / `method` / `scenarios`: 基于论文标题和分类生成合理的简短描述
- `code` / `notes` / `archImage`: 空字符串

### 3. 修改 `init()` 函数追加新论文
**文件**: `docs/index.html`
**位置**: `init()` 函数内部（line ~3550 附近）
**操作**: 在 `allPapers = parseLibrary(md)` 之后、`applySavedStatus()` 之前，插入以下逻辑：
```js
// Append hard-coded new papers, deduplicate by title
const existingTitles = new Set(allPapers.map(p => p.title));
for (const newP of DEFAULT_NEW_PAPERS) {
  if (!existingTitles.has(newP.title)) {
    allPapers.push(newP);
  }
}
```
**原因**: 
- 保留 `papers-library.md` 中的原有论文
- 在末尾追加12篇新论文
- 通过标题去重，避免重复添加（如果 papers-library.md 中已存在同名论文）
- 在 `applySavedStatus()` 之前追加，确保新论文不会被 localStorage 中的旧状态意外覆盖

### 4. 验证去重和状态逻辑
- `applySavedStatus()` 使用论文标题作为 key 从 localStorage 读取状态，新论文由于标题不在 localStorage 中，自然保持默认"待读"
- 去重逻辑确保刷新页面不会重复添加
- `renderSidebar()` 会自动统计新分类的论文数量

## 假设与决策
1. **假设**: 用户提供的12篇论文标题与 `papers-library.md` 中现有论文标题不完全相同。如果存在同名，去重逻辑会跳过。
2. **假设**: "待补充"的论文链接（第9、10、12篇）在 `arxivUrl` 中留空，不影响页面显示。
3. **决策**: 新论文的 `summary/painPoints/method/scenarios` 字段使用基于标题和分类的简短描述，而非空字符串，确保卡片和详情页有内容可显示。
4. **决策**: 不在 `papers-library.md` 中修改内容，所有新论文完全通过 JS 硬编码追加，保持单点修改。

## 验证步骤
1. 本地打开 `docs/index.html`，确认网络面板中 `papers-library.md` 加载成功
2. 检查侧边栏是否出现"🌧️ 去雨"分类及正确数量
3. 检查论文卡片网格中是否出现12篇新论文，状态为"待读"
4. 点击新论文卡片，确认详情弹窗显示正常（概览/方法分析/标签等）
5. 将某篇原有论文标为已读，刷新页面，确认状态保持，且新论文未被重复添加
6. `git add docs/index.html && git commit && git push`，确认 GitHub Pages 自动部署后线上正常
