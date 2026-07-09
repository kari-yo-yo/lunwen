# 论文热力图功能完善计划

## Summary

修复当前热力图布局错乱问题（单列垂直堆叠），并增强整体用户体验：添加月份/星期标签、自定义 Tooltip、点击详情、水平滚动支持，以及更合理的颜色层级。

---

## Current State Analysis

### 问题 1：布局根因 —— 列被强制垂直堆叠

`renderHeatmap()` 中设置了 `grid.style.flexDirection = 'column'`，导致 21 个列 `<div>` 全部纵向排列，用户只能看到一列灰色方块。

此外，`.sidebar` 宽度约 260px，而 21 列 × (13px 宽 + 3px 间距) = 336px，超出容器宽度，CSS 中没有溢出处理机制。

### 问题 2：JS 逻辑冗余

`renderHeatmap()` 先用双重循环构建了一次扁平的 147 个 cell 字符串，随后立即丢弃；接着又用几乎相同的逻辑重新构建 `columnHtml`。第一次计算完全无意义。

### 问题 3：标签与交互缺失

- 无月份标签、无星期标签，用户无法定位时间
- 仅依赖浏览器原生 `title` 做 tooltip，体验简陋
- 无点击交互，无法查看某天具体新增了哪些论文

---

## Proposed Changes

### 改动 1：HTML 结构调整

**文件**：`papers/index.html`（约第 1465-1479 行）

**What**：为热力图增加语义化容器层：左侧星期标签列、顶部月份标签行、可滚动网格区域。

**Why**：原有结构只有单层 `<div class="heatmap-grid">`，无法支撑标签对齐和滚动控制。

**How**：

替换原热力图 HTML 为：

```html
  <!-- Heatmap -->
  <div class="heatmap-section" id="heatmapSection">
    <div class="heatmap-title">📅 论文添加热力图</div>
    <div class="heatmap-body" id="heatmapBody">
      <!-- 左侧星期标签 -->
      <div class="heatmap-weekdays" id="heatmapWeekdays"></div>
      <!-- 右侧滚动区域：月份 + 网格 -->
      <div class="heatmap-scroll-wrap">
        <div class="heatmap-months" id="heatmapMonths"></div>
        <div class="heatmap-grid" id="heatmapGrid">
          <div class="heatmap-empty">暂无添加记录</div>
        </div>
      </div>
    </div>
    <div class="heatmap-legend">
      <span>少</span>
      <span class="heatmap-cell level-1"></span>
      <span class="heatmap-cell level-2"></span>
      <span class="heatmap-cell level-3"></span>
      <span class="heatmap-cell level-4"></span>
      <span class="heatmap-cell level-5"></span>
      <span>多</span>
    </div>
  </div>
```

---

### 改动 2：CSS 布局修复与增强

**文件**：`papers/index.html`（约第 1217-1261 行）

**What**：
- 新增 `.heatmap-scroll-wrap` 实现水平滚动
- 新增 `.heatmap-column` 作为每周列容器
- 新增 `.heatmap-months` / `.heatmap-month-label` 月份标签
- 新增 `.heatmap-weekdays` / `.heatmap-weekday-label` 星期标签
- 修复 `.heatmap-grid` 为 `flex-direction: row; flex-wrap: nowrap`
- 增加 `level-5` 颜色层级
- 调整低层级颜色透明度，提升可见度

**Why**：确保 21 列横向排列，超出宽度时通过滚动查看；标签帮助用户定位时间；更丰富的颜色层级让数据分布更清晰。

**How**：替换 `.heatmap-section` 至 `.heatmap-empty` 之间的全部 CSS 为：

```css
/* --- Heatmap / Reading Calendar --- */
.heatmap-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 24px;
  min-width: 0;
}
.heatmap-title {
  font-size: 14px; font-weight: 600; color: var(--text);
  margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.heatmap-body {
  display: flex;
  flex-direction: row;
  gap: 6px;
}
.heatmap-scroll-wrap {
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 6px;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}
.heatmap-scroll-wrap::-webkit-scrollbar {
  height: 6px;
}
.heatmap-scroll-wrap::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}
.heatmap-grid {
  display: flex;
  flex-direction: row;
  gap: 3px;
  flex-wrap: nowrap;
  align-items: flex-start;
}
.heatmap-column {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.heatmap-cell {
  width: 13px; height: 13px;
  border-radius: 2px;
  background: var(--border);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  cursor: pointer;
  position: relative;
}
.heatmap-cell.level-1 { background: rgba(99,102,241,0.30); }
.heatmap-cell.level-2 { background: rgba(99,102,241,0.50); }
.heatmap-cell.level-3 { background: rgba(99,102,241,0.70); }
.heatmap-cell.level-4 { background: rgba(99,102,241,0.90); }
.heatmap-cell.level-5 { background: rgba(129,140,248,1); }
.heatmap-cell:hover {
  transform: scale(1.6);
  z-index: 5;
  box-shadow: 0 2px 8px rgba(99,102,241,0.4);
  outline: 1px solid var(--primary-light);
}
.heatmap-months {
  display: flex;
  flex-direction: row;
  gap: 3px;
  margin-bottom: 6px;
}
.heatmap-month-label {
  font-size: 10px;
  color: var(--text-muted);
  height: 14px;
  line-height: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.heatmap-weekdays {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding-top: 20px;
}
.heatmap-weekday-label {
  font-size: 10px;
  color: var(--text-muted);
  width: 22px;
  height: 13px;
  line-height: 13px;
  text-align: right;
}
.heatmap-weekday-label.empty {
  visibility: hidden;
}
.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px; font-size: 11px; color: var(--text-dim);
}
.heatmap-empty {
  font-size: 13px; color: var(--text-dim);
  padding: 16px 0;
}
@media (max-width: 768px) {
  .heatmap-section { padding: 16px; }
  .heatmap-cell { width: 10px; height: 10px; }
  .heatmap-column { gap: 2px; }
  .heatmap-grid { gap: 2px; }
  .heatmap-weekday-label { width: 20px; height: 10px; line-height: 10px; font-size: 9px; }
}
```

---

### 改动 3：JS `renderHeatmap()` 重构

**文件**：`papers/index.html`（约第 2651-2735 行）

**What**：
- 移除冗余的第一次 `html` 拼接
- 移除 `grid.style.display = 'flex'` 和 `grid.style.flexDirection = 'column'` 错误内联样式
- 一次性生成星期标签、月份标签、网格列
- 增加 `data-*` 属性（date, count, weekday）供交互使用
- 调整 level 阈值匹配 5 级颜色

**Why**：修复布局、去除冗余、为后续交互提供数据基础。

**How**：替换 `renderHeatmap()` 为：

```javascript
function renderHeatmap() {
  const grid = document.getElementById('heatmapGrid');
  const monthsRow = document.getElementById('heatmapMonths');
  const weekdaysCol = document.getElementById('heatmapWeekdays');
  if (!grid || !monthsRow || !weekdaysCol) return;

  const activityMap = {};
  allPapers.forEach(p => {
    if (p.date) {
      const d = parseDateKey(p.date);
      if (d) activityMap[d] = (activityMap[d] || 0) + 1;
    }
  });
  try {
    const history = JSON.parse(localStorage.getItem('paper_activity_history') || '{}');
    for (const [dateKey, count] of Object.entries(history)) {
      activityMap[dateKey] = (activityMap[dateKey] || 0) + count;
    }
  } catch {}

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const startDate = new Date(today);
  startDate.setDate(startDate.getDate() - 139);
  const dayOfWeek = startDate.getDay();
  startDate.setDate(startDate.getDate() - dayOfWeek);

  const totalWeeks = 21;
  const dayNames = ['日', '一', '二', '三', '四', '五', '六'];
  const monthNames = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];

  // 星期标签：只在周一/三/五显示
  let weekdayHtml = '';
  for (let row = 0; row < 7; row++) {
    const showLabel = (row === 1 || row === 3 || row === 5);
    weekdayHtml += `<div class="heatmap-weekday-label ${showLabel ? '' : 'empty'}">${showLabel ? '周' + dayNames[row] : ''}</div>`;
  }
  weekdaysCol.innerHTML = weekdayHtml;

  // 月份标签
  const monthLabelMap = new Array(totalWeeks).fill('');
  for (let col = 0; col < totalWeeks; col++) {
    const firstDayOfCol = new Date(startDate);
    firstDayOfCol.setDate(firstDayOfCol.getDate() + col * 7);
    const m = firstDayOfCol.getMonth();
    const y = firstDayOfCol.getFullYear();
    if (col === 0) {
      monthLabelMap[col] = `${y}年${monthNames[m]}`;
    } else {
      const prevDay = new Date(startDate);
      prevDay.setDate(prevDay.getDate() + (col - 1) * 7);
      if (prevDay.getMonth() !== m) {
        monthLabelMap[col] = monthNames[m];
      }
    }
  }

  let monthsHtml = '';
  for (let col = 0; col < totalWeeks; col++) {
    monthsHtml += `<div class="heatmap-month-label" style="width:13px;">${monthLabelMap[col]}</div>`;
  }
  monthsRow.innerHTML = monthsHtml;

  // 网格列
  let gridHtml = '';
  for (let col = 0; col < totalWeeks; col++) {
    gridHtml += '<div class="heatmap-column">';
    for (let row = 0; row < 7; row++) {
      const cellDate = new Date(startDate);
      cellDate.setDate(cellDate.getDate() + col * 7 + row);
      const key = dateKey(cellDate);
      const count = activityMap[key] || 0;

      let level = 0;
      if (count >= 1 && count <= 2) level = 1;
      else if (count >= 3 && count <= 4) level = 2;
      else if (count >= 5 && count <= 7) level = 3;
      else if (count >= 8 && count <= 11) level = 4;
      else if (count >= 12) level = 5;

      const ymd = cellDate.toISOString().split('T')[0];
      const weekdayText = '周' + dayNames[cellDate.getDay()];
      gridHtml += `<div class="heatmap-cell level-${level}"
        data-date="${ymd}"
        data-count="${count}"
        data-weekday="${weekdayText}"
        title="${ymd} ${weekdayText}：${count} 次操作"></div>`;
    }
    gridHtml += '</div>';
  }

  if (gridHtml) {
    grid.innerHTML = gridHtml;
    // 同步月份标签宽度与 cell 宽度（响应式适配）
    const cells = grid.querySelectorAll('.heatmap-cell');
    const labels = monthsRow.querySelectorAll('.heatmap-month-label');
    const cellW = cells.length ? cells[0].offsetWidth : 13;
    labels.forEach(l => l.style.width = cellW + 'px');
  } else {
    grid.innerHTML = '<div class="heatmap-empty">暂无添加记录</div>';
  }
}
```

---

### 改动 4：热力图交互增强（Tooltip + 点击详情）

**文件**：`papers/index.html`（在 `renderHeatmap()` 之后新增）

**What**：
- 自定义浮动 Tooltip（替代浏览器原生 `title`）
- 点击 cell 弹出当日详情：操作次数 + 当日新增论文列表

**Why**：提升用户体验，让用户能快速了解某天具体发生了什么。

**How**：在 `renderHeatmap()` 之后、`initParticles()` 之前新增：

```javascript
// ============================================================
//  Heatmap Interactions (Tooltip + Click Detail)
// ============================================================
(function initHeatmapInteractions() {
  const grid = document.getElementById('heatmapGrid');
  if (!grid) return;

  let tooltip = document.getElementById('heatmapTooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'heatmapTooltip';
    tooltip.className = 'card-tooltip';
    tooltip.style.pointerEvents = 'none';
    document.body.appendChild(tooltip);
  }

  function showTooltip(el, date, count, weekday) {
    tooltip.innerHTML = `
      <div style="font-weight:600;margin-bottom:4px;">${date} ${weekday}</div>
      <div style="font-size:13px;color:var(--text-dim);">当日操作：${count} 次</div>
    `;
    tooltip.classList.add('visible');

    const rect = el.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2;
    let top = rect.top - tooltip.offsetHeight - 8;
    if (left < 8) left = 8;
    if (left + tooltip.offsetWidth > window.innerWidth - 8) {
      left = window.innerWidth - tooltip.offsetWidth - 8;
    }
    if (top < 8) top = rect.bottom + 8;
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
  }

  function hideTooltip() {
    tooltip.classList.remove('visible');
  }

  grid.addEventListener('mouseover', (e) => {
    const cell = e.target.closest('.heatmap-cell');
    if (!cell) return;
    const date = cell.dataset.date;
    const count = parseInt(cell.dataset.count || '0', 10);
    const weekday = cell.dataset.weekday || '';
    showTooltip(cell, date, count, weekday);
  });

  grid.addEventListener('mouseout', (e) => {
    if (e.target.closest('.heatmap-cell')) hideTooltip();
  });

  grid.addEventListener('click', (e) => {
    const cell = e.target.closest('.heatmap-cell');
    if (!cell) return;
    const date = cell.dataset.date;
    const count = parseInt(cell.dataset.count || '0', 10);
    if (count === 0) {
      showToast(`${date} 无操作记录`);
      return;
    }
    const papersOnDate = allPapers.filter(p => parseDateKey(p.date) === date);
    let detail = `<strong>${date}</strong><br>当日操作 ${count} 次<br><br>`;
    if (papersOnDate.length) {
      detail += '新增论文：<ul style="margin:4px 0;padding-left:16px;">';
      papersOnDate.forEach(p => {
        detail += `<li>${escapeHtml(p.title || '未命名')}</li>`;
      });
      detail += '</ul>';
    } else {
      detail += '（主要为状态变更记录）';
    }
    showModal('当日详情', detail);
  });
})();
```

> 页面中已存在 `showToast()` 和 `showModal()` 函数，直接复用。

---

## Assumptions & Decisions

1. **复用现有 Tooltip 样式**：页面已有 `.card-tooltip` 样式类，热力图 Tooltip 直接复用，保持视觉一致性。
2. **水平滚动优于缩放**：sidebar 宽度固定，21 列必然溢出。选择水平滚动而非缩小 cell，确保每个格子仍可清晰辨识。
3. **5 级颜色层级**：从 4 级增至 5 级，并将阈值放宽，让数据分布更均匀、视觉更丰富。
4. **星期标签间隔显示**：为避免 7 行全部显示文字导致左列过宽，只在周一/三/五显示 "周一"、"周三"、"周五"。
5. **月份标签只在切换时显示**：仅在月份发生变化的列顶部显示月份名，第一列额外带年份，保持简洁。

---

## Verification Steps

1. **布局验证**：在桌面端打开网站，检查热力图是否显示为 21 列 × 7 行的横向网格，而非单列。
2. **滚动验证**：缩小浏览器窗口或检查 sidebar 区域，确认内容溢出时出现水平滚动条。
3. **标签验证**：
   - 左侧是否显示 "周一"、"周三"、"周五" 标签
   - 顶部是否在月份切换处显示月份名
4. **交互验证**：
   - 鼠标悬停 cell 是否出现精致的浮动 Tooltip（替代原生 title）
   - 点击有数据的 cell 是否弹出 "当日详情" Modal
   - 点击无数据的 cell 是否弹出 "无操作记录" Toast
5. **颜色验证**：检查 5 级颜色是否正确渲染，低频次也能明显辨识。
6. **响应式验证**：在移动端（或缩小窗口至 < 768px）检查 cell 是否缩小为 10px，布局是否正常。
