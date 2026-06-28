const state = {
  posts: [],
  total: 0,
  offset: 0,
  pageStart: 0,
  limit: 50,
  loading: false,
  chatBusy: false,
  filter: "all",
  minStar: 0,
  minReply: 0,
  sort: "recent",
  query: "",
  datePreset: "",
  startDate: "",
  endDate: "",
  calendarMonth: localMonthKey(new Date()),
  calendar: null,
  collections: [],
  collectionFilter: "",
  selectedPid: null,
  selectedPost: null,
  stats: null,
  agentCursor: 0,
  renderedEvents: new Set(),
  assistantByJob: new Map(),
  assistantTextNodes: new Map(),
  statusByJobKey: new Map(),
  pollTimer: null,
  polling: false,
  agentActive: false,
  chatAutoScroll: true,
  scrollFrame: 0,
  workbenchStatusTimer: null,
  pendingPostReload: null,
  pidImportBusy: false,
  titleGenerationBusy: false,
};

const nodes = {
  indexSummary: document.querySelector("#indexSummary"),
  searchInput: document.querySelector("#searchInput"),
  datePresetSelect: document.querySelector("#datePresetSelect"),
  dateCustomControls: document.querySelector("#dateCustomControls"),
  startDateInput: document.querySelector("#startDateInput"),
  endDateInput: document.querySelector("#endDateInput"),
  collectionFilterSelect: document.querySelector("#collectionFilterSelect"),
  newCollectionButton: document.querySelector("#newCollectionButton"),
  generateTitlesButton: document.querySelector("#generateTitlesButton"),
  minStarInput: document.querySelector("#minStarInput"),
  minReplyInput: document.querySelector("#minReplyInput"),
  sortSelect: document.querySelector("#sortSelect"),
  calendarGrid: document.querySelector("#calendarGrid"),
  calendarTitle: document.querySelector("#calendarTitle"),
  calendarTodayLabel: document.querySelector("#calendarTodayLabel"),
  calendarPrevButton: document.querySelector("#calendarPrevButton"),
  calendarNextButton: document.querySelector("#calendarNextButton"),
  postList: document.querySelector("#postList"),
  detailPane: document.querySelector("#detailPane"),
  resultCount: document.querySelector("#resultCount"),
  chatLog: document.querySelector("#chatLog"),
  chatForm: document.querySelector("#chatForm"),
  chatInput: document.querySelector("#chatInput"),
  sessionLabel: document.querySelector("#sessionLabel"),
  modelLabel: document.querySelector("#modelLabel"),
  workbenchStatus: document.querySelector("#workbenchStatus"),
  pidImportButton: document.querySelector("#pidImportButton"),
  pidImportModal: document.querySelector("#pidImportModal"),
  pidImportForm: document.querySelector("#pidImportForm"),
  pidImportText: document.querySelector("#pidImportText"),
  pidImportCount: document.querySelector("#pidImportCount"),
  pidImportChips: document.querySelector("#pidImportChips"),
  pidImportResult: document.querySelector("#pidImportResult"),
  pidImportCloseButton: document.querySelector("#pidImportCloseButton"),
  pidImportCancelButton: document.querySelector("#pidImportCancelButton"),
  pidImportSubmitButton: document.querySelector("#pidImportSubmitButton"),
  reindexButton: document.querySelector("#reindexButton"),
  prevPageButton: document.querySelector("#prevPageButton"),
  nextPageButton: document.querySelector("#nextPageButton"),
  loadMoreButton: document.querySelector("#loadMoreButton"),
  pageInfo: document.querySelector("#pageInfo"),
  appShell: document.querySelector(".app-shell"),
  contentGrid: document.querySelector("#contentGrid"),
  resizers: document.querySelectorAll(".column-resizer"),
};

const LAYOUT_STORAGE_KEY = "treehole-layout-v1";

function icon(name) {
  return `<svg class="icon"><use href="#icon-${name}"></use></svg>`;
}

function formatPid(pid) {
  return `#${pid}`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function localDateKey(date) {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
}

function localMonthKey(date) {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}`;
}

function parseMonth(value) {
  const [year, month] = String(value || "").split("-").map((item) => Number.parseInt(item, 10));
  if (!Number.isFinite(year) || !Number.isFinite(month)) return new Date();
  return new Date(year, month - 1, 1);
}

function addMonths(value, delta) {
  const date = parseMonth(value);
  date.setMonth(date.getMonth() + delta);
  return localMonthKey(date);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function buildPostsUrl(offset = 0) {
  const params = new URLSearchParams({
    offset: String(offset),
    limit: String(state.limit),
    q: state.query,
    filter: state.filter,
    min_star: String(state.minStar || 0),
    min_reply: String(state.minReply || 0),
    sort: state.sort,
  });
  if (state.datePreset && state.datePreset !== "custom") params.set("date_preset", state.datePreset);
  if (state.datePreset === "custom") {
    if (state.startDate) params.set("start_date", state.startDate);
    if (state.endDate) params.set("end_date", state.endDate);
  }
  if (state.collectionFilter) params.set("collection", state.collectionFilter);
  return `/api/posts?${params.toString()}`;
}

function readThresholdInput(input) {
  const value = Number.parseInt(input?.value || "0", 10);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

function normalizePidDigits(value) {
  return String(value || "").replace(/[\uff10-\uff19]/g, (char) =>
    String.fromCharCode(char.charCodeAt(0) - 0xff10 + 48),
  );
}

function extractPidsFromText(value) {
  const text = normalizePidDigits(value);
  const regex = /(^|[^\d])(\d{5,10})(?!\d)/g;
  const pids = [];
  const seen = new Set();
  let match = regex.exec(text);
  while (match) {
    const pid = Number.parseInt(match[2], 10);
    if (Number.isFinite(pid) && pid > 0 && !seen.has(pid)) {
      seen.add(pid);
      pids.push(pid);
    }
    match = regex.exec(text);
  }
  return pids;
}

function setWorkbenchStatus(text, { timeout = 3200 } = {}) {
  if (!nodes.workbenchStatus) return;
  nodes.workbenchStatus.textContent = text;
  if (state.workbenchStatusTimer) clearTimeout(state.workbenchStatusTimer);
  if (timeout > 0) {
    state.workbenchStatusTimer = setTimeout(() => {
      state.workbenchStatusTimer = null;
      updateWorkbenchStatus();
    }, timeout);
  }
}

function updateWorkbenchStatus() {
  if (!nodes.workbenchStatus) return;
  if (state.loading) {
    nodes.workbenchStatus.textContent = state.query.trim() ? "本地检索中" : "读取本地缓存";
  } else if (state.collectionFilter) {
    nodes.workbenchStatus.textContent = currentCollection()?.name || "收藏筛选";
  } else {
    nodes.workbenchStatus.textContent = "本地缓存";
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const error = new Error(payload?.error || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.data = payload;
    throw error;
  }
  return payload || {};
}

async function copyText(text) {
  const value = String(text || "");
  if (!value) return;
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.append(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function currentCollection() {
  return state.collections.find((item) => item.id === state.collectionFilter) || null;
}

function activeCollectionId() {
  return state.collectionFilter || state.collections[0]?.id || "default";
}

function activeCollectionName() {
  return state.collections.find((item) => item.id === activeCollectionId())?.name || "默认收藏";
}

function renderCollectionOptions() {
  const previous = state.collectionFilter;
  const options = [`<option value="">全部帖子</option>`].concat(
    state.collections.map((item) => {
      const label = `${item.name}${Number.isFinite(item.count) ? ` (${item.count})` : ""}`;
      return `<option value="${escapeHtml(item.id)}">${escapeHtml(label)}</option>`;
    }),
  );
  nodes.collectionFilterSelect.innerHTML = options.join("");
  const exists = !previous || state.collections.some((item) => item.id === previous);
  state.collectionFilter = exists ? previous : "";
  nodes.collectionFilterSelect.value = state.collectionFilter;
}

async function loadCollections() {
  const data = await fetchJson("/api/collections");
  state.collections = data.collections || [];
  renderCollectionOptions();
}

async function createCollection() {
  const name = window.prompt("新收藏栏名称");
  if (!name?.trim()) return;
  try {
    const data = await fetchJson("/api/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    state.collections = data.collections || [];
    state.collectionFilter = data.collection?.id || state.collectionFilter;
    renderCollectionOptions();
    await loadPosts({ reset: true, offset: 0 });
    setWorkbenchStatus(`已新建收藏栏：${data.collection?.name || name}`);
  } catch (error) {
    setWorkbenchStatus(`新建收藏栏失败：${error.message}`, { timeout: 5200 });
  }
}

function renderDateControls() {
  nodes.datePresetSelect.value = state.datePreset;
  nodes.dateCustomControls.hidden = state.datePreset !== "custom";
  nodes.startDateInput.value = state.startDate;
  nodes.endDateInput.value = state.endDate;
}

function resetSelectionForFilter() {
  state.selectedPid = null;
  state.selectedPost = null;
}

async function loadCalendar(month = state.calendarMonth) {
  try {
    const data = await fetchJson(`/api/calendar?month=${encodeURIComponent(month)}`);
    state.calendar = data;
    state.calendarMonth = data.month || month;
    renderCalendar();
  } catch (error) {
    nodes.calendarGrid.innerHTML = `<div class="calendar-empty">日历加载失败</div>`;
  }
}

function renderCalendar() {
  const data = state.calendar;
  if (!data) return;
  const monthDate = parseMonth(data.month);
  const monthLabel = `${monthDate.getFullYear()}-${pad2(monthDate.getMonth() + 1)}`;
  nodes.calendarTitle.textContent = monthLabel;
  nodes.calendarTodayLabel.textContent = `今天 ${data.today}`;

  const days = data.days || [];
  if (!days.length) {
    nodes.calendarGrid.innerHTML = `<div class="calendar-empty">暂无日期</div>`;
    return;
  }

  const firstWeekday = (monthDate.getDay() + 6) % 7;
  const blanks = Array.from({ length: firstWeekday }, () => `<span class="calendar-blank" aria-hidden="true"></span>`);
  const dayButtons = days.map((item) => {
    const dayNo = Number.parseInt(item.date.slice(-2), 10);
    const selected = state.datePreset === "custom" && state.startDate === item.date && state.endDate === item.date;
    const classes = [
      "calendar-day",
      item.coverage || "empty",
      item.complete ? "complete" : "",
      item.date === data.today ? "today" : "",
      selected ? "selected" : "",
    ]
      .filter(Boolean)
      .join(" ");
    const title = `${item.date} · ${item.count} 帖 · ${item.coverageLabel || ""}${item.firstTime ? ` · ${item.firstTime}-${item.lastTime}` : ""}`;
    return `
      <button class="${classes}" data-date="${item.date}" type="button" title="${escapeHtml(title)}">
        <span>${dayNo}</span>
        <strong>${item.count || ""}</strong>
      </button>
    `;
  });
  nodes.calendarGrid.innerHTML = blanks.concat(dayButtons).join("");
  nodes.calendarGrid.querySelectorAll(".calendar-day").forEach((button) => {
    button.addEventListener("click", () => {
      const date = button.dataset.date;
      state.datePreset = "custom";
      state.startDate = date;
      state.endDate = date;
      state.calendarMonth = date.slice(0, 7);
      renderDateControls();
      renderCalendar();
      resetSelectionForFilter();
      loadPosts({ reset: true, offset: 0 });
    });
  });
}

function setPidImportBusy(isBusy) {
  state.pidImportBusy = isBusy;
  nodes.pidImportText.disabled = isBusy;
  nodes.pidImportButton.disabled = isBusy;
  nodes.pidImportSubmitButton.textContent = isBusy ? "抓取中" : "提取并抓取";
  renderPidImportPreview();
}

function renderPidImportPreview() {
  const pids = extractPidsFromText(nodes.pidImportText.value);
  nodes.pidImportCount.textContent = `${pids.length} 个 PID`;
  nodes.pidImportSubmitButton.disabled = state.pidImportBusy || !pids.length;

  if (!pids.length) {
    nodes.pidImportChips.innerHTML = `<span class="pid-import-empty">暂无 PID</span>`;
    return pids;
  }

  const visible = pids.slice(0, 80).map((pid) => `<span class="pid-chip">${formatPid(pid)}</span>`);
  if (pids.length > 80) {
    visible.push(`<span class="pid-chip">+${pids.length - 80}</span>`);
  }
  nodes.pidImportChips.innerHTML = visible.join("");
  return pids;
}

function renderPidImportResults(data) {
  const summary = data.summary || {};
  const results = data.results || [];
  const statusText = {
    fetched: "已抓取",
    deleted: "已删除",
    failed: "失败",
  };
  const rows = results
    .slice(0, 120)
    .map((item) => {
      const status = item.status || "failed";
      const safeStatus = statusText[status] ? status : "failed";
      const commentTotal = Number(item.comment_total || 0);
      const commentCount = Number(item.comments || 0);
      const commentInfo =
        status === "fetched" ? ` · 预览 ${commentCount}${commentTotal ? `/${commentTotal}` : ""}` : "";
      const title = `${item.title || item.error || ""}${commentInfo}`;
      return `
        <div class="pid-import-row">
          <span class="post-pid">${formatPid(item.pid)}</span>
          <span class="pid-import-row-status ${safeStatus}">${statusText[status] || status}</span>
          <span class="pid-import-row-title" title="${escapeHtml(title)}">${escapeHtml(title)}</span>
        </div>
      `;
    })
    .join("");
  const overflow = results.length > 120 ? `<div class="pid-import-row-title">还有 ${results.length - 120} 条结果未展开</div>` : "";
  const preview = summary.comment_preview_limit ? `，每帖评论预览 ${summary.comment_preview_limit} 条` : "";
  nodes.pidImportResult.innerHTML = `
    <div class="pid-import-summary">
      完成 ${summary.requested || results.length} 个：已抓取 ${summary.fetched || 0}，已删除 ${summary.deleted || 0}，失败 ${summary.failed || 0}${preview}
    </div>
    ${rows || "暂无结果"}
    ${overflow}
  `;
}

function openPidImportDialog() {
  nodes.pidImportModal.hidden = false;
  renderPidImportPreview();
  requestAnimationFrame(() => nodes.pidImportText.focus());
}

function closePidImportDialog() {
  nodes.pidImportModal.hidden = true;
}

async function submitPidImport() {
  const pids = renderPidImportPreview();
  if (!pids.length || state.pidImportBusy) return;

  setPidImportBusy(true);
  nodes.pidImportResult.textContent = `正在抓取 ${pids.length} 个 PID...`;
  setWorkbenchStatus(`导入 ${pids.length} 个 PID 中`, { timeout: 0 });

  try {
    const data = await fetchJson("/api/pids/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: nodes.pidImportText.value }),
    });
    renderPidImportResults(data);
    state.stats = data.stats || state.stats;
    state.selectedPid = null;
    state.selectedPost = null;
    await Promise.all([loadCollections(), loadCalendar()]);
    await loadPosts({ reset: true, offset: 0 });
    const summary = data.summary || {};
    setWorkbenchStatus(`PID 导入完成：成功 ${(summary.fetched || 0) + (summary.deleted || 0)}，失败 ${summary.failed || 0}`);
  } catch (error) {
    nodes.pidImportResult.innerHTML = `<div class="pid-import-summary">导入失败：${escapeHtml(error.message)}</div>`;
    setWorkbenchStatus(`PID 导入失败：${error.message}`, { timeout: 5200 });
  } finally {
    setPidImportBusy(false);
  }
}

async function loadPosts({ reset = false, offset = null } = {}) {
  if (state.loading) {
    if (reset) state.pendingPostReload = { reset, offset };
    return;
  }
  state.loading = true;
  updateWorkbenchStatus();
  if (reset) {
    state.pageStart = Number(offset || 0);
    state.offset = state.pageStart;
    state.posts = [];
    renderPostList();
  }
  try {
    const data = await fetchJson(buildPostsUrl(state.offset));
    state.total = data.total || 0;
    state.stats = data.stats || state.stats;
    state.posts = reset ? data.posts : mergePostLists(state.posts, data.posts);
    state.offset = state.pageStart + state.posts.length;
    updateSummary();
    renderPostList();
    updatePager();
    if (!state.selectedPid && state.posts.length) {
      await selectPost(state.posts[0].pid);
    } else if (state.selectedPid && !state.posts.some((post) => post.pid === state.selectedPid)) {
      state.selectedPid = null;
      state.selectedPost = null;
      renderDetail();
    }
  } catch (error) {
    setWorkbenchStatus(`加载失败：${error.message}`, { timeout: 5200 });
  } finally {
    state.loading = false;
    updatePager();
    if (!state.workbenchStatusTimer) updateWorkbenchStatus();
    if (state.pendingPostReload) {
      const pending = state.pendingPostReload;
      state.pendingPostReload = null;
      loadPosts(pending);
    }
  }
}

function mergePostLists(oldList, newList) {
  const seen = new Set(oldList.map((post) => post.pid));
  const merged = oldList.slice();
  for (const post of newList || []) {
    if (!seen.has(post.pid)) {
      seen.add(post.pid);
      merged.push(post);
    }
  }
  return merged;
}

async function selectPost(pid) {
  state.selectedPid = Number(pid);
  renderPostList();
  nodes.detailPane.innerHTML = `<div class="empty-state">正在读取 ${formatPid(pid)}</div>`;
  try {
    state.selectedPost = await fetchJson(`/api/posts/${pid}`);
    renderDetail();
  } catch (error) {
    nodes.detailPane.innerHTML = `<div class="empty-state">读取失败：${escapeHtml(error.message)}</div>`;
  }
}

function updateSummary() {
  const stats = state.stats;
  if (!stats) return;
  if (state.query.trim() || state.datePreset || state.collectionFilter) {
    nodes.indexSummary.textContent = `${state.total} 条本地匹配 · ${stats.posts} 帖缓存池`;
  } else {
    nodes.indexSummary.textContent = `${stats.posts} 帖 · ${stats.comment_files} 评论缓存 · 离线`;
  }
  nodes.resultCount.textContent = String(state.total);
  if (stats.model && nodes.modelLabel) nodes.modelLabel.textContent = stats.model;
  if (!state.workbenchStatusTimer) updateWorkbenchStatus();
}

function updatePager() {
  const shownStart = state.total && state.posts.length ? state.pageStart + 1 : 0;
  const shownEnd = state.pageStart + state.posts.length;
  const loadedEnd = state.pageStart + state.posts.length;
  nodes.pageInfo.textContent = `${shownStart}-${shownEnd} / ${state.total || 0}`;
  nodes.prevPageButton.disabled = state.loading || state.pageStart <= 0;
  nodes.nextPageButton.disabled = state.loading || state.pageStart + state.limit >= state.total;
  nodes.loadMoreButton.disabled = state.loading || loadedEnd >= state.total;
  nodes.loadMoreButton.textContent = state.loading ? "加载中" : "加载更多";
}

function renderPostList() {
  nodes.resultCount.textContent = String(state.total || state.posts.length);
  if (!state.posts.length) {
    const loadingText = state.query.trim() ? "正在检索本地缓存" : "正在载入本地缓存";
    nodes.postList.innerHTML = `<div class="empty-state">${state.loading ? loadingText : "没有匹配结果"}</div>`;
    return;
  }

  nodes.postList.innerHTML = state.posts
    .map((post) => {
      const active = post.pid === state.selectedPid ? " active" : "";
      const tags = (post.tags || []).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
      const deleted = post.deleted ? `<span class="deleted-badge">已删除</span>` : "";
      const custom = post.custom_title ? `<span class="local-title-badge">备注</span>` : "";
      const favorite = post.is_favorite ? `<span class="local-title-badge favorite">收藏</span>` : "";
      return `
        <button class="post-row${active}" data-pid="${post.pid}" type="button">
          <div class="post-meta-line">
            <div class="post-stamp">
              <span class="post-pid">${formatPid(post.pid)}</span>
              <span class="post-date">${escapeHtml(post.dateLabel || "")}</span>
              <span class="post-relative">${escapeHtml(post.relative || "")}</span>
              ${deleted}
              ${custom}
              ${favorite}
            </div>
            <div class="post-signals">
              <span class="signal">${post.reply || 0}${icon("chat")}</span>
              <span class="signal">${post.star || 0}${icon("star")}</span>
            </div>
          </div>
          <div class="post-row-title">${escapeHtml(post.title || "无正文帖子")}</div>
          <div class="post-row-note">${escapeHtml(post.text || "")}</div>
          <div class="tag-list">${tags}</div>
        </button>
      `;
    })
    .join("");

  document.querySelectorAll(".post-row").forEach((button) => {
    button.addEventListener("click", () => selectPost(button.dataset.pid));
  });
}

function formatPostForClipboard(post) {
  const comments = (post.comments || [])
    .slice(0, 30)
    .map((comment, idx) => `${idx + 1}. [${comment.name || "洞友"}] ${comment.text || ""}`)
    .join("\n");
  return [
    `${formatPid(post.pid)} ${post.title || ""}`,
    post.post_time || "",
    "",
    post.text || "",
    comments ? "\n评论：\n" + comments : "",
  ].join("\n").trim();
}

function renderDetail() {
  const post = state.selectedPost;
  if (!post) {
    nodes.detailPane.innerHTML = `<div class="empty-state">请选择左侧帖子</div>`;
    return;
  }
  const tags = (post.tags || []).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
  const deleted = post.deleted ? `<span class="deleted-badge detail">已删除</span>` : "";
  const activeCid = activeCollectionId();
  const favoriteActive = (post.collection_ids || []).includes(activeCid);
  const collectionChoices = state.collections
    .map((item) => {
      const checked = (post.collection_ids || []).includes(item.id) ? " checked" : "";
      return `
        <label class="collection-choice">
          <input type="checkbox" data-collection-id="${escapeHtml(item.id)}"${checked} />
          <span>${escapeHtml(item.name)}</span>
        </label>
      `;
    })
    .join("");
  const comments = (post.comments || [])
    .map(
      (comment) => `
        <article class="comment">
          <div class="comment-meta">
            <span class="comment-speaker">[${escapeHtml(comment.name || "洞友")}]</span>
            <span>${escapeHtml(comment.reply_time || comment.time || "")}</span>
          </div>
          <p>${escapeHtml(comment.text || "")}</p>
        </article>
      `,
    )
    .join("");
  const emptyCommentsText =
    post.comment_total > 0
      ? `帖子有 ${post.comment_total} 条回复；当前没有帖子接口自带预览`
      : "本地缓存暂无评论";

  nodes.detailPane.innerHTML = `
    <article class="thread-paper">
      <div class="thread-meta-line">
        <div class="post-stamp">
          <span class="post-pid">${formatPid(post.pid)}</span>
          <span class="post-date">${escapeHtml(post.dateLabel || "")}</span>
          <span class="post-relative">${escapeHtml(post.relative || "")}</span>
          ${deleted}
        </div>
        <div class="post-signals">
          <span class="signal">${post.reply || 0}${icon("chat")}</span>
          <span class="signal">${post.star || 0}${icon("star")}</span>
        </div>
      </div>
      <div class="thread-title-row">
        <div class="title-editor thread-title-editor">
          <svg class="icon"><use href="#icon-edit"></use></svg>
          <input
            class="thread-title-input"
            id="customTitleInput"
            type="text"
            maxlength="80"
            value="${escapeHtml(post.custom_title || post.title || "")}"
            placeholder="${escapeHtml(post.original_title || post.title || "无正文帖子")}"
          />
          <button class="thread-action" id="saveTitleButton" title="保存标题" aria-label="保存标题">${icon("check")}</button>
          <button class="thread-action" id="clearTitleButton" title="清空标题" aria-label="清空标题">×</button>
        </div>
        <button class="thread-action${favoriteActive ? " active" : ""}" id="favoritePostButton" title="切换当前收藏栏" aria-label="切换当前收藏栏">${icon("star")}</button>
      </div>
      <p class="thread-body">${escapeHtml(post.text || "")}</p>
      <div class="thread-tags">${tags}</div>
      <div class="collection-editor">
        ${collectionChoices || `<span class="empty-comment">暂无收藏栏</span>`}
      </div>
      <div class="detail-actions">
        <button class="thread-action" id="refreshPostButton" title="刷新" aria-label="刷新">${icon("loader")}</button>
        <button class="thread-action primary" id="askPostButton" title="问这条" aria-label="问这条">${icon("chat")}</button>
        <button class="thread-action" id="copyPostButton" title="复制帖子" aria-label="复制帖子">${icon("copy")}</button>
      </div>
    </article>

    <section class="comments-paper">
      <h3>评论 ${post.comment_total ? `<span>${post.comment_total}</span>` : ""}</h3>
      <div class="comments-list">
        ${comments || `<div class="empty-comment">${escapeHtml(emptyCommentsText)}</div>`}
      </div>
    </section>

    <section class="source-paper">
      <h3>来源</h3>
      <div class="source-list">
        <span class="source-link">${icon("database")} ${escapeHtml(post.source || `pid_post_cache/${post.pid}.json`)}</span>
        ${post.original_title && post.original_title !== post.title ? `<span class="source-link">${icon("edit")} 原标题：${escapeHtml(post.original_title)}</span>` : ""}
      </div>
    </section>
  `;

  document.querySelector("#favoritePostButton").addEventListener("click", () => togglePostCollection(post.pid, activeCid));
  document.querySelector("#saveTitleButton").addEventListener("click", () => savePostTitle(post.pid));
  document.querySelector("#clearTitleButton").addEventListener("click", () => savePostTitle(post.pid, ""));
  document.querySelectorAll(".collection-choice input").forEach((input) => {
    input.addEventListener("change", () => togglePostCollection(post.pid, input.dataset.collectionId));
  });
  document.querySelector("#refreshPostButton").addEventListener("click", () => refreshSelectedPost());
  document.querySelector("#askPostButton").addEventListener("click", () => {
    nodes.chatInput.value = `结合 ${formatPid(post.pid)} 总结一下这条帖子和评论`;
    nodes.chatInput.focus();
  });
  document.querySelector("#copyPostButton").addEventListener("click", async () => {
    await copyText(formatPostForClipboard(post));
    setWorkbenchStatus(`已复制 ${formatPid(post.pid)} 帖子`);
  });
}

function applyPostFromServer(post) {
  if (!post) return;
  state.selectedPost = post;
  state.posts = state.posts.map((item) =>
    item.pid === post.pid
      ? {
          ...item,
          ...post,
          comments: [],
          text: post.text?.slice(0, 180) || "",
        }
      : item,
  );
  renderPostList();
  renderDetail();
}

function updatePostListMetaFromServer(post) {
  if (!post) return;
  state.posts = state.posts.map((item) =>
    item.pid === post.pid
      ? {
          ...item,
          ...post,
          comments: [],
          text: post.text?.slice(0, 180) || item.text || "",
        }
      : item,
  );
  renderPostList();
}

async function togglePostCollection(pid, collectionId) {
  if (!collectionId) return;
  try {
    const data = await fetchJson(`/api/posts/${pid}/metadata`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ toggle_collection: collectionId }),
    });
    state.collections = data.collections || state.collections;
    renderCollectionOptions();
    state.selectedPost = data.post || state.selectedPost;
    updatePostListMetaFromServer(data.post);
    renderDetail();
    setWorkbenchStatus(`已更新 ${formatPid(pid)} 收藏栏`);
  } catch (error) {
    setWorkbenchStatus(`收藏更新失败：${error.message}`, { timeout: 5200 });
    if (state.selectedPost?.pid === Number(pid)) renderDetail();
  }
}

async function savePostTitle(pid, forcedValue = null) {
  const input = document.querySelector("#customTitleInput");
  const title = forcedValue === null ? input?.value || "" : forcedValue;
  try {
    const data = await fetchJson(`/api/posts/${pid}/metadata`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ custom_title: title }),
    });
    applyPostFromServer(data.post);
    setWorkbenchStatus(title.trim() ? `已保存 ${formatPid(pid)} 标题` : `已清空 ${formatPid(pid)} 标题`);
  } catch (error) {
    setWorkbenchStatus(`标题保存失败：${error.message}`, { timeout: 5200 });
  }
}

async function generateMissingTitles() {
  if (state.titleGenerationBusy) return;
  const collectionId = activeCollectionId();
  state.titleGenerationBusy = true;
  nodes.generateTitlesButton.disabled = true;
  setWorkbenchStatus(`为「${activeCollectionName()}」补标题中`, { timeout: 0 });
  try {
    const data = await fetchJson("/api/titles/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collection_id: collectionId }),
    });
    state.collections = data.collections || state.collections;
    renderCollectionOptions();
    await loadPosts({ reset: true, offset: state.pageStart });
    if (state.selectedPid) await selectPost(state.selectedPid);
    const summary = data.summary || {};
    setWorkbenchStatus(`标题生成完成：更新 ${summary.updated || 0}，失败 ${summary.failed || 0}`);
  } catch (error) {
    setWorkbenchStatus(`标题生成失败：${error.message}`, { timeout: 6200 });
  } finally {
    state.titleGenerationBusy = false;
    nodes.generateTitlesButton.disabled = false;
  }
}

function setChatBusy(isBusy) {
  state.chatBusy = isBusy;
  nodes.chatForm.querySelector("button[type='submit']").disabled = isBusy;
}

function chatIsNearBottom() {
  return nodes.chatLog.scrollHeight - nodes.chatLog.scrollTop - nodes.chatLog.clientHeight < 80;
}

function scrollChatToBottom({ force = false } = {}) {
  if (!force && !state.chatAutoScroll) return;
  if (state.scrollFrame) return;
  state.scrollFrame = requestAnimationFrame(() => {
    nodes.chatLog.scrollTop = nodes.chatLog.scrollHeight;
    state.scrollFrame = 0;
  });
}

function renderSessionLabel(jobStatus = null) {
  if (jobStatus?.status === "running" || jobStatus?.status === "queued") {
    nodes.sessionLabel.textContent = "agent.py 正在运行";
  } else {
    nodes.sessionLabel.textContent = "agent.py CLI";
  }
}

function addSystemLine(text, { jobId = "", key = "" } = {}) {
  const systemKey = key ? `${jobId}:${key}` : "";
  let line = systemKey ? state.statusByJobKey.get(systemKey) : null;
  if (!line) {
    line = document.createElement("div");
    line.className = "system-line";
    if (systemKey) state.statusByJobKey.set(systemKey, line);
    nodes.chatLog.append(line);
  }
  line.textContent = text;
  scrollChatToBottom();
}

function addMessage(role, body, { jobId = "" } = {}) {
  const message = document.createElement("article");
  message.className = `message ${role}`;
  message.innerHTML = `
    <div class="message-top">
      <div class="message-role">${escapeHtml(role)}</div>
    </div>
    <div class="message-body">${escapeHtml(body)}</div>
    <div class="message-actions">
      <button class="message-copy" type="button" title="复制" aria-label="复制消息">${icon("copy")}</button>
    </div>
  `;
  message.querySelector(".message-copy").addEventListener("click", async (event) => {
    event.stopPropagation();
    await copyText(message.querySelector(".message-body")?.innerText || "");
    setWorkbenchStatus("已复制侧边栏消息");
  });
  nodes.chatLog.append(message);
  scrollChatToBottom();
  if (role === "assistant" && jobId) {
    const bodyNode = message.querySelector(".message-body");
    let textNode = bodyNode.firstChild;
    if (!textNode || textNode.nodeType !== Node.TEXT_NODE) {
      textNode = document.createTextNode(bodyNode.textContent || "");
      bodyNode.textContent = "";
      bodyNode.append(textNode);
    }
    state.assistantByJob.set(jobId, bodyNode);
    state.assistantTextNodes.set(jobId, textNode);
  }
  return message;
}

function appendAssistantDelta(jobId, chunk) {
  let body = state.assistantByJob.get(jobId);
  if (!body) {
    body = addMessage("assistant", "", { jobId }).querySelector(".message-body");
  }
  let textNode = state.assistantTextNodes.get(jobId);
  if (!textNode) {
    textNode = document.createTextNode(body.textContent || "");
    body.textContent = "";
    body.append(textNode);
    state.assistantTextNodes.set(jobId, textNode);
  }
  textNode.appendData(chunk);
  scrollChatToBottom();
}

function renderAgentEvent(event) {
  if (!event || state.renderedEvents.has(event.id)) return;
  state.renderedEvents.add(event.id);
  state.agentCursor = Math.max(state.agentCursor, Number(event.id) || 0);

  if (event.role === "user") {
    addMessage("user", event.content || "", { jobId: event.job_id });
  } else if (event.role === "assistant") {
    if (event.kind === "delta") {
      appendAssistantDelta(event.job_id || "default", event.content || "");
    } else {
      addMessage("assistant", event.content || "", { jobId: event.job_id });
    }
  } else if (event.role === "error") {
    addMessage("assistant", `调用失败：${event.content || ""}`, { jobId: event.job_id });
  } else if (event.role === "system") {
    addSystemLine(event.content || "", { jobId: event.job_id, key: event.kind === "status" ? event.key || "status" : "" });
  }
}

async function pollAgentHistory({ reset = false } = {}) {
  if (state.polling) return;
  state.polling = true;
  const after = reset ? 0 : state.agentCursor;
  try {
    const data = await fetchJson(`/api/agent/history?after=${after}`);
    if (data.model && nodes.modelLabel) nodes.modelLabel.textContent = data.model;
    if (reset) {
      nodes.chatLog.innerHTML = "";
      state.renderedEvents = new Set();
      state.assistantByJob = new Map();
      state.assistantTextNodes = new Map();
      state.statusByJobKey = new Map();
      state.agentCursor = 0;
      state.chatAutoScroll = true;
    }
    for (const event of data.events || []) renderAgentEvent(event);
    renderSessionLabel(data.activeJobStatus);
    state.agentActive = data.activeJobStatus?.status === "queued" || data.activeJobStatus?.status === "running";
    setChatBusy(state.agentActive);
    if (reset) scrollChatToBottom({ force: true });
  } catch (error) {
    addSystemLine(`侧栏历史读取失败：${error.message}`);
  } finally {
    state.polling = false;
  }
}

async function runAgentPollingLoop({ reset = false } = {}) {
  await pollAgentHistory({ reset });
  const delay = state.agentActive || state.chatBusy ? 160 : 900;
  state.pollTimer = setTimeout(() => runAgentPollingLoop(), delay);
}

function startAgentPolling() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  runAgentPollingLoop({ reset: true });
}

async function submitChat(question) {
  if (state.chatBusy) return;
  question = question.trim();
  if (!question) return;
  state.chatAutoScroll = true;
  setChatBusy(true);
  try {
    await fetchJson("/api/agent/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    await pollAgentHistory();
    scrollChatToBottom({ force: true });
  } catch (error) {
    addMessage("assistant", `提交失败：${error.message}`);
    setChatBusy(false);
  }
}

async function refreshSelectedPost() {
  const post = state.selectedPost;
  if (!post) return;
  const button = document.querySelector("#refreshPostButton");
  if (button) button.disabled = true;
  setWorkbenchStatus(`刷新 ${formatPid(post.pid)} 中`, { timeout: 0 });
  try {
    const data = await fetchJson(`/api/posts/${post.pid}/refresh`, { method: "POST" });
    applyPostFromServer(data.post);
    await Promise.all([loadCollections(), loadCalendar()]);
    setWorkbenchStatus(`${formatPid(post.pid)} 已刷新`);
  } catch (error) {
    setWorkbenchStatus(`${formatPid(post.pid)} 刷新失败：${error.message}`, { timeout: 5200 });
  } finally {
    const latestButton = document.querySelector("#refreshPostButton");
    if (latestButton) latestButton.disabled = false;
  }
}

function debounce(fn, delay = 260) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

const reloadPostsDebounced = debounce(() => loadPosts({ reset: true }), 260);

function applyThresholdFilters() {
  state.minStar = readThresholdInput(nodes.minStarInput);
  state.minReply = readThresholdInput(nodes.minReplyInput);
  resetSelectionForFilter();
  reloadPostsDebounced();
}

nodes.minStarInput.addEventListener("input", applyThresholdFilters);
nodes.minReplyInput.addEventListener("input", applyThresholdFilters);

nodes.searchInput.addEventListener("input", (event) => {
  state.query = event.target.value;
  resetSelectionForFilter();
  reloadPostsDebounced();
});

nodes.datePresetSelect.addEventListener("change", (event) => {
  state.datePreset = event.target.value;
  if (state.datePreset !== "custom") {
    state.startDate = "";
    state.endDate = "";
  }
  renderDateControls();
  resetSelectionForFilter();
  loadPosts({ reset: true, offset: 0 });
});

nodes.startDateInput.addEventListener("change", (event) => {
  state.datePreset = "custom";
  state.startDate = event.target.value;
  renderDateControls();
  resetSelectionForFilter();
  loadPosts({ reset: true, offset: 0 });
});

nodes.endDateInput.addEventListener("change", (event) => {
  state.datePreset = "custom";
  state.endDate = event.target.value;
  renderDateControls();
  resetSelectionForFilter();
  loadPosts({ reset: true, offset: 0 });
});

nodes.collectionFilterSelect.addEventListener("change", (event) => {
  state.collectionFilter = event.target.value;
  resetSelectionForFilter();
  updateWorkbenchStatus();
  loadPosts({ reset: true, offset: 0 });
});

nodes.newCollectionButton.addEventListener("click", () => {
  createCollection();
});

nodes.generateTitlesButton.addEventListener("click", () => {
  generateMissingTitles();
});

nodes.sortSelect.addEventListener("change", (event) => {
  state.sort = event.target.value;
  resetSelectionForFilter();
  loadPosts({ reset: true, offset: 0 });
});

nodes.calendarPrevButton.addEventListener("click", () => {
  state.calendarMonth = addMonths(state.calendarMonth, -1);
  loadCalendar();
});

nodes.calendarNextButton.addEventListener("click", () => {
  state.calendarMonth = addMonths(state.calendarMonth, 1);
  loadCalendar();
});

nodes.postList.addEventListener("scroll", () => {
  const nearBottom = nodes.postList.scrollTop + nodes.postList.clientHeight > nodes.postList.scrollHeight - 220;
  if (nearBottom && state.pageStart + state.posts.length < state.total) {
    loadPosts();
  }
});

nodes.prevPageButton.addEventListener("click", () => {
  const nextOffset = Math.max(0, state.pageStart - state.limit);
  resetSelectionForFilter();
  loadPosts({ reset: true, offset: nextOffset });
});

nodes.nextPageButton.addEventListener("click", () => {
  const nextOffset = Math.min(Math.max(0, state.total - 1), state.pageStart + state.limit);
  resetSelectionForFilter();
  loadPosts({ reset: true, offset: nextOffset });
});

nodes.loadMoreButton.addEventListener("click", () => {
  loadPosts();
});

nodes.pidImportButton.addEventListener("click", () => {
  openPidImportDialog();
});

nodes.pidImportCloseButton.addEventListener("click", () => {
  closePidImportDialog();
});

nodes.pidImportCancelButton.addEventListener("click", () => {
  closePidImportDialog();
});

nodes.pidImportModal.addEventListener("click", (event) => {
  if (event.target === nodes.pidImportModal) closePidImportDialog();
});

nodes.pidImportText.addEventListener("input", () => {
  nodes.pidImportResult.textContent = "";
  renderPidImportPreview();
});

nodes.pidImportForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitPidImport();
});

nodes.reindexButton.addEventListener("click", async () => {
  setWorkbenchStatus("重建本地索引中", { timeout: 0 });
  try {
    const data = await fetchJson("/api/reindex", { method: "POST" });
    state.stats = data.stats;
    state.selectedPid = null;
    state.selectedPost = null;
    await Promise.all([loadCollections(), loadCalendar()]);
    await loadPosts({ reset: true, offset: 0 });
    setWorkbenchStatus(`索引完成：${data.stats.posts} 帖`);
  } catch (error) {
    setWorkbenchStatus(`索引失败：${error.message}`, { timeout: 5200 });
  }
});

nodes.chatLog.addEventListener("scroll", () => {
  state.chatAutoScroll = chatIsNearBottom();
});

nodes.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.chatBusy) return;
  const question = nodes.chatInput.value.trim();
  if (!question) return;
  nodes.chatInput.value = "";
  submitChat(question);
});

nodes.chatInput.addEventListener("keydown", (event) => {
  if (event.isComposing) return;
  if (event.key !== "Enter") return;
  if (event.shiftKey) return;
  event.preventDefault();
  if (state.chatBusy) return;
  const question = nodes.chatInput.value.trim();
  if (!question) return;
  nodes.chatInput.value = "";
  submitChat(question);
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !nodes.pidImportModal.hidden) {
    closePidImportDialog();
  }
});

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

function readLayout() {
  try {
    return JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function writeLayout(layout) {
  localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout));
}

function setFeedWidth(width) {
  const gridWidth = nodes.contentGrid.getBoundingClientRect().width;
  const max = gridWidth - 8 - 320;
  document.documentElement.style.setProperty("--feed-width", `${clamp(width, 300, max)}px`);
}

function setAssistantWidth(width) {
  const shellWidth = nodes.appShell.getBoundingClientRect().width;
  const max = shellWidth - 8 - 640;
  document.documentElement.style.setProperty("--assistant-width", `${clamp(width, 300, max)}px`);
}

function currentFeedWidth() {
  return document.querySelector(".feed-pane").getBoundingClientRect().width;
}

function currentAssistantWidth() {
  return document.querySelector(".assistant-panel").getBoundingClientRect().width;
}

function saveCurrentLayout() {
  writeLayout({
    feedWidth: Math.round(currentFeedWidth()),
    assistantWidth: Math.round(currentAssistantWidth()),
  });
}

function setupColumnResizers() {
  const isResizableLayout = () => !window.matchMedia("(max-width: 1180px)").matches;
  const saved = readLayout();
  if (isResizableLayout()) {
    if (Number.isFinite(saved.feedWidth)) setFeedWidth(saved.feedWidth);
    if (Number.isFinite(saved.assistantWidth)) setAssistantWidth(saved.assistantWidth);
  }

  nodes.resizers.forEach((resizer) => {
    resizer.addEventListener("pointerdown", (event) => {
      if (!isResizableLayout()) return;
      event.preventDefault();
      resizer.setPointerCapture(event.pointerId);
      resizer.classList.add("active");
      document.body.classList.add("is-resizing");

      const kind = resizer.dataset.resizer;
      const startX = event.clientX;
      const startFeed = currentFeedWidth();
      const startAssistant = currentAssistantWidth();

      const onPointerMove = (moveEvent) => {
        const deltaX = moveEvent.clientX - startX;
        if (kind === "feed") {
          setFeedWidth(startFeed + deltaX);
        } else {
          setAssistantWidth(startAssistant - deltaX);
        }
      };

      const onPointerUp = () => {
        resizer.classList.remove("active");
        document.body.classList.remove("is-resizing");
        saveCurrentLayout();
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        window.removeEventListener("pointercancel", onPointerUp);
      };

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp, { once: true });
      window.addEventListener("pointercancel", onPointerUp, { once: true });
    });

    resizer.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      if (resizer.dataset.resizer === "feed") {
        setFeedWidth(currentFeedWidth() + direction * 24);
      } else {
        setAssistantWidth(currentAssistantWidth() - direction * 24);
      }
      saveCurrentLayout();
    });
  });

  window.addEventListener("resize", () => {
    if (!isResizableLayout()) return;
    setFeedWidth(currentFeedWidth());
    setAssistantWidth(currentAssistantWidth());
    saveCurrentLayout();
  });
}

async function init() {
  setupColumnResizers();
  renderSessionLabel();
  renderDateControls();
  updatePager();
  try {
    await Promise.all([loadCollections(), loadCalendar()]);
  } catch (error) {
    setWorkbenchStatus(`初始化失败：${error.message}`, { timeout: 5200 });
  }
  startAgentPolling();
  loadPosts({ reset: true, offset: 0 });
}

init();
