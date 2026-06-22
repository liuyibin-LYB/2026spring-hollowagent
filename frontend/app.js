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
  scrollFrame: 0,
  workbenchStatusTimer: null,
  pendingPostReload: null,
};

const nodes = {
  indexSummary: document.querySelector("#indexSummary"),
  searchInput: document.querySelector("#searchInput"),
  minStarInput: document.querySelector("#minStarInput"),
  minReplyInput: document.querySelector("#minReplyInput"),
  sortSelect: document.querySelector("#sortSelect"),
  postList: document.querySelector("#postList"),
  detailPane: document.querySelector("#detailPane"),
  resultCount: document.querySelector("#resultCount"),
  chatLog: document.querySelector("#chatLog"),
  chatForm: document.querySelector("#chatForm"),
  chatInput: document.querySelector("#chatInput"),
  sessionLabel: document.querySelector("#sessionLabel"),
  modelLabel: document.querySelector("#modelLabel"),
  workbenchStatus: document.querySelector("#workbenchStatus"),
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
  return `/api/posts?${params.toString()}`;
}

function readThresholdInput(input) {
  const value = Number.parseInt(input?.value || "0", 10);
  return Number.isFinite(value) && value > 0 ? value : 0;
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
  if (state.query.trim()) {
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
      return `
        <button class="post-row${active}" data-pid="${post.pid}" type="button">
          <div class="post-meta-line">
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

function renderDetail() {
  const post = state.selectedPost;
  if (!post) {
    nodes.detailPane.innerHTML = `<div class="empty-state">请选择左侧帖子</div>`;
    return;
  }
  const tags = (post.tags || []).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
  const deleted = post.deleted ? `<span class="deleted-badge detail">已删除</span>` : "";
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
      <h2 class="thread-title">${escapeHtml(post.title || "无正文帖子")}</h2>
      <p class="thread-body">${escapeHtml(post.text || "")}</p>
      <div class="thread-tags">${tags}</div>
      <div class="detail-actions">
        <button class="thread-action" id="refreshPostButton" title="刷新" aria-label="刷新">${icon("loader")}</button>
        <button class="thread-action primary" id="askPostButton" title="问这条" aria-label="问这条">${icon("chat")}</button>
        <button class="thread-action" id="copyPostButton" title="复制正文" aria-label="复制正文">${icon("copy")}</button>
      </div>
    </article>

    <section class="comments-paper">
      <h3>评论 ${post.comment_total ? `<span>${post.comment_total}</span>` : ""}</h3>
      <div class="comments-list">
        ${comments || `<div class="empty-comment">本地缓存暂无评论</div>`}
      </div>
    </section>

    <section class="source-paper">
      <h3>来源</h3>
      <div class="source-list">
        <span class="source-link">${icon("database")} ${escapeHtml(post.source || `pid_post_cache/${post.pid}.json`)}</span>
      </div>
    </section>
  `;

  document.querySelector("#refreshPostButton").addEventListener("click", () => refreshSelectedPost());
  document.querySelector("#askPostButton").addEventListener("click", () => {
    nodes.chatInput.value = `结合 ${formatPid(post.pid)} 总结一下这条帖子和评论`;
    nodes.chatInput.focus();
  });
  document.querySelector("#copyPostButton").addEventListener("click", async () => {
    await navigator.clipboard?.writeText(post.text || "");
    setWorkbenchStatus(`已复制 ${formatPid(post.pid)} 正文`);
  });
}

function setChatBusy(isBusy) {
  state.chatBusy = isBusy;
  nodes.chatForm.querySelector("button[type='submit']").disabled = isBusy;
}

function scrollChatToBottom() {
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
    <div class="message-role">${escapeHtml(role)}</div>
    <div class="message-body">${escapeHtml(body)}</div>
  `;
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
    }
    for (const event of data.events || []) renderAgentEvent(event);
    renderSessionLabel(data.activeJobStatus);
    state.agentActive = data.activeJobStatus?.status === "queued" || data.activeJobStatus?.status === "running";
    setChatBusy(state.agentActive);
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
  setChatBusy(true);
  try {
    await fetchJson("/api/agent/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    await pollAgentHistory();
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
    state.selectedPost = data.post;
    state.posts = state.posts.map((item) =>
      item.pid === data.post.pid
        ? {
            ...item,
            ...data.post,
            comments: [],
            text: data.post.text?.slice(0, 180) || "",
          }
        : item,
    );
    renderPostList();
    renderDetail();
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
  state.selectedPid = null;
  state.selectedPost = null;
  reloadPostsDebounced();
}

nodes.minStarInput.addEventListener("input", applyThresholdFilters);
nodes.minReplyInput.addEventListener("input", applyThresholdFilters);

nodes.searchInput.addEventListener("input", (event) => {
  state.query = event.target.value;
  state.selectedPid = null;
  state.selectedPost = null;
  reloadPostsDebounced();
});

nodes.sortSelect.addEventListener("change", (event) => {
  state.sort = event.target.value;
  state.selectedPid = null;
  state.selectedPost = null;
  loadPosts({ reset: true, offset: 0 });
});

nodes.postList.addEventListener("scroll", () => {
  const nearBottom = nodes.postList.scrollTop + nodes.postList.clientHeight > nodes.postList.scrollHeight - 220;
  if (nearBottom && state.pageStart + state.posts.length < state.total) {
    loadPosts();
  }
});

nodes.prevPageButton.addEventListener("click", () => {
  const nextOffset = Math.max(0, state.pageStart - state.limit);
  state.selectedPid = null;
  state.selectedPost = null;
  loadPosts({ reset: true, offset: nextOffset });
});

nodes.nextPageButton.addEventListener("click", () => {
  const nextOffset = Math.min(Math.max(0, state.total - 1), state.pageStart + state.limit);
  state.selectedPid = null;
  state.selectedPost = null;
  loadPosts({ reset: true, offset: nextOffset });
});

nodes.loadMoreButton.addEventListener("click", () => {
  loadPosts();
});

nodes.reindexButton.addEventListener("click", async () => {
  setWorkbenchStatus("重建本地索引中", { timeout: 0 });
  try {
    const data = await fetchJson("/api/reindex", { method: "POST" });
    state.stats = data.stats;
    state.selectedPid = null;
    state.selectedPost = null;
    await loadPosts({ reset: true, offset: 0 });
    setWorkbenchStatus(`索引完成：${data.stats.posts} 帖`);
  } catch (error) {
    setWorkbenchStatus(`索引失败：${error.message}`, { timeout: 5200 });
  }
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
  document.documentElement.style.setProperty("--feed-width", `${clamp(width, 280, max)}px`);
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

setupColumnResizers();
renderSessionLabel();
updatePager();
startAgentPolling();
loadPosts({ reset: true, offset: 0 });
