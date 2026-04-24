"""
PKU Treehole RAG Agent

Main agent class implementing mode 2 only:
Auto keyword extraction with staged retrieval (multi-turn capable)
"""

import json
import os
import random
import re
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

import requests

from client import TreeholeClient
from utils import (
    format_posts_batch,
    save_json,
    load_json,
    get_cache_key,
    is_cache_valid,
    print_header,
    print_separator,
)

# Agent debug message prefix
AGENT_PREFIX = "[Agent] "

# Project root directory (stable regardless of current working directory)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_KNOWLEDGE_FILE = os.path.join(PROJECT_DIR, "agent.md")
TASK_MEMORY_DIR = os.path.join(PROJECT_DIR, "data", "task_memory")

# Conversation history persistence file
CONVERSATION_FILE = os.path.join(PROJECT_DIR, "data", "conversation_history.json")

try:
    import config_private as _cfg
except ImportError:
    import config as _cfg

USERNAME = _cfg.USERNAME
PASSWORD = _cfg.PASSWORD
LLM_API_KEY = _cfg.LLM_API_KEY
LLM_API_BASE = _cfg.LLM_API_BASE
LLM_MODEL = _cfg.LLM_MODEL
LLM_CHAT_COMPLETIONS_PATH = getattr(_cfg, "LLM_CHAT_COMPLETIONS_PATH", "/chat/completions")
LLM_EXTRA_HEADERS = getattr(_cfg, "LLM_EXTRA_HEADERS", {})
LLM_EXTRA_BODY = getattr(_cfg, "LLM_EXTRA_BODY", {})
LLM_ENABLE_PARALLEL_TOOL_CALLS = getattr(_cfg, "LLM_ENABLE_PARALLEL_TOOL_CALLS", True)
LLM_REASONING_EFFORT = getattr(_cfg, "LLM_REASONING_EFFORT", None)
LLM_THINKING_TYPE = getattr(_cfg, "LLM_THINKING_TYPE", "auto")

MAX_SEARCH_RESULTS = _cfg.MAX_SEARCH_RESULTS
MAX_CONTEXT_POSTS = _cfg.MAX_CONTEXT_POSTS
MAX_COMMENTS_PER_POST = _cfg.MAX_COMMENTS_PER_POST

TEMPERATURE = _cfg.TEMPERATURE
MAX_RESPONSE_TOKENS = _cfg.MAX_RESPONSE_TOKENS

SEARCH_DELAY_MIN = _cfg.SEARCH_DELAY_MIN
SEARCH_DELAY_MAX = _cfg.SEARCH_DELAY_MAX

ENABLE_CACHE = _cfg.ENABLE_CACHE
CACHE_DIR = _cfg.CACHE_DIR
CACHE_EXPIRATION = _cfg.CACHE_EXPIRATION

# Optional settings with backward-compatible defaults
SEARCH_PAGE_LIMIT = getattr(_cfg, "SEARCH_PAGE_LIMIT", 30)
SEARCH_COMMENT_LIMIT = getattr(_cfg, "SEARCH_COMMENT_LIMIT", 10)
INCLUDE_IMAGE_POSTS = getattr(_cfg, "INCLUDE_IMAGE_POSTS", True)
MAX_COMMENT_FETCH_POSTS = getattr(_cfg, "MAX_COMMENT_FETCH_POSTS", 6)
COMMENT_FETCH_MAX_PARALLEL = getattr(_cfg, "COMMENT_FETCH_MAX_PARALLEL", 10)
COMMENT_FETCH_MAX_REQUESTS_PER_SECOND = getattr(_cfg, "COMMENT_FETCH_MAX_REQUESTS_PER_SECOND", 20.0)
LLM_RETRY_MAX_ATTEMPTS = getattr(_cfg, "LLM_RETRY_MAX_ATTEMPTS", 5)
LLM_RETRY_SLEEP_SECONDS = getattr(_cfg, "LLM_RETRY_SLEEP_SECONDS", 5)

# Two-stage auto-search planning
BROAD_SEARCH_MIN = getattr(_cfg, "BROAD_SEARCH_MIN", 10)
BROAD_SEARCH_MAX = getattr(_cfg, "BROAD_SEARCH_MAX", 20)
FOCUSED_SEARCH_MIN = getattr(_cfg, "FOCUSED_SEARCH_MIN", 5)
FOCUSED_SEARCH_MAX = getattr(_cfg, "FOCUSED_SEARCH_MAX", 10)

# Normalize cache path to an absolute path under project root when configured as relative.
if not os.path.isabs(CACHE_DIR):
    CACHE_DIR = os.path.join(PROJECT_DIR, CACHE_DIR)


class _TokenBucket:
    """Thread-safe token bucket for request rate limiting."""

    def __init__(self, refill_rate: float, capacity: Optional[float] = None):
        self.refill_rate = max(0.0, float(refill_rate))
        self.capacity = max(1.0, float(capacity if capacity is not None else refill_rate or 1.0))
        self.tokens = self.capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Acquire one token, waiting when necessary."""
        if self.refill_rate <= 0:
            return

        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                    self.last_refill = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

                wait_s = (1.0 - self.tokens) / self.refill_rate

            time.sleep(min(wait_s, 0.05))


class TreeholeRAGAgent:
    """
    RAG Agent for PKU Treehole with an OpenAI-compatible LLM backend.
    Supports staged automatic keyword-based retrieval.
    """

    # ------------------------------------------------------------------ #
    #                        Search tool definition                       #
    # ------------------------------------------------------------------ #
    SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "search_treehole",
            "description": "在北大树洞中搜索相关帖子。如果当前信息不足以回答问题，可以使用不同的关键词多次调用此函数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，精准的1-2个词，不要包含多个概念"
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么需要搜索这个关键词（可选）"
                    }
                },
                "required": ["keyword"]
            }
        }
    }

    GET_POST_TOOL = {
        "type": "function",
        "function": {
            "name": "get_post_by_pid",
            "description": "通过帖子 PID 直接获取单帖详情，可选补拉该帖评论。适合在已知 PID 时精确抓取，减少无效搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "integer",
                        "description": "帖子 PID"
                    },
                    "include_comments": {
                        "type": "boolean",
                        "description": "是否拉取评论"
                    },
                    "max_comments": {
                        "type": "integer",
                        "description": "最多拉取评论条数，-1 表示尽量全量"
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么需要抓取该 PID（可选）"
                    }
                },
                "required": ["pid"]
            }
        }
    }

    GET_COMMENT_TOOL = {
        "type": "function",
        "function": {
            "name": "get_comments_by_pid",
            "description": "通过帖子 PID 直接获取评论。适用于已确定高价值帖子后的精确补拉。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "integer",
                        "description": "帖子 PID"
                    },
                    "max_comments": {
                        "type": "integer",
                        "description": "最多拉取评论条数，-1 表示尽量全量"
                    },
                    "sort": {
                        "type": "string",
                        "description": "排序方式，asc 或 desc"
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么要拉这条帖子的评论（可选）"
                    }
                },
                "required": ["pid"]
            }
        }
    }

    # System prompt shared by mode_auto_search and multi-turn conversation
    _AUTO_SEARCH_SYSTEM_PROMPT = (
        "你是一个北大树洞问答助手。你有三个工具：\n"
        "1) search_treehole: 关键词搜索帖子\n"
        "2) get_post_by_pid: 按 PID 精确抓取单帖\n"
        "3) get_comments_by_pid: 按 PID 精确抓取评论\n\n"
        "建议采用两阶段策略（软约束，可自行判断收敛）：\n"
        "阶段A（宽泛探索）: 建议 {broad_min}-{broad_max} 次搜索。"
        "目标是识别黑话/别称/简称（课程、院系、老师）并扩展关键词。\n"
        "阶段B（高质量聚焦）: 建议 {focused_min}-{focused_max} 次搜索。"
        "优先寻找高价值帖子，最重要指标是 reply 与 star，其次是时间与内容相关性。\n\n"
        "执行规则：\n"
        "- 已知 PID 时优先用 get_post_by_pid，减少冗余检索\n"
        "- 需要评论细节时用 get_comments_by_pid，避免在宽泛阶段过早拉评论\n"
        "- 优先在同一轮一次性返回多个 tool_calls（建议 2-4 个），减少模型往返次数\n"
        "- 仅当必须依赖上一批工具结果再决策时，才拆成下一轮调用\n"
        "- 若当前信息已足够支持结论，可提前结束当前阶段并进入下一阶段或直接总结\n"
        "- 搜索关键词优先 1-2 个核心词，必要时组合黑话\n"
        "- 严禁编造信息，只基于检索结果回答\n"
        "- 信息不足时明确说明\n"
        "- 保持客观，综合多条高质量帖子后再总结\n"
        "- 在回答前，先确保高价值帖子已补充评论信息"
    )

    def __init__(self, interactive=True, cookies_file=None):
        """Initialize the agent with Treehole client and an OpenAI-compatible LLM API.

        Args:
            interactive (bool): Whether to allow interactive prompts for login verification.
                              Set to False when running as a service.
            cookies_file (str): Path to user-specific cookies file. If None, uses default.
        """
        self.client = TreeholeClient(cookies_file=cookies_file)
        self.api_key = LLM_API_KEY
        self.api_base = LLM_API_BASE
        self.model = LLM_MODEL
        self.chat_completions_path = LLM_CHAT_COMPLETIONS_PATH
        self.extra_headers = dict(LLM_EXTRA_HEADERS) if isinstance(LLM_EXTRA_HEADERS, dict) else {}
        self.extra_body = dict(LLM_EXTRA_BODY) if isinstance(LLM_EXTRA_BODY, dict) else {}
        self.enable_parallel_tool_calls = bool(LLM_ENABLE_PARALLEL_TOOL_CALLS)
        self.reasoning_effort = LLM_REASONING_EFFORT
        self.thinking_type = LLM_THINKING_TYPE
        self._all_comments_cache: Dict[int, List[Dict[str, Any]]] = {}
        self.stream_callback = None  # Optional callback for streaming output
        self.info_callback = None    # Callback for progress/info messages

        # Multi-turn conversation state (mode 2)
        self._conversation_history: List[Dict[str, Any]] = []
        self._conversation_searched_posts: List[Dict[str, Any]] = []
        self._conversation_search_count: int = 0
        self._task_memory_file: Optional[str] = None
        self._agent_knowledge: str = ""
        self._last_memory_snapshot_hash: str = ""
        self._comment_rate_limiter = _TokenBucket(COMMENT_FETCH_MAX_REQUESTS_PER_SECOND)
        self._comment_metrics_lock = threading.Lock()
        self._comment_fetch_stats: Dict[str, Any] = self._new_comment_fetch_stats()
        self._post_cache: Dict[int, Dict[str, Any]] = {}

        # Load persistent agent knowledge and ensure per-task memory directory exists.
        os.makedirs(TASK_MEMORY_DIR, exist_ok=True)
        self._agent_knowledge = self._load_agent_knowledge()

        # Ensure login
        if not self.client.ensure_login(USERNAME, PASSWORD, interactive=interactive):
            raise RuntimeError("Failed to login to Treehole. Try running interactively first to save cookies.")

        # Create cache directory
        if ENABLE_CACHE:
            os.makedirs(CACHE_DIR, exist_ok=True)

        print(f"{AGENT_PREFIX}✓ 树洞 RAG Agent 初始化成功")

    # ------------------------------------------------------------------ #
    #                        Random delay helpers                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _human_delay(min_s: float, max_s: float, label: str = "") -> float:
        """Sleep for a random duration to simulate human browsing.

        Args:
            min_s: Minimum delay in seconds.
            max_s: Maximum delay in seconds.
            label: Optional label for debug output.

        Returns:
            The actual delay applied (seconds).
        """
        delay = random.uniform(min_s, max_s)
        if label:
            print(f"{AGENT_PREFIX}⏳ {label}: 等待 {delay:.2f}s")
        time.sleep(delay)
        return delay

    @staticmethod
    def _format_unix_timestamp(ts: Optional[int]) -> str:
        """Convert unix timestamp to readable local time string."""
        if not ts:
            return "unknown"
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "unknown"

    def _normalize_comment_metadata(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize comment metadata used by downstream formatting and LLM context."""
        ts = comment.get("timestamp")
        comment["reply_time"] = self._format_unix_timestamp(ts)
        return comment

    def _normalize_post_metadata(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize post metadata so all modes can consume stable fields."""
        ts = post.get("timestamp")
        post["post_time"] = self._format_unix_timestamp(ts)
        post["reply_count"] = post.get("reply", 0)
        post["star_count"] = post.get("likenum", 0)
        post["has_image"] = post.get("type") == "image" or bool(post.get("media_ids"))

        comments = post.get("comments") or post.get("comment_list") or []
        post["comments"] = [self._normalize_comment_metadata(c) for c in comments]
        return post

    @staticmethod
    def _resolve_search_comment_limit() -> int:
        """Resolve comment_limit sent to search API from config value."""
        if SEARCH_COMMENT_LIMIT == -1:
            return 10
        if SEARCH_COMMENT_LIMIT < 0:
            return 10
        return SEARCH_COMMENT_LIMIT

    def _load_agent_knowledge(self) -> str:
        """Load persistent agent knowledge from agent.md (if exists)."""
        if not os.path.exists(AGENT_KNOWLEDGE_FILE):
            return ""
        try:
            with open(AGENT_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                print(f"{AGENT_PREFIX}✓ 已加载 agent.md 经验库")
            return content
        except Exception as e:
            print(f"{AGENT_PREFIX}加载 agent.md 失败: {e}")
            return ""

    def _start_task_memory(self, user_question: str, mode: str) -> None:
        """Create a per-task markdown memory file for this run/conversation."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_{ts}.md"
        self._task_memory_file = os.path.join(TASK_MEMORY_DIR, filename)
        self._last_memory_snapshot_hash = ""
        header = (
            f"# Task Memory\n\n"
            f"- created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- mode: {mode}\n"
            f"- question: {user_question}\n\n"
            "## Notes\n"
        )
        try:
            with open(self._task_memory_file, "w", encoding="utf-8") as f:
                f.write(header)
            print(f"{AGENT_PREFIX}✓ 已创建任务 memory: {self._task_memory_file}")
        except Exception as e:
            print(f"{AGENT_PREFIX}创建任务 memory 失败: {e}")
            self._task_memory_file = None

    def _append_task_memory(self, line: str) -> None:
        """Append one line to per-task markdown memory."""
        if not self._task_memory_file:
            return
        try:
            with open(self._task_memory_file, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            pass

    def _read_task_memory(self, max_chars: int = 3500) -> str:
        """Read recent task memory snippet for prompt conditioning."""
        if not self._task_memory_file or not os.path.exists(self._task_memory_file):
            return ""
        try:
            with open(self._task_memory_file, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) <= max_chars:
                return content
            return content[-max_chars:]
        except Exception:
            return ""

    def _inject_task_memory_snapshot(self, messages: List[Dict[str, Any]]) -> None:
        """Inject latest task memory into messages if updated."""
        snapshot = self._read_task_memory(max_chars=2800)
        if not snapshot:
            return
        snapshot_hash = hashlib.md5(snapshot.encode("utf-8")).hexdigest()
        if snapshot_hash == self._last_memory_snapshot_hash:
            return

        messages.append({
            "role": "system",
            "content": (
                "以下是当前会话持续任务 memory（每轮更新，始终作为上下文）：\n"
                f"{snapshot}"
            ),
        })
        self._last_memory_snapshot_hash = snapshot_hash

    def _build_auto_search_system_prompt(self) -> str:
        """Build dynamic auto-search system prompt with agent/task memories."""
        base = self._AUTO_SEARCH_SYSTEM_PROMPT.format(
            broad_min=BROAD_SEARCH_MIN,
            broad_max=BROAD_SEARCH_MAX,
            focused_min=FOCUSED_SEARCH_MIN,
            focused_max=FOCUSED_SEARCH_MAX,
        )
        sections = [base]

        if self._agent_knowledge:
            sections.append("\n\n[agent.md 经验库]\n" + self._agent_knowledge)

        task_memory = self._read_task_memory()
        if task_memory:
            sections.append("\n\n[当前任务临时 memory]\n" + task_memory)

        return "\n".join(sections)

    def _upsert_posts(self, target_posts: List[Dict[str, Any]], posts: List[Dict[str, Any]]) -> None:
        """Insert/replace posts by pid in-place."""
        mapping = {p.get("pid"): p for p in target_posts if p.get("pid") is not None}
        for post in posts:
            pid = post.get("pid")
            if pid is None:
                continue
            mapping[pid] = post
        target_posts[:] = list(mapping.values())

    def get_post_by_pid(
        self,
        pid: int,
        include_comments: bool = False,
        max_comments: int = MAX_COMMENTS_PER_POST,
    ) -> Optional[Dict[str, Any]]:
        """Fetch one post by pid and optionally hydrate comments."""
        if pid in self._post_cache:
            cached_post = dict(self._post_cache[pid])
            if include_comments:
                cached_post["comments"] = self._load_comments_for_post(cached_post, max_comments)
            return cached_post

        try:
            result = self.client.get_post(pid)
            if not result.get("success"):
                return None
            post = self._normalize_post_metadata(result.get("data", {}))
            self._post_cache[pid] = dict(post)
            if include_comments:
                post["comments"] = self._load_comments_for_post(post, max_comments)
            return post
        except Exception as e:
            print(f"{AGENT_PREFIX}获取 PID {pid} 失败: {e}")
            return None

    def get_comments_by_pid(
        self,
        pid: int,
        max_comments: int = MAX_COMMENTS_PER_POST,
        sort: str = "asc",
    ) -> List[Dict[str, Any]]:
        """Fetch comments by pid directly with pagination and global rate limit."""
        if pid in self._all_comments_cache:
            cached = self._all_comments_cache[pid]
            if max_comments == -1:
                return cached
            return cached[:max_comments]

        comments: List[Dict[str, Any]] = []
        page = 1
        while True:
            self._comment_rate_limiter.acquire()
            self._record_comment_api_request()
            result = self.client.get_comment(pid, page=page, limit=100, sort=sort)
            if not result.get("success"):
                break

            page_data = result.get("data", {})
            page_comments = [
                self._normalize_comment_metadata(c)
                for c in page_data.get("data", [])
            ]
            comments.extend(page_comments)

            if max_comments != -1 and len(comments) >= max_comments:
                comments = comments[:max_comments]
                break

            last_page = page_data.get("last_page", page)
            if page >= last_page or not page_comments:
                break
            page += 1

        self._all_comments_cache[pid] = comments
        if max_comments == -1:
            return comments
        return comments[:max_comments]

    @staticmethod
    def _new_comment_fetch_stats() -> Dict[str, Any]:
        """Build an empty comment-fetch metrics snapshot."""
        return {
            "started_at": 0.0,
            "api_requests": 0,
            "cache_hits": 0,
            "selected_posts": 0,
            "completed_posts": 0,
            "failed_posts": 0,
            "request_timestamps": [],
        }

    def _reset_comment_fetch_stats(self, selected_posts: int) -> None:
        """Reset per-batch metrics before a hydration run."""
        with self._comment_metrics_lock:
            self._comment_fetch_stats = self._new_comment_fetch_stats()
            self._comment_fetch_stats["started_at"] = time.time()
            self._comment_fetch_stats["selected_posts"] = selected_posts

    def _inc_comment_stat(self, key: str, step: int = 1) -> None:
        with self._comment_metrics_lock:
            self._comment_fetch_stats[key] = self._comment_fetch_stats.get(key, 0) + step

    def _record_comment_api_request(self) -> None:
        """Record one outbound comment API request."""
        with self._comment_metrics_lock:
            self._comment_fetch_stats["api_requests"] += 1
            self._comment_fetch_stats["request_timestamps"].append(time.time())

    def _log_comment_fetch_stats(self) -> None:
        """Print concise request/concurrency metrics for one hydration batch."""
        with self._comment_metrics_lock:
            stats = dict(self._comment_fetch_stats)
            req_ts = list(stats.get("request_timestamps", []))

        started_at = float(stats.get("started_at") or time.time())
        elapsed = max(0.001, time.time() - started_at)
        api_requests = int(stats.get("api_requests", 0))
        selected_posts = int(stats.get("selected_posts", 0))
        completed_posts = int(stats.get("completed_posts", 0))
        failed_posts = int(stats.get("failed_posts", 0))
        cache_hits = int(stats.get("cache_hits", 0))

        if len(req_ts) > 1:
            request_span = max(0.001, req_ts[-1] - req_ts[0])
            submit_rate = (len(req_ts) - 1) / request_span
        else:
            submit_rate = len(req_ts) / elapsed

        msg = (
            "评论抓取统计: "
            f"候选 {selected_posts} 帖, 完成 {completed_posts}, 失败 {failed_posts}, "
            f"缓存命中 {cache_hits}, 请求 {api_requests}, "
            f"耗时 {elapsed:.2f}s, 平均发起速率 {submit_rate:.2f} req/s, "
            f"并发上限 {max(1, COMMENT_FETCH_MAX_PARALLEL)}, "
            f"速率上限 {COMMENT_FETCH_MAX_REQUESTS_PER_SECOND:.2f} req/s"
        )
        print(f"{AGENT_PREFIX}{msg}")

    @staticmethod
    def _compact_text_preview(text: str, max_len: int = 120) -> str:
        """Normalize and truncate text for compact ranking context."""
        cleaned = (text or "").replace("\n", " ").strip()
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[:max_len] + "..."

    def _format_search_brief(
        self,
        posts: List[Dict[str, Any]],
        max_items: int = 15,
        include_comment_preview: bool = False,
        preview_comments_per_post: int = 3,
    ) -> str:
        """Build lightweight search summary for tool feedback.

        When include_comment_preview is True, comments are shown from search response
        (comment_list/comments) only, without extra comment API calls.
        """
        if not posts:
            return "无结果"
        lines = []
        for post in posts[:max_items]:
            pid = post.get("pid", "unknown")
            reply_count = post.get("reply_count", post.get("reply", 0))
            star_count = post.get("star_count", post.get("likenum", 0))
            post_time = post.get("post_time", "unknown")
            text_preview = self._compact_text_preview(post.get("text", ""), max_len=90)
            lines.append(
                f"- pid={pid} | reply={reply_count} | star={star_count} | time={post_time} | {text_preview}"
            )
            if include_comment_preview:
                comments = post.get("comments") or post.get("comment_list") or []
                for i, c in enumerate(comments[:max(0, preview_comments_per_post)], 1):
                    c_preview = self._compact_text_preview(c.get("text", ""), max_len=70)
                    lines.append(f"  - 评论预览{i}: {c_preview}")
        if len(posts) > max_items:
            lines.append(f"- ... 还有 {len(posts) - max_items} 条结果未展示")
        return "\n".join(lines)

    @staticmethod
    def _parse_selected_pids(raw: str, allowed_pids: Set[int], max_selected: int) -> List[int]:
        """Parse LLM-selected pid list from JSON-like output."""
        if not raw:
            return []

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        candidates: List[int] = []
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                pids = payload.get("selected_pids", [])
            elif isinstance(payload, list):
                pids = payload
            else:
                pids = []
            for pid in pids:
                if isinstance(pid, int):
                    candidates.append(pid)
                elif isinstance(pid, str) and pid.isdigit():
                    candidates.append(int(pid))
        except Exception:
            for token in re.findall(r"\d+", raw):
                try:
                    candidates.append(int(token))
                except Exception:
                    continue

        result: List[int] = []
        seen: Set[int] = set()
        for pid in candidates:
            if pid in allowed_pids and pid not in seen:
                seen.add(pid)
                result.append(pid)
            if len(result) >= max_selected:
                break
        return result

    def _heuristic_select_posts_for_comments(
        self,
        posts: List[Dict[str, Any]],
        query: str,
        max_selected: int,
    ) -> List[int]:
        """Fallback ranking when LLM selection fails."""
        query_tokens = [tok.lower() for tok in re.split(r"\s+", (query or "").strip()) if tok]

        scored: List[Any] = []
        for post in posts:
            pid = post.get("pid")
            if not pid:
                continue
            reply_count = max(0, int(post.get("reply_count", post.get("reply", 0)) or 0))
            star_count = max(0, int(post.get("star_count", post.get("likenum", 0)) or 0))
            comment_total = max(0, int(post.get("comment_total", 0) or 0))
            text = (post.get("text") or "").lower()
            relevance = 0.0
            if query_tokens:
                hit = sum(1 for tok in query_tokens if tok and tok in text)
                relevance = hit / len(query_tokens)
            score = relevance * 4.0 + reply_count * 0.35 + star_count * 0.55 + comment_total * 0.25
            scored.append((score, pid))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [pid for _, pid in scored[:max_selected]]

    def _select_posts_for_comment_fetch(
        self,
        posts: List[Dict[str, Any]],
        query: str,
        max_selected: int,
    ) -> List[int]:
        """Ask LLM to pick high-value posts for deeper comment fetching."""
        if max_selected <= 0:
            return []

        valid_posts = [post for post in posts if post.get("pid")]
        if len(valid_posts) <= max_selected:
            return [int(post["pid"]) for post in valid_posts]

        allowed_pids = {int(post["pid"]) for post in valid_posts}
        compact_posts = [
            {
                "pid": int(post["pid"]),
                "reply_count": int(post.get("reply_count", post.get("reply", 0)) or 0),
                "star_count": int(post.get("star_count", post.get("likenum", 0)) or 0),
                "comment_total": int(post.get("comment_total", 0) or 0),
                "post_time": post.get("post_time", "unknown"),
                "text_preview": self._compact_text_preview(post.get("text", ""), max_len=110),
            }
            for post in valid_posts
        ]

        system_message = (
            "你是树洞检索优化助手。你需要从候选帖子中挑选最值得深入翻评论的帖子。"
            "优先考虑与查询相关、reply_count高、star_count高、comment_total高的帖子。"
            "尽量避免选择明显低价值帖子。"
            "输出严格 JSON：{\"selected_pids\":[...],\"reason\":\"...\"}。"
        )
        user_message = (
            f"用户问题/查询：{query or '无'}\n"
            f"最多可选帖子数：{max_selected}\n"
            "候选帖子（JSON）：\n"
            f"{json.dumps(compact_posts, ensure_ascii=False)}\n\n"
            "请只返回 JSON，不要输出额外文字。"
        )

        raw = self.call_llm(
            user_message=user_message,
            system_message=system_message,
            temperature=0.0,
            stream=False,
        )

        selected = self._parse_selected_pids(raw, allowed_pids, max_selected)
        if selected:
            return selected

        return self._heuristic_select_posts_for_comments(valid_posts, query, max_selected)

    def _load_comments_for_post(self, post: Dict[str, Any], max_comments: int) -> List[Dict[str, Any]]:
        """Load comments for a post with configurable cap (-1 for all)."""
        pid = post.get("pid")
        existing_comments = [
            self._normalize_comment_metadata(c)
            for c in (post.get("comments") or post.get("comment_list") or [])
        ]

        if not pid:
            if max_comments == -1:
                return existing_comments
            return existing_comments[:max_comments] if max_comments >= 0 else existing_comments

        comment_total = post.get("comment_total", len(existing_comments))

        if max_comments == 0:
            return []

        cached_comments = self._all_comments_cache.get(pid)
        if cached_comments:
            self._inc_comment_stat("cache_hits")
            if max_comments == -1:
                if comment_total and len(cached_comments) < comment_total:
                    pass
                else:
                    return cached_comments
            elif len(cached_comments) >= max_comments:
                return cached_comments[:max_comments]

        if max_comments != -1 and len(existing_comments) >= max_comments:
            return existing_comments[:max_comments]

        if max_comments == -1 and comment_total and len(existing_comments) >= comment_total:
            self._all_comments_cache[pid] = existing_comments
            return existing_comments

        target_count = comment_total if max_comments == -1 else max_comments
        if target_count <= 0 and max_comments == -1:
            return existing_comments

        if max_comments == -1:
            print(f"{AGENT_PREFIX}  正在获取帖子 #{pid} 的全部 {comment_total} 条评论...")
        else:
            print(f"{AGENT_PREFIX}  正在获取帖子 #{pid} 的评论（目标 {target_count} 条）...")

        all_comments: List[Dict[str, Any]] = []
        page = 1
        while True:
            self._comment_rate_limiter.acquire()
            self._record_comment_api_request()
            page_result = self.client.get_comment(pid, page=page, limit=100)
            if not page_result.get("success"):
                break

            page_data = page_result.get("data", {})
            page_comments = [
                self._normalize_comment_metadata(c)
                for c in page_data.get("data", [])
            ]
            all_comments.extend(page_comments)

            if max_comments != -1 and len(all_comments) >= max_comments:
                all_comments = all_comments[:max_comments]
                break

            last_page = page_data.get("last_page", page)
            if page >= last_page or not page_comments:
                break
            page += 1

        if not all_comments:
            all_comments = existing_comments

        self._all_comments_cache[pid] = all_comments
        if max_comments == -1:
            return all_comments
        return all_comments[:max_comments]

    def _hydrate_posts_for_context(
        self,
        posts: List[Dict[str, Any]],
        max_comments: int,
        selection_query: str = "",
        context_post_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Hydrate context with AI-selected, rate-limited comment fetching.

        Selection is run on all candidate posts first, then context_post_limit is applied.
        """
        normalized_posts = [self._normalize_post_metadata(post) for post in posts]

        if max_comments == 0:
            for post in normalized_posts:
                post["comments"] = []
            return normalized_posts

        selected_limit = max(0, min(MAX_COMMENT_FETCH_POSTS, len(normalized_posts)))
        selected_pids = self._select_posts_for_comment_fetch(
            normalized_posts,
            query=selection_query,
            max_selected=selected_limit,
        )
        selected_set = set(selected_pids)

        # Reorder so selected posts are prioritized into limited context.
        selected_posts = [p for p in normalized_posts if p.get("pid") in selected_set]
        unselected_posts = [p for p in normalized_posts if p.get("pid") not in selected_set]
        ordered_posts = selected_posts + unselected_posts

        if context_post_limit is not None and context_post_limit > 0:
            context_posts = ordered_posts[:context_post_limit]
        else:
            context_posts = ordered_posts

        effective_selected_set = {p.get("pid") for p in context_posts if p.get("pid") in selected_set}

        print(
            f"{AGENT_PREFIX}评论补拉策略: 候选 {len(normalized_posts)} 帖, "
            f"AI 选择 {len(selected_pids)} 帖补拉评论, "
            f"上下文使用 {len(context_posts)} 帖"
        )
        if selected_pids:
            print(f"{AGENT_PREFIX}AI 选中帖子 PID: {', '.join(str(pid) for pid in selected_pids)}")

        comments_by_pid: Dict[int, List[Dict[str, Any]]] = {}
        if effective_selected_set:
            self._reset_comment_fetch_stats(len(effective_selected_set))
            max_workers = max(1, min(COMMENT_FETCH_MAX_PARALLEL, len(effective_selected_set)))

            def fetch_for_post(p: Dict[str, Any]) -> Any:
                pid = int(p.get("pid"))
                comments = self._load_comments_for_post(p, max_comments)
                return pid, comments

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pid = {
                    executor.submit(fetch_for_post, post): int(post["pid"])
                    for post in context_posts
                    if post.get("pid") in effective_selected_set
                }
                for future in as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    try:
                        _, comments = future.result()
                        comments_by_pid[pid] = comments
                        self._inc_comment_stat("completed_posts")
                    except Exception as e:
                        self._inc_comment_stat("failed_posts")
                        print(f"{AGENT_PREFIX}  警告: 获取帖子 {pid} 的评论失败: {e}")

            self._log_comment_fetch_stats()

        hydrated: List[Dict[str, Any]] = []
        for post in context_posts:
            pid = post.get("pid")
            if pid in comments_by_pid:
                post["comments"] = comments_by_pid[pid]
            else:
                base_comments = post.get("comments") or post.get("comment_list") or []
                if max_comments == -1:
                    post["comments"] = base_comments
                else:
                    post["comments"] = base_comments[:max_comments]
            hydrated.append(post)

        return hydrated

    # ------------------------------------------------------------------ #
    #                        Treehole search                              #
    # ------------------------------------------------------------------ #

    def search_treehole(
        self,
        keyword: str,
        max_results: int = MAX_SEARCH_RESULTS,
        use_cache: bool = ENABLE_CACHE
    ) -> List[Dict[str, Any]]:
        """
        Search Treehole for posts matching keyword.
        """
        # Check cache first
        if use_cache:
            cache_key = get_cache_key(
                f"{keyword}|{max_results}|{SEARCH_PAGE_LIMIT}|{SEARCH_COMMENT_LIMIT}|{INCLUDE_IMAGE_POSTS}"
            )
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

            if is_cache_valid(cache_file, CACHE_EXPIRATION):
                print(f"{AGENT_PREFIX}使用缓存结果: {keyword}")
                cached = load_json(cache_file) or []
                return [self._normalize_post_metadata(post) for post in cached]

        print(f"{AGENT_PREFIX}正在搜索树洞: {keyword}")

        try:
            all_posts: List[Dict[str, Any]] = []
            page = 1
            search_comment_limit = self._resolve_search_comment_limit()
            page_limit = max(1, SEARCH_PAGE_LIMIT)

            while len(all_posts) < max_results:
                request_limit = min(page_limit, max_results - len(all_posts))
                self._human_delay(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX, f"搜索请求 第{page}页")

                search_result = self.client.search_posts(
                    keyword,
                    page=page,
                    limit=request_limit,
                    comment_limit=search_comment_limit,
                )

                if not search_result.get("success"):
                    msg = f"搜索失败: {search_result.get('message', '未知错误')}"
                    print(f"{AGENT_PREFIX}{msg}")
                    if self.info_callback:
                        self.info_callback(msg)
                    break

                page_posts = search_result.get("data", {}).get("data", [])
                if not page_posts:
                    break

                normalized_posts = [self._normalize_post_metadata(post) for post in page_posts]
                if not INCLUDE_IMAGE_POSTS:
                    normalized_posts = [post for post in normalized_posts if not post.get("has_image")]

                all_posts.extend(normalized_posts)

                last_page = search_result.get("data", {}).get("last_page", page)
                if page >= last_page or len(page_posts) < request_limit:
                    break
                page += 1

            enriched_posts = all_posts[:max_results]

            if use_cache:
                save_json(enriched_posts, cache_file)

            msg = f"✓ 找到 {len(enriched_posts)} 个帖子"
            print(f"{AGENT_PREFIX}{msg}")
            if self.info_callback:
                self.info_callback(msg)
            return enriched_posts

        except Exception as e:
            print(f"{AGENT_PREFIX}搜索树洞时出错: {e}")
            return []

    # ------------------------------------------------------------------ #
    #                  OpenAI-compatible LLM API calls                    #
    # ------------------------------------------------------------------ #

    def _get_chat_completions_url(self) -> str:
        """Build the chat completions endpoint URL for the configured backend."""
        base = str(self.api_base or "").rstrip("/")
        path = str(self.chat_completions_path or "/chat/completions").strip()
        if not base:
            raise ValueError("LLM_API_BASE is empty")
        if base.endswith("/chat/completions"):
            return base
        if not path:
            path = "/chat/completions"
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    def _build_llm_headers(self) -> Dict[str, str]:
        """Build request headers for the configured backend."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for key, value in self.extra_headers.items():
            headers[str(key)] = str(value)
        return headers

    def _is_deepseek_v4_model(self) -> bool:
        """Return True when the configured model looks like a DeepSeek V4 chat model."""
        model_name = str(self.model or "").lower()
        return "deepseek-v4" in model_name

    def _normalize_reasoning_effort(self) -> Optional[str]:
        """Normalize configured reasoning effort to a lowercase string."""
        if not isinstance(self.reasoning_effort, str):
            return None
        normalized = self.reasoning_effort.strip().lower()
        return normalized or None

    def _build_llm_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        stream: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a chat-completions payload with optional provider-specific extras."""
        data: Dict[str, Any] = dict(self.extra_body)
        data.update({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": MAX_RESPONSE_TOKENS,
            "stream": stream,
        })
        if tools:
            data["tools"] = tools
            data["tool_choice"] = "auto"

        reasoning_effort = self._normalize_reasoning_effort()
        thinking_type = None
        if isinstance(self.thinking_type, str):
            candidate = self.thinking_type.strip().lower()
            if candidate in {"enabled", "disabled"}:
                thinking_type = candidate

        if self._is_deepseek_v4_model():
            # DeepSeek V4 uses OpenAI-compatible reasoning_effort, but its effective
            # values are narrowed to high/max. Keep a single config surface and map.
            mapped_effort = None
            if reasoning_effort not in {None, "auto"}:
                if reasoning_effort in {"none", "off", "disabled"}:
                    thinking_type = "disabled"
                elif reasoning_effort in {"xhigh", "max"}:
                    mapped_effort = "max"
                    if thinking_type is None:
                        thinking_type = "enabled"
                else:
                    mapped_effort = "high"
                    if thinking_type is None:
                        thinking_type = "enabled"

            if thinking_type in {"enabled", "disabled"}:
                data["thinking"] = {"type": thinking_type}
            if mapped_effort and thinking_type != "disabled":
                data["reasoning_effort"] = mapped_effort
        else:
            if reasoning_effort and reasoning_effort != "auto":
                data["reasoning_effort"] = reasoning_effort
        return data

    def _extract_llm_error_detail(self, error: Exception) -> str:
        """Extract readable error details from the configured LLM backend."""
        if isinstance(error, requests.exceptions.HTTPError) and getattr(error, "response", None) is not None:
            try:
                payload = error.response.json()
                if isinstance(payload, dict):
                    nested = payload.get("error")
                    if isinstance(nested, dict) and nested.get("message"):
                        return str(nested.get("message"))
                    if payload.get("message"):
                        return str(payload.get("message"))
                return error.response.text
            except Exception:
                return str(error)
        return str(error)

    def _is_retryable_llm_error(self, error: Exception) -> bool:
        """Return True for transient errors that should be retried."""
        if isinstance(error, requests.exceptions.Timeout):
            return True
        if isinstance(error, requests.exceptions.ConnectionError):
            return True
        if isinstance(error, requests.exceptions.HTTPError):
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600):
                return True

        error_text = self._extract_llm_error_detail(error).lower()
        return (
            "rate limit" in error_text
            or "too many requests" in error_text
            or "429" in error_text
        )

    def _wait_before_llm_retry(self, attempt: int, error: Exception) -> None:
        detail = self._extract_llm_error_detail(error)
        print(
            f"{AGENT_PREFIX}错误: LLM API 调用失败 - {detail}"
            f"（第{attempt}/{LLM_RETRY_MAX_ATTEMPTS}次）"
        )
        print(f"{AGENT_PREFIX}等待 {LLM_RETRY_SLEEP_SECONDS} 秒后重试...")
        time.sleep(LLM_RETRY_SLEEP_SECONDS)

    def call_llm(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = TEMPERATURE,
        stream: bool = True,
        callback: Optional[callable] = None
    ) -> str:
        """Call the configured OpenAI-compatible LLM API for chat completion."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        headers = self._build_llm_headers()
        data = self._build_llm_payload(messages, temperature=temperature, stream=stream)
        url = self._get_chat_completions_url()

        for attempt in range(1, LLM_RETRY_MAX_ATTEMPTS + 1):
            try:
                if stream:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=120,
                        stream=True,
                    )
                    response.raise_for_status()

                    full_content = ""
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            cb = self.stream_callback or callback
                                            if cb:
                                                cb(content)
                                            else:
                                                print(content, end='', flush=True)
                                            full_content += content
                                except json.JSONDecodeError:
                                    continue

                    if not (self.stream_callback or callback):
                        print()
                    return full_content
                else:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < LLM_RETRY_MAX_ATTEMPTS and self._is_retryable_llm_error(e):
                    self._wait_before_llm_retry(attempt, e)
                    continue

                detail = self._extract_llm_error_detail(e)
                print(f"{AGENT_PREFIX}调用 LLM API 时出错: {detail}")
                return f"抱歉，调用 LLM API 时出错: {detail}"

    @staticmethod
    def _extract_reasoning_content(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract reasoning content from a response payload or streaming delta."""
        if not isinstance(payload, dict):
            return None
        reasoning = payload.get("reasoning_content")
        if reasoning is None:
            reasoning = payload.get("reasoning")
        return reasoning

    @staticmethod
    def _build_assistant_message(
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        reasoning_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an assistant message while preserving reasoning/tool-call context."""
        message: Dict[str, Any] = {"role": "assistant"}
        if content is not None:
            message["content"] = content
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        return message

    def _call_llm_with_tools(self, messages: List[Dict], tools: List[Dict], stream: bool = False) -> Dict:
        """Call the configured OpenAI-compatible LLM API with tool support."""
        headers = self._build_llm_headers()
        data = self._build_llm_payload(messages, temperature=TEMPERATURE, stream=stream, tools=tools)
        url = self._get_chat_completions_url()
        use_parallel_tool_calls = bool(tools) and self.enable_parallel_tool_calls

        for attempt in range(1, LLM_RETRY_MAX_ATTEMPTS + 1):
            try:
                req_data = dict(data)
                if use_parallel_tool_calls:
                    # Hint model/runtime to emit multiple tool calls in one assistant turn.
                    req_data["parallel_tool_calls"] = True

                if stream:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=req_data,
                        timeout=120,
                        stream=True,
                    )
                    response.raise_for_status()

                    full_content = ""
                    full_reasoning_content = ""
                    tool_calls_raw = None

                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        reasoning_content = self._extract_reasoning_content(delta)
                                        if reasoning_content:
                                            full_reasoning_content += reasoning_content
                                        if delta.get('tool_calls') and tool_calls_raw is None:
                                            tool_calls_raw = delta['tool_calls']
                                        content = delta.get('content', '')
                                        if content:
                                            if self.stream_callback:
                                                self.stream_callback(content)
                                            else:
                                                print(content, end='', flush=True)
                                            full_content += content
                                except json.JSONDecodeError:
                                    continue

                    if tool_calls_raw:
                        return {
                            "content": full_content,
                            "tool_calls": tool_calls_raw,
                            "reasoning_content": full_reasoning_content,
                        }
                    return {
                        "content": full_content,
                        "tool_calls": None,
                        "reasoning_content": full_reasoning_content,
                    }
                else:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=req_data,
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()
                    choice = result["choices"][0]
                    message = choice["message"]
                    return {
                        "content": message.get("content"),
                        "tool_calls": message.get("tool_calls"),
                        "reasoning_content": self._extract_reasoning_content(message),
                    }
            except Exception as e:
                detail_lower = self._extract_llm_error_detail(e).lower()
                if (
                    use_parallel_tool_calls
                    and "parallel_tool_calls" in detail_lower
                    and any(k in detail_lower for k in ["unknown", "unsupported", "invalid", "extra", "not allowed"])
                ):
                    use_parallel_tool_calls = False
                    continue

                if attempt < LLM_RETRY_MAX_ATTEMPTS and self._is_retryable_llm_error(e):
                    self._wait_before_llm_retry(attempt, e)
                    continue

                error_detail = self._extract_llm_error_detail(e)
                print(f"{AGENT_PREFIX}错误: LLM API 调用失败 - {error_detail}")
                return {"content": None, "tool_calls": None, "reasoning_content": None}

    # ------------------------------------------------------------------ #
    #         Internal: execute one round of search-then-answer           #
    # ------------------------------------------------------------------ #

    def _execute_search_loop(
        self,
        messages: List[Dict],
        all_searched_posts: List[Dict],
        search_history: List[Dict],
        search_count: int,
        phase_max: int,
        phase_name: str,
        phase_posts: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Run one staged search loop with min/max search counts for this phase."""

        phase_used = 0
        broad_preview_used = False

        while phase_used < phase_max:
            self._inject_task_memory_snapshot(messages)
            response = self._call_llm_with_tools(
                messages,
                [self.SEARCH_TOOL, self.GET_POST_TOOL, self.GET_COMMENT_TOOL],
                stream=False,
            )

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                break

            messages.append(
                self._build_assistant_message(
                    content=response.get("content"),
                    tool_calls=tool_calls,
                    reasoning_content=response.get("reasoning_content"),
                )
            )

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"].get("arguments", "{}") or "{}")

                if name == "search_treehole":
                    search_count += 1
                    phase_used += 1
                    keyword = args.get("keyword", "")
                    reason = args.get("reason", "")

                    search_history.append({
                        "iteration": search_count,
                        "phase": phase_name,
                        "action": "search",
                        "keyword": keyword,
                        "reason": reason,
                    })

                    self._append_task_memory(
                        f"- [{phase_name}] search#{search_count}: keyword={keyword} reason={reason or '-'}"
                    )

                    temp_callback = self.info_callback
                    self.info_callback = None
                    posts = self.search_treehole(keyword, max_results=MAX_SEARCH_RESULTS)
                    self._upsert_posts(all_searched_posts, posts)
                    if phase_posts is not None:
                        self._upsert_posts(phase_posts, posts)
                    self.info_callback = temp_callback

                    search_msg = f"[{phase_name}] 第{search_count}次搜索 关键词: {keyword}"
                    if reason:
                        search_msg += f"\n搜索原因: {reason}"
                    search_msg += f"\n✓ 找到 {len(posts)} 个帖子"
                    print(f"\n{AGENT_PREFIX}{search_msg}")
                    if self.info_callback:
                        self.info_callback(search_msg)

                    if posts:
                        include_preview = (phase_name.startswith("阶段A") and not broad_preview_used)
                        brief = self._format_search_brief(
                            posts,
                            max_items=min(15, MAX_CONTEXT_POSTS),
                            include_comment_preview=include_preview,
                            preview_comments_per_post=3,
                        )
                        if include_preview:
                            broad_preview_used = True
                            self._append_task_memory("- 阶段A首次搜索已展示每帖最多3条评论预览（来自search响应）")
                            result_summary = (
                                f"搜索到 {len(posts)} 个帖子。轻量摘要（含一次评论预览）如下：\n{brief}"
                            )
                        else:
                            result_summary = f"搜索到 {len(posts)} 个帖子。轻量摘要如下：\n{brief}"
                    else:
                        result_summary = f"未找到关于「{keyword}」的相关帖子。"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "search_treehole",
                        "content": result_summary,
                    })

                    self._human_delay(0.2, 0.8, "工具调用间隔")

                elif name == "get_post_by_pid":
                    pid = int(args.get("pid", 0) or 0)
                    include_comments = bool(args.get("include_comments", False))
                    max_comments = int(args.get("max_comments", MAX_COMMENTS_PER_POST) or MAX_COMMENTS_PER_POST)
                    reason = args.get("reason", "")

                    search_history.append({
                        "iteration": search_count,
                        "phase": phase_name,
                        "action": "get_post",
                        "pid": pid,
                        "reason": reason,
                    })
                    self._append_task_memory(
                        f"- [{phase_name}] get_post: pid={pid} include_comments={include_comments} max_comments={max_comments}"
                    )

                    post = self.get_post_by_pid(pid, include_comments=include_comments, max_comments=max_comments)
                    if post:
                        self._upsert_posts(all_searched_posts, [post])
                        if phase_posts is not None:
                            self._upsert_posts(phase_posts, [post])
                        post_text = format_posts_batch(
                            [post],
                            include_comments=True,
                            max_comments=max_comments if include_comments else 0,
                        )
                        result_summary = f"已获取帖子 #{pid}：\n\n{post_text}"
                    else:
                        result_summary = f"未能获取帖子 #{pid}。"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "get_post_by_pid",
                        "content": result_summary,
                    })

                elif name == "get_comments_by_pid":
                    pid = int(args.get("pid", 0) or 0)
                    max_comments = int(args.get("max_comments", MAX_COMMENTS_PER_POST) or MAX_COMMENTS_PER_POST)
                    sort = str(args.get("sort", "asc") or "asc").lower()
                    reason = args.get("reason", "")

                    search_history.append({
                        "iteration": search_count,
                        "phase": phase_name,
                        "action": "get_comment",
                        "pid": pid,
                        "max_comments": max_comments,
                        "reason": reason,
                    })
                    self._append_task_memory(
                        f"- [{phase_name}] get_comment: pid={pid} max_comments={max_comments} sort={sort}"
                    )

                    comments = self.get_comments_by_pid(pid, max_comments=max_comments, sort=sort)

                    # Merge comments into post cache if post already exists in working set.
                    for post in all_searched_posts:
                        if post.get("pid") == pid:
                            post["comments"] = comments
                            break

                    preview_lines = []
                    for i, c in enumerate(comments[:10], 1):
                        ctext = self._compact_text_preview(c.get("text", ""), max_len=80)
                        preview_lines.append(f"{i}. {ctext}")
                    preview = "\n".join(preview_lines) if preview_lines else "无评论或拉取失败"
                    result_summary = (
                        f"已获取帖子 #{pid} 评论，共 {len(comments)} 条（展示前10条）：\n{preview}"
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "get_comments_by_pid",
                        "content": result_summary,
                    })

            if phase_used >= phase_max:
                break

        return search_count

    # ------------------------------------------------------------------ #
    #              Mode 2: Auto search (single-shot, legacy)              #
    # ------------------------------------------------------------------ #

    def mode_auto_search(self, user_question: str) -> Dict[str, Any]:
        """Mode 2 (single-shot): Automatic keyword extraction with iterative search."""
        print_header("模式 2: 智能自动检索（支持多轮搜索）")

        self._start_task_memory(user_question, mode="auto_single")
        self._append_task_memory("- strategy: 阶段A宽泛探索 + 阶段B高质量聚焦")

        messages = [
            {
                "role": "system",
                "content": self._build_auto_search_system_prompt(),
            },
            {"role": "user", "content": f"用户问题：{user_question}"},
        ]

        all_searched_posts: List[Dict[str, Any]] = []
        stage_b_posts: List[Dict[str, Any]] = []
        search_history: List[Dict] = []

        msg = "✓ LLM 开始分析问题..."
        print(f"{AGENT_PREFIX}{msg}")
        if self.info_callback:
            self.info_callback(msg)

        messages.append({
            "role": "user",
            "content": (
                f"请执行阶段A（宽泛探索）：建议 {BROAD_SEARCH_MIN}-{BROAD_SEARCH_MAX} 次。"
                "重点识别院系/老师/课程/等的黑话、简称、别称；若信息已收敛可提前结束。"
            ),
        })
        search_count = self._execute_search_loop(
            messages,
            all_searched_posts,
            search_history,
            0,
            phase_max=BROAD_SEARCH_MAX,
            phase_name="阶段A-宽泛探索",
            phase_posts=None,
        )

        messages.append({
            "role": "user",
            "content": (
                f"请执行阶段B（高质量聚焦）：建议 {FOCUSED_SEARCH_MIN}-{FOCUSED_SEARCH_MAX} 次。"
                "优先筛选高 reply / 高 star 的帖子，再补充时间与内容质量；若证据已充分可提前结束。"
            ),
        })
        search_count = self._execute_search_loop(
            messages,
            all_searched_posts,
            search_history,
            search_count,
            phase_max=FOCUSED_SEARCH_MAX,
            phase_name="阶段B-高质量聚焦",
            phase_posts=stage_b_posts,
        )

        unique_posts = list({p.get("pid"): p for p in all_searched_posts if p.get("pid") is not None}.values())
        stage_b_unique_posts = list({p.get("pid"): p for p in stage_b_posts if p.get("pid") is not None}.values())
        final_candidates = stage_b_unique_posts if stage_b_unique_posts else unique_posts

        msg = (
            f"✓ 累计找到 {len(unique_posts)} 个不重复帖子，"
            f"阶段B候选 {len(stage_b_unique_posts)} 个，正在生成回答..."
        )
        print(f"\n{AGENT_PREFIX}{msg}")
        if self.info_callback:
            self.info_callback(msg)

        final_context_posts = self._hydrate_posts_for_context(
            final_candidates,
            MAX_COMMENTS_PER_POST,
            selection_query=user_question,
            context_post_limit=MAX_CONTEXT_POSTS,
        )
        final_context_text = format_posts_batch(
            final_context_posts,
            include_comments=True,
            max_comments=MAX_COMMENTS_PER_POST,
        )

        self._append_task_memory(
            (
                f"- final_context_posts={len(final_context_posts)} "
                f"(stage_b_candidates={len(stage_b_unique_posts)}, total_unique={len(unique_posts)})"
            )
        )

        print_separator("-")
        print("\n【最终回答】\n")

        messages.append({
            "role": "user",
            "content": (
                f"以下是最终高质量上下文（共 {len(final_context_posts)} 帖，已按策略补充关键评论）：\n\n"
                f"{final_context_text}"
            ),
        })

        messages.append({
            "role": "user",
            "content": "好的，你已经完成了所有搜索。请现在基于你已经检索到的所有树洞内容，用中文给出完整、有条理的回答。只使用已检索到的内容，不要编造信息；如果信息不足请诚实说明。",
        })
        response = self._call_llm_with_tools(messages, tools=[], stream=True)
        final_answer = response.get("content") or ""

        print("\n")
        print_separator("-")

        return {
            "answer": final_answer,
            "search_count": search_count,
            "search_history": search_history,
            "keywords": [],
            "sources": [{"pid": p.get("pid"), "text": p.get("text", "")[:100] + "..."} for p in final_candidates[:20]],
            "num_sources": len(final_candidates),
        }

    # ------------------------------------------------------------------ #
    #           Mode 2 Multi-turn: persistent conversation                #
    # ------------------------------------------------------------------ #

    def save_conversation(self):
        """Save conversation history to disk."""
        data = {
            "history": self._conversation_history,
            "search_count": self._conversation_search_count,
        }
        os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{AGENT_PREFIX}✓ 对话已保存到 {CONVERSATION_FILE}")

    def load_conversation(self) -> bool:
        """Load conversation history from disk. Returns True if loaded."""
        if not os.path.exists(CONVERSATION_FILE):
            return False
        try:
            with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._conversation_history = data.get("history", [])
            self._conversation_search_count = data.get("search_count", 0)
            turns = sum(1 for m in self._conversation_history if m["role"] == "user") // 2
            print(f"{AGENT_PREFIX}✓ 已恢复上次对话（{turns} 轮，{self._conversation_search_count} 次搜索）")
            return True
        except Exception as e:
            print(f"{AGENT_PREFIX}加载对话记录失败: {e}")
            return False

    def reset_conversation(self):
        """Reset conversation state. Archive old record if exists."""
        self._conversation_history.clear()
        self._conversation_searched_posts.clear()
        self._conversation_search_count = 0
        self._task_memory_file = None
        if os.path.exists(CONVERSATION_FILE):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = CONVERSATION_FILE.replace(".json", f"_{ts}.json")
            os.rename(CONVERSATION_FILE, archive_path)
            print(f"{AGENT_PREFIX}✓ 旧对话已归档: {archive_path}")
        print(f"{AGENT_PREFIX}✓ 对话已重置")

    def mode_auto_search_multi_turn(self, user_question: str) -> Dict[str, Any]:
        """Mode 2 Multi-turn: auto search with persistent conversation history.

        Each call appends the new user question to the existing conversation,
        allowing follow-up questions that leverage previously retrieved context.
        Call ``reset_conversation()`` to start fresh.
        """
        # Lazy-init system message on first turn
        if not self._conversation_history:
            print_header("模式 2: 智能自动检索（多轮对话）")
            self._start_task_memory(user_question, mode="auto_multi")
            self._conversation_history.append({
                "role": "system",
                "content": self._build_auto_search_system_prompt(),
            })
        else:
            print_separator("-")
            print(f"\n{AGENT_PREFIX}继续多轮对话（已有 {self._conversation_search_count} 次搜索记录）\n")
            if not self._task_memory_file:
                self._start_task_memory(user_question, mode="auto_multi_resume")

        self._append_task_memory(f"- user_turn: {user_question}")

        # Inject latest task memory snapshot for this turn.
        self._conversation_history.append({
            "role": "system",
            "content": self._build_auto_search_system_prompt(),
        })

        # Append user message
        self._conversation_history.append({"role": "user", "content": user_question})

        search_history_this_turn: List[Dict] = []
        stage_b_posts_this_turn: List[Dict[str, Any]] = []

        msg = "✓ LLM 分析中..."
        print(f"{AGENT_PREFIX}{msg}")
        if self.info_callback:
            self.info_callback(msg)

        self._conversation_history.append({
            "role": "user",
            "content": (
                f"请执行阶段A（宽泛探索）：建议 {BROAD_SEARCH_MIN}-{BROAD_SEARCH_MAX} 次。"
                "目标是识别黑话/简称/别称，并扩大关键词覆盖；若信息已收敛可提前结束。"
            ),
        })
        self._conversation_search_count = self._execute_search_loop(
            self._conversation_history,
            self._conversation_searched_posts,
            search_history_this_turn,
            self._conversation_search_count,
            phase_max=BROAD_SEARCH_MAX,
            phase_name="阶段A-宽泛探索",
            phase_posts=None,
        )

        self._conversation_history.append({
            "role": "user",
            "content": (
                f"请执行阶段B（高质量聚焦）：建议 {FOCUSED_SEARCH_MIN}-{FOCUSED_SEARCH_MAX} 次。"
                "优先抓取高 reply / 高 star 的帖子，必要时用 get_post_by_pid 精确补洞；证据充分可提前结束。"
            ),
        })
        self._conversation_search_count = self._execute_search_loop(
            self._conversation_history,
            self._conversation_searched_posts,
            search_history_this_turn,
            self._conversation_search_count,
            phase_max=FOCUSED_SEARCH_MAX,
            phase_name="阶段B-高质量聚焦",
            phase_posts=stage_b_posts_this_turn,
        )

        unique_posts = list({p.get("pid"): p for p in self._conversation_searched_posts if p.get("pid") is not None}.values())
        stage_b_unique_posts = list({p.get("pid"): p for p in stage_b_posts_this_turn if p.get("pid") is not None}.values())
        final_candidates = stage_b_unique_posts if stage_b_unique_posts else unique_posts

        if search_history_this_turn:
            msg = (
                f"✓ 本轮搜索 {len(search_history_this_turn)} 次，累计 {len(unique_posts)} 个不重复帖子，"
                f"阶段B候选 {len(stage_b_unique_posts)} 个"
            )
            print(f"\n{AGENT_PREFIX}{msg}")
            if self.info_callback:
                self.info_callback(msg)

        final_context_posts = self._hydrate_posts_for_context(
            final_candidates,
            MAX_COMMENTS_PER_POST,
            selection_query=user_question,
            context_post_limit=MAX_CONTEXT_POSTS,
        )
        final_context_text = format_posts_batch(
            final_context_posts,
            include_comments=True,
            max_comments=MAX_COMMENTS_PER_POST,
        )

        self._append_task_memory(
            (
                f"- turn_final_context_posts={len(final_context_posts)} "
                f"(stage_b_candidates={len(stage_b_unique_posts)}, total_unique={len(unique_posts)})"
            )
        )

        # Ask LLM for final answer
        print_separator("-")
        print("\n【回答】\n")

        self._conversation_history.append({
            "role": "user",
            "content": (
                f"以下是本轮最终高质量上下文（共 {len(final_context_posts)} 帖，已按策略补充关键评论）：\n\n"
                f"{final_context_text}"
            ),
        })

        self._conversation_history.append({
            "role": "user",
            "content": "请基于你已经检索到的所有树洞内容，用中文给出完整、有条理的回答。只使用已检索到的内容，不要编造信息；如果信息不足请诚实说明。",
        })

        response = self._call_llm_with_tools(self._conversation_history, tools=[], stream=True)
        final_answer = response.get("content") or ""

        # Save assistant reply to history for next turn
        self._conversation_history.append(
            self._build_assistant_message(
                content=final_answer,
                reasoning_content=response.get("reasoning_content"),
            )
        )

        print("\n")
        print_separator("-")

        return {
            "answer": final_answer,
            "search_count": self._conversation_search_count,
            "search_history": search_history_this_turn,
            "sources": [{"pid": p.get("pid"), "text": p.get("text", "")[:100] + "..."} for p in final_candidates[:20]],
            "num_sources": len(final_candidates),
            "conversation_turns": sum(1 for m in self._conversation_history if m["role"] == "user") // 2,
        }

    # ------------------------------------------------------------------ #
    #                      Interactive CLI                                #
    # ------------------------------------------------------------------ #

    def interactive_mode(self):
        """Interactive mode for mode 2 multi-turn conversation only."""
        print_header("PKU Treehole RAG Agent - Mode 2 Interactive")

        if self.load_conversation():
            resume = input("检测到上次对话记录，是否继续？(Y/n): ").strip().lower()
            if resume not in ('y', ''):
                self.reset_conversation()
        else:
            self.reset_conversation()

        print(f"\n{AGENT_PREFIX}模式2多轮对话（/reset 重置, /save 存档, /quit 退出）\n")

        while True:
            user_question = input("\n你: ").strip()
            if not user_question:
                continue
            if user_question == "/quit":
                self.save_conversation()
                print(f"{AGENT_PREFIX}退出多轮对话")
                break
            if user_question == "/reset":
                self.reset_conversation()
                continue
            if user_question == "/save":
                self.save_conversation()
                continue

            self.mode_auto_search_multi_turn(user_question)


def main():
    """Main entry point."""
    try:
        agent = TreeholeRAGAgent()
        agent.interactive_mode()
    except KeyboardInterrupt:
        print("\n\n[Agent] 程序被用户中断")
    except Exception as e:
        print(f"\n[Agent] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
