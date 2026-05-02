"""
PKU Treehole RAG Agent

Main agent class implementing mode 2 only:
Auto keyword extraction with staged retrieval (multi-turn capable)
"""

import json
import os
import random
import re
import difflib
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter

from client import TreeholeClient
from utils import (
    extract_keywords,
    format_posts_batch,
    format_post_to_text,
    save_json,
    load_json,
    get_cache_key,
    is_cache_valid,
    print_header,
    print_separator,
    truncate_text,
)

# Agent debug message prefix
AGENT_PREFIX = "[Agent] "

# Project root directory (stable regardless of current working directory)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_KNOWLEDGE_FILE = os.path.join(PROJECT_DIR, "agent.md")
TASK_MEMORY_DIR = os.path.join(PROJECT_DIR, "data", "task_memory")
SESSIONS_DIR = os.path.join(PROJECT_DIR, "data", "sessions")
ACTIVE_SESSION_FILE = os.path.join(PROJECT_DIR, "data", "active_session.json")
CONTEXT_CACHE_DIR = os.path.join(PROJECT_DIR, "data", "context_cache")
THOROUGH_SEARCH_DIR = os.path.join(PROJECT_DIR, "data", "thorough_search")
DAILY_DIGEST_DIR = os.path.join(PROJECT_DIR, "data", "daily_digest")
PID_POST_CACHE_DIR = os.path.join(PROJECT_DIR, "data", "pid_post_cache")
COMMENT_CACHE_DIR = os.path.join(PROJECT_DIR, "data", "comment_cache")
LATEST_PID_STATE_FILE = os.path.join(PROJECT_DIR, "data", "latest_pid_state.json")

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
CACHE_EXPIRATION = getattr(_cfg, "CACHE_EXPIRATION", 7 * 24 * 3600)

# Optional settings with backward-compatible defaults
SEARCH_PAGE_LIMIT = getattr(_cfg, "SEARCH_PAGE_LIMIT", 30)
SEARCH_COMMENT_LIMIT = getattr(_cfg, "SEARCH_COMMENT_LIMIT", 10)
INCLUDE_IMAGE_POSTS = getattr(_cfg, "INCLUDE_IMAGE_POSTS", True)
MAX_COMMENT_FETCH_POSTS = getattr(_cfg, "MAX_COMMENT_FETCH_POSTS", 6)
COMMENT_FETCH_MAX_PARALLEL = getattr(_cfg, "COMMENT_FETCH_MAX_PARALLEL", 10)
COMMENT_FETCH_MAX_REQUESTS_PER_SECOND = getattr(_cfg, "COMMENT_FETCH_MAX_REQUESTS_PER_SECOND", 20.0)
PID_FETCH_MAX_PARALLEL = getattr(_cfg, "PID_FETCH_MAX_PARALLEL", 20)
PID_FETCH_MAX_REQUESTS_PER_SECOND = getattr(_cfg, "PID_FETCH_MAX_REQUESTS_PER_SECOND", 40.0)
PID_POST_CACHE_EXPIRATION = getattr(_cfg, "PID_POST_CACHE_EXPIRATION", 7 * 24 * 3600)
PID_MISS_CACHE_EXPIRATION = getattr(_cfg, "PID_MISS_CACHE_EXPIRATION", 30 * 60)
COMMENT_CACHE_EXPIRATION = getattr(_cfg, "COMMENT_CACHE_EXPIRATION", 7 * 24 * 3600)
LLM_RETRY_MAX_ATTEMPTS = getattr(_cfg, "LLM_RETRY_MAX_ATTEMPTS", 5)
LLM_RETRY_SLEEP_SECONDS = getattr(_cfg, "LLM_RETRY_SLEEP_SECONDS", 5)
SEARCH_MAX_REQUESTS_PER_SECOND = getattr(_cfg, "SEARCH_MAX_REQUESTS_PER_SECOND", PID_FETCH_MAX_REQUESTS_PER_SECOND)
QUICK_QA_MAX_TURNS = getattr(_cfg, "QUICK_QA_MAX_TURNS", 5)
QUICK_QA_MAX_TOOL_ROUNDS = getattr(_cfg, "QUICK_QA_MAX_TOOL_ROUNDS", 4)
QUICK_QA_SEARCH_BUDGET = getattr(_cfg, "QUICK_QA_SEARCH_BUDGET", 12)
DEEP_RESEARCH_MAX_TOOL_ROUNDS = getattr(_cfg, "DEEP_RESEARCH_MAX_TOOL_ROUNDS", 10)
DEEP_RESEARCH_SEARCH_BUDGET = getattr(_cfg, "DEEP_RESEARCH_SEARCH_BUDGET", 30)
RECENT_PID_SCAN_HINT = getattr(_cfg, "RECENT_PID_SCAN_HINT", 8000000)
RECENT_PID_SCAN_STEP = getattr(_cfg, "RECENT_PID_SCAN_STEP", 5000)
RECENT_PID_SCAN_MAX_PROBES = getattr(_cfg, "RECENT_PID_SCAN_MAX_PROBES", 1500)
DAILY_DIGEST_RECENT_POSTS = getattr(_cfg, "DAILY_DIGEST_RECENT_POSTS", 4000)
DAILY_DIGEST_TOP_POSTS = getattr(_cfg, "DAILY_DIGEST_TOP_POSTS", 12)
THOROUGH_SEARCH_MAX_RESULTS_PER_KEYWORD = getattr(_cfg, "THOROUGH_SEARCH_MAX_RESULTS_PER_KEYWORD", -1)
THOROUGH_SEARCH_MAX_CONTEXT_POSTS = getattr(_cfg, "THOROUGH_SEARCH_MAX_CONTEXT_POSTS", max(MAX_CONTEXT_POSTS, 30))
SESSION_RECENT_TURNS = getattr(_cfg, "SESSION_RECENT_TURNS", 5)
SESSION_CONTEXT_MAX_POSTS = getattr(_cfg, "SESSION_CONTEXT_MAX_POSTS", max(MAX_CONTEXT_POSTS, 40))

# Backward-compatible legacy knobs (still used in docs/prompts as soft hints)
BROAD_SEARCH_MIN = getattr(_cfg, "BROAD_SEARCH_MIN", 10)
BROAD_SEARCH_MAX = getattr(_cfg, "BROAD_SEARCH_MAX", 20)
FOCUSED_SEARCH_MIN = getattr(_cfg, "FOCUSED_SEARCH_MIN", 5)
FOCUSED_SEARCH_MAX = getattr(_cfg, "FOCUSED_SEARCH_MAX", 10)

# Normalize cache path to an absolute path under project root when configured as relative.
if not os.path.isabs(CACHE_DIR):
    CACHE_DIR = os.path.join(PROJECT_DIR, CACHE_DIR)


class _TokenBucket:
    """Thread-safe token bucket for request rate limiting."""

    def __init__(
        self,
        refill_rate: float,
        capacity: Optional[float] = None,
        initial_tokens: Optional[float] = None,
    ):
        self.refill_rate = max(0.0, float(refill_rate))
        self.capacity = max(1.0, float(capacity if capacity is not None else refill_rate or 1.0))
        if initial_tokens is None:
            initial_tokens = self.capacity
        self.tokens = max(0.0, min(self.capacity, float(initial_tokens)))
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

            time.sleep(min(wait_s, 0.01))


class _LiveStatusTimer:
    """Render a single-line, real-time elapsed timer for slow LLM phases."""

    def __init__(self, label: str, interval: float = 0.1):
        self.label = label
        self.interval = max(0.05, interval)
        self.start_ts = time.time()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_len = 0
        self._wrote = False

    def start(self) -> "_LiveStatusTimer":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def elapsed(self) -> float:
        return time.time() - self.start_ts

    def stop(self, clear: bool = True) -> float:
        elapsed = self.elapsed()
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval + 0.2)
        if clear and self._wrote:
            print("\r" + (" " * self._last_len) + "\r", end="", flush=True)
        return elapsed

    def _run(self) -> None:
        while not self._stop_event.is_set():
            line = f"{AGENT_PREFIX}{self.label}: {self.elapsed():.1f}s"
            padding = " " * max(0, self._last_len - len(line))
            print("\r" + line + padding, end="", flush=True)
            self._last_len = len(line)
            self._wrote = True
            self._stop_event.wait(self.interval)


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
            "description": "在北大树洞中按关键词搜索主帖。搜索只匹配主帖正文，不匹配评论；多关键词是严格同时匹配；英文按完整词匹配，不做前缀匹配；返回结果默认按发帖时间从新到旧排序。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词。优先 1-2 个核心词；如果要扩大覆盖，宁可分多次搜，也不要把很多概念硬塞进一次搜索。"
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
        "搜索工具特性：\n"
        "- search_treehole 只能搜到主帖正文，搜不到评论里的关键词\n"
        "- 多关键词搜索是严格匹配，关键词越多越容易漏掉结果\n"
        "- 英文更接近完整词匹配，不会稳定匹配前缀变化\n"
        "- 搜索结果默认按发帖时间从新到旧排序，不是按热度排序\n\n"
        "帖子质量判断：\n"
        "- 一般来说，reply 越多代表热度越高\n"
        "- 一般来说，star 越多代表质量/收藏价值越高\n"
        "- 时间和相关性仍然重要，不能只看热度\n\n"
        "搜索策略：\n"
        "- 可以先宽后窄，但不要把阶段写死；是否进入更聚焦的检索由你自行判断\n"
        "- 评论信息通常也有价值；一旦帖子看起来重要，尽快补评论，不必等到某个固定阶段\n"
        "- 单轮尽量多做几次工具调用（建议 4-8 个），优先一次性发出多组搜索请求提高效率\n\n"
        "执行规则：\n"
        "- 已知 PID 时优先用 get_post_by_pid，减少冗余检索\n"
        "- 需要评论细节时用 get_comments_by_pid，不要只停留在主帖标题级信息\n"
        "- 优先在同一轮一次性返回多个 tool_calls（建议 4-8 个），减少模型往返次数\n"
        "- 仅当必须依赖上一批工具结果再决策时，才拆成下一轮调用\n"
        "- 若当前信息已足够支持结论，可直接总结，不需要机械分阶段\n"
        "- 搜索关键词优先 1-2 个核心词，必要时组合黑话\n"
        "- 严禁编造信息，只基于检索结果回答\n"
        "- 信息不足时明确说明\n"
        "- 保持客观，综合多条高质量帖子后再总结\n"
        "- 在回答前，先确保真正关键的帖子已经补充评论信息\n\n"
        "参考预算（软约束）：\n"
        "- 宽泛探索可参考 {broad_min}-{broad_max} 次搜索\n"
        "- 聚焦深入可参考 {focused_min}-{focused_max} 次搜索"
    )

    MODE_PROFILES = {
        "quick": {
            "label": "日常Q&A",
            "session_mode": True,
            "max_turns": QUICK_QA_MAX_TURNS,
            "max_tool_rounds": QUICK_QA_MAX_TOOL_ROUNDS,
            "search_budget": QUICK_QA_SEARCH_BUDGET,
            "search_results_per_call": min(MAX_SEARCH_RESULTS, 25),
            "context_post_limit": min(SESSION_CONTEXT_MAX_POSTS, max(12, MAX_CONTEXT_POSTS)),
            "comment_fetch_posts": min(MAX_COMMENT_FETCH_POSTS, max(6, MAX_COMMENT_FETCH_POSTS)),
            "comment_limit": min(MAX_COMMENTS_PER_POST if MAX_COMMENTS_PER_POST > 0 else 8, 12) if MAX_COMMENTS_PER_POST != -1 else 12,
            "query_style": "快速检索，优先解决当前问题，适合连续追问，默认控制在五轮以内。",
        },
        "deep": {
            "label": "Deep Research",
            "session_mode": True,
            "max_turns": 999,
            "max_tool_rounds": DEEP_RESEARCH_MAX_TOOL_ROUNDS,
            "search_budget": DEEP_RESEARCH_SEARCH_BUDGET,
            "search_results_per_call": max(MAX_SEARCH_RESULTS, 40),
            "context_post_limit": max(SESSION_CONTEXT_MAX_POSTS, MAX_CONTEXT_POSTS),
            "comment_fetch_posts": max(MAX_COMMENT_FETCH_POSTS, 12),
            "comment_limit": MAX_COMMENTS_PER_POST,
            "query_style": "在预算内渐进式研究，可先广泛摸底，再逐步聚焦，并自主决定研究方向。",
        },
    }

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
        self._llm_request_seq = 0
        self._tool_request_seq = 0
        self._seq_lock = threading.Lock()
        self._search_rate_limiter = _TokenBucket(
            SEARCH_MAX_REQUESTS_PER_SECOND,
            capacity=SEARCH_MAX_REQUESTS_PER_SECOND,
            initial_tokens=0,
        )
        self._pid_fetch_rate_limiter = _TokenBucket(
            PID_FETCH_MAX_REQUESTS_PER_SECOND,
            capacity=PID_FETCH_MAX_REQUESTS_PER_SECOND,
            initial_tokens=0,
        )

        # Session state
        self._conversation_history: List[Dict[str, Any]] = []
        self._conversation_searched_posts: List[Dict[str, Any]] = []
        self._conversation_search_count: int = 0
        self._active_session_id: Optional[str] = None
        self._session_mode: str = "quick"
        self._session_title: str = ""
        self._session_created_at: str = ""
        self._session_updated_at: str = ""
        self._session_turns: List[Dict[str, Any]] = []
        self._session_posts: Dict[int, Dict[str, Any]] = {}
        self._default_cli_mode: str = "quick"
        self._task_memory_file: Optional[str] = None
        self._agent_knowledge: str = ""
        self._last_memory_snapshot_hash: str = ""
        self._comment_rate_limiter = _TokenBucket(
            COMMENT_FETCH_MAX_REQUESTS_PER_SECOND,
            capacity=COMMENT_FETCH_MAX_REQUESTS_PER_SECOND,
            initial_tokens=0,
        )
        self._comment_metrics_lock = threading.Lock()
        self._comment_fetch_stats: Dict[str, Any] = self._new_comment_fetch_stats()
        self._post_cache: Dict[int, Dict[str, Any]] = {}
        self._post_miss_cache: Dict[int, float] = {}

        # Load persistent agent knowledge and ensure data directories exist.
        os.makedirs(TASK_MEMORY_DIR, exist_ok=True)
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        os.makedirs(CONTEXT_CACHE_DIR, exist_ok=True)
        os.makedirs(THOROUGH_SEARCH_DIR, exist_ok=True)
        os.makedirs(DAILY_DIGEST_DIR, exist_ok=True)
        os.makedirs(PID_POST_CACHE_DIR, exist_ok=True)
        os.makedirs(COMMENT_CACHE_DIR, exist_ok=True)
        self._agent_knowledge = self._load_agent_knowledge()
        self._configure_http_pool()

        # Ensure login
        if not self.client.ensure_login(USERNAME, PASSWORD, interactive=interactive):
            raise RuntimeError("Failed to login to Treehole. Try running interactively first to save cookies.")

        # Create cache directory
        if ENABLE_CACHE:
            os.makedirs(CACHE_DIR, exist_ok=True)

        print(f"{AGENT_PREFIX}✓ 树洞 RAG Agent 初始化成功")

    def _configure_http_pool(self) -> None:
        """Align requests' connection pool with our worker concurrency."""
        session = getattr(self.client, "session", None)
        if session is None:
            return
        pool_size = max(
            10,
            int(COMMENT_FETCH_MAX_PARALLEL),
            int(PID_FETCH_MAX_PARALLEL),
            int(SEARCH_MAX_REQUESTS_PER_SECOND),
        )
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, pool_block=True)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

    def _emit_info(self, message: str) -> None:
        """Print and forward user-visible runtime updates."""
        print(f"{AGENT_PREFIX}{message}")
        if self.info_callback:
            self.info_callback(message)

    def _next_tool_request_id(self) -> int:
        with self._seq_lock:
            self._tool_request_seq += 1
            return self._tool_request_seq

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

    @staticmethod
    def _now_str() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _slug_from_text(text: str, max_len: int = 24) -> str:
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "-", (text or "").strip())
        cleaned = cleaned.strip("-_")
        return (cleaned[:max_len] or "session").lower()

    def _build_session_title(self, seed_text: str) -> str:
        question = (seed_text or "").strip()
        if not question:
            return "未命名会话"
        return truncate_text(question.replace("\n", " "), 40)

    def _session_file(self, session_id: str) -> str:
        return os.path.join(SESSIONS_DIR, f"{session_id}.json")

    def _context_cache_file(self, session_id: str) -> str:
        return os.path.join(CONTEXT_CACHE_DIR, f"{session_id}.json")

    def _write_active_session_pointer(self) -> None:
        if not self._active_session_id:
            return
        save_json(
            {
                "active_session_id": self._active_session_id,
                "updated_at": self._now_str(),
            },
            ACTIVE_SESSION_FILE,
        )

    def _read_active_session_id(self) -> Optional[str]:
        data = load_json(ACTIVE_SESSION_FILE) or {}
        session_id = data.get("active_session_id")
        return str(session_id) if session_id else None

    def _reset_session_state(self) -> None:
        self._conversation_history.clear()
        self._conversation_searched_posts.clear()
        self._conversation_search_count = 0
        self._active_session_id = None
        self._session_mode = self._default_cli_mode
        self._session_title = ""
        self._session_created_at = ""
        self._session_updated_at = ""
        self._session_turns = []
        self._session_posts = {}
        self._task_memory_file = None
        self._last_memory_snapshot_hash = ""

    def _new_session_id(self, seed_text: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{self._slug_from_text(seed_text)}"

    def _serialize_session(self) -> Dict[str, Any]:
        posts = sorted(
            [self._normalize_post_metadata(dict(post)) for post in self._session_posts.values() if post.get("pid")],
            key=lambda item: int(item.get("pid", 0)),
        )
        return {
            "session_id": self._active_session_id,
            "title": self._session_title,
            "mode": self._session_mode,
            "created_at": self._session_created_at,
            "updated_at": self._session_updated_at or self._now_str(),
            "search_count": self._conversation_search_count,
            "turns": self._session_turns,
            "posts": posts,
            "task_memory_file": self._task_memory_file,
        }

    def _persist_context_cache(self) -> None:
        if not self._active_session_id:
            return
        ranked_posts = self._rank_posts_for_query(
            list(self._session_posts.values()),
            query=" ".join(turn.get("content", "") for turn in self._session_turns[-SESSION_RECENT_TURNS:]),
            limit=min(SESSION_CONTEXT_MAX_POSTS, max(20, len(self._session_posts))),
        )
        compact = []
        for post in ranked_posts:
            comments = post.get("comments") or post.get("comment_list") or []
            compact.append(
                {
                    "pid": post.get("pid"),
                    "post_time": post.get("post_time"),
                    "reply_count": post.get("reply_count", post.get("reply", 0)),
                    "star_count": post.get("star_count", post.get("likenum", 0)),
                    "comment_total": post.get("comment_total", len(comments)),
                    "text_preview": self._compact_text_preview(post.get("text", ""), max_len=180),
                    "comment_preview": [
                        self._compact_text_preview(comment.get("text", ""), max_len=90)
                        for comment in comments[:3]
                    ],
                }
            )
        save_json(
            {
                "session_id": self._active_session_id,
                "updated_at": self._now_str(),
                "posts": compact,
            },
            self._context_cache_file(self._active_session_id),
        )

    def _save_session(self) -> None:
        if not self._active_session_id:
            return
        self._session_updated_at = self._now_str()
        payload = self._serialize_session()
        save_json(payload, self._session_file(self._active_session_id))
        self._persist_context_cache()
        self._write_active_session_pointer()
        save_json(
            {
                "active_session_id": self._active_session_id,
                "history": self._session_turns,
                "search_count": self._conversation_search_count,
                "mode": self._session_mode,
            },
            CONVERSATION_FILE,
        )

    def _load_session_payload(self, payload: Dict[str, Any]) -> bool:
        session_id = payload.get("session_id")
        if not session_id:
            return False
        self._reset_session_state()
        self._active_session_id = str(session_id)
        self._session_mode = str(payload.get("mode") or "quick")
        self._default_cli_mode = self._session_mode if self._session_mode in self.MODE_PROFILES else "quick"
        self._session_title = str(payload.get("title") or "未命名会话")
        self._session_created_at = str(payload.get("created_at") or self._now_str())
        self._session_updated_at = str(payload.get("updated_at") or self._session_created_at)
        self._conversation_search_count = int(payload.get("search_count") or 0)
        self._session_turns = list(payload.get("turns") or [])
        self._task_memory_file = payload.get("task_memory_file") or None
        raw_posts = payload.get("posts") or []
        self._session_posts = {}
        for post in raw_posts:
            normalized = self._normalize_post_metadata(dict(post))
            pid = normalized.get("pid")
            if pid:
                self._session_posts[int(pid)] = normalized
        self._conversation_searched_posts = list(self._session_posts.values())
        return True

    def _begin_new_session(self, seed_text: str, profile: str) -> None:
        self._reset_session_state()
        self._active_session_id = self._new_session_id(seed_text)
        self._session_mode = profile
        self._session_title = self._build_session_title(seed_text)
        self._session_created_at = self._now_str()
        self._session_updated_at = self._session_created_at
        self._start_task_memory(seed_text, mode=profile)
        self._save_session()
        self._emit_info(f"已创建新会话: {self._session_title} ({self._active_session_id})")

    def _load_session_by_id(self, session_id: str) -> bool:
        session_path = self._session_file(session_id)
        if not os.path.exists(session_path):
            return False
        payload = load_json(session_path)
        if not isinstance(payload, dict):
            return False
        ok = self._load_session_payload(payload)
        if ok:
            self._write_active_session_pointer()
        return ok

    def _load_latest_session(self) -> bool:
        active_session_id = self._read_active_session_id()
        if active_session_id and self._load_session_by_id(active_session_id):
            return True
        sessions = self.list_sessions()
        if sessions:
            return self._load_session_by_id(sessions[0]["session_id"])
        return False

    def _ensure_session_for_mode(self, profile: str, seed_text: str) -> None:
        session_turn_count = sum(1 for turn in self._session_turns if turn.get("role") == "user")
        if not self._active_session_id:
            self._begin_new_session(seed_text, profile)
            return
        if self._session_mode != profile:
            self._save_session()
            self._begin_new_session(seed_text, profile)
            return
        if profile == "quick" and session_turn_count >= QUICK_QA_MAX_TURNS:
            self._save_session()
            self._begin_new_session(seed_text, profile)
            self._emit_info("快速问答会话已达到五轮上限，已自动开启新会话。")

    def _append_session_turn(self, role: str, content: str, mode: Optional[str] = None, **extra: Any) -> None:
        self._session_turns.append(
            {
                "role": role,
                "content": content,
                "mode": mode or self._session_mode,
                "created_at": self._now_str(),
                **extra,
            }
        )
        self._session_updated_at = self._now_str()

    def _upsert_session_posts(self, posts: List[Dict[str, Any]]) -> None:
        for post in posts:
            normalized = self._normalize_post_metadata(dict(post))
            pid = normalized.get("pid")
            if pid:
                self._session_posts[int(pid)] = normalized
        self._conversation_searched_posts = list(self._session_posts.values())

    def _build_recent_turns_snippet(self, max_turns: int = SESSION_RECENT_TURNS) -> str:
        if not self._session_turns:
            return ""
        snippets = []
        relevant_turns = self._session_turns[-max_turns * 2:]
        for turn in relevant_turns:
            role = "用户" if turn.get("role") == "user" else "助手"
            content = truncate_text(str(turn.get("content") or "").replace("\n", " "), 220)
            snippets.append(f"- {role}: {content}")
        return "\n".join(snippets)

    def _rank_posts_for_query(
        self,
        posts: List[Dict[str, Any]],
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        tokens = [tok.lower() for tok in extract_keywords(query or "")]
        ranked = []
        for post in posts:
            text = str(post.get("text") or "")
            text_lower = text.lower()
            reply_count = max(0, int(post.get("reply_count", post.get("reply", 0)) or 0))
            star_count = max(0, int(post.get("star_count", post.get("likenum", 0)) or 0))
            comment_total = max(0, int(post.get("comment_total", 0) or 0))
            match_score = 0.0
            if tokens:
                hits = sum(1 for token in tokens if token and token in text_lower)
                match_score = hits / max(1, len(tokens))
            recency_score = int(post.get("pid", 0) or 0) / 1_000_000.0
            score = match_score * 5.0 + star_count * 0.7 + reply_count * 0.5 + comment_total * 0.2 + recency_score
            ranked.append((score, post))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [post for _, post in ranked[:max(1, limit)]]

    def _build_compact_context_snippet(
        self,
        query: str,
        limit: int = 18,
        max_chars: int = 8000,
    ) -> str:
        if not self._session_posts:
            return ""
        ranked_posts = self._rank_posts_for_query(list(self._session_posts.values()), query=query, limit=limit)
        lines = []
        for post in ranked_posts:
            pid = post.get("pid")
            lines.append(
                f"- pid={pid} | time={post.get('post_time')} | reply={post.get('reply_count', 0)} | "
                f"star={post.get('star_count', 0)} | {self._compact_text_preview(post.get('text', ''), max_len=160)}"
            )
            comments = post.get("comments") or post.get("comment_list") or []
            for idx, comment in enumerate(comments[:2], 1):
                lines.append(f"  - 评论{idx}: {self._compact_text_preview(comment.get('text', ''), max_len=90)}")
        compact = "\n".join(lines)
        return compact[:max_chars]

    def _save_text_artifact(self, path: str, content: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        if not os.path.exists(SESSIONS_DIR):
            return sessions
        for name in os.listdir(SESSIONS_DIR):
            if not name.endswith(".json"):
                continue
            path = os.path.join(SESSIONS_DIR, name)
            payload = load_json(path)
            if not isinstance(payload, dict):
                continue
            sessions.append(
                {
                    "session_id": payload.get("session_id") or name[:-5],
                    "title": payload.get("title") or "未命名会话",
                    "mode": payload.get("mode") or "quick",
                    "updated_at": payload.get("updated_at") or "",
                    "turns": sum(1 for turn in (payload.get("turns") or []) if turn.get("role") == "user"),
                    "search_count": int(payload.get("search_count") or 0),
                }
            )
        sessions.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return sessions

    def render_session_list(self) -> str:
        sessions = self.list_sessions()
        if not sessions:
            return "暂无历史会话。"
        lines = []
        for item in sessions[:20]:
            lines.append(
                f"- {item['session_id']} | {item['mode']} | {item['updated_at']} | "
                f"{item['turns']}轮 | {item['search_count']}次搜索 | {item['title']}"
            )
        return "\n".join(lines)

    def render_session_history(self, session_id: Optional[str] = None, max_turns: int = 12) -> str:
        payload = None
        if session_id:
            payload = load_json(self._session_file(session_id))
        elif self._active_session_id:
            payload = self._serialize_session()
        if not isinstance(payload, dict):
            return "未找到会话。"
        turns = payload.get("turns") or []
        title = payload.get("title") or "未命名会话"
        lines = [f"# {title}", ""]
        for turn in turns[-max_turns * 2:]:
            role = "你" if turn.get("role") == "user" else "助手"
            lines.append(f"## {role}")
            lines.append(str(turn.get("content") or "").strip())
            lines.append("")
        return "\n".join(lines).strip()

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

    def _pid_post_cache_file(self, pid: int) -> str:
        return os.path.join(PID_POST_CACHE_DIR, f"{int(pid)}.json")

    def _load_pid_post_cache(self, pid: int) -> Optional[Any]:
        """Load persistent PID cache entry. Returns dict post, None for cached miss, or False for no cache."""
        now = time.time()
        if pid in self._post_cache:
            return dict(self._post_cache[pid])
        miss_ts = self._post_miss_cache.get(pid)
        if miss_ts and now - miss_ts <= PID_MISS_CACHE_EXPIRATION:
            return None

        path = self._pid_post_cache_file(pid)
        payload = load_json(path)
        if not isinstance(payload, dict):
            return False

        cached_at = float(payload.get("cached_at") or 0.0)
        found = bool(payload.get("found"))
        ttl = PID_POST_CACHE_EXPIRATION if found else PID_MISS_CACHE_EXPIRATION
        if ttl > 0 and cached_at > 0 and now - cached_at > ttl:
            return False

        if not found:
            self._post_miss_cache[pid] = cached_at or now
            return None

        post = payload.get("post")
        if not isinstance(post, dict):
            return False
        normalized = self._normalize_post_metadata(dict(post))
        self._post_cache[pid] = dict(normalized)
        return normalized

    def _save_pid_post_cache(self, pid: int, post: Optional[Dict[str, Any]]) -> None:
        """Persist one PID lookup result for later daily scans."""
        payload = {
            "pid": int(pid),
            "found": bool(post),
            "post": post if post else None,
            "cached_at": time.time(),
            "updated_at": self._now_str(),
        }
        save_json(payload, self._pid_post_cache_file(pid))
        if post:
            self._post_cache[int(pid)] = dict(post)
        else:
            self._post_miss_cache[int(pid)] = payload["cached_at"]

    def _comment_cache_file(self, pid: int) -> str:
        return os.path.join(COMMENT_CACHE_DIR, f"{int(pid)}.json")

    def _load_persistent_comments(self, pid: int) -> Optional[List[Dict[str, Any]]]:
        payload = load_json(self._comment_cache_file(pid))
        if not isinstance(payload, dict):
            return None
        cached_at = float(payload.get("cached_at") or 0.0)
        if COMMENT_CACHE_EXPIRATION > 0 and cached_at > 0 and time.time() - cached_at > COMMENT_CACHE_EXPIRATION:
            return None
        comments = payload.get("comments")
        if not isinstance(comments, list):
            return None
        normalized = [self._normalize_comment_metadata(c) for c in comments if isinstance(c, dict)]
        self._all_comments_cache[int(pid)] = normalized
        return normalized

    def _save_persistent_comments(self, pid: int, comments: List[Dict[str, Any]]) -> None:
        save_json(
            {
                "pid": int(pid),
                "comments": comments,
                "cached_at": time.time(),
                "updated_at": self._now_str(),
            },
            self._comment_cache_file(pid),
        )

    def get_post_by_pid(
        self,
        pid: int,
        include_comments: bool = False,
        max_comments: int = MAX_COMMENTS_PER_POST,
        quiet: bool = False,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Fetch one post by pid and optionally hydrate comments."""
        pid = int(pid)
        tool_id = None
        if not quiet:
            tool_id = self._next_tool_request_id()
        start_ts = time.time()
        cached = self._load_pid_post_cache(pid) if use_cache else False
        if isinstance(cached, dict):
            cached_post = dict(cached)
            if include_comments:
                cached_post["comments"] = self._load_comments_for_post(cached_post, max_comments)
            if not quiet:
                self._emit_info(
                    f"Tool#{tool_id} get_post_by_pid 命中缓存: pid={pid}, 耗时 {time.time() - start_ts:.2f}s"
                )
            return cached_post
        if cached is None:
            if not quiet:
                self._emit_info(
                    f"Tool#{tool_id} get_post_by_pid 命中未找到缓存: pid={pid}, 耗时 {time.time() - start_ts:.2f}s"
                )
            return None

        try:
            if not quiet:
                self._emit_info(f"Tool#{tool_id} get_post_by_pid 开始: pid={pid}")
            self._pid_fetch_rate_limiter.acquire()
            result = self.client.get_post(pid)
            if not result.get("success"):
                self._save_pid_post_cache(pid, None)
                if not quiet:
                    self._emit_info(f"Tool#{tool_id} get_post_by_pid 未找到: pid={pid}")
                return None
            post = self._normalize_post_metadata(result.get("data", {}))
            self._save_pid_post_cache(pid, post)
            if include_comments:
                post["comments"] = self._load_comments_for_post(post, max_comments)
            if not quiet:
                self._emit_info(
                    f"Tool#{tool_id} get_post_by_pid 完成: pid={pid}, 耗时 {time.time() - start_ts:.2f}s"
                )
            return post
        except Exception as e:
            if not quiet:
                self._emit_info(f"Tool#{tool_id} get_post_by_pid 失败: pid={pid}, error={e}")
            return None

    def get_comments_by_pid(
        self,
        pid: int,
        max_comments: int = MAX_COMMENTS_PER_POST,
        sort: str = "asc",
    ) -> List[Dict[str, Any]]:
        """Fetch comments by pid directly with pagination and global rate limit."""
        pid = int(pid)
        tool_id = self._next_tool_request_id()
        start_ts = time.time()
        cached = self._all_comments_cache.get(pid)
        if cached is None:
            cached = self._load_persistent_comments(pid)
        if cached is not None:
            if max_comments == -1:
                self._emit_info(
                    f"Tool#{tool_id} get_comments_by_pid 命中缓存: pid={pid}, comments={len(cached)}, 耗时 {time.time() - start_ts:.2f}s"
                )
                return cached
            result = cached[:max_comments]
            self._emit_info(
                f"Tool#{tool_id} get_comments_by_pid 命中缓存: pid={pid}, comments={len(result)}, 耗时 {time.time() - start_ts:.2f}s"
            )
            return result

        comments: List[Dict[str, Any]] = []
        page = 1
        self._emit_info(f"Tool#{tool_id} get_comments_by_pid 开始: pid={pid}, max_comments={max_comments}, sort={sort}")
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

            last_page = int(page_data.get("last_page") or page)
            if page >= last_page or not page_comments:
                break
            page += 1

        self._all_comments_cache[pid] = comments
        self._save_persistent_comments(pid, comments)
        if max_comments == -1:
            self._emit_info(
                f"Tool#{tool_id} get_comments_by_pid 完成: pid={pid}, comments={len(comments)}, 耗时 {time.time() - start_ts:.2f}s"
            )
            return comments
        result = comments[:max_comments]
        self._emit_info(
            f"Tool#{tool_id} get_comments_by_pid 完成: pid={pid}, comments={len(result)}, 耗时 {time.time() - start_ts:.2f}s"
        )
        return result

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
            "active_posts": 0,
            "peak_active_posts": 0,
            "request_timestamps": [],
            "completion_timestamps": [],
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

    def _record_comment_worker_started(self) -> None:
        """Record one active post-level comment worker."""
        with self._comment_metrics_lock:
            active = int(self._comment_fetch_stats.get("active_posts", 0)) + 1
            self._comment_fetch_stats["active_posts"] = active
            self._comment_fetch_stats["peak_active_posts"] = max(
                int(self._comment_fetch_stats.get("peak_active_posts", 0)),
                active,
            )

    def _record_comment_worker_finished(self) -> None:
        """Record one completed post-level comment worker."""
        with self._comment_metrics_lock:
            active = max(0, int(self._comment_fetch_stats.get("active_posts", 0)) - 1)
            self._comment_fetch_stats["active_posts"] = active
            self._comment_fetch_stats["completion_timestamps"].append(time.time())

    def _log_comment_fetch_stats(self) -> None:
        """Print concise request/concurrency metrics for one hydration batch."""
        with self._comment_metrics_lock:
            stats = dict(self._comment_fetch_stats)
            req_ts = list(stats.get("request_timestamps", []))
            done_ts = list(stats.get("completion_timestamps", []))

        started_at = float(stats.get("started_at") or time.time())
        elapsed = max(0.001, time.time() - started_at)
        api_requests = int(stats.get("api_requests", 0))
        selected_posts = int(stats.get("selected_posts", 0))
        completed_posts = int(stats.get("completed_posts", 0))
        failed_posts = int(stats.get("failed_posts", 0))
        cache_hits = int(stats.get("cache_hits", 0))
        peak_active_posts = int(stats.get("peak_active_posts", 0))
        submit_rate = api_requests / elapsed
        completion_rate = (completed_posts + failed_posts) / elapsed
        burst_submit_rate = 0.0
        if len(req_ts) > 1:
            request_span = max(0.001, req_ts[-1] - req_ts[0])
            burst_submit_rate = (len(req_ts) - 1) / request_span
        first_req_delay = (req_ts[0] - started_at) if req_ts else 0.0
        last_done_delay = (done_ts[-1] - started_at) if done_ts else elapsed

        msg = (
            "评论抓取统计: "
            f"候选 {selected_posts} 帖, 完成 {completed_posts}, 失败 {failed_posts}, "
            f"缓存命中 {cache_hits}, 请求 {api_requests}, "
            f"耗时 {elapsed:.2f}s, 完成吞吐 {completion_rate:.2f} posts/s, "
            f"平均提交 {submit_rate:.2f} req/s, 瞬时提交峰值 {burst_submit_rate:.2f} req/s, "
            f"活跃worker峰值 {peak_active_posts}/{max(1, COMMENT_FETCH_MAX_PARALLEL)}, "
            f"首请求 {first_req_delay:.2f}s, 最后完成 {last_done_delay:.2f}s, "
            f"提交速率上限 {COMMENT_FETCH_MAX_REQUESTS_PER_SECOND:.2f} req/s"
        )
        self._emit_info(msg)

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
        query_tokens = [tok.lower() for tok in extract_keywords(query or "")]

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

    @staticmethod
    def _nonnegative_int(value: Any, default: int = 0) -> int:
        try:
            return max(0, int(value or default))
        except Exception:
            return default

    @staticmethod
    def _int_or_default(value: Any, default: int = 0) -> int:
        try:
            if value is None or value == "":
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    def _post_has_sufficient_comments(self, post: Dict[str, Any], max_comments: int) -> bool:
        """Return True when comment hydration would not add useful context."""
        if max_comments == 0:
            return True

        existing_comments = [
            self._normalize_comment_metadata(c)
            for c in (post.get("comments") or post.get("comment_list") or [])
            if isinstance(c, dict)
        ]
        raw_comment_total = post.get("comment_total")
        comment_total = self._nonnegative_int(raw_comment_total, 0) if raw_comment_total is not None else -1

        if max_comments != -1 and len(existing_comments) >= max_comments:
            return True
        if max_comments == -1:
            if comment_total == 0:
                return True
            if comment_total > 0 and len(existing_comments) >= comment_total:
                return True

        pid = post.get("pid")
        if not pid:
            return False
        try:
            pid_int = int(pid)
        except Exception:
            return False

        cached_comments = self._all_comments_cache.get(pid_int)
        if cached_comments is None:
            cached_comments = self._load_persistent_comments(pid_int)
        if cached_comments is None:
            return False

        if max_comments == -1:
            return bool(comment_total > 0 and len(cached_comments) >= comment_total)
        return len(cached_comments) >= max_comments or bool(comment_total > 0 and len(cached_comments) >= comment_total)

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
        if cached_comments is None:
            cached_comments = self._load_persistent_comments(pid)
        if cached_comments is not None:
            self._inc_comment_stat("cache_hits")
            if max_comments == -1:
                if comment_total and len(cached_comments) < comment_total:
                    pass
                else:
                    return cached_comments
            elif len(cached_comments) >= max_comments or (comment_total and len(cached_comments) >= comment_total):
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
            self._emit_info(f"正在获取帖子 #{pid} 的全部 {comment_total} 条评论...")
        else:
            self._emit_info(f"正在获取帖子 #{pid} 的评论（目标 {target_count} 条）...")

        all_comments: List[Dict[str, Any]] = []
        page = 1
        last_progress_ts = time.time()
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

            last_page = int(page_data.get("last_page") or page)
            if max_comments == -1 and last_page > 1:
                now = time.time()
                if page == 1 or page >= last_page or page % 5 == 0 or now - last_progress_ts >= 3.0:
                    total_hint = f"/{comment_total}" if comment_total else ""
                    self._emit_info(
                        f"评论抓取进度: pid={pid}, 第 {page}/{last_page} 页, "
                        f"累计 {len(all_comments)}{total_hint} 条"
                    )
                    last_progress_ts = now

            if max_comments != -1 and len(all_comments) >= max_comments:
                all_comments = all_comments[:max_comments]
                break

            if page >= last_page or not page_comments:
                break
            page += 1

        if not all_comments:
            all_comments = existing_comments

        self._all_comments_cache[pid] = all_comments
        self._save_persistent_comments(pid, all_comments)
        if max_comments == -1:
            return all_comments
        return all_comments[:max_comments]

    def _hydrate_posts_for_context(
        self,
        posts: List[Dict[str, Any]],
        max_comments: int,
        selection_query: str = "",
        context_post_limit: Optional[int] = None,
        selected_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Hydrate context with AI-selected, rate-limited comment fetching.

        Selection is run on all candidate posts first, then context_post_limit is applied.
        """
        normalized_posts = [self._normalize_post_metadata(post) for post in posts]

        if max_comments == 0:
            for post in normalized_posts:
                post["comments"] = []
            return normalized_posts

        fetch_candidates = [
            post
            for post in normalized_posts
            if not self._post_has_sufficient_comments(post, max_comments)
        ]
        effective_selected_limit = selected_limit if selected_limit is not None else MAX_COMMENT_FETCH_POSTS
        effective_selected_limit = max(0, min(effective_selected_limit, len(fetch_candidates)))
        if fetch_candidates and effective_selected_limit > 0:
            selected_pids = self._select_posts_for_comment_fetch(
                fetch_candidates,
                query=selection_query,
                max_selected=effective_selected_limit,
            )
        else:
            selected_pids = []
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

        self._emit_info(
            f"评论补拉策略: 候选 {len(normalized_posts)} 帖, "
            f"AI 选择 {len(selected_pids)} 帖补拉评论, "
            f"上下文使用 {len(context_posts)} 帖"
        )
        if selected_pids:
            self._emit_info(f"AI 选中帖子 PID: {', '.join(str(pid) for pid in selected_pids)}")
        context_pids = [str(p.get("pid")) for p in context_posts if p.get("pid")]
        if context_pids:
            suffix = "" if len(context_pids) <= 40 else f" ...（共 {len(context_pids)} 帖）"
            self._emit_info(f"进入 LLM 上下文 PID: {', '.join(context_pids[:40])}{suffix}")

        comments_by_pid: Dict[int, List[Dict[str, Any]]] = {}
        if effective_selected_set:
            fetch_pids = [str(pid) for pid in selected_pids if pid in effective_selected_set]
            if fetch_pids:
                self._emit_info(f"实际补拉评论 PID: {', '.join(fetch_pids)}")
            self._reset_comment_fetch_stats(len(effective_selected_set))
            max_workers = max(1, min(COMMENT_FETCH_MAX_PARALLEL, len(effective_selected_set)))

            def fetch_for_post(p: Dict[str, Any]) -> Any:
                self._record_comment_worker_started()
                try:
                    pid = int(p.get("pid"))
                    comments = self._load_comments_for_post(p, max_comments)
                    return pid, comments
                finally:
                    self._record_comment_worker_finished()

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
                        self._emit_info(f"警告: 获取帖子 {pid} 的评论失败: {e}")

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
        normalized_max_results = max_results
        if normalized_max_results == 0:
            return []
        if normalized_max_results is None:
            normalized_max_results = MAX_SEARCH_RESULTS

        # Check cache first
        if use_cache:
            cache_key = get_cache_key(
                f"{keyword}|{normalized_max_results}|{SEARCH_PAGE_LIMIT}|{SEARCH_COMMENT_LIMIT}|{INCLUDE_IMAGE_POSTS}"
            )
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

            if is_cache_valid(cache_file, CACHE_EXPIRATION):
                self._emit_info(f"搜索缓存命中: {keyword}")
                cached = load_json(cache_file) or []
                return [self._normalize_post_metadata(post) for post in cached]

        tool_id = self._next_tool_request_id()
        start_ts = time.time()
        self._emit_info(f"Tool#{tool_id} search_treehole 开始: {keyword}")

        try:
            all_posts: List[Dict[str, Any]] = []
            page = 1
            search_comment_limit = self._resolve_search_comment_limit()
            page_limit = max(1, SEARCH_PAGE_LIMIT)
            unlimited = normalized_max_results == -1
            if unlimited:
                self._emit_info(
                    f"Tool#{tool_id} search_treehole 分页抓取: "
                    f"page_limit={page_limit}, comment_limit={search_comment_limit}, "
                    f"速率上限 {SEARCH_MAX_REQUESTS_PER_SECOND:.2f} req/s"
                )

            while unlimited or len(all_posts) < normalized_max_results:
                request_limit = page_limit if unlimited else min(page_limit, normalized_max_results - len(all_posts))
                self._search_rate_limiter.acquire()
                if page == 1 and (SEARCH_DELAY_MAX > 0 or SEARCH_DELAY_MIN > 0):
                    self._human_delay(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX, f"搜索请求 {keyword}")

                search_result = self.client.search_posts(
                    keyword,
                    page=page,
                    limit=request_limit,
                    comment_limit=search_comment_limit,
                )

                if not search_result.get("success"):
                    msg = f"搜索失败: {search_result.get('message', '未知错误')}"
                    self._emit_info(msg)
                    break

                page_posts = search_result.get("data", {}).get("data", [])
                if not page_posts:
                    self._emit_info(
                        f"Tool#{tool_id} search_treehole 进度: {keyword}, "
                        f"第 {page} 页无结果，累计 {len(all_posts)} 帖"
                    )
                    break

                normalized_posts = [self._normalize_post_metadata(post) for post in page_posts]
                if not INCLUDE_IMAGE_POSTS:
                    normalized_posts = [post for post in normalized_posts if not post.get("has_image")]

                all_posts.extend(normalized_posts)

                page_data = search_result.get("data", {})
                total = int(page_data.get("total", 0) or 0)
                last_page = int(page_data.get("last_page") or page)
                if unlimited or page > 1 or page < last_page:
                    total_hint = f"/{total}" if total > 0 else ""
                    elapsed = max(0.001, time.time() - start_ts)
                    self._emit_info(
                        f"Tool#{tool_id} search_treehole 进度: {keyword}, "
                        f"第 {page}/{last_page} 页, 本页 {len(page_posts)} 帖, "
                        f"累计 {len(all_posts)}{total_hint} 帖, 耗时 {elapsed:.2f}s"
                    )
                if page >= last_page or len(page_posts) < request_limit:
                    break
                page += 1

            enriched_posts = all_posts if unlimited else all_posts[:normalized_max_results]

            if use_cache:
                save_json(enriched_posts, cache_file)

            msg = (
                f"Tool#{tool_id} search_treehole 完成: {len(enriched_posts)} 帖, "
                f"耗时 {time.time() - start_ts:.2f}s"
            )
            self._emit_info(msg)
            return enriched_posts

        except Exception as e:
            self._emit_info(f"Tool#{tool_id} search_treehole 失败: {e}")
            return []

    def search_treehole_exhaustive(
        self,
        keyword: str,
        use_cache: bool = ENABLE_CACHE,
    ) -> List[Dict[str, Any]]:
        """Fetch all pages for a keyword unless a config cap is specified."""
        max_results = THOROUGH_SEARCH_MAX_RESULTS_PER_KEYWORD
        return self.search_treehole(keyword, max_results=max_results, use_cache=use_cache)

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
            phase_timer: Optional[_LiveStatusTimer] = None
            thinking_timer: Optional[_LiveStatusTimer] = None
            try:
                self._llm_request_seq += 1
                request_id = self._llm_request_seq
                start_ts = time.time()
                self._emit_info(
                    f"LLM#{request_id} 开始请求（stream={'on' if stream else 'off'}, messages={len(messages)}, attempt={attempt}）"
                )
                phase_timer = _LiveStatusTimer(f"LLM#{request_id} 等待响应").start()
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
                    first_reasoning_ts = None
                    first_output_ts = None
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
                                        reasoning = self._extract_reasoning_content(delta)
                                        if reasoning and first_reasoning_ts is None:
                                            first_reasoning_ts = time.time()
                                            response_elapsed = phase_timer.stop() if phase_timer else first_reasoning_ts - start_ts
                                            phase_timer = None
                                            self._emit_info(
                                                f"LLM#{request_id} 开始思考: 响应延迟 {response_elapsed:.2f}s"
                                            )
                                            thinking_timer = _LiveStatusTimer(f"LLM#{request_id} 思考中").start()
                                        content = delta.get('content', '')
                                        if content:
                                            if first_output_ts is None:
                                                first_output_ts = time.time()
                                                if thinking_timer:
                                                    thinking_elapsed = thinking_timer.stop()
                                                    thinking_timer = None
                                                    self._emit_info(
                                                        f"LLM#{request_id} 开始输出: 思考 {thinking_elapsed:.2f}s, 首字延迟 {first_output_ts - start_ts:.2f}s"
                                                    )
                                                elif phase_timer:
                                                    response_elapsed = phase_timer.stop()
                                                    phase_timer = None
                                                    self._emit_info(
                                                        f"LLM#{request_id} 开始输出: 响应延迟 {response_elapsed:.2f}s"
                                                    )
                                                else:
                                                    self._emit_info(
                                                        f"LLM#{request_id} 开始输出: 首字延迟 {first_output_ts - start_ts:.2f}s"
                                                    )
                                            cb = self.stream_callback or callback
                                            if cb:
                                                cb(content)
                                            else:
                                                print(content, end='', flush=True)
                                            full_content += content
                                except json.JSONDecodeError:
                                    continue

                    if thinking_timer:
                        thinking_elapsed = thinking_timer.stop()
                        thinking_timer = None
                        self._emit_info(f"LLM#{request_id} 思考结束: {thinking_elapsed:.2f}s")
                    if phase_timer:
                        phase_timer.stop()
                        phase_timer = None
                    if not (self.stream_callback or callback):
                        print()
                    self._emit_info(
                        f"LLM#{request_id} 完成: {time.time() - start_ts:.2f}s, 输出 {len(full_content)} 字符"
                    )
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
                    if phase_timer:
                        phase_timer.stop()
                        phase_timer = None
                    self._emit_info(
                        f"LLM#{request_id} 完成: {time.time() - start_ts:.2f}s（非流式）"
                    )
                    return result["choices"][0]["message"]["content"]
            except Exception as e:
                if thinking_timer:
                    thinking_timer.stop()
                    thinking_timer = None
                if phase_timer:
                    phase_timer.stop()
                    phase_timer = None
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
            phase_timer: Optional[_LiveStatusTimer] = None
            thinking_timer: Optional[_LiveStatusTimer] = None
            try:
                self._llm_request_seq += 1
                request_id = self._llm_request_seq
                start_ts = time.time()
                req_data = dict(data)
                if use_parallel_tool_calls:
                    # Hint model/runtime to emit multiple tool calls in one assistant turn.
                    req_data["parallel_tool_calls"] = True
                self._emit_info(
                    f"LLM#{request_id} 开始工具请求（stream={'on' if stream else 'off'}, tools={len(tools)}, messages={len(messages)}, attempt={attempt}）"
                )
                phase_timer = _LiveStatusTimer(f"LLM#{request_id} 等待响应").start()

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
                    first_reasoning_ts = None
                    first_output_ts = None

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
                                            if first_reasoning_ts is None:
                                                first_reasoning_ts = time.time()
                                                response_elapsed = phase_timer.stop() if phase_timer else first_reasoning_ts - start_ts
                                                phase_timer = None
                                                self._emit_info(
                                                    f"LLM#{request_id} 开始思考: 响应延迟 {response_elapsed:.2f}s"
                                                )
                                                thinking_timer = _LiveStatusTimer(f"LLM#{request_id} 思考中").start()
                                            full_reasoning_content += reasoning_content
                                        if delta.get('tool_calls') and tool_calls_raw is None:
                                            tool_calls_raw = delta['tool_calls']
                                        content = delta.get('content', '')
                                        if content:
                                            if first_output_ts is None:
                                                first_output_ts = time.time()
                                                if thinking_timer:
                                                    thinking_elapsed = thinking_timer.stop()
                                                    thinking_timer = None
                                                    self._emit_info(
                                                        f"LLM#{request_id} 开始输出: 思考 {thinking_elapsed:.2f}s, 首字延迟 {first_output_ts - start_ts:.2f}s"
                                                    )
                                                elif phase_timer:
                                                    response_elapsed = phase_timer.stop()
                                                    phase_timer = None
                                                    self._emit_info(
                                                        f"LLM#{request_id} 开始输出: 响应延迟 {response_elapsed:.2f}s"
                                                    )
                                                else:
                                                    self._emit_info(
                                                        f"LLM#{request_id} 开始输出: 首字延迟 {first_output_ts - start_ts:.2f}s"
                                                    )
                                            if self.stream_callback:
                                                self.stream_callback(content)
                                            else:
                                                print(content, end='', flush=True)
                                            full_content += content
                                except json.JSONDecodeError:
                                    continue

                    if thinking_timer:
                        thinking_elapsed = thinking_timer.stop()
                        thinking_timer = None
                        self._emit_info(f"LLM#{request_id} 思考结束: {thinking_elapsed:.2f}s")
                    if phase_timer:
                        phase_timer.stop()
                        phase_timer = None
                    if tool_calls_raw:
                        self._emit_info(
                            f"LLM#{request_id} 完成: {time.time() - start_ts:.2f}s, tool_calls={len(tool_calls_raw)}, 输出 {len(full_content)} 字符"
                        )
                        return {
                            "content": full_content,
                            "tool_calls": tool_calls_raw,
                            "reasoning_content": full_reasoning_content,
                        }
                    self._emit_info(
                        f"LLM#{request_id} 完成: {time.time() - start_ts:.2f}s, tool_calls=0, 输出 {len(full_content)} 字符"
                    )
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
                    if phase_timer:
                        phase_timer.stop()
                        phase_timer = None
                    self._emit_info(
                        f"LLM#{request_id} 完成: {time.time() - start_ts:.2f}s（非流式）"
                    )
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
                    if thinking_timer:
                        thinking_timer.stop()
                        thinking_timer = None
                    if phase_timer:
                        phase_timer.stop()
                        phase_timer = None
                    use_parallel_tool_calls = False
                    continue

                if thinking_timer:
                    thinking_timer.stop()
                    thinking_timer = None
                if phase_timer:
                    phase_timer.stop()
                    phase_timer = None
                if attempt < LLM_RETRY_MAX_ATTEMPTS and self._is_retryable_llm_error(e):
                    self._wait_before_llm_retry(attempt, e)
                    continue

                error_detail = self._extract_llm_error_detail(e)
                print(f"{AGENT_PREFIX}错误: LLM API 调用失败 - {error_detail}")
                return {"content": None, "tool_calls": None, "reasoning_content": None}

    # ------------------------------------------------------------------ #
    #         Internal: execute one round of search-then-answer           #
    # ------------------------------------------------------------------ #

    def _build_research_system_prompt(self, profile: str) -> str:
        profile_cfg = self.MODE_PROFILES[profile]
        return (
            self._build_auto_search_system_prompt()
            + "\n\n[当前模式]\n"
            + f"- 模式: {profile_cfg['label']}\n"
            + f"- 风格: {profile_cfg['query_style']}\n"
            + f"- 工具轮次预算: {profile_cfg['max_tool_rounds']}\n"
            + f"- search_treehole 预算: {profile_cfg['search_budget']}\n"
            + f"- 每次 search_treehole 拉取帖子上限: {profile_cfg['search_results_per_call']}\n"
            + f"- 最终上下文帖子上限: {profile_cfg['context_post_limit']}\n"
        )

    def _build_turn_messages(self, user_question: str, profile: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_research_system_prompt(profile),
            }
        ]
        recent_turns = self._build_recent_turns_snippet()
        if recent_turns:
            messages.append(
                {
                    "role": "system",
                    "content": "以下是最近几轮对话摘要，请只把它当作线索，不要被旧结论绑死：\n" + recent_turns,
                }
            )
        compact_context = self._build_compact_context_snippet(
            query=user_question,
            limit=min(SESSION_CONTEXT_MAX_POSTS, 18),
        )
        if compact_context:
            messages.append(
                {
                    "role": "system",
                    "content": "以下是当前会话的 compact 证据缓存（来自已抓到的帖子和评论摘要）：\n" + compact_context,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"当前用户问题：{user_question}\n"
                    "请在预算内自主决定是先泛化还是先聚焦。优先一次性发出多条 tool_calls，提高效率；"
                    "如果当前已有证据足够，也可以直接停止调用工具并准备总结。"
                ),
            }
        )
        return messages

    def _hydrate_all_posts_with_comments(
        self,
        posts: List[Dict[str, Any]],
        max_comments: int = -1,
    ) -> List[Dict[str, Any]]:
        normalized_posts = [self._normalize_post_metadata(dict(post)) for post in posts if post.get("pid")]
        if not normalized_posts:
            return []
        max_workers = max(1, min(COMMENT_FETCH_MAX_PARALLEL, len(normalized_posts)))
        results: Dict[int, List[Dict[str, Any]]] = {}
        self._reset_comment_fetch_stats(len(normalized_posts))
        batch_start = time.time()
        self._emit_info(
            f"批量评论抓取开始: 候选 {len(normalized_posts)} 帖, "
            f"并发 {max_workers}, 每帖上限 {'全部' if max_comments == -1 else max_comments}"
        )

        def fetch(post: Dict[str, Any]) -> Any:
            self._record_comment_worker_started()
            try:
                pid = int(post["pid"])
                comments = self._load_comments_for_post(post, max_comments=max_comments)
                return pid, comments
            finally:
                self._record_comment_worker_finished()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(fetch, post): int(post["pid"]) for post in normalized_posts}
            finished = 0
            comment_count = 0
            progress_step = max(1, min(50, len(normalized_posts) // 20 or 1))
            last_progress_ts = batch_start
            for future in as_completed(future_map):
                pid = future_map[future]
                try:
                    _, comments = future.result()
                    results[pid] = comments
                    comment_count += len(comments)
                    self._inc_comment_stat("completed_posts")
                except Exception as e:
                    self._inc_comment_stat("failed_posts")
                    self._emit_info(f"批量评论抓取失败: pid={pid}, error={e}")
                finally:
                    finished += 1
                    now = time.time()
                    if (
                        finished == 1
                        or finished == len(normalized_posts)
                        or finished % progress_step == 0
                        or now - last_progress_ts >= 3.0
                    ):
                        with self._comment_metrics_lock:
                            api_requests = int(self._comment_fetch_stats.get("api_requests", 0))
                            failed_posts = int(self._comment_fetch_stats.get("failed_posts", 0))
                            active_posts = int(self._comment_fetch_stats.get("active_posts", 0))
                            cache_hits = int(self._comment_fetch_stats.get("cache_hits", 0))
                        elapsed = max(0.001, now - batch_start)
                        self._emit_info(
                            f"批量评论抓取进度: {finished}/{len(normalized_posts)} 帖, "
                            f"评论 {comment_count} 条, 失败 {failed_posts}, 缓存命中 {cache_hits}, "
                            f"活跃 {active_posts}, 请求 {api_requests}, "
                            f"耗时 {elapsed:.2f}s, 吞吐 {finished / elapsed:.2f} posts/s"
                        )
                        last_progress_ts = now

        self._log_comment_fetch_stats()
        hydrated = []
        for post in normalized_posts:
            pid = int(post["pid"])
            post["comments"] = results.get(pid, post.get("comments") or post.get("comment_list") or [])
            hydrated.append(post)
        return hydrated

    def _summarize_source_refs(self, posts: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        refs = []
        for post in posts[:limit]:
            refs.append(
                {
                    "pid": post.get("pid"),
                    "text": self._compact_text_preview(post.get("text", ""), max_len=100),
                }
            )
        return refs

    @staticmethod
    def _parse_tool_call_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        try:
            args = json.loads(tool_call["function"].get("arguments", "{}") or "{}")
            return args if isinstance(args, dict) else {}
        except Exception:
            return {}

    def _execute_tool_calls_batch(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        working_posts: List[Dict[str, Any]],
        search_history: List[Dict[str, Any]],
        profile: str,
        search_used: int,
        search_budget: int,
        search_results_per_call: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Execute one assistant tool-call batch concurrently, then merge in original order."""
        profile_cfg = self.MODE_PROFILES[profile]
        prepared: List[Dict[str, Any]] = []

        for idx, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            name = function.get("name", "")
            args = self._parse_tool_call_args(tool_call)
            item: Dict[str, Any] = {
                "idx": idx,
                "tool_call": tool_call,
                "name": name,
                "args": args,
                "execute": False,
                "skip_content": None,
            }

            if name == "search_treehole":
                if search_used >= search_budget:
                    item["skip_content"] = "search_treehole 预算已用尽，请改用已有帖子继续分析，或直接准备总结。"
                else:
                    keyword = str(args.get("keyword", "") or "").strip()
                    reason = str(args.get("reason", "") or "").strip()
                    search_used += 1
                    self._conversation_search_count += 1
                    iteration = self._conversation_search_count
                    item.update({"execute": True, "keyword": keyword, "reason": reason, "iteration": iteration})
                    search_history.append(
                        {
                            "iteration": iteration,
                            "mode": profile,
                            "action": "search",
                            "keyword": keyword,
                            "reason": reason,
                        }
                    )
                    self._append_task_memory(
                        f"- [{profile}] search#{iteration}: keyword={keyword or '-'} reason={reason or '-'}"
                    )
            elif name == "get_post_by_pid":
                pid = self._int_or_default(args.get("pid"), 0)
                include_comments = bool(args.get("include_comments", True))
                max_comments = self._int_or_default(
                    args.get("max_comments"),
                    profile_cfg["comment_limit"],
                )
                reason = str(args.get("reason", "") or "").strip()
                item.update(
                    {
                        "execute": pid > 0,
                        "pid": pid,
                        "include_comments": include_comments,
                        "max_comments": max_comments,
                        "reason": reason,
                    }
                )
                if pid <= 0:
                    item["skip_content"] = "get_post_by_pid 缺少有效 PID。"
                search_history.append(
                    {
                        "iteration": self._conversation_search_count,
                        "mode": profile,
                        "action": "get_post",
                        "pid": pid,
                        "reason": reason,
                    }
                )
                self._append_task_memory(
                    f"- [{profile}] get_post: pid={pid} include_comments={include_comments} max_comments={max_comments}"
                )
            elif name == "get_comments_by_pid":
                pid = self._int_or_default(args.get("pid"), 0)
                max_comments = self._int_or_default(
                    args.get("max_comments"),
                    profile_cfg["comment_limit"],
                )
                sort = str(args.get("sort", "asc") or "asc").lower()
                reason = str(args.get("reason", "") or "").strip()
                item.update(
                    {
                        "execute": pid > 0,
                        "pid": pid,
                        "max_comments": max_comments,
                        "sort": sort,
                        "reason": reason,
                    }
                )
                if pid <= 0:
                    item["skip_content"] = "get_comments_by_pid 缺少有效 PID。"
                search_history.append(
                    {
                        "iteration": self._conversation_search_count,
                        "mode": profile,
                        "action": "get_comment",
                        "pid": pid,
                        "max_comments": max_comments,
                        "reason": reason,
                    }
                )
                self._append_task_memory(
                    f"- [{profile}] get_comment: pid={pid} max_comments={max_comments} sort={sort}"
                )
            else:
                item["skip_content"] = f"未知工具: {name or '-'}"

            prepared.append(item)

        def run_tool(item: Dict[str, Any]) -> Dict[str, Any]:
            name = item["name"]
            if name == "search_treehole":
                return {
                    "posts": self.search_treehole(
                        item.get("keyword", ""),
                        max_results=search_results_per_call,
                    )
                }
            if name == "get_post_by_pid":
                return {
                    "post": self.get_post_by_pid(
                        item["pid"],
                        include_comments=item["include_comments"],
                        max_comments=item["max_comments"],
                    )
                }
            if name == "get_comments_by_pid":
                return {
                    "comments": self.get_comments_by_pid(
                        item["pid"],
                        max_comments=item["max_comments"],
                        sort=item["sort"],
                    )
                }
            return {}

        results_by_idx: Dict[int, Dict[str, Any]] = {}
        executable = [item for item in prepared if item.get("execute")]
        if executable:
            max_workers = max(1, min(len(executable), max(4, int(COMMENT_FETCH_MAX_PARALLEL))))
            if len(executable) > 1:
                self._emit_info(f"并发执行本轮 {len(executable)} 个工具调用，workers={max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {executor.submit(run_tool, item): item for item in executable}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        results_by_idx[item["idx"]] = future.result()
                    except Exception as e:
                        results_by_idx[item["idx"]] = {"error": str(e)}

        batch_new_posts: List[Dict[str, Any]] = []
        for item in prepared:
            tool_call = item["tool_call"]
            name = item["name"]
            result = results_by_idx.get(item["idx"], {})
            result_summary = item.get("skip_content")
            if not result_summary:
                if result.get("error"):
                    result_summary = f"{name} 执行失败: {result['error']}"
                elif name == "search_treehole":
                    posts = result.get("posts") or []
                    self._upsert_posts(working_posts, posts)
                    self._upsert_session_posts(posts)
                    batch_new_posts.extend(posts)
                    keyword = item.get("keyword", "")
                    if posts:
                        result_summary = (
                            f"搜索到 {len(posts)} 个帖子。以下是带评论预览的轻量摘要：\n"
                            + self._format_search_brief(
                                posts,
                                max_items=min(12, profile_cfg["context_post_limit"]),
                                include_comment_preview=True,
                                preview_comments_per_post=3,
                            )
                        )
                    else:
                        result_summary = f"未找到关于「{keyword}」的相关帖子。"
                elif name == "get_post_by_pid":
                    post = result.get("post")
                    pid = item.get("pid", 0)
                    include_comments = item.get("include_comments", True)
                    max_comments = item.get("max_comments", profile_cfg["comment_limit"])
                    if post:
                        self._upsert_posts(working_posts, [post])
                        self._upsert_session_posts([post])
                        batch_new_posts.append(post)
                        result_summary = (
                            f"已获取帖子 #{pid}：\n\n"
                            f"{format_post_to_text(post, include_comments=include_comments, max_comments=max_comments)}"
                        )
                    else:
                        result_summary = f"未能获取帖子 #{pid}。"
                elif name == "get_comments_by_pid":
                    pid = item.get("pid", 0)
                    comments = result.get("comments") or []
                    for post in working_posts:
                        if self._int_or_default(post.get("pid"), 0) == pid:
                            post["comments"] = comments
                            break
                    if pid in self._session_posts:
                        self._session_posts[pid]["comments"] = comments

                    preview_lines = [
                        f"{idx}. {self._compact_text_preview(comment.get('text', ''), max_len=80)}"
                        for idx, comment in enumerate(comments[:10], 1)
                    ]
                    result_summary = (
                        f"已获取帖子 #{pid} 评论，共 {len(comments)} 条（展示前 10 条）：\n"
                        + ("\n".join(preview_lines) if preview_lines else "无评论或拉取失败")
                    )
                else:
                    result_summary = f"未知工具: {name or '-'}"

            messages_entry = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": result_summary,
            }
            messages.append(messages_entry)

        return batch_new_posts, search_used

    def _execute_research_loop(
        self,
        messages: List[Dict[str, Any]],
        working_posts: List[Dict[str, Any]],
        search_history: List[Dict[str, Any]],
        user_question: str,
        profile: str,
    ) -> None:
        profile_cfg = self.MODE_PROFILES[profile]
        max_tool_rounds = int(profile_cfg["max_tool_rounds"])
        search_budget = int(profile_cfg["search_budget"])
        search_results_per_call = int(profile_cfg["search_results_per_call"])
        search_used = 0
        tool_round = 0

        while tool_round < max_tool_rounds:
            self._inject_task_memory_snapshot(messages)
            response = self._call_llm_with_tools(
                messages,
                [self.SEARCH_TOOL, self.GET_POST_TOOL, self.GET_COMMENT_TOOL],
                stream=False,
            )

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                break

            tool_round += 1
            messages.append(
                self._build_assistant_message(
                    content=response.get("content"),
                    tool_calls=tool_calls,
                    reasoning_content=response.get("reasoning_content"),
                )
            )

            batch_new_posts: List[Dict[str, Any]] = []
            batch_new_posts, search_used = self._execute_tool_calls_batch(
                messages=messages,
                tool_calls=tool_calls,
                working_posts=working_posts,
                search_history=search_history,
                profile=profile,
                search_used=search_used,
                search_budget=search_budget,
                search_results_per_call=search_results_per_call,
            )
            if batch_new_posts:
                self._emit_info(
                    f"本轮工具调用后新增/更新 {len({p.get('pid') for p in batch_new_posts if p.get('pid')})} 帖，当前会话累计 {len(self._session_posts)} 帖"
                )
                self._save_session()

        if search_used >= search_budget:
            self._emit_info(f"{self.MODE_PROFILES[profile]['label']} 搜索预算已用满。")

    def _run_profiled_turn(self, user_question: str, profile: str) -> Dict[str, Any]:
        profile_cfg = self.MODE_PROFILES[profile]
        self._ensure_session_for_mode(profile, user_question)
        self._append_task_memory(f"- user_turn[{profile}]: {user_question}")
        self._append_session_turn("user", user_question, mode=profile)

        messages = self._build_turn_messages(user_question, profile)
        working_posts = self._rank_posts_for_query(
            list(self._session_posts.values()),
            query=user_question,
            limit=max(profile_cfg["context_post_limit"], 20),
        )
        search_history: List[Dict[str, Any]] = []

        self._emit_info(f"进入 {profile_cfg['label']} 模式，开始检索...")
        self._execute_research_loop(
            messages=messages,
            working_posts=working_posts,
            search_history=search_history,
            user_question=user_question,
            profile=profile,
        )

        candidate_posts = self._rank_posts_for_query(
            list(self._session_posts.values()),
            query=user_question,
            limit=max(profile_cfg["context_post_limit"] * 2, 24),
        )
        final_context_posts = self._hydrate_posts_for_context(
            candidate_posts,
            profile_cfg["comment_limit"],
            selection_query=user_question,
            context_post_limit=profile_cfg["context_post_limit"],
            selected_limit=profile_cfg["comment_fetch_posts"],
        )
        final_context_text = format_posts_batch(
            final_context_posts,
            include_comments=True,
            max_comments=profile_cfg["comment_limit"],
        )
        self._append_task_memory(
            f"- final_context[{profile}]: posts={len(final_context_posts)} total_session_posts={len(self._session_posts)}"
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"以下是当前问题的最终上下文（共 {len(final_context_posts)} 帖，已补关键评论）：\n\n"
                    f"{final_context_text}"
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "请基于以上已检索内容，用中文给出完整、有条理的回答。只使用已检索到的内容，不要编造；如果证据不足请明确说明。",
            }
        )

        print_separator("-")
        print(f"\n【{profile_cfg['label']} 回答】\n")
        response = self._call_llm_with_tools(messages, tools=[], stream=True)
        final_answer = response.get("content") or ""
        print("\n")
        print_separator("-")

        self._append_session_turn(
            "assistant",
            final_answer,
            mode=profile,
            search_history=search_history,
            source_pids=[post.get("pid") for post in final_context_posts[:20]],
        )
        self._save_session()

        return {
            "answer": final_answer,
            "search_count": self._conversation_search_count,
            "search_history": search_history,
            "sources": self._summarize_source_refs(final_context_posts),
            "num_sources": len(final_context_posts),
            "conversation_turns": sum(1 for turn in self._session_turns if turn.get("role") == "user"),
            "session_id": self._active_session_id,
            "mode": profile,
        }

    def _load_latest_pid_state(self) -> int:
        payload = load_json(LATEST_PID_STATE_FILE) or {}
        pid = int(payload.get("latest_pid") or 0)
        if pid > 0:
            return pid
        session_max = max((int(pid) for pid in self._session_posts.keys()), default=0)
        return max(session_max, RECENT_PID_SCAN_HINT)

    def _save_latest_pid_state(self, latest_pid: int) -> None:
        if latest_pid <= 0:
            return
        save_json(
            {
                "latest_pid": latest_pid,
                "updated_at": self._now_str(),
            },
            LATEST_PID_STATE_FILE,
        )

    @staticmethod
    def _build_pid_probe_steps(initial_step: int) -> List[int]:
        """Build coarse-to-fine PID probe steps, e.g. 5000 -> 1000 -> 200 -> 50 -> 10 -> 1."""
        preferred_steps = [5000, 1000, 200, 50, 10, 1]
        step = max(1, int(initial_step or 1))
        steps: List[int] = [step]
        for preferred in preferred_steps:
            if preferred < step and preferred not in steps:
                steps.append(preferred)
        if steps[-1] != 1:
            steps.append(1)
        return steps

    def _find_existing_anchor_pid(self, seed_pid: int, max_backtrack: int = 1000) -> Tuple[int, int]:
        """Find a confirmed existing PID at or below seed_pid for upward probing."""
        lower_bound = max(1, seed_pid - max(0, max_backtrack))
        probes = 0
        for candidate_pid in range(seed_pid, lower_bound - 1, -1):
            post = self.get_post_by_pid(candidate_pid, include_comments=False, use_cache=False)
            probes += 1
            if post:
                if candidate_pid != seed_pid:
                    self._emit_info(f"起点 PID {seed_pid} 不可见，回退到已确认存在的 PID {candidate_pid}")
                return candidate_pid, probes
        self._emit_info(f"未能在 {seed_pid} 下方 {max_backtrack} 范围内确认存在 PID，暂以 {seed_pid} 为锚点")
        return seed_pid, probes

    def _discover_latest_pid(self) -> int:
        seed_pid = self._load_latest_pid_state()
        steps = self._build_pid_probe_steps(max(1, RECENT_PID_SCAN_STEP))
        best_pid, probes = self._find_existing_anchor_pid(seed_pid)
        miss_threshold = 5
        probe_budget = max(RECENT_PID_SCAN_MAX_PROBES, len(steps) * miss_threshold + 20)
        self._emit_info(f"开始探测最新 PID，起点 {seed_pid}，步长序列 {steps}")

        for step in steps:
            if probes >= probe_budget:
                break
            consecutive_misses = 0
            hits = 0
            candidate_pid = best_pid + step
            while probes < probe_budget and consecutive_misses < miss_threshold:
                post = self.get_post_by_pid(candidate_pid, include_comments=False, use_cache=False)
                probes += 1
                if post:
                    best_pid = candidate_pid
                    hits += 1
                    consecutive_misses = 0
                    candidate_pid = best_pid + step
                else:
                    consecutive_misses += 1
                    candidate_pid += step

            self._emit_info(
                f"PID 探测步长 {step} 完成: 命中 {hits} 次, best={best_pid}, 连续空洞={consecutive_misses}, probes={probes}"
            )

        if probes >= probe_budget:
            self._emit_info("PID 探测达到最大探测次数，使用当前 best 作为最新 PID 估计")

        self._save_latest_pid_state(best_pid)
        self._emit_info(f"最新有效 PID 估计为 {best_pid}")
        return best_pid

    def _collect_recent_posts_by_pid(
        self,
        count: int,
        latest_pid: Optional[int] = None,
        stop_pid: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        latest_pid = latest_pid or self._discover_latest_pid()
        posts: List[Dict[str, Any]] = []
        pid = latest_pid
        attempts = 0
        lower_bound = max(0, int(stop_pid or 0))
        max_attempts = max(count * 8, count + 1000)
        max_workers = max(1, int(PID_FETCH_MAX_PARALLEL))
        batch_size = min(100, max(max_workers, max(1, count) * 2))
        batch_no = 0
        self._emit_info(
            f"最近 PID 并发扫描开始: 目标 {count} 帖, latest_pid={latest_pid}, "
            f"并发 {max_workers}, 提交速率上限 {PID_FETCH_MAX_REQUESTS_PER_SECOND:.2f} req/s"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pid > lower_bound and len(posts) < count and attempts < max_attempts:
                candidate_pids: List[int] = []
                while (
                    pid > lower_bound
                    and attempts + len(candidate_pids) < max_attempts
                    and len(candidate_pids) < batch_size
                ):
                    candidate_pids.append(pid)
                    pid -= 1
                if not candidate_pids:
                    break

                batch_no += 1
                batch_start = time.time()
                future_to_pid = {
                    executor.submit(self.get_post_by_pid, candidate_pid, False, MAX_COMMENTS_PER_POST, True): candidate_pid
                    for candidate_pid in candidate_pids
                }
                batch_results: Dict[int, Dict[str, Any]] = {}
                for future in as_completed(future_to_pid):
                    candidate_pid = future_to_pid[future]
                    try:
                        post = future.result()
                        if post:
                            batch_results[candidate_pid] = post
                    except Exception as e:
                        self._emit_info(f"警告: 并发获取 PID {candidate_pid} 失败: {e}")

                attempts += len(candidate_pids)
                for candidate_pid in candidate_pids:
                    post = batch_results.get(candidate_pid)
                    if post:
                        posts.append(post)
                        if len(posts) >= count:
                            break

                self._emit_info(
                    f"最近 PID 扫描批次 {batch_no}: "
                    f"pid {candidate_pids[0]}->{candidate_pids[-1]}, "
                    f"命中 {len(batch_results)}/{len(candidate_pids)}, "
                    f"累计 {len(posts)}/{count}, 耗时 {time.time() - batch_start:.2f}s"
                )

        misses = attempts - len(posts)
        range_hint = f"{latest_pid}->{lower_bound + 1}" if lower_bound else f"latest_pid={latest_pid}"
        self._emit_info(
            f"最近 PID 扫描完成: 命中 {len(posts)} 帖, 空洞 {misses} 个, 扫描范围 {range_hint}"
        )
        return posts

    @staticmethod
    def _score_daily_post(post: Dict[str, Any]) -> float:
        reply_count = max(0, int(post.get("reply_count", post.get("reply", 0)) or 0))
        star_count = max(0, int(post.get("star_count", post.get("likenum", 0)) or 0))
        pid_bonus = int(post.get("pid", 0) or 0) / 1_000_000.0
        reply_aux = min(reply_count, 50) * 0.12 + max(0, reply_count - 50) * 0.02
        return star_count * 10.0 + reply_aux + pid_bonus

    def _format_daily_ranked_index(
        self,
        posts: List[Dict[str, Any]],
        latest_pid: int,
        previous_latest_pid: int,
        candidate_count: int,
        limit: int = 120,
    ) -> str:
        lines = [
            "# 每日神帖候选排序",
            "",
            f"- previous_latest_pid: {previous_latest_pid}",
            f"- latest_pid: {latest_pid}",
            f"- candidate_count: {candidate_count}",
            "- score: star_count * 10 + capped_reply_aux + tiny_pid_bonus",
            "",
        ]
        for idx, post in enumerate(posts[:limit], 1):
            pid = post.get("pid")
            reply_count = int(post.get("reply_count", post.get("reply", 0)) or 0)
            star_count = int(post.get("star_count", post.get("likenum", 0)) or 0)
            lines.append(
                f"{idx}. pid={pid} | score={self._score_daily_post(post):.2f} | "
                f"star={star_count} | reply={reply_count} | time={post.get('post_time')} | "
                f"{self._compact_text_preview(post.get('text', ''), max_len=120)}"
            )
        return "\n".join(lines)

    def mode_daily_hot_digest(self, recent_post_count: int = DAILY_DIGEST_RECENT_POSTS) -> Dict[str, Any]:
        print_header("模式 1: 每日神帖汇总")
        previous_latest_pid = self._load_latest_pid_state()
        latest_pid = self._discover_latest_pid()
        recent_posts = self._collect_recent_posts_by_pid(
            recent_post_count,
            latest_pid=latest_pid,
        )
        ranked_posts = sorted(recent_posts, key=self._score_daily_post, reverse=True)
        top_posts = ranked_posts[:max(DAILY_DIGEST_TOP_POSTS, 1)]
        self._emit_info(
            f"每日候选 {len(recent_posts)} 帖，按收藏优先排序后取前 {len(top_posts)} 帖进入日报上下文"
        )
        hydrated_posts = self._hydrate_posts_for_context(
            top_posts,
            max_comments=min(MAX_COMMENTS_PER_POST if MAX_COMMENTS_PER_POST > 0 else 10, 10) if MAX_COMMENTS_PER_POST != -1 else 10,
            selection_query="每日神帖 高质量 高收藏 收藏量优先 回复数仅辅助",
            context_post_limit=len(top_posts),
            selected_limit=len(top_posts),
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(DAILY_DIGEST_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        candidates_json_path = os.path.join(run_dir, "daily_candidates.json")
        ranked_index_path = os.path.join(run_dir, "daily_ranked_index.md")
        raw_json_path = os.path.join(run_dir, "daily_hot_posts.json")
        raw_md_path = os.path.join(run_dir, "daily_hot_posts.md")
        save_json(recent_posts, candidates_json_path)
        self._save_text_artifact(
            ranked_index_path,
            self._format_daily_ranked_index(
                ranked_posts,
                latest_pid=latest_pid,
                previous_latest_pid=previous_latest_pid,
                candidate_count=len(recent_posts),
            ),
        )
        save_json(hydrated_posts, raw_json_path)
        self._save_text_artifact(
            raw_md_path,
            format_posts_batch(hydrated_posts, include_comments=True, max_comments=10),
        )

        system_message = (
            "你是北大树洞日报助手。请根据给定的近期帖子，产出一份“每日神帖汇总”。"
            "筛选标准以收藏量和信息质量为主，回复数只作为辅助信号，因为高回复可能来自少数人在评论区聊天。"
            "请说明为什么这些帖子值得看，并明确引用 pid。"
        )
        user_message = (
            f"最近抓取的帖子来自 PID {latest_pid} 附近，共扫描 {len(recent_posts)} 帖有效候选，"
            f"按收藏优先筛出 {len(hydrated_posts)} 帖高质量候选。\n"
            f"请输出一份简洁但信息充分的中文日报。\n\n"
            f"{format_posts_batch(hydrated_posts, include_comments=True, max_comments=10)}"
        )

        print("\n【每日神帖汇总】\n")
        answer = self.call_llm(user_message=user_message, system_message=system_message, stream=True)
        print("\n")
        print_separator("-")

        answer_path = os.path.join(run_dir, "daily_hot_digest.md")
        self._save_text_artifact(answer_path, answer)
        return {
            "answer": answer,
            "search_count": self._conversation_search_count,
            "search_history": [],
            "sources": self._summarize_source_refs(hydrated_posts),
            "num_sources": len(hydrated_posts),
            "artifacts": {
                "candidates": candidates_json_path,
                "ranked_index": ranked_index_path,
                "json": raw_json_path,
                "markdown": raw_md_path,
                "digest": answer_path,
            },
        }

    def mode_thorough_search(
        self,
        keywords: List[str],
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        print_header("模式 4: Thorough Search")
        unique_keywords = []
        seen_keywords: Set[str] = set()
        for keyword in keywords:
            cleaned = str(keyword or "").strip()
            if cleaned and cleaned not in seen_keywords:
                seen_keywords.add(cleaned)
                unique_keywords.append(cleaned)
        if not unique_keywords:
            raise ValueError("thorough search 至少需要一个关键词")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_slug = self._slug_from_text("_".join(unique_keywords), max_len=32)
        run_dir = os.path.join(THOROUGH_SEARCH_DIR, f"{run_id}_{run_slug}")
        os.makedirs(run_dir, exist_ok=True)
        self._emit_info(
            f"Thorough Search 开始: {len(unique_keywords)} 个关键词, "
            f"输出目录 {run_dir}"
        )

        merged_posts: Dict[int, Dict[str, Any]] = {}
        search_history = []
        for idx, keyword in enumerate(unique_keywords, 1):
            before_count = len(merged_posts)
            self._emit_info(f"Thorough Search 搜索关键词 {idx}/{len(unique_keywords)}: {keyword}")
            posts = self.search_treehole_exhaustive(keyword)
            search_history.append({"action": "search_all", "keyword": keyword, "count": len(posts)})
            for post in posts:
                pid = int(post.get("pid", 0) or 0)
                if pid:
                    merged_posts[pid] = post
            self._emit_info(
                f"Thorough Search 关键词完成 {idx}/{len(unique_keywords)}: {keyword}, "
                f"拉取 {len(posts)} 帖, 新增 {len(merged_posts) - before_count} 帖, "
                f"去重后累计 {len(merged_posts)} 帖"
            )

        all_posts = sorted(merged_posts.values(), key=lambda item: int(item.get("pid", 0) or 0), reverse=True)
        self._emit_info(f"Thorough Search 开始全量补评论: 去重后 {len(all_posts)} 帖")
        hydrated_posts = self._hydrate_all_posts_with_comments(all_posts, max_comments=-1)
        self._emit_info(f"Thorough Search 评论补拉完成: {len(hydrated_posts)} 帖")

        corpus_json_path = os.path.join(run_dir, "corpus.json")
        corpus_md_path = os.path.join(run_dir, "corpus.md")
        corpus_index_path = os.path.join(run_dir, "corpus_index.md")
        self._emit_info("Thorough Search 正在保存语料文件...")
        save_json(hydrated_posts, corpus_json_path)
        self._save_text_artifact(corpus_md_path, format_posts_batch(hydrated_posts, include_comments=True, max_comments=-1))
        self._save_text_artifact(
            corpus_index_path,
            self._format_search_brief(
                hydrated_posts,
                max_items=len(hydrated_posts),
                include_comment_preview=True,
                preview_comments_per_post=2,
            ),
        )
        self._emit_info(
            f"Thorough Search 语料已保存: json={corpus_json_path}, markdown={corpus_md_path}, index={corpus_index_path}"
        )

        answer = ""
        answer_path = os.path.join(run_dir, "answer.md")
        if question:
            selected_posts = self._rank_posts_for_query(
                hydrated_posts,
                query=question,
                limit=max(THOROUGH_SEARCH_MAX_CONTEXT_POSTS, 20),
            )
            context_posts = self._hydrate_posts_for_context(
                selected_posts,
                min(MAX_COMMENTS_PER_POST if MAX_COMMENTS_PER_POST > 0 else 12, 12) if MAX_COMMENTS_PER_POST != -1 else 12,
                selection_query=question,
                context_post_limit=THOROUGH_SEARCH_MAX_CONTEXT_POSTS,
                selected_limit=min(len(selected_posts), max(MAX_COMMENT_FETCH_POSTS, 12)),
            )
            system_message = (
                "你是北大树洞语料问答助手。你正在使用用户手工指定关键词抓下来的完整语料回答问题。"
                "请优先引用 pid，并对证据不足处明确说明。"
            )
            user_message = (
                f"关键词: {', '.join(unique_keywords)}\n"
                f"问题: {question}\n\n"
                f"以下是从大语料中筛出的高相关上下文：\n\n"
                f"{format_posts_batch(context_posts, include_comments=True, max_comments=12)}"
            )
            print("\n【Thorough Search 回答】\n")
            answer = self.call_llm(user_message=user_message, system_message=system_message, stream=True)
            print("\n")
            self._save_text_artifact(answer_path, answer)

        return {
            "answer": answer,
            "search_count": self._conversation_search_count,
            "search_history": search_history,
            "sources": self._summarize_source_refs(hydrated_posts),
            "num_sources": len(hydrated_posts),
            "artifacts": {
                "json": corpus_json_path,
                "markdown": corpus_md_path,
                "index": corpus_index_path,
                "answer": answer_path if answer else None,
            },
        }

    # ------------------------------------------------------------------ #
    #              Public modes / backward-compatible wrappers            #
    # ------------------------------------------------------------------ #

    def mode_quick_qa(self, user_question: str) -> Dict[str, Any]:
        print_header("模式 2: 日常 Q&A")
        return self._run_profiled_turn(user_question, profile="quick")

    def mode_deep_research(self, user_question: str) -> Dict[str, Any]:
        print_header("模式 3: Deep Research")
        return self._run_profiled_turn(user_question, profile="deep")

    def mode_auto_search(self, user_question: str) -> Dict[str, Any]:
        """Backward-compatible entrypoint."""
        return self.mode_deep_research(user_question)

    def mode_auto_search_multi_turn(self, user_question: str) -> Dict[str, Any]:
        """Backward-compatible multi-turn entrypoint."""
        return self.mode_quick_qa(user_question)

    def save_conversation(self):
        """Save current session to disk."""
        self._save_session()
        if self._active_session_id:
            self._emit_info(f"会话已保存: {self._active_session_id}")

    def load_conversation(self) -> bool:
        """Load the active or latest session from disk."""
        ok = self._load_latest_session()
        if ok:
            turns = sum(1 for turn in self._session_turns if turn.get("role") == "user")
            self._emit_info(
                f"已恢复会话: {self._active_session_id}（{turns} 轮，{self._conversation_search_count} 次搜索）"
            )
        return ok

    def reset_conversation(self):
        """Start a fresh quick-QA session."""
        self._save_session()
        self._begin_new_session("新会话", self._default_cli_mode)

    def _handle_cli_command(self, user_input: str) -> bool:
        command = user_input.strip()
        if not command.startswith("/"):
            return False

        parts = command.split(maxsplit=1)
        command_name = parts[0].lower()
        command_args = parts[1].strip() if len(parts) > 1 else ""
        known_commands = [
            "/help",
            "/?",
            "/mode",
            "/daily",
            "/thorough",
            "/sessions",
            "/resume",
            "/history",
            "/new",
            "/reset",
            "/save",
            "/quit",
        ]

        if command_name in {"/help", "/?"}:
            print(
                "\n可用命令:\n"
                "/mode quick|deep      切换默认输入模式\n"
                "/daily N              生成每日神帖汇总，例如 /daily 4000\n"
                "/thorough kw1,kw2 | 问题   做 thorough search 并回答\n"
                "/sessions             查看历史会话列表\n"
                "/resume <session_id>  恢复指定会话\n"
                "/history [session_id] 查看当前/指定会话内容\n"
                "/new 或 /reset        新建会话\n"
                "/save                 保存当前会话\n"
                "/quit                 保存并退出\n"
            )
            return True

        if command_name == "/mode":
            if not command_args:
                self._emit_info("用法: /mode quick|deep")
                return True
            mode = command_args.lower()
            if mode not in self.MODE_PROFILES:
                self._emit_info(f"未知模式: {mode}")
                return True
            self._default_cli_mode = mode
            self._emit_info(f"默认模式已切换为 {self.MODE_PROFILES[mode]['label']}")
            return True

        if command_name == "/daily":
            count = DAILY_DIGEST_RECENT_POSTS
            if command_args:
                raw_count = command_args
                bracketed_count = re.fullmatch(r"\[(\d+)\]", raw_count)
                if bracketed_count:
                    raw_count = bracketed_count.group(1)
                if not raw_count.isdigit():
                    self._emit_info("用法: /daily [N]，N 必须是正整数")
                    return True
                count = int(raw_count)
                if count <= 0:
                    self._emit_info("用法: /daily [N]，N 必须是正整数")
                    return True
            self.mode_daily_hot_digest(recent_post_count=count)
            return True

        if command_name == "/thorough":
            if not command_args:
                self._emit_info("用法: /thorough kw1,kw2 | 问题")
                return True
            payload = command_args
            keywords_part, _, question_part = payload.partition("|")
            keywords = [item.strip() for item in re.split(r"[,，、;；\n]+", keywords_part) if item.strip()]
            if not keywords:
                self._emit_info("用法: /thorough kw1,kw2 | 问题，至少需要一个关键词")
                return True
            question = question_part.strip() or None
            self.mode_thorough_search(keywords=keywords, question=question)
            return True

        if command_name == "/sessions":
            print("\n" + self.render_session_list() + "\n")
            return True

        if command_name == "/resume":
            if not command_args:
                self._emit_info("用法: /resume <session_id>")
                return True
            session_id = command_args
            if self._load_session_by_id(session_id):
                self._emit_info(f"已恢复会话: {session_id}")
            else:
                self._emit_info(f"未找到会话: {session_id}")
            return True

        if command_name == "/history":
            session_id = command_args or None
            print("\n" + self.render_session_history(session_id=session_id) + "\n")
            return True

        if command_name in {"/new", "/reset"}:
            self.reset_conversation()
            return True

        if command_name == "/save":
            self.save_conversation()
            return True

        if command_name == "/quit":
            self.save_conversation()
            self._emit_info("退出多轮对话")
            raise KeyboardInterrupt

        suggestion = difflib.get_close_matches(command_name, known_commands, n=1)
        if suggestion:
            self._emit_info(f"未知命令: {command_name}。你是不是想输入 {suggestion[0]}？输入 /help 查看全部命令。")
        else:
            self._emit_info(f"未知命令: {command_name}。输入 /help 查看全部命令。")
        return True

    # ------------------------------------------------------------------ #
    #                      Interactive CLI                                #
    # ------------------------------------------------------------------ #

    def interactive_mode(self):
        """Interactive mode with mode switching and session history."""
        print_header("PKU Treehole RAG Agent")

        if not self.load_conversation():
            self._begin_new_session("启动会话", self._default_cli_mode)

        print(
            f"\n{AGENT_PREFIX}默认模式: {self.MODE_PROFILES[self._default_cli_mode]['label']}\n"
            f"{AGENT_PREFIX}输入 /help 查看命令，直接提问会走默认模式。\n"
        )

        while True:
            try:
                user_question = input("\n你: ").strip()
            except EOFError:
                self.save_conversation()
                break
            if not user_question:
                continue
            try:
                if self._handle_cli_command(user_question):
                    continue
            except KeyboardInterrupt:
                break

            if self._default_cli_mode == "quick":
                self.mode_quick_qa(user_question)
            else:
                self.mode_deep_research(user_question)


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
