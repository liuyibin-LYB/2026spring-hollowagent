"""
Configuration file for PKU Treehole RAG Agent.

Copy this file to config_private.py and fill in your credentials.
config_private.py is gitignored for security.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Treehole Credentials ====================
# Your PKU credentials for Treehole login
USERNAME = "2xx00xxxxx"  # Replace with your student ID
PASSWORD = "xxxxxx"      # Replace with your password

# ==================== LLM API Configuration ====================
# Any OpenAI-compatible chat-completions endpoint should work here.
LLM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LLM_API_BASE = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-pro"

# Optional overrides for non-default compatible backends.
LLM_CHAT_COMPLETIONS_PATH = "/chat/completions"
LLM_EXTRA_HEADERS = {}
LLM_EXTRA_BODY = {}
LLM_ENABLE_PARALLEL_TOOL_CALLS = True
# Unified reasoning knob:
# - OpenAI GPT-5.4-style models: none / low / medium / high / xhigh
# - DeepSeek V4-style models: high / max (low/medium -> high, xhigh -> max, none -> thinking disabled)
LLM_REASONING_EFFORT = "high"
# Optional thinking toggle for compatible backends such as DeepSeek V4.
# Use "auto" in most cases; set to "disabled" to force non-thinking mode.
LLM_THINKING_TYPE = "auto"
LLM_RETRY_MAX_ATTEMPTS = 5
LLM_RETRY_SLEEP_SECONDS = 5

# ==================== Agent Configuration ====================
# --- Quick Tuning（常调参数） ---
# 1) 每次搜索最多拉多少帖子（每次 search_treehole 上限）
MAX_SEARCH_RESULTS = 40

# 2) 最终上下文帖子上限
MAX_CONTEXT_POSTS = 30

# 3) AI 选多少帖子做“深挖评论”
MAX_COMMENT_FETCH_POSTS = 6

# 4) 每帖最多给 LLM 看多少评论（0=不看评论，-1=尽量全量）
MAX_COMMENTS_PER_POST = 5

# 5) 软参考搜索次数（不再驱动硬编码阶段，只作为提示词参考）
# Broad exploration hints: discover aliases/slang/synonyms
BROAD_SEARCH_MIN = 10
BROAD_SEARCH_MAX = 20

# Focused research hints: prioritize high reply/star posts
FOCUSED_SEARCH_MIN = 5
FOCUSED_SEARCH_MAX = 10

# 6) 生成参数
TEMPERATURE = 0.7
MAX_RESPONSE_TOKENS = 4096

# ==================== Rate Limiting ====================
# --- Search pacing ---
# Delay between search requests (seconds) — random uniform in [MIN, MAX]
SEARCH_DELAY_MIN = 1.0
SEARCH_DELAY_MAX = 3.0

# ==================== Cache Configuration ====================
# Enable caching of search results
ENABLE_CACHE = True

# Cache directory
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "cache")

# Cache expiration time (seconds), 1 day = 86400
CACHE_EXPIRATION = 86400

# ==================== Fixed Defaults（通常无需调整） ====================
# 这些参数通常保持默认，日常不建议调整。

# 评论补拉吞吐控制
COMMENT_FETCH_MAX_REQUESTS_PER_SECOND = 20.0
COMMENT_FETCH_MAX_PARALLEL = 10
SEARCH_MAX_REQUESTS_PER_SECOND = 6.0

# 模式预算
QUICK_QA_MAX_TURNS = 5
QUICK_QA_MAX_TOOL_ROUNDS = 4
QUICK_QA_SEARCH_BUDGET = 12
DEEP_RESEARCH_MAX_TOOL_ROUNDS = 10
DEEP_RESEARCH_SEARCH_BUDGET = 30

# 会话与上下文
SESSION_RECENT_TURNS = 5
SESSION_CONTEXT_MAX_POSTS = 40

# Daily 神帖模式（通过最近 PID 扫描）
RECENT_PID_SCAN_HINT = 8000000
RECENT_PID_SCAN_STEP = 120
RECENT_PID_SCAN_MAX_PROBES = 60
DAILY_DIGEST_RECENT_POSTS = 60
DAILY_DIGEST_TOP_POSTS = 12

# Thorough search
# -1 表示尽量抓全；如果关键词过泛，建议在 config_private.py 中改成更小上限
THOROUGH_SEARCH_MAX_RESULTS_PER_KEYWORD = -1
THOROUGH_SEARCH_MAX_CONTEXT_POSTS = 40

# 搜索接口细节
SEARCH_PAGE_LIMIT = 30
SEARCH_COMMENT_LIMIT = 10
INCLUDE_IMAGE_POSTS = True
