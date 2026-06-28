"""
Configuration file for PKU Treehole RAG Agent.

Copy this file to config_private.py and fill in your credentials.
config_private.py is gitignored for security.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Treehole Credentials ====================
# Your PKU credentials for Treehole login.
USERNAME = "2xx00xxxxx"  # Replace with your student ID.
PASSWORD = "xxxxxx"      # Replace with your password.

# ==================== LLM API Configuration ====================
# Any OpenAI-compatible chat-completions endpoint should work here.
# DeepSeek official:
#   LLM_API_BASE = "https://api.deepseek.com"
#   LLM_CHAT_COMPLETIONS_PATH = "/chat/completions"
#   LLM_MODEL = "deepseek-v4-pro"
# SiliconFlow:
#   LLM_API_BASE = "https://api.siliconflow.cn/v1"
#   LLM_CHAT_COMPLETIONS_PATH = "/chat/completions"
#   LLM_MODEL = "<copy the model id from SiliconFlow Models, e.g. deepseek-ai/DeepSeek-V3.2>"
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
# - DeepSeek V4-style models: high / max
# - SiliconFlow: use provider-specific fields in LLM_EXTRA_BODY when needed.
LLM_REASONING_EFFORT = "high"

# Optional thinking toggle for compatible backends such as DeepSeek V4.
# Use "auto" in most cases; set to "disabled" to force non-thinking mode.
LLM_THINKING_TYPE = "auto"
LLM_RETRY_MAX_ATTEMPTS = 5
LLM_RETRY_SLEEP_SECONDS = 5

# ==================== Agent Configuration ====================
# Comment limits: 0 = no comments, -1 = fetch as many as possible.

# Quick Q&A mode.
QUICK_QA_MAX_TURNS = 10
QUICK_QA_MAX_TOOL_ROUNDS = 10
QUICK_QA_SEARCH_BUDGET = 15
QUICK_QA_SEARCH_RESULTS_PER_CALL = 30
QUICK_QA_CONTEXT_POST_LIMIT = 50
QUICK_QA_COMMENT_FETCH_POSTS = 10
QUICK_QA_COMMENTS_PER_POST = 10

# Deep Research mode.
DEEP_RESEARCH_MAX_TOOL_ROUNDS = 20
DEEP_RESEARCH_SEARCH_BUDGET = 30
DEEP_RESEARCH_SEARCH_RESULTS_PER_CALL = 60
DEEP_RESEARCH_CONTEXT_POST_LIMIT = 100
DEEP_RESEARCH_COMMENT_FETCH_POSTS = 20
DEEP_RESEARCH_COMMENTS_PER_POST = 10

# Defaults for direct tool calls outside a quick/deep profile.
DEFAULT_SEARCH_RESULTS_PER_CALL = 60
DEFAULT_COMMENT_FETCH_POSTS = 10
DEFAULT_COMMENTS_PER_POST = 10

# Session context.
SESSION_RECENT_TURNS = 5
SESSION_CONTEXT_MAX_POSTS = 100

# Quick Q&A prompt search-count hints; these are not hard limits.
QUICK_QA_BROAD_SEARCH_HINT_MIN = 4
QUICK_QA_BROAD_SEARCH_HINT_MAX = 8
QUICK_QA_FOCUSED_SEARCH_HINT_MIN = 2
QUICK_QA_FOCUSED_SEARCH_HINT_MAX = 4

# Deep Research prompt search-count hints; these are not hard limits.
DEEP_RESEARCH_BROAD_SEARCH_HINT_MIN = 10
DEEP_RESEARCH_BROAD_SEARCH_HINT_MAX = 20
DEEP_RESEARCH_FOCUSED_SEARCH_HINT_MIN = 5
DEEP_RESEARCH_FOCUSED_SEARCH_HINT_MAX = 10

# Generation settings.
TEMPERATURE = 0.7
MAX_RESPONSE_TOKENS = 16384

# ==================== Rate Limiting ====================
# All network-heavy Treehole paths use the same throughput knobs.
REQUEST_MAX_PARALLEL = 20
REQUEST_MAX_REQUESTS_PER_SECOND = 40.0
TREEHOLE_REQUEST_TIMEOUT = (10, 30)

# ==================== Cache Configuration ====================
ENABLE_CACHE = True
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "cache")
CACHE_EXPIRATION = 7 * 24 * 3600

# ==================== Fixed Defaults ====================
PID_POST_CACHE_EXPIRATION = 7 * 24 * 3600
PID_MISS_CACHE_EXPIRATION = 30 * 60
COMMENT_CACHE_EXPIRATION = 7 * 24 * 3600

# Daily digest mode.
RECENT_PID_SCAN_HINT = 8000000
RECENT_PID_SCAN_STEP = 5000
RECENT_PID_SCAN_MAX_PROBES = 1500
DAILY_DIGEST_RECENT_POSTS = 4000
DAILY_DIGEST_TOP_POSTS = 10
DAILY_DIGEST_COMMENTS_PER_POST = 10

# Thorough search mode.
THOROUGH_SEARCH_MAX_RESULTS_PER_KEYWORD = 4000
THOROUGH_SEARCH_MAX_CONTEXT_POSTS = 40
THOROUGH_SEARCH_CONTEXT_COMMENTS_PER_POST = 10

# Treehole search API details.
SEARCH_PAGE_LIMIT = 30
SEARCH_COMMENT_LIMIT = 10
INCLUDE_IMAGE_POSTS = True
