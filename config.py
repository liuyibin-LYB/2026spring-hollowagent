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

# ==================== DeepSeek API Configuration ====================
# Get your API key from: https://platform.deepseek.com/
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # or "deepseek-reasoner"

# ==================== Agent Configuration ====================
# --- Quick Tuning（模式2常调参数） ---
# 1) 每次搜索最多拉多少帖子（每次 search_treehole 上限）
MAX_SEARCH_RESULTS = 40

# 2) 最终上下文帖子上限（优先来自阶段B检索到的高质量帖子）
MAX_CONTEXT_POSTS = 30

# 3) AI 选多少帖子做“深挖评论”
MAX_COMMENT_FETCH_POSTS = 6

# 4) 每帖最多给 LLM 看多少评论（0=不看评论，-1=尽量全量）
MAX_COMMENTS_PER_POST = 5

# 5) 两阶段搜索次数（A:宽泛探索，B:高质量聚焦）
# Stage A (broad exploration): discover aliases/slang/synonyms
BROAD_SEARCH_MIN = 10
BROAD_SEARCH_MAX = 20

# Stage B (high-quality focus): prioritize high reply/star posts
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

# 搜索接口细节
SEARCH_PAGE_LIMIT = 30
SEARCH_COMMENT_LIMIT = 10
INCLUDE_IMAGE_POSTS = True
