# PKU Treehole Search Agent

基于北大树洞的检索增强问答与研究工具。现在支持四类核心工作流：

1. `每日神帖汇总`
2. `日常 Q&A`
3. `Deep Research`
4. `Thorough Search`

同时内置：

- LLM 请求实时状态显示（请求开始、首个思考片段、首个输出片段、总耗时）
- 历史会话列表、查看、恢复
- compact 上下文缓存，避免对话上下文无限膨胀
- 搜索缓存、评论并发抓取、任务 memory

## 快速开始

```bash
git clone git@github.com:SunVapor/pku-treehole-search-agent.git
cd ./pku-treehole-search-agent
bash start.sh
```

或手动：

```bash
pip install requests
cp config.py config_private.py
python agent.py
```

在 `config_private.py` 中填入：

```python
USERNAME = "你的学号"
PASSWORD = "你的密码"
LLM_API_KEY = "sk-xxx"
LLM_API_BASE = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-pro"
```

## 现在的四种模式

### 1. 每日神帖汇总

用途：

- 扫描最近一批 PID
- 按热度/质量筛选高价值帖子
- 生成“今日神帖”汇总

实现要点：

- 没有依赖树洞“热门榜”接口，而是通过最近 PID 扫描获取新帖
- `reply` 越多越偏热度
- `star` 越多越偏质量/收藏价值

产物会保存在 `data/daily_digest/<timestamp>/`。

### 2. 日常 Q&A

用途：

- 五轮以内的连续追问
- 快速检索、快速补评论、快速回答

特点：

- 默认 CLI 模式
- 会把会话保存成独立 session
- 会优先重用当前 session 的 compact 证据缓存

### 3. Deep Research

用途：

- 让 LLM 在预算内自主决定研究方向
- 更强调渐进式摸底、转向、聚焦和总结

特点：

- 不再把阶段 A/B 写死在流程里
- 可以先广后窄，但是否切换由模型自己判断
- 一旦发现关键帖子，就尽快抓取评论

### 4. Thorough Search

用途：

- 用户手工指定关键词
- 把这些关键词下的帖子和评论尽量抓全
- 保存大语料文件，再做问答

产物会保存在 `data/thorough_search/<timestamp>_<slug>/`，包括：

- `corpus.json`
- `corpus.md`
- `corpus_index.md`
- `answer.md`（如果同时给了问题）

这个工作流参考了 [PKU_Treehole_Starred_Saver](https://github.com/EmptyBlueBox/PKU_Treehole_Starred_Saver) 的思路：把“原始抓取产物”和“便于阅读/后处理的导出产物”分开存，并显式关注速率和并发。

## CLI 命令

启动后，直接输入问题会走默认模式（默认是 `quick`）。

可用命令：

```text
/help
/mode quick|deep
/daily [N]
/thorough kw1,kw2 | 问题
/sessions
/resume <session_id>
/history [session_id]
/new
/reset
/save
/quit
```

说明：

- `/mode quick|deep`：切换“直接输入问题”时的默认模式
- `/daily [N]`：扫描最近 `N` 个有效帖子候选并生成日报
- `/thorough kw1,kw2 | 问题`：先抓全关键词语料，再回答问题
- `/sessions`：查看历史会话列表
- `/resume <session_id>`：恢复指定会话
- `/history [session_id]`：查看当前或指定会话内容

## 搜索工具的重要限制

这部分也会同步注入给 LLM：

- `search_treehole` 只能搜主帖正文，搜不到评论里的关键词
- 多关键词搜索是严格匹配，关键词越多越容易漏结果
- 英文更接近完整词匹配，不稳定支持前缀匹配
- 搜索结果默认按发布时间从新到旧排序，不按热度排序

因此推荐：

- 一轮内发出多次短关键词搜索
- 发现关键 PID 后尽快抓单帖和评论
- 不要把希望寄托在“一次超长关键词搜索”上

## 运行时可见性

现在会实时打印：

- LLM 请求开始
- 首个思考片段延迟
- 首个输出片段延迟
- LLM 总耗时
- 每个工具调用的开始/结束/缓存命中
- 评论抓取统计

## 上下文与会话管理

旧版本会把越来越多的完整上下文直接堆进对话历史里。现在改成了两层：

1. `data/sessions/<session_id>.json`
   - 保存会话标题、模式、轮次、搜索次数、已抓帖子
2. `data/context_cache/<session_id>.json`
   - 保存 compact 过的帖子/评论摘要

这样做的好处：

- 恢复会话时更像“继续工作”而不是“继续堆 prompt”
- 历史可浏览、可恢复
- 新一轮回答主要依赖 compact 证据缓存和最近几轮对话

## 配置项

`config.py` / `config_private.py` 中常用的新参数：

```python
MAX_SEARCH_RESULTS = 40
MAX_CONTEXT_POSTS = 30
MAX_COMMENT_FETCH_POSTS = 6
MAX_COMMENTS_PER_POST = 5

SEARCH_DELAY_MIN = 1.0
SEARCH_DELAY_MAX = 3.0
SEARCH_MAX_REQUESTS_PER_SECOND = 6.0

QUICK_QA_MAX_TURNS = 5
QUICK_QA_MAX_TOOL_ROUNDS = 4
QUICK_QA_SEARCH_BUDGET = 12

DEEP_RESEARCH_MAX_TOOL_ROUNDS = 10
DEEP_RESEARCH_SEARCH_BUDGET = 30

RECENT_PID_SCAN_HINT = 8000000
RECENT_PID_SCAN_STEP = 120
RECENT_PID_SCAN_MAX_PROBES = 60
DAILY_DIGEST_RECENT_POSTS = 60
DAILY_DIGEST_TOP_POSTS = 12

THOROUGH_SEARCH_MAX_RESULTS_PER_KEYWORD = -1
THOROUGH_SEARCH_MAX_CONTEXT_POSTS = 40
```

说明：

- 现在不再依赖“硬编码阶段 A/B 次数”驱动主流程
- `BROAD_SEARCH_*` / `FOCUSED_SEARCH_*` 仍保留，主要作为提示词中的软参考

## 项目结构

```text
pku-treehole-search-agent/
├── agent.py
├── agent.md
├── client.py
├── config.py
├── README.md
├── data/
│   ├── cache/
│   ├── context_cache/
│   ├── daily_digest/
│   ├── sessions/
│   ├── task_memory/
│   └── thorough_search/
└── email_bot/
```

## 兼容性说明

- `mode_auto_search()` 现在默认走 `Deep Research`
- `mode_auto_search_multi_turn()` 现在默认走 `日常 Q&A`
- 老的 `save_conversation()` / `load_conversation()` 仍可用，但底层已经切到 session 文件

## License

MIT
