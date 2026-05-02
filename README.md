# PKU Treehole Search Agent

基于北大树洞的检索增强问答与研究工具。当前支持四类核心工作流：

1. `每日神贴汇总`
2. `日常 Q&A`
3. `Deep Research`
4. `Thorough Search`

同时内置：

- LLM 请求实时计时：等待响应每 0.1s 刷新；进入思考后单独计时；开始输出后停止计时并直接流式打印
- 历史会话列表、查看、恢复
- compact 上下文缓存，避免对话上下文无限膨胀
- 搜索缓存、评论并发抓取、任务 memory
- `/` 开头输入统一按 CLI 命令处理，输错命令会提示，不会误发给 LLM

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

## 四种模式

### 1. 每日神贴汇总

用途：

- 从上一次记录的最新 PID 出发，探测当前最新 PID
- 扫描近期有效帖子，默认候选量为 `4000`
- 以收藏量为主要质量信号筛选高价值帖子
- 给候选神贴补拉评论后生成日报

具体流程：

1. 读取 `data/latest_pid_state.json` 中的 `latest_pid` 作为探测起点；它只用来加速寻找当前最新 PID，不会限制日报候选扫描范围。
2. 先确认起点 PID 本身存在；如果起点已删除，就向下回退找到一个确实存在的锚点。
3. 从这个已确认存在的 `best_pid` 开始，只向更大的 PID 试探。默认步长序列是 `5000 -> 1000 -> 200 -> 50 -> 10 -> 1`；每个步长连续 5 次 miss 后，回到当前 `best_pid` 并降低步长。
4. 保存新的最新 PID。
5. 从最新 PID 往回扫描，跳过已删除或不可见 PID，收集最多 `DAILY_DIGEST_RECENT_POSTS` 个有效帖子；例如 `/daily 1000` 会尽量收集最近 1000 个有效帖子。
6. 排序时 `star_count` 权重最高，`reply_count` 只作为很小的辅助信号，因为高回复可能只是少数人在评论区聊天。
7. 取前 `DAILY_DIGEST_TOP_POSTS` 个帖子补拉评论，并将这些帖子放入最终 LLM 上下文。

PID 扫描是并发的：任务会批量提交到线程池，`PID_FETCH_MAX_PARALLEL` 控制同时在飞的请求数，`PID_FETCH_MAX_REQUESTS_PER_SECOND` 控制提交速率。单帖结果会写入 `data/pid_post_cache/`，评论会写入 `data/comment_cache/`，所以同一批 `/daily 1000` 第二次运行通常会直接命中缓存；未找到的 PID 也会短时间缓存，避免反复扫删除空洞。

产物保存在 `data/daily_digest/<timestamp>/`：

- `daily_candidates.json`：本次扫描到的全部有效候选帖子
- `daily_ranked_index.md`：按收藏优先排序后的候选索引
- `daily_hot_posts.json`：进入日报上下文的帖子及评论
- `daily_hot_posts.md`：可读版上下文
- `daily_hot_digest.md`：LLM 生成的日报

### 2. 日常 Q&A

用途：

- 五轮以内的连续追问
- 快速检索、快速补评论、快速回答

流程：

1. 构造系统提示、最近几轮对话摘要、compact 证据缓存和本轮用户问题。
2. LLM 在预算内自主调用 `search_treehole`、`get_post_by_pid`、`get_comments_by_pid`。
3. 工具结果会进入本轮消息，也会写入 session 帖子缓存。
4. 回答前按当前问题重新给 session 内帖子排序。
5. LLM 选择最值得补评论的帖子，补拉评论后形成最终上下文。
6. 最后一次 LLM 请求只基于最终上下文回答。

### 3. Deep Research

用途：

- 让 LLM 在更大预算内自主决定研究方向
- 适合渐进式摸底、转向、聚焦和总结

流程与日常 Q&A 相同，但工具轮次、搜索预算、上下文帖子上限和评论补拉上限更高。它不再写死“阶段 A/B”，是否先广后窄由 LLM 根据证据自主决定。

### 4. Thorough Search

用途：

- 用户手工指定关键词
- 尽量抓全这些关键词下的帖子和评论
- 保存大语料文件，再做问答

流程：

1. 对每个关键词运行 exhaustive search。
2. 按 PID 合并去重。
3. 对全部抓到的帖子尽量补拉评论。
4. 保存完整语料。
5. 如果用户同时给了问题，再从完整语料中排序筛选上下文，让 LLM 回答。

产物保存在 `data/thorough_search/<timestamp>_<slug>/`：

- `corpus.json`
- `corpus.md`
- `corpus_index.md`
- `answer.md`（如果同时给了问题）

## 每个模式里“哪些帖子去了哪里”

| 模式 | 初始帖子来源 | 评论补拉 | LLM 筛选 | 最终 LLM 上下文 |
| --- | --- | --- | --- | --- |
| 每日神贴 | 最新 PID 往回扫描得到的有效候选 | 排名前 `DAILY_DIGEST_TOP_POSTS` 的帖子都会补拉 | 代码按收藏优先排序，不让回复数主导 | `daily_hot_posts.json/md` 中的帖子 |
| 日常 Q&A | 当前 session 缓存，加上本轮工具搜索的新帖子 | LLM 从候选中选 `comment_fetch_posts` 个帖子补拉 | LLM 在工具轮中决定搜索、抓单帖、抓评论；回答前再次选择补评论帖子 | 当前问题排序后的 `context_post_limit` 个帖子 |
| Deep Research | 同日常 Q&A，但预算更大 | 同日常 Q&A，上限更高 | 同日常 Q&A | 同日常 Q&A |
| Thorough Search | 用户指定关键词下的全部搜索结果 | 全语料尽量补拉评论 | 若给了问题，代码先按问题排序，再让 LLM 基于筛出的上下文回答 | `answer.md` 对应的筛选上下文 |

运行时会打印：

- LLM 选择补拉评论的 PID
- 实际补拉评论的 PID
- 进入最终 LLM 上下文的 PID
- 每个 Treehole API 工具调用的开始、结束、缓存命中和耗时
- 评论抓取批次的完成吞吐、平均提交速率、瞬时提交峰值和活跃 worker 峰值

## CLI 命令

启动后，直接输入问题会走默认模式（默认是 `quick`）。所有 `/` 开头的输入都会被当作命令处理；如果命令不存在，会提示相近命令或让你查看 `/help`。

```text
/help
/mode quick|deep
/daily N
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
- `/daily N`：扫描最近 `N` 个有效帖子候选并生成日报，例如 `/daily 4000`；直接输入 `/daily` 使用默认 `4000`
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

LLM 请求会分三段显示：

- `等待响应`：请求发出后每 0.1s 刷新，例如 `LLM#3 等待响应: 0.4s`
- `思考中`：收到首个 reasoning 片段后切换到思考计时
- `开始输出`：收到首个正文片段后停止计时，之后只流式打印正文

Treehole API 请求通常很快，所以仍保持开始、结束、缓存命中和耗时的日志方式。

## 上下文与会话管理

现在使用两层持久化：

1. `data/sessions/<session_id>.json`
   - 保存会话标题、模式、轮次、搜索次数、已抓帖子
2. `data/context_cache/<session_id>.json`
   - 保存 compact 过的帖子和评论摘要

这样恢复会话时更像“继续工作”，而不是不断把完整历史堆进 prompt。

## 配置项

`config.py` / `config_private.py` 中常用参数：

```python
MAX_SEARCH_RESULTS = 40
MAX_CONTEXT_POSTS = 30
MAX_COMMENT_FETCH_POSTS = 6
MAX_COMMENTS_PER_POST = 5

SEARCH_DELAY_MIN = 1.0
SEARCH_DELAY_MAX = 1.0
SEARCH_MAX_REQUESTS_PER_SECOND = 40.0
COMMENT_FETCH_MAX_REQUESTS_PER_SECOND = 20.0
COMMENT_FETCH_MAX_PARALLEL = 10
PID_FETCH_MAX_REQUESTS_PER_SECOND = 40.0
PID_FETCH_MAX_PARALLEL = 20
PID_POST_CACHE_EXPIRATION = 7 * 24 * 3600
PID_MISS_CACHE_EXPIRATION = 30 * 60
COMMENT_CACHE_EXPIRATION = 7 * 24 * 3600

QUICK_QA_MAX_TURNS = 5
QUICK_QA_MAX_TOOL_ROUNDS = 4
QUICK_QA_SEARCH_BUDGET = 12

DEEP_RESEARCH_MAX_TOOL_ROUNDS = 10
DEEP_RESEARCH_SEARCH_BUDGET = 30

RECENT_PID_SCAN_HINT = 8000000
RECENT_PID_SCAN_STEP = 5000
RECENT_PID_SCAN_MAX_PROBES = 1500
DAILY_DIGEST_RECENT_POSTS = 4000
DAILY_DIGEST_TOP_POSTS = 12

THOROUGH_SEARCH_MAX_RESULTS_PER_KEYWORD = -1
THOROUGH_SEARCH_MAX_CONTEXT_POSTS = 40
```

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
│   ├── comment_cache/
│   ├── context_cache/
│   ├── daily_digest/
│   ├── pid_post_cache/
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
