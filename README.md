# PKU Treehole Search Agent

基于北大树洞的 RAG（检索增强生成）智能问答系统，当前聚焦模式2自动检索问答。

**🆕 新功能：邮件机器人** - 通过邮件远程查询，随时随地获取课程信息！

## ⚡ 快速开始

### 方式 1: 命令行交互

```bash
git clone git@github.com:SunVapor/pku-treehole-search-agent.git
cd ./pku-treehole-search-agent
bash start.sh
```

### 方式 2: 邮件机器人 🆕

通过邮件远程查询，无需登录服务器！

```bash
cd ~/pku-treehole-search-agent/email_bot

# 1. 配置邮箱
cp email_config_template.py email_config.py
# 编辑 email_config.py 填入邮箱信息

# 2. 启动邮件机器人
bash start_email_bot.sh

# 或部署为系统服务（开机自启）
sudo bash deploy_service.sh
```

发送邮件到配置的邮箱，主题包含 `树洞` 即可自动回复！

详见 [email_bot/README.md](email_bot/README.md)

### 手动配置

1. **安装依赖**
   ```bash
   pip install requests
   ```

2. **配置账号**
   ```bash
   cp config.py config_private.py
   ```
   
   编辑 `config_private.py`：
   ```python
   USERNAME = "你的学号"
   PASSWORD = "你的密码"
   DEEPSEEK_API_KEY = "sk-xxx..."  # 从 https://platform.deepseek.com/ 获取
   ```

3. **运行 Agent**
   ```bash
   python3 agent.py
   ```

## 🎯 功能特性

### 核心模式

#### 模式 2: 智能自动检索 🆕⭐
- **只需输入自然语言问题**
- **LLM 自主决策搜索策略**
- **支持两阶段多轮搜索（A宽泛探索 + B高质量聚焦）**
- **工作流程**：
   1. LLM 分析问题并决定搜索关键词
   2. 阶段A宽泛探索，识别黑话/别称/简称
   3. 阶段B高质量聚焦，优先高 reply/star 帖子
   4. 对高价值帖子补拉评论
   5. 综合检索结果给出最终回答
- **类似 MCP 工具调用模式**：LLM 像使用工具一样调用搜索功能

### 核心特性

- ✅ 流式输出（实时显示 AI 回答）
- ✅ 检索内容预览（先看数据源，再看分析）
- ✅ 搜索结果缓存（避免重复请求）
- ✅ Cookie 持久化（免重复登录）
- ✅ 智能 Token 优化


## ⚙️ 配置说明

建议先改这一组模式2常调参数（在 `config_private.py` 中修改）：

```python
# --- Quick Tuning（模式2常调）---
MAX_SEARCH_RESULTS = 40              # 每次搜索最多拉取帖子数
MAX_CONTEXT_POSTS = 30               # 最终上下文帖子上限（优先阶段B帖子）
MAX_COMMENT_FETCH_POSTS = 6          # AI 选中“深挖评论”的帖子上限
MAX_COMMENTS_PER_POST = 5            # 每帖评论上限（0禁用，-1尽量全量）
BROAD_SEARCH_MIN = 10
BROAD_SEARCH_MAX = 20
FOCUSED_SEARCH_MIN = 5
FOCUSED_SEARCH_MAX = 10
TEMPERATURE = 0.7
MAX_RESPONSE_TOKENS = 4096
SEARCH_DELAY_MIN = 1.0
SEARCH_DELAY_MAX = 3.0
ENABLE_CACHE = True
CACHE_EXPIRATION = 86400

# --- Fixed Defaults（通常不改）---
COMMENT_FETCH_MAX_REQUESTS_PER_SECOND = 20.0
COMMENT_FETCH_MAX_PARALLEL = 10
SEARCH_PAGE_LIMIT = 30
SEARCH_COMMENT_LIMIT = 10
INCLUDE_IMAGE_POSTS = True
```

## 📁 项目结构

```
pku-treehole-search-agent/
├── agent.md               # Agent 持久经验库（黑话、策略、工具说明）
├── README.md              # 项目文档
├── start.sh               # 一键启动脚本
├── config.py              # 配置模板
├── config_private.py      # 私有配置（自动创建）
├── client.py              # 树洞 API 客户端
├── agent.py               # RAG Agent 主逻辑
├── utils.py               # 工具函数
├── email_bot/             # 邮件机器人 🆕
│   ├── README.md          # 邮件机器人文档
│   ├── bot_email.py       # 邮件机器人主程序
│   ├── email_config_template.py  # 配置模板
│   └── ...                # 其他脚本和配置
└── data/
   ├── cache/             # 搜索结果缓存
   └── task_memory/       # 单次任务/对话临时 memory（memory_时间戳.md）
```

## 🔧 API 实现细节

### 树洞搜索 API

**端点**: `GET /chapi/api/v3/hole/list_comments`

**参数**:
- `keyword`: 搜索关键词
- `page`: 页码
- `limit`: 每页帖子数
- `comment_limit`: 每个帖子返回的评论数

**当前实现补充**:
- 支持分页拉取，直到达到 `MAX_SEARCH_RESULTS` 或无更多结果
- 支持通过 `INCLUDE_IMAGE_POSTS` 过滤图片帖（不下载图片文件）
- 模式2采用“两阶段策略”：先进行 10-20 次宽泛探索（识别黑话/别称），再进行 5-10 次高质量聚焦（优先 reply/star）
- 阶段A第一次搜索会展示“每帖最多 3 条评论预览”（来自 search 接口返回的 `comment_list`），用于快速识别黑话语境，后续不重复预览
- 在已知 PID 的情况下，模型可直接调用 `get_post_by_pid` 获取单帖，减少无效搜索上下文
- 在已确定高价值 PID 后，模型可调用 `get_comments_by_pid` 精确补拉评论
- 模式2在 `MAX_SEARCH_RESULTS` 范围内让 AI 选择 `MAX_COMMENT_FETCH_POSTS` 个高价值帖子，再按 `MAX_CONTEXT_POSTS` 组装上下文并仅对入选上下文的已选帖子补拉评论
- 补拉评论阶段使用受控并发 + 请求速率上限，默认通过 `COMMENT_FETCH_MAX_PARALLEL` 与 `COMMENT_FETCH_MAX_REQUESTS_PER_SECOND` 限制压力
- 自动检索的系统提示词会注入 `agent.md`（持久经验库）与单次任务临时 memory（`data/task_memory/`）

**响应格式**:
```json
{
  "code": 20000,
  "data": {
    "list": [
      {
        "pid": 8001234,
        "text": "帖子内容",
        "comment_total": 45,
        "comment_list": [...]
      }
    ],
    "total": 15
  }
}
```

### 认证系统

#### 登录流程
1. **OAuth 登录**: `oauth_login(username, password)` → 获取 token
2. **SSO 登录**: `sso_login(token)` → 获取 authorization
3. **额外验证**（如需要）:
   - SMS 验证: `send_message()` + `login_by_message(code)`
   - Mobile Token: `login_by_token(code)` ⚠️ 注意：参数名为 `code`

#### Cookie 持久化
- 默认保存在项目目录下的 `.treehole_cookies.json`
- 如需自定义路径，可在初始化 `TreeholeRAGAgent` 时传入 `cookies_file`
- 自动加载和保存，避免频繁登录

#### 非交互模式 🆕
```python
# 后台服务部署时使用非交互模式
agent = TreeholeRAGAgent(interactive=False)

# 交互模式（命令行使用）
agent = TreeholeRAGAgent(interactive=True)  # 默认
```

**用途**:
- `interactive=False`: 无法读取 stdin，登录失败时直接返回（适合systemd服务）
- `interactive=True`: 可以提示用户输入验证码/token（适合命令行）

**首次部署**:
```bash
# 1. 先交互式登录一次，保存 cookies
python3 agent.py

# 2. 然后部署为服务（会自动使用保存的 cookies）
cd email_bot && sudo bash deploy_service.sh
```

### 评论获取 API

**端点**: `GET /api/pku_comment_v3/{pid}`

**参数**:
- `page`: 页码
- `limit`: 每页评论数
- `sort`: 排序方式（`asc` / `desc`）

**特性**:
- 模式2会按策略对高价值帖子补拉评论
- 支持分页，自动合并结果

### 发给 AI 的帖子打包字段

在模式2的上下文构造中，Agent 会把帖子和评论整理成稳定字段后再发送给 LLM，包含：

- 帖子编号：`pid`
- 帖子 reply 数：`reply_count`（来自 `reply`）
- 帖子 star 数：`star_count`（来自 `likenum`）
- 帖子发表时间：`post_time`（由 `timestamp` 格式化）
- 评论回复时间：`reply_time`（由评论 `timestamp` 格式化）
- 是否有图片：`has_image`（由 `type` / `media_ids` 推断）

这些字段会以文本元数据形式写入上下文，便于 LLM 在回答时引用。

## 🚨 故障排除

### 问题 1: 登录失败 / 需要令牌验证

**现象**: 登录时提示 "Mobile token:" 或 "请进行令牌验证"

**解决方案**:
```bash
# 删除旧 cookie
rm ./.treehole_cookies.json

# 交互式重新登录
python3 agent.py

# 输入你的 PKU 手机令牌（6位数字，从 PKU Helper App 获取）
```

**邮件机器人部署**:
```bash
# 1. 先在命令行交互式登录，保存 cookies
python3 agent.py

# 2. 然后重启邮件机器人服务
sudo systemctl restart treehole-email-bot
```

### 问题 2: DeepSeek API 错误

**检查**:
- API Key 是否正确
- 网络连接是否正常
- 账户余额是否充足

### 问题 3: 搜索限流

**解决**: 增加请求延迟
```python
# 在 config_private.py 中
SEARCH_DELAY_MIN = 1.5
SEARCH_DELAY_MAX = 3.5
```

### 问题 4: 邮件机器人无法启动

**检查日志**:
```bash
# 查看服务状态
sudo systemctl status treehole-email-bot

# 查看详细日志
tail -f ~/pku-treehole-search-agent/logs/bot.log
```

**常见原因**:
- **EOF when reading a line**: Cookies 已过期，需要交互式重新登录
- **Failed to login**: 检查 `config_private.py` 中的账号密码
- **IMAP/SMTP error**: 检查 `email_config.py` 中的邮箱授权码

## 💡 注意事项

1. **隐私安全**
   - `config_private.py` 已加入 `.gitignore`
   - 不要将包含密码的文件提交到 Git

2. **费用控制**
   - DeepSeek API 按 Token 计费
   - 通过 `MAX_CONTEXT_POSTS` 控制成本

3. **搜索缓存**
   - 默认缓存 24 小时
   - 缓存文件保存在 `data/cache/`

## 📝 开源协议

MIT License
