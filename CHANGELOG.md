# 更新日志 (CHANGELOG)


**改动总结1st**
1. **模式 2 多轮对话**
- 新增 `_conversation_history` / `_conversation_searched_posts` 实例属性，跨轮次持久化
- 新增 `mode_auto_search_multi_turn()` — 每次追加 user 消息到同一个 [messages]，LLM 可按需决定是否再搜索
- 新增 `reset_conversation()` 清空对话
- [interactive_mode] 中模式 2 改为内层 `while True` 循环，支持 `/reset` 和 `/quit` 命令
- 原单轮 [mode_auto_search()] 保留，供 email_bot 等非交互场景使用

2. **随机延迟模拟真人**
- config 中 [SEARCH_DELAY = 1.0] → `SEARCH_DELAY_MIN / MAX` + `COMMENT_DELAY_MIN / MAX` 两组区间
- 新增 `_human_delay(min_s, max_s, label)` 方法，用 [random.uniform] 生成随机延迟并打印日志
- **所有树洞 API 调用前**都加了随机延迟：
    - 搜索请求：1.0~3.0s
    - 评论分页：0.3~1.0s
    - 搜索间隔额外 jitter：0.3~1.0s

3. **代码重构**
- 提取 `SEARCH_TOOL` 和 `_AUTO_SEARCH_SYSTEM_PROMPT` 为类常量，避免重复定义
- 提取 `_execute_search_loop()` 方法，消除 [mode_auto_search] 和 `mode_auto_search_multi_turn` 的搜索循环代码重复

**修改汇总2nd（[agent.py]）：**
1. **新增 [from datetime import datetime] 导入**
2. **新增 [CONVERSATION_FILE] 常量** — 路径为 [data/conversation_history.json]
3. **新增 [save_conversation()]** — 将 [_conversation_history] 和 [_conversation_search_count] 序列化为 JSON
4. **新增 [load_conversation()]** — 启动时从磁盘恢复对话，返回是否成功
5. **[reset_conversation()] 增加归档** — 旧记录自动重命名为 [conversation_history_YYYYMMDD_HHMMSS.json]，不会丢失
6. **模式 2 交互逻辑改进**：
    - 进入时自动检测并提示是否恢复上次对话
    - `/quit` 退出时自动存档
    - 新增 `/save` 手动存档命令
    - `/reset` 归档旧记录后重置


## [v2.0.0] - 2026-02-23

### 🚀 新功能: 智能迭代检索（MCP-like Function Calling）

#### 主要特性
- **模式 2 升级**：从简单的"提取关键词→搜索一次"升级为"智能迭代检索"
- **LLM 自主决策**：LLM 自己决定何时搜索、搜索什么关键词
- **多轮迭代**：如果首次搜索结果不满意，LLM 可以自动再次搜索（使用不同关键词）
- **MCP 风格工具调用**：类似 Model Context Protocol，LLM 将搜索功能作为工具使用

#### 工作流程
1. LLM 分析用户问题
2. LLM 决定搜索策略（关键词）
3. 执行搜索，返回结果
4. LLM 评估信息充分性
5. 如需要，LLM 再次搜索（换用不同关键词）
6. 迭代直至信息充足（最多 3 次）
7. 综合所有搜索结果生成最终回答

#### 技术实现
- 新增 `_call_deepseek_with_tools()` 方法支持 function calling API
- 定义 `search_treehole` 作为 LLM 可调用的工具
- 实现迭代循环，处理 `tool_calls` 并反馈结果
- LLM 自主判断是否需要更多信息

#### 配置项
- 新增 `MAX_SEARCH_ITERATIONS` 配置（默认 3 次）
- 控制单次查询的最大搜索轮数

#### 实测效果
以"AI和机器学习相关课程推荐"为例：
- **第 1 次搜索**: "AI 机器学习 课程"
- **第 2 次搜索**: "人工智能 课程 推荐"
- **第 3 次搜索**: "AI引论 机器学习 课程测评"
- 最终找到 34 个不重复帖子，生成综合性回答

#### 适用场景
- 问题复杂，单一关键词覆盖不全
- 需要综合多方面信息
- 用户不确定精确关键词

---

## [v1.2.0] - 2026-02-23

### ✨ 优化评论处理和 Token 估算

#### Bug 修复
- **Token 估算不准确**：`smart_truncate_posts` 现在正确包含评论内容
- 避免超出 LLM 上下文限制

#### 新增功能
- **可配置评论数量**：`MAX_COMMENTS_PER_POST` 配置项
  - 支持 0（禁用评论）
  - 支持正整数（指定数量）
  - 支持 -1（无限制）
- 应用于模式 1 和模式 2

---

## [v1.1.0] - 2026-02-22

### 🤖 邮件机器人

#### 新增功能
- 远程邮件查询系统
- systemd 服务部署
- 非交互模式登录

#### Cookie 管理优化
- 统一 Cookie 存储位置：`~/.treehole_cookies.json`
- 支持非交互式环境运行

#### 项目重组
- 邮件机器人独立文件夹：`email_bot/`
- 完善的文档和部署脚本

---

## [v1.0.0] - 初始版本

### 核心功能
- 三种检索模式：手动、自动、课程测评
- 流式输出
- 搜索缓存
- 中文界面
