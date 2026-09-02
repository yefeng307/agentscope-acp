# agentscope-acp

> ACP (Agent Client Protocol) Agent —— 将 AgentScope Agent 包装为 ACP stdio 子进程，供 agent-work（AionUI）等 ACP 客户端直接调用。

## 项目简介

`agentscope-acp` 是一个 **ACP Agent 端实现**（Python）。它在进程内直接实例化 AgentScope 的
`Agent`（ReAct 循环），把 `reply_stream()` 事件流翻译为 ACP `session/update` 通知，
通过 stdin/stdout 上的 JSON-RPC 2.0（NDJSON）与 ACP 客户端通信。

```
agent-work (ACP Client)                     agentscope-acp (本项目)
┌──────────────────┐   spawn + JSON-RPC    ┌────────────────────────────┐
│ AcpDriver        │ ────────────────────→ │ acp.Agent 子类              │
│ (packages/server │ ←──────────────────── │   session/update 流式通知   │
│  /agent/acp/)    │   stdio (NDJSON)      │        │ 进程内调用          │
└──────────────────┘                       │        ▼                   │
                                           │ AgentScope Agent            │
                                           │   reply_stream() 事件流      │
                                           └────────────────────────────┘
```

## 功能范围

| ACP 方法 | 状态 | 说明 |
|---------|:---:|------|
| `initialize` | ✅ | 握手，声明 load_session 能力 |
| `session/new` | ✅ | 创建 AgentScope Agent 会话，附带 models 状态（前端模型选择器数据源） |
| `session/prompt` | ✅ | 驱动 `reply_stream()`，流式推送 `agent_message_chunk` |
| `session/cancel` | ✅ | 取消当前 prompt task（stop_reason=cancelled） |
| 工具调用流式展示 | ✅ | 内置工具集（Bash/Read/Write/Edit/Grep/Glob）→ `tool_call` / `tool_call_update`；放行策略由 `AGENTSCOPE_ACP_PERMISSION_MODE` 控制 |
| 权限审批 | ✅ | `request_permission` 四选项（本次/永久放行、本次/永久拒绝）→ 永久项转 PermissionRule 回传引擎 |
| 思考流 | ✅ | ThinkingBlockDeltaEvent → `agent_thought_chunk`（不污染正文） |
| 会话持久化 | ✅ | AgentState → 本地 JSON；`session/load` 恢复（声明 load_session 能力） |
| @文件引用 | ✅ | `resource_link` 块经 `fs/read_text_file` 拉取内容（按客户端 fs 能力门控）；`resource` 内嵌块直接纳入正文；解析失败跳过不阻断 |
| 模型切换 | ✅ | `session/set_config_option` 运行时换模型（响应回传新配置） |
| usage 统计 | ✅ | 每轮 `usage_update`（输入+输出 token） |
| `session/list` / `close` | ✅ | 列表（cwd 过滤）/ 关闭（保存状态 + 断开 MCP） |
| MCP 集成 | ✅ | `session/new` 的 mcp_servers（stdio/http）→ 引擎 MCPClient |


## 环境变量

| 变量名 | 必填 | 默认值 | 说明 |
|--------|:---:|--------|------|
| `OPENAI_API_KEY` | 是 | — | OpenAI 兼容 API key（DashScope compatible-mode、DeepSeek、vLLM 等端点通用） |
| `OPENAI_BASE_URL` | 否 | — | OpenAI 兼容端点地址（缺省走 openai 官方端点） |
| `AGENTSCOPE_ACP_MODEL` | 否 | `qwen3.6-plus` | 当前模型 |
| `AGENTSCOPE_ACP_AVAILABLE_MODELS` | 否 | 同当前模型 | 逗号分隔的模型列表，填充 session/new 的 models |
| `AGENTSCOPE_ACP_SYSTEM_PROMPT` | 否 | 内置默认 | Agent 系统提示词 |
| `AGENTSCOPE_ACP_SKILLS_DIR` | 否 | 关闭 | Agent Skills 目录（含 `SKILL.md` 的目录，见下方 Skill 章节），启用渐进式披露技能 |
| `AGENTSCOPE_ACP_PERMISSION_MODE` | 否 | `ask` | `accept_edits`（编辑器操作自动放行，其余拦截）或 `ask`（逐操作审批，默认） |
| `AGENTSCOPE_ACP_SESSIONS_DIR` | 否 | `~/.agentscope-acp/sessions` | AgentState 持久化目录（每会话一个 JSON） |
| `AGENTSCOPE_ACP_LOG` | 否 | 关闭 | 文件日志路径（stdout 被 ACP 协议占用，绝不写 stdout） |

## Skill（Agent Skills）

设置 `AGENTSCOPE_ACP_SKILLS_DIR` 指向一个目录，其中的每个子目录是一个标准
Agent Skill（Anthropic 的 SKILL.md 格式，与 Claude Code / Codex / pi 等生态互通）：

```
skills/                  ← AGENTSCOPE_ACP_SKILLS_DIR 指向这里
├── pdf-processing/
│   ├── SKILL.md         # frontmatter 必填 name + description
│   └── scripts/...      # 可选资源（相对路径引用）
└── git-commit-style/
    └── SKILL.md
```

AgentScope 的 `LocalSkillLoader` 扫描每个 `SKILL.md`，把 `name`/`description`
注入系统提示词（渐进式披露：全文本体按需通过内置 `skill_viewer` 工具读取）。
目录不存在时仅记录警告、按无 skill 启动，不会崩溃。

## 快速开始

### 安装依赖

```bash
cd agentscope-acp
uv sync
```

依赖中的 `agentscope` 使用官方 PyPI 发布版（`2.0.7.post1`）

### 冒烟测试

```bash
# 手动 stdin 发送 JSON-RPC（每行一条）
OPENAI_API_KEY=sk-... uv run agentscope-acp <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}
{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/tmp"}}
{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"<上一步返回的id>","prompt":[{"type":"text","text":"你好"}]}}
EOF
```

## agent-work 接入

agent-work 的 ACP Client 链路已完备（AcpDriver → AcpAgentTask → probe 预取模型），
安装本包后添加一个自定义 Agent（设置页「自定义 Agent」或 agent_catalog 种子）：

```bash
uv tool install agentscope-acp
```

```json
{
  "command": "agentscope-acp",
  "args": [],
  "nativeSkillsDirs": ["~/.agentscope-acp/skills"],
  "env": {
    "OPENAI_API_KEY": "sk-...",
    "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
    "AGENTSCOPE_ACP_MODEL": "DeepSeek-V4-Flash",
    "AGENTSCOPE_ACP_SKILLS_DIR": "~/.agentscope-acp/skills"
  }
}
```

说明：
- 启动方式与 pi-agent 一致（seed.ts 里 `command: "pi-acp"`）：agent-work 用
  `spawn(command, args, { env })` 拉起子进程，command 通过服务器进程的 PATH 解析。
  若 agent-work 以服务方式运行，需保证安装目录（uv tool 默认 `~/.local/bin`）在其
  PATH 中，或 command 写安装后的绝对路径。
- Skill：agent-work 会把前端启用的 skills 软链进 `nativeSkillsDirs`（与 pi-agent 的
  `~/.pi/agent/skills/` 同机制），本进程通过 `AGENTSCOPE_ACP_SKILLS_DIR` 读同一目录
  的 `SKILL.md`——两者必须指向同一路径。
- Guid 首页的模型选择器数据来自 `probeAgentHandshake`（spawn 子进程 → `initialize` +
  `session/new` → 读取响应中的 `models` 字段），本项目在 `session/new` 响应中返回
  `SessionModelState`（current_model_id + available_models）即可被前端识别。
- 会话恢复：本进程实现 `session/load`（声明 load_session 能力，AgentState 从本地
  JSON 恢复）；进程重启后内存会话表清空，已落盘的状态仍可通过 `session/load` 恢复。

## 离线交付（wheel 包）

本体是纯 Python wheel（`py3-none-any`，跨平台跨 Python 版本），但依赖树里含二进制
扩展（pydantic-core、numpy、aiohttp 等，带 `cpXX`/平台标签），交付包必须按**目标
Python 版本 + 平台**生成：

```bash
# 1. 打本体 wheel（生成 dist/agentscope_acp-0.1.0-py3-none-any.whl）
uv build

# 2. 按目标环境下载全部依赖（示例：Python 3.13 + Windows x64）
uv pip download --python-version 3.13 --platform win_amd64 \
    --python-platform windows \
    dist/agentscope_acp-0.1.0-py3-none-any.whl -d dist/deps

# 3. 本体与依赖放进同一目录，整体交付 dist/deps
cp dist/agentscope_acp-0.1.0-py3-none-any.whl dist/deps/
```

目标机离线安装（无需网络）：

```bash
uv tool install --no-index --find-links dist/deps agentscope-acp
# 或
pip install --no-index --find-links dist/deps agentscope_acp-0.1.0-py3-none-any.whl
```

Linux 目标把 `--platform win_amd64 --python-platform windows` 换成
`--platform manylinux_2_17_x86_64 --python-platform linux`（或对应平台标签）；
Python 版本不同则改 `--python-version`。

## 源码结构

```
src/agentscope_acp/
├── __init__.py     # 包声明
├── __main__.py     # 入口：asyncio.run → run_agent(AgentScopeAcpAgent())
├── config.py       # 环境变量解析 + 模型/权限/MCP 构建
├── agent.py        # AgentScopeAcpAgent(acp.Agent)：ACP 方法 + 会话/权限/持久化
└── translate.py    # AgentScope 事件 → ACP SessionUpdate（纯函数，可单测）

tests/
├── test_translate.py   # 翻译层单元测试（事件构造 → ACP update 形状断言）
├── test_agent.py       # ACP 方法测试（Fake Agent + 录制型连接）
└── smoke_stdio.py      # 真实进程 stdio 协议冒烟测试
```

## 测试

```bash
uv run pytest                    # 单元测试（translate/agent 层，无需 API key）
uv run python tests/smoke_stdio.py  # 真实进程 stdio 冒烟（假 key，验证协议栈与错误处理）
```

冒烟测试会用 SDK 客户端 spawn 真实进程，验证 initialize 握手、session/new 模型列表、
prompt 流式推送与模型调用失败时的 in-band 错误消息（进程不崩溃）。

## 待办（Roadmap）

已实现功能见上方「功能范围」表，剩余待办及留待原因：

- [ ] 多模态 ContentBlock（图片输入）：协议有 `ImageBlock` 类型但生态零实现——
      QwenPaw ACP 只抽 text（图片块被静默丢弃）、agent-work host 不发图片；
      引擎侧有 `DataBlock` 底子，做成将是首个支持多模态的 ACP 实现
- [ ] 动态模型查询：引擎 `ModelClient.list_models()` 是扫描本地 YAML 卡片目录，
      QwenPaw 的模型列表同样来自 provider 配置文件（配置态）；调供应商 models API
      需按 provider 自建调用，目前用 `AGENTSCOPE_ACP_AVAILABLE_MODELS` 静态配置
- [ ] session/delete：SDK 0.10 接口未暴露该方法，QwenPaw 与 agent-work 协议层
      均无实现，等生态对齐后补

## License

MIT
