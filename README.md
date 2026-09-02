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

## MVP 功能范围

| ACP 方法 | 状态 | 说明 |
|---------|:---:|------|
| `initialize` | ✅ | 握手，声明 baseline 能力 |
| `session/new` | ✅ | 创建 AgentScope Agent 会话，附带 models 状态（前端模型选择器数据源） |
| `session/prompt` | ✅ | 驱动 `reply_stream()`，流式推送 `agent_message_chunk` |
| `session/cancel` | ✅ | 取消当前 prompt task（stop_reason=cancelled） |
| 工具调用流式展示 | ✅ | 内置工具集（Bash/Read/Write/Edit/Grep/Glob）→ `tool_call` / `tool_call_update`，`ACCEPT_EDITS` 自动放行 |
| 其他（权限审批/持久化等） | 🗓 | 见下方 Roadmap |

## 环境变量

| 变量名 | 必填 | 默认值 | 说明 |
|--------|:---:|--------|------|
| `AGENTSCOPE_ACP_PROVIDER` | 否 | `dashscope` | `dashscope` 或 `openai-compat` |
| `DASHSCOPE_API_KEY` | 条件 | — | provider=dashscope 时必填 |
| `OPENAI_API_KEY` | 条件 | — | provider=openai-compat 时必填 |
| `OPENAI_BASE_URL` | 否 | — | openai-compat 自定义端点 |
| `AGENTSCOPE_ACP_MODEL` | 否 | `qwen3.6-plus` | 当前模型 |
| `AGENTSCOPE_ACP_AVAILABLE_MODELS` | 否 | 同当前模型 | 逗号分隔的模型列表，填充 session/new 的 models |
| `AGENTSCOPE_ACP_SYSTEM_PROMPT` | 否 | 内置默认 | Agent 系统提示词 |
| `AGENTSCOPE_ACP_TOOLS` | 否 | 开启 | 是否启用内置工具集；设 `0`/`false`/`no`/`off` 关闭（纯对话无工具） |
| `AGENTSCOPE_ACP_TOOL_NAMES` | 否 | `Bash,Read,Write,Edit,Grep,Glob` | 工具白名单（逗号分隔类名，可选加 `PowerShell`）；引擎新增工具时改这里即可启用，无需发版 |
| `AGENTSCOPE_ACP_SKILLS_DIR` | 否 | 关闭 | Agent Skills 目录（含 `SKILL.md` 的目录，见下方 Skill 章节），启用渐进式披露技能 |
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

依赖中的 `agentscope` 通过本地路径（editable）指向工作区的 `../agentscope(ref-backend)`。

### 冒烟测试

```bash
# 手动 stdin 发送 JSON-RPC（每行一条）
DASHSCOPE_API_KEY=sk-... uv run agentscope-acp <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}
{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/tmp"}}
{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"<上一步返回的id>","prompt":[{"type":"text","text":"你好"}]}}
EOF
```

## agent-work 接入（不改前端代码）

agent-work 的 ACP Client 链路已完备（AcpDriver → AcpAgentTask → probe 预取模型），
只需添加一个自定义 Agent（设置页「自定义 Agent」或 agent_catalog 种子）：

```json
{
  "command": "uv",
  "args": ["run", "--directory", "/home/llm/zhangle/deerflow-agent-work/agentscope-acp", "agentscope-acp"],
  "env": {
    "DASHSCOPE_API_KEY": "sk-...",
    "AGENTSCOPE_ACP_MODEL": "qwen3.6-plus"
  }
}
```

说明：
- Guid 首页的模型选择器数据来自 `probeAgentHandshake`（spawn 子进程 → `initialize` +
  `session/new` → 读取响应中的 `models` 字段），本项目在 `session/new` 响应中返回
  `SessionModelState`（current_model_id + available_models）即可被前端识别。
- 会话恢复：ACP 会话状态在进程内存中（`dict[session_id, Agent]`）；前端重连时若
  `session/load` 未实现（load_session=False），会自动回退 `session/new` 新建会话。

## 源码结构

```
src/agentscope_acp/
├── __init__.py     # 包声明
├── __main__.py     # 入口：asyncio.run → run_agent(AgentScopeAcpAgent())
├── config.py       # 环境变量解析 + AgentScope 模型实例化
├── agent.py        # AgentScopeAcpAgent(acp.Agent)：ACP 方法 + 会话表
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

## Roadmap（后续迭代）

- [x] 工具调用流式展示（ToolCallStart/Delta/End → tool_call/tool_call_update；内置
      Bash/Read/Write/Edit/Grep/Glob 工具集，默认启用，`ACCEPT_EDITS` 自动放行）
- [ ] 权限审批（RequireUserConfirmEvent → session/request_permission；
      UserConfirmResultEvent 回传恢复；把 `ACCEPT_EDITS` 升级为前端可审批）
- [ ] 思考流输出（ThinkingBlockDeltaEvent → agent_thought_chunk）
- [ ] 会话持久化与恢复（AgentState 序列化到本地 JSON → session/load；声明 load_session 能力）
- [ ] 模型切换（session/set_config_option + 动态模型查询接口 —— 调供应商 models API
      填充 available_models，替换静态配置）
- [ ] usage 统计（ModelCallEndEvent → usage_update）
- [ ] session/list / session/close / session/delete
- [ ] MCP 服务器集成（session/new 的 mcp_servers → Toolkit）
- [ ] 结构化输出、图片等多模态 ContentBlock

### 演进方向（记录）

若后续需要多客户端共享会话/集中部署，可将本进程内的 AgentScope 调用替换为对已部署
AgentScope agent_service（FastAPI, `examples/agent_service`）的 HTTP/SSE 调用，
协议翻译层（translate.py）保持不变。

## License

MIT
