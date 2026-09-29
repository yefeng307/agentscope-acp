# -*- coding: utf-8 -*-
"""Environment-based configuration for agentscope-acp.

Configuration comes from three layers, highest priority first:

1. environment variables (agent-work's ``agent_catalog`` env row, so ACP
   clients keep per-agent control),
2. an optional YAML config file (see ``find_config_file`` for the search
   order),
3. built-in defaults.

stdout is reserved for the ACP protocol — logging goes to stderr or a
file only.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import SecretStr

logger = logging.getLogger(__name__)

ENV_MODEL = "AGENTSCOPE_ACP_MODEL"
ENV_AVAILABLE_MODELS = "AGENTSCOPE_ACP_AVAILABLE_MODELS"
ENV_SYSTEM_PROMPT = "AGENTSCOPE_ACP_SYSTEM_PROMPT"
ENV_TOOLS = "AGENTSCOPE_ACP_TOOLS"
ENV_TOOL_NAMES = "AGENTSCOPE_ACP_TOOL_NAMES"
ENV_SKILLS_DIR = "AGENTSCOPE_ACP_SKILLS_DIR"
ENV_PERMISSION_MODE = "AGENTSCOPE_ACP_PERMISSION_MODE"
ENV_SESSIONS_DIR = "AGENTSCOPE_ACP_SESSIONS_DIR"
ENV_LOG = "AGENTSCOPE_ACP_LOG"
ENV_CONFIG = "AGENTSCOPE_ACP_CONFIG"

# Config-file search order (first existing file wins; only ONE file is
# loaded). AGENTSCOPE_ACP_CONFIG points at an explicit file: a missing or
# unparseable file is then a hard startup error. The two search paths are
# optional — problems there are warned and skipped.
CONFIG_FILE_PROJECT = "agentscope-acp.yaml"
# agent-work spawns the process with cwd=workspace, so the project-local
# file above is per-workspace; this one is the per-user global fallback.
CONFIG_FILE_USER = "~/.agentscope-acp/config.yaml"

PERMISSION_ACCEPT_EDITS = "accept_edits"
PERMISSION_ASK = "ask"
PERMISSION_MODES = (PERMISSION_ACCEPT_EDITS, PERMISSION_ASK)

DEFAULT_MODEL = "qwen3.6-plus"

DEFAULT_SESSIONS_DIR = "~/.agentscope-acp/sessions"

# The default tool set. Explicit by design: the tool set is the agent's
# capability boundary (Bash/Write carry permission implications), and the
# ACP translation layer maps tool names onto ToolKinds. New tools are
# opt-in via AGENTSCOPE_ACP_TOOL_NAMES rather than silently gained on an
# engine upgrade.
DEFAULT_TOOL_NAMES = ("Bash", "Read", "Write", "Edit", "Grep", "Glob")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant powered by AgentScope. "
    "Reply in the user's language with clear, well-structured Markdown."
)


@dataclass(frozen=True)
class ModelEntry:
    """One selectable model advertised to ACP clients."""

    model_id: str
    name: str


@dataclass
class AcpConfig:
    """Runtime configuration resolved from environment variables."""

    api_key: SecretStr
    model: str
    base_url: str | None = None
    available_models: list[ModelEntry] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    enable_tools: bool = True
    tool_names: list[str] | None = None
    skills_dir: str | None = None
    permission_mode: str = PERMISSION_ACCEPT_EDITS
    sessions_dir: str = DEFAULT_SESSIONS_DIR
    log_path: str | None = None

    @classmethod
    def from_env(cls) -> "AcpConfig":
        # OpenAI-compatible only: any endpoint that speaks the OpenAI chat
        # completions format (DashScope compatible-mode, DeepSeek, vLLM,
        # Ollama's OpenAI endpoint, ...) via OPENAI_API_KEY/OPENAI_BASE_URL
        # (or the ``api_key``/``base_url`` keys of the config file).
        found = find_config_file()
        file_values = load_config_file(*found) if found else {}

        api_key = SecretStr(
            _pick(
                os.environ.get("OPENAI_API_KEY", ""),
                file_values.get("api_key"),
                "",
            ),
        )
        base_url = (
            _pick(
                os.environ.get("OPENAI_BASE_URL", ""),
                file_values.get("base_url"),
                "",
            )
            or None
        )
        model = _pick(
            os.environ.get(ENV_MODEL, ""),
            file_values.get("model"),
            DEFAULT_MODEL,
        )

        raw_list = _pick(
            os.environ.get(ENV_AVAILABLE_MODELS, ""),
            _as_csv(file_values.get("available_models")),
            "",
        )
        if raw_list:
            entries = [
                ModelEntry(model_id=part.strip(), name=part.strip())
                for part in raw_list.split(",")
                if part.strip()
            ]
        else:
            entries = [ModelEntry(model_id=model, name=model)]
        if model not in {e.model_id for e in entries}:
            entries.insert(0, ModelEntry(model_id=model, name=model))

        raw_names = _pick(
            os.environ.get(ENV_TOOL_NAMES, ""),
            _as_csv(file_values.get("tool_names")),
            "",
        )
        tool_names = (
            [part.strip() for part in raw_names.split(",") if part.strip()]
            or None
        )

        permission_mode = _pick(
            os.environ.get(ENV_PERMISSION_MODE, ""),
            file_values.get("permission_mode"),
            PERMISSION_ACCEPT_EDITS,
        ).lower()
        if permission_mode not in PERMISSION_MODES:
            logger.warning(
                "unknown %s %r — falling back to %r",
                ENV_PERMISSION_MODE,
                permission_mode,
                PERMISSION_ACCEPT_EDITS,
            )
            permission_mode = PERMISSION_ACCEPT_EDITS

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            available_models=entries,
            system_prompt=_pick(
                os.environ.get(ENV_SYSTEM_PROMPT, ""),
                file_values.get("system_prompt"),
                DEFAULT_SYSTEM_PROMPT,
            ),
            enable_tools=_parse_bool(
                _pick(
                    os.environ.get(ENV_TOOLS, ""),
                    file_values.get("tools"),
                    "",
                ),
                True,
            ),
            tool_names=tool_names,
            # expanduser like sessions_dir: hosts seed env values with "~"
            # (agent-work's nativeSkillsDirs and AGENTSCOPE_ACP_SKILLS_DIR
            # point at the same ~/.agentscope-acp/skills directory), and
            # os.path.isdir would silently fail on the unexpanded form.
            skills_dir=os.path.expanduser(
                _pick(
                    os.environ.get(ENV_SKILLS_DIR, ""),
                    file_values.get("skills_dir"),
                    "",
                ),
            )
            or None,
            permission_mode=permission_mode,
            sessions_dir=_pick(
                os.environ.get(ENV_SESSIONS_DIR, ""),
                file_values.get("sessions_dir"),
                DEFAULT_SESSIONS_DIR,
            ),
            log_path=_pick(
                os.environ.get(ENV_LOG, ""),
                file_values.get("log"),
                "",
            )
            or None,
        )


def find_config_file() -> tuple[Path, bool] | None:
    """Locate the YAML config file to load.

    Returns ``(path, explicit)``. ``explicit`` is True when the path came
    from ``AGENTSCOPE_ACP_CONFIG`` — a missing or unparseable file is then
    a hard startup error. For search-path hits (False) the caller treats
    problems as warnings only. ``None`` means no config file.
    """
    raw = os.environ.get(ENV_CONFIG, "").strip()
    if raw:
        return Path(raw).expanduser(), True
    for candidate in (CONFIG_FILE_PROJECT, CONFIG_FILE_USER):
        path = Path(candidate).expanduser()
        if path.is_file():
            return path, False
    return None


def load_config_file(path: Path, explicit: bool) -> dict[str, Any]:
    """Parse a YAML config file into a flat mapping.

    Failure handling: explicit files raise ``ValueError`` (aborts startup
    with exit code 2), search-path files log a warning and return ``{}``
    so the agent still starts on env/defaults.
    """
    try:
        import yaml  # declared in pyproject; also a transitive dep of agentscope

        # utf-8-sig tolerates the UTF-8 BOM Notepad adds on Windows.
        with open(path, encoding="utf-8-sig") as fh:
            raw = yaml.safe_load(fh)
    except Exception as exc:
        message = f"cannot load config file {path}: {exc}"
        if explicit:
            raise ValueError(message) from None
        logger.warning("%s — ignoring", message)
        return {}
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        message = f"config file {path} must contain a top-level mapping"
        if explicit:
            raise ValueError(message)
        logger.warning("%s — ignoring", message)
        return {}
    return dict(raw)


def _pick(env_value: str, file_value: Any, default: str) -> str:
    """First non-empty of: env variable, config-file value, default."""
    if env_value and env_value.strip():
        return env_value.strip()
    if file_value is not None and str(file_value).strip():
        return str(file_value).strip()
    return default


def _as_csv(value: Any) -> str:
    """Normalize a config value (list or string) to comma-separated text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(
            str(item).strip() for item in value if str(item).strip()
        )
    return str(value).strip()


def sessions_path(config: AcpConfig) -> Path:
    """Resolve the session persistence directory from ``config``."""
    return Path(
        config.sessions_dir,
    ).expanduser() if config.sessions_dir else Path(
        DEFAULT_SESSIONS_DIR,
    ).expanduser()


def _parse_bool(raw: str, default: bool) -> bool:
    """Tolerant boolean parsing for on/off env switches."""
    if not raw or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def build_toolkit(
    enable_tools: bool,
    skills_dir: str | None = None,
    tool_names: list[str] | None = None,
    mcps: list[Any] | None = None,
):
    """Build the built-in coding tool set, or ``None`` for a tool-less agent.

    ``tool_names`` is a whitelist of built-in tool class names; ``None``
    (default) enables :data:`DEFAULT_TOOL_NAMES`. Unknown names are warned
    and skipped, so new AgentScope tools can be opted in via
    ``AGENTSCOPE_ACP_TOOL_NAMES`` without a code change — they are never
    enabled implicitly.

    ``skills_dir`` registers Agent Skills (folders containing a ``SKILL.md``
    with ``name``/``description`` frontmatter) via
    ``Toolkit.skills_or_loaders``; see README for the directory layout. A
    missing directory is ignored with a warning so a stale env value never
    breaks startup.

    ``mcps`` registers already-connected :class:`~agentscope.mcp.MCPClient`
    instances (built by :func:`build_mcp_clients` from the ACP
    ``session/new`` ``mcp_servers`` payload).

    Imports are deferred so a tool-less configuration (and the unit tests
    that avoid AgentScope's tool stack) never pay the import cost.
    """
    if not enable_tools:
        return None
    from agentscope.skill import LocalSkillLoader
    from agentscope.tool import (
        Bash,
        Edit,
        Glob,
        Grep,
        PowerShell,
        Read,
        Toolkit,
        Write,
    )

    # Tool-name registry. Task* tools are intentionally absent: they drive
    # AgentScope's internal task system, not ACP-facing workflows.
    _TOOL_CLASSES = {
        "Bash": Bash,
        "PowerShell": PowerShell,
        "Read": Read,
        "Write": Write,
        "Edit": Edit,
        "Grep": Grep,
        "Glob": Glob,
    }

    names = list(tool_names) if tool_names else list(DEFAULT_TOOL_NAMES)
    tools = []
    for name in names:
        cls = _TOOL_CLASSES.get(name)
        if cls is None:
            logger.warning(
                "unknown tool %r — skipping (available: %s)",
                name,
                ", ".join(_TOOL_CLASSES),
            )
            continue
        tools.append(cls())

    skills_or_loaders = None
    if skills_dir:
        if os.path.isdir(skills_dir):
            # scan_subdir=True accepts both layouts: the directory being
            # one skill itself, or containing multiple skill subdirectories
            # (the standard Agent Skills ecosystem layout).
            skills_or_loaders = [
                LocalSkillLoader(directory=skills_dir, scan_subdir=True),
            ]
        else:
            logger.warning(
                "skills dir %r does not exist — ignoring %s",
                skills_dir,
                ENV_SKILLS_DIR,
            )

    return Toolkit(
        tools=tools,
        skills_or_loaders=skills_or_loaders,
        mcps=mcps or [],
    )


def build_mcp_clients(mcp_servers: list[Any] | None) -> list[Any]:
    """Convert ACP ``mcp_servers`` entries into AgentScope ``MCPClient``s.

    Pure configuration conversion — no IO. Connection is performed by
    :func:`connect_mcp_clients`. ``stdio`` and ``http`` transports are
    supported; ``sse``/``acp`` servers are unsupported by the AgentScope
    engine and skipped with a warning.
    """
    if not mcp_servers:
        return []

    from agentscope.mcp import HttpMCPConfig, MCPClient, StdioMCPConfig

    clients: list[Any] = []
    for server in mcp_servers:
        # McpServerStdio carries no discriminator ``type`` field (unlike
        # the Http/Sse/Acp variants), so default to stdio.
        server_type = getattr(server, "type", None) or "stdio"
        name = getattr(server, "name", None) or server_type or "mcp"
        try:
            if server_type == "stdio":
                env_list = getattr(server, "env", None) or []
                client = MCPClient(
                    name=name,
                    is_stateful=True,
                    mcp_config=StdioMCPConfig(
                        command=getattr(server, "command", ""),
                        args=getattr(server, "args", None) or None,
                        env={
                            item.name: item.value for item in env_list
                        } or None,
                    ),
                )
            elif server_type == "http":
                headers = getattr(server, "headers", None) or []
                client = MCPClient(
                    name=name,
                    is_stateful=False,
                    mcp_config=HttpMCPConfig(
                        url=getattr(server, "url", ""),
                        headers={
                            item.name: item.value for item in headers
                        } or None,
                    ),
                )
            else:
                logger.warning(
                    "unsupported MCP server type %r (%s) — skipping",
                    server_type,
                    name,
                )
                continue
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "invalid MCP server config for %r: %s — skipping",
                name,
                exc,
            )
            continue
        clients.append(client)
    return clients


async def connect_mcp_clients(clients: list[Any]) -> list[Any]:
    """Connect stateful MCP clients, dropping the ones that fail.

    Stateless (HTTP) clients need no explicit connection.
    """
    connected: list[Any] = []
    for client in clients:
        if not client.is_stateful:
            connected.append(client)
            continue
        try:
            await client.connect()
        except Exception as exc:  # pragma: no cover - IO dependent
            logger.warning(
                "failed to connect MCP %r: %s — skipping",
                client.name,
                exc,
            )
            continue
        connected.append(client)
    return connected


def configure_permissions(
    agent,
    cwd: str | None,
    permission_mode: str = PERMISSION_ACCEPT_EDITS,
) -> None:
    """Configure the agent's permission context.

    ``accept_edits`` (default): auto-approve file operations inside ``cwd``
    without a frontend round-trip.

    ``ask``: keep the engine default (``ASK``) so every tool call surfaces
    a ``session/request_permission`` popup in the client.
    """
    from agentscope.permission import (
        AdditionalWorkingDirectory,
        PermissionMode,
    )

    context = agent.state.permission_context
    if permission_mode == PERMISSION_ACCEPT_EDITS:
        context.mode = PermissionMode.ACCEPT_EDITS
    else:
        # DEFAULT = explicit permission per action (ASK behavior).
        context.mode = PermissionMode.DEFAULT
    if cwd:
        context.working_directories[cwd] = AdditionalWorkingDirectory(
            path=cwd,
            source="agentscope-acp",
        )


def build_chat_model(config: AcpConfig, model: str | None = None):
    """Instantiate the AgentScope chat model described by ``config``.

    ``model`` overrides ``config.model`` — used by ``set_config_option``
    for runtime model switching.

    Credentials are read at call time (not import time) so tests can patch
    the environment. Raises ``ValueError`` with an actionable message when
    the API key is missing — surfaced by the ACP agent as a protocol error.
    """
    model_name = model or config.model

    if not config.api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY: set it in the agent's environment "
            "(agent-work agent_catalog env or shell).",
        )

    key = (
        config.api_key.get_secret_value()
        if isinstance(config.api_key, SecretStr)
        else config.api_key
    )

    from agentscope.credential import OpenAICredential
    from agentscope.model import OpenAIChatModel

    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=SecretStr(key),
            base_url=config.base_url,
        ),
        model=model_name,
    )
