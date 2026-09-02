# -*- coding: utf-8 -*-
"""Environment-based configuration for agentscope-acp.

All knobs are environment variables so that ACP clients (agent-work's
``agent_catalog`` row) can configure the agent per-agent without config
files. stdout is reserved for the ACP protocol — logging goes to stderr
or a file only.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from pydantic import SecretStr

logger = logging.getLogger(__name__)

ENV_PROVIDER = "AGENTSCOPE_ACP_PROVIDER"
ENV_MODEL = "AGENTSCOPE_ACP_MODEL"
ENV_AVAILABLE_MODELS = "AGENTSCOPE_ACP_AVAILABLE_MODELS"
ENV_SYSTEM_PROMPT = "AGENTSCOPE_ACP_SYSTEM_PROMPT"
ENV_TOOLS = "AGENTSCOPE_ACP_TOOLS"
ENV_TOOL_NAMES = "AGENTSCOPE_ACP_TOOL_NAMES"
ENV_SKILLS_DIR = "AGENTSCOPE_ACP_SKILLS_DIR"
ENV_LOG = "AGENTSCOPE_ACP_LOG"

PROVIDER_DASHSCOPE = "dashscope"
PROVIDER_OPENAI_COMPAT = "openai-compat"

DEFAULT_MODEL = "qwen3.6-plus"

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

    provider: str
    api_key: SecretStr
    model: str
    available_models: list[ModelEntry] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    enable_tools: bool = True
    tool_names: list[str] | None = None
    skills_dir: str | None = None
    log_path: str | None = None

    @classmethod
    def from_env(cls) -> "AcpConfig":
        provider = os.environ.get(ENV_PROVIDER, PROVIDER_DASHSCOPE).strip().lower()
        if provider not in (PROVIDER_DASHSCOPE, PROVIDER_OPENAI_COMPAT):
            raise ValueError(
                f"Unsupported {ENV_PROVIDER}: {provider!r} "
                f"(expected {PROVIDER_DASHSCOPE!r} or {PROVIDER_OPENAI_COMPAT!r})",
            )

        if provider == PROVIDER_DASHSCOPE:
            api_key = SecretStr(os.environ.get("DASHSCOPE_API_KEY", ""))
        else:
            api_key = SecretStr(os.environ.get("OPENAI_API_KEY", ""))

        model = os.environ.get(ENV_MODEL, "").strip() or DEFAULT_MODEL

        raw_list = os.environ.get(ENV_AVAILABLE_MODELS, "").strip()
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

        raw_names = os.environ.get(ENV_TOOL_NAMES, "").strip()
        tool_names = (
            [part.strip() for part in raw_names.split(",") if part.strip()]
            or None
        )

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            available_models=entries,
            system_prompt=os.environ.get(ENV_SYSTEM_PROMPT, "").strip()
            or DEFAULT_SYSTEM_PROMPT,
            enable_tools=_parse_bool(os.environ.get(ENV_TOOLS, ""), True),
            tool_names=tool_names,
            skills_dir=os.environ.get(ENV_SKILLS_DIR, "").strip() or None,
            log_path=os.environ.get(ENV_LOG, "").strip() or None,
        )


def _parse_bool(raw: str, default: bool) -> bool:
    """Tolerant boolean parsing for on/off env switches."""
    if not raw or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def api_key_env_name(provider: str) -> str:
    """Human-readable env var name for error messages."""
    return (
        "DASHSCOPE_API_KEY"
        if provider == PROVIDER_DASHSCOPE
        else "OPENAI_API_KEY"
    )


def build_toolkit(
    enable_tools: bool,
    skills_dir: str | None = None,
    tool_names: list[str] | None = None,
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
    )


def configure_permissions(agent, cwd: str | None) -> None:
    """Auto-approve file operations inside the working directory.

    Tool support "path B" (todo 1.1): run with ``ACCEPT_EDITS`` so the
    agent can read/write inside ``cwd`` without a frontend approval
    round-trip. Interactive ``session/request_permission`` is todo 1.2.
    """
    from agentscope.permission import (
        AdditionalWorkingDirectory,
        PermissionMode,
    )

    context = agent.state.permission_context
    context.mode = PermissionMode.ACCEPT_EDITS
    if cwd:
        context.working_directories[cwd] = AdditionalWorkingDirectory(
            path=cwd,
            source="agentscope-acp",
        )


def build_chat_model(config: AcpConfig):
    """Instantiate the AgentScope chat model described by ``config``.

    Credentials are read at call time (not import time) so tests can patch
    the environment. Raises ``ValueError`` with an actionable message when
    the API key is missing — surfaced by the ACP agent as a protocol error.
    """
    if not config.api_key:
        env_name = api_key_env_name(config.provider)
        raise ValueError(
            f"Missing {env_name}: set it in the agent's environment "
            "(agent-work agent_catalog env or shell).",
        )

    key = (
        config.api_key.get_secret_value()
        if isinstance(config.api_key, SecretStr)
        else config.api_key
    )

    if config.provider == PROVIDER_DASHSCOPE:
        from agentscope.credential import DashScopeCredential
        from agentscope.model import DashScopeChatModel

        return DashScopeChatModel(
            credential=DashScopeCredential(
                api_key=SecretStr(key),
            ),
            model=config.model,
        )

    from agentscope.credential import OpenAICredential
    from agentscope.model import OpenAIChatModel

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=SecretStr(key),
            base_url=base_url,
        ),
        model=config.model,
    )
