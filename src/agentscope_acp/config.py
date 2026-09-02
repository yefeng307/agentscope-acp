# -*- coding: utf-8 -*-
"""Environment-based configuration for agentscope-acp.

All knobs are environment variables so that ACP clients (agent-work's
``agent_catalog`` row) can configure the agent per-agent without config
files. stdout is reserved for the ACP protocol — logging goes to stderr
or a file only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV_PROVIDER = "AGENTSCOPE_ACP_PROVIDER"
ENV_MODEL = "AGENTSCOPE_ACP_MODEL"
ENV_AVAILABLE_MODELS = "AGENTSCOPE_ACP_AVAILABLE_MODELS"
ENV_SYSTEM_PROMPT = "AGENTSCOPE_ACP_SYSTEM_PROMPT"
ENV_TOOLS = "AGENTSCOPE_ACP_TOOLS"
ENV_LOG = "AGENTSCOPE_ACP_LOG"

PROVIDER_DASHSCOPE = "dashscope"
PROVIDER_OPENAI_COMPAT = "openai-compat"

DEFAULT_MODEL = "qwen3.6-plus"

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
    api_key: str
    model: str
    available_models: list[ModelEntry] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    enable_tools: bool = True
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
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

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

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            available_models=entries,
            system_prompt=os.environ.get(ENV_SYSTEM_PROMPT, "").strip()
            or DEFAULT_SYSTEM_PROMPT,
            enable_tools=_parse_bool(os.environ.get(ENV_TOOLS, ""), True),
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


def build_toolkit(enable_tools: bool):
    """Build the built-in coding tool set, or ``None`` for a tool-less agent.

    Imports are deferred so a tool-less configuration (and the unit tests
    that avoid AgentScope's tool stack) never pay the import cost.
    """
    if not enable_tools:
        return None
    from agentscope.tool import Bash, Edit, Glob, Grep, Read, Toolkit, Write

    return Toolkit(tools=[Bash(), Read(), Write(), Edit(), Grep(), Glob()])


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
    from pydantic import SecretStr

    if not config.api_key:
        env_name = api_key_env_name(config.provider)
        raise ValueError(
            f"Missing {env_name}: set it in the agent's environment "
            "(agent-work agent_catalog env or shell).",
        )

    if config.provider == PROVIDER_DASHSCOPE:
        from agentscope.credential import DashScopeCredential
        from agentscope.model import DashScopeChatModel

        return DashScopeChatModel(
            credential=DashScopeCredential(
                api_key=SecretStr(config.api_key),
            ),
            model=config.model,
        )

    from agentscope.credential import OpenAICredential
    from agentscope.model import OpenAIChatModel

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=SecretStr(config.api_key),
            base_url=base_url,
        ),
        model=config.model,
    )
