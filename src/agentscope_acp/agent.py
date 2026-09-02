# -*- coding: utf-8 -*-
"""ACP Agent implementation backed by in-process AgentScope Agents.

Follows the QwenPaw ACP server pattern (``QwenPawACPAgent``): subclass
``acp.Agent``, keep one AgentScope ``Agent`` per ACP session in memory,
and translate ``reply_stream`` events into ``session/update`` notifications
through :mod:`agentscope_acp.translate`.

Scope: ``initialize`` / ``session/new`` / ``session/prompt`` /
``session/cancel`` / ``session/load`` / ``session/list`` / ``session/close``
/ ``set_config_option``, with streaming text + reasoning (thinking) +
tool-call surfacing, interactive permission approval, runtime model
switching, MCP server attachment and JSON-file session persistence.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from acp import (
    Agent,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
)
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    Implementation,
    ListSessionsResponse,
    ModelInfo,
    PermissionOption,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionInfo,
    SessionModelState,
    SetSessionConfigOptionResponse,
    ToolCallUpdate,
    UsageUpdate,
)

from agentscope.agent import Agent as AgentScopeAgent
from agentscope.event import ConfirmResult, RequireUserConfirmEvent, UserConfirmResultEvent
from agentscope.message import UserMsg
from agentscope.permission import PermissionBehavior, PermissionRule
from agentscope.state import AgentState

from . import __version__
from .config import (
    AcpConfig,
    build_chat_model,
    build_mcp_clients,
    build_toolkit,
    configure_permissions,
    connect_mcp_clients,
    sessions_path,
)
from .translate import (
    STOP_CANCELLED,
    TurnTranslator,
    _tool_kind,
    agent_message_chunk,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "agentscope-acp"
AGENT_TITLE = "AgentScope"
USER_NAME = "user"

CONFIG_OPTION_MODEL = "model"

# Permission options offered for each tool call. ``option_id`` values are
# mapped back onto ConfirmResult in ``_confirm_result``.
PERMISSION_OPTIONS = [
    PermissionOption(option_id="allow_once", name="允许一次", kind="allow_once"),
    PermissionOption(
        option_id="allow_always",
        name="始终允许",
        kind="allow_always",
    ),
    PermissionOption(option_id="reject_once", name="拒绝一次", kind="reject_once"),
    PermissionOption(
        option_id="reject_always",
        name="始终拒绝",
        kind="reject_always",
    ),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _confirm_result(tool_call: Any, outcome: Any) -> ConfirmResult:
    """Map an ACP permission outcome onto an AgentScope ``ConfirmResult``.

    ``selected`` + allow_* -> confirmed (with a permanent PermissionRule for
    ``allow_always``); ``selected`` + reject_*/unknown or ``cancelled`` ->
    rejected.

    Two shapes are accepted: the SDK's ``ClientConnection.request_permission``
    returns a ``RequestPermissionResponse`` whose ``.outcome`` is the
    AllowedOutcome/DeniedOutcome model, while simplified clients/tests may
    hand back the bare outcome model directly.
    """
    selected = getattr(outcome, "outcome", None)
    option_id = getattr(outcome, "option_id", None)
    if selected is not None and hasattr(selected, "outcome"):
        # Nested wrapper: RequestPermissionResponse.outcome -> AllowedOutcome.
        option_id = getattr(selected, "option_id", None)
        selected = getattr(selected, "outcome", None)
    if selected != "selected":
        return ConfirmResult(confirmed=False, tool_call=tool_call)
    rules = None
    if option_id in ("allow_always", "reject_always"):
        behavior = (
            PermissionBehavior.ALLOW
            if option_id == "allow_always"
            else PermissionBehavior.DENY
        )
        rules = [
            PermissionRule(
                tool_name=tool_call.name,
                rule_content=None,
                behavior=behavior,
                source="acp-client",
            ),
        ]
    confirmed = option_id in ("allow_once", "allow_always")
    return ConfirmResult(confirmed=confirmed, tool_call=tool_call, rules=rules)


def _uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI to an absolute local path.

    ``fs/read_text_file`` expects an absolute path, while clients such as
    Zed reference @mentioned files in resource links as URIs. Non-file
    URIs and plain paths are passed through unchanged.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri
    path = unquote(parsed.path)
    if os.name == "nt":
        netloc = parsed.netloc
        if len(netloc) == 2 and netloc[1] == ":" and netloc[0].isalpha():
            # file://D:/x — the drive letter lands in the netloc.
            path = netloc + path
        elif len(path) >= 3 and path[0] == "/" and path[2] == ":":
            # file:///D:/x — drop the leading slash.
            path = path[1:]
    return path


@dataclass
class SessionRecord:
    """Bookkeeping for one ACP session."""

    agent: AgentScopeAgent
    cwd: str
    mcp_clients: list[Any] = field(default_factory=list)
    title: str | None = None
    updated_at: str | None = None
    model_id: str | None = None


class AgentScopeAcpAgent(Agent):
    """ACP agent bridging to in-process AgentScope Agents.

    Each ACP session owns one AgentScope ``Agent`` (its ``AgentState``
    keeps the conversation context across turns within the process, and
    is persisted to ``AGENTSCOPE_ACP_SESSIONS_DIR`` for ``session/load``).
    """

    _conn: Client

    def __init__(
        self,
        config: AcpConfig | None = None,
        agent_factory: Any | None = None,
    ) -> None:
        """
        Args:
            config (`AcpConfig | None`, optional):
                Resolved configuration; read from the environment when
                omitted.
            agent_factory (`Any | None`, optional):
                Callable ``agent_factory(config, cwd, mcp_clients=None,
                state=None) -> AgentScopeAgent`` replacing the default
                Agent construction — used by tests to inject a fake Agent
                without touching models.
        """
        self._config = config or AcpConfig.from_env()
        self._agent_factory = agent_factory or self._default_agent_factory
        self._records: dict[str, SessionRecord] = {}
        self._prompt_tasks: dict[str, asyncio.Task[Any]] = {}
        self._client_capabilities: Any | None = None

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # ACP protocol methods
    # ------------------------------------------------------------------

    async def initialize(  # pylint: disable=unused-argument
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._client_capabilities = client_capabilities
        logger.info(
            "initialize: protocol_version=%s client_info=%s",
            protocol_version,
            client_info,
        )
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(load_session=True),
            agent_info=Implementation(
                name=AGENT_NAME,
                title=AGENT_TITLE,
                version=__version__,
            ),
        )

    async def new_session(  # pylint: disable=unused-argument
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        try:
            clients = await connect_mcp_clients(
                build_mcp_clients(mcp_servers),
            )
            agent = self._agent_factory(
                self._config,
                cwd,
                mcp_clients=clients,
            )
            # The protocol session id follows the engine's AgentState id
            # so persisted state and load_session line up.
            state = getattr(agent, "state", None)
            session_id = (
                getattr(state, "session_id", None) or uuid4().hex
            )
            record = SessionRecord(
                agent=agent,
                cwd=cwd,
                mcp_clients=clients,
                updated_at=_now_iso(),
                model_id=self._config.model,
            )
            self._records[session_id] = record
            # Persist immediately: if the client (e.g. Zed) restarts and
            # kills this process, session/load must still find the session
            # on disk.
            self._save_state(record, session_id)
        except ValueError as exc:
            # e.g. missing API key — surface an actionable protocol error.
            logger.error("new_session failed: %s", exc)
            raise RequestError.invalid_params({"details": str(exc)}) from None
        logger.info(
            "new_session: id=%s cwd=%s mcps=%d",
            session_id,
            cwd,
            len(self._records[session_id].mcp_clients),
        )
        return NewSessionResponse(
            session_id=session_id,
            models=self._build_model_state(self._config.model),
            config_options=self._build_config_options(self._config.model),
        )

    async def load_session(  # pylint: disable=unused-argument
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        path = sessions_path(self._config) / f"{session_id}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            state = AgentState.model_validate(raw)
        except FileNotFoundError:
            raise RequestError.invalid_params(
                {"details": f"No saved session: {session_id!r}"},
            ) from None
        except Exception as exc:
            raise RequestError.invalid_params(
                {"details": f"Corrupt session state: {exc}"},
            ) from None

        clients = await connect_mcp_clients(build_mcp_clients(mcp_servers))
        try:
            agent = self._agent_factory(
                self._config,
                cwd,
                mcp_clients=clients,
                state=state,
            )
        except ValueError as exc:
            raise RequestError.invalid_params({"details": str(exc)}) from None
        self._records[session_id] = SessionRecord(
            agent=agent,
            cwd=cwd,
            mcp_clients=clients,
            updated_at=_now_iso(),
            model_id=self._config.model,
        )
        logger.info("load_session: id=%s cwd=%s", session_id, cwd)
        return LoadSessionResponse(
            config_options=self._build_config_options(self._config.model),
        )

    async def _resolve_prompt_text(
        self,
        session_id: str,
        prompt: list[Any],
    ) -> str:
        """Extract text from ACP content blocks, resolving resource links.

        Clients such as Zed send @file mentions as ``resource_link`` blocks
        carrying only a URI; the protocol expects the agent to fetch the
        content itself through ``fs/read_text_file`` (gated on the client's
        ``fs.readTextFile`` capability). Embedded ``resource`` blocks are
        included verbatim. Resolution failures are best-effort — an
        unreadable link is skipped instead of failing the whole prompt.
        """
        parts: list[str] = []
        for block in prompt or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
                continue
            if kind == "resource":
                resource = getattr(block, "resource", None)
                text = getattr(resource, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
                continue
            if kind == "resource_link":
                name = getattr(block, "name", None) or "file"
                uri = getattr(block, "uri", None)
                if not isinstance(uri, str):
                    continue
                fs_caps = getattr(self._client_capabilities, "fs", None)
                if not getattr(fs_caps, "read_text_file", None):
                    logger.debug(
                        "client does not support fs/read_text_file; "
                        "skipping resource link: uri=%s",
                        uri,
                    )
                    continue
                try:
                    response = await self._conn.read_text_file(
                        session_id,
                        _uri_to_path(uri),
                    )
                    content = getattr(response, "content", None)
                except Exception:
                    logger.warning(
                        "failed to resolve resource link: uri=%s",
                        uri,
                        exc_info=True,
                    )
                    continue
                if isinstance(content, str) and content:
                    parts.append(
                        f"<attached_file name={name!r}>\n"
                        f"{content}\n"
                        f"</attached_file>"
                    )
        return "\n".join(parts)

    async def prompt(  # pylint: disable=unused-argument
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        record = self._records.get(session_id)
        if record is None:
            raise RequestError.invalid_params(
                {"details": f"Unknown session: {session_id!r}"},
            )

        text = await self._resolve_prompt_text(session_id, prompt)
        if not text:
            return PromptResponse(stop_reason="end_turn")

        if not record.title:
            first_line = text.strip().splitlines()[0].strip()
            record.title = (first_line[:60] or "conversation")
        record.updated_at = _now_iso()

        logger.info("prompt: session=%s length=%d", session_id, len(text))

        prompt_task = asyncio.current_task()
        if prompt_task is not None:
            self._prompt_tasks[session_id] = prompt_task

        translator = TurnTranslator()
        try:
            await self._consume_events(
                session_id,
                record.agent,
                translator,
                record.agent.reply_stream(
                    UserMsg(name=USER_NAME, content=text),
                ),
            )
        except asyncio.CancelledError:
            # session/cancel -> the dispatcher task was cancelled here;
            # absorb it and answer the still-pending prompt request.
            logger.info("prompt cancelled: session=%s", session_id)
            return PromptResponse(stop_reason=STOP_CANCELLED)
        except Exception as exc:
            # Report the failure in-band as a message and end the turn
            # instead of crashing the agent process.
            logger.exception("prompt failed: session=%s", session_id)
            error_id = f"error:{uuid4().hex[:8]}"
            try:
                await self._conn.session_update(
                    session_id=session_id,
                    update=agent_message_chunk(
                        error_id,
                        f"AgentScope error: {exc}",
                    ),
                )
            except Exception:
                logger.exception(
                    "failed to deliver error message: session=%s",
                    session_id,
                )
            return PromptResponse(stop_reason="end_turn")
        finally:
            self._prompt_tasks.pop(session_id, None)
            # Persist on every path (success, failure, cancel): the client
            # may be killed at any moment and session/load after a restart
            # depends on this file existing.
            record.updated_at = _now_iso()
            self._save_state(record, session_id)

        await self._send_usage(session_id, translator, record.agent)

        logger.info(
            "prompt finished: session=%s stop_reason=%s tokens(in/out)=%d/%d",
            session_id,
            translator.stop_reason(),
            translator.input_tokens,
            translator.output_tokens,
        )
        return PromptResponse(stop_reason=translator.stop_reason())

    async def cancel(  # pylint: disable=unused-argument
        self,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        logger.info("cancel: session=%s", session_id)
        prompt_task = self._prompt_tasks.get(session_id)
        current_task = asyncio.current_task()
        if (
            prompt_task is None
            or prompt_task is current_task
            or prompt_task.done()
        ):
            return

        prompt_task.cancel()
        try:
            await prompt_task
        except asyncio.CancelledError:
            pass
        except Exception:
            # The prompt handler already reported the error in-band.
            logger.exception(
                "prompt task failed while cancelling: session=%s",
                session_id,
            )

    async def list_sessions(  # pylint: disable=unused-argument
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        sessions: dict[str, SessionInfo] = {}
        for session_id, record in self._records.items():
            sessions[session_id] = SessionInfo(
                session_id=session_id,
                cwd=record.cwd,
                title=record.title,
                updated_at=record.updated_at,
            )
        # Merge persisted sessions from disk so the table survives a
        # process restart (in-memory records are empty after the client
        # kills us).
        sessions_dir = sessions_path(self._config)
        if sessions_dir.is_dir():
            for path in sessions_dir.glob("*.json"):
                session_id = path.stem
                if session_id in sessions:
                    continue
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    meta = raw.get("_acp_meta", {})
                except Exception:
                    continue
                sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    cwd=meta.get("cwd", ""),
                    title=meta.get("title"),
                    updated_at=meta.get("updated_at") or _now_iso(),
                )
        result = [
            info
            for info in sessions.values()
            if not cwd or info.cwd == cwd
        ]
        return ListSessionsResponse(sessions=result)

    async def close_session(  # pylint: disable=unused-argument
        self,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        logger.info("close_session: session=%s", session_id)
        record = self._records.pop(session_id, None)
        if record is None:
            return
        self._save_state(record, session_id)
        for client in record.mcp_clients:
            if client.is_stateful:
                try:
                    await client.close()
                except Exception:
                    logger.warning(
                        "failed to close MCP %r: session=%s",
                        client.name,
                        session_id,
                        exc_info=True,
                    )

    async def set_config_option(  # pylint: disable=unused-argument
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse | None:
        record = self._records.get(session_id)
        if record is None:
            raise RequestError.invalid_params(
                {"details": f"Unknown session: {session_id!r}"},
            )
        if config_id != CONFIG_OPTION_MODEL:
            raise RequestError.invalid_params(
                {"details": f"Unknown config option: {config_id!r}"},
            )
        model_id = str(value)
        try:
            record.agent.model = build_chat_model(
                self._config,
                model=model_id,
            )
        except ValueError as exc:
            raise RequestError.invalid_params({"details": str(exc)}) from None
        record.model_id = model_id
        logger.info(
            "set_config_option: session=%s model=%s",
            session_id,
            model_id,
        )
        options = self._build_config_options(model_id)
        # Note: no config_option_update notification here — the response
        # itself carries the new options and ACP is a single-client link
        # (QwenPaw likewise relies on the response only).
        return SetSessionConfigOptionResponse(config_options=options)

    # ------------------------------------------------------------------
    # Reply-stream handling
    # ------------------------------------------------------------------

    async def _consume_events(
        self,
        session_id: str,
        agent: AgentScopeAgent,
        translator: TurnTranslator,
        event_stream: Any,
    ) -> None:
        """Drain one ``reply_stream`` generator into ACP updates.

        A ``RequireUserConfirmEvent`` ends the generator by design: the
        engine resumes from its saved state when fed the resulting
        ``UserConfirmResultEvent`` back through a fresh ``reply_stream``
        call — recursed here so multi-step turns (asking several times)
        keep streaming into the same ACP turn.
        """
        async for event in event_stream:
            if isinstance(event, RequireUserConfirmEvent):
                confirm_event = await self._request_permission(
                    session_id,
                    event,
                )
                await self._consume_events(
                    session_id,
                    agent,
                    translator,
                    agent.reply_stream(confirm_event),
                )
                return
            for update in translator.process(event):
                await self._conn.session_update(
                    session_id=session_id,
                    update=update,
                )

    async def _request_permission(
        self,
        session_id: str,
        event: RequireUserConfirmEvent,
    ) -> UserConfirmResultEvent:
        """Ask the client for permission and build the resume event.

        ACP ``request_permission`` covers one tool call, so multi-tool
        confirmations are asked sequentially.
        """
        confirm_results = []
        for tool_call in event.tool_calls:
            outcome = await self._conn.request_permission(
                session_id=session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id=tool_call.id,
                    kind=_tool_kind(tool_call.name),
                    status="pending",
                    title=tool_call.name,
                    raw_input=tool_call.input or None,
                ),
                options=PERMISSION_OPTIONS,
            )
            confirm_results.append(_confirm_result(tool_call, outcome))
        return UserConfirmResultEvent(
            reply_id=event.reply_id,
            confirm_results=confirm_results,
        )

    async def _send_usage(
        self,
        session_id: str,
        translator: TurnTranslator,
        agent: AgentScopeAgent,
    ) -> None:
        """Report turn token usage once (after the final model call)."""
        if not translator.total_tokens:
            return
        size = int(getattr(agent.model, "context_size", 0) or 0)
        await self._conn.session_update(
            session_id=session_id,
            update=UsageUpdate(
                sessionUpdate="usage_update",
                used=translator.total_tokens,
                size=size,
            ),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _state_path(self, session_id: str):
        return sessions_path(self._config) / f"{session_id}.json"

    def _save_state(self, record: SessionRecord, session_id: str) -> None:
        """Persist the agent state to disk (best-effort).

        The engine's AgentState is saved together with a small `_acp_meta`
        block (cwd/title/timestamps) so `list_sessions` can rebuild the
        session table after a process restart (extra keys are ignored by
        AgentState.model_validate).
        """
        try:
            path = self._state_path(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.loads(record.agent.state.model_dump_json())
            payload["_acp_meta"] = {
                "cwd": record.cwd,
                "title": record.title,
                "updated_at": record.updated_at,
                "model_id": record.model_id,
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.warning(
                "failed to persist session %s",
                session_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_agent_factory(
        config: AcpConfig,
        cwd: str,
        mcp_clients: list[Any] | None = None,
        state: AgentState | None = None,
    ) -> AgentScopeAgent:
        """Build the AgentScope Agent for one session.

        ``state`` restores a persisted conversation (``session/load``);
        ``mcp_clients`` attaches ACP ``mcp_servers`` to the toolkit.
        """
        agent = AgentScopeAgent(
            name=AGENT_NAME,
            system_prompt=config.system_prompt,
            model=build_chat_model(config),
            toolkit=build_toolkit(
                config.enable_tools,
                config.skills_dir,
                config.tool_names,
                mcp_clients,
            ),
            state=state,
        )
        configure_permissions(agent, cwd, config.permission_mode)
        return agent

    def _build_model_state(self, current_model: str) -> SessionModelState:
        """Advertise selectable models to ACP clients."""
        return SessionModelState(
            available_models=[
                ModelInfo(model_id=entry.model_id, name=entry.name)
                for entry in self._config.available_models
            ],
            current_model_id=current_model,
        )

    def _build_config_options(
        self,
        current_model: str,
    ) -> list[SessionConfigOptionSelect]:
        """Advertise the runtime config options (model picker)."""
        return [
            SessionConfigOptionSelect(
                type="select",
                id=CONFIG_OPTION_MODEL,
                name="模型",
                description="Select the model used by the agent",
                current_value=current_model,
                options=[
                    SessionConfigSelectOption(
                        value=entry.model_id,
                        name=entry.name,
                    )
                    for entry in self._config.available_models
                ],
            ),
        ]
