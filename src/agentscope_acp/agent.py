# -*- coding: utf-8 -*-
"""ACP Agent implementation backed by in-process AgentScope Agents.

Follows the QwenPaw ACP server pattern (``QwenPawACPAgent``): subclass
``acp.Agent``, keep one AgentScope ``Agent`` per ACP session in memory,
and translate ``reply_stream`` events into ``session/update`` notifications
through :mod:`agentscope_acp.translate`.

Scope: ``initialize`` / ``session/new`` / ``session/prompt`` /
``session/cancel`` with streaming text and tool-call surfacing. Built-in
tools run with auto-approve permissions (``ACCEPT_EDITS``); interactive
permission prompts and session persistence remain on the README Roadmap.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from acp import Agent, InitializeResponse, NewSessionResponse, PromptResponse
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    Implementation,
    ModelInfo,
    SessionModelState,
)

from agentscope.agent import Agent as AgentScopeAgent
from agentscope.message import UserMsg

from . import __version__
from .config import (
    AcpConfig,
    build_chat_model,
    build_toolkit,
    configure_permissions,
)
from .translate import (
    STOP_CANCELLED,
    TurnTranslator,
    agent_message_chunk,
    extract_text,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "agentscope-acp"
AGENT_TITLE = "AgentScope"
USER_NAME = "user"


class AgentScopeAcpAgent(Agent):
    """ACP agent bridging to in-process AgentScope Agents.

    Each ACP session owns one AgentScope ``Agent`` (its ``AgentState``
    keeps the conversation context across turns within the process).
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
                Callable ``agent_factory(config, cwd) -> AgentScopeAgent``
                replacing the default Agent construction — used by tests
                to inject a fake Agent without touching models.
        """
        self._config = config or AcpConfig.from_env()
        self._agent_factory = agent_factory or self._default_agent_factory
        self._sessions: dict[str, AgentScopeAgent] = {}
        self._prompt_tasks: dict[str, asyncio.Task[Any]] = {}

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
        logger.info(
            "initialize: protocol_version=%s client_info=%s",
            protocol_version,
            client_info,
        )
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(load_session=False),
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
        session_id = uuid4().hex
        try:
            self._sessions[session_id] = self._agent_factory(
                self._config,
                cwd,
            )
        except ValueError as exc:
            # e.g. missing API key — surface an actionable protocol error.
            logger.error("new_session failed: %s", exc)
            raise RequestError.invalid_params({"details": str(exc)}) from None
        logger.info("new_session: id=%s cwd=%s", session_id, cwd)
        return NewSessionResponse(
            session_id=session_id,
            models=self._build_model_state(),
        )

    async def prompt(  # pylint: disable=unused-argument
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        agent = self._sessions.get(session_id)
        if agent is None:
            raise RequestError.invalid_params(
                {"details": f"Unknown session: {session_id!r}"},
            )

        text = extract_text(prompt)
        if not text:
            return PromptResponse(stop_reason="end_turn")

        logger.info("prompt: session=%s length=%d", session_id, len(text))

        prompt_task = asyncio.current_task()
        if prompt_task is not None:
            self._prompt_tasks[session_id] = prompt_task

        translator = TurnTranslator()
        try:
            async for event in agent.reply_stream(
                UserMsg(name=USER_NAME, content=text),
            ):
                for update in translator.process(event):
                    await self._conn.session_update(
                        session_id=session_id,
                        update=update,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_agent_factory(
        config: AcpConfig,
        cwd: str,
    ) -> AgentScopeAgent:
        """Build the AgentScope Agent for one session.

        When tools are enabled (the default) the agent gets the built-in
        coding tool set and runs with ``ACCEPT_EDITS`` permissions, so file
        operations inside the session working directory are auto-approved
        without a frontend round-trip (interactive approval is todo 1.2).
        """
        agent = AgentScopeAgent(
            name=AGENT_NAME,
            system_prompt=config.system_prompt,
            model=build_chat_model(config),
            toolkit=build_toolkit(
                config.enable_tools,
                config.skills_dir,
                config.tool_names,
            ),
        )
        configure_permissions(agent, cwd)
        return agent

    def _build_model_state(self) -> SessionModelState:
        """Advertise selectable models to ACP clients.

        The list is static (from the environment) in the MVP; the dynamic
        provider models API is on the README Roadmap.
        """
        return SessionModelState(
            available_models=[
                ModelInfo(model_id=entry.model_id, name=entry.name)
                for entry in self._config.available_models
            ],
            current_model_id=self._config.model,
        )
