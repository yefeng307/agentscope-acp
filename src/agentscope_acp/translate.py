# -*- coding: utf-8 -*-
"""Translate AgentScope ``reply_stream`` events into ACP session updates.

This module is intentionally free of connection/IO logic: every function
takes AgentScope event objects (pydantic models) and returns plain ACP
update models, so the translation layer is unit-testable without a live
model or ACP client.

Mapping (see README Roadmap for the rest):

- ``TextBlockDeltaEvent``     -> ``agent_message_chunk`` (streamed text)
- ``ThinkingBlockDeltaEvent`` -> ``agent_thought_chunk`` (reasoning stream)
- ``ToolCallStartEvent``      -> ``tool_call`` (tool starts, in_progress)
- ``ToolCallEndEvent``        -> ``tool_call_update`` (args complete)
- ``ToolResultEndEvent``      -> ``tool_call_update`` (completed/failed)
- ``ReplyEndEvent``           -> turn-stop reason for ``PromptResponse``
- ``ModelCallEndEvent``       -> token accounting (usage_update)
- result deltas               -> buffered and flushed on the end event
"""
from __future__ import annotations

import logging
from typing import Any

from acp import start_tool_call, text_block, tool_content, update_tool_call
from acp.schema import AgentMessageChunk, AgentThoughtChunk

from agentscope.event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    TextBlockDeltaEvent,
    ThinkingBlockDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.event._event import ToolResultState
from agentscope.types import ReplyFinishedReason

logger = logging.getLogger(__name__)

STOP_END_TURN = "end_turn"
STOP_CANCELLED = "cancelled"
STOP_MAX_TURN_REQUESTS = "max_turn_requests"

# ACP ToolCall "kind" per built-in AgentScope tool name.
_TOOL_KIND = {
    "Bash": "execute",
    "PowerShell": "execute",
    "Read": "read",
    "Write": "edit",
    "Edit": "edit",
    "Grep": "search",
    "Glob": "search",
}


def _tool_kind(name: str) -> str:
    """Map an AgentScope tool name onto an ACP ``ToolKind``."""
    return _TOOL_KIND.get(name, "other")


def extract_text(prompt: list[Any] | None) -> str:
    """Pull plain text out of ACP prompt content blocks.

    Mirrors QwenPaw's ``_extract_text``: non-text blocks (image/audio/
    resource) are skipped in the MVP — see README Roadmap.
    """
    parts: list[str] = []
    for block in prompt or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def agent_message_chunk(message_id: str, text: str) -> AgentMessageChunk:
    """Build an ``agent_message_chunk`` update carrying one text delta."""
    return AgentMessageChunk(
        session_update="agent_message_chunk",
        content=text_block(text),
        message_id=message_id,
    )


def agent_thought_chunk(message_id: str, text: str) -> AgentThoughtChunk:
    """Build an ``agent_thought_chunk`` update carrying one reasoning delta."""
    return AgentThoughtChunk(
        session_update="agent_thought_chunk",
        content=text_block(text),
        message_id=message_id,
    )


class TurnTranslator:
    """Stateful translator for one ``session/prompt`` turn.

    One AgentScope *reply* maps to one ACP message: all text deltas of a
    reply share the same ``message_id`` (``{reply_id}:{block_id}`` so two
    distinct text blocks inside a single reply still open separate ACP
    messages, which matches how clients accumulate chunks by id).
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.finished_reason: ReplyFinishedReason | None = None
        # Per-tool-call buffers keyed by tool_call_id. Tool args and output
        # stream in as deltas; both are flushed on their respective end
        # event so clients receive one compact update each.
        self._tool_inputs: dict[str, list[str]] = {}
        self._tool_outputs: dict[str, list[str]] = {}
        # Complete tool args keyed by tool_call_id, kept after the
        # in_progress update so the terminal update can repeat them.
        self._tool_inputs_done: dict[str, str] = {}

    @property
    def total_tokens(self) -> int:
        """Tokens consumed by the turn's model calls (in + out)."""
        return self.input_tokens + self.output_tokens

    def process(self, event: Any) -> list[Any]:
        """Convert one AgentScope event into zero or more ACP updates."""
        if isinstance(event, TextBlockDeltaEvent):
            if not event.delta:
                return []
            message_id = f"{event.reply_id}:{event.block_id}"
            return [agent_message_chunk(message_id, event.delta)]

        if isinstance(event, ThinkingBlockDeltaEvent):
            if not event.delta:
                return []
            message_id = f"{event.reply_id}:{event.block_id}"
            return [agent_thought_chunk(message_id, event.delta)]

        if isinstance(event, ModelCallEndEvent):
            self.input_tokens += event.input_tokens
            self.output_tokens += event.output_tokens
            return []

        if isinstance(event, ReplyEndEvent):
            self.finished_reason = event.finished_reason
            if event.finished_reason == ReplyFinishedReason.ERROR:
                logger.error(
                    "AgentScope reply ended with error: %s",
                    event.error,
                )
            return []

        if isinstance(event, ToolCallStartEvent):
            return [
                start_tool_call(
                    event.tool_call_id,
                    event.tool_call_name,
                    kind=_tool_kind(event.tool_call_name),
                    status="in_progress",
                ),
            ]

        if isinstance(event, ToolCallDeltaEvent):
            self._tool_inputs.setdefault(event.tool_call_id, []).append(
                event.delta,
            )
            return []

        if isinstance(event, ToolCallEndEvent):
            # Arguments are complete — surface them for the client to
            # render, still in_progress until the tool result arrives.
            raw_input = "".join(
                self._tool_inputs.pop(event.tool_call_id, []),
            )
            # Nothing streamed (no arguments): the start event already
            # advertised the call, so there is nothing new to update.
            if not raw_input:
                return []
            self._tool_inputs_done[event.tool_call_id] = raw_input
            return [
                update_tool_call(
                    event.tool_call_id,
                    status="in_progress",
                    raw_input=raw_input,
                ),
            ]

        if isinstance(event, ToolResultStartEvent):
            # Already advertised as in_progress at ToolCallStart.
            return []

        if isinstance(event, ToolResultTextDeltaEvent):
            self._tool_outputs.setdefault(event.tool_call_id, []).append(
                event.delta,
            )
            return []

        if isinstance(event, ToolResultEndEvent):
            output = "".join(self._tool_outputs.pop(event.tool_call_id, []))
            # ACP has no "cancelled" ToolCallStatus — an interrupted turn
            # must not show the tool as failed. The turn-level
            # stop_reason=cancelled already signals the abort, so skip
            # the final update for INTERRUPTED results.
            if event.state == ToolResultState.INTERRUPTED:
                self._tool_inputs_done.pop(event.tool_call_id, None)
                return []
            failed = event.state != ToolResultState.SUCCESS
            return [
                update_tool_call(
                    event.tool_call_id,
                    status="failed" if failed else "completed",
                    # Repeat the args on the terminal update: clients that
                    # only render raw_input on completed tool cards (Zed)
                    # would otherwise never show the command.
                    raw_input=self._tool_inputs_done.pop(
                        event.tool_call_id,
                        None,
                    ),
                    content=(
                        [tool_content(text_block(output))] if output else None
                    ),
                    raw_output=output or None,
                ),
            ]

        # Data-block / hint events are not surfaced.
        return []

    def stop_reason(self) -> str:
        """Map the AgentScope finished reason onto an ACP ``StopReason``."""
        if self.finished_reason == ReplyFinishedReason.INTERRUPTED:
            return STOP_CANCELLED
        if self.finished_reason == ReplyFinishedReason.EXCEED_MAX_ITERS:
            return STOP_MAX_TURN_REQUESTS
        return STOP_END_TURN
