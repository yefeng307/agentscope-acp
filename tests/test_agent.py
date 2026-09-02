# -*- coding: utf-8 -*-
"""Unit tests for AgentScopeAcpAgent — ACP methods against a fake AgentScope Agent.

The ACP connection is replaced by a recorder, and the AgentScope Agent by a
fake yielding a scripted event stream, so no model/network is involved.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.schema import TextContentBlock
from agentscope.event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    TextBlockDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.event._event import ToolResultState
from agentscope.types import ReplyFinishedReason
from pydantic import SecretStr

from agentscope_acp.agent import AgentScopeAcpAgent
from agentscope_acp.config import AcpConfig, ModelEntry, build_toolkit


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class RecordingConn:
    """Fake ``acp.interfaces.Client`` recording session_update calls."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))


class FakeAgentScopeAgent:
    """Fake AgentScope Agent with a scripted reply_stream."""

    def __init__(self, events: list[Any] | None = None, error: Exception | None = None):
        self.events = events or []
        self.error = error
        self.received: list[Any] = []

    async def reply_stream(self, inputs: Any = None, **kwargs: Any):
        self.received.append(inputs)
        for event in self.events:
            await asyncio.sleep(0)
            yield event
        if self.error is not None:
            raise self.error


class ForeverAgent:
    """Agent whose stream never ends — used for cancel testing."""

    def __init__(self) -> None:
        self.received: list[Any] = []

    async def reply_stream(self, inputs: Any = None, **kwargs: Any):
        self.received.append(inputs)
        while True:
            await asyncio.sleep(3600)
            yield  # pragma: no cover — never reached


def _config() -> AcpConfig:
    return AcpConfig(
        provider="dashscope",
        api_key=SecretStr("sk-test"),
        model="qwen3.6-plus",
        available_models=[
            ModelEntry(model_id="qwen3.6-plus", name="qwen3.6-plus"),
            ModelEntry(model_id="qwen3.6-max", name="qwen3.6-max"),
        ],
    )


def _prompt_blocks(text: str) -> list[TextContentBlock]:
    return [TextContentBlock(type="text", text=text)]


def _stream_events() -> list[Any]:
    return [
        TextBlockDeltaEvent(reply_id="r1", block_id="b1", delta="Hel"),
        TextBlockDeltaEvent(reply_id="r1", block_id="b1", delta="lo"),
        ModelCallEndEvent(reply_id="r1", input_tokens=10, output_tokens=5),
        ReplyEndEvent(session_id="s", reply_id="r1"),
    ]


def _make_agent(fake: Any) -> AgentScopeAcpAgent:
    acp_agent = AgentScopeAcpAgent(
        _config(),
        agent_factory=lambda cfg, cwd: fake,
    )
    acp_agent.on_connect(RecordingConn())
    return acp_agent


# ----------------------------------------------------------------------
# initialize / new_session
# ----------------------------------------------------------------------

async def test_initialize_declares_baseline_capabilities():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    response = await acp_agent.initialize(protocol_version=1)
    assert response.agent_info.name == "agentscope-acp"
    assert response.agent_capabilities.load_session is False


async def test_new_session_advertises_models():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    response = await acp_agent.new_session(cwd="/tmp")
    models = response.models
    assert models.current_model_id == "qwen3.6-plus"
    assert [m.model_id for m in models.available_models] == [
        "qwen3.6-plus",
        "qwen3.6-max",
    ]
    # Session registered for subsequent prompts.
    assert response.session_id in acp_agent._sessions


async def test_new_session_invalid_config_raises_request_error():
    def failing_factory(cfg: AcpConfig, cwd: str):
        raise ValueError("Missing DASHSCOPE_API_KEY")

    acp_agent = AgentScopeAcpAgent(_config(), agent_factory=failing_factory)
    with pytest.raises(RequestError):
        await acp_agent.new_session(cwd="/tmp")


# ----------------------------------------------------------------------
# prompt
# ----------------------------------------------------------------------

async def test_prompt_streams_chunks_and_ends_turn():
    fake = FakeAgentScopeAgent(_stream_events())
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("hi"),
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    conn = acp_agent._conn
    assert len(conn.updates) == 2
    ids = {u.message_id for _, u in conn.updates}
    assert ids == {"r1:b1"}
    texts = "".join(u.content.text for _, u in conn.updates)
    assert texts == "Hello"
    # The user message reached the AgentScope agent.
    assert len(fake.received) == 1


async def test_prompt_empty_text_returns_immediately():
    acp_agent = _make_agent(FakeAgentScopeAgent(_stream_events()))
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(prompt=[], session_id=session.session_id)
    assert response.stop_reason == "end_turn"
    assert acp_agent._conn.updates == []


async def test_prompt_unknown_session_raises():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    with pytest.raises(RequestError):
        await acp_agent.prompt(prompt=_prompt_blocks("hi"), session_id="nope")


async def test_prompt_error_is_reported_in_band():
    fake = FakeAgentScopeAgent(error=RuntimeError("boom"))
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("hi"),
        session_id=session.session_id,
    )

    # Error surfaces as a message chunk and the turn ends normally.
    assert response.stop_reason == "end_turn"
    conn = acp_agent._conn
    assert len(conn.updates) == 1
    assert "boom" in conn.updates[0][1].content.text


# ----------------------------------------------------------------------
# cancel
# ----------------------------------------------------------------------

async def test_cancel_aborts_prompt_with_cancelled_stop_reason():
    acp_agent = _make_agent(ForeverAgent())
    session = await acp_agent.new_session(cwd="/tmp")

    prompt_task = asyncio.create_task(
        acp_agent.prompt(
            prompt=_prompt_blocks("slow question"),
            session_id=session.session_id,
        ),
    )
    # Give the prompt task a chance to register itself and start streaming.
    await asyncio.sleep(0.05)

    await acp_agent.cancel(session_id=session.session_id)
    response = await prompt_task

    assert response.stop_reason == "cancelled"
    # Prompt bookkeeping is cleaned up.
    assert session.session_id not in acp_agent._prompt_tasks


async def test_cancel_without_active_prompt_is_noop():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    await acp_agent.cancel(session_id="whatever")


# ----------------------------------------------------------------------
# second turn reuses the same session agent (context continuity)
# ----------------------------------------------------------------------

async def test_prompt_second_turn_uses_same_agent_instance():
    fake = FakeAgentScopeAgent(_stream_events())
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(prompt=_prompt_blocks("one"), session_id=session.session_id)
    await acp_agent.prompt(prompt=_prompt_blocks("two"), session_id=session.session_id)

    assert len(fake.received) == 2


# ----------------------------------------------------------------------
# tool-call surfacing through the prompt loop
# ----------------------------------------------------------------------

def _tool_stream_events() -> list[Any]:
    return [
        ToolCallStartEvent(
            reply_id="r1", tool_call_id="c1", tool_call_name="Bash",
        ),
        ToolCallDeltaEvent(reply_id="r1", tool_call_id="c1", delta="ls"),
        ToolCallEndEvent(reply_id="r1", tool_call_id="c1"),
        ToolResultStartEvent(
            reply_id="r1", tool_call_id="c1", tool_call_name="Bash",
        ),
        ToolResultTextDeltaEvent(reply_id="r1", tool_call_id="c1", delta="out"),
        ToolResultEndEvent(
            reply_id="r1", tool_call_id="c1", state=ToolResultState.SUCCESS,
        ),
        TextBlockDeltaEvent(reply_id="r1", block_id="b1", delta="done"),
        ReplyEndEvent(session_id="s", reply_id="r1"),
    ]


async def test_prompt_surfaces_tool_call_updates():
    fake = FakeAgentScopeAgent(_tool_stream_events())
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("list files"),
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    kinds = [u.session_update for _, u in acp_agent._conn.updates]
    assert kinds == [
        "tool_call",
        "tool_call_update",
        "tool_call_update",
        "agent_message_chunk",
    ]
    # The tool_call names the invocation; the final update carries the result.
    first = acp_agent._conn.updates[0][1]
    assert first.title == "Bash" and first.kind == "execute"
    last_tool = acp_agent._conn.updates[2][1]
    assert last_tool.status == "completed" and last_tool.raw_output == "out"


# ----------------------------------------------------------------------
# skills configuration
# ----------------------------------------------------------------------

async def test_from_env_reads_skills_dir(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_SKILLS_DIR", "/tmp/skills")
    assert AcpConfig.from_env().skills_dir == "/tmp/skills"


async def test_build_toolkit_registers_skills_from_dir(tmp_path):
    skill_dir = tmp_path / "greet"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: greet\ndescription: greet the user politely\n---\n\n"
        "# Instructions\nSay hello.\n",
        encoding="utf-8",
    )

    toolkit = build_toolkit(True, skills_dir=str(tmp_path))
    instructions = await toolkit.get_skill_instructions() or ""
    assert "greet" in instructions
    assert "greet the user politely" in instructions


async def test_build_toolkit_ignores_missing_skills_dir(tmp_path):
    toolkit = build_toolkit(True, skills_dir=str(tmp_path / "nope"))
    assert await toolkit.get_skill_instructions() is None
