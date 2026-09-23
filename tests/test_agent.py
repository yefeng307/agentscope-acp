# -*- coding: utf-8 -*-
"""Unit tests for AgentScopeAcpAgent — ACP methods against a fake AgentScope Agent.

The ACP connection is replaced by a recorder, and the AgentScope Agent by a
fake yielding a scripted event stream, so no model/network is involved.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    FileSystemCapabilities,
    PermissionOption,
    ReadTextFileResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
    ToolCallUpdate,
)
from agentscope.event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    RequireUserConfirmEvent,
    TextBlockDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
    UserConfirmResultEvent,
    UserInterruptEvent,
)
from agentscope.event._event import ToolResultState
from agentscope.message import ToolCallBlock, ToolCallState, ToolResultBlock
from agentscope.state import AgentState
from agentscope.types import ReplyFinishedReason
from pydantic import SecretStr

from agentscope_acp.agent import AgentScopeAcpAgent, _uri_to_path
from agentscope_acp.config import AcpConfig, ModelEntry, build_toolkit


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class RecordingConn:
    """Fake ``acp.interfaces.Client`` recording session_update calls."""

    def __init__(
        self,
        permission_outcomes: list[Any] | None = None,
        file_contents: dict[str, str] | None = None,
    ) -> None:
        self.updates: list[tuple[str, Any]] = []
        self.permission_requests: list[tuple[str, Any, list[Any]]] = []
        self.read_requests: list[tuple[str, str]] = []
        self._outcomes = list(permission_outcomes or [])
        self._file_contents = dict(file_contents or {})

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))

    async def send_request(self, method: str, params: Any, **kwargs: Any) -> Any:
        """Raw request channel — the agent sends session/request_permission
        through it (see _request_permission_outcome) so the reply shape is
        whatever the fake hands back: SDK models, the standard wire dict, or
        a host shortcut like agent-work's {option: {kind}}."""
        assert method == "session/request_permission", method
        # Rebuild models from the wire dicts so existing property-access
        # assertions keep working.
        self.permission_requests.append(
            (
                params["sessionId"],
                ToolCallUpdate.model_validate(params["toolCall"]),
                [PermissionOption.model_validate(o) for o in params["options"]],
            )
        )
        if self._outcomes:
            return self._outcomes.pop(0)
        # Default to the standard wire dict shape (what a spec-compliant
        # host answers).
        return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        self.read_requests.append((session_id, path))
        # Unknown paths raise KeyError, exercising the resolver's
        # best-effort failure path.
        return ReadTextFileResponse(content=self._file_contents[path])


class FakeAgentScopeAgent:
    """Fake AgentScope Agent with a scripted, cursor-advancing reply_stream.

    Mirroring the engine, a ``RequireUserConfirmEvent`` ends the generator;
    the next ``reply_stream`` call resumes right after it (the resume event
    passed in ``inputs``).
    """

    name = "agentscope-acp"

    def __init__(self, events: list[Any] | None = None, error: Exception | None = None):
        self.events = events or []
        self.error = error
        self.received: list[Any] = []
        self.state = AgentState()
        self.model: Any = None
        self._pos = 0
        self._error_raised = False

    async def reply_stream(self, inputs: Any = None, **kwargs: Any):
        self.received.append(inputs)
        while self._pos < len(self.events):
            event = self.events[self._pos]
            self._pos += 1
            await asyncio.sleep(0)
            yield event
            if isinstance(event, RequireUserConfirmEvent):
                return
        if self.error is not None and not self._error_raised:
            self._error_raised = True
            raise self.error


class ForeverAgent:
    """Agent whose stream never ends — used for cancel testing."""

    name = "agentscope-acp"

    def __init__(self) -> None:
        self.received: list[Any] = []

    async def reply_stream(self, inputs: Any = None, **kwargs: Any):
        self.received.append(inputs)
        while True:
            await asyncio.sleep(3600)
            yield  # pragma: no cover — never reached


def _config() -> AcpConfig:
    return AcpConfig(
        api_key=SecretStr("sk-test"),
        model="qwen3.6-plus",
        available_models=[
            ModelEntry(model_id="qwen3.6-plus", name="qwen3.6-plus"),
            ModelEntry(model_id="qwen3.6-max", name="qwen3.6-max"),
        ],
    )


def _prompt_blocks(text: str) -> list[TextContentBlock]:
    return [TextContentBlock(type="text", text=text)]


def _received_text(user_msg: Any) -> str:
    """Flatten a UserMsg's text blocks into plain text."""
    return "".join(
        getattr(block, "text", None) or ""
        for block in (getattr(user_msg, "content", None) or [])
    )


def _stream_events() -> list[Any]:
    return [
        TextBlockDeltaEvent(reply_id="r1", block_id="b1", delta="Hel"),
        TextBlockDeltaEvent(reply_id="r1", block_id="b1", delta="lo"),
        ModelCallEndEvent(reply_id="r1", input_tokens=10, output_tokens=5),
        ReplyEndEvent(session_id="s", reply_id="r1"),
    ]


def _make_agent(
    fake: Any,
    permission_outcomes: list[Any] | None = None,
) -> AgentScopeAcpAgent:
    acp_agent = AgentScopeAcpAgent(
        _config(),
        agent_factory=lambda cfg, cwd, mcp_clients=None, state=None: fake,
    )
    acp_agent.on_connect(RecordingConn(permission_outcomes))
    return acp_agent


@pytest.fixture(autouse=True)
def _isolate_sessions(tmp_path, monkeypatch):
    """Point all session persistence at a per-test temp dir so tests
    never touch the real ~/.agentscope-acp/sessions."""
    monkeypatch.setattr(
        "agentscope_acp.agent.sessions_path",
        lambda config: tmp_path,
    )


# ----------------------------------------------------------------------
# initialize / new_session
# ----------------------------------------------------------------------

async def test_initialize_declares_baseline_capabilities():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    response = await acp_agent.initialize(protocol_version=1)
    assert response.agent_info.name == "agentscope-acp"
    assert response.agent_capabilities.load_session is True
    caps = response.agent_capabilities.session_capabilities
    assert caps is not None
    assert caps.close is not None
    assert caps.list is not None
    assert caps.resume is not None


async def test_new_session_advertises_models_and_config_options():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    response = await acp_agent.new_session(cwd="/tmp")
    models = response.models
    assert models.current_model_id == "qwen3.6-plus"
    assert [m.model_id for m in models.available_models] == [
        "qwen3.6-plus",
        "qwen3.6-max",
    ]
    # Model picker advertised as a runtime config option.
    assert len(response.config_options) == 1
    option = response.config_options[0]
    assert option.id == "model" and option.current_value == "qwen3.6-plus"
    assert [o.value for o in option.options] == [
        "qwen3.6-plus",
        "qwen3.6-max",
    ]
    # Session registered for subsequent prompts.
    assert response.session_id in acp_agent._records


async def test_new_session_invalid_config_raises_request_error():
    def failing_factory(
        cfg: AcpConfig,
        cwd: str,
        mcp_clients=None,
        state=None,
    ):
        raise ValueError("Missing OPENAI_API_KEY")

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
    # 2 text chunks + 1 usage update at the end of the turn.
    assert len(conn.updates) == 3
    ids = {u.message_id for _, u in conn.updates if u.session_update == "agent_message_chunk"}
    assert ids == {"r1:b1"}
    texts = "".join(
        u.content.text
        for _, u in conn.updates
        if u.session_update == "agent_message_chunk"
    )
    assert texts == "Hello"
    # Usage was reported from the ModelCallEndEvent.
    usage = conn.updates[-1][1]
    assert usage.session_update == "usage_update"
    assert usage.used == 15
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


async def test_prompt_without_model_calls_skips_usage():
    fake = FakeAgentScopeAgent(
        [ReplyEndEvent(session_id="s", reply_id="r1")],
    )
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(prompt=_prompt_blocks("hi"), session_id=session.session_id)
    assert acp_agent._conn.updates == []


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


async def test_close_session_cancels_inflight_prompt():
    acp_agent = _make_agent(ForeverAgent())
    session = await acp_agent.new_session(cwd="/tmp")

    prompt_task = asyncio.create_task(
        acp_agent.prompt(
            prompt=_prompt_blocks("slow question"),
            session_id=session.session_id,
        ),
    )
    # Give the prompt task a chance to register itself.
    await asyncio.sleep(0.05)

    await acp_agent.close_session(session_id=session.session_id)
    response = await prompt_task

    assert response.stop_reason == "cancelled"
    assert acp_agent._records == {}
    assert session.session_id not in acp_agent._prompt_tasks


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
    # The terminal update repeats the args so clients that only render
    # raw_input on completed tool cards still show the command.
    assert last_tool.raw_input == "ls"


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


# ----------------------------------------------------------------------
# tool whitelist configuration
# ----------------------------------------------------------------------

async def test_from_env_reads_tool_names(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_TOOL_NAMES", "Bash, Read")
    assert AcpConfig.from_env().tool_names == ["Bash", "Read"]


async def test_build_toolkit_whitelist_selects_tools():
    toolkit = build_toolkit(True, tool_names=["Read", "PowerShell"])
    names = [t.name for t in toolkit.tool_groups[0].tools]
    assert names == ["Read", "PowerShell"]


async def test_build_toolkit_unknown_tool_is_skipped():
    toolkit = build_toolkit(True, tool_names=["Nope"])
    assert toolkit.tool_groups[0].tools == []


async def test_build_toolkit_default_set():
    toolkit = build_toolkit(True)
    names = [t.name for t in toolkit.tool_groups[0].tools]
    assert names == ["Bash", "Read", "Write", "Edit", "Grep", "Glob"]


# ----------------------------------------------------------------------
# interactive permission approval
# ----------------------------------------------------------------------

def _permission_events() -> list[Any]:
    return [
        TextBlockDeltaEvent(reply_id="r1", block_id="b1", delta="pre"),
        RequireUserConfirmEvent(
            reply_id="r1",
            tool_calls=[
                ToolCallBlock(id="c1", name="Bash", input='{"cmd": "rm x"}'),
            ],
        ),
        TextBlockDeltaEvent(reply_id="r1", block_id="b2", delta="post"),
        ReplyEndEvent(session_id="s", reply_id="r1"),
    ]


async def test_prompt_requests_permission_and_resumes_on_allow():
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[
            AllowedOutcome(outcome="selected", option_id="allow_once"),
        ],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    conn = acp_agent._conn
    # The client was asked before the tool ran.
    assert len(conn.permission_requests) == 1
    _, tool_call_update, options = conn.permission_requests[0]
    assert tool_call_update.tool_call_id == "c1"
    assert tool_call_update.title == "Bash"
    assert tool_call_update.kind == "execute"
    assert [o.kind for o in options] == [
        "allow_once",
        "allow_always",
        "reject_once",
        "reject_always",
    ]
    # The engine was resumed with a confirmed result and kept streaming.
    assert len(fake.received) == 2
    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert resume.confirm_results[0].confirmed is True
    assert resume.confirm_results[0].rules is None
    texts = [
        u.content.text
        for _, u in conn.updates
        if u.session_update == "agent_message_chunk"
    ]
    assert texts == ["pre", "post"]


async def test_prompt_denied_permission_resumes_with_rejection():
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[
            AllowedOutcome(outcome="selected", option_id="reject_once"),
        ],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert resume.confirm_results[0].confirmed is False


async def test_prompt_allow_always_adds_permission_rule():
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[
            AllowedOutcome(outcome="selected", option_id="allow_always"),
        ],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    resume = fake.received[1]
    rule = resume.confirm_results[0].rules[0]
    assert rule.tool_name == "Bash"
    assert rule.behavior.value == "allow"


async def test_prompt_cancelled_permission_is_rejection():
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[DeniedOutcome(outcome="cancelled")],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    resume = fake.received[1]
    assert resume.confirm_results[0].confirmed is False


async def test_prompt_permission_nested_response_shape():
    """Real clients wrap the outcome: RequestPermissionResponse.outcome is the
    AllowedOutcome model (verified against a Zed capture)."""
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[
            RequestPermissionResponse(
                outcome=AllowedOutcome(
                    outcome="selected",
                    option_id="allow_once",
                ),
            ),
        ],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert resume.confirm_results[0].confirmed is True


async def test_permission_host_option_kind_shape_allows():
    """agent-work answers session/request_permission with {option: {kind}}
    instead of the standard outcome envelope (packages/server/src/agent/acp/
    task.ts resolvePermission). allow_* kinds must permit the tool."""
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[{"option": {"kind": "allow_once"}}],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert resume.confirm_results[0].confirmed is True


async def test_permission_host_option_kind_shape_rejects():
    """agent-work's 'deny' (cancel/timeout semantic) and reject_once kinds
    must refuse the tool instead of permitting it."""
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[{"option": {"kind": "deny"}}],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert resume.confirm_results[0].confirmed is False


async def test_permission_standard_wire_dict_shape():
    """Spec-compliant hosts answer with the camelCase outcome envelope."""
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[
            {"outcome": {"outcome": "selected", "optionId": "allow_once"}},
        ],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert resume.confirm_results[0].confirmed is True


async def test_permission_unknown_shape_fails_closed():
    """Unrecognized reply shapes must refuse (never hang or permit)."""
    fake = FakeAgentScopeAgent(_permission_events())
    acp_agent = _make_agent(
        fake,
        permission_outcomes=[{"unexpected": True}],
    )
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=_prompt_blocks("delete it"),
        session_id=session.session_id,
    )

    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert resume.confirm_results[0].confirmed is False


# ----------------------------------------------------------------------
# concurrent confirmations (one parked stream, several asks)
# ----------------------------------------------------------------------

class ConcurrentConfirmFakeAgent:
    """Mirrors the engine's concurrent tool batch: several asks surface in
    ONE parked stream, and a resume must cover every ask because the engine
    never re-sends asks it already surfaced (``Agent.reply_stream`` note)."""

    name = "agentscope-acp"

    def __init__(self) -> None:
        self.received: list[Any] = []
        self.state = AgentState()
        self.model: Any = None

    async def reply_stream(self, inputs: Any = None, **kwargs: Any):
        self.received.append(inputs)
        if not isinstance(inputs, UserConfirmResultEvent):
            # First prompt: the batch parks with two asks, one per tool call.
            yield ToolCallStartEvent(
                reply_id="r1", tool_call_id="c1", tool_call_name="Read",
            )
            yield ToolCallStartEvent(
                reply_id="r1", tool_call_id="c2", tool_call_name="Read",
            )
            yield RequireUserConfirmEvent(
                reply_id="r1",
                tool_calls=[
                    ToolCallBlock(
                        id="c1", name="Read", input='{"file_path": "a"}',
                    ),
                ],
            )
            yield RequireUserConfirmEvent(
                reply_id="r1",
                tool_calls=[
                    ToolCallBlock(
                        id="c2", name="Read", input='{"file_path": "b"}',
                    ),
                ],
            )
            return
        # Resume: without a decision for every ASKING call the engine parks
        # again silently - it never re-asks - and the turn dead-ends.
        covered = {result.tool_call.id for result in inputs.confirm_results}
        if covered != {"c1", "c2"}:
            return
        for tool_call_id in ("c1", "c2"):
            yield ToolResultStartEvent(
                reply_id="r1", tool_call_id=tool_call_id,
                tool_call_name="Read",
            )
            yield ToolResultTextDeltaEvent(
                reply_id="r1", tool_call_id=tool_call_id, delta="body",
            )
            yield ToolResultEndEvent(
                reply_id="r1", tool_call_id=tool_call_id,
                state=ToolResultState.SUCCESS,
            )
        yield ReplyEndEvent(session_id="s", reply_id="r1")


async def test_prompt_answers_all_concurrent_confirms_before_resuming():
    """A concurrent batch parks with one ask per tool call; answering only
    the first stranded the rest in ASKING and poisoned the next prompt
    ("Agent is waiting for 1 tool calls ... but received no event")."""
    fake = ConcurrentConfirmFakeAgent()
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("read both files"),
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    conn = acp_agent._conn
    # Every ask of the parked batch reached the client...
    assert [r[1].tool_call_id for r in conn.permission_requests] == [
        "c1",
        "c2",
    ]
    # ...and ONE resume event carried both decisions.
    assert len(fake.received) == 2
    resume = fake.received[1]
    assert isinstance(resume, UserConfirmResultEvent)
    assert [result.tool_call.id for result in resume.confirm_results] == [
        "c1",
        "c2",
    ]
    # The batch executed to completion instead of dead-ending parked.
    statuses = [
        (u.tool_call_id, u.status)
        for _, u in conn.updates
        if u.session_update == "tool_call_update"
    ]
    assert statuses == [("c1", "completed"), ("c2", "completed")]


# ----------------------------------------------------------------------
# recovery of sessions parked by an earlier aborted turn
# ----------------------------------------------------------------------

class ParkedStateFakeAgent:
    """Agent whose state still parks an ASKING tool call from an earlier
    aborted turn; mirrors the engine's interrupt short-circuit."""

    name = "agentscope-acp"

    def __init__(self) -> None:
        self.state = AgentState()
        self.state.append_context(
            self.name,
            [
                ToolCallBlock(
                    id="c9",
                    name="Bash",
                    input='{"cmd": "rm x"}',
                    state=ToolCallState.ASKING,
                ),
            ],
        )
        self.received: list[Any] = []
        self.model: Any = None

    async def reply_stream(self, inputs: Any = None, **kwargs: Any):
        self.received.append(inputs)
        if isinstance(inputs, UserInterruptEvent):
            # Engine: close every awaiting call, end the reply INTERRUPTED.
            for block in self.state.get_unfinished_tool_calls(self.name):
                block.state = ToolCallState.FINISHED
                self.state.context[-1].content.append(
                    ToolResultBlock(
                        id=block.id,
                        name=block.name,
                        output="interrupted",
                        state=ToolResultState.INTERRUPTED,
                    ),
                )
                yield ToolResultEndEvent(
                    reply_id=self.state.reply_id,
                    tool_call_id=block.id,
                    state=ToolResultState.INTERRUPTED,
                )
            yield ReplyEndEvent(
                session_id="s",
                reply_id=self.state.reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )
            return
        # Engine _check_incoming_event: a new message while parked raises.
        if self.state.has_awaiting_tool_calls(self.name):
            raise ValueError(
                "Agent is waiting for 1 tool calls and external execution "
                "results for 0 tool calls, but received no event.",
            )
        yield TextBlockDeltaEvent(reply_id="r2", block_id="b1", delta="ok")
        yield ReplyEndEvent(session_id="s", reply_id="r2")


async def test_prompt_recovers_session_parked_on_earlier_turn():
    """A prompt on a session left parked (client died mid-ask, permission
    round-trip failed, ...) must close the awaiting calls and proceed
    instead of failing with the engine's "waiting for N tool calls" error."""
    fake = ParkedStateFakeAgent()
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("hello again"),
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    # The parked reply was closed with an interrupt BEFORE the new message.
    assert isinstance(fake.received[0], UserInterruptEvent)
    assert fake.received[0].reply_id
    assert _received_text(fake.received[1]) == "hello again"
    # The turn streamed normally: no in-band error chunk, and no cancelled
    # stop reason leaked from the recovery drain.
    texts = [
        u.content.text
        for _, u in acp_agent._conn.updates
        if u.session_update == "agent_message_chunk"
    ]
    assert texts == ["ok"]


# ----------------------------------------------------------------------
# prompt content resolution (@file / resource blocks)
# ----------------------------------------------------------------------

def test_uri_to_path_converts_file_uris():
    if os.name == "nt":
        # Zed's fs/read_text_file only accepts backslash paths (verified
        # against a live capture: file:// URIs and forward slashes return
        # -32002 Resource not found).
        assert _uri_to_path("file:///D:/proj/x.txt") == "D:\\proj\\x.txt"
        assert _uri_to_path("file://D:/proj/x.txt") == "D:\\proj\\x.txt"
    else:
        assert _uri_to_path("file:///home/u/x.txt") == "/home/u/x.txt"
    # Non-file URIs pass through unchanged.
    assert _uri_to_path("https://example.com/x.txt") == "https://example.com/x.txt"


def _fs_enabled() -> ClientCapabilities:
    return ClientCapabilities(
        fs=FileSystemCapabilities(read_text_file=True),
    )


async def test_prompt_resolves_resource_link_and_embedded_resource():
    fake = FakeAgentScopeAgent(_stream_events())
    acp_agent = _make_agent(fake)
    acp_agent._client_capabilities = _fs_enabled()
    conn = acp_agent._conn
    conn._file_contents[_uri_to_path("file:///D:/proj/x.txt")] = "file body"
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=[
            TextContentBlock(type="text", text="look at this"),
            ResourceContentBlock(
                type="resource_link",
                uri="file:///D:/proj/x.txt",
                name="x.txt",
            ),
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    text="inline content",
                    uri="file:///D:/proj/inline.txt",
                ),
            ),
        ],
        session_id=session.session_id,
    )

    assert len(conn.read_requests) == 1
    request_session, path = conn.read_requests[0]
    assert request_session == session.session_id
    assert path.replace("\\", "/").endswith("D:/proj/x.txt")
    user_text = _received_text(fake.received[0])
    assert "look at this" in user_text
    assert "file body" in user_text
    assert "inline content" in user_text


async def test_prompt_skips_unresolvable_resource_link():
    fake = FakeAgentScopeAgent(_stream_events())
    acp_agent = _make_agent(fake)
    acp_agent._client_capabilities = _fs_enabled()
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=[
            TextContentBlock(type="text", text="hello"),
            ResourceContentBlock(
                type="resource_link",
                uri="file:///D:/missing.txt",
                name="missing.txt",
            ),
        ],
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    assert _received_text(fake.received[0]) == "hello"


async def test_prompt_skips_resource_link_without_fs_capability():
    fake = FakeAgentScopeAgent(_stream_events())
    acp_agent = _make_agent(fake)
    conn = acp_agent._conn
    conn._file_contents[_uri_to_path("file:///D:/proj/x.txt")] = "file body"
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(
        prompt=[
            TextContentBlock(type="text", text="hello"),
            ResourceContentBlock(
                type="resource_link",
                uri="file:///D:/proj/x.txt",
                name="x.txt",
            ),
        ],
        session_id=session.session_id,
    )

    # Without client fs capability the link must not be resolved.
    assert conn.read_requests == []
    assert _received_text(fake.received[0]) == "hello"


# ----------------------------------------------------------------------
# runtime model switching
# ----------------------------------------------------------------------

async def test_set_config_option_switches_model():
    fake = FakeAgentScopeAgent(_stream_events())
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.set_config_option(
        config_id="model",
        session_id=session.session_id,
        value="qwen3.6-max",
    )

    assert response.config_options[0].current_value == "qwen3.6-max"
    # The live agent now uses the new model.
    assert fake.model.model == "qwen3.6-max"


async def test_set_config_option_unknown_id_raises():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    session = await acp_agent.new_session(cwd="/tmp")
    with pytest.raises(RequestError):
        await acp_agent.set_config_option(
            config_id="nope",
            session_id=session.session_id,
            value="x",
        )


# ----------------------------------------------------------------------
# session persistence / lifecycle
# ----------------------------------------------------------------------

async def test_prompt_persists_state_to_disk(tmp_path):
    fake = FakeAgentScopeAgent(_stream_events())
    config = _config()
    config.sessions_dir = str(tmp_path)
    acp_agent = AgentScopeAcpAgent(
        config,
        agent_factory=lambda cfg, cwd, mcp_clients=None, state=None: fake,
    )
    acp_agent.on_connect(RecordingConn())
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.prompt(prompt=_prompt_blocks("hi"), session_id=session.session_id)

    saved = tmp_path / f"{session.session_id}.json"
    assert saved.exists()
    raw = json.loads(saved.read_text(encoding="utf-8"))
    assert raw["session_id"] == session.session_id


async def test_load_session_restores_state(tmp_path):
    config = _config()
    config.sessions_dir = str(tmp_path)
    state = AgentState()
    saved = tmp_path / "saved1.json"
    saved.write_text(state.model_dump_json(), encoding="utf-8")

    fake = FakeAgentScopeAgent()
    captured: dict[str, Any] = {}

    def factory(cfg, cwd, mcp_clients=None, state=None):
        captured["state"] = state
        return fake

    acp_agent = AgentScopeAcpAgent(config, agent_factory=factory)
    acp_agent.on_connect(RecordingConn())

    response = await acp_agent.load_session(cwd="/tmp", session_id="saved1")

    assert "saved1" in acp_agent._records
    assert response.config_options[0].current_value == "qwen3.6-plus"
    # The factory received the restored state.
    assert isinstance(captured["state"], AgentState)


async def test_load_session_missing_raises(tmp_path):
    config = _config()
    config.sessions_dir = str(tmp_path)
    acp_agent = AgentScopeAcpAgent(
        config,
        agent_factory=lambda cfg, cwd, mcp_clients=None, state=None: None,
    )
    acp_agent.on_connect(RecordingConn())
    with pytest.raises(RequestError):
        await acp_agent.load_session(cwd="/tmp", session_id="ghost")


async def test_new_session_persists_immediately(tmp_path):
    """The session file must exist right after session/new so a client
    restart (which kills this process) can still session/load it."""
    acp_agent = _make_agent(FakeAgentScopeAgent())
    session = await acp_agent.new_session(cwd="/tmp")

    saved = tmp_path / f"{session.session_id}.json"
    assert saved.exists()
    raw = json.loads(saved.read_text(encoding="utf-8"))
    assert raw["session_id"] == session.session_id
    assert raw["_acp_meta"]["cwd"] == "/tmp"


async def test_prompt_failure_still_persists(tmp_path):
    """A failed prompt turn must still persist state — the process may be
    killed right after and session/load depends on the file existing."""
    fake = FakeAgentScopeAgent(error=RuntimeError("boom"))
    acp_agent = _make_agent(fake)
    session = await acp_agent.new_session(cwd="/tmp")

    response = await acp_agent.prompt(
        prompt=_prompt_blocks("hi"),
        session_id=session.session_id,
    )
    assert response.stop_reason == "end_turn"

    saved = tmp_path / f"{session.session_id}.json"
    assert saved.exists()
    raw = json.loads(saved.read_text(encoding="utf-8"))
    assert raw["session_id"] == session.session_id


async def test_list_sessions_includes_disk_sessions(tmp_path):
    """After a process restart the in-memory table is empty; the session
    list must still surface sessions persisted on disk."""
    acp_agent = _make_agent(FakeAgentScopeAgent())
    session = await acp_agent.new_session(cwd="/tmp")

    # Simulate a restart: fresh agent instance, empty in-memory table.
    fresh = _make_agent(FakeAgentScopeAgent())
    response = await fresh.list_sessions()

    ids = [s.session_id for s in response.sessions]
    assert session.session_id in ids
    info = next(s for s in response.sessions if s.session_id == session.session_id)
    assert info.cwd == "/tmp"


async def test_list_sessions_filters_by_cwd(tmp_path):
    acp_agent = _make_agent(FakeAgentScopeAgent())
    await acp_agent.new_session(cwd="/project/a")
    await acp_agent.new_session(cwd="/project/b")

    response = await acp_agent.list_sessions(cwd="/project/b")

    assert len(response.sessions) == 1
    assert response.sessions[0].cwd == "/project/b"


async def test_list_sessions_reports_open_sessions():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    await acp_agent.new_session(cwd="/tmp")
    response = await acp_agent.list_sessions()

    assert len(response.sessions) == 1
    info = response.sessions[0]
    assert info.cwd == "/tmp"
    assert info.updated_at is not None


async def test_close_session_removes_record(tmp_path):
    acp_agent = _make_agent(FakeAgentScopeAgent())
    session = await acp_agent.new_session(cwd="/tmp")

    await acp_agent.close_session(session_id=session.session_id)

    assert acp_agent._records == {}
    # The persisted state is dropped too — session/list must not
    # resurrect a closed session after a restart.
    assert not (tmp_path / f"{session.session_id}.json").exists()


async def test_resume_session_reuses_inmemory_record():
    acp_agent = _make_agent(FakeAgentScopeAgent())
    session = await acp_agent.new_session(cwd="/tmp")
    original = acp_agent._records[session.session_id]

    response = await acp_agent.resume_session(
        cwd="/elsewhere",
        session_id=session.session_id,
    )

    assert acp_agent._records[session.session_id] is original
    assert original.cwd == "/elsewhere"
    assert response.config_options[0].current_value == "qwen3.6-plus"


async def test_resume_session_restores_from_disk(tmp_path):
    config = _config()
    config.sessions_dir = str(tmp_path)
    state = AgentState()
    saved = tmp_path / "resume1.json"
    saved.write_text(state.model_dump_json(), encoding="utf-8")

    fake = FakeAgentScopeAgent()
    captured: dict[str, Any] = {}

    def factory(cfg, cwd, mcp_clients=None, state=None):
        captured["state"] = state
        return fake

    acp_agent = AgentScopeAcpAgent(config, agent_factory=factory)
    acp_agent.on_connect(RecordingConn())

    response = await acp_agent.resume_session(cwd="/tmp", session_id="resume1")

    assert "resume1" in acp_agent._records
    assert response.config_options[0].current_value == "qwen3.6-plus"
    assert isinstance(captured["state"], AgentState)


async def test_resume_session_creates_fresh_when_unknown(tmp_path):
    """An unknown id degrades to a fresh session under the client's id,
    persisted immediately like session/new."""
    acp_agent = _make_agent(FakeAgentScopeAgent())

    response = await acp_agent.resume_session(cwd="/tmp", session_id="brandnew")

    assert "brandnew" in acp_agent._records
    assert response.config_options[0].current_value == "qwen3.6-plus"
    saved = tmp_path / "brandnew.json"
    assert saved.exists()


def _state_with_saved_model(model_id: str) -> str:
    """A persisted session payload carrying the _acp_meta block written by
    _save_state (cwd/title/.../model_id)."""
    payload = json.loads(AgentState().model_dump_json())
    payload["_acp_meta"] = {"cwd": "/tmp", "model_id": model_id}
    return json.dumps(payload)


async def test_load_session_restores_saved_model(tmp_path):
    """The model chosen at runtime must survive a session/load (process
    restart): without it the rebuilt agent runs the env model and the
    selector snaps back to it."""
    config = _config()
    config.sessions_dir = str(tmp_path)
    saved = tmp_path / "saved2.json"
    saved.write_text(
        _state_with_saved_model("qwen3.6-max"),
        encoding="utf-8",
    )

    fake = FakeAgentScopeAgent()
    acp_agent = AgentScopeAcpAgent(
        config,
        agent_factory=lambda cfg, cwd, mcp_clients=None, state=None: fake,
    )
    acp_agent.on_connect(RecordingConn())

    response = await acp_agent.load_session(cwd="/tmp", session_id="saved2")

    assert response.config_options[0].current_value == "qwen3.6-max"
    assert fake.model.model == "qwen3.6-max"
    assert acp_agent._records["saved2"].model_id == "qwen3.6-max"


async def test_resume_session_restores_saved_model_from_disk(tmp_path):
    config = _config()
    config.sessions_dir = str(tmp_path)
    saved = tmp_path / "resume2.json"
    saved.write_text(
        _state_with_saved_model("qwen3.6-max"),
        encoding="utf-8",
    )

    fake = FakeAgentScopeAgent()
    acp_agent = AgentScopeAcpAgent(
        config,
        agent_factory=lambda cfg, cwd, mcp_clients=None, state=None: fake,
    )
    acp_agent.on_connect(RecordingConn())

    response = await acp_agent.resume_session(cwd="/tmp", session_id="resume2")

    assert response.config_options[0].current_value == "qwen3.6-max"
    assert fake.model.model == "qwen3.6-max"


# ----------------------------------------------------------------------
# MCP server conversion
# ----------------------------------------------------------------------

async def test_build_mcp_clients_stdio_and_http():
    from acp.schema import EnvVariable, HttpMcpServer, McpServerStdio
    from agentscope_acp.config import build_mcp_clients

    clients = build_mcp_clients(
        [
            McpServerStdio(
                name="fs",
                command="mcp-fs",
                args=["--root", "/data"],
                env=[EnvVariable(name="K", value="V")],
            ),
            HttpMcpServer(
                type="http",
                name="web",
                url="https://example.com/mcp",
                headers=[],
            ),
        ],
    )

    assert len(clients) == 2
    stdio, http = clients
    assert stdio.name == "fs" and stdio.is_stateful is True
    assert stdio.mcp_config.type == "stdio_mcp"
    assert stdio.mcp_config.args == ["--root", "/data"]
    assert stdio.mcp_config.env == {"K": "V"}
    assert http.name == "web" and http.is_stateful is False
    assert http.mcp_config.type == "http_mcp"
    assert http.mcp_config.url == "https://example.com/mcp"


async def test_build_mcp_clients_skips_unsupported_types():
    from acp.schema import SseMcpServer
    from agentscope_acp.config import build_mcp_clients

    clients = build_mcp_clients(
        [
            SseMcpServer(
                type="sse",
                name="sse",
                url="https://example.com/sse",
                headers=[],
            ),
        ],
    )
    assert clients == []
