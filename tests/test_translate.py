# -*- coding: utf-8 -*-
"""Unit tests for the AgentScope -> ACP translation layer.

No live model or ACP client needed: events are constructed directly and
the ACP side is asserted on pydantic model shapes.
"""
from __future__ import annotations

from acp.schema import TextContentBlock
from agentscope.event import (
    ModelCallEndEvent,
    ReplyEndEvent,
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

from agentscope_acp.translate import (
    STOP_CANCELLED,
    STOP_END_TURN,
    STOP_MAX_TURN_REQUESTS,
    TurnTranslator,
    agent_message_chunk,
    extract_text,
)


class TestExtractText:
    def test_joins_text_blocks(self):
        blocks = [
            TextContentBlock(type="text", text="hello "),
            TextContentBlock(type="text", text="world"),
        ]
        assert extract_text(blocks) == "hello \nworld"

    def test_skips_non_text_blocks(self):
        blocks = [
            TextContentBlock(type="text", text="only this"),
            type("FakeImage", (), {"type": "image", "data": "..."})(),
        ]
        assert extract_text(blocks) == "only this"

    def test_empty_prompt(self):
        assert extract_text(None) == ""
        assert extract_text([]) == ""


class TestAgentMessageChunk:
    def test_shape(self):
        chunk = agent_message_chunk("r1:b1", "hi")
        data = chunk.model_dump(exclude_none=True, by_alias=True)
        assert data["sessionUpdate"] == "agent_message_chunk"
        assert data["messageId"] == "r1:b1"
        assert data["content"] == {"type": "text", "text": "hi"}


class TestTurnTranslator:
    def test_text_delta_produces_chunk_with_stable_message_id(self):
        translator = TurnTranslator()
        first = translator.process(
            _text_delta("r1", "b1", "Hel"),
        )
        second = translator.process(
            _text_delta("r1", "b1", "lo"),
        )
        assert len(first) == 1 and len(second) == 1
        assert first[0].message_id == second[0].message_id == "r1:b1"
        assert first[0].content.text == "Hel"
        assert second[0].content.text == "lo"

    def test_distinct_blocks_get_distinct_message_ids(self):
        translator = TurnTranslator()
        a = translator.process(_text_delta("r1", "b1", "one"))[0]
        b = translator.process(_text_delta("r1", "b2", "two"))[0]
        assert a.message_id != b.message_id

    def test_empty_delta_is_dropped(self):
        translator = TurnTranslator()
        assert translator.process(_text_delta("r1", "b1", "")) == []

    def test_usage_accumulates_across_model_calls(self):
        translator = TurnTranslator()
        translator.process(_model_call_end("r1", 10, 20))
        translator.process(_model_call_end("r1", 5, 7))
        assert translator.input_tokens == 15
        assert translator.output_tokens == 27

    def test_stop_reason_mapping(self):
        translator = TurnTranslator()
        assert translator.stop_reason() == STOP_END_TURN

        translator.process(_reply_end(ReplyFinishedReason.COMPLETED))
        assert translator.stop_reason() == STOP_END_TURN

        translator.process(_reply_end(ReplyFinishedReason.INTERRUPTED))
        assert translator.stop_reason() == STOP_CANCELLED

        translator.process(_reply_end(ReplyFinishedReason.EXCEED_MAX_ITERS))
        assert translator.stop_reason() == STOP_MAX_TURN_REQUESTS

    def test_ignored_events(self):
        translator = TurnTranslator()
        ignored = [
            ThinkingBlockDeltaEvent(reply_id="r1", block_id="t1", delta="hm"),
            ToolResultStartEvent(
                reply_id="r1",
                tool_call_id="c1",
                tool_call_name="Bash",
            ),
        ]
        for event in ignored:
            assert translator.process(event) == []


# ----------------------------------------------------------------------
# Event constructors
# ----------------------------------------------------------------------

def _text_delta(reply_id: str, block_id: str, delta: str):
    from agentscope.event import TextBlockDeltaEvent

    return TextBlockDeltaEvent(
        reply_id=reply_id,
        block_id=block_id,
        delta=delta,
    )


def _model_call_end(reply_id: str, input_tokens: int, output_tokens: int):
    return ModelCallEndEvent(
        reply_id=reply_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _reply_end(reason: ReplyFinishedReason):
    return ReplyEndEvent(
        session_id="s",
        reply_id="r1",
        finished_reason=reason,
    )


class TestToolCallTranslation:
    def test_full_tool_call_lifecycle(self):
        translator = TurnTranslator()

        # ToolCallStart -> a tool_call update advertising the invocation.
        start = translator.process(
            ToolCallStartEvent(
                reply_id="r1",
                tool_call_id="c1",
                tool_call_name="Bash",
            ),
        )
        assert len(start) == 1
        call = start[0]
        assert call.session_update == "tool_call"
        assert call.tool_call_id == "c1"
        assert call.title == "Bash"
        assert call.kind == "execute"
        assert call.status == "in_progress"

        # Argument deltas are buffered; ToolCallEnd flushes them as raw_input.
        translator.process(
            ToolCallDeltaEvent(
                reply_id="r1", tool_call_id="c1", delta='{"command":',
            ),
        )
        translator.process(
            ToolCallDeltaEvent(
                reply_id="r1", tool_call_id="c1", delta='"ls"}',
            ),
        )
        end_call = translator.process(
            ToolCallEndEvent(reply_id="r1", tool_call_id="c1"),
        )
        assert len(end_call) == 1
        assert end_call[0].session_update == "tool_call_update"
        assert end_call[0].raw_input == '{"command":"ls"}'
        assert end_call[0].status == "in_progress"

        # Result deltas are buffered; ToolResultEnd flushes output + status.
        translator.process(
            ToolResultStartEvent(
                reply_id="r1", tool_call_id="c1", tool_call_name="Bash",
            ),
        )
        translator.process(
            ToolResultTextDeltaEvent(
                reply_id="r1", tool_call_id="c1", delta="file.",
            ),
        )
        translator.process(
            ToolResultTextDeltaEvent(
                reply_id="r1", tool_call_id="c1", delta="txt",
            ),
        )
        end_result = translator.process(
            ToolResultEndEvent(
                reply_id="r1", tool_call_id="c1", state=ToolResultState.SUCCESS,
            ),
        )
        assert len(end_result) == 1
        assert end_result[0].session_update == "tool_call_update"
        assert end_result[0].status == "completed"
        assert end_result[0].raw_output == "file.txt"
        assert end_result[0].content[0].content.text == "file.txt"

    def test_failed_tool_result_maps_to_failed_status(self):
        translator = TurnTranslator()
        translator.process(
            ToolCallStartEvent(
                reply_id="r1", tool_call_id="c2", tool_call_name="Write",
            ),
        )
        result = translator.process(
            ToolResultEndEvent(
                reply_id="r1", tool_call_id="c2", state=ToolResultState.ERROR,
            ),
        )
        assert len(result) == 1
        assert result[0].status == "failed"
        # No streamed output -> neither content nor raw_output is set.
        assert result[0].raw_output is None
        assert result[0].content is None

    def test_unknown_tool_kind_falls_back_to_other(self):
        translator = TurnTranslator()
        start = translator.process(
            ToolCallStartEvent(
                reply_id="r1", tool_call_id="c3", tool_call_name="CustomTool",
            ),
        )
        assert start[0].kind == "other"
