# -*- coding: utf-8 -*-
"""End-to-end stdio smoke test against the real agentscope-acp process.

Spawns ``uv run agentscope-acp`` through the ACP SDK client and exercises
the full JSON-RPC stack (framing, routing, dispatch) without a real model:

1. ``initialize`` handshake succeeds and advertises agent info.
2. ``session/new`` returns a session id plus the model list.
3. ``session/prompt`` with a fake API key fails at the model call — the
   error is delivered in-band as an ``agent_message_chunk`` and the turn
   ends with ``end_turn`` instead of crashing the process.

Run: ``uv run python tests/smoke_stdio.py``
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client


class PrintClient(Client):
    """Records session updates pushed by the agent."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, str, str]] = []

    async def request_permission(
        self,
        options: list[Any],
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> Any:
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        kind = getattr(update, "session_update", None)
        text = ""
        content = getattr(update, "content", None)
        if content is not None:
            text = getattr(content, "text", "") or ""
        self.updates.append((session_id, str(kind), text))
        print(f"[update] {kind}: {text!r}")


async def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    # Fake key: model construction succeeds, the actual call fails later —
    # which is exactly the in-band error path we want to verify.
    env["DASHSCOPE_API_KEY"] = "sk-smoke-test"

    client = PrintClient()
    async with spawn_agent_process(
        client,
        "uv",
        "run",
        "--project",
        str(project_dir),
        "agentscope-acp",
        env=env,
    ) as (conn, _proc):
        init = await conn.initialize(protocol_version=PROTOCOL_VERSION)
        print(
            f"[init] agent={init.agent_info.name} "
            f"version={init.agent_info.version} "
            f"load_session={init.agent_capabilities.load_session}",
        )
        assert init.agent_info.name == "agentscope-acp"

        session = await conn.new_session(cwd=str(project_dir), mcp_servers=[])
        models = session.models
        print(
            f"[session] id={session.session_id[:8]}... "
            f"current={models.current_model_id if models else None} "
            f"available={[m.model_id for m in models.available_models] if models else []}",
        )
        assert models is not None and models.current_model_id

        response = await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("Say hello in one word.")],
        )
        print(f"[prompt] stop_reason={response.stop_reason}")
        assert response.stop_reason == "end_turn"

        # The fake key makes the model call fail; the error must have been
        # delivered in-band as an agent_message_chunk.
        assert any(kind == "agent_message_chunk" for _, kind, _ in client.updates), (
            "expected an in-band error message chunk"
        )
        error_text = "".join(t for _, k, t in client.updates if k == "agent_message_chunk")
        print(f"[prompt] in-band message: {error_text[:200]!r}")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
