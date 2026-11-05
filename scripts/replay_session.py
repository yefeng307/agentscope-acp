# -*- coding: utf-8 -*-
"""Replay a persisted agentscope-acp session against the model endpoint.

Diagnoses the "old session keeps failing with server_error while new
sessions work" symptom: loads the persisted AgentState, formats its
history exactly like the engine does (OpenAIChatFormatter), and sends it
to the OpenAI-compatible endpoint with NO retry (so the raw server
response is visible). With ``--bisect`` it progressively drops history
to find the smallest prefix that still triggers the failure.

Usage (run from the repo root, key read from env / config file):
    uv run python scripts/replay_session.py <session_id>
    uv run python scripts/replay_session.py <session_id> --bisect
    uv run python scripts/replay_session.py <session_id> --model deepseek-v4-flash

Environment:
    OPENAI_API_KEY / OPENAI_BASE_URL, or the api_key/base_url keys of
    ~/.agentscope-acp/agentscope-acp.yaml (the file the agent itself uses).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

SESSIONS_DIR = Path(
    os.environ.get(
        "AGENTSCOPE_ACP_SESSIONS_DIR",
        "~/.agentscope-acp/sessions",
    ),
).expanduser()
CONFIG_FILE = Path(
    os.environ.get("AGENTSCOPE_ACP_CONFIG", "~/.agentscope-acp/agentscope-acp.yaml"),
).expanduser()


def _load_config_file() -> dict:
    if not CONFIG_FILE.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8-sig")) or {}
    except Exception as exc:  # pragma: no cover - IO
        print(f"[warn] cannot read config file {CONFIG_FILE}: {exc}")
        return {}


async def _call(model: str, base_url: str | None, api_key: str, messages: list[dict]):
    """One raw request, no retries. Returns (ok, detail_text)."""
    from openai import AsyncOpenAI

    kwargs: dict = {"api_key": api_key, "max_retries": 0, "timeout": 120}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncOpenAI(**kwargs)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=32,
        )
        return True, f"OK: {resp.choices[0].message.content!r}"
    except Exception as exc:
        detail = str(exc)
        status = getattr(getattr(exc, "response", None), "status_code", "?")
        body = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                body = json.dumps(resp.json(), ensure_ascii=False)[:500]
            except Exception:
                body = getattr(resp, "text", "?")
        return False, f"HTTP {status} | {detail[:300]} | body={body}"


async def _replay(session_id: str, model: str, text_only: bool) -> tuple[list[dict], bool, str]:
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.state import AgentState

    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.is_file():
        sys.exit(f"session file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    state = AgentState.model_validate(raw)
    messages = state.context  # list[Msg]

    if text_only:
        kept = []
        for msg in messages:
            blocks = [
                b
                for b in msg.content
                if getattr(b, "type", None) not in ("tool_call", "tool_result", "thinking")
            ]
            if blocks:
                kept.append(msg.model_copy(update={"content": blocks}))
        messages = kept

    cfg = _load_config_file()
    api_key = os.environ.get("OPENAI_API_KEY") or str(cfg.get("api_key") or "")
    base_url = os.environ.get("OPENAI_BASE_URL") or str(cfg.get("base_url") or "") or None
    if not api_key:
        sys.exit("no api key: set OPENAI_API_KEY or the api_key key of the config file")

    formatter = OpenAIChatFormatter()
    formatted = await formatter.format(list(messages))
    ok, detail = await _call(model, base_url, api_key, formatted)
    return formatted, ok, detail


async def _bisect(model: str, formatted: list[dict]):
    """Binary-search the history prefix for the failure trigger."""
    # formatted list order = chronological; keep first block (system, if any)
    system_prefix = []
    rest = list(formatted)
    if rest and rest[0].get("role") == "system":
        system_prefix = [rest.pop(0)]

    async def ok_with(prefix: list[dict]) -> bool:
        ok, _ = await _call_with(model, system_prefix + prefix)
        return ok

    # Quick check: full history fails? (caller already verified, but re-verify)
    if await ok_with(rest):
        print("[bisect] full history succeeds — nothing to bisect")
        return

    lo, hi = 1, len(rest)
    while lo < hi:
        mid = (lo + hi) // 2
        ok, detail = await _call_with(model, system_prefix + rest[:mid])
        print(f"[bisect] prefix={mid}/{len(rest)} -> {'OK' if ok else 'FAIL'}")
        if ok:
            lo = mid + 1
        else:
            hi = mid
    print(f"[bisect] failure appears once message #{lo} is included "
          f"(of {len(rest)} history messages)")
    print(f"[bisect] offending message: {json.dumps(rest[lo-1], ensure_ascii=False)[:800]}")


async def _call_with(model, messages):
    cfg = _load_config_file()
    api_key = os.environ.get("OPENAI_API_KEY") or str(cfg.get("api_key") or "")
    base_url = os.environ.get("OPENAI_BASE_URL") or str(cfg.get("base_url") or "") or None
    return await _call(model, base_url, api_key, messages)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", help="session id (filename stem under the sessions dir)")
    parser.add_argument("--model", help="model override (default: persisted _acp_meta.model_id)")
    parser.add_argument("--bisect", action="store_true", help="binary-search history for the failing message")
    parser.add_argument("--text-only", action="store_true", help="strip tool_call/tool_result/thinking blocks before replay")
    args = parser.parse_args()

    path = SESSIONS_DIR / f"{args.session_id}.json"
    if not path.is_file():
        sys.exit(f"session file not found: {path}")
    meta = json.loads(path.read_text(encoding="utf-8")).get("_acp_meta", {})
    model = args.model or meta.get("model_id") or "deepseek-v4-flash"

    formatted, ok, detail = await _replay(args.session_id, model, args.text_only)
    print(f"session={args.session_id} model={model} history_msgs={len(formatted)}")
    print("RESULT:", detail)
    if not ok and args.bisect:
        await _bisect(model, formatted)
    if not ok:
        print("\nHint: a 4xx (e.g. 400/422) is a request-shape problem;")
        print("a 5xx (500/502/503) means the endpoint failed on THIS history.")


if __name__ == "__main__":
    asyncio.run(main())
