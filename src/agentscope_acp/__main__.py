# -*- coding: utf-8 -*-
"""agentscope-acp entry point: run the ACP agent over stdio.

stdout carries the ACP JSON-RPC protocol — all logging goes to stderr
(unless ``AGENTSCOPE_ACP_LOG`` redirects it to a file).
"""
from __future__ import annotations

import asyncio
import logging
import sys

from acp import run_agent

from .agent import AgentScopeAcpAgent
from .config import AcpConfig

logger = logging.getLogger(__name__)


def _setup_logging(config: AcpConfig) -> None:
    """Wire logging to stderr / file — never stdout."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Drop any pre-configured handlers that might write to stdout.
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if config.log_path:
        handler = logging.FileHandler(config.log_path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)


def main() -> None:
    try:
        config = AcpConfig.from_env()
    except ValueError as exc:
        print(f"agentscope-acp: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    _setup_logging(config)
    logger.info(
        "starting agentscope-acp: provider=%s model=%s models=%d",
        config.provider,
        config.model,
        len(config.available_models),
    )

    agent = AgentScopeAcpAgent(config)
    try:
        asyncio.run(run_agent(agent))
    except KeyboardInterrupt:
        logger.info("interrupted, exiting")


if __name__ == "__main__":
    main()
