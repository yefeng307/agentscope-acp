# -*- coding: utf-8 -*-
"""agentscope-acp — expose an AgentScope Agent as an ACP stdio agent.

This package wraps :class:`agentscope.agent.Agent` (in-process) behind the
Agent Client Protocol so that any ACP client — agent-work, Zed, or the
reference python-sdk client — can drive it over NDJSON JSON-RPC on stdio.

MVP scope: ``initialize`` / ``session/new`` / ``session/prompt`` /
``session/cancel`` with streaming text output. See README.md for the
roadmap (tool calls, permissions, session persistence, ...).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
