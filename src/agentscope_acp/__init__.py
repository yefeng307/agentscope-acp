# -*- coding: utf-8 -*-
"""agentscope-acp — expose an AgentScope Agent as an ACP stdio agent.

This package wraps :class:`agentscope.agent.Agent` (in-process) behind the
Agent Client Protocol so that any ACP client — agent-work, Zed, or the
reference python-sdk client — can drive it over NDJSON JSON-RPC on stdio.

MVP scope: ``initialize`` / ``session/new`` / ``session/prompt`` /
``session/cancel`` with streaming text and tool-call updates (built-in
Bash/Read/Write/Edit/Grep/Glob tools, auto-approved inside ``cwd``).
See README.md for the roadmap (permissions, session persistence, ...).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
