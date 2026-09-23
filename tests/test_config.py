# -*- coding: utf-8 -*-
"""Unit tests for config.from_env — env parsing edge cases (no network)."""
from agentscope_acp.config import AcpConfig


def test_skills_dir_expands_tilde(monkeypatch):
    """Hosts seed "~" env values (agent-work's nativeSkillsDirs and
    AGENTSCOPE_ACP_SKILLS_DIR point at the same ~/.agentscope-acp/skills
    directory); os.path.isdir would silently fail on the unexpanded form."""
    monkeypatch.setenv("AGENTSCOPE_ACP_SKILLS_DIR", "~/.agentscope-acp/skills")
    config = AcpConfig.from_env()
    assert config.skills_dir is not None
    assert "~" not in config.skills_dir
    assert config.skills_dir.endswith(".agentscope-acp/skills")


def test_skills_dir_unset_is_none(monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_ACP_SKILLS_DIR", raising=False)
    assert AcpConfig.from_env().skills_dir is None


def test_available_models_include_current(monkeypatch):
    """AGENTSCOPE_ACP_MODEL stays selectable even when absent from the
    available list — the Guid page's model picker needs it as a value."""
    monkeypatch.setenv("AGENTSCOPE_ACP_MODEL", "m1")
    monkeypatch.setenv("AGENTSCOPE_ACP_AVAILABLE_MODELS", "m2, m3")
    config = AcpConfig.from_env()
    assert [e.model_id for e in config.available_models] == ["m1", "m2", "m3"]
