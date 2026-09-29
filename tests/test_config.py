# -*- coding: utf-8 -*-
"""Unit tests for config.from_env — env parsing edge cases (no network)."""
import pytest

from agentscope_acp.config import DEFAULT_SYSTEM_PROMPT, AcpConfig


@pytest.fixture(autouse=True)
def _isolate_config_lookup(tmp_path, monkeypatch):
    """Config lookup probes ./agentscope-acp.yaml and
    ~/.agentscope-acp/config.yaml — keep tests off any real user files."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("AGENTSCOPE_ACP_CONFIG", raising=False)


def _write(tmp_path, content, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


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


# ── Config file ────────────────────────────────────────────────────────────


def test_config_file_explicit_path(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_ACP_MODEL", raising=False)
    monkeypatch.delenv("AGENTSCOPE_ACP_SYSTEM_PROMPT", raising=False)
    path = _write(tmp_path, "system_prompt: hi from file\nmodel: m-file\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    config = AcpConfig.from_env()
    assert config.system_prompt == "hi from file"
    assert config.model == "m-file"


def test_env_overrides_file(tmp_path, monkeypatch):
    path = _write(tmp_path, "model: m-file\nsystem_prompt: from file\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    monkeypatch.setenv("AGENTSCOPE_ACP_MODEL", "m-env")
    config = AcpConfig.from_env()
    assert config.model == "m-env"
    assert config.system_prompt == "from file"


def test_file_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_ACP_SYSTEM_PROMPT", raising=False)
    path = _write(tmp_path, "system_prompt: custom\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    config = AcpConfig.from_env()
    assert config.system_prompt == "custom"
    assert config.system_prompt != DEFAULT_SYSTEM_PROMPT


def test_base_url_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    path = _write(tmp_path, "base_url: https://api.example.com/v1\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    assert AcpConfig.from_env().base_url == "https://api.example.com/v1"


def test_env_overrides_base_url(tmp_path, monkeypatch):
    path = _write(tmp_path, "base_url: https://file.example.com/v1\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
    assert AcpConfig.from_env().base_url == "https://env.example.com/v1"


def test_available_models_list_in_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_ACP_AVAILABLE_MODELS", raising=False)
    path = _write(tmp_path, "available_models: [m-a, m-b]\nmodel: m-a\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    config = AcpConfig.from_env()
    assert [e.model_id for e in config.available_models] == ["m-a", "m-b"]


def test_project_search_path_used(tmp_path):
    """./agentscope-acp.yaml is picked up from the process cwd
    (agent-work spawns with cwd=workspace)."""
    _write(tmp_path, "system_prompt: from cwd\n", name="agentscope-acp.yaml")
    config = AcpConfig.from_env()
    assert config.system_prompt == "from cwd"


def test_no_config_file_defaults(monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_ACP_SYSTEM_PROMPT", raising=False)
    assert AcpConfig.from_env().system_prompt == DEFAULT_SYSTEM_PROMPT


def test_explicit_config_missing_raises(tmp_path, monkeypatch):
    """Explicit AGENTSCOPE_ACP_CONFIG is a hard error, not a silent skip."""
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(tmp_path / "nope.yaml"))
    with pytest.raises(ValueError):
        AcpConfig.from_env()


def test_explicit_config_bad_yaml_raises(tmp_path, monkeypatch):
    path = _write(tmp_path, "{ not: [valid yaml\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    with pytest.raises(ValueError):
        AcpConfig.from_env()


def test_search_path_bad_yaml_warns(tmp_path):
    """An unparseable search-path file must not break startup — env/defaults
    still apply."""
    _write(tmp_path, "{ not: [valid yaml\n", name="agentscope-acp.yaml")
    config = AcpConfig.from_env()
    assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
