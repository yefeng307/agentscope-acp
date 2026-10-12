# -*- coding: utf-8 -*-
"""Unit tests for config.from_env — env parsing edge cases (no network)."""
import os
import subprocess

import pytest

from agentscope_acp.config import (
    DEFAULT_SYSTEM_PROMPT,
    AcpConfig,
    _iter_skill_dirs,
    build_toolkit,
)


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


class TestIterSkillDirs:
    def _skill(self, root, rel, name):
        d = root / rel
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}\n---\n# {name}",
            encoding="utf-8",
        )

    def test_finds_nested_skill_roots(self, tmp_path):
        self._skill(tmp_path, "skill-a", "a")
        self._skill(tmp_path, "group/skill-b", "b")
        (tmp_path / "empty-dir").mkdir()
        roots = _iter_skill_dirs(str(tmp_path))
        assert sorted(roots) == [
            str(tmp_path / "group" / "skill-b"),
            str(tmp_path / "skill-a"),
        ]

    def test_skill_root_not_descended(self, tmp_path):
        # A skill dir with a nested SKILL.md inside: the outer root wins
        # and is not descended further (no duplicate registration).
        self._skill(tmp_path, "outer", "outer")
        self._skill(tmp_path, "outer/inner", "inner")
        assert _iter_skill_dirs(str(tmp_path)) == [str(tmp_path / "outer")]

    def test_entries_without_skill_md_ignored(self, tmp_path):
        # Plain files and dirs without a SKILL.md register nothing.
        (tmp_path / "note.txt").write_text("hi", encoding="utf-8")
        (tmp_path / "empty-dir").mkdir()
        self._skill(tmp_path, "skill-a", "a")
        assert _iter_skill_dirs(str(tmp_path)) == [str(tmp_path / "skill-a")]

    @staticmethod
    def _make_dir_link(target, link):
        """Link ``link`` → ``target`` (both path-like).

        Prefers ``os.symlink``; on Windows without the symlink privilege
        (developer mode / admin) falls back to a junction (``mklink /J``),
        which needs no privilege and is followed by ``os.path.isfile``
        just the same. Returns True on success.
        """
        try:
            os.symlink(str(target), str(link), target_is_directory=True)
            return True
        except OSError:
            pass
        if os.name == "nt":
            try:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                    check=True,
                    capture_output=True,
                )
                return True
            except (OSError, subprocess.CalledProcessError):
                return False
        return False

    def test_follows_symlinked_skill(self, tmp_path, tmp_path_factory):
        # The point of the symlink-aware walk: agent-work enables a skill
        # by symlinking it into the skills dir, and the engine's
        # os.walk-based scan would miss it. The target lives outside the
        # skills tree (as in production), so cycle protection cannot
        # dedupe it away.
        target = tmp_path_factory.mktemp("real-skill")
        (target / "SKILL.md").write_text(
            "---\nname: linked\ndescription: linked\n---\n# linked",
            encoding="utf-8",
        )
        link = tmp_path / "linked-skill"
        if not self._make_dir_link(target, link):
            pytest.skip("symlink/junction unsupported on this platform")
        assert _iter_skill_dirs(str(tmp_path)) == [str(link)]

    def test_symlink_cycle_terminates(self, tmp_path):
        # A symlink pointing back at the skills dir itself (or any
        # ancestor) must not loop: realpath dedup stops the walk.
        self._skill(tmp_path, "skill-a", "a")
        loop = tmp_path / "loop"
        if not self._make_dir_link(tmp_path, loop):
            pytest.skip("symlink/junction unsupported on this platform")
        assert _iter_skill_dirs(str(tmp_path)) == [str(tmp_path / "skill-a")]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert _iter_skill_dirs(str(tmp_path / "nope")) == []


class TestBuildToolkitSkills:
    async def test_single_loader_loads_all_skills(self, tmp_path):
        for rel, name in [("a", "a"), ("group/b", "b")]:
            d = tmp_path / rel
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {name}\n---\n# {name}",
                encoding="utf-8",
            )
        toolkit = build_toolkit(
            enable_tools=False,
            skills_dir=str(tmp_path),
        )
        # enable_tools=False short-circuits before skills are consulted.
        assert toolkit is None

        toolkit = build_toolkit(
            enable_tools=True,
            skills_dir=str(tmp_path),
            tool_names=["Bash"],
        )
        loaders = toolkit.tool_groups[0].skills_or_loaders
        # Exactly one custom loader handles the whole skills dir.
        assert len(loaders) == 1
        skills = await loaders[0].list_skills()
        assert {s.name for s in skills} == {"a", "b"}
        assert {s.markdown for s in skills} == {"# a", "# b"}
        # Nested and flat roots are both found; the mtime cache returns
        # equal skills on the second call.
        assert await loaders[0].list_skills() == skills

    async def test_loader_follows_symlinked_skill(
        self, tmp_path, tmp_path_factory
    ):
        # End to end: a symlinked skill (target outside the tree, as in
        # production) is found and read; Skill.dir keeps the link path so
        # disabling the skill (link removed) also hides its content.
        target = tmp_path_factory.mktemp("real-skill")
        (target / "SKILL.md").write_text(
            "---\nname: linked\ndescription: linked\n---\n# linked",
            encoding="utf-8",
        )
        link = tmp_path / "linked-skill"
        if not TestIterSkillDirs._make_dir_link(target, link):
            pytest.skip("symlink/junction unsupported on this platform")
        toolkit = build_toolkit(
            enable_tools=True,
            skills_dir=str(tmp_path),
            tool_names=["Bash"],
        )
        skills = await toolkit.tool_groups[0].skills_or_loaders[0].list_skills()
        assert [s.name for s in skills] == ["linked"]
        assert skills[0].dir == str(link)

    async def test_loader_tolerates_crlf_and_bom(self, tmp_path):
        d = tmp_path / "crlf-skill"
        d.mkdir()
        (d / "SKILL.md").write_bytes(
            b"\xef\xbb\xbf---\r\n"
            b"name: crlf\r\ndescription: crlf\r\n"
            b"---\r\n# body"
        )
        toolkit = build_toolkit(
            enable_tools=True,
            skills_dir=str(tmp_path),
            tool_names=["Bash"],
        )
        skills = await toolkit.tool_groups[0].skills_or_loaders[0].list_skills()
        assert len(skills) == 1
        assert skills[0].name == "crlf"
        assert skills[0].markdown == "# body"

    async def test_skills_dir_without_skill_md(self, tmp_path):
        (tmp_path / "empty").mkdir()
        toolkit = build_toolkit(
            enable_tools=True,
            skills_dir=str(tmp_path),
            tool_names=["Bash"],
        )
        loaders = toolkit.tool_groups[0].skills_or_loaders
        assert len(loaders) == 1
        assert await loaders[0].list_skills() == []


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


# ── context window sizes ────────────────────────────────────────────────────


def test_context_size_default_from_env(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_CONTEXT_SIZE", "64000")
    config = AcpConfig.from_env()
    assert config.resolve_context_size("any-model") == 64000


def test_context_sizes_per_model_from_env(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_CONTEXT_SIZES", "m-a:1000, m-b:2000")
    config = AcpConfig.from_env()
    assert config.resolve_context_size("m-a") == 1000
    assert config.resolve_context_size("m-b") == 2000
    assert config.resolve_context_size("other") is None


def test_context_size_from_file(tmp_path, monkeypatch):
    path = _write(
        tmp_path,
        "context_size: 96000\n"
        "context_sizes:\n"
        "  m-a: 1000\n"
        "  m-b: 2000\n",
    )
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    config = AcpConfig.from_env()
    assert config.resolve_context_size("m-a") == 1000
    assert config.resolve_context_size("m-b") == 2000
    assert config.resolve_context_size("other") == 96000


def test_context_size_env_overrides_file(tmp_path, monkeypatch):
    path = _write(tmp_path, "context_size: 96000\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    monkeypatch.setenv("AGENTSCOPE_ACP_CONTEXT_SIZE", "48000")
    assert AcpConfig.from_env().default_context_size == 48000


def test_context_size_unset_is_none(monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_ACP_CONTEXT_SIZE", raising=False)
    monkeypatch.delenv("AGENTSCOPE_ACP_CONTEXT_SIZES", raising=False)
    config = AcpConfig.from_env()
    assert config.default_context_size is None
    assert config.context_sizes == {}


def test_context_size_bad_value_ignored(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_CONTEXT_SIZE", "abc")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONTEXT_SIZES", "m-a:12k")
    config = AcpConfig.from_env()
    assert config.default_context_size is None
    assert config.context_sizes == {}


def test_resolve_context_size_prefers_per_model(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_CONTEXT_SIZE", "64000")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONTEXT_SIZES", "m-a:1000")
    config = AcpConfig.from_env()
    assert config.resolve_context_size("m-a") == 1000
    assert config.resolve_context_size("other") == 64000


# ── max output tokens ──────────────────────────────────────────────────────


def test_max_tokens_default_from_env(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_MAX_TOKENS", "8192")
    config = AcpConfig.from_env()
    assert config.resolve_max_tokens("any-model") == 8192


def test_max_tokens_per_model_from_env(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_MAX_TOKENS_BY_MODEL", "m-a:1000, m-b:2000")
    config = AcpConfig.from_env()
    assert config.resolve_max_tokens("m-a") == 1000
    assert config.resolve_max_tokens("m-b") == 2000
    assert config.resolve_max_tokens("other") is None


def test_max_tokens_from_file(tmp_path, monkeypatch):
    path = _write(
        tmp_path,
        "max_tokens: 4096\n"
        "max_tokens_by_model:\n"
        "  m-a: 1000\n"
        "  m-b: 2000\n",
    )
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    config = AcpConfig.from_env()
    assert config.resolve_max_tokens("m-a") == 1000
    assert config.resolve_max_tokens("m-b") == 2000
    assert config.resolve_max_tokens("other") == 4096


def test_max_tokens_env_overrides_file(tmp_path, monkeypatch):
    path = _write(tmp_path, "max_tokens: 4096\n")
    monkeypatch.setenv("AGENTSCOPE_ACP_CONFIG", str(path))
    monkeypatch.setenv("AGENTSCOPE_ACP_MAX_TOKENS", "2048")
    assert AcpConfig.from_env().default_max_tokens == 2048


def test_max_tokens_unset_is_none(monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_ACP_MAX_TOKENS", raising=False)
    monkeypatch.delenv("AGENTSCOPE_ACP_MAX_TOKENS_BY_MODEL", raising=False)
    config = AcpConfig.from_env()
    assert config.default_max_tokens is None
    assert config.max_tokens_by_model == {}


def test_max_tokens_bad_value_ignored(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_MAX_TOKENS", "abc")
    monkeypatch.setenv("AGENTSCOPE_ACP_MAX_TOKENS_BY_MODEL", "m-a:8k")
    config = AcpConfig.from_env()
    assert config.default_max_tokens is None
    assert config.max_tokens_by_model == {}


def test_resolve_max_tokens_prefers_per_model(monkeypatch):
    monkeypatch.setenv("AGENTSCOPE_ACP_MAX_TOKENS", "8192")
    monkeypatch.setenv("AGENTSCOPE_ACP_MAX_TOKENS_BY_MODEL", "m-a:1000")
    config = AcpConfig.from_env()
    assert config.resolve_max_tokens("m-a") == 1000
    assert config.resolve_max_tokens("other") == 8192
