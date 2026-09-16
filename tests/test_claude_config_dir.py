"""Tests for ``CLAUDE_CONFIG_DIR``, Claude Code's configuration-home override.

Claude Code resolves its config home as the variable when set and ``~/.claude``
otherwise, and moves ``.claude.json`` *into* that directory along with it. ``sx``
mirrors both, so every path it reads, previews and deletes follows.

The adapter tests here all build a **decoy** store under the patched ``home()``
alongside the real one under the override. Without the decoy a passing test
proves nothing: it could not tell "followed the override" from "found the only
store there was".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.memory import discover_memories
from sx.util import (
    claude_config_dir,
    claude_config_file,
    claude_config_override,
    claude_config_warning,
    is_within,
    scratchpad_root,
)

HOME = Path("/h")
PROJECT = "/work/proj"
SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# --- the pure helpers -----------------------------------------------------


def test_config_home_defaults_to_dot_claude() -> None:
    assert claude_config_dir(HOME) == Path("/h/.claude")


def test_an_absolute_override_replaces_the_home_default(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/alt/cfg")
    assert claude_config_dir(HOME) == Path("/alt/cfg")


def test_the_global_config_moves_into_the_override(monkeypatch) -> None:
    """``.claude.json`` follows the override, and not the way the layout hints.

    Unset it sits *beside* ``~/.claude``; set it sits *inside* the override.
    There is no expression in terms of the config directory that is right in
    both cases, which is why it is derived separately.
    """
    assert claude_config_file(HOME) == Path("/h/.claude.json")

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/alt/cfg")
    resolved = claude_config_file(HOME)
    assert resolved == Path("/alt/cfg/.claude.json")
    assert resolved != Path("/h/.claude.json")  # missed the move entirely
    assert resolved != Path("/alt/cfg/.claude/.claude.json")  # derived from the dir
    assert resolved != Path("/alt/.claude.json")  # derived from the dir's parent


def test_a_relative_override_is_ignored(monkeypatch) -> None:
    """A relative value must never be resolved against the working directory.

    ``store_roots()`` is the delete allowlist. Honoring ``CLAUDE_CONFIG_DIR=
    .claude`` would aim it at whatever directory ``sx`` happened to launch from
    — and this repository has its own ``.claude/``.
    """
    for value in (".claude", "alt/cfg", "./x"):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", value)
        assert claude_config_override() is None
        assert claude_config_dir(HOME) == Path("/h/.claude")
        assert str(Path.cwd()) not in str(claude_config_dir(HOME))


def test_a_tilde_override_is_ignored(monkeypatch) -> None:
    """Claude Code does no tilde expansion and rejects the result as relative.

    Expanding it here would have ``sx`` confidently reading a directory the
    harness itself refuses to start against.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/alt")
    assert claude_config_dir(HOME) == Path("/h/.claude")
    assert claude_config_dir(HOME) != Path.home() / "alt"


def test_a_blank_override_is_ignored(monkeypatch) -> None:
    for value in ("", "   "):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", value)
        assert claude_config_override() is None
        assert claude_config_dir(HOME) == Path("/h/.claude")


def test_the_home_directory_is_refused_as_an_override(monkeypatch) -> None:
    """``CLAUDE_CONFIG_DIR=$HOME`` would put ``~/projects`` on the allowlist."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(Path.home()))
    assert claude_config_override() is None
    assert "home directory" in (claude_config_warning() or "")


def test_a_filesystem_root_is_refused_as_an_override(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/")
    assert claude_config_override() is None
    assert "filesystem root" in (claude_config_warning() or "")


def test_an_ignored_override_is_never_silent(monkeypatch) -> None:
    assert claude_config_warning() is None
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "relative/path")
    assert "not an absolute path" in (claude_config_warning() or "")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/alt/cfg")
    assert claude_config_warning() is None


def test_the_override_is_read_at_call_time(monkeypatch) -> None:
    """Guards against anyone caching the value in a module-level constant.

    ``sx.util`` is imported long before any ``setenv``, so a captured value
    would freeze to "unset" and every override test would pass vacuously.
    """
    assert claude_config_dir(HOME) == Path("/h/.claude")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/alt/cfg")
    assert claude_config_dir(HOME) == Path("/alt/cfg")


def test_the_scratchpad_root_ignores_the_override(monkeypatch) -> None:
    """The scratchpad is uid-derived, so it does not move with the config home."""
    before = scratchpad_root()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/alt/cfg")
    assert scratchpad_root() == before
    assert "/alt/cfg" not in str(scratchpad_root())


def test_the_suite_scrubs_the_variable() -> None:
    """Documents the conftest invariant; fails loudly if the fixture is removed."""
    assert os.environ.get("CLAUDE_CONFIG_DIR") is None


# --- the adapter ----------------------------------------------------------


def _store(base: Path, *, sid: str = SID, prompt: str = "hi") -> None:
    """Populate a full Claude store rooted at ``base``."""
    folder = base / "projects" / encode_project_dir(PROJECT)
    folder.mkdir(parents=True)
    (folder / f"{sid}.jsonl").write_text(
        json.dumps({"type": "user", "cwd": PROJECT, "message": {"content": prompt}}) + "\n"
    )
    for sub in ("file-history", "session-env", "tasks"):
        (base / sub / sid).mkdir(parents=True)
        (base / sub / sid / "data").write_text("x")
    (base / "security" / "agent-sdk-venv").mkdir(parents=True)
    (base / "security" / f"security_warnings_state_{sid}.json").write_text("{}")
    (base / "history.jsonl").write_text(
        json.dumps({"display": prompt, "project": PROJECT, "sessionId": sid}) + "\n"
    )
    (folder / "memory").mkdir()
    (folder / "memory" / "MEMORY.md").write_text("# Memory Index\n")
    (folder / "memory" / f"{prompt}-fact.md").write_text(f"---\nname: {prompt}\n---\nbody\n")


@pytest.fixture()
def redirected(tmp_path: Path, monkeypatch) -> tuple[ClaudeAdapter, Path, Path]:
    """A real store under an override, plus a populated decoy under ``home()``.

    Returns:
        ``(adapter, override_dir, decoy_home)``.
    """
    decoy_home = tmp_path / "decoy-home"
    cfg = tmp_path / "cfg"
    _store(decoy_home / ".claude", prompt="decoy")
    (decoy_home / ".claude.json").write_text(
        json.dumps({"projects": {PROJECT: {"allowedTools": ["Decoy"]}}})
    )
    _store(cfg, prompt="real")
    (cfg / ".claude.json").write_text(
        json.dumps({"projects": {PROJECT: {"allowedTools": ["Bash"]}}})
    )

    monkeypatch.setattr(claude_mod, "home", lambda: decoy_home)
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: tmp_path / "scratch")
    monkeypatch.setattr("sx.memory.home", lambda: decoy_home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg))
    return ClaudeAdapter(), cfg, decoy_home


def test_every_store_root_follows_the_override(redirected) -> None:
    adapter, cfg, decoy_home = redirected
    roots = adapter.store_roots()
    assert roots[0] == cfg / "projects"
    assert not any(is_within(root, [decoy_home]) for root in roots)


def test_the_config_home_is_never_itself_a_store_root(redirected) -> None:
    """The blast-radius rule survives the override: name subdirectories, not the base."""
    adapter, cfg, _ = redirected
    assert cfg not in adapter.store_roots()


def test_the_decoy_store_is_invisible(redirected) -> None:
    adapter, cfg, _ = redirected
    files = list(adapter.session_files())
    assert files == [cfg / "projects" / encode_project_dir(PROJECT) / f"{SID}.jsonl"]
    assert [s.title for s in adapter.discover()] == ["real"]


def test_derived_paths_follow_the_override(redirected) -> None:
    adapter, cfg, _ = redirected
    assert adapter._config_dir() == cfg
    assert adapter._config_file() == cfg / ".claude.json"
    assert adapter._history_path() == cfg / "history.jsonl"
    assert adapter._projects_root() == cfg / "projects"
    assert all(is_within(p, [cfg]) for p in adapter.protected_paths())


def test_the_cascade_follows_the_override(redirected) -> None:
    """A delete must take the override's artifacts and leave the decoy's alone."""
    adapter, cfg, decoy_home = redirected
    session = next(iter(adapter.discover()))
    targets = adapter.correlated_paths(session)
    assert targets, "cascade found nothing to remove"
    assert all(is_within(p, [cfg, cfg.parent / "scratch"]) for p in targets)
    assert not any(is_within(p, [decoy_home]) for p in targets)

    adapter.delete(session)
    for sub in ("file-history", "session-env", "tasks"):
        assert not (cfg / sub / SID).exists()
        assert (decoy_home / ".claude" / sub / SID).exists()


def test_protected_paths_follow_the_override(redirected) -> None:
    adapter, cfg, _ = redirected
    venv = cfg / "security" / "agent-sdk-venv"
    result = adapter._delete_paths([venv], dry_run=True)
    assert result.skipped.get(venv) == "protected path (refused)"
    assert venv.exists()


def test_memory_discovery_follows_the_override(redirected) -> None:
    _, cfg, decoy_home = redirected
    found = discover_memories()
    assert found, "no memories discovered"
    assert all(is_within(m.path, [cfg]) for m in found)
    assert not any(is_within(m.path, [decoy_home]) for m in found)
    assert "decoy" not in {m.name for m in found}


def test_stale_artifact_labels_name_the_real_directory(redirected) -> None:
    """A label saying ``~/.claude/tasks`` while deleting elsewhere would be a lie."""
    adapter, cfg, _ = redirected
    (cfg / "tasks" / "11111111-2222-3333-4444-555555555555").mkdir(parents=True)
    reasons = [o.reason for o in adapter.find_orphans()]
    stale = [r for r in reasons if "tasks" in r]
    assert stale, "no stale-task orphan was produced"
    assert all(str(cfg) in r for r in stale)
    assert not any("~/.claude" in r for r in stale)


# --- strictness and isolation ---------------------------------------------


def test_a_missing_override_reports_the_harness_unavailable(tmp_path, monkeypatch) -> None:
    """Strict, no fallback — and not defeated by the uid-derived scratchpad root.

    The inherited ``available()`` is "any store root exists", and the scratchpad
    root is a store root that exists on any machine that has ever run Claude
    Code. Keying on it would call a redirected-but-absent store "ready".
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: scratch)
    (tmp_path / ".claude" / "projects").mkdir(parents=True)

    adapter = ClaudeAdapter()
    assert adapter.available() is True

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "nope"))
    assert adapter.available() is False
    assert list(adapter.session_files()) == []


def test_resolution_never_escapes_a_patched_home(tmp_path, monkeypatch) -> None:
    """The regression guard for this design, in the spirit of ``test_memory.py``.

    The util helpers take the home directory as an argument precisely so they
    never resolve ``home`` in ``sx.util``'s namespace. Were they to do so, every
    adapter-level ``monkeypatch.setattr(claude_mod, "home", ...)`` would slip and
    the suite would read — and delete from — the developer's real store.
    """
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setattr("sx.util.home", lambda: Path("/should/never/be/used"))

    adapter = ClaudeAdapter()
    for path in (
        adapter._config_dir(),
        adapter._config_file(),
        adapter._history_path(),
        adapter._projects_root(),
        *adapter.protected_paths(),
    ):
        assert is_within(path, [tmp_path]), f"{path} escaped the sandbox"
