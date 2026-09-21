from __future__ import annotations

import os
import subprocess
from pathlib import Path

from vault_rag.service.git import GitRemote, ManagedGit


def git(*args: str, cwd: Path, env: dict[str, str] | None = None) -> None:
    environment = dict(os.environ)
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    if env is not None:
        environment.update(env)
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def commit(repository: Path, message: str) -> None:
    git("add", ".", cwd=repository)
    git("commit", "-m", message, cwd=repository)
    git("push", "origin", "HEAD:main", cwd=repository)


def test_managed_git_stages_blobs_and_activates_two_safe_commits(tmp_path: Path) -> None:
    """A recursive fetch or enabled hooks would initialize the submodule or write the marker."""
    dependency_remote = tmp_path / "dependency.git"
    dependency_source = tmp_path / "dependency-source"
    dependency_remote.mkdir()
    git("init", "--bare", "--initial-branch=main", cwd=dependency_remote)
    dependency_source.mkdir()
    git("init", "--initial-branch=main", cwd=dependency_source)
    git("config", "user.name", "Test User", cwd=dependency_source)
    git("config", "user.email", "test@example.invalid", cwd=dependency_source)
    (dependency_source / "submodule-note.md").write_text("dependency\n", encoding="utf-8")
    git("add", "submodule-note.md", cwd=dependency_source)
    git("commit", "-m", "dependency", cwd=dependency_source)
    git("remote", "add", "origin", str(dependency_remote), cwd=dependency_source)
    git("push", "origin", "HEAD:main", cwd=dependency_source)

    remote_path = tmp_path / "vault.git"
    source = tmp_path / "source"
    remote_path.mkdir()
    git("init", "--bare", "--initial-branch=main", cwd=remote_path)
    source.mkdir()
    git("init", "--initial-branch=main", cwd=source)
    git("config", "user.name", "Test User", cwd=source)
    git("config", "user.email", "test@example.invalid", cwd=source)
    git("remote", "add", "origin", str(remote_path), cwd=source)
    (source / "note.md").write_text("first\n", encoding="utf-8")
    (source / "manifest.json").write_text('{"version": 1}\n', encoding="utf-8")
    (source / "crlf.txt").write_bytes(b"first\r\n")
    (source / "invalid.bin").write_bytes(b"\xff\x80")
    git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(dependency_remote),
        "dependency",
        cwd=source,
    )
    commit(source, "first")

    checkout = tmp_path / "checkout"
    managed = ManagedGit()
    remote = GitRemote(str(remote_path), "refs/heads/main", None)

    first = managed.fetch(remote, checkout)
    assert first.prepared_checkout is not None
    assert not checkout.exists()
    assert (
        managed.read_blob(first, first.prepared_checkout, "manifest.json", max_bytes=128)
        == b'{"version": 1}\n'
    )
    crlf_blob = managed.read_blob(first, first.prepared_checkout, "crlf.txt", max_bytes=128)
    assert crlf_blob == b"first\r\n"
    assert (
        managed.read_blob(first, first.prepared_checkout, "invalid.bin", max_bytes=128)
        == b"\xff\x80"
    )
    managed.activate(first, checkout)
    assert (checkout / "note.md").read_text(encoding="utf-8") == "first\n"
    assert not (checkout / "dependency" / "submodule-note.md").exists()

    hook_marker = tmp_path / "hook-ran"
    hook = checkout / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\nprintf hook > {hook_marker}\n", encoding="utf-8")
    hook.chmod(0o755)
    local_file = checkout / "local-only.md"
    local_file.write_text("not published\n", encoding="utf-8")
    (source / "note.md").write_text("second\n", encoding="utf-8")
    commit(source, "second")

    second = managed.fetch(remote, checkout)
    assert second.sha != first.sha
    assert (checkout / "note.md").read_text(encoding="utf-8") == "first\n"
    assert local_file.exists()
    managed.activate(second, checkout)
    assert (checkout / "note.md").read_text(encoding="utf-8") == "second\n"
    assert not local_file.exists()
    assert not hook_marker.exists()
    assert not (checkout / "dependency" / "submodule-note.md").exists()
    assert managed.head(checkout) == second.sha
