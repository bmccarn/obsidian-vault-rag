from __future__ import annotations

import subprocess
from base64 import b64encode
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from vault_rag.errors import RepositorySyncError
from vault_rag.service.git import FetchedCommit, GitCommandRunner, GitRemote, ManagedGit

SHA = "a" * 40
TOKEN = "super-secret-token"


class RecordingRunner:
    def __init__(self, *, blob_size: int = 4, blob_kind: str = "blob") -> None:
        self.blob_size = blob_size
        self.blob_kind = blob_kind
        self.calls: list[tuple[tuple[str, ...], Path | None, dict[str, str], float]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None,
        environ: Mapping[str, str],
        timeout: float,
    ) -> str:
        command = tuple(args)
        self.calls.append((command, cwd, dict(environ), timeout))
        if command[1:4] == ("rev-parse", "--verify", "FETCH_HEAD^{commit}"):
            return f"{SHA}\n"
        if command[1:4] == ("rev-parse", "--verify", "HEAD^{commit}"):
            return f"{SHA}\n"
        if command[1:4] == ("cat-file", "-t", f"{SHA}:manifest.json"):
            return f"{self.blob_kind}\n"
        if command[1:4] == ("cat-file", "-s", f"{SHA}:manifest.json"):
            return f"{self.blob_size}\n"
        if command[1] == "show":
            return "data"
        return ""

    def run_bytes(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None,
        environ: Mapping[str, str],
        timeout: float,
    ) -> bytes:
        return self.run(args, cwd=cwd, environ=environ, timeout=timeout).encode("utf-8")


def configured_values(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        environ[f"GIT_CONFIG_KEY_{index}"]: environ[f"GIT_CONFIG_VALUE_{index}"]
        for index in range(int(environ["GIT_CONFIG_COUNT"]))
    }


def test_existing_fetch_keeps_credentials_out_of_commands_and_enforces_git_safety(
    tmp_path: Path,
) -> None:
    """Removing any secure Git configuration makes this test fail."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    runner = RecordingRunner()
    remote = GitRemote("https://example.test/vault.git", "refs/heads/main", TOKEN)

    fetched = ManagedGit(runner).fetch(remote, checkout)

    assert fetched == FetchedCommit(sha=SHA, prepared_checkout=None)
    assert TOKEN not in repr(remote)
    assert all(TOKEN not in " ".join(args) for args, _, _, _ in runner.calls)
    assert all(environ["GIT_TERMINAL_PROMPT"] == "0" for _, _, environ, _ in runner.calls)
    for args, _, environ, _ in runner.calls:
        values = configured_values(environ)
        assert values["credential.helper"] == ""
        assert values["core.hooksPath"] == "/dev/null"
        assert values["fetch.recurseSubmodules"] == "false"
        assert values["submodule.recurse"] == "false"
        assert values["http.followRedirects"] == "false"
        if args[1] == "fetch":
            assert values["http.https://example.test/.extraHeader"] == (
                f"Authorization: Bearer {TOKEN}"
            )
        else:
            assert not any(key.startswith("http.https://") for key in values)
    assert any(args[1] == "fetch" for args, _, _, _ in runner.calls)
    assert any(args[1] == "rev-parse" for args, _, _, _ in runner.calls)


def test_github_fetch_uses_basic_smart_http_token_transport(tmp_path: Path) -> None:
    """GitHub's Git endpoint rejects the API-style Bearer token transport."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    runner = RecordingRunner()

    ManagedGit(runner).fetch(
        GitRemote("https://github.com/example/private-vault.git", "refs/heads/main", TOKEN),
        checkout,
    )

    encoded = b64encode(f"x-access-token:{TOKEN}".encode()).decode("ascii")
    fetch_environments = [environ for args, _, environ, _ in runner.calls if args[1] == "fetch"]
    assert len(fetch_environments) == 1
    values = configured_values(fetch_environments[0])
    header = values["http.https://github.com/.extraHeader"]
    assert header == f"Authorization: Basic {encoded}"
    assert TOKEN not in header


def test_fetch_strips_hostile_transport_environment_and_scopes_trust_to_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Restoring an ambient transport override would make the fetch unsafe."""
    for name in (
        "GIT_SSL_NO_VERIFY",
        "GIT_SSL_CAINFO",
        "GIT_SSL_CAPATH",
        "GIT_SSL_CERT",
        "GIT_SSL_KEY",
        "GIT_SSL_PINNEDPUBLICKEY",
        "GIT_SSL_VERSION",
        "GIT_SSL_CIPHER_LIST",
        "GIT_HTTP_PROXY",
        "GIT_HTTP_USER_AGENT",
        "GIT_PROXY_COMMAND",
        "GIT_ALLOW_PROTOCOL",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "GIT_TRACE",
        "GIT_TRACE_CURL",
        "GIT_TRACE_REDACT",
        "GIT_TRACE2",
        "GIT_EXEC_PATH",
        "GIT_SSH_COMMAND",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "GIT_FUTURE_KNOB",
        "LD_PRELOAD",
    ):
        monkeypatch.setenv(name, "hostile")
    monkeypatch.setenv("PATH", "/safe/runtime/path")
    monkeypatch.setenv("HOME", "/safe/runtime/home")
    runner = RecordingRunner()
    remote = GitRemote(
        "https://git-fixture.test/vault.git",
        "refs/heads/main",
        TOKEN,
        ca_cert_path="/fixture/ca/root.crt",
    )

    ManagedGit(runner).fetch(remote, tmp_path / "checkout")

    for args, _, environ, _ in runner.calls:
        assert not {
            "GIT_SSL_NO_VERIFY",
            "GIT_SSL_CAINFO",
            "GIT_SSL_CAPATH",
            "GIT_SSL_CERT",
            "GIT_SSL_KEY",
            "GIT_SSL_PINNEDPUBLICKEY",
            "GIT_SSL_VERSION",
            "GIT_SSL_CIPHER_LIST",
            "GIT_HTTP_PROXY",
            "GIT_HTTP_USER_AGENT",
            "GIT_PROXY_COMMAND",
            "GIT_ALLOW_PROTOCOL",
            "GIT_CONFIG",
            "GIT_CONFIG_PARAMETERS",
            "GIT_TRACE",
            "GIT_TRACE_CURL",
            "GIT_TRACE_REDACT",
            "GIT_TRACE2",
            "GIT_EXEC_PATH",
            "GIT_SSH_COMMAND",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "GIT_FUTURE_KNOB",
            "LD_PRELOAD",
        }.intersection(environ)
        assert environ["PATH"] == "/safe/runtime/path"
        assert environ["HOME"] == "/safe/runtime/home"
        values = configured_values(environ)
        if args[1] == "fetch":
            assert values["http.https://git-fixture.test/.sslCAInfo"] == "/fixture/ca/root.crt"
            assert values["http.https://git-fixture.test/.extraHeader"] == (
                f"Authorization: Bearer {TOKEN}"
            )
        else:
            assert not any(key.startswith("http.https://") for key in values)
        assert "http.extraHeader" not in values


def test_plain_local_remote_uses_hardened_base_configuration_without_https_scoping(
    tmp_path: Path,
) -> None:
    """Local contract fixtures do not need HTTPS-only credentials or trust settings."""
    runner = RecordingRunner()

    fetched = ManagedGit(runner).fetch(
        GitRemote((tmp_path / "remote.git").as_uri(), "refs/heads/main", None),
        tmp_path / "checkout",
    )

    assert fetched.sha == SHA
    for _, _, environ, _ in runner.calls:
        values = configured_values(environ)
        assert values["credential.helper"] == ""
        assert values["http.followRedirects"] == "false"
        assert not any(key.startswith("http.https://") for key in values)


def test_first_fetch_stages_a_same_parent_checkout_without_publishing(tmp_path: Path) -> None:
    """Replacing staging with a direct clone would make the active path appear too early."""
    checkout = tmp_path / "active"
    runner = RecordingRunner()

    fetched = ManagedGit(runner).fetch(
        GitRemote("https://example.test/vault.git", "refs/heads/main", None), checkout
    )

    assert fetched.sha == SHA
    assert fetched.prepared_checkout is not None
    assert fetched.prepared_checkout.parent == checkout.parent
    assert fetched.prepared_checkout != checkout
    assert not checkout.exists()
    assert any(args[1:3] == ("init", "--quiet") for args, _, _, _ in runner.calls)
    assert any(args[1] == "checkout" for args, _, _, _ in runner.calls)


def test_activate_replaces_an_absent_destination_or_updates_an_existing_checkout(
    tmp_path: Path,
) -> None:
    """Removing the atomic rename or forced clean checkout changes the managed tree contract."""
    managed = ManagedGit(RecordingRunner())
    checkout = tmp_path / "checkout"
    prepared = tmp_path / ".checkout-stage"
    prepared.mkdir()
    (prepared / "note.md").write_text("staged\n", encoding="utf-8")

    managed.activate(FetchedCommit(SHA, prepared), checkout)

    assert (checkout / "note.md").read_text(encoding="utf-8") == "staged\n"
    assert not prepared.exists()

    runner = RecordingRunner()
    ManagedGit(runner).activate(FetchedCommit(SHA, None), checkout)

    assert [args[1] for args, _, _, _ in runner.calls] == ["checkout", "clean"]
    assert runner.calls[0][0] == ("git", "checkout", "--detach", "--force", SHA)
    assert runner.calls[1][0] == ("git", "clean", "-ffd")


def test_read_blob_is_bounded_and_rejects_non_files_without_git_output(tmp_path: Path) -> None:
    """Skipping object type or size checks could expose directories or oversized content."""
    runner = RecordingRunner(blob_size=5)
    managed = ManagedGit(runner)

    assert (
        managed.read_blob(FetchedCommit(SHA, None), tmp_path, "manifest.json", max_bytes=5)
        == b"data"
    )

    oversized = RecordingRunner(blob_size=6)
    with pytest.raises(RepositorySyncError) as error:
        ManagedGit(oversized).read_blob(
            FetchedCommit(SHA, None), tmp_path, "manifest.json", max_bytes=5
        )
    assert str(error.value) == "git blob exceeds configured size limit"
    assert all(args[1] != "show" for args, _, _, _ in oversized.calls)

    non_file = RecordingRunner(blob_kind="tree")
    with pytest.raises(RepositorySyncError) as error:
        ManagedGit(non_file).read_blob(
            FetchedCommit(SHA, None), tmp_path, "manifest.json", max_bytes=5
        )
    assert str(error.value) == "git blob is not a file"
    assert all(args[1] != "show" for args, _, _, _ in non_file.calls)


def test_read_blob_hides_invalid_size_output_in_its_exception_chain(tmp_path: Path) -> None:
    """Chaining the parser error would expose untrusted Git stdout."""

    class InvalidSizeRunner(RecordingRunner):
        def run(self, *args: Any, **kwargs: Any) -> str:
            output = super().run(*args, **kwargs)
            if args[0][1:3] == ("cat-file", "-s"):
                return f"{TOKEN} is not a size"
            return output

    with pytest.raises(RepositorySyncError) as error:
        ManagedGit(InvalidSizeRunner()).read_blob(
            FetchedCommit(SHA, None), tmp_path, "manifest.json", max_bytes=5
        )

    assert str(error.value) == "git blob is unavailable"
    assert error.value.__cause__ is None


def test_activate_hides_prepared_rename_errors_in_its_exception_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Chaining an OS rename error would expose private filesystem paths."""
    prepared = tmp_path / ".prepared"
    checkout = tmp_path / "checkout"
    prepared.mkdir()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError(f"{TOKEN} from {self} to {target}")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(RepositorySyncError) as error:
        ManagedGit(RecordingRunner()).activate(FetchedCommit(SHA, prepared), checkout)

    assert str(error.value) == "prepared checkout could not be activated"
    assert error.value.__cause__ is None


@pytest.mark.parametrize("relative_path", ["", "/manifest.json", "../manifest.json", "a/../../b"])
def test_read_blob_rejects_unsafe_paths_before_invoking_git(
    tmp_path: Path, relative_path: str
) -> None:
    """Allowing revision-like or traversal paths would change the Git object selected."""
    runner = RecordingRunner()

    with pytest.raises(RepositorySyncError) as error:
        ManagedGit(runner).read_blob(FetchedCommit(SHA, None), tmp_path, relative_path, max_bytes=5)

    assert str(error.value) == "git blob path is invalid"
    assert runner.calls == []


def test_invalid_commit_output_is_rejected_without_exposing_it(tmp_path: Path) -> None:
    """Returning an unverified revision expression would let Git select a different object."""

    class InvalidShaRunner(RecordingRunner):
        def run(self, *args: Any, **kwargs: Any) -> str:
            super().run(*args, **kwargs)
            return "not-a-commit-and-not-a-secret\n"

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    with pytest.raises(RepositorySyncError) as error:
        ManagedGit(InvalidShaRunner()).fetch(
            GitRemote("https://example.test/vault.git", "refs/heads/main", TOKEN), checkout
        )

    assert str(error.value) == "git returned an invalid commit identifier"
    assert TOKEN not in str(error.value)


def test_head_returns_none_for_missing_checkout_and_verified_commit_for_existing_one(
    tmp_path: Path,
) -> None:
    """Changing head to expose an invalid revision would violate the commit identifier boundary."""
    managed = ManagedGit(RecordingRunner())

    assert managed.head(tmp_path / "missing") is None
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert managed.head(checkout) == SHA


def test_head_treats_invalid_commit_output_as_missing(tmp_path: Path) -> None:
    """Malformed local Git output must preserve the fail-closed optional contract."""

    class InvalidShaRunner(RecordingRunner):
        def run(self, *args: Any, **kwargs: Any) -> str:
            super().run(*args, **kwargs)
            return "not-a-commit-and-not-a-secret\n"

    checkout = tmp_path / "checkout"
    checkout.mkdir()

    assert ManagedGit(InvalidShaRunner()).head(checkout) is None


def test_command_runner_uses_a_list_without_shell_and_sanitizes_failed_process_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shell mode or subprocess output would leak credentials and allow injection."""
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            [], 23, stdout="ignored", stderr=f"{TOKEN}\n" + "x" * 5000
        )

    monkeypatch.setattr("vault_rag.service.git.subprocess.run", fake_run)

    with pytest.raises(RepositorySyncError) as error:
        GitCommandRunner().run(
            ["git", "fetch", "https://example.test/vault.git"],
            cwd=tmp_path,
            environ={"AUTH": TOKEN},
            timeout=1.0,
        )

    command = seen["args"][0]  # type: ignore[index]
    kwargs = seen["kwargs"]  # type: ignore[assignment]
    assert command == ["git", "fetch", "https://example.test/vault.git"]
    assert isinstance(kwargs, dict)
    assert kwargs["shell"] is False
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["capture_output"] is True
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs
    assert kwargs["check"] is False
    assert TOKEN not in str(error.value)
    assert "example.test" not in str(error.value)
    assert len(str(error.value)) < 100

    assert error.value.__cause__ is None


def test_command_runner_sanitizes_timeouts_without_command_or_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Returning TimeoutExpired details would include command arguments and environment secrets."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["git", TOKEN], 1.0)

    monkeypatch.setattr("vault_rag.service.git.subprocess.run", fake_run)

    with pytest.raises(RepositorySyncError) as error:
        GitCommandRunner().run(["git", "fetch"], cwd=tmp_path, environ={"AUTH": TOKEN}, timeout=1.0)

    assert str(error.value) == "git fetch timed out"
    assert TOKEN not in repr(error.value)
    assert error.value.__cause__ is None
