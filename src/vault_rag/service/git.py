"""Safe, managed Git operations for private vault synchronization."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from base64 import b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol, cast
from urllib.parse import urlsplit

from vault_rag.errors import RepositorySyncError

_GIT_TIMEOUT_SECONDS = 30.0
_SHA_LENGTHS = frozenset({40, 64})
_RUNTIME_ENVIRONMENT_NAMES = frozenset(
    {
        "HOME",
        "LANG",
        "LANGUAGE",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
    }
)
_RUNTIME_ENVIRONMENT_PREFIXES = ("LC_",)


@dataclass(frozen=True, slots=True)
class GitRemote:
    """The remote, optional exact-origin trust path, and private credential."""

    url: str
    ref: str
    credential: str | None = field(repr=False)
    ca_cert_path: str | None = None


@dataclass(frozen=True, slots=True)
class FetchedCommit:
    """A verified commit, optionally staged outside its active checkout path."""

    sha: str
    prepared_checkout: Path | None


class GitRunner(Protocol):
    """The narrow execution dependency used by :class:`ManagedGit`."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None,
        environ: Mapping[str, str],
        timeout: float,
    ) -> str: ...

    def run_bytes(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None,
        environ: Mapping[str, str],
        timeout: float,
    ) -> bytes: ...


class GitCommandRunner:
    """Run a Git command without exposing its input or output in errors."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None,
        environ: Mapping[str, str],
        timeout: float,
    ) -> str:
        output = self._execute(args, cwd=cwd, environ=environ, timeout=timeout, text=True)
        assert isinstance(output, str)
        return output

    def run_bytes(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None,
        environ: Mapping[str, str],
        timeout: float,
    ) -> bytes:
        output = self._execute(args, cwd=cwd, environ=environ, timeout=timeout, text=False)
        assert isinstance(output, bytes)
        return output

    def _execute(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None,
        environ: Mapping[str, str],
        timeout: float,
        text: bool,
    ) -> str | bytes:
        operation = _operation_name(args)
        try:
            completed = subprocess.run(
                list(args),
                cwd=cwd,
                env=dict(environ),
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=text,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RepositorySyncError(f"git {operation} timed out") from None
        except (OSError, UnicodeError):
            raise RepositorySyncError(f"git {operation} could not run") from None

        if completed.returncode != 0:
            raise RepositorySyncError(
                f"git {operation} failed with exit status {completed.returncode}"
            )
        return cast(str | bytes, completed.stdout)


class ManagedGit:
    """Fetch, inspect, and activate a service-owned Git checkout."""

    def __init__(
        self, runner: GitRunner | None = None, *, timeout: float = _GIT_TIMEOUT_SECONDS
    ) -> None:
        self._runner = runner if runner is not None else GitCommandRunner()
        self._timeout = timeout

    def fetch(self, remote: GitRemote, checkout: Path) -> FetchedCommit:
        """Fetch ``remote.ref`` without mutating the active worktree when it exists."""
        checkout.parent.mkdir(parents=True, exist_ok=True)
        self._discard_staged_checkouts(checkout)
        if checkout.exists():
            self._run(
                ("git", "fetch", "--no-recurse-submodules", remote.url, remote.ref),
                cwd=checkout,
                remote=remote,
            )
            return FetchedCommit(self._fetch_head(checkout), None)

        prepared = Path(tempfile.mkdtemp(prefix=f".{checkout.name}.", dir=checkout.parent))
        try:
            self._run(("git", "init", "--quiet", str(prepared)), cwd=None, remote=None)
            self._run(("git", "remote", "add", "origin", remote.url), cwd=prepared, remote=None)
            self._run(
                ("git", "fetch", "--no-recurse-submodules", "origin", remote.ref),
                cwd=prepared,
                remote=remote,
            )
            sha = self._fetch_head(prepared)
            self._run(("git", "checkout", "--detach", "--force", sha), cwd=prepared, remote=None)
        except Exception:
            shutil.rmtree(prepared, ignore_errors=True)
            raise
        return FetchedCommit(sha, prepared)

    def activate(self, fetched: FetchedCommit, checkout: Path) -> None:
        """Publish a staged checkout or force an existing checkout to ``fetched.sha``."""
        sha = _verified_sha(fetched.sha)
        prepared = fetched.prepared_checkout
        if prepared is not None:
            if checkout.exists():
                raise RepositorySyncError(
                    "prepared checkout cannot replace an existing destination"
                )
            if prepared.parent.resolve() != checkout.parent.resolve() or not prepared.is_dir():
                raise RepositorySyncError("prepared checkout is invalid")
            try:
                prepared.replace(checkout)
            except OSError:
                raise RepositorySyncError("prepared checkout could not be activated") from None
            return

        if not checkout.is_dir():
            raise RepositorySyncError("managed checkout is unavailable")
        self._run(("git", "checkout", "--detach", "--force", sha), cwd=checkout, remote=None)
        self._run(("git", "clean", "-ffd"), cwd=checkout, remote=None)

    def discard(self, fetched: FetchedCommit, checkout: Path) -> None:
        """Remove an unpublished staged checkout without touching the active checkout."""
        prepared = fetched.prepared_checkout
        if prepared is not None:
            self._discard_staged_checkout(prepared, checkout)

    def _discard_staged_checkouts(self, checkout: Path) -> None:
        for prepared in checkout.parent.glob(f".{checkout.name}.*"):
            self._discard_staged_checkout(prepared, checkout)

    @staticmethod
    def _discard_staged_checkout(prepared: Path, checkout: Path) -> None:
        try:
            managed = (
                prepared.parent.resolve() == checkout.parent.resolve()
                and prepared.name.startswith(f".{checkout.name}.")
                and prepared.is_dir()
            )
        except OSError:
            return
        if managed:
            shutil.rmtree(prepared, ignore_errors=True)

    def read_blob(
        self,
        fetched: FetchedCommit,
        checkout: Path,
        relative_path: str,
        *,
        max_bytes: int,
    ) -> bytes:
        """Read one bounded blob from a fetched commit before it is activated."""
        sha = _verified_sha(fetched.sha)
        if max_bytes < 0:
            raise RepositorySyncError("git blob size limit is invalid")
        path = _verified_relative_path(relative_path)
        object_name = f"{sha}:{path}"
        try:
            object_type = self._run(
                ("git", "cat-file", "-t", object_name), cwd=checkout, remote=None
            )
        except RepositorySyncError:
            raise RepositorySyncError("git blob is unavailable") from None
        if object_type.strip() != "blob":
            raise RepositorySyncError("git blob is not a file")
        try:
            size_text = self._run(("git", "cat-file", "-s", object_name), cwd=checkout, remote=None)
            size = int(size_text.strip())
        except (RepositorySyncError, ValueError):
            raise RepositorySyncError("git blob is unavailable") from None
        if size < 0:
            raise RepositorySyncError("git blob is unavailable")
        if size > max_bytes:
            raise RepositorySyncError("git blob exceeds configured size limit")
        try:
            content = self._run_bytes(
                ("git", "show", "--no-ext-diff", "--format=", object_name),
                cwd=checkout,
                remote=None,
            )
        except RepositorySyncError:
            raise RepositorySyncError("git blob is unavailable") from None
        if len(content) > max_bytes:
            raise RepositorySyncError("git blob exceeds configured size limit")
        return content

    def head(self, checkout: Path) -> str | None:
        """Return the verified active commit, if the checkout currently has one."""
        if not checkout.is_dir():
            return None
        try:
            output = self._run(
                ("git", "rev-parse", "--verify", "HEAD^{commit}"), cwd=checkout, remote=None
            )
            return _verified_sha(output.strip())
        except RepositorySyncError:
            return None

    def _fetch_head(self, checkout: Path) -> str:
        output = self._run(
            ("git", "rev-parse", "--verify", "FETCH_HEAD^{commit}"), cwd=checkout, remote=None
        )
        return _verified_sha(output.strip())

    def _run(self, args: Sequence[str], *, cwd: Path | None, remote: GitRemote | None) -> str:
        return self._runner.run(
            args,
            cwd=cwd,
            environ=_git_environment(remote),
            timeout=self._timeout,
        )

    def _run_bytes(
        self, args: Sequence[str], *, cwd: Path | None, remote: GitRemote | None
    ) -> bytes:
        return self._runner.run_bytes(
            args,
            cwd=cwd,
            environ=_git_environment(remote),
            timeout=self._timeout,
        )


def _operation_name(args: Sequence[str]) -> str:
    if len(args) > 1 and args[0] == "git":
        return args[1]
    return "command"


def _git_environment(remote: GitRemote | None) -> dict[str, str]:
    """Build a minimal process environment for non-interactive Git operations."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in _RUNTIME_ENVIRONMENT_NAMES or name.startswith(_RUNTIME_ENVIRONMENT_PREFIXES)
    }
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    config = [
        ("credential.helper", ""),
        ("core.hooksPath", os.devnull),
        ("fetch.recurseSubmodules", "false"),
        ("submodule.recurse", "false"),
        ("http.followRedirects", "false"),
    ]
    if remote is not None and (remote.ca_cert_path is not None or remote.credential is not None):
        origin = _https_origin(remote.url)
        if remote.ca_cert_path is not None:
            config.append((f"http.{origin}.sslCAInfo", remote.ca_cert_path))
        if remote.credential is not None:
            config.append(
                (
                    f"http.{origin}.extraHeader",
                    _authorization_header(remote.url, remote.credential),
                )
            )
    environment["GIT_CONFIG_COUNT"] = str(len(config))
    for index, (key, value) in enumerate(config):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    return environment


def _authorization_header(url: str, credential: str) -> str:
    """Use GitHub's documented Smart HTTP token transport without changing generic remotes."""
    hostname = urlsplit(url).hostname
    if hostname is not None and hostname.lower() == "github.com":
        encoded = b64encode(f"x-access-token:{credential}".encode()).decode("ascii")
        return f"Authorization: Basic {encoded}"
    return f"Authorization: Bearer {credential}"


def _https_origin(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RepositorySyncError("git remote must use an HTTPS origin")
    origin = f"https://{parsed.hostname}"
    if parsed.port is not None:
        origin = f"{origin}:{parsed.port}"
    return f"{origin}/"


def _verified_sha(value: str) -> str:
    if len(value) not in _SHA_LENGTHS or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RepositorySyncError("git returned an invalid commit identifier")
    return value


def _verified_relative_path(value: str) -> str:
    if not value or "\x00" in value or any(ord(character) < 32 for character in value):
        raise RepositorySyncError("git blob path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise RepositorySyncError("git blob path is invalid")
    return value
