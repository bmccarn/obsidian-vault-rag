"""Manifest-governed, vault-contained source discovery."""

import hashlib
import os
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pathspec  # pyright: ignore[reportMissingImports]

from vault_rag.config.models import ResolvedVault
from vault_rag.domain import SourceKind
from vault_rag.errors import ParseError, SecurityError
from vault_rag.security.paths import (
    folded_path_key,
    reject_unsafe_vault_root,
    secure_relative_path,
)

from .models import DiscoveredSource, DiscoveryFailure, DiscoveryResult

_DEFAULT_EXCLUDES = (
    ".git/**",
    "**/.git/**",
    ".obsidian/**",
    "**/.obsidian/**",
)
_KIND_BY_SUFFIX = {
    ".md": SourceKind.MARKDOWN,
    ".txt": SourceKind.TEXT,
    ".log": SourceKind.LOG,
    ".json": SourceKind.JSON,
}


def _case_variant(name: str) -> str:
    for index, character in enumerate(name):
        if character.isalpha():
            replacement = character.lower() if character.isupper() else character.upper()
            return f"{name[:index]}{replacement}{name[index + 1 :]}"
    return name


def is_case_insensitive(root: Path) -> bool:
    """Detect whether the root inode is reachable through case-variant spelling."""
    variant_name = _case_variant(root.name)
    if variant_name == root.name:
        return False
    try:
        return os.path.samefile(root, root.with_name(variant_name))
    except OSError:
        return False


def _walk_files(root: Path) -> Iterator[Path]:
    """Walk ordinary files and file links without following directory links."""
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name, reverse=True)
        except OSError as exc:
            raise SecurityError(f"could not inspect vault directory: {directory}") from exc

        for entry in ordered:
            candidate = Path(entry.path)
            try:
                if entry.is_symlink():
                    if candidate.suffix:
                        yield candidate
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(candidate)
                elif entry.is_file(follow_symlinks=False):
                    yield candidate
            except OSError as exc:
                raise SecurityError(f"could not inspect vault path: {candidate}") from exc


def _source_kind(relative_path: str) -> SourceKind:
    suffix = Path(relative_path).suffix.lower()
    try:
        return _KIND_BY_SUFFIX[suffix]
    except KeyError as exc:
        raise ParseError(
            f"unsupported source extension: {relative_path}",
            details={"path": relative_path},
        ) from exc


def _lexical_relative_path(root: Path, candidate: Path) -> str:
    """Return the walk-relative POSIX path used only for manifest matching.

    ``_walk_files`` never follows a directory link and never leaves ``root``, so
    this string is a safe *selection* key. It is never stored: every accepted
    file still receives its authoritative path from ``secure_relative_path``.
    """
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SecurityError(f"walked path escaped the vault root: {candidate}") from exc
    return unicodedata.normalize("NFC", relative.as_posix())


def discover_sources(vault: ResolvedVault) -> DiscoveryResult:
    """Discover, validate, decode, and hash sources selected by a vault manifest.

    Include and exclude patterns are evaluated against a lexical vault-relative
    path before any containment, extension, or decoding work, so an excluded
    symlink or excluded binary is never inspected. A selected file that fails
    containment, carries an unsupported extension, cannot be read, or is not
    valid UTF-8 becomes one bounded ``DiscoveryFailure`` and is skipped while
    every other file still reconciles. Root problems and case-fold collisions
    remain vault-level failures.
    """
    reject_unsafe_vault_root(vault.root)
    include_spec = pathspec.PathSpec.from_lines("gitwildmatch", vault.manifest.include)
    default_exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", _DEFAULT_EXCLUDES)
    manifest_exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", vault.manifest.exclude)

    accepted: list[tuple[str, str, SourceKind, Path]] = []
    failures: list[DiscoveryFailure] = []
    folded_paths: dict[str, str] = {}
    check_collisions = is_case_insensitive(vault.root)

    for candidate in _walk_files(vault.root):
        selected_path = _lexical_relative_path(vault.root, candidate)
        if not include_spec.match_file(selected_path):
            continue
        if default_exclude_spec.match_file(selected_path):
            continue
        if manifest_exclude_spec.match_file(selected_path):
            continue

        try:
            relative_path = secure_relative_path(vault.root, candidate)
        except SecurityError as exc:
            # The raised message names the absolute path and can name a link
            # target, so a bounded relative message is recorded instead.
            failures.append(
                DiscoveryFailure(
                    relative_path=selected_path,
                    category=exc.code,
                    message=f"selected path failed containment and was skipped: {selected_path}",
                )
            )
            continue
        try:
            kind = _source_kind(relative_path)
        except ParseError as exc:
            failures.append(
                DiscoveryFailure(
                    relative_path=relative_path,
                    category=exc.code,
                    message=exc.message,
                )
            )
            continue

        folded_path = folded_path_key(relative_path)
        previous_path = folded_paths.get(folded_path)
        if check_collisions and previous_path is not None and previous_path != relative_path:
            raise SecurityError(
                f"case-insensitive collision in vault: {previous_path} and {relative_path}",
                details={"path": relative_path, "conflicting_path": previous_path},
            )
        folded_paths[folded_path] = relative_path
        accepted.append((relative_path, folded_path, kind, candidate))

    sources: list[DiscoveredSource] = []
    for relative_path, folded_path, kind, candidate in sorted(accepted):
        try:
            content = candidate.read_bytes()
            metadata = candidate.stat(follow_symlinks=False)
        except OSError:
            failures.append(
                DiscoveryFailure(
                    relative_path=relative_path,
                    category=ParseError.code,
                    message=f"could not read source: {relative_path}",
                )
            )
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            failures.append(
                DiscoveryFailure(
                    relative_path=relative_path,
                    category=ParseError.code,
                    message=f"source is not valid UTF-8 at byte {exc.start}: {relative_path}",
                )
            )
            continue

        sources.append(
            DiscoveredSource(
                vault_id=vault.manifest.id,
                root=vault.root,
                relative_path=relative_path,
                folded_path=folded_path,
                kind=kind,
                text=text,
                content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
                size_bytes=len(content),
                mtime_ns=metadata.st_mtime_ns,
            )
        )

    if failures:
        reject_unsafe_vault_root(vault.root)
    return DiscoveryResult(sources=tuple(sources), failures=tuple(sorted(failures)))
