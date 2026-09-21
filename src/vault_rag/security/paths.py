"""Filesystem path guards for vault-contained sources."""

import os
import stat
import unicodedata
from pathlib import Path

from vault_rag.errors import SecurityError


def _reject_symlink_components(path: Path) -> None:
    """Reject symlinks in each existing component of an absolute path."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise SecurityError(f"symlink is not permitted in vault path: {current}")


def _samefile_ancestor_relative_path(root: Path, candidate: Path) -> Path:
    """Return a relative suffix when an existing ancestor is the root inode."""
    for ancestor in candidate.parents:
        try:
            if os.path.samefile(ancestor, root):
                return candidate.relative_to(ancestor)
        except OSError as exc:
            raise SecurityError("could not establish path containment") from exc
    raise SecurityError("path escapes the vault root")


def _relative_to_root(root: Path, candidate: Path) -> Path:
    """Use lexical containment first, then verify a case-variant root inode."""
    try:
        return candidate.relative_to(root)
    except ValueError:
        return _samefile_ancestor_relative_path(root, candidate)


def secure_relative_path(root: Path, candidate: Path) -> str:
    """Return a normalized relative path only when it remains inside ``root``.

    Existing path components are inspected without following links before the
    resolved path is checked, so a link cannot redirect a source outside a
    registered vault.
    """
    raw_root = Path(root).expanduser()
    raw_candidate = Path(candidate).expanduser()
    requested = raw_candidate if raw_candidate.is_absolute() else raw_root / raw_candidate

    try:
        _reject_symlink_components(raw_root.absolute())
        _reject_symlink_components(requested.absolute())
        canonical_root = raw_root.resolve(strict=True)
        canonical_candidate = requested.resolve(strict=False)
    except FileNotFoundError as exc:
        raise SecurityError(f"vault root does not exist: {root}") from exc
    except OSError as exc:
        raise SecurityError(f"could not inspect vault path: {candidate}") from exc

    relative = _relative_to_root(canonical_root, canonical_candidate)

    relative_text = unicodedata.normalize("NFC", relative.as_posix())
    if not relative_text or relative_text == "." or Path(relative_text).is_absolute():
        raise SecurityError("path must be a non-empty relative vault path")
    return relative_text


def folded_path_key(relative_path: str) -> str:
    """Produce a Unicode- and case-folded key for collision detection."""
    return unicodedata.normalize("NFC", relative_path).casefold()


def reject_unsafe_vault_root(root: Path) -> None:
    """Fail closed when a vault root is missing or has a symlinked component.

    Discovery converts per-file containment failures into bounded diagnostics.
    Checking the root once keeps a relocated, missing, or symlinked root a loud
    vault-level failure rather than one diagnostic per selected file — and stops
    such a root from reporting an empty source set that would delete a healthy
    index.
    """
    absolute = Path(root).expanduser().absolute()
    _reject_symlink_components(absolute)
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise SecurityError(f"vault root does not exist: {root}") from exc
    if not resolved.is_dir():
        raise SecurityError(f"vault root is not a directory: {root}")
