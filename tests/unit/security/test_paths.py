from pathlib import Path

import pytest

from vault_rag.errors import SecurityError
from vault_rag.security.paths import folded_path_key, reject_unsafe_vault_root, secure_relative_path


def test_secure_relative_path_rejects_parent_escape(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    with pytest.raises(SecurityError):
        secure_relative_path(root, root / ".." / "secret.md")


def test_secure_relative_path_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    target = tmp_path / "secret.md"
    target.write_text("secret")
    (root / "linked.md").symlink_to(target)
    with pytest.raises(SecurityError, match="symlink"):
        secure_relative_path(root, root / "linked.md")


def test_secure_relative_path_uses_samefile_ancestor_for_case_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registered-root"
    root.mkdir()
    # Separate names make this deterministic even on a case-sensitive host;
    # the mock models the same inode reached through a case-variant spelling.
    case_variant_root = tmp_path / "case-variant-root"
    case_variant_root.mkdir()
    candidate = case_variant_root / "Notes.md"
    candidate.write_text("note")

    def samefile(left: str | Path, right: str | Path) -> bool:
        return {Path(left), Path(right)} == {root, case_variant_root}

    monkeypatch.setattr("vault_rag.security.paths.os.path.samefile", samefile)

    assert secure_relative_path(root, candidate) == "Notes.md"


def test_secure_relative_path_samefile_fallback_rejects_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "Vault"
    root.mkdir()
    sibling = tmp_path / "Vault-sibling"
    sibling.mkdir()
    candidate = sibling / "secret.md"
    candidate.write_text("secret")
    monkeypatch.setattr("vault_rag.security.paths.os.path.samefile", lambda _left, _right: False)

    with pytest.raises(SecurityError, match="escapes"):
        secure_relative_path(root, candidate)


def test_secure_relative_path_samefile_fallback_fails_closed_on_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "registered-root"
    root.mkdir()
    candidate_root = tmp_path / "case-variant-root"
    candidate_root.mkdir()
    candidate = candidate_root / "note.md"
    candidate.write_text("note")

    def failing_samefile(_left: str | Path, _right: str | Path) -> bool:
        raise OSError("samefile failed")

    monkeypatch.setattr("vault_rag.security.paths.os.path.samefile", failing_samefile)

    with pytest.raises(SecurityError, match="could not establish"):
        secure_relative_path(root, candidate)


def test_secure_relative_path_normalizes_unicode_but_preserves_display_case(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    display_name = "Notes/Cafe\u0301.MD"
    candidate = root / display_name
    candidate.parent.mkdir()
    candidate.write_text("note")

    assert secure_relative_path(root, candidate) == "Notes/Café.MD"


def test_folded_path_key_detects_case_and_unicode_collisions() -> None:
    assert folded_path_key("Notes/Café.MD") == folded_path_key("notes/Cafe\u0301.md")


def test_reject_unsafe_vault_root_rejects_a_symlinked_component(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(SecurityError, match="symlink"):
        reject_unsafe_vault_root(linked)


def test_reject_unsafe_vault_root_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(SecurityError, match="does not exist"):
        reject_unsafe_vault_root(tmp_path / "absent")


def test_reject_unsafe_vault_root_rejects_a_file(tmp_path: Path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("not a vault")

    with pytest.raises(SecurityError, match="vault root is not a directory"):
        reject_unsafe_vault_root(root)


def test_reject_unsafe_vault_root_accepts_a_plain_directory(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()

    reject_unsafe_vault_root(root)
