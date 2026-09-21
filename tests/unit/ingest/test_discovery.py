# mypy: disable-error-code=import-untyped
# pyright: reportMissingImports=false

import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from vault_rag.config.loader import load_manifest
from vault_rag.config.models import EgressPolicy, ResolvedVault, VaultManifest
from vault_rag.domain import SourceKind
from vault_rag.errors import SecurityError
from vault_rag.ingest.discovery import _lexical_relative_path, discover_sources

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "vault-a"


def write_note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_vault(
    root: Path,
    *,
    include: tuple[str, ...] = ("**/*.md",),
    exclude: tuple[str, ...] = (),
) -> ResolvedVault:
    root.mkdir(parents=True, exist_ok=True)
    manifest = VaultManifest(
        schema_version=1,
        id="test-vault",
        egress_policy=EgressPolicy.REMOTE_ALLOWED,
        include=include,
        exclude=exclude,
    )
    return ResolvedVault(root=root.resolve(), manifest=manifest)


@pytest.fixture
def resolved_vault(tmp_path: Path) -> ResolvedVault:
    root = tmp_path / "vault-a"
    shutil.copytree(
        FIXTURE_ROOT,
        root,
        ignore=shutil.ignore_patterns("CLAUDE.local.md"),
    )
    return ResolvedVault(root=root.resolve(), manifest=load_manifest(root / ".vault-rag.toml"))


def test_discovery_obeys_excludes_and_hashes_content(resolved_vault: ResolvedVault) -> None:
    sources = discover_sources(resolved_vault).sources
    relative_paths = [source.relative_path for source in sources]

    assert "templates/excluded.md" not in relative_paths
    assert relative_paths == [
        "attachments/events.log",
        "projects/example.md",
    ]
    assert all(source.content_hash.startswith("sha256:") for source in sources)


def test_discovery_returns_deterministic_posix_order(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault")
    write_note(vault.root / "z-last.md", "last")
    write_note(vault.root / "A" / "middle.md", "middle")
    write_note(vault.root / "a-first.md", "first")

    assert [source.relative_path for source in discover_sources(vault).sources] == [
        "A/middle.md",
        "a-first.md",
        "z-last.md",
    ]


def test_discovery_applies_default_private_directory_excludes(tmp_path: Path) -> None:
    vault = make_vault(
        tmp_path / "vault",
        include=("**/*.md", ".git/**", ".obsidian/**"),
    )
    write_note(vault.root / "visible.md", "visible")
    write_note(vault.root / ".git" / "private.md", "git")
    write_note(vault.root / ".obsidian" / "private.md", "obsidian")

    assert [source.relative_path for source in discover_sources(vault).sources] == ["visible.md"]


def test_discovery_maps_each_supported_extension(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault", include=("**/*",))
    expected = {
        "data.json": SourceKind.JSON,
        "events.log": SourceKind.LOG,
        "note.md": SourceKind.MARKDOWN,
        "plain.txt": SourceKind.TEXT,
    }
    for name in expected:
        (vault.root / name).write_text("content", encoding="utf-8")

    assert {
        source.relative_path: source.kind for source in discover_sources(vault).sources
    } == expected


def test_excluded_symlink_is_skipped_before_containment(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault", exclude=("templates/**",))
    outside = tmp_path / "outside.md"
    write_note(outside, "secret")
    write_note(vault.root / "visible.md", "visible")
    (vault.root / "templates").mkdir()
    (vault.root / "templates" / "linked.md").symlink_to(outside)
    (vault.root / ".git" / "refs").mkdir(parents=True)
    (vault.root / ".git" / "refs" / "leak.md").symlink_to(outside)

    result = discover_sources(vault)

    assert [source.relative_path for source in result.sources] == ["visible.md"]
    assert result.failures == ()


def test_included_symlink_becomes_a_bounded_failure_without_being_followed(
    tmp_path: Path,
) -> None:
    vault = make_vault(tmp_path / "vault")
    outside = tmp_path / "outside.md"
    write_note(outside, "SECRETLINKTARGET")
    write_note(vault.root / "visible.md", "visible")
    (vault.root / "linked.md").symlink_to(outside)

    result = discover_sources(vault)

    assert [source.relative_path for source in result.sources] == ["visible.md"]
    assert [(failure.relative_path, failure.category) for failure in result.failures] == [
        ("linked.md", "security_error")
    ]
    message = result.failures[0].message
    assert "linked.md" in message
    assert "SECRETLINKTARGET" not in message
    assert str(tmp_path) not in message


def test_undecodable_included_file_becomes_a_bounded_failure(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault")
    write_note(vault.root / "visible.md", "visible")
    (vault.root / "broken.md").write_bytes(b"valid\n\xff")

    result = discover_sources(vault)

    assert [source.relative_path for source in result.sources] == ["visible.md"]
    assert [(failure.relative_path, failure.category) for failure in result.failures] == [
        ("broken.md", "parse_error")
    ]
    assert "not valid UTF-8" in result.failures[0].message


def test_root_disappearing_after_walk_remains_a_loud_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = make_vault(tmp_path / "vault")
    write_note(vault.root / "first.md", "first")
    write_note(vault.root / "second.md", "second")
    candidates = (vault.root / "first.md", vault.root / "second.md")

    def racing_walk(_root: Path) -> Iterator[Path]:
        yield from candidates

    monkeypatch.setattr("vault_rag.ingest.discovery._walk_files", racing_walk)
    original_read_bytes = Path.read_bytes
    removed = False

    def racing_read_bytes(path: Path) -> bytes:
        nonlocal removed
        if path == candidates[0] and not removed:
            shutil.rmtree(vault.root)
            removed = True
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)

    try:
        result = discover_sources(vault)
    except SecurityError as exc:
        assert removed
        assert "does not exist" in str(exc)
        return

    assert removed
    assert result.sources == ()
    assert result.failures
    pytest.fail("root disappearance returned DiscoveryResult instead of raising SecurityError")


def test_selected_file_oserror_is_a_bounded_failure_with_healthy_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = make_vault(tmp_path / "vault")
    bad = vault.root / "bad.md"
    good = vault.root / "good.md"
    write_note(bad, "bad")
    write_note(good, "good")
    original_read_bytes = Path.read_bytes

    def failing_read_bytes(path: Path) -> bytes:
        if path == bad:
            raise OSError("secret absolute failure: /private/secret")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)

    result = discover_sources(vault)

    assert [source.relative_path for source in result.sources] == ["good.md"]
    assert [
        (failure.relative_path, failure.category, failure.message) for failure in result.failures
    ] == [("bad.md", "parse_error", "could not read source: bad.md")]
    assert "/private/secret" not in result.failures[0].message


def test_unsupported_included_extension_becomes_a_bounded_failure(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault", include=("**/*",))
    write_note(vault.root / "visible.md", "visible")
    (vault.root / "attachment.bin").write_bytes(b"unsupported")

    result = discover_sources(vault)

    assert [source.relative_path for source in result.sources] == ["visible.md"]
    assert [(failure.relative_path, failure.category) for failure in result.failures] == [
        ("attachment.bin", "parse_error")
    ]


def test_discovery_fails_closed_on_a_symlinked_vault_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-vault"
    real_root.mkdir()
    write_note(real_root / "visible.md", "visible")
    linked_root = tmp_path / "linked-vault"
    linked_root.symlink_to(real_root, target_is_directory=True)
    manifest = make_vault(real_root).manifest

    with pytest.raises(SecurityError, match="symlink"):
        discover_sources(ResolvedVault(root=linked_root, manifest=manifest))


def test_lexical_relative_path_fails_closed_outside_the_root(tmp_path: Path) -> None:
    with pytest.raises(SecurityError, match="escaped the vault root"):
        _lexical_relative_path(tmp_path / "vault", tmp_path / "elsewhere" / "note.md")


def test_discovery_does_not_follow_directory_symlinks(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault")
    outside = tmp_path / "outside"
    write_note(outside / "secret.md", "secret")
    write_note(vault.root / "visible.md", "visible")
    (vault.root / "linked-directory").symlink_to(outside, target_is_directory=True)

    assert [source.relative_path for source in discover_sources(vault).sources] == ["visible.md"]


def test_discovery_hashes_exact_utf8_bytes(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault")
    raw = "café\r\nsecond line\n".encode()
    (vault.root / "exact.md").write_bytes(raw)

    source = discover_sources(vault).sources[0]

    assert source.text == raw.decode("utf-8")
    assert source.size_bytes == len(raw)
    assert source.content_hash == f"sha256:{hashlib.sha256(raw).hexdigest()}"


def test_discovery_rejects_case_fold_collision(
    resolved_vault: ResolvedVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_note(resolved_vault.root / "Note.md", "one")
    write_note(resolved_vault.root / "note.md", "two")
    monkeypatch.setattr("vault_rag.ingest.discovery.is_case_insensitive", lambda _: True)
    monkeypatch.setattr(
        "vault_rag.ingest.discovery._walk_files",
        lambda _: iter((resolved_vault.root / "Note.md", resolved_vault.root / "note.md")),
    )
    monkeypatch.setattr(
        "vault_rag.ingest.discovery.secure_relative_path",
        lambda _root, candidate: candidate.name,
    )
    with pytest.raises(SecurityError, match="case-insensitive collision"):
        discover_sources(resolved_vault)
