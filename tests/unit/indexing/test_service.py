import os
import sqlite3
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.config import (  # type: ignore[import-untyped]
    EgressPolicy,
    EmbeddingConfig,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)
from vault_rag.embedding import EmbeddingBatch, EmbeddingFailure  # type: ignore[import-untyped]
from vault_rag.errors import RebuildRequiredError, StorageError  # type: ignore[import-untyped]
from vault_rag.indexing import PARSER_SCHEMA_VERSION, Indexer  # type: ignore[import-untyped]
from vault_rag.ingest.chunker import CHUNKER_SCHEMA_VERSION  # type: ignore[import-untyped]
from vault_rag.storage import (  # type: ignore[import-untyped]
    SCHEMA_VERSION,
    LexicalCandidate,
    LexicalRequest,
    SourceRevision,
    SQLiteStore,
    StorageFilters,
    VectorLoadRequest,
)


class WhitespaceCounter:
    def count(self, text: str) -> int:
        return len(text.split())

    def take_tail(self, text: str, tokens: int) -> str:
        return " ".join(text.split()[-tokens:])


class FakeEmbedding:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.failure: str | None = None
        self.failed_indexes: tuple[int, ...] = ()

    def fail_next(self, message: str) -> None:
        self.failure = message

    def fail_indexes_next(self, *indexes: int) -> None:
        self.failed_indexes = indexes

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls.append(tuple(texts))
        if self.failure is not None:
            message = self.failure
            self.failure = None
            return EmbeddingBatch(
                (),
                tuple(EmbeddingFailure(index, "transport", message) for index in range(len(texts))),
                None,
            )
        failed = set(self.failed_indexes)
        self.failed_indexes = ()
        return EmbeddingBatch(
            tuple(
                np.asarray([float(index + 1), 1.0], dtype=np.float32)
                for index in range(len(texts))
                if index not in failed
            ),
            tuple(
                EmbeddingFailure(index, "transport", "selected failure") for index in sorted(failed)
            ),
            2,
        )


def embedding_config(*, endpoint_class: str = "local", model_env: str = "MODEL") -> EmbeddingConfig:
    base_url = (
        "https://embedding.example.test/v1"
        if endpoint_class == "remote"
        else "http://127.0.0.1:4000/v1"
    )
    return EmbeddingConfig.model_validate(
        {
            "base_url": base_url,
            "model_env": model_env,
            "endpoint_class": endpoint_class,
            "target_min_tokens": 10,
            "target_max_tokens": 30,
            "overlap_tokens": 2,
            "max_input_tokens": 100,
        }
    )


def write_manifest(
    root: Path,
    *,
    include: tuple[str, ...] = ("**/*.md",),
    exclude: tuple[str, ...] = (),
    egress: EgressPolicy = EgressPolicy.REMOTE_ALLOWED,
    fields: tuple[str, ...] = ("ticket", "aliases"),
) -> VaultManifest:
    manifest = VaultManifest.model_validate(
        {
            "schema_version": 1,
            "id": "vault-a",
            "egress_policy": egress.value,
            "include": include,
            "exclude": exclude,
            "metadata": {"frontmatter_fields": fields},
        }
    )
    return manifest


def make_indexer(
    root: Path,
    database: Path,
    embedding: FakeEmbedding,
    *,
    config: EmbeddingConfig | None = None,
    manifest: VaultManifest | None = None,
) -> tuple[Indexer, SQLiteStore]:
    config = config or embedding_config()
    manifest = manifest or write_manifest(root)
    profile = ResolvedProfile(
        name="default",
        vaults=(ResolvedVault(root=root, manifest=manifest),),
        embedding_model="embed-v1",
        api_key_env=None,
        effective_policy=manifest.egress_policy,
        semantic_enabled=not (
            manifest.egress_policy is EgressPolicy.LOCAL_ONLY and config.endpoint_class == "remote"
        ),
        semantic_disabled_reason=None,
    )
    store = SQLiteStore(database)
    return Indexer(profile, config, embedding, store, counter=WhitespaceCounter()), store


def lexical(store: SQLiteStore, text: str) -> tuple[LexicalCandidate, ...]:
    return cast(
        tuple[LexicalCandidate, ...],
        store.lexical_search(LexicalRequest(("vault-a",), f'"{text}"', StorageFilters(), 10)),
    )


def test_scoped_atomic_rebuild_preserves_unrelated_vault_sources(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    root_a = tmp_path / "vault-a"
    root_b = tmp_path / "vault-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.md").write_text("# A\n\nalpha initial", encoding="utf-8")
    (root_b / "b.md").write_text("# B\n\nbeta retained", encoding="utf-8")
    config = embedding_config()
    embedding = FakeEmbedding()
    manifest_a = write_manifest(root_a).model_copy(update={"id": "vault-a"})
    manifest_b = write_manifest(root_b).model_copy(update={"id": "vault-b"})

    def indexer(root: Path, manifest: VaultManifest) -> Indexer:
        profile = ResolvedProfile(
            name=manifest.id,
            vaults=(ResolvedVault(root=root, manifest=manifest),),
            embedding_model="embed-v1",
            api_key_env=None,
            effective_policy=manifest.egress_policy,
            semantic_enabled=True,
            semantic_disabled_reason=None,
        )
        return Indexer(
            profile,
            config,
            embedding,
            SQLiteStore(database),
            counter=WhitespaceCounter(),
        )

    indexer(root_a, manifest_a).run(rebuild=True)
    indexer(root_b, manifest_b).run(rebuild=True)
    before_b = SQLiteStore(database).source_hashes(("vault-b",))

    (root_a / "a.md").write_text("# A\n\nalpha replacement", encoding="utf-8")
    indexer(root_a, manifest_a).run(rebuild=True)

    store = SQLiteStore(database)
    assert store.source_hashes(("vault-b",)) == before_b
    assert store.source_hashes(("vault-a",))
    candidates = store.lexical_search(
        LexicalRequest(("vault-a", "vault-b"), '"beta retained"', StorageFilters(), 10)
    )
    assert [candidate.vault_id for candidate in candidates] == ["vault-b"]


def test_failed_scoped_rebuild_preserves_active_target_and_unrelated_vault(
    tmp_path: Path,
) -> None:
    class ExplodingEmbedding(FakeEmbedding):
        def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
            del texts
            raise RuntimeError("synthetic failure")

    database = tmp_path / "index.sqlite3"
    roots = {"vault-a": tmp_path / "vault-a", "vault-b": tmp_path / "vault-b"}
    for vault_id, root in roots.items():
        root.mkdir()
        (root / "note.md").write_text(f"# {vault_id}\n\noriginal {vault_id}", encoding="utf-8")
    config = embedding_config()

    def indexer(vault_id: str, embedding: FakeEmbedding) -> Indexer:
        manifest = write_manifest(roots[vault_id]).model_copy(update={"id": vault_id})
        profile = ResolvedProfile(
            name=vault_id,
            vaults=(ResolvedVault(root=roots[vault_id], manifest=manifest),),
            embedding_model="embed-v1",
            api_key_env=None,
            effective_policy=manifest.egress_policy,
            semantic_enabled=True,
            semantic_disabled_reason=None,
        )
        return Indexer(
            profile, config, embedding, SQLiteStore(database), counter=WhitespaceCounter()
        )

    embedding = FakeEmbedding()
    indexer("vault-a", embedding).run(rebuild=True)
    indexer("vault-b", embedding).run(rebuild=True)
    store = SQLiteStore(database)
    before = store.source_hashes(("vault-a", "vault-b"))
    (roots["vault-a"] / "note.md").write_text("# vault-a\n\nreplacement", encoding="utf-8")

    with pytest.raises(RuntimeError, match="synthetic"):
        indexer("vault-a", ExplodingEmbedding()).run(rebuild=True)

    assert store.source_hashes(("vault-a", "vault-b")) == before


def test_incompatible_shared_store_rebuilds_fresh_then_later_vault_recovers(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    roots = {"vault-a": tmp_path / "vault-a", "vault-b": tmp_path / "vault-b"}
    for vault_id, root in roots.items():
        root.mkdir()
        (root / "note.md").write_text(f"# {vault_id}\n\ncontent {vault_id}", encoding="utf-8")
    config = embedding_config()
    embedding = FakeEmbedding()

    def indexer(vault_id: str) -> Indexer:
        manifest = write_manifest(roots[vault_id]).model_copy(update={"id": vault_id})
        profile = ResolvedProfile(
            name=vault_id,
            vaults=(ResolvedVault(root=roots[vault_id], manifest=manifest),),
            embedding_model="embed-v1",
            api_key_env=None,
            effective_policy=manifest.egress_policy,
            semantic_enabled=True,
            semantic_disabled_reason=None,
        )
        return Indexer(
            profile, config, embedding, SQLiteStore(database), counter=WhitespaceCounter()
        )

    indexer("vault-a").run(rebuild=True)
    indexer("vault-b").run(rebuild=True)
    with sqlite3.connect(database) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    indexer("vault-a").run(rebuild=True)

    store = SQLiteStore(database)
    assert store.source_hashes(("vault-a",))
    assert store.source_hashes(("vault-b",)) == {}

    indexer("vault-b").run(rebuild=True)
    assert store.source_hashes(("vault-a",))
    assert store.source_hashes(("vault-b",))


def test_unchanged_fully_embedded_scan_makes_zero_embedding_calls(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\nsearchable body", encoding="utf-8")
    embedding = FakeEmbedding()
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", embedding)

    first = indexer.run()
    assert first.ready_chunks == 1
    embedding.calls.clear()
    second = indexer.run()

    assert second.unchanged_sources == first.total_sources == 1
    assert second.embedding_requests == 0
    assert embedding.calls == []
    assert store.snapshot(("vault-a",)).ready_count == 1


def test_changed_failure_activates_lexical_then_unchanged_pending_retries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "example.md"
    note.write_text("# Example\n\nold searchable state", encoding="utf-8")
    embedding = FakeEmbedding()
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", embedding)
    indexer.run()

    note.write_text("# Example\n\nnew searchable state", encoding="utf-8")
    embedding.fail_next("endpoint unavailable")
    failed = indexer.run()
    assert failed.changed_sources == 1
    assert failed.pending_chunks == 1
    assert failed.embedding_failures == 1
    assert lexical(store, "new searchable")
    assert not lexical(store, "old searchable")

    calls_before_retry = len(embedding.calls)
    retried = indexer.run()
    assert retried.unchanged_sources == 1
    assert retried.pending_chunks == 0
    assert retried.ready_chunks == 1
    assert len(embedding.calls) == calls_before_retry + 1


def test_selected_frontmatter_and_section_metadata_reach_lexical_storage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text(
        "---\nticket: DIS-123\naliases: [front-alias]\nignored: secret-value\n---\n"
        "# Example\n\n[[Target Note|section-alias]] #project/tag ordinary body",
        encoding="utf-8",
    )
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", FakeEmbedding())

    indexer.run()

    for term in ("DIS", "front", "section", "project", "Target"):
        assert lexical(store, term), term
    candidate = lexical(store, "ordinary")[0]
    assert candidate.metadata == {
        "aliases": ["front-alias", "section-alias"],
        "tags": ["project/tag"],
        "ticket": "DIS-123",
        "wikilinks": ["Target Note"],
    }


def test_remote_endpoint_prohibited_by_one_local_only_vault_indexes_disabled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\nlexical only", encoding="utf-8")
    embedding = FakeEmbedding()
    manifest = write_manifest(root, egress=EgressPolicy.LOCAL_ONLY)
    indexer, store = make_indexer(
        root,
        tmp_path / "index.sqlite3",
        embedding,
        config=embedding_config(endpoint_class="remote"),
        manifest=manifest,
    )

    report = indexer.run()

    assert embedding.calls == []
    assert report.ready_chunks == report.pending_chunks == report.embedding_requests == 0
    assert lexical(store, "lexical")


def test_policy_change_disables_existing_remote_vectors_without_egress(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\npolicy searchable", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    embedding = FakeEmbedding()
    remote = embedding_config(endpoint_class="remote")
    allowed, store = make_indexer(root, database, embedding, config=remote)
    assert allowed.run().ready_chunks == 1
    embedding.calls.clear()

    local_manifest = write_manifest(root, egress=EgressPolicy.LOCAL_ONLY)
    prohibited, _ = make_indexer(root, database, embedding, config=remote, manifest=local_manifest)
    report = prohibited.run()

    assert embedding.calls == []
    assert report.ready_chunks == report.pending_chunks == 0
    assert store.snapshot(("vault-a",)).observed_fingerprints == ()
    assert lexical(store, "policy")

    embedding.calls.clear()
    relaxed_manifest = write_manifest(root, egress=EgressPolicy.REMOTE_ALLOWED)
    relaxed, _ = make_indexer(root, database, embedding, config=remote, manifest=relaxed_manifest)
    restored = relaxed.run()
    assert restored.ready_chunks == 1
    assert restored.embedding_requests == 1
    assert not any(item.category == "policy_transition" for item in restored.diagnostics)
    assert len(embedding.calls) == 1

    embedding.calls.clear()
    repeated = relaxed.run()
    assert repeated.unchanged_sources == 1
    assert repeated.embedding_requests == 0
    assert embedding.calls == []


def test_relaxing_profile_policy_requeues_unchanged_vault_without_manifest_change(
    tmp_path: Path,
) -> None:
    config = embedding_config(endpoint_class="remote")
    embedding = FakeEmbedding()
    vaults: list[ResolvedVault] = []
    for vault_id, policy in (
        ("vault-a", EgressPolicy.REMOTE_ALLOWED),
        ("vault-b", EgressPolicy.LOCAL_ONLY),
    ):
        root = tmp_path / vault_id
        root.mkdir()
        (root / "note.md").write_text(f"# {vault_id}\n\n{vault_id} searchable", encoding="utf-8")
        manifest = VaultManifest.model_validate(
            {
                "schema_version": 1,
                "id": vault_id,
                "egress_policy": policy.value,
                "include": ["**/*.md"],
            }
        )
        vaults.append(ResolvedVault(root=root, manifest=manifest))
    restricted_profile = ResolvedProfile(
        name="restricted",
        vaults=tuple(vaults),
        embedding_model="embed-v1",
        api_key_env=None,
        effective_policy=EgressPolicy.LOCAL_ONLY,
        semantic_enabled=False,
        semantic_disabled_reason="remote endpoint prohibited by local-only vault",
    )
    store = SQLiteStore(tmp_path / "index.sqlite3")
    restricted = Indexer(restricted_profile, config, embedding, store, counter=WhitespaceCounter())
    restricted_report = restricted.run()
    assert restricted_report.ready_chunks == restricted_report.embedding_requests == 0
    assert embedding.calls == []

    relaxed_profile = restricted_profile.model_copy(
        update={
            "name": "relaxed",
            "vaults": (vaults[0],),
            "effective_policy": EgressPolicy.REMOTE_ALLOWED,
            "semantic_enabled": True,
            "semantic_disabled_reason": None,
        }
    )
    relaxed = Indexer(relaxed_profile, config, embedding, store, counter=WhitespaceCounter())
    report = relaxed.run()
    assert report.unchanged_sources == 1
    assert report.ready_chunks == report.embedding_requests == 1
    assert len(embedding.calls) == 1
    assert any(
        item.vault_id == "vault-a" and item.category == "policy_transition"
        for item in report.diagnostics
    )

    embedding.calls.clear()
    repeated = relaxed.run()
    assert repeated.embedding_requests == 0
    assert embedding.calls == []


def test_manifest_frontmatter_widening_and_narrowing_reindexes_all_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    for name, owner in (("a.md", "alice"), ("b.md", "bob")):
        (root / name).write_text(
            f"---\nticket: T-{name[0].upper()}\nowner: {owner}\n---\n# Note\n\n{name} body",
            encoding="utf-8",
        )
    database = tmp_path / "index.sqlite3"
    embedding = FakeEmbedding()
    initial_manifest = write_manifest(root, fields=("ticket",))
    initial, store = make_indexer(root, database, embedding, manifest=initial_manifest)
    initial.run()
    initial_fingerprint = store.snapshot(("vault-a",)).manifest_fingerprint

    embedding.calls.clear()
    widened_manifest = write_manifest(root, fields=("ticket", "owner"))
    widened, _ = make_indexer(root, database, embedding, manifest=widened_manifest)
    widened_report = widened.run()
    widened_fingerprint = store.snapshot(("vault-a",)).manifest_fingerprint
    assert (widened_report.changed_sources, widened_report.unchanged_sources) == (2, 0)
    assert widened_report.embedding_requests == 2
    assert len(embedding.calls) == 2
    for body, owner in (("a.md", "alice"), ("b.md", "bob")):
        assert lexical(store, body)[0].metadata == {
            "owner": owner,
            "ticket": f"T-{body[0].upper()}",
        }
        assert lexical(store, owner)
    assert widened_fingerprint not in {None, initial_fingerprint}

    embedding.calls.clear()
    narrowed_manifest = write_manifest(root, fields=("owner",))
    narrowed, _ = make_indexer(root, database, embedding, manifest=narrowed_manifest)
    narrowed_report = narrowed.run()
    narrowed_fingerprint = store.snapshot(("vault-a",)).manifest_fingerprint
    assert (narrowed_report.changed_sources, narrowed_report.unchanged_sources) == (2, 0)
    assert narrowed_report.embedding_requests == 2
    assert len(embedding.calls) == 2
    for body, owner in (("a.md", "alice"), ("b.md", "bob")):
        assert lexical(store, body)[0].metadata == {"owner": owner}
    assert not lexical(store, "T-A")
    assert narrowed_fingerprint not in {None, widened_fingerprint}


def test_add_delete_manifest_reconciliation_and_empty_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    projects = root / "projects"
    projects.mkdir(parents=True)
    (root / "root.md").write_text("root searchable", encoding="utf-8")
    (projects / "kept.md").write_text("kept searchable", encoding="utf-8")
    embedding = FakeEmbedding()
    database = tmp_path / "index.sqlite3"
    indexer, store = make_indexer(root, database, embedding)
    first = indexer.run()
    assert (first.total_sources, first.added_sources, first.deleted_sources) == (2, 2, 0)

    narrowed = write_manifest(root, include=("projects/**",))
    narrowed_indexer, _ = make_indexer(root, database, embedding, manifest=narrowed)
    second = narrowed_indexer.run()

    assert (
        second.total_sources,
        second.changed_sources,
        second.unchanged_sources,
        second.deleted_sources,
    ) == (1, 1, 0, 1)
    assert not lexical(store, "root")
    assert lexical(store, "kept")
    assert store.snapshot(("vault-a",)).manifest_fingerprint is not None

    (projects / "kept.md").unlink()
    empty = narrowed_indexer.run()
    assert empty.total_sources == 0
    assert empty.deleted_sources == 1
    assert store.snapshot(("vault-a",)).manifest_fingerprint is not None


def test_parse_failure_on_changed_bytes_suppresses_stale_revision(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "example.md"
    note.write_text("# Example\n\nold searchable", encoding="utf-8")
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", FakeEmbedding())
    indexer.run()

    note.write_text("---\nbroken: [\n---\n# Example\n\nnew bytes", encoding="utf-8")
    report = indexer.run()

    assert report.changed_sources == report.parse_failures == 1
    assert not lexical(store, "old searchable")
    assert store.snapshot(("vault-a",)).source_count == 0
    assert report.diagnostics[0].path == "example.md"


def test_parse_failure_hash_allows_revert_recovery_without_reparsing_same_breakage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "example.md"
    good = "# Example\n\nrestored searchable"
    broken = "---\nbroken: [\n---\n# Example\n\nstill broken"
    note.write_text(good, encoding="utf-8")
    embedding = FakeEmbedding()
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", embedding)
    assert indexer.run().ready_chunks == 1

    note.write_text(broken, encoding="utf-8")
    failed = indexer.run()
    broken_hash = store.source_hashes(("vault-a",))[("vault-a", "example.md")]
    assert failed.changed_sources == failed.parse_failures == 1
    assert not lexical(store, "restored")

    calls_after_failure = len(embedding.calls)
    unchanged_broken = indexer.run()
    assert unchanged_broken.unchanged_sources == 1
    assert unchanged_broken.changed_sources == unchanged_broken.parse_failures == 0
    assert len(embedding.calls) == calls_after_failure
    assert store.source_hashes(("vault-a",))[("vault-a", "example.md")] == broken_hash

    note.write_text(good, encoding="utf-8")
    recovered = indexer.run()
    assert recovered.changed_sources == 1
    assert recovered.unchanged_sources == recovered.parse_failures == 0
    assert recovered.ready_chunks == 1
    assert recovered.embedding_requests == 1
    assert lexical(store, "restored")


def test_global_failure_indexes_do_not_shift_compact_vectors(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text(
        "# First\n\nfirst body\n\n# Second\n\nsecond body", encoding="utf-8"
    )
    embedding = FakeEmbedding()
    embedding.fail_indexes_next(0)
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", embedding)

    report = indexer.run()

    assert report.pending_chunks == report.ready_chunks == 1
    pending_id = store.pending_chunks(("vault-a",))[0].id
    snapshot = store.snapshot(("vault-a",))
    loaded = store.load_vectors(
        VectorLoadRequest(("vault-a",), snapshot.observed_fingerprints[0], StorageFilters(), 10)
    )
    assert len(loaded) == 1
    assert loaded[0].chunk_id != pending_id
    np.testing.assert_allclose(
        loaded[0].vector,
        np.asarray([2.0, 1.0], dtype=np.float32) / np.sqrt(5.0),
    )


def test_multi_vault_manifest_disagreement_is_not_a_rebuild_trigger(
    tmp_path: Path,
) -> None:
    config = embedding_config()
    embedding = FakeEmbedding()
    vaults: list[ResolvedVault] = []
    for vault_id, fields in (("vault-a", ("ticket",)), ("vault-b", ("owner",))):
        root = tmp_path / vault_id
        root.mkdir()
        (root / "note.md").write_text(f"# {vault_id}\n\nsearchable", encoding="utf-8")
        manifest = VaultManifest.model_validate(
            {
                "schema_version": 1,
                "id": vault_id,
                "egress_policy": "remote-allowed",
                "include": ["**/*.md"],
                "metadata": {"frontmatter_fields": fields},
            }
        )
        vaults.append(ResolvedVault(root=root, manifest=manifest))
    profile = ResolvedProfile(
        name="both",
        vaults=tuple(vaults),
        embedding_model="embed-v1",
        api_key_env=None,
        effective_policy=EgressPolicy.REMOTE_ALLOWED,
        semantic_enabled=True,
    )
    store = SQLiteStore(tmp_path / "index.sqlite3")
    indexer = Indexer(profile, config, embedding, store, counter=WhitespaceCounter())
    indexer.run()
    assert store.snapshot(("vault-a", "vault-b")).manifest_fingerprint is None
    embedding.calls.clear()

    second = indexer.run()

    assert second.unchanged_sources == 2
    assert embedding.calls == []

    changed_manifest = VaultManifest.model_validate(
        {
            **vaults[0].manifest.model_dump(mode="python"),
            "metadata": {"frontmatter_fields": ("ticket", "owner")},
        }
    )
    changed_profile = profile.model_copy(
        update={
            "vaults": (
                vaults[0].model_copy(update={"manifest": changed_manifest}),
                vaults[1],
            )
        }
    )
    changed_indexer = Indexer(
        changed_profile, config, embedding, store, counter=WhitespaceCounter()
    )
    changed = changed_indexer.run()
    assert (changed.changed_sources, changed.unchanged_sources) == (1, 1)
    assert len(embedding.calls) == 0


def test_failed_manifest_reconciliation_keeps_old_fingerprint_until_coherent_retry(
    tmp_path: Path,
) -> None:
    class FailOnePromotionStore(SQLiteStore):  # type: ignore[misc]
        fail_key: tuple[str, str] | None = None

        def replace_source(
            self, revision: SourceRevision, *, replacing_path: str | None = None
        ) -> None:
            key = (revision.source.vault_id, revision.source.relative_path)
            if key == self.fail_key:
                raise StorageError("synthetic manifest source failure")
            super().replace_source(revision, replacing_path=replacing_path)

    def manifest(
        vault_id: str,
        *,
        fields: tuple[str, ...],
        include: tuple[str, ...] = ("**/*.md",),
    ) -> VaultManifest:
        return VaultManifest.model_validate(
            {
                "schema_version": 1,
                "id": vault_id,
                "egress_policy": "remote-allowed",
                "include": include,
                "metadata": {"frontmatter_fields": fields},
            }
        )

    def note(path: Path, ticket: str, owner: str, body: str) -> None:
        path.write_text(
            f"---\nticket: {ticket}\nowner: {owner}\n---\n# Note\n\n{body}",
            encoding="utf-8",
        )

    root_a = tmp_path / "vault-a"
    root_b = tmp_path / "vault-b"
    root_a.mkdir()
    root_b.mkdir()
    note(root_a / "a.md", "A-1", "alice", "alpha body")
    note(root_a / "b.md", "B-1", "bob", "bravo body")
    note(root_a / "gone.md", "G-1", "gary", "gone body")
    note(root_b / "d.md", "D-1", "dana", "delta body")
    initial_a = manifest("vault-a", fields=("ticket",))
    initial_b = manifest("vault-b", fields=("ticket",))
    config = embedding_config()
    embedding = FakeEmbedding()
    store = FailOnePromotionStore(tmp_path / "index.sqlite3")

    def make_profile(manifest_a: VaultManifest, manifest_b: VaultManifest) -> ResolvedProfile:
        return ResolvedProfile(
            name="both",
            vaults=(
                ResolvedVault(root=root_a, manifest=manifest_a),
                ResolvedVault(root=root_b, manifest=manifest_b),
            ),
            embedding_model="embed-v1",
            api_key_env=None,
            effective_policy=EgressPolicy.REMOTE_ALLOWED,
            semantic_enabled=True,
        )

    initial = Indexer(
        make_profile(initial_a, initial_b),
        config,
        embedding,
        store,
        counter=WhitespaceCounter(),
    )
    initial.run()
    old_a = store.snapshot(("vault-a",)).manifest_fingerprint
    old_b = store.snapshot(("vault-b",)).manifest_fingerprint

    note(root_a / "a.md", "A-1", "alice", "alpha changed body")
    note(root_a / "c.md", "C-1", "carol", "charlie body")
    widened_a = manifest(
        "vault-a",
        fields=("ticket", "owner"),
        include=("a.md", "b.md", "c.md"),
    )
    widened_b = manifest("vault-b", fields=("ticket", "owner"))
    widened = Indexer(
        make_profile(widened_a, widened_b),
        config,
        embedding,
        store,
        counter=WhitespaceCounter(),
    )
    store.fail_key = ("vault-a", "b.md")
    failed = widened.run()

    assert any(
        item.path == "b.md" and item.category == "storage_error" for item in failed.diagnostics
    )
    assert store.snapshot(("vault-a",)).manifest_fingerprint == old_a
    assert store.snapshot(("vault-b",)).manifest_fingerprint not in {None, old_b}
    assert lexical(store, "alpha")[0].metadata == {
        "owner": "alice",
        "ticket": "A-1",
    }
    assert lexical(store, "charlie")[0].metadata == {
        "owner": "carol",
        "ticket": "C-1",
    }
    assert lexical(store, "bravo")[0].metadata == {"ticket": "B-1"}
    assert not lexical(store, "gone")
    vault_b = store.lexical_search(LexicalRequest(("vault-b",), '"delta"', StorageFilters(), 10))
    assert vault_b[0].metadata == {"owner": "dana", "ticket": "D-1"}

    embedding.calls.clear()
    store.fail_key = None
    retried = widened.run()
    assert (retried.changed_sources, retried.unchanged_sources) == (3, 1)
    # b.md's bytes are unchanged, but adding ``owner`` changes its embedding prefix/hash.
    # Its synthetic replacement failure did not persist that vector, so only b.md is retried.
    assert retried.embedding_requests == 1
    assert len(embedding.calls) == 1
    assert store.snapshot(("vault-a",)).manifest_fingerprint not in {None, old_a}
    for term, metadata in (
        ("alpha", {"owner": "alice", "ticket": "A-1"}),
        ("bravo", {"owner": "bob", "ticket": "B-1"}),
        ("charlie", {"owner": "carol", "ticket": "C-1"}),
    ):
        assert lexical(store, term)[0].metadata == metadata

    embedding.calls.clear()
    repeated = widened.run()
    assert repeated.unchanged_sources == 4
    assert repeated.embedding_requests == 0
    assert embedding.calls == []


def test_large_policy_transition_aggregates_notices_and_preserves_parse_failure(
    tmp_path: Path,
) -> None:
    config = embedding_config(endpoint_class="remote")
    embedding = FakeEmbedding()
    root_a = tmp_path / "vault-a"
    root_b = tmp_path / "vault-b"
    root_a.mkdir()
    root_b.mkdir()
    for index in range(123):
        (root_a / f"a{index:03}.md").write_text(
            f"# Note {index}\n\nsearchable {index}", encoding="utf-8"
        )
    victim = root_a / "zz-existing.md"
    victim.write_text("# Victim\n\nvisible before break", encoding="utf-8")
    (root_b / "local.md").write_text("# Local\n\nrestrictive sibling", encoding="utf-8")
    manifest_a = VaultManifest.model_validate(
        {
            "schema_version": 1,
            "id": "vault-a",
            "egress_policy": "remote-allowed",
            "include": ["**/*.md"],
        }
    )
    manifest_b = VaultManifest.model_validate(
        {
            "schema_version": 1,
            "id": "vault-b",
            "egress_policy": "local-only",
            "include": ["**/*.md"],
        }
    )
    restricted_profile = ResolvedProfile(
        name="restricted",
        vaults=(
            ResolvedVault(root=root_a, manifest=manifest_a),
            ResolvedVault(root=root_b, manifest=manifest_b),
        ),
        embedding_model="embed-v1",
        api_key_env=None,
        effective_policy=EgressPolicy.LOCAL_ONLY,
        semantic_enabled=False,
        semantic_disabled_reason="remote endpoint prohibited by local-only vault",
    )
    store = SQLiteStore(tmp_path / "index.sqlite3")
    restricted = Indexer(restricted_profile, config, embedding, store, counter=WhitespaceCounter())
    restricted.run()
    assert embedding.calls == []

    (root_a / "a122.md").unlink()
    victim.write_text("---\nbroken: [\n---\n# Victim\n\nlate failure", encoding="utf-8")
    relaxed_profile = restricted_profile.model_copy(
        update={
            "name": "relaxed",
            "vaults": (ResolvedVault(root=root_a, manifest=manifest_a),),
            "effective_policy": EgressPolicy.REMOTE_ALLOWED,
            "semantic_enabled": True,
            "semantic_disabled_reason": None,
        }
    )
    relaxed = Indexer(relaxed_profile, config, embedding, store, counter=WhitespaceCounter())
    report = relaxed.run()

    transitions = [item for item in report.diagnostics if item.category == "policy_transition"]
    parse_diagnostics = [item for item in report.diagnostics if item.category == "parse_error"]
    assert report.deleted_sources == report.parse_failures == 1
    assert report.ready_chunks == report.embedding_requests == 122
    assert len(transitions) == 1
    assert transitions[0].vault_id == "vault-a"
    assert transitions[0].message == "semantic embedding re-enabled for 122 chunks"
    assert [item.path for item in parse_diagnostics] == ["zz-existing.md"]
    assert all("a122.md" not in item.message for item in transitions)
    assert len(embedding.calls) == 1
    assert len(embedding.calls[0]) == 122
    assert not lexical(store, "visible before break")

    embedding.calls.clear()
    repeated = relaxed.run()
    assert repeated.embedding_requests == 0
    assert embedding.calls == []


def test_configuration_mismatch_requires_atomic_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\nsearchable", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    embedding = FakeEmbedding()
    first, store = make_indexer(root, database, embedding)
    first.run()
    original_hashes = store.source_hashes(("vault-a",))

    changed_config = embedding_config(model_env="OTHER_MODEL").model_copy(
        update={"tokenizer": "o200k_base"}
    )
    changed, _ = make_indexer(root, database, embedding, config=changed_config)
    with pytest.raises(RebuildRequiredError):
        changed.run()
    assert store.source_hashes(("vault-a",)) == original_hashes

    rebuilt = changed.run(rebuild=True)
    assert rebuilt.ready_chunks == 1
    reopened = SQLiteStore(database)
    reopened.initialize()
    assert reopened.snapshot(("vault-a",)).embedding_config_fingerprint is not None
    assert not list(tmp_path.glob(".index.sqlite3.*.rebuild*"))


def test_failed_rebuild_preserves_old_active_index_and_cleans_temp(tmp_path: Path) -> None:
    class ExplodingEmbedding(FakeEmbedding):
        def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
            del texts
            raise RuntimeError("synthetic failure")

    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\nold searchable", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    first, store = make_indexer(root, database, FakeEmbedding())
    first.run()
    (root / "example.md").write_text("# Example\n\nreplacement", encoding="utf-8")
    failing, _ = make_indexer(root, database, ExplodingEmbedding())

    with pytest.raises(RuntimeError, match="synthetic"):
        failing.run(rebuild=True)

    assert lexical(store, "old searchable")
    assert not lexical(store, "replacement")
    assert not list(tmp_path.glob(".index.sqlite3.*.rebuild*"))


def test_case_only_rename_reports_delete_add_and_activates_new_path(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    (root / "Notes").mkdir(parents=True)
    old = root / "Notes" / "A.md"
    old.write_text("case searchable", encoding="utf-8")
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", FakeEmbedding())
    indexer.run()
    new = root / "Notes" / "a.md"
    old.rename(new)

    report = indexer.run()

    assert (report.added_sources, report.deleted_sources) == (1, 1)
    assert lexical(store, "case")[0].relative_path == "Notes/a.md"


def test_one_source_transaction_failure_does_not_rollback_successful_sibling(
    tmp_path: Path,
) -> None:
    class FailOneStore(SQLiteStore):  # type: ignore[misc]
        def replace_source(
            self, revision: SourceRevision, *, replacing_path: str | None = None
        ) -> None:
            if revision.source.relative_path == "bad.md":
                raise StorageError("synthetic source transaction failure")
            super().replace_source(revision, replacing_path=replacing_path)

    root = tmp_path / "vault"
    root.mkdir()
    (root / "bad.md").write_text("bad searchable", encoding="utf-8")
    (root / "good.md").write_text("good searchable", encoding="utf-8")
    config = embedding_config()
    manifest = write_manifest(root)
    profile = ResolvedProfile(
        name="default",
        vaults=(ResolvedVault(root=root, manifest=manifest),),
        embedding_model="embed-v1",
        api_key_env=None,
        effective_policy=EgressPolicy.REMOTE_ALLOWED,
        semantic_enabled=True,
    )
    store = FailOneStore(tmp_path / "index.sqlite3")
    indexer = Indexer(profile, config, FakeEmbedding(), store, counter=WhitespaceCounter())

    report = indexer.run()

    assert (report.total_sources, report.added_sources) == (2, 2)
    assert lexical(store, "good")
    assert not lexical(store, "bad")
    assert store.snapshot(("vault-a",)).indexed_at is None
    assert any(
        item.path == "bad.md" and item.category == "storage_error" for item in report.diagnostics
    )
    assert report.blocking_failures == 1


def test_rebuild_leaves_the_replacement_database_user_only(tmp_path: Path) -> None:
    """The rebuild must not publish chunk bodies under a permissive umask.

    The assertion runs directly against ``Indexer.run(rebuild=True)`` with no
    intervening ``AppFactory.store()`` call, because that helper re-chmods the
    active database on every other code path and would mask the defect.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\nsearchable body", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    indexer, _store = make_indexer(root, database, FakeEmbedding())

    previous_umask = os.umask(0o022)
    try:
        report = indexer.run(rebuild=True)
    finally:
        os.umask(previous_umask)

    assert report.ready_chunks == 1
    assert stat.S_IMODE(database.stat().st_mode) == 0o600


def test_line_index_fix_is_gated_by_new_schema_versions() -> None:
    """Indexes built by the shifted line index must be rebuild-gated.

    Both fingerprints are output-affecting for sources containing a
    non-CommonMark separator, so neither version may silently revert.
    """
    assert PARSER_SCHEMA_VERSION >= 4
    assert CHUNKER_SCHEMA_VERSION >= 3


def test_excluded_symlink_does_not_block_reconciliation(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    (root / "templates").mkdir(parents=True)
    (root / "example.md").write_text("# Example\n\nsearchable body", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n\nlinked body", encoding="utf-8")
    (root / "templates" / "linked.md").symlink_to(outside)
    manifest = write_manifest(root, exclude=("templates/**",))
    indexer, store = make_indexer(
        root, tmp_path / "index.sqlite3", FakeEmbedding(), manifest=manifest
    )

    report = indexer.run()

    assert report.total_sources == 1
    assert report.ready_chunks == 1
    assert report.diagnostics == ()
    assert lexical(store, "searchable body")
    assert not lexical(store, "linked body")


def test_undecodable_file_diagnoses_and_stops_being_searchable(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\nsearchable body", encoding="utf-8")
    (root / "other.md").write_text("# Other\n\nsibling body", encoding="utf-8")
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", FakeEmbedding())
    indexer.run()

    (root / "example.md").write_bytes(b"# Example\n\xff\xfe\n")
    report = indexer.run()

    assert report.parse_failures == 1
    assert report.deleted_sources == 1
    assert [(item.path, item.category) for item in report.diagnostics] == [
        ("example.md", "parse_error")
    ]
    assert not lexical(store, "searchable body")
    assert lexical(store, "sibling body")


def test_source_replaced_by_a_symlink_is_diagnosed_and_removed(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "example.md").write_text("# Example\n\nsearchable body", encoding="utf-8")
    (root / "other.md").write_text("# Other\n\nsibling body", encoding="utf-8")
    indexer, store = make_indexer(root, tmp_path / "index.sqlite3", FakeEmbedding())
    indexer.run()

    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n\nlinked body", encoding="utf-8")
    (root / "example.md").unlink()
    (root / "example.md").symlink_to(outside)
    report = indexer.run()

    assert report.parse_failures == 0
    assert report.deleted_sources == 1
    assert [(item.path, item.category) for item in report.diagnostics] == [
        ("example.md", "security_error")
    ]
    assert report.blocking_failures == 1
    assert not lexical(store, "searchable body")
    assert not lexical(store, "linked body")
    assert lexical(store, "sibling body")


def test_blocking_failures_remain_accurate_after_diagnostic_truncation(tmp_path: Path) -> None:
    """Deriving completion from retained diagnostics hides later discovery failures."""
    root = tmp_path / "vault"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n\nlinked body", encoding="utf-8")
    for number in range(101):
        (root / f"{number:03d}.md").symlink_to(outside)
    indexer, _store = make_indexer(root, tmp_path / "index.sqlite3", FakeEmbedding())

    report = indexer.run()

    assert len(report.diagnostics) == 100
    assert report.blocking_failures == 101
