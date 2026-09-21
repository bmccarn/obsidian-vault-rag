import hashlib
import os
import unicodedata
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.config import (  # type: ignore[import-untyped]
    EgressPolicy,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)
from vault_rag.domain import JsonValue, LineRange, SourceKind  # type: ignore[import-untyped]
from vault_rag.errors import (  # type: ignore[import-untyped]
    SecurityError,
    SemanticUnavailableError,
    StaleSourceError,
)
from vault_rag.ingest.chunker import ChunkRecord  # type: ignore[import-untyped]
from vault_rag.ingest.models import DiscoveredSource  # type: ignore[import-untyped]
from vault_rag.retrieval import (  # type: ignore[import-untyped]
    ReadRequest,
    RetrievalService,
    SearchFilters,
    SearchMode,
    SearchRequest,
)
from vault_rag.storage import (  # type: ignore[import-untyped]
    SourceRevision,
    SQLiteStore,
    StorageFilters,
    VaultFingerprint,
    VectorRecord,
)


class FakeEmbedding:
    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.failure = False
        self.calls: list[tuple[str, int]] = []

    def fail_queries(self) -> None:
        self.failure = True

    def embed_query(self, text: str, expected_dimensions: int) -> np.ndarray:
        self.calls.append((text, expected_dimensions))
        if self.failure:
            raise SemanticUnavailableError(
                "provider secret-token failed", details={"secret": "secret-token"}
            )
        if self.vector.size != expected_dimensions:
            raise SemanticUnavailableError("dimension mismatch")
        return self.vector


def manifest(vault_id: str, *, fields: tuple[str, ...] = ("ticket", "owner")) -> VaultManifest:
    return VaultManifest.model_validate(
        {
            "schema_version": 1,
            "id": vault_id,
            "egress_policy": "remote-allowed",
            "include": ["**/*.md"],
            "metadata": {"frontmatter_fields": fields},
        }
    )


def profile(roots: Mapping[str, Path], *, semantic_enabled: bool = True) -> ResolvedProfile:
    vaults = tuple(
        ResolvedVault(root=root, manifest=manifest(vault_id)) for vault_id, root in roots.items()
    )
    return ResolvedProfile(
        name="test",
        vaults=vaults,
        embedding_model="embed-v1",
        api_key_env=None,
        effective_policy=EgressPolicy.REMOTE_ALLOWED,
        semantic_enabled=semantic_enabled,
        semantic_disabled_reason=None if semantic_enabled else "policy disabled",
    )


def add_source(
    store: SQLiteStore,
    root: Path,
    *,
    vault_id: str,
    path: str,
    text: str,
    chunk_id: str,
    body: str | None = None,
    heading: tuple[str, ...] = ("Heading",),
    lines: LineRange | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
    vector: tuple[float, ...] | None = (1.0, 0.0),
    observed_fingerprint: str = "observed-v1",
    title: str | None = None,
) -> None:
    source_path = root / path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(text.encode("utf-8"))
    raw = source_path.read_bytes()
    source = DiscoveredSource(
        vault_id=vault_id,
        root=root,
        relative_path=path,
        folded_path=path.casefold(),
        kind=SourceKind.MARKDOWN,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        size_bytes=len(raw),
        mtime_ns=source_path.stat().st_mtime_ns,
    )
    source_lines = text.splitlines()
    chunk_lines = lines or LineRange(1, max(1, len(source_lines)))
    chunk_body = text if body is None else body
    chunk = ChunkRecord(
        id=chunk_id,
        vault_id=vault_id,
        relative_path=path,
        ordinal=0,
        title=Path(path).stem if title is None else title,
        heading=heading,
        lines=chunk_lines,
        text=chunk_body,
        embedding_text=chunk_body,
        token_count=len(chunk_body.split()),
        metadata=dict(metadata or {}),
        content_hash=f"sha256:{hashlib.sha256(chunk_body.encode()).hexdigest()}",
    )
    vectors: tuple[VectorRecord, ...] = ()
    if vector is not None:
        vectors = (
            VectorRecord(
                chunk_id=chunk_id,
                dimensions=len(vector),
                config_fingerprint="config-v1",
                observed_fingerprint=observed_fingerprint,
                vector=np.asarray(vector, dtype=np.float32),
            ),
        )
    store.replace_source(
        SourceRevision(
            source=source,
            chunks=(chunk,),
            vectors=vectors,
            pending=(),
            manifest_fingerprint="manifest-v1",
            parser_fingerprint="parser-v1",
            chunker_fingerprint="chunker-v1",
            embedding_config_fingerprint="config-v1",
        )
    )


def add_chunked_source(
    store: SQLiteStore,
    root: Path,
    *,
    path: str,
    text: str,
    chunks: tuple[tuple[str, tuple[str, ...], LineRange], ...],
) -> None:
    source_path = root / path
    source_path.write_bytes(text.encode("utf-8"))
    raw = source_path.read_bytes()
    source = DiscoveredSource(
        "vault-a",
        root,
        path,
        path.casefold(),
        SourceKind.MARKDOWN,
        text,
        f"sha256:{hashlib.sha256(raw).hexdigest()}",
        len(raw),
        source_path.stat().st_mtime_ns,
    )
    records = tuple(
        ChunkRecord(
            chunk_id,
            "vault-a",
            path,
            ordinal,
            Path(path).stem,
            heading,
            lines,
            "".join(text.splitlines(keepends=True)[lines.start - 1 : lines.end]),
            chunk_id,
            1,
            {},
            f"sha256:{chunk_id}",
        )
        for ordinal, (chunk_id, heading, lines) in enumerate(chunks)
    )
    store.replace_source(
        SourceRevision(
            source,
            records,
            (),
            (),
            "manifest-v1",
            "parser-v1",
            "chunker-v1",
            "config-v1",
        )
    )


class ConcurrentInsertStore(SQLiteStore):  # type: ignore[misc]
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.insert_during_load: Callable[[], None] | None = None
        self.inserted = False

    def load_vectors_complete(
        self,
        vault_ids: tuple[str, ...],
        observed_fingerprint: str,
        filters: StorageFilters,
    ) -> tuple[VectorRecord, ...]:
        if self.insert_during_load is not None:
            callback = self.insert_during_load
            self.insert_during_load = None
            callback()
            self.inserted = True
        return cast(
            tuple[VectorRecord, ...],
            super().load_vectors_complete(vault_ids, observed_fingerprint, filters),
        )


class TransactionTrackingStore(SQLiteStore):  # type: ignore[misc]
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.in_consistent_read = False

    @contextmanager
    def consistent_read(self) -> Generator["TransactionTrackingStore"]:
        self.in_consistent_read = True
        try:
            yield self
        finally:
            self.in_consistent_read = False


@dataclass
class RetrievalFixture:
    root: Path
    store: SQLiteStore
    embedding: FakeEmbedding
    service: RetrievalService


@pytest.fixture
def retrieval_fixture(tmp_path: Path) -> RetrievalFixture:
    root = tmp_path / "vault-a"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="prs/dzte-infra-298.md",
        text="# PR\n\ntracked exact state\n",
        body="tracked exact state",
        chunk_id="exact",
        lines=LineRange(3, 3),
        metadata={"aliases": ["dzte-infra #298"], "ticket": "DIS-298"},
        vector=(0.0, 1.0),
    )
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="notes/semantic.md",
        text="# Semantic\n\nexact lexical phrase and state\n",
        body="exact lexical phrase and state",
        chunk_id="semantic",
        lines=LineRange(3, 3),
        vector=(1.0, 0.0),
    )
    store.delete_sources(
        (),
        fingerprints=(
            VaultFingerprint("vault-a", "manifest-v1", "parser-v1", "chunker-v1", "config-v1"),
        ),
    )
    embedding = FakeEmbedding()
    service = RetrievalService(profile({"vault-a": root}), store, embedding, "observed-v1")
    return RetrievalFixture(root, store, embedding, service)


def test_lexical_mode_skips_dense_retrieval(retrieval_fixture: RetrievalFixture) -> None:
    response = retrieval_fixture.service.search(
        SearchRequest(query="exact lexical phrase", limit=5, mode=SearchMode.LEXICAL)
    )

    assert retrieval_fixture.embedding.calls == []
    assert response.degraded.semantic_search is False
    assert response.hits
    assert all(hit.scores.dense_rank is None for hit in response.hits)


def test_dense_mode_excludes_lexical_and_exact_identifier_tiers(
    retrieval_fixture: RetrievalFixture,
) -> None:
    response = retrieval_fixture.service.search(
        SearchRequest(query="dzte-infra #298", limit=5, mode=SearchMode.DENSE)
    )

    assert retrieval_fixture.embedding.calls == [("dzte-infra #298", 2)]
    assert response.hits
    assert all(hit.scores.lexical_rank is None for hit in response.hits)
    assert all(hit.scores.exact_identifier is False for hit in response.hits)


def test_semantic_policy_degradation_applies_only_to_semantic_modes(
    retrieval_fixture: RetrievalFixture,
) -> None:
    service = RetrievalService(
        profile({"vault-a": retrieval_fixture.root}, semantic_enabled=False),
        retrieval_fixture.store,
        retrieval_fixture.embedding,
        "observed-v1",
    )

    lexical = service.search(SearchRequest("project state", mode=SearchMode.LEXICAL))
    dense = service.search(SearchRequest("project state", mode=SearchMode.DENSE))
    hybrid = service.search(SearchRequest("project state", mode=SearchMode.HYBRID))

    assert lexical.degraded.semantic_search is False
    assert dense.degraded.reason == "semantic_disabled_by_policy"
    assert hybrid.degraded.reason == "semantic_disabled_by_policy"


def test_exact_identifier_tier_beats_higher_semantic_score(
    retrieval_fixture: RetrievalFixture,
) -> None:
    response = retrieval_fixture.service.search(SearchRequest(query="dzte-infra #298", limit=5))
    assert response.hits[0].ref.path == "prs/dzte-infra-298.md"
    assert response.hits[0].scores.exact_identifier is True


@pytest.mark.parametrize(
    ("query", "target_path", "metadata", "title"),
    [
        ("DIS-400", "tickets/real.md", {"ticket": "DIS-400"}, None),
        ("repo-core #77", "prs/real.md", {"aliases": ["repo-core #77"]}, None),
        ("commit:abcdef1", "commits/real.md", {"ticket": "commit:abcdef1"}, None),
        ("projects/exact-note.md", "projects/exact-note.md", {}, None),
        ("DIS-500", "titles/real.md", {}, "DIS-500"),
    ],
)
def test_exact_scan_promotes_matches_outside_lexical_and_dense_windows(
    tmp_path: Path,
    query: str,
    target_path: str,
    metadata: Mapping[str, JsonValue],
    title: str | None,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    for index in range(45):
        add_source(
            store,
            root,
            vault_id="vault-a",
            path=f"noise/{index:02}.md",
            text=(query + " ") * 12,
            chunk_id=f"noise-{index:02}",
            vector=(1.0, 0.0),
        )
    add_source(
        store,
        root,
        vault_id="vault-a",
        path=target_path,
        text="# Authoritative\n\n" + ("long padding " * 500),
        chunk_id="authoritative",
        metadata=metadata,
        vector=(0.0, 1.0),
        title=title,
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    hit = service.search(SearchRequest(query, limit=1)).hits[0]

    assert hit.chunk_id == "authoritative"
    assert hit.scores.exact_identifier
    assert hit.scores.lexical_rank is None
    assert hit.scores.dense_rank is None


def test_exact_scan_applies_profile_and_structured_filters_before_matching(
    tmp_path: Path,
) -> None:
    roots = {"vault-a": tmp_path / "a", "vault-b": tmp_path / "b"}
    for root in roots.values():
        root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    for index in range(45):
        add_source(
            store,
            roots["vault-a"],
            vault_id="vault-a",
            path=f"allowed/noise-{index:02}.md",
            text="DIS-777 " * 12,
            chunk_id=f"noise-{index:02}",
            metadata={"owner": "active"},
        )
    add_source(
        store,
        roots["vault-a"],
        vault_id="vault-a",
        path="allowed/real.md",
        text="long " * 500,
        chunk_id="allowed",
        metadata={"ticket": "DIS-777", "owner": "active"},
        vector=(0.0, 1.0),
    )
    add_source(
        store,
        roots["vault-a"],
        vault_id="vault-a",
        path="excluded/real.md",
        text="short",
        chunk_id="excluded",
        metadata={"ticket": "DIS-777", "owner": "active"},
    )
    add_source(
        store,
        roots["vault-b"],
        vault_id="vault-b",
        path="allowed/foreign.md",
        text="short",
        chunk_id="foreign",
        metadata={"ticket": "DIS-777", "owner": "active"},
    )
    service = RetrievalService(
        profile({"vault-a": roots["vault-a"]}), store, FakeEmbedding(), "observed-v1"
    )

    response = service.search(
        SearchRequest(
            "DIS-777",
            SearchFilters(path_prefix="allowed", frontmatter={"owner": "active"}),
            limit=5,
        )
    )

    assert response.hits[0].chunk_id == "allowed"
    assert response.hits[0].scores.exact_identifier
    assert response.hits[0].scores.lexical_rank is None
    assert response.hits[0].scores.dense_rank is None
    assert {hit.chunk_id for hit in response.hits}.isdisjoint({"excluded", "foreign"})


def test_nfd_path_with_punctuation_receives_exact_tier_outside_windows(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    for index in range(45):
        add_source(
            store,
            root,
            vault_id="vault-a",
            path=f"noise/{index:02}.md",
            text="notes café md " * 12,
            chunk_id=f"noise-{index:02}",
        )
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="notes/café.md",
        text="long " * 500,
        chunk_id="normalized-path",
        vector=(0.0, 1.0),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")
    query = unicodedata.normalize("NFD", "notes/café.md") + "."

    hit = service.search(SearchRequest(query, limit=1)).hits[0]

    assert hit.chunk_id == "normalized-path"
    assert hit.scores.exact_identifier


def test_bare_number_never_creates_exact_tier(retrieval_fixture: RetrievalFixture) -> None:
    response = retrieval_fixture.service.search(SearchRequest(query="298", limit=5))
    assert all(not hit.scores.exact_identifier for hit in response.hits)


def test_exact_path_and_typed_frontmatter_values_create_priority_tier(
    retrieval_fixture: RetrievalFixture,
) -> None:
    path_hit = retrieval_fixture.service.search(
        SearchRequest("prs/dzte-infra-298.md", limit=5)
    ).hits[0]
    ticket_hit = retrieval_fixture.service.search(SearchRequest("DIS-298", limit=5)).hits[0]

    assert path_hit.ref.path == "prs/dzte-infra-298.md"
    assert path_hit.scores.exact_identifier
    assert ticket_hit.ref.path == "prs/dzte-infra-298.md"
    assert ticket_hit.scores.exact_identifier


def test_query_embedding_failure_returns_lexical_degraded_without_secret(
    retrieval_fixture: RetrievalFixture,
) -> None:
    retrieval_fixture.embedding.fail_queries()
    response = retrieval_fixture.service.search(
        SearchRequest(query="exact lexical phrase", limit=5)
    )
    assert response.hits
    assert response.degraded.semantic_search is True
    assert response.degraded.reason is not None
    assert "secret-token" not in response.degraded.reason


def test_invalid_query_vector_degrades_to_lexical(retrieval_fixture: RetrievalFixture) -> None:
    retrieval_fixture.embedding.vector = np.asarray([np.nan, 0.0], dtype=np.float32)

    response = retrieval_fixture.service.search(SearchRequest("state"))

    assert response.hits
    assert response.degraded.semantic_search
    assert response.degraded.reason == "query_embedding_failed"


def test_lexical_empty_query_can_still_use_dense_and_ties_are_stable(
    retrieval_fixture: RetrievalFixture,
) -> None:
    response = retrieval_fixture.service.search(SearchRequest(query="!!!", limit=5))
    assert [hit.ref.path for hit in response.hits] == [
        "notes/semantic.md",
        "prs/dzte-infra-298.md",
    ]
    assert all(hit.scores.lexical_rank is None for hit in response.hits)


def test_no_vectors_and_empty_query_have_bounded_degradation(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="note.md",
        text="lexical only",
        chunk_id="lexical",
        vector=None,
    )
    embedding = FakeEmbedding()
    service = RetrievalService(profile({"vault-a": root}), store, embedding, "observed-v1")

    lexical = service.search(SearchRequest("lexical"))
    empty = service.search(SearchRequest(""))

    assert lexical.hits and lexical.degraded.semantic_search
    assert lexical.degraded.reason == "no_compatible_vectors"
    assert embedding.calls == []
    assert empty.hits == ()


def test_profile_and_filters_apply_before_both_rank_paths_and_limits(tmp_path: Path) -> None:
    roots = {"vault-a": tmp_path / "a", "vault-b": tmp_path / "b"}
    for root in roots.values():
        root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    for index in range(24):
        add_source(
            store,
            roots["vault-a"],
            vault_id="vault-a",
            path=f"other/dominant-{index}.md",
            text="shared phrase " * 8,
            chunk_id=f"foreign-{index}",
            vector=(1.0, 0.0),
        )
    add_source(
        store,
        roots["vault-b"],
        vault_id="vault-b",
        path="projects/target.md",
        text="shared phrase",
        chunk_id="target",
        metadata={"owner": None},
        vector=(0.0, 1.0),
    )
    service = RetrievalService(
        profile({"vault-b": roots["vault-b"]}), store, FakeEmbedding(), "observed-v1"
    )

    response = service.search(
        SearchRequest(
            "shared phrase",
            SearchFilters(path_prefix="projects", frontmatter={"owner": None}),
            limit=1,
        )
    )

    assert len(response.hits) == 1
    assert response.hits[0].ref.vault_id == "vault-b"
    assert response.hits[0].ref.path == "projects/target.md"
    assert response.hits[0].scores.lexical_rank == 1
    assert response.hits[0].scores.dense_rank == 1


def test_vector_dimensions_are_scoped_by_observed_fingerprint_and_filters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="projects/target.md",
        text="target project",
        chunk_id="target",
        vector=(1.0, 0.0),
    )
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="archive/other.md",
        text="other archive",
        chunk_id="other",
        vector=(1.0, 0.0, 0.0),
        observed_fingerprint="stale-observed-v2",
    )
    embedding = FakeEmbedding()
    service = RetrievalService(profile({"vault-a": root}), store, embedding, "observed-v1")

    response = service.search(
        SearchRequest("target", SearchFilters(path_prefix="projects"), limit=1)
    )

    assert response.degraded.semantic_search is False
    assert response.hits[0].chunk_id == "target"
    assert response.hits[0].scores.dense_rank == 1
    assert embedding.calls == [("target", 2)]


def test_query_embedding_does_not_hold_consistent_read_transaction(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = TransactionTrackingStore(tmp_path / "index.sqlite3")
    store.initialize()
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="target.md",
        text="target",
        chunk_id="target",
    )

    class OutsideTransactionEmbedding(FakeEmbedding):
        def embed_query(self, text: str, expected_dimensions: int) -> np.ndarray:
            assert store.in_consistent_read is False
            return super().embed_query(text, expected_dimensions)

    service = RetrievalService(
        profile({"vault-a": root}),
        store,
        OutsideTransactionEmbedding(),
        "observed-v1",
    )

    response = service.search(SearchRequest("target", mode=SearchMode.HYBRID))

    assert response.degraded.semantic_search is False
    assert response.hits[0].chunk_id == "target"


def test_equal_dense_scores_use_stable_source_tie_breakers(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    for path, chunk_id in (("zeta.md", "z"), ("alpha.md", "a")):
        add_source(
            store,
            root,
            vault_id="vault-a",
            path=path,
            text="ordinary",
            chunk_id=chunk_id,
            vector=(1.0, 0.0),
        )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    response = service.search(SearchRequest("!!!", limit=2))

    assert [hit.ref.path for hit in response.hits] == ["alpha.md", "zeta.md"]


def test_scope_complete_dense_load_includes_tail_best_after_concurrent_insert(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = ConcurrentInsertStore(tmp_path / "index.sqlite3")
    store.initialize()
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="zz0.md",
        text="ordinary",
        chunk_id="old-worse",
        vector=(0.0, 1.0),
    )
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="zzz-best.md",
        text="ordinary",
        chunk_id="true-tail-best",
        vector=(1.0, 0.0),
    )

    def insert_early_rows() -> None:
        writer = SQLiteStore(store.path)
        for index in range(3):
            add_source(
                writer,
                root,
                vault_id="vault-a",
                path=f"aa{index}.md",
                text="ordinary",
                chunk_id=f"inserted-{index}",
                vector=(1.0, 1.0),
            )

    store.insert_during_load = insert_early_rows
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    hit = service.search(SearchRequest("!!!", limit=1)).hits[0]

    assert store.inserted
    assert hit.chunk_id == "true-tail-best"
    assert hit.scores.dense_score == pytest.approx(1.0)


def test_dense_load_does_not_truncate_profile_matrix(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    for index in range(25):
        add_source(
            store,
            root,
            vault_id="vault-a",
            path=f"notes/{index:02}.md",
            text="ordinary",
            chunk_id=f"chunk-{index:02}",
            vector=(0.0, 1.0) if index < 24 else (1.0, 0.0),
        )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    response = service.search(SearchRequest("!!!", limit=1))

    assert response.hits[0].ref.path == "notes/24.md"


def test_duplicate_candidates_are_deduplicated_and_rrf_is_one_based(
    retrieval_fixture: RetrievalFixture,
) -> None:
    response = retrieval_fixture.service.search(SearchRequest("exact lexical phrase", limit=5))
    assert len({hit.chunk_id for hit in response.hits}) == len(response.hits)
    hit = next(hit for hit in response.hits if hit.chunk_id == "semantic")
    assert hit.scores.lexical_rank == 1
    assert hit.scores.dense_rank == 1
    assert hit.scores.fused_score == pytest.approx(2 / 61)


def test_search_excerpt_has_explicit_continuation_and_valid_citation(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "é" * 2000
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="long.md",
        text=text,
        chunk_id="long",
        vector=(1.0, 0.0),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    hit = service.search(SearchRequest("!!!", limit=1)).hits[0]

    assert len(hit.text) == 1800
    assert hit.continuation is not None
    assert hit.continuation.remaining_characters == 200
    assert hit.ref.lines == LineRange(1, 1)
    assert hit.ref.citation.endswith("#L1-L1")


def test_search_filters_copy_values_and_reject_unsafe_or_out_of_profile_values(
    retrieval_fixture: RetrievalFixture,
) -> None:
    aliases: list[JsonValue] = ["dzte-infra #298"]
    frontmatter: dict[str, JsonValue] = {"ticket": aliases}
    filters = SearchFilters(vault_ids=["vault-a"], frontmatter=frontmatter)
    aliases.append("changed")
    frontmatter["ticket"] = "changed"
    assert filters.frontmatter["ticket"] == ("dzte-infra #298",)

    with pytest.raises(ValueError):
        SearchFilters(path_prefix="../escape")
    with pytest.raises(ValueError):
        retrieval_fixture.service.search(
            SearchRequest("state", SearchFilters(vault_ids=["vault-b"]))
        )
    with pytest.raises(ValueError):
        retrieval_fixture.service.search(
            SearchRequest("state", SearchFilters(frontmatter={"not-configured": "x"}))
        )


def test_adversarial_fts_text_cannot_change_sql(retrieval_fixture: RetrievalFixture) -> None:
    response = retrieval_fixture.service.search(SearchRequest('" OR path:*; DROP TABLE chunks --'))
    assert isinstance(response.hits, tuple)
    assert retrieval_fixture.service.search(SearchRequest("state")).hits


def test_single_vault_read_infers_vault_and_hash_verifies(
    retrieval_fixture: RetrievalFixture,
) -> None:
    response = retrieval_fixture.service.read(
        ReadRequest(path="prs/dzte-infra-298.md", lines=LineRange(3, 3))
    )
    assert response.text == "tracked exact state\n"
    assert response.ref.vault_id == "vault-a"
    assert response.ref.lines == LineRange(3, 3)


def test_read_rejects_source_changed_after_index(retrieval_fixture: RetrievalFixture) -> None:
    path = retrieval_fixture.root / "prs/dzte-infra-298.md"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(StaleSourceError):
        retrieval_fixture.service.read(ReadRequest(path="prs/dzte-infra-298.md"))


@pytest.mark.parametrize(
    ("expected_source_hash", "error"),
    [
        (None, None),
        ("sha256:4755db938696de059a6beeff63961a2c738d436281161ce6aa8f7e095e9401b1", None),
        ("sha256:" + "A" * 64, ValueError),
        ("sha256:" + "0" * 64, StaleSourceError),
    ],
)
def test_read_expected_source_hash_is_validated_and_fail_closed(
    retrieval_fixture: RetrievalFixture,
    expected_source_hash: str | None,
    error: type[Exception] | None,
) -> None:
    """A caller hash must be canonical and agree before any source text is returned."""
    if error is ValueError:
        with pytest.raises(ValueError):
            ReadRequest(
                path="prs/dzte-infra-298.md",
                expected_source_hash=expected_source_hash,
            )
        return
    request = ReadRequest(
        path="prs/dzte-infra-298.md",
        expected_source_hash=expected_source_hash,
    )

    if error is StaleSourceError:
        with pytest.raises(StaleSourceError):
            retrieval_fixture.service.read(request)
        return

    response = retrieval_fixture.service.read(request)

    assert response.text == "# PR\n\ntracked exact state\n"


def test_read_request_rejects_non_string_expected_source_hash() -> None:
    with pytest.raises(ValueError, match="expected_source_hash"):
        ReadRequest(path="notes/read.md", expected_source_hash=123)  # type: ignore[arg-type]


def test_multi_vault_read_requires_vault_and_stays_scoped(tmp_path: Path) -> None:
    roots = {"vault-a": tmp_path / "a", "vault-b": tmp_path / "b"}
    for root in roots.values():
        root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    for vault_id, root in roots.items():
        add_source(
            store,
            root,
            vault_id=vault_id,
            path="same.md",
            text=f"{vault_id}\n",
            chunk_id=vault_id,
        )
    service = RetrievalService(profile(roots), store, FakeEmbedding(), "observed-v1")

    with pytest.raises(ValueError, match="vault_id is required"):
        service.read(ReadRequest(path="same.md"))
    assert service.read(ReadRequest(vault_id="vault-b", path="same.md")).text == "vault-b\n"
    with pytest.raises(ValueError):
        service.read(ReadRequest(vault_id="foreign", path="same.md"))


def test_read_rejects_traversal_symlinks_and_mutually_exclusive_selectors(
    retrieval_fixture: RetrievalFixture, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        ReadRequest(
            path="prs/dzte-infra-298.md",
            heading="PR",
            lines=LineRange(1, 1),
        )
    with pytest.raises(SecurityError):
        retrieval_fixture.service.read(ReadRequest(path="../outside.md"))

    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = retrieval_fixture.root / "linked.md"
    os.symlink(outside, link)
    with pytest.raises(SecurityError):
        retrieval_fixture.service.read(ReadRequest(path="linked.md"))


def test_heading_read_handles_ambiguity_and_exact_breadcrumb(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "# A\nfirst\n## State\nalpha\n# B\nsecond\n## State\nbeta\n"
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="headings.md",
        text=text,
        chunk_id="a-state",
        body="## State\nalpha\n",
        heading=("A", "State"),
        lines=LineRange(3, 4),
    )
    # Add the second heading as a second chunk in one replacement.
    first = store.chunks_by_ids(("a-state",), vault_ids=("vault-a",))["a-state"]
    raw = (root / "headings.md").read_bytes()
    source = DiscoveredSource(
        "vault-a",
        root,
        "headings.md",
        "headings.md",
        SourceKind.MARKDOWN,
        text,
        f"sha256:{hashlib.sha256(raw).hexdigest()}",
        len(raw),
        (root / "headings.md").stat().st_mtime_ns,
    )
    chunks = (
        ChunkRecord(
            "a-state",
            "vault-a",
            "headings.md",
            0,
            "headings",
            ("A", "State"),
            LineRange(3, 4),
            "## State\nalpha\n",
            "alpha",
            1,
            {},
            first.content_hash,
        ),
        ChunkRecord(
            "b-state",
            "vault-a",
            "headings.md",
            1,
            "headings",
            ("B", "State"),
            LineRange(7, 8),
            "## State\nbeta\n",
            "beta",
            1,
            {},
            "sha256:b",
        ),
    )
    store.replace_source(
        SourceRevision(
            source,
            chunks,
            (),
            (),
            "manifest-v1",
            "parser-v1",
            "chunker-v1",
            "config-v1",
        )
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    with pytest.raises(ValueError, match=r"A > State.*B > State"):
        service.read(ReadRequest(path="headings.md", heading="State"))
    response = service.read(ReadRequest(path="headings.md", heading=("B", "State")))
    assert response.text == "## State\nbeta\n"
    assert response.ref.heading == ("B", "State")
    assert response.ref.lines == LineRange(7, 8)


def test_repeated_exact_heading_breadcrumbs_are_ambiguous_with_line_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "# A\n## State\nalpha\n# A\n## State\nbeta\n"
    add_chunked_source(
        store,
        root,
        path="repeated.md",
        text=text,
        chunks=(
            ("first", ("A", "State"), LineRange(2, 3)),
            ("second", ("A", "State"), LineRange(5, 6)),
        ),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    with pytest.raises(ValueError, match=r"A > State \(L2-L3\).*A > State \(L5-L6\)"):
        service.read(ReadRequest(path="repeated.md", heading=("A", "State")))


def test_adjacent_repeated_headings_are_distinct_occurrences(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "# A\n## State\nalpha\n## State\nbeta\n"
    add_chunked_source(
        store,
        root,
        path="adjacent.md",
        text=text,
        chunks=(
            ("first", ("A", "State"), LineRange(2, 3)),
            ("second", ("A", "State"), LineRange(4, 5)),
        ),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    with pytest.raises(ValueError, match=r"L2-L3.*L4-L5"):
        service.read(ReadRequest(path="adjacent.md", heading=("A", "State")))


def test_one_heading_section_split_over_chunks_resolves_once(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "# A\n## State\nfirst line\nsecond line\nthird line\n"
    add_chunked_source(
        store,
        root,
        path="split.md",
        text=text,
        chunks=(
            ("left", ("A", "State"), LineRange(2, 4)),
            ("right", ("A", "State"), LineRange(4, 5)),
        ),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    response = service.read(ReadRequest(path="split.md", heading=("A", "State")))

    assert response.ref.lines == LineRange(2, 5)
    assert response.text == "## State\nfirst line\nsecond line\nthird line\n"


def test_unique_heading_resolution_preserves_crlf(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "# A\r\n## State\r\nbeta\r\n"
    add_chunked_source(
        store,
        root,
        path="crlf.md",
        text=text,
        chunks=(("state", ("A", "State"), LineRange(2, 3)),),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    response = service.read(ReadRequest(path="crlf.md", heading=("A", "State")))

    assert response.ref.lines == LineRange(2, 3)
    assert response.text == "## State\r\nbeta\r\n"


def test_read_preserves_crlf_unicode_line_slicing_and_continuation(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "\u03b1\r\n\u03b2\r\n" + ("\u754c" * 2000) + "\r\n"
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="unicode.md",
        text=text,
        chunk_id="unicode",
        lines=LineRange(1, 3),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    line = service.read(ReadRequest(path="unicode.md", lines=LineRange(2, 2)))
    long = service.read(ReadRequest(path="unicode.md", lines=LineRange(3, 3)))

    assert line.text == "β\r\n"
    assert len(long.text) == 1800
    assert long.continuation is not None
    assert long.continuation.next_line == 3
    assert long.continuation.next_character == 1801


def test_search_index_age_is_vault_scoped_and_non_negative(
    retrieval_fixture: RetrievalFixture,
) -> None:
    response = retrieval_fixture.service.search(SearchRequest("state"))
    assert response.index_age is not None
    assert response.index_age >= 0
    assert response.elapsed_ms >= 0


def test_read_line_ranges_follow_commonmark_lines_only(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "# A\n\nalpha\u2028more alpha\n\n## State\n\nbeta\n"
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="drift.md",
        text=text,
        chunk_id="drift",
        heading=("A", "State"),
        lines=LineRange(5, 7),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    explicit = service.read(ReadRequest(path="drift.md", lines=LineRange(5, 7)))
    by_heading = service.read(ReadRequest(path="drift.md", heading=("A", "State")))

    assert explicit.text == "## State\n\nbeta\n"
    assert by_heading.ref.lines == LineRange(5, 7)
    assert by_heading.text == "## State\n\nbeta\n"


def test_read_continuation_counts_carriage_return_line_endings(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.initialize()
    text = "".join(f"line{index:04d}\r" for index in range(300))
    add_source(
        store,
        root,
        vault_id="vault-a",
        path="cr.md",
        text=text,
        chunk_id="carriage-return",
        lines=LineRange(1, 300),
    )
    service = RetrievalService(profile({"vault-a": root}), store, FakeEmbedding(), "observed-v1")

    response = service.read(ReadRequest(path="cr.md", lines=LineRange(1, 300)))

    assert len(response.text) == 1800
    assert response.continuation is not None
    assert response.continuation.next_line == 201
    assert response.continuation.next_character == 1
    assert response.continuation.remaining_characters == 900
