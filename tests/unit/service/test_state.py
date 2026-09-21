import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from vault_rag.indexing import IndexDiagnostic, IndexReport
from vault_rag.service.state import IndexReportState, SyncStateStore, VaultSyncState


def report(
    *,
    blocking_failures: int = 0,
    diagnostics: tuple[IndexDiagnostic, ...] = (),
) -> IndexReport:
    return IndexReport(
        total_sources=8,
        added_sources=2,
        changed_sources=1,
        unchanged_sources=4,
        deleted_sources=1,
        ready_chunks=12,
        pending_chunks=3,
        parse_failures=1,
        embedding_failures=2,
        embedding_requests=5,
        diagnostics=diagnostics,
        elapsed_ms=123,
        blocking_failures=blocking_failures,
    )


def state() -> VaultSyncState:
    timestamp = datetime(2026, 8, 7, 12, 30, tzinfo=UTC)
    return VaultSyncState(
        vault_id="vault-a",
        configured_ref="refs/heads/main",
        configured_sha="a" * 40,
        fetched_sha="b" * 64,
        checkout_sha="c" * 40,
        attempted_sha="d" * 64,
        reconciled_sha="e" * 40,
        configured_at=timestamp,
        fetched_at=timestamp,
        checkout_at=timestamp,
        attempted_at=timestamp,
        reconciled_at=timestamp,
        report=IndexReportState.from_report(
            report(
                diagnostics=(
                    IndexDiagnostic("vault-a", "notes/a.md", "parse_error", "bad heading"),
                )
            )
        ),
        sync_degraded_reason="index_pending",
        sync_degraded_at=timestamp,
    )


def test_store_round_trips_exact_state_without_repository_secrets(tmp_path: Path) -> None:
    store = SyncStateStore(tmp_path / "state")
    expected = state()

    store.write(expected)

    result = store.read("vault-a", "refs/heads/main")
    payload = (tmp_path / "state/vault-a.json").read_text(encoding="utf-8")
    assert result.state == expected
    assert result.warning is None
    assert "https://example.test/private.git" not in payload
    assert "credential-value" not in payload


def test_store_retains_blocking_failures_and_reads_legacy_report_state(tmp_path: Path) -> None:
    store = SyncStateStore(tmp_path / "state")
    persisted_report = IndexReportState.from_report(report(blocking_failures=3))
    expected = state().model_copy(update={"report": persisted_report})

    store.write(expected)
    round_tripped = store.read("vault-a", "refs/heads/main")
    legacy_payload = expected.model_dump(mode="json")
    del legacy_payload["report"]["blocking_failures"]
    (tmp_path / "state/vault-a.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy = store.read("vault-a", "refs/heads/main")

    assert round_tripped.state.report is not None
    assert round_tripped.state.report.blocking_failures == 3
    assert legacy.state.report is not None
    assert legacy.state.report.blocking_failures == 0


def test_missing_state_returns_clean_initial_snapshot(tmp_path: Path) -> None:
    result = SyncStateStore(tmp_path / "state").read("vault-a", "refs/heads/main")

    assert result.state == VaultSyncState(vault_id="vault-a", configured_ref="refs/heads/main")
    assert result.warning is None


def test_corrupt_state_returns_fresh_degraded_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "state/vault-a.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")

    result = SyncStateStore(tmp_path / "state").read("vault-a", "refs/heads/main")

    assert result.state.vault_id == "vault-a"
    assert result.state.sync_degraded_reason == "state_invalid"
    assert result.warning == "stored synchronization state is invalid"


def test_unreadable_state_returns_safe_degraded_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state/vault-a.json"
    path.parent.mkdir()
    path.write_text("ignored", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def deny_state_read(candidate: Path) -> bytes:
        if candidate == path:
            raise PermissionError("credential-value must not leak")
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", deny_state_read)

    result = SyncStateStore(path.parent).read("vault-a", "refs/heads/main")

    assert result.state == VaultSyncState(
        vault_id="vault-a",
        configured_ref="refs/heads/main",
        sync_degraded_reason="state_invalid",
    )
    assert result.warning == "stored synchronization state is invalid"
    assert "credential-value" not in result.warning


@pytest.mark.parametrize(
    "replacement",
    [
        {"vault_id": "vault-b"},
        {"configured_ref": "refs/heads/release"},
        {"configured_sha": "A" * 40},
    ],
)
def test_invalid_stored_identity_or_sha_returns_degraded_snapshot(
    tmp_path: Path, replacement: dict[str, str]
) -> None:
    store = SyncStateStore(tmp_path / "state")
    store.write(state())
    path = tmp_path / "state/vault-a.json"
    payload = state().model_dump(mode="json")
    payload.update(replacement)
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = store.read("vault-a", "refs/heads/main")

    assert result.state == VaultSyncState(
        vault_id="vault-a",
        configured_ref="refs/heads/main",
        sync_degraded_reason="state_invalid",
    )
    assert result.warning == "stored synchronization state is invalid"


def test_oversized_stored_state_returns_degraded_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "state/vault-a.json"
    path.parent.mkdir()
    path.write_bytes(b" " * (1024 * 1024 + 1))

    result = SyncStateStore(tmp_path / "state").read("vault-a", "refs/heads/main")

    assert result.state.sync_degraded_reason == "state_invalid"
    assert result.warning == "stored synchronization state is invalid"


def test_index_report_state_bounds_diagnostics_and_messages() -> None:
    source_report = report(
        diagnostics=tuple(
            IndexDiagnostic("vault-a", f"notes/{number}.md", "parse_error", "x" * 501)
            for number in range(101)
        )
    )

    persisted = IndexReportState.from_report(source_report)

    assert len(persisted.diagnostics) == 100
    assert all(len(diagnostic.message) == 500 for diagnostic in persisted.diagnostics)
    with pytest.raises(ValidationError):
        IndexReportState.model_validate(
            persisted.model_dump(mode="json") | {"diagnostics": [{}] * 101}
        )


def test_sync_degradation_reason_is_bounded() -> None:
    with pytest.raises(ValidationError):
        VaultSyncState(
            vault_id="vault-a",
            configured_ref="refs/heads/main",
            sync_degraded_reason="x" * 501,
        )


def test_store_creates_restricted_directory_and_state_file(tmp_path: Path) -> None:
    store = SyncStateStore(tmp_path / "state")

    store.write(state())

    assert (tmp_path / "state").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "state/vault-a.json").stat().st_mode & 0o777 == 0o600


def test_failed_replace_removes_temporary_file(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    (state_dir / "vault-a.json").mkdir()

    with pytest.raises(OSError):
        SyncStateStore(state_dir).write(state())

    assert tuple(state_dir.glob(".vault-a.json.*")) == ()
