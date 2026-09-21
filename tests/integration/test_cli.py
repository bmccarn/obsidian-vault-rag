import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner  # pyright: ignore[reportMissingImports]

from vault_rag.app import AppFactory  # type: ignore[import-untyped]
from vault_rag.cli import create_app  # type: ignore[import-untyped]
from vault_rag.errors import ConfigError  # type: ignore[import-untyped]

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class FakeEmbeddingTransport:
    fail: bool = False
    calls: int = 0
    dimensions: int = 3
    models: list[str] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        assert request.url == httpx.URL("https://embedding.test/v1/embeddings")
        assert request.headers.get("authorization") == "Bearer top-secret-key"
        payload = json.loads(request.content)
        if self.models is None:
            self.models = []
        self.models.append(payload["model"])
        if self.fail:
            return httpx.Response(503, json={"error": "provider-secret-body"})
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": index,
                        "embedding": [float(index + 1)] + [1.0] * (self.dimensions - 1),
                    }
                    for index, _text in enumerate(payload["input"])
                ]
            },
        )


@dataclass
class CliFixture:
    runner: CliRunner
    app: Any
    vault: Path
    config: Path
    data_home: Path
    environ: dict[str, str]
    transport: FakeEmbeddingTransport

    def invoke(self, arguments: list[str]) -> Any:
        return self.runner.invoke(self.app, arguments, env=self.environ)

    @staticmethod
    def payload(result: Any) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(result.stdout))


@pytest.fixture
def cli_fixture(tmp_path: Path) -> CliFixture:
    config_home = tmp_path / "config"
    data_home = tmp_path / "data"
    registry_dir = config_home / "vault-rag"
    registry_dir.mkdir(parents=True)
    config = registry_dir / "config.toml"
    config.write_text(
        """
[profiles.vault-a]
vaults = ["vault-a"]

[embedding]
base_url = "https://embedding.test/v1"
api_key_env = "TEST_API_KEY"
model_env = "TEST_MODEL"
endpoint_class = "remote"
batch_size = 8
target_min_tokens = 10
target_max_tokens = 30
overlap_tokens = 2
max_input_tokens = 8191
tokenizer = "cl100k_base"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    os.chmod(config, 0o600)
    vault = tmp_path / "vault-a"
    vault.mkdir()
    (vault / ".vault-rag.toml").write_text(
        """
schema_version = 1
id = "vault-a"
egress_policy = "remote-allowed"
include = ["**/*.md"]

[metadata]
frontmatter_fields = ["owner", "large"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (vault / "project.md").write_text(
        "---\nowner: core\n---\n# Project State\n\nproject state is ready for retrieval\n",
        encoding="utf-8",
    )
    transport = FakeEmbeddingTransport()
    http_client = httpx.Client(transport=httpx.MockTransport(transport))

    def factory_builder(config_path: Path | None, environ: Mapping[str, str]) -> AppFactory:
        return AppFactory.from_environment(
            config_path,
            environ,
            http_client=http_client,
            sleep=lambda _delay: None,
        )

    return CliFixture(
        CliRunner(),
        create_app(factory_builder),
        vault,
        config,
        data_home,
        {
            "XDG_CONFIG_HOME": str(config_home),
            "XDG_DATA_HOME": str(data_home),
            "TEST_MODEL": "embed-test",
            "TEST_API_KEY": "top-secret-key",
        },
        transport,
    )


def register_and_index(cli: CliFixture) -> None:
    registered = cli.invoke(["register", str(cli.vault), "--json"])
    assert registered.exit_code == 0, registered.stdout
    indexed = cli.invoke(["index", "--profile", "vault-a", "--json"])
    assert indexed.exit_code == 0, indexed.stdout


def test_version_command() -> None:
    from vault_rag.cli import app

    result = CliRunner().invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    assert result.stdout == '{"version":"0.1.0"}\n'


@pytest.mark.parametrize(
    "args",
    [
        ["register", "--help"],
        ["profile", "list", "--help"],
        ["index", "--help"],
        ["search", "--help"],
        ["read", "--help"],
        ["status", "--help"],
        ["doctor", "--help"],
        ["evaluate", "--help"],
    ],
)
def test_implemented_commands_have_help(args: list[str]) -> None:
    from vault_rag.cli import app

    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0


def test_search_and_evaluate_help_expose_hardening_contracts() -> None:
    from vault_rag.cli import app

    search_help = CliRunner().invoke(app, ["search", "--help"], color=False, terminal_width=200)
    evaluate_help = CliRunner().invoke(app, ["evaluate", "--help"], color=False, terminal_width=200)

    assert search_help.exit_code == 0
    assert evaluate_help.exit_code == 0
    search_output = _ANSI_ESCAPE.sub("", search_help.stdout)
    evaluate_output = _ANSI_ESCAPE.sub("", evaluate_help.stdout)
    for option in (
        "--mode",
        "--vault",
        "--path-prefix",
        "--source-kind",
        "--frontmatter",
    ):
        assert option in search_output
    assert "--mode" in evaluate_output


def test_built_wheel_runs_version(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    subprocess.run(["uv", "build", "--out-dir", str(dist)], check=True)
    wheel = next(dist.glob("vault_rag-*.whl"))
    result = subprocess.run(
        ["uvx", "--from", str(wheel), "vault-rag", "version", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout == '{"version":"0.1.0"}\n'


def test_register_index_search_read_json(cli_fixture: CliFixture) -> None:
    registered = cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"])
    assert registered.exit_code == 0
    assert cli_fixture.payload(registered)["vault_id"] == "vault-a"

    indexed = cli_fixture.invoke(["index", "--profile", "vault-a", "--json"])
    assert indexed.exit_code == 0
    assert cli_fixture.payload(indexed)["ready_chunks"] > 0

    searched = cli_fixture.invoke(
        ["search", "--profile", "vault-a", "project state", "--limit", "5", "--json"]
    )
    assert searched.exit_code == 0
    search_payload = cli_fixture.payload(searched)
    assert search_payload["degraded"]["semantic_search"] is False
    hit = search_payload["hits"][0]
    assert hit["citation"].startswith("vault://vault-a/")

    read = cli_fixture.invoke(
        [
            "read",
            "--profile",
            "vault-a",
            hit["path"],
            "--start-line",
            str(hit["start_line"]),
            "--end-line",
            str(hit["end_line"]),
            "--expected-source-hash",
            hit["source_hash"],
            "--json",
        ]
    )
    assert read.exit_code == 0
    assert cli_fixture.payload(read)["source_hash"] == hit["source_hash"]
    assert read.stdout == read.stdout.strip() + "\n"
    assert (
        read.stdout
        == json.dumps(json.loads(read.stdout), sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_reregister_changed_manifest_id_reports_persisted_identity(
    cli_fixture: CliFixture,
) -> None:
    contents = cli_fixture.config.read_text(encoding="utf-8")
    cli_fixture.config.write_text(
        contents.replace(
            "[profiles.vault-a]",
            '[profiles.vault-b]\nvaults = ["vault-b"]\n\n[profiles.vault-a]',
        ),
        encoding="utf-8",
    )
    first = cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"])
    assert cli_fixture.payload(first)["vault_id"] == "vault-a"
    manifest = cli_fixture.vault / ".vault-rag.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('id = "vault-a"', 'id = "vault-b"'),
        encoding="utf-8",
    )

    second = cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"])

    assert second.exit_code == 0
    assert cli_fixture.payload(second)["vault_id"] == "vault-b"
    old = cli_fixture.invoke(["status", "--profile", "vault-a", "--json"])
    assert old.exit_code == 2
    assert "unregistered vault: vault-a" in old.stdout
    new = cli_fixture.invoke(["status", "--profile", "vault-b", "--json"])
    assert new.exit_code == 0
    assert cli_fixture.payload(new)["vault_ids"] == ["vault-b"]


def test_search_exposes_modes_and_structured_filters(cli_fixture: CliFixture) -> None:
    (cli_fixture.vault / "other.md").write_text(
        "---\nowner: other\n---\n# Other\n\nproject state alternate\n",
        encoding="utf-8",
    )
    register_and_index(cli_fixture)
    calls_before = cli_fixture.transport.calls

    filtered = cli_fixture.invoke(
        [
            "search",
            "--profile",
            "vault-a",
            "--mode",
            "lexical",
            "--vault",
            "vault-a",
            "--path-prefix",
            "project.md",
            "--source-kind",
            "markdown",
            "--frontmatter",
            'owner="core"',
            "project state",
            "--json",
        ]
    )

    assert filtered.exit_code == 0, filtered.stdout
    assert [hit["path"] for hit in cli_fixture.payload(filtered)["hits"]] == ["project.md"]
    assert cli_fixture.transport.calls == calls_before

    for extra in (
        ["--vault", "foreign"],
        ["--frontmatter", 'ticket="DIS-1"'],
        ["--frontmatter", 'owner="core"', "--frontmatter", 'owner="other"'],
        ["--frontmatter", "not-json"],
    ):
        result = cli_fixture.invoke(
            ["search", "--profile", "vault-a", *extra, "project state", "--json"]
        )
        assert result.exit_code == 2
        assert cli_fixture.payload(result)["error"]["code"] == "config_error"


def test_text_modes_are_human_readable_and_bounded(cli_fixture: CliFixture) -> None:
    register_and_index(cli_fixture)
    search = cli_fixture.invoke(["search", "--profile", "vault-a", "project state"])
    assert search.exit_code == 0
    assert "vault://vault-a/project.md" in search.stdout
    assert len(search.stdout) < 20_000
    status = cli_fixture.invoke(["status", "--profile", "vault-a"])
    assert status.exit_code == 0
    assert "Profile: vault-a" in status.stdout
    assert "Schema:" in status.stdout
    assert "Fingerprint state:" in status.stdout
    assert "Index age seconds:" in status.stdout
    assert "Semantic degradation:" in status.stdout
    doctor = cli_fixture.invoke(["doctor", "--profile", "vault-a"])
    assert doctor.exit_code == 0
    assert "FTS5" in doctor.stdout


def test_factory_repr_and_status_never_expose_environment_secrets(
    cli_fixture: CliFixture,
) -> None:
    factory = AppFactory.from_environment(None, cli_fixture.environ)
    rendered = repr(factory)
    assert "top-secret-key" not in rendered
    assert "embedding.test" not in rendered
    factory.close()

    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    same_value_env = {**cli_fixture.environ, "TEST_MODEL": "top-secret-key"}
    status = cli_fixture.runner.invoke(
        cli_fixture.app,
        ["status", "--profile", "vault-a", "--json"],
        env=same_value_env,
    )
    assert status.exit_code == 0
    assert "top-secret-key" not in status.stdout
    assert cli_fixture.payload(status)["model"] == "<redacted-model>"


def test_expected_errors_have_stable_json_and_exit_codes(cli_fixture: CliFixture) -> None:
    missing = cli_fixture.invoke(["search", "query", "--json"])
    assert missing.exit_code == 2
    assert cli_fixture.payload(missing)["error"]["code"] == "config_error"
    assert missing.stdout == (
        '{"error":{"code":"config_error","details":{},"message":"profile is required"}}\n'
    )

    register_and_index(cli_fixture)
    traversal = cli_fixture.invoke(["read", "--profile", "vault-a", "../outside.md", "--json"])
    assert traversal.exit_code == 3
    assert cli_fixture.payload(traversal)["error"]["code"] == "security_error"

    caller_stale = cli_fixture.invoke(
        [
            "read",
            "--profile",
            "vault-a",
            "project.md",
            "--expected-source-hash",
            "sha256:" + "0" * 64,
            "--json",
        ]
    )
    assert caller_stale.exit_code == 3
    assert cli_fixture.payload(caller_stale) == {
        "error": {
            "code": "stale_source",
            "details": {"path": "project.md", "vault_id": "vault-a"},
            "message": "source hash does not match caller expectation",
        }
    }

    (cli_fixture.vault / "project.md").write_text("changed after index", encoding="utf-8")
    stale = cli_fixture.invoke(["read", "--profile", "vault-a", "project.md", "--json"])
    assert stale.exit_code == 3
    assert cli_fixture.payload(stale)["error"]["code"] == "stale_source"


def test_pending_index_exit_four_and_degraded_search_exit_zero(
    cli_fixture: CliFixture,
) -> None:
    registered = cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"])
    assert registered.exit_code == 0
    cli_fixture.transport.fail = True
    pending = cli_fixture.invoke(["index", "--profile", "vault-a", "--json"])
    assert pending.exit_code == 4
    pending_payload = cli_fixture.payload(pending)
    assert pending_payload["pending_chunks"] > 0
    assert "provider-secret-body" not in pending.stdout
    assert "top-secret-key" not in pending.stdout

    status = cli_fixture.invoke(["status", "--profile", "vault-a", "--json"])
    assert status.exit_code == 0
    status_payload = cli_fixture.payload(status)
    assert status_payload["pending"] is True
    assert status_payload["semantic_degradation"]["reason"] == "pending_embeddings"

    doctor = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    assert doctor.exit_code == 0
    model = next(item for item in cli_fixture.payload(doctor)["checks"] if item["name"] == "model")
    assert model["status"] == "semantic-unavailable"
    assert "provider-secret-body" not in doctor.stdout

    degraded = cli_fixture.invoke(["search", "--profile", "vault-a", "project state", "--json"])
    assert degraded.exit_code == 0
    assert cli_fixture.payload(degraded)["degraded"] == {
        "reason": "pending_embeddings",
        "semantic_search": True,
    }
    assert "top-secret-key" not in degraded.stdout


def test_profile_list_handles_empty_registry(cli_fixture: CliFixture) -> None:
    contents = cli_fixture.config.read_text(encoding="utf-8")
    cli_fixture.config.write_text(
        contents.replace('[profiles.vault-a]\nvaults = ["vault-a"]\n\n', ""),
        encoding="utf-8",
    )
    environ = dict(cli_fixture.environ)
    del environ["TEST_MODEL"]
    result = cli_fixture.runner.invoke(cli_fixture.app, ["profile", "list", "--json"], env=environ)
    assert result.exit_code == 0
    assert cli_fixture.payload(result) == {"profiles": [], "profiles_truncated": 0}


def test_profile_list_is_secret_free_and_does_not_resolve_model(
    cli_fixture: CliFixture,
) -> None:
    environ = dict(cli_fixture.environ)
    del environ["TEST_MODEL"]
    environ["TEST_API_KEY"] = "do-not-print-this-secret"
    result = cli_fixture.runner.invoke(cli_fixture.app, ["profile", "list", "--json"], env=environ)
    assert result.exit_code == 0
    assert cli_fixture.payload(result) == {
        "profiles": [{"name": "vault-a", "vaults": ["vault-a"], "vaults_truncated": 0}],
        "profiles_truncated": 0,
    }
    assert "do-not-print-this-secret" not in result.stdout
    assert "TEST_API_KEY" not in result.stdout


@pytest.mark.parametrize("json_output", [True, False])
def test_invalid_config_never_prints_inline_secrets_or_credential_urls(
    cli_fixture: CliFixture,
    json_output: bool,
) -> None:
    secret = "sk-inline-supersecret-value"
    contents = cli_fixture.config.read_text(encoding="utf-8")
    cli_fixture.config.write_text(
        contents.replace(
            'base_url = "https://embedding.test/v1"',
            'base_url = "http://user:password@[/v1"',
        )
        + f'api_key = "{secret}"\n',
        encoding="utf-8",
    )
    arguments = ["profile", "list"] + (["--json"] if json_output else [])

    result = cli_fixture.invoke(arguments)

    assert result.exit_code == 2
    combined = result.stdout + result.stderr
    assert secret not in combined
    assert "user:password" not in combined
    assert "input_value" not in combined


def test_status_and_doctor_report_structured_safe_diagnostics(
    cli_fixture: CliFixture,
) -> None:
    register_and_index(cli_fixture)
    status = cli_fixture.invoke(["status", "--profile", "vault-a", "--json"])
    assert status.exit_code == 0
    payload = cli_fixture.payload(status)
    assert payload["profile"] == "vault-a"
    assert payload["vault_ids"] == ["vault-a"]
    assert payload["effective_egress_policy"] == "remote-allowed"
    assert payload["model"] == "embed-test"
    assert payload["endpoint_class"] == "remote"
    assert payload["embedding_revision"] == "1"
    assert payload["requested_dimensions"] is None
    assert payload["max_batch_tokens"] == 300_000
    assert payload["database_path"].endswith("vault-rag/index.sqlite3")
    assert payload["schema"]["compatible"] is True
    assert payload["fingerprints"]["state"] == "current"
    assert payload["counts"]["ready"] > 0
    assert payload["pending"] is False
    assert payload["last_successful_reconciliation"] is not None
    assert payload["index_age_seconds"] >= 0
    assert "top-secret-key" not in status.stdout

    doctor = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    assert doctor.exit_code == 0
    report = cli_fixture.payload(doctor)
    names = {item["name"] for item in report["checks"]}
    assert {
        "config_permissions",
        "vault_roots",
        "manifests",
        "path_collisions",
        "fts5",
        "database_permissions",
        "model",
    } <= names
    model = next(item for item in report["checks"] if item["name"] == "model")
    assert model["dimensions"] == 3
    assert model["revision"] == "1"
    assert model["requested_dimensions"] is None
    assert "top-secret-key" not in doctor.stdout
    assert "provider-secret-body" not in doctor.stdout


def test_doctor_reports_insecure_config_permissions(cli_fixture: CliFixture) -> None:
    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    os.chmod(cli_fixture.config, 0o644)
    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    assert result.exit_code == 0
    permission = next(
        item
        for item in cli_fixture.payload(result)["checks"]
        if item["name"] == "config_permissions"
    )
    assert permission["ok"] is False
    assert permission["status"] == "insecure"


def test_doctor_respects_remote_egress_policy(cli_fixture: CliFixture) -> None:
    manifest = cli_fixture.vault / ".vault-rag.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("remote-allowed", "local-only"),
        encoding="utf-8",
    )
    registered = cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"])
    assert registered.exit_code == 0
    before = cli_fixture.transport.calls
    doctor = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    assert doctor.exit_code == 0
    assert cli_fixture.transport.calls == before
    model = next(item for item in cli_fixture.payload(doctor)["checks"] if item["name"] == "model")
    assert model["status"] == "policy-disabled"
    assert model["semantic_available"] is False
    status = cli_fixture.invoke(["status", "--profile", "vault-a", "--json"])
    assert cli_fixture.payload(status)["semantic_degradation"] == {
        "reason": "semantic_disabled_by_policy",
        "semantic_search": True,
    }


@pytest.mark.parametrize(
    ("manifest_text", "expected_status"),
    [
        ("not = [valid", "invalid"),
        (
            'schema_version = 1\nid = "other-vault"\n'
            'egress_policy = "remote-allowed"\ninclude = ["**/*.md"]\n',
            "invalid",
        ),
    ],
)
def test_doctor_reports_manifest_failures_without_preemptive_error(
    cli_fixture: CliFixture,
    manifest_text: str,
    expected_status: str,
) -> None:
    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    (cli_fixture.vault / ".vault-rag.toml").write_text(manifest_text, encoding="utf-8")
    calls_before = cli_fixture.transport.calls

    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])

    assert result.exit_code == 0
    payload = cli_fixture.payload(result)
    manifest = next(item for item in payload["checks"] if item["name"] == "manifests")
    model = next(item for item in payload["checks"] if item["name"] == "model")
    assert manifest["status"] == expected_status
    assert model["status"] == "blocked"
    assert cli_fixture.transport.calls == calls_before


def test_doctor_reports_missing_root_and_model_environment(
    cli_fixture: CliFixture,
) -> None:
    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    missing_root = cli_fixture.vault.with_name("moved-vault")
    cli_fixture.vault.rename(missing_root)
    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    roots = next(
        item for item in cli_fixture.payload(result)["checks"] if item["name"] == "vault_roots"
    )
    assert result.exit_code == 0
    assert roots["ok"] is False

    missing_root.rename(cli_fixture.vault)
    environ = dict(cli_fixture.environ)
    del environ["TEST_MODEL"]
    del environ["TEST_API_KEY"]
    missing_env = cli_fixture.runner.invoke(
        cli_fixture.app, ["doctor", "--profile", "vault-a", "--json"], env=environ
    )
    model = next(
        item for item in cli_fixture.payload(missing_env)["checks"] if item["name"] == "model"
    )
    assert missing_env.exit_code == 0
    assert model["status"] == "semantic-unavailable"
    assert model["category"] == "configuration"


def test_doctor_reports_store_setup_failure_as_checks(cli_fixture: CliFixture) -> None:
    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    database = cli_fixture.data_home / "vault-rag" / "index.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"not a sqlite database")
    os.chmod(database, 0o600)

    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])

    assert result.exit_code == 0
    payload = cli_fixture.payload(result)
    fts = next(item for item in payload["checks"] if item["name"] == "fts5")
    database_check = next(
        item for item in payload["checks"] if item["name"] == "database_permissions"
    )
    assert fts["ok"] is False
    assert database_check["ok"] is True


def test_doctor_reports_insecure_database_permissions(cli_fixture: CliFixture) -> None:
    register_and_index(cli_fixture)
    database = cli_fixture.data_home / "vault-rag" / "index.sqlite3"
    os.chmod(database, 0o644)

    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])

    assert result.exit_code == 0
    payload = cli_fixture.payload(result)
    fts = next(item for item in payload["checks"] if item["name"] == "fts5")
    database_check = next(
        item for item in payload["checks"] if item["name"] == "database_permissions"
    )
    assert fts["ok"] is True
    assert database_check["ok"] is False


def test_doctor_reports_case_fold_collision(
    cli_fixture: CliFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    monkeypatch.setattr(
        "vault_rag.app.discover_sources",
        lambda _vault: SimpleNamespace(
            sources=(
                SimpleNamespace(folded_path="same.md"),
                SimpleNamespace(folded_path="same.md"),
            ),
            failures=(),
        ),
    )

    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])

    collision = next(
        item for item in cli_fixture.payload(result)["checks"] if item["name"] == "path_collisions"
    )
    assert collision["ok"] is False
    assert "collide" in collision["message"]


def test_doctor_provider_failure_includes_safe_category(cli_fixture: CliFixture) -> None:
    register_and_index(cli_fixture)
    cli_fixture.transport.fail = True
    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    model = next(item for item in cli_fixture.payload(result)["checks"] if item["name"] == "model")
    assert model["category"] == "http"
    assert "provider-secret-body" not in result.stdout


def test_search_metadata_output_has_an_explicit_bound(cli_fixture: CliFixture) -> None:
    huge = "x" * 5_000
    (cli_fixture.vault / "large.md").write_text(
        f"---\nlarge: {huge}\nowner: top-secret-key\n---\n"
        "# Large\n\nbounded metadata target top-secret-key\n",
        encoding="utf-8",
    )
    register_and_index(cli_fixture)
    result = cli_fixture.invoke(
        ["search", "--profile", "vault-a", "bounded metadata target", "--json"]
    )
    assert result.exit_code == 0
    hit = next(item for item in cli_fixture.payload(result)["hits"] if item["path"] == "large.md")
    assert hit["metadata_truncated"] is True
    assert "top-secret-key" in result.stdout
    assert len(result.stdout) < 30_000


def test_uncreatable_xdg_homes_are_config_errors_or_doctor_checks(
    cli_fixture: CliFixture,
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked", encoding="utf-8")

    config_env = {**cli_fixture.environ, "XDG_CONFIG_HOME": str(blocker)}
    config_failure = cli_fixture.runner.invoke(
        cli_fixture.app, ["profile", "list", "--json"], env=config_env
    )
    assert config_failure.exit_code == 2
    assert cli_fixture.payload(config_failure)["error"]["code"] == "config_error"
    assert "Traceback" not in config_failure.stdout

    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    data_env = {**cli_fixture.environ, "XDG_DATA_HOME": str(blocker)}
    status = cli_fixture.runner.invoke(
        cli_fixture.app, ["status", "--profile", "vault-a", "--json"], env=data_env
    )
    assert status.exit_code == 2
    assert cli_fixture.payload(status)["error"]["code"] == "config_error"

    doctor = cli_fixture.runner.invoke(
        cli_fixture.app, ["doctor", "--profile", "vault-a", "--json"], env=data_env
    )
    assert doctor.exit_code == 0
    checks = {item["name"]: item for item in cli_fixture.payload(doctor)["checks"]}
    assert checks["fts5"]["ok"] is False
    assert checks["database_permissions"]["ok"] is False


def test_search_does_not_use_vectors_from_a_stale_model_fingerprint(
    cli_fixture: CliFixture,
) -> None:
    register_and_index(cli_fixture)
    calls_before = cli_fixture.transport.calls
    environ = {**cli_fixture.environ, "TEST_MODEL": "different-model"}
    result = cli_fixture.runner.invoke(
        cli_fixture.app,
        ["search", "--profile", "vault-a", "project state", "--json"],
        env=environ,
    )
    assert result.exit_code == 0
    payload = cli_fixture.payload(result)
    assert payload["hits"]
    assert payload["degraded"] == {
        "reason": "rebuild_required",
        "semantic_search": True,
    }
    assert cli_fixture.transport.calls == calls_before


def test_status_reports_ambiguous_and_missing_observed_vectors(
    cli_fixture: CliFixture,
) -> None:
    (cli_fixture.vault / "second.md").write_text("second semantic source", encoding="utf-8")
    register_and_index(cli_fixture)
    database = cli_fixture.data_home / "vault-rag" / "index.sqlite3"
    with sqlite3.connect(database) as connection:
        second = connection.execute(
            "SELECT chunk_id FROM chunk_vectors ORDER BY chunk_id LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE chunk_vectors SET observed_fingerprint = 'synthetic-other' WHERE chunk_id = ?",
            (second,),
        )

    ambiguous = cli_fixture.payload(
        cli_fixture.invoke(["status", "--profile", "vault-a", "--json"])
    )
    assert ambiguous["semantic_degradation"] == {
        "reason": "ambiguous_vectors",
        "semantic_search": True,
    }

    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM chunk_vectors")
    missing = cli_fixture.payload(cli_fixture.invoke(["status", "--profile", "vault-a", "--json"]))
    assert missing["semantic_degradation"] == {
        "reason": "no_compatible_vectors",
        "semantic_search": True,
    }


def test_status_and_doctor_report_model_swap_as_rebuild_required(
    cli_fixture: CliFixture,
) -> None:
    register_and_index(cli_fixture)
    environ = {**cli_fixture.environ, "TEST_MODEL": "different-model"}

    status = cli_fixture.runner.invoke(
        cli_fixture.app, ["status", "--profile", "vault-a", "--json"], env=environ
    )
    assert status.exit_code == 0
    assert cli_fixture.payload(status)["semantic_degradation"] == {
        "reason": "rebuild_required",
        "semantic_search": True,
    }

    doctor = cli_fixture.runner.invoke(
        cli_fixture.app, ["doctor", "--profile", "vault-a", "--json"], env=environ
    )
    report = cli_fixture.payload(doctor)
    model = next(item for item in report["checks"] if item["name"] == "model")
    assert report["healthy"] is False
    assert model["status"] == "rebuild-required"


def test_doctor_detects_provider_width_drift_and_healthy_current_index(
    cli_fixture: CliFixture,
) -> None:
    register_and_index(cli_fixture)
    healthy = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    healthy_payload = cli_fixture.payload(healthy)
    assert healthy_payload["healthy"] is True
    assert (
        next(item for item in healthy_payload["checks"] if item["name"] == "model")["status"]
        == "ok"
    )

    cli_fixture.transport.dimensions = 7
    drifted = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])
    drifted_payload = cli_fixture.payload(drifted)
    model = next(item for item in drifted_payload["checks"] if item["name"] == "model")
    assert drifted_payload["healthy"] is False
    assert model["status"] == "dimension-mismatch"
    assert model["dimensions"] == 7
    assert model["stored_dimensions"] == [3]


def test_last_reconciliation_advances_on_unchanged_index(cli_fixture: CliFixture) -> None:
    register_and_index(cli_fixture)
    first = cli_fixture.payload(cli_fixture.invoke(["status", "--profile", "vault-a", "--json"]))[
        "last_successful_reconciliation"
    ]
    time.sleep(0.002)
    second_index = cli_fixture.invoke(["index", "--profile", "vault-a", "--json"])
    assert second_index.exit_code == 0
    second = cli_fixture.payload(cli_fixture.invoke(["status", "--profile", "vault-a", "--json"]))[
        "last_successful_reconciliation"
    ]
    assert second > first


def test_reconciliation_timestamps_are_profile_scoped_and_multi_vault_oldest(
    cli_fixture: CliFixture,
    tmp_path: Path,
) -> None:
    vault_b = tmp_path / "vault-b"
    vault_b.mkdir()
    (vault_b / ".vault-rag.toml").write_text(
        'schema_version = 1\nid = "vault-b"\negress_policy = "remote-allowed"\n'
        'include = ["**/*.md"]\n',
        encoding="utf-8",
    )
    (vault_b / "note.md").write_text("second vault text", encoding="utf-8")
    with cli_fixture.config.open("a", encoding="utf-8") as config_file:
        config_file.write(
            '\n[profiles.vault-b]\nvaults = ["vault-b"]\n'
            '\n[profiles.both]\nvaults = ["vault-a", "vault-b"]\n'
        )
    assert cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"]).exit_code == 0
    assert cli_fixture.invoke(["register", str(vault_b), "--json"]).exit_code == 0

    assert cli_fixture.invoke(["index", "--profile", "vault-a", "--json"]).exit_code == 0
    first_a = cli_fixture.payload(cli_fixture.invoke(["status", "--profile", "vault-a", "--json"]))[
        "last_successful_reconciliation"
    ]
    never_b_payload = cli_fixture.payload(
        cli_fixture.invoke(["status", "--profile", "vault-b", "--json"])
    )
    never_b = never_b_payload["last_successful_reconciliation"]
    assert first_a is not None
    assert never_b is None
    assert never_b_payload["semantic_degradation"] == {
        "reason": "no_compatible_vectors",
        "semantic_search": True,
    }

    time.sleep(0.002)
    assert cli_fixture.invoke(["index", "--profile", "vault-b", "--json"]).exit_code == 0
    after_b_a = cli_fixture.payload(
        cli_fixture.invoke(["status", "--profile", "vault-a", "--json"])
    )["last_successful_reconciliation"]
    second_b = cli_fixture.payload(
        cli_fixture.invoke(["status", "--profile", "vault-b", "--json"])
    )["last_successful_reconciliation"]
    both = cli_fixture.payload(cli_fixture.invoke(["status", "--profile", "both", "--json"]))[
        "last_successful_reconciliation"
    ]
    assert after_b_a == first_a
    assert second_b > first_a
    assert both == first_a


def test_success_payloads_preserve_source_identity_prose_and_nested_keys(
    cli_fixture: CliFixture,
) -> None:
    source = (
        "---\nowner:\n  top-secret-key: 'Authorization: pending review'\n---\n"
        "# Project State\n\nThe bearer of the ring visits project.md.\n\n"
        "Authorization: pending review by legal.\n"
    )
    (cli_fixture.vault / "project.md").write_text(source, encoding="utf-8")
    register_and_index(cli_fixture)
    environ = {
        **cli_fixture.environ,
        "CI_JOB_TOKEN": "project",
        "BUILD_KEY": "vault-a",
    }

    read = cli_fixture.runner.invoke(
        cli_fixture.app,
        ["read", "--profile", "vault-a", "project.md", "--json"],
        env=environ,
    )
    read_payload = cli_fixture.payload(read)
    assert read.exit_code == 0
    assert read_payload["text"] == source
    assert read_payload["source_hash"] == (
        f"sha256:{hashlib.sha256(source.encode('utf-8')).hexdigest()}"
    )
    assert read_payload["profile"] == "vault-a"
    assert read_payload["vault_id"] == "vault-a"
    assert read_payload["path"] == "project.md"
    assert read_payload["citation"].startswith("vault://vault-a/project.md")

    search = cli_fixture.runner.invoke(
        cli_fixture.app,
        ["search", "--profile", "vault-a", "bearer ring", "--json"],
        env=environ,
    )
    hit = cli_fixture.payload(search)["hits"][0]
    assert hit["path"] == "project.md"
    assert hit["metadata"]["owner"]["top-secret-key"] == "Authorization: pending review"
    assert "bearer of the ring" in hit["text"]


def test_diagnostic_redaction_only_matches_credential_shaped_authorization(
    cli_fixture: CliFixture,
) -> None:
    def failing_factory(config_path: Path | None, environ: Mapping[str, str]) -> AppFactory:
        del config_path, environ
        raise ConfigError(
            "Authorization: abcdefghijkl and Bearer abcdefghijkl; "
            "Authorization: pending review; " + ("x" * 430) + "top-secret-key"
        )

    result = CliRunner().invoke(
        create_app(failing_factory), ["profile", "list", "--json"], env=cli_fixture.environ
    )
    assert result.exit_code == 2
    assert "abcdefghijkl" not in result.stdout
    assert "Authorization: pending review" in result.stdout
    assert "top-secret" not in result.stdout


def test_evaluate_json_is_complete_canonical_and_source_body_free(
    cli_fixture: CliFixture, tmp_path: Path
) -> None:
    register_and_index(cli_fixture)
    evaluation_file = tmp_path / "eval.toml"
    case_blocks = [
        "[[cases]]\n"
        f'id = "semantic-{index:02}"\n'
        'kind = "semantic"\n'
        'query = "project state"\n'
        'expected_sources = [{ vault_id = "vault-a", path = "project.md" }]'
        for index in range(24)
    ]
    case_blocks.append(
        "[[cases]]\n"
        'id = "identifier-project"\n'
        'kind = "identifier"\n'
        'query = "project.md"\n'
        'expected_sources = [{ vault_id = "vault-a", path = "project.md" }]'
    )
    evaluation_file.write_text(
        "schema_version = 1\n\n"
        "[thresholds]\nrecall_at_5 = 1.0\nidentifier_rank_1 = 1.0\n\n"
        + "\n\n".join(case_blocks)
        + "\n",
        encoding="utf-8",
    )

    result = cli_fixture.invoke(
        ["evaluate", "--profile", "vault-a", "--file", str(evaluation_file), "--json"]
    )

    assert result.exit_code == 0, result.stdout
    payload = cli_fixture.payload(result)
    assert payload["profile"] == "vault-a"
    assert payload["case_count"] == 25
    assert payload["mode"] == "hybrid"
    assert payload["degraded_cases"] == 0
    assert payload["passed"] is True
    assert payload["invalid_citations"] == 0
    assert "project state is ready for retrieval" not in result.stdout
    assert "query" not in result.stdout
    assert result.stdout == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def test_evaluate_threshold_failure_returns_complete_report_and_stable_exit(
    cli_fixture: CliFixture, tmp_path: Path
) -> None:
    register_and_index(cli_fixture)
    evaluation_file = tmp_path / "eval.toml"
    evaluation_file.write_text(
        """
schema_version = 1

[thresholds]
recall_at_5 = 1.0
identifier_rank_1 = 1.0

[[cases]]
id = "semantic-only"
kind = "semantic"
query = "project state"
expected_paths = ["project.md"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = cli_fixture.invoke(
        ["evaluate", "--profile", "vault-a", "--file", str(evaluation_file), "--json"]
    )

    assert result.exit_code == 5
    payload = cli_fixture.payload(result)
    assert payload["case_count"] == 1
    assert payload["recall_at_5"] == 1.0
    assert payload["identifier_rank_1"] == 0.0
    assert payload["passed"] is False
    assert len(payload["cases"]) == 1


def test_evaluate_fails_when_hybrid_semantics_degrade_despite_lexical_recall(
    cli_fixture: CliFixture, tmp_path: Path
) -> None:
    register_and_index(cli_fixture)
    cli_fixture.transport.fail = True
    evaluation_file = tmp_path / "eval.toml"
    cases = "\n".join(
        "[[cases]]\n"
        f'id = "degraded-{index:02}"\n'
        'kind = "semantic"\n'
        'query = "project state"\n'
        'expected_sources = [{ vault_id = "vault-a", path = "project.md" }]\n'
        for index in range(25)
    )
    evaluation_file.write_text(
        "schema_version = 1\n[thresholds]\nrecall_at_5 = 1.0\nidentifier_rank_1 = 0.0\n" + cases,
        encoding="utf-8",
    )

    result = cli_fixture.invoke(
        ["evaluate", "--profile", "vault-a", "--file", str(evaluation_file), "--json"]
    )

    assert result.exit_code == 5
    payload = cli_fixture.payload(result)
    assert payload["recall_at_5"] == 1.0
    assert payload["degraded_cases"] == 25
    assert payload["degradation_reasons"] == ["query_embedding_failed"]
    assert payload["passed"] is False


def test_evaluate_schema_errors_use_config_error_contract(
    cli_fixture: CliFixture, tmp_path: Path
) -> None:
    evaluation_file = tmp_path / "eval.toml"
    evaluation_file.write_text("schema_version = 2\nsecret = 'do-not-print-this-secret'\n")
    result = cli_fixture.invoke(
        ["evaluate", "--profile", "vault-a", "--file", str(evaluation_file), "--json"]
    )
    assert result.exit_code == 2
    assert cli_fixture.payload(result)["error"]["code"] == "config_error"
    assert "do-not-print-this-secret" not in result.stdout


def test_evaluate_text_is_bounded_and_reports_threshold_verdict(
    cli_fixture: CliFixture, tmp_path: Path
) -> None:
    register_and_index(cli_fixture)
    evaluation_file = tmp_path / "eval.toml"
    cases = "\n".join(
        "[[cases]]\n"
        f'id = "case-{index:02}"\n'
        'kind = "semantic"\n'
        'query = "project state"\n'
        'expected_paths = ["project.md"]\n'
        for index in range(25)
    )
    evaluation_file.write_text(
        "schema_version = 1\n[thresholds]\nrecall_at_5 = 0.0\nidentifier_rank_1 = 0.0\n" + cases,
        encoding="utf-8",
    )
    result = cli_fixture.invoke(
        ["evaluate", "--profile", "vault-a", "--file", str(evaluation_file)]
    )
    assert result.exit_code == 0
    assert "Evaluation: PASS" in result.stdout
    assert "Recall@5:" in result.stdout
    assert "project state is ready for retrieval" not in result.stdout
    assert len(result.stdout) < 20_000


def test_multi_vault_read_requires_explicit_vault(tmp_path: Path) -> None:
    # The service-owned rule must remain visible through the CLI rather than being
    # replaced with implicit all-vault or first-vault behavior.
    config_home = tmp_path / "config" / "vault-rag"
    config_home.mkdir(parents=True)
    roots: list[Path] = []
    vault_tables: list[str] = []
    for vault_id in ("vault-a", "vault-b"):
        root = tmp_path / vault_id
        root.mkdir()
        roots.append(root)
        (root / ".vault-rag.toml").write_text(
            f'schema_version = 1\nid = "{vault_id}"\n'
            'egress_policy = "remote-allowed"\ninclude = ["**/*.md"]\n',
            encoding="utf-8",
        )
        (root / "same.md").write_text(f"{vault_id} shared text\n", encoding="utf-8")
        vault_tables.append(f'[vaults.{vault_id}]\npath = "{root}"')
    config = config_home / "config.toml"
    config.write_text(
        "\n\n".join(vault_tables)
        + '\n\n[profiles.both]\nvaults = ["vault-a", "vault-b"]\n\n'
        + '[embedding]\nbase_url = "https://embedding.test/v1"\n'
        + 'api_key_env = "TEST_API_KEY"\nmodel_env = "TEST_MODEL"\n'
        + 'endpoint_class = "remote"\ntarget_min_tokens = 10\n'
        + "target_max_tokens = 30\noverlap_tokens = 2\nmax_input_tokens = 100\n",
        encoding="utf-8",
    )
    transport = FakeEmbeddingTransport()
    client = httpx.Client(transport=httpx.MockTransport(transport))

    def builder(config_path: Path | None, environ: Mapping[str, str]) -> AppFactory:
        return AppFactory.from_environment(
            config_path, environ, http_client=client, sleep=lambda _: None
        )

    app = create_app(builder)
    env = {
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "TEST_MODEL": "embed-test",
        "TEST_API_KEY": "top-secret-key",
    }
    runner = CliRunner()
    assert runner.invoke(app, ["index", "--profile", "both", "--json"], env=env).exit_code == 0
    ambiguous = runner.invoke(app, ["read", "--profile", "both", "same.md", "--json"], env=env)
    assert ambiguous.exit_code == 2
    explicit = runner.invoke(
        app,
        ["read", "--profile", "both", "same.md", "--vault", "vault-b", "--json"],
        env=env,
    )
    assert explicit.exit_code == 0
    assert json.loads(explicit.stdout)["text"] == "vault-b shared text\n"


def test_search_citations_resolve_to_physical_source_lines(cli_fixture: CliFixture) -> None:
    note = cli_fixture.vault / "drift.md"
    note.write_text(
        "# Drift Note\n\nalpha\u2028more alpha\n\n## Target Section\n\ntargetdrift body line\n",
        encoding="utf-8",
    )
    register_and_index(cli_fixture)

    searched = cli_fixture.invoke(["search", "--profile", "vault-a", "targetdrift", "--json"])
    assert searched.exit_code == 0, searched.stdout
    hit = next(item for item in CliFixture.payload(searched)["hits"] if item["path"] == "drift.md")

    physical_lines = note.read_text(encoding="utf-8").split("\n")[:-1]
    assert len(physical_lines) == 7
    assert hit["end_line"] <= len(physical_lines)

    read = cli_fixture.invoke(
        [
            "read",
            "--profile",
            "vault-a",
            "drift.md",
            "--start-line",
            str(hit["start_line"]),
            "--end-line",
            str(hit["end_line"]),
            "--json",
        ]
    )
    assert read.exit_code == 0, read.stdout
    expected = "".join(
        f"{line}\n" for line in physical_lines[hit["start_line"] - 1 : hit["end_line"]]
    )
    assert CliFixture.payload(read)["text"] == expected
    assert "targetdrift" in CliFixture.payload(read)["text"]


def test_index_reports_bounded_discovery_failures_without_aborting(
    cli_fixture: CliFixture,
) -> None:
    (cli_fixture.vault / "broken.md").write_bytes(b"# Broken\n\xff\xfe\n")

    registered = cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"])
    assert registered.exit_code == 0, registered.stdout
    indexed = cli_fixture.invoke(["index", "--profile", "vault-a", "--json"])
    assert indexed.exit_code == 0, indexed.stdout

    payload = CliFixture.payload(indexed)
    assert payload["total_sources"] == 1
    assert payload["parse_failures"] == 1
    assert [(item["path"], item["category"]) for item in payload["diagnostics"]] == [
        ("broken.md", "parse_error")
    ]

    searched = cli_fixture.invoke(["search", "--profile", "vault-a", "project state", "--json"])
    assert searched.exit_code == 0, searched.stdout
    assert [hit["path"] for hit in CliFixture.payload(searched)["hits"]] == ["project.md"]


def test_doctor_leaves_per_file_discovery_failures_to_index(cli_fixture: CliFixture) -> None:
    """Per-file discovery problems are reported by ``index``, not by ``doctor``.

    ``doctor`` keeps reporting vault-level problems — a missing or symlinked
    root and a case-folded path collision — but one undecodable note no longer
    fails its ``path_collisions`` check.
    """
    (cli_fixture.vault / "broken.md").write_bytes(b"# Broken\n\xff\xfe\n")
    registered = cli_fixture.invoke(["register", str(cli_fixture.vault), "--json"])
    assert registered.exit_code == 0, registered.stdout

    result = cli_fixture.invoke(["doctor", "--profile", "vault-a", "--json"])

    assert result.exit_code == 0, result.stdout
    collision = next(
        item for item in CliFixture.payload(result)["checks"] if item["name"] == "path_collisions"
    )
    assert collision["ok"] is True
    assert collision["status"] == "ok"


def test_serve_builds_runtime_and_runs_single_worker_without_access_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vault_rag.cli as cli

    calls: list[object] = []

    def build(config_path: Path, data_root: Path, environ: Mapping[str, str]) -> object:
        calls.append((config_path, data_root, dict(environ)))
        return object()

    def run(app: object, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr(cli, "build_runtime", build, raising=False)
    monkeypatch.setattr(cli.uvicorn, "run", run, raising=False)

    config_path = tmp_path / "service.yaml"
    data_root = tmp_path / "data"
    result = CliRunner().invoke(
        cli.create_app(),
        [
            "serve",
            "--service-config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--host",
            "127.0.0.1",
            "--port",
            "9090",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert cast(tuple[Path, Path, object], calls[0])[:2] == (config_path, data_root)
    assert cast(tuple[object, dict[str, object]], calls[1])[1] == {
        "host": "127.0.0.1",
        "port": 9090,
        "workers": 1,
        "access_log": False,
    }


def test_api_builds_read_only_postgresql_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vault_rag.cli as cli

    calls: list[object] = []
    runtime = object()

    monkeypatch.setattr(
        cli,
        "build_api_runtime",
        lambda config, data, environ: calls.append((config, data, dict(environ))) or runtime,
        raising=False,
    )
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
        raising=False,
    )
    config_path = tmp_path / "service.yaml"
    data_root = tmp_path / "data"

    result = CliRunner().invoke(
        cli.create_app(),
        [
            "api",
            "--service-config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--host",
            "127.0.0.1",
            "--port",
            "9090",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert cast(tuple[Path, Path, object], calls[0])[:2] == (config_path, data_root)
    assert cast(tuple[object, dict[str, object]], calls[1])[1] == {
        "host": "127.0.0.1",
        "port": 9090,
        "workers": 1,
        "access_log": False,
    }


def test_worker_and_nested_database_commands_own_their_runtimes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vault_rag.cli as cli

    calls: list[object] = []

    class Runtime:
        def run_forever(self) -> None:
            calls.append("run")

        def close(self) -> None:
            calls.append("close")

        def request_stop(self) -> None:
            calls.append("stop")

    monkeypatch.setattr(
        cli,
        "build_worker_runtime",
        lambda *_args, **_kwargs: calls.append("worker") or Runtime(),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "migrate_database",
        lambda *_args, **_kwargs: calls.append("migrate") or (1,),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "check_database",
        lambda *_args, **_kwargs: calls.append("check") or (1, 2),
        raising=False,
    )
    config_path = tmp_path / "service.yaml"
    data_root = tmp_path / "data"

    worker = CliRunner().invoke(
        cli.create_app(),
        [
            "worker",
            "--service-config",
            str(config_path),
            "--data-root",
            str(data_root),
        ],
    )
    migration = CliRunner().invoke(
        cli.create_app(),
        ["db", "migrate", "--service-config", str(config_path)],
    )
    check = CliRunner().invoke(
        cli.create_app(),
        ["db", "check", "--service-config", str(config_path)],
    )

    assert worker.exit_code == 0, worker.stdout
    assert migration.exit_code == 0, migration.stdout
    assert check.exit_code == 0, check.stdout
    assert calls == ["worker", "run", "close", "migrate", "check"]
    assert "Applied PostgreSQL migrations: 1" in migration.stdout
    assert "Compatible PostgreSQL migrations: 1, 2" in check.stdout


def test_db_cleanup_is_explicit_and_reports_bounded_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vault_rag.cli as cli

    calls: list[tuple[Path, str]] = []

    class Result:
        revisions_deleted = 2
        blobs_deleted = 3
        embeddings_deleted = 5
        elapsed_ms = 8

    def cleanup(config_path: Path, _environ: object, vault_id: str) -> Result:
        calls.append((config_path, vault_id))
        return Result()

    monkeypatch.setattr(cli, "cleanup_database", cleanup)
    config_path = tmp_path / "service.yaml"
    result = CliRunner().invoke(
        cli.create_app(),
        ["db", "cleanup", "vault-a", "--service-config", str(config_path), "--json"],
    )

    assert result.exit_code == 0, result.stdout
    assert calls == [(config_path, "vault-a")]
    assert '"revisions_deleted":2' in result.stdout
    assert '"blobs_deleted":3' in result.stdout
    assert '"embeddings_deleted":5' in result.stdout


def test_worker_installs_graceful_signal_handlers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vault_rag.cli as cli

    installed: dict[int, object] = {}
    signal_calls: list[tuple[int, object]] = []
    stopped: list[str] = []

    def install(signum: int, handler: object) -> object:
        previous = installed.get(signum, f"previous-{signum}")
        installed[signum] = handler
        signal_calls.append((signum, handler))
        return previous

    class Runtime:
        def run_forever(self) -> None:
            handler = installed[cli.signal.SIGTERM]
            assert callable(handler)
            handler(cli.signal.SIGTERM, None)

        def request_stop(self) -> None:
            stopped.append("stop")

        def close(self) -> None:
            stopped.append("close")

    monkeypatch.setattr(cli.signal, "signal", install)
    monkeypatch.setattr(cli, "build_worker_runtime", lambda *_args, **_kwargs: Runtime())

    result = CliRunner().invoke(
        cli.create_app(),
        ["worker", "--service-config", str(tmp_path / "service.yaml")],
    )

    assert result.exit_code == 0, result.stdout
    assert stopped == ["stop", "close"]
    assert [signum for signum, _handler in signal_calls] == [
        cli.signal.SIGINT,
        cli.signal.SIGTERM,
        cli.signal.SIGTERM,
        cli.signal.SIGINT,
    ]


def test_db_migrate_redacts_failures_and_supports_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vault_rag.cli as cli

    monkeypatch.setenv("VAULT_RAG_DATABASE_URL", "postgresql://user:secret@db/vault")

    def fail(*_args: object, **_kwargs: object) -> tuple[int, ...]:
        raise ValueError("failed postgresql://user:secret@db/vault")

    monkeypatch.setattr(cli, "migrate_database", fail)
    result = CliRunner().invoke(
        cli.create_app(),
        ["db", "migrate", "--service-config", str(tmp_path / "service.yaml"), "--json"],
    )

    assert result.exit_code == 2
    assert "secret" not in result.stdout
    assert "postgresql://user:" not in result.stdout
    assert '"code":"config_error"' in result.stdout
