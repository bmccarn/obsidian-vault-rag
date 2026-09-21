"""Release-artifact content contract.

Only member names are inspected.  No archive member is ever extracted or read,
so this test cannot itself surface local transcript, private-repository, or
credential content.
"""

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pathspec
import pytest  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = Path(__file__).parents[2]

ALLOWED_SDIST_ROOTS = frozenset(
    {
        # hatchling force-includes every VCS exclusion file into the sdist
        # (hatchling/builders/sdist.py:338-340), after include/exclude
        # selection and outside its reach.  The root .gitignore is a tracked
        # public file, and in a released sdist it usefully documents that
        # .pi-subagents/ and .superpowers/ are local harness state.
        ".gitignore",
        "PKG-INFO",
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "examples",
        "deploy",
        "charts",
        "compose.yaml",
        "Dockerfile",
        ".dockerignore",
        "docs",
        "pyproject.toml",
        "src",
        "tests",
        "uv.lock",
    }
)
FORBIDDEN_BASENAMES = frozenset({".coverage", "CLAUDE.local.md"})
FORBIDDEN_PREFIXES = (".pi-subagents/", ".superpowers/", "docs/superpowers/")
PRIVATE_DOCKER_CONTEXT_PATHS = (
    ".git/HEAD",
    ".secrets/github_token",
    "secrets/litellm_api_key",
    "data/index.sqlite3",
    "repos/private-vault/.git/config",
    "state/private-vault.json",
    "CLAUDE.local.md",
    "deploy/docker/service.local.yaml",
)


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("release")
    subprocess.run(
        ["uv", "build", "--out-dir", str(out)],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return next(out.glob("vault_rag-*.tar.gz")), next(out.glob("vault_rag-*.whl"))


def sdist_members(archive: Path) -> list[str]:
    """Return archive-relative paths with the single top-level prefix removed."""
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
    return [name.split("/", 1)[1] for name in names if "/" in name]


def test_packaged_runtime_content_excludes_observability_sentinels(
    built_artifacts: tuple[Path, Path],
) -> None:
    """Release code must not contain request, source, database, or provider sample secrets."""
    sdist, wheel = built_artifacts
    sentinels = (
        "task6-sentinel-query",
        "task6-sentinel-source",
        "task6-sentinel-path",
        "task6-sentinel-dsn",
        "task6-sentinel-credential",
        "task6-sentinel-vector",
        "task6-sentinel-fingerprint",
        "task6-sentinel-provider",
    )
    with tarfile.open(sdist) as archive:
        sdist_content = b"\n".join(
            member_file.read()
            for member in archive.getmembers()
            if "/src/vault_rag/" in member.name
            and member.isfile()
            and (member_file := archive.extractfile(member)) is not None
        ).decode("utf-8", errors="replace")
    with zipfile.ZipFile(wheel) as archive:
        wheel_content = b"\n".join(
            archive.read(name) for name in archive.namelist() if name.startswith("vault_rag/")
        ).decode("utf-8", errors="replace")
    for sentinel in sentinels:
        assert sentinel not in sdist_content
        assert sentinel not in wheel_content


def test_sdist_ships_only_allowlisted_release_paths(
    built_artifacts: tuple[Path, Path],
) -> None:
    sdist, _wheel = built_artifacts
    roots = {member.split("/", 1)[0] for member in sdist_members(sdist)}

    assert sorted(root for root in roots if root not in ALLOWED_SDIST_ROOTS) == []


def test_sdist_excludes_local_and_harness_artifacts(
    built_artifacts: tuple[Path, Path],
) -> None:
    sdist, _wheel = built_artifacts
    members = sdist_members(sdist)

    assert [name for name in members if Path(name).name in FORBIDDEN_BASENAMES] == []
    assert [name for name in members if name.startswith(FORBIDDEN_PREFIXES)] == []


def test_docker_context_excludes_private_operator_inputs() -> None:
    dockerignore = PROJECT_ROOT / ".dockerignore"
    patterns = dockerignore.read_text(encoding="utf-8").splitlines()
    ignored = pathspec.GitIgnoreSpec.from_lines(patterns)

    assert [path for path in PRIVATE_DOCKER_CONTEXT_PATHS if not ignored.match_file(path)] == []


def test_sdist_still_ships_the_installable_project(
    built_artifacts: tuple[Path, Path],
) -> None:
    sdist, _wheel = built_artifacts
    members = set(sdist_members(sdist))

    assert {
        "README.md",
        "docs/configuration.md",
        "docs/mcp.md",
        "docs/retrieval.md",
        "pyproject.toml",
        "src/vault_rag/cli.py",
        "tests/integration/test_cli.py",
        "uv.lock",
    } <= members


def test_operator_docs_cover_phase_1_hardening_contracts() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    configuration = (PROJECT_ROOT / "docs/configuration.md").read_text(encoding="utf-8")
    retrieval = (PROJECT_ROOT / "docs/retrieval.md").read_text(encoding="utf-8")

    for term in ("lexical", "dense", "hybrid", "--frontmatter", "--path-prefix"):
        assert term in readme
    for term in (
        "max_batch_tokens",
        "revision",
        "dimensions",
        "HTTPS",
        "loopback",
        "minimum_cases",
        "p95_latency_ms",
        "expected_sources",
        "vault_id",
    ):
        assert term in configuration
    for term in (
        "degraded cases",
        "25",
        "500 ms",
        "identifier-shaped",
        "dense-only",
        "lexical-only",
        "Example dimension baseline",
        "text-embedding-3-large",
        "dimensions = 1024",
    ):
        assert term in retrieval


def test_release_docs_keep_operator_deployment_inputs_out_of_artifacts() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    configuration = (PROJECT_ROOT / "docs/configuration.md").read_text(encoding="utf-8")

    assert "Private deployment inputs are not release artifacts." in readme
    assert "Private deployment inputs are not release artifacts." in configuration


def test_wheel_ships_only_the_package_and_metadata(
    built_artifacts: tuple[Path, Path],
) -> None:
    _sdist, wheel = built_artifacts
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
    roots = sorted({name.split("/", 1)[0] for name in names})

    assert roots == ["vault_rag", "vault_rag-0.1.0.dist-info"]
    assert [name for name in names if Path(name).name in FORBIDDEN_BASENAMES] == []
