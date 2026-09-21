import unicodedata

import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.retrieval import (  # type: ignore[import-untyped]
    IdentifierMatch,
    build_fts_query,
    recognize_identifiers,
)


@pytest.mark.parametrize(
    ("query", "kind", "value"),
    [
        ("DIS-12345", "jira", "DIS-12345"),
        ("dzte-infra #298", "pr", "dzte-infra#298"),
        ("commit:7427888", "git", "7427888"),
        ("sha:1234567", "git", "1234567"),
        ("7427abc", "git", "7427abc"),
        (
            "projects/snowyowl-environment-standup.md",
            "path",
            "projects/snowyowl-environment-standup.md",
        ),
    ],
)
def test_recognizes_complete_identifiers(query: str, kind: str, value: str) -> None:
    assert recognize_identifiers(query) == (IdentifierMatch(kind=kind, value=value),)


def test_bare_numbers_partial_hex_and_unsafe_paths_do_not_trigger_priority() -> None:
    assert recognize_identifiers("298") == ()
    assert recognize_identifiers("123456") == ()
    assert recognize_identifiers("1234567") == ()
    assert recognize_identifiers("../projects/example.md") == ()
    assert recognize_identifiers("projects/example.exe") == ()


def test_recognizers_normalize_only_case_insensitive_identifiers_and_deduplicate() -> None:
    assert recognize_identifiers("sha:ABCDEF1 abcdef1") == (
        IdentifierMatch(kind="git", value="abcdef1"),
    )
    assert recognize_identifiers("Repo-Name #42") == (
        IdentifierMatch(kind="pr", value="repo-name#42"),
    )


def test_fts_query_quotes_tokens_and_never_preserves_operators_or_raw_quotes() -> None:
    query = build_fts_query('alpha" OR path:*; DROP TABLE chunks -- beta')
    assert query == '"alpha" OR "OR" OR "path" OR "DROP" OR "TABLE" OR "chunks" OR "beta"'
    assert "*" not in query
    assert ";" not in query
    assert "--" not in query
    assert build_fts_query('"* : ( )') == ""


def test_fts_query_preserves_meaningful_identifier_punctuation_inside_quotes() -> None:
    assert build_fts_query("DIS-123 dzte-infra #298 commit:7427888") == (
        '"DIS-123" OR "dzte-infra" OR "298" OR "commit:7427888"'
    )


def test_path_recognizer_normalizes_nfd_query_before_matching() -> None:
    nfd_path = unicodedata.normalize("NFD", "notes/café.md")

    assert recognize_identifiers(nfd_path) == (IdentifierMatch(kind="path", value="notes/café.md"),)


@pytest.mark.parametrize(
    "query",
    [
        "open notes/café.md.",
        "open notes/café.md, please",
        "open (notes/café.md)",
        "notes/café.md...",
    ],
)
def test_path_recognizer_stops_at_supported_suffix_before_sentence_punctuation(
    query: str,
) -> None:
    assert recognize_identifiers(query) == (IdentifierMatch(kind="path", value="notes/café.md"),)


@pytest.mark.parametrize(
    "query",
    [
        "notes/café.md.backup",
        "notes/café.md2",
        "notes/café.md/more",
        "https://example.test/notes/café.md",
    ],
)
def test_path_recognizer_rejects_false_token_boundaries(query: str) -> None:
    assert recognize_identifiers(query) == ()


def test_path_recognizer_accepts_dotted_posix_components() -> None:
    assert recognize_identifiers("projects.v2/notes/exact.note.md,") == (
        IdentifierMatch(kind="path", value="projects.v2/notes/exact.note.md"),
    )


def test_path_recognizer_handles_long_punctuation_run_without_partial_match() -> None:
    assert recognize_identifiers(("-" * 9_000) + " notes/example.md.") == (
        IdentifierMatch(kind="path", value="notes/example.md"),
    )
