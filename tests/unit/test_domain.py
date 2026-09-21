import pytest

from vault_rag.domain import LineRange, SourceRef


def test_source_ref_formats_stable_citation() -> None:
    ref = SourceRef(
        vault_id="example-vault",
        path="projects/example.md",
        heading=("Example", "State"),
        lines=LineRange(start=12, end=18),
        source_hash="sha256:abc",
    )
    assert ref.citation == "vault://example-vault/projects/example.md#L12-L18"


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 1), (1, 0), (2, 1)],
)
def test_line_range_rejects_invalid_ranges(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        LineRange(start=start, end=end)


@pytest.mark.parametrize(
    "path",
    ["", "/absolute/path.md", "folder\\note.md", "..", "folder/../note.md"],
)
def test_source_ref_rejects_invalid_paths(path: str) -> None:
    with pytest.raises(ValueError):
        SourceRef(
            vault_id="example-vault",
            path=path,
            heading=(),
            lines=LineRange(start=1, end=1),
            source_hash="sha256:abc",
        )
