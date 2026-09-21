from __future__ import annotations

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.storage.postgres.index_store import _parse_pgvector_text


def test_pgvector_text_parser_accepts_whitespace_and_expected_dimensions() -> None:
    vector = _parse_pgvector_text("  [ 1.25, -2e0 ]  ", dimensions=2)

    assert vector is not None
    np.testing.assert_array_equal(vector, np.asarray([1.25, -2.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("value", "dimensions"),
    [
        ("[1, nope]", 2),
        ("[1, 2] trailing", 2),
        ("[1, 2]", 3),
        ("[NaN, 2]", 2),
        ("[true, 2]", 2),
    ],
)
def test_pgvector_text_parser_rejects_malformed_or_incompatible_values(
    value: str, dimensions: int
) -> None:
    assert _parse_pgvector_text(value, dimensions=dimensions) is None
