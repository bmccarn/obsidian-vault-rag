import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.embedding.batching import plan_batches  # type: ignore[import-untyped]


def token_count(text: str) -> int:
    return len(text)


def test_plan_batches_respects_item_and_token_budgets_in_stable_order() -> None:
    plan = plan_batches(
        ["aa", "bbb", "c", "dddd", "ee"],
        max_items=3,
        max_tokens=6,
        count_tokens=token_count,
    )

    assert [[item.index for item in batch] for batch in plan.batches] == [
        [0, 1, 2],
        [3, 4],
    ]
    assert [[item.tokens for item in batch] for batch in plan.batches] == [
        [2, 3, 1],
        [4, 2],
    ]
    assert plan.oversized_indexes == ()


def test_plan_batches_separates_oversized_inputs_without_reordering_neighbors() -> None:
    plan = plan_batches(
        ["aa", "toolong", "b", "cccc"],
        max_items=8,
        max_tokens=4,
        count_tokens=token_count,
    )

    assert [[item.index for item in batch] for batch in plan.batches] == [[0, 2], [3]]
    assert plan.oversized_indexes == (1,)


@pytest.mark.parametrize(
    ("max_items", "max_tokens"),
    [(0, 10), (2, 0)],
)
def test_plan_batches_rejects_invalid_budgets(max_items: int, max_tokens: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        plan_batches(
            ["a"],
            max_items=max_items,
            max_tokens=max_tokens,
            count_tokens=token_count,
        )


def test_plan_batches_rejects_negative_token_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        plan_batches(
            ["a"],
            max_items=2,
            max_tokens=10,
            count_tokens=lambda _text: -1,
        )


def test_plan_batches_starts_a_new_batch_before_exceeding_item_budget() -> None:
    plan = plan_batches(
        ["a", "b", "c"],
        max_items=2,
        max_tokens=10,
        count_tokens=token_count,
    )

    assert [[item.index for item in batch] for batch in plan.batches] == [[0, 1], [2]]
