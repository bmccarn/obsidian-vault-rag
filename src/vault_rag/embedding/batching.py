"""Stable token- and item-bounded embedding batch planning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexedInput:
    """One embedding input with its original index and token count."""

    index: int
    text: str
    tokens: int


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Request batches plus original indexes that exceed the token budget."""

    batches: tuple[tuple[IndexedInput, ...], ...]
    oversized_indexes: tuple[int, ...]


def plan_batches(
    texts: Sequence[str],
    *,
    max_items: int,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> BatchPlan:
    """Pack inputs in stable order without exceeding either request budget."""
    if isinstance(max_items, bool) or max_items < 1:
        raise ValueError("max_items must be positive")
    if isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    batches: list[tuple[IndexedInput, ...]] = []
    current: list[IndexedInput] = []
    current_tokens = 0
    oversized: list[int] = []
    for index, text in enumerate(texts):
        tokens = count_tokens(text)
        if isinstance(tokens, bool) or tokens < 0:
            raise ValueError("token counts must be non-negative")
        if tokens > max_tokens:
            oversized.append(index)
            continue
        if current and (len(current) >= max_items or current_tokens + tokens > max_tokens):
            batches.append(tuple(current))
            current = []
            current_tokens = 0
        current.append(IndexedInput(index, text, tokens))
        current_tokens += tokens
    if current:
        batches.append(tuple(current))
    return BatchPlan(tuple(batches), tuple(oversized))
