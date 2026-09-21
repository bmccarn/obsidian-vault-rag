"""OpenAI-compatible embedding requests with bounded, non-secret failures."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx
import numpy as np  # pyright: ignore[reportMissingImports]
import tiktoken  # pyright: ignore[reportMissingImports]

from vault_rag.config import EmbeddingConfig
from vault_rag.errors import SemanticUnavailableError

from .batching import plan_batches

_MAX_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 10.0
_MAX_MESSAGE_LENGTH = 1_000
_BACKOFF_SECONDS = (0.5, 1.0)


@dataclass(frozen=True, slots=True)
class EmbeddingFailure:
    """Failure associated with one input's index in the original sequence."""

    index: int
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Successful vectors and per-input failures from one embedding operation."""

    vectors: tuple[np.ndarray, ...]
    failures: tuple[EmbeddingFailure, ...]
    dimensions: int | None


@dataclass(frozen=True, slots=True)
class ModelProbe:
    """Observed model identity and vector width."""

    model: str
    dimensions: int


@dataclass(frozen=True, slots=True)
class _BatchError:
    category: str
    message: str


class _InvalidResponseError(ValueError):
    pass


def _bounded(message: str) -> str:
    """Bound diagnostics before callers can persist them."""
    return message[:_MAX_MESSAGE_LENGTH]


def _retry_after(response: httpx.Response) -> float | None:
    raw_value = response.headers.get("retry-after")
    if raw_value is None:
        return None
    try:
        delay = float(raw_value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw_value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = (retry_at - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return min(delay, _MAX_RETRY_AFTER_SECONDS)


def _failure_for_response(response: httpx.Response) -> _BatchError:
    # Provider bodies and endpoint URLs are deliberately omitted: either may echo
    # authorization values, query credentials, or credential-bearing upstream URLs.
    return _BatchError("http", _bounded(f"embedding endpoint returned HTTP {response.status_code}"))


def _normalized_vectors(
    response: httpx.Response,
    input_count: int,
    expected_dimensions: int | None,
) -> tuple[tuple[np.ndarray, ...], int]:
    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise _InvalidResponseError("embedding response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise _InvalidResponseError("embedding response must be an object")
    data: Any = payload.get("data")
    if not isinstance(data, list) or len(data) != input_count:
        raise _InvalidResponseError("embedding response count did not match request")

    by_index: dict[int, np.ndarray] = {}
    dimensions: int | None = None
    for item in data:
        if not isinstance(item, dict):
            raise _InvalidResponseError("embedding response item must be an object")
        index: Any = item.get("index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < input_count
            or index in by_index
        ):
            raise _InvalidResponseError("embedding response indexes were invalid")
        raw_vector: Any = item.get("embedding")
        if not isinstance(raw_vector, list) or not raw_vector:
            raise _InvalidResponseError("embedding vector was invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_vector
        ):
            raise _InvalidResponseError("embedding vector was invalid")
        try:
            vector = np.asarray(raw_vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _InvalidResponseError("embedding vector was invalid") from exc
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise _InvalidResponseError("embedding vector contained invalid values")
        item_dimensions = int(vector.size)
        if dimensions is None:
            dimensions = item_dimensions
        elif dimensions != item_dimensions:
            raise _InvalidResponseError("embedding response contained mixed dimensions")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise _InvalidResponseError("embedding vector must have non-zero finite norm")
        vector = np.asarray(vector / norm, dtype=np.float32)
        vector.setflags(write=False)
        by_index[index] = vector

    if dimensions is None or len(by_index) != input_count:
        raise _InvalidResponseError("embedding response was incomplete")
    if expected_dimensions is not None and dimensions != expected_dimensions:
        raise _InvalidResponseError("embedding dimensions changed between batches")
    return tuple(by_index[index] for index in range(input_count)), dimensions


class EmbeddingClient:
    """Call a configured OpenAI-compatible embeddings endpoint.

    API-key values are retained only in a private attribute and are excluded from
    representations and all surfaced diagnostics.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        environ: Mapping[str, str],
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._model = environ.get(config.model_env, "")
        self._api_key = environ.get(config.api_key_env, "") if config.api_key_env else None
        self._owns_http_client = http_client is None
        self._http_client = httpx.Client() if http_client is None else http_client
        self._closed = False
        self._sleep = sleep
        self._encoding = tiktoken.get_encoding(config.tokenizer)
        self._url = f"{str(config.base_url).rstrip('/')}/embeddings"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(endpoint_class={self._config.endpoint_class!r}, "
            f"batch_size={self._config.batch_size})"
        )

    @property
    def model(self) -> str:
        """Return the resolved embedding model identifier without allowing mutation."""
        return self._model

    def close(self) -> None:
        """Close the internally-created HTTP client, if this instance owns it."""
        if self._owns_http_client and not self._closed:
            self._http_client.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _configuration_error(self) -> _BatchError | None:
        if not self._model:
            return _BatchError("configuration", "embedding model is unavailable")
        if self._config.api_key_env and not self._api_key:
            return _BatchError("configuration", "embedding API key is unavailable")
        return None

    def _request(self, texts: Sequence[str]) -> httpx.Response | _BatchError:
        headers = {"authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        for attempt in range(_MAX_ATTEMPTS):
            try:
                payload: dict[str, object] = {
                    "model": self._model,
                    "input": list(texts),
                }
                if self._config.dimensions is not None:
                    payload["dimensions"] = self._config.dimensions
                response = self._http_client.post(
                    self._url,
                    json=payload,
                    headers=headers,
                )
            except httpx.TransportError:
                if attempt == _MAX_ATTEMPTS - 1:
                    return _BatchError("transport", "embedding endpoint is unavailable")
                self._sleep(_BACKOFF_SECONDS[attempt])
                continue
            except httpx.RequestError:
                return _BatchError("transport", "embedding endpoint is unavailable")

            retryable = response.status_code == 429 or 500 <= response.status_code <= 599
            if retryable and attempt < _MAX_ATTEMPTS - 1:
                retry_after = _retry_after(response)
                self._sleep(retry_after if retry_after is not None else _BACKOFF_SECONDS[attempt])
                continue
            if response.is_error:
                return _failure_for_response(response)
            return response
        return _BatchError("transport", "embedding endpoint is unavailable")

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed texts, degrading failed inputs instead of raising."""
        if not texts:
            return EmbeddingBatch((), (), None)

        configuration_error = self._configuration_error()
        if configuration_error is not None:
            config_failures = tuple(
                EmbeddingFailure(index, configuration_error.category, configuration_error.message)
                for index in range(len(texts))
            )
            return EmbeddingBatch((), config_failures, None)

        plan = plan_batches(
            texts,
            max_items=self._config.batch_size,
            max_tokens=self._config.max_batch_tokens,
            count_tokens=lambda text: len(self._encoding.encode_ordinary(text)),
        )
        vectors: list[np.ndarray] = []
        failures = [
            EmbeddingFailure(
                index,
                "input_too_large",
                "embedding input exceeded request token budget",
            )
            for index in plan.oversized_indexes
        ]
        dimensions = self._config.dimensions
        for planned_batch in plan.batches:
            batch = tuple(item.text for item in planned_batch)
            response = self._request(batch)
            if isinstance(response, _BatchError):
                failures.extend(
                    EmbeddingFailure(item.index, response.category, response.message)
                    for item in planned_batch
                )
                continue
            try:
                batch_vectors, batch_dimensions = _normalized_vectors(
                    response, len(batch), dimensions
                )
            except _InvalidResponseError as exc:
                category = (
                    "dimension_mismatch"
                    if str(exc) == "embedding dimensions changed between batches"
                    else "invalid_response"
                )
                message = _bounded(str(exc))
                failures.extend(
                    EmbeddingFailure(item.index, category, message) for item in planned_batch
                )
                continue
            vectors.extend(batch_vectors)
            dimensions = batch_dimensions

        failures.sort(key=lambda failure: failure.index)
        return EmbeddingBatch(tuple(vectors), tuple(failures), dimensions)

    def probe(self) -> ModelProbe:
        """Probe model connectivity and vector width without surfacing probe content."""
        result = self.embed(["vault-rag model connectivity probe"])
        if result.failures or not result.vectors or result.dimensions is None:
            category = result.failures[0].category if result.failures else "unavailable"
            raise SemanticUnavailableError(
                "embedding model probe unavailable",
                details={"category": category},
            )
        return ModelProbe(model=self._model, dimensions=result.dimensions)

    def embed_query(self, text: str, expected_dimensions: int) -> np.ndarray:
        """Embed a query or raise the typed semantic-degradation signal."""
        if isinstance(expected_dimensions, bool) or expected_dimensions < 1:
            raise ValueError("expected_dimensions must be positive")
        result = self.embed([text])
        if result.failures or not result.vectors:
            category = result.failures[0].category if result.failures else "unavailable"
            raise SemanticUnavailableError(
                "query embedding unavailable",
                details={"category": category},
            )
        vector = result.vectors[0]
        actual_dimensions = int(vector.size)
        if actual_dimensions != expected_dimensions:
            raise SemanticUnavailableError(
                "query embedding dimensions do not match the active index",
                details={
                    "expected_dimensions": expected_dimensions,
                    "actual_dimensions": actual_dimensions,
                },
            )
        return vector
