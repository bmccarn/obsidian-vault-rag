from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from hashlib import sha256
from typing import Any

import httpx
import numpy as np
import pytest

from vault_rag.config import EmbeddingConfig
from vault_rag.embedding import (
    EmbeddingClient,
    configuration_fingerprint,
    observed_fingerprint,
)
from vault_rag.errors import SemanticUnavailableError


def config(**overrides: Any) -> EmbeddingConfig:
    values: dict[str, Any] = {
        "base_url": "http://127.0.0.1:4000/v1",
        "api_key_env": "LITELLM_API_KEY",
        "model_env": "EMBEDDING_MODEL",
        "endpoint_class": "local",
        "batch_size": 64,
    }
    values.update(overrides)
    return EmbeddingConfig.model_validate(values)


def embedding_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = "secret",
    client_config: EmbeddingConfig | None = None,
    sleep: Callable[[float], None] = lambda _delay: None,
) -> EmbeddingClient:
    environ = {"EMBEDDING_MODEL": "embed-v1"}
    if api_key is not None:
        environ["LITELLM_API_KEY"] = api_key
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return EmbeddingClient(client_config or config(), environ, http_client=http_client, sleep=sleep)


def response(
    *vectors: list[float],
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        json={
            "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
            "model": "embed-v1",
        },
    )


def test_embed_posts_openai_compatible_request_and_normalizes() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response([3.0, 4.0])

    client = embedding_client(handler, api_key="top-secret-value")
    result = client.embed(["hello"])

    assert requests[0].url == httpx.URL("http://127.0.0.1:4000/v1/embeddings")
    assert requests[0].headers["authorization"] == "Bearer top-secret-value"
    assert json.loads(requests[0].content) == {"model": "embed-v1", "input": ["hello"]}
    assert result.failures == ()
    assert result.dimensions == 2
    np.testing.assert_allclose(result.vectors[0], np.array([0.6, 0.8], dtype=np.float32))
    assert result.vectors[0].dtype == np.float32
    assert result.vectors[0].flags.writeable is False
    assert "top-secret-value" not in repr(client)
    assert "127.0.0.1" not in repr(client)


def test_configured_dimensions_are_requested_and_enforced() -> None:
    requests: list[httpx.Request] = []

    def valid_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response([3.0, 4.0])

    result = embedding_client(valid_handler, client_config=config(dimensions=2)).embed(["hello"])

    assert json.loads(requests[0].content) == {
        "model": "embed-v1",
        "input": ["hello"],
        "dimensions": 2,
    }
    assert result.dimensions == 2
    assert len(result.vectors) == 1

    mismatch = embedding_client(
        lambda _request: response([1.0, 0.0, 0.0]),
        client_config=config(dimensions=2),
    ).embed(["hello"])
    assert mismatch.vectors == ()
    assert mismatch.failures[0].category == "dimension_mismatch"


def test_429_and_5xx_retry_then_succeed() -> None:
    statuses = iter((429, 500, 200))
    sleep_calls: list[float] = []

    client = embedding_client(
        lambda _request: response([3.0, 4.0], status=next(statuses)),
        sleep=sleep_calls.append,
    )

    assert len(client.embed(["hello"]).vectors) == 1
    assert sleep_calls == [0.5, 1.0]


@pytest.mark.parametrize("failure", ["timeout", "network", "429", "500"])
def test_retryable_failures_stop_after_three_attempts(failure: str) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("contains secret", request=request)
        if failure == "network":
            raise httpx.ConnectError("authorization=Bearer secret", request=request)
        return response(status=int(failure))

    result = embedding_client(handler, sleep=sleeps.append).embed(["a", "b"])

    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert result.vectors == ()
    assert [item.index for item in result.failures] == [0, 1]
    assert all("secret" not in item.message for item in result.failures)
    assert all(len(item.message) <= 1_000 for item in result.failures)


def test_retry_after_is_honored_and_bounded() -> None:
    replies: Iterator[httpx.Response] = iter(
        (
            response(status=429, headers={"Retry-After": "7"}),
            response(status=503, headers={"Retry-After": "99"}),
            response([1.0, 0.0]),
        )
    )
    sleeps: list[float] = []

    result = embedding_client(lambda _request: next(replies), sleep=sleeps.append).embed(["x"])

    assert len(result.vectors) == 1
    assert sleeps == [7.0, 10.0]


def test_other_4xx_is_not_retried() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, text="Bearer secret and https://user:password@example.test")

    result = embedding_client(handler).embed(["hello"])

    assert attempts == 1
    assert result.failures[0].category == "http"
    assert "secret" not in result.failures[0].message
    assert "password" not in result.failures[0].message


def test_non_transport_request_error_degrades_without_retry_or_disclosure() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.DecodingError("raw authorization secret", request=request)

    result = embedding_client(handler, sleep=sleeps.append).embed(["a", "b"])

    assert attempts == 1
    assert sleeps == []
    assert result.vectors == ()
    assert [failure.index for failure in result.failures] == [0, 1]
    assert all(failure.category == "transport" for failure in result.failures)
    assert all(
        failure.message == "embedding endpoint is unavailable" for failure in result.failures
    )


def test_response_indexes_are_reordered_to_input_order() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ]
            },
        )

    result = embedding_client(handler).embed(["first", "second"])

    np.testing.assert_array_equal(result.vectors[0], [1.0, 0.0])
    np.testing.assert_array_equal(result.vectors[1], [0.0, 1.0])


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ]
        },
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [1.0, 0.0, 0.0]},
            ]
        },
        {
            "data": [
                {"index": 0, "embedding": [float("nan"), 1.0]},
                {"index": 1, "embedding": [1.0, 0.0]},
            ]
        },
        {
            "data": [
                {"index": 0, "embedding": [float("inf"), 1.0]},
                {"index": 1, "embedding": [1.0, 0.0]},
            ]
        },
        {
            "data": [
                {"index": 0, "embedding": [0.0, 0.0]},
                {"index": 1, "embedding": [1.0, 0.0]},
            ]
        },
        {
            "data": [
                {"index": 0, "embedding": [True, 0.0]},
                {"index": 1, "embedding": [1.0, 0.0]},
            ]
        },
    ],
    ids=[
        "wrong-count",
        "duplicate-index",
        "mixed-dimensions",
        "nan",
        "infinity",
        "zero",
        "boolean",
    ],
)
def test_invalid_response_fails_every_input_without_raising(payload: object) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(payload),
            headers={"content-type": "application/json"},
        )

    result = embedding_client(handler).embed(["a", "b"])

    assert result.vectors == ()
    assert result.dimensions is None
    assert [failure.index for failure in result.failures] == [0, 1]
    assert all(failure.category == "invalid_response" for failure in result.failures)


def test_missing_api_key_environment_degrades_without_disclosure_or_request() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return response([1.0, 0.0])

    client = embedding_client(handler, api_key=None)
    result = client.embed(["hello"])

    assert called is False
    assert result.failures[0].category == "configuration"
    assert result.failures[0].message == "embedding API key is unavailable"
    assert "LITELLM_API_KEY" not in repr(client)


def test_token_budget_skips_oversized_input_and_embeds_neighbors() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        requests.append(inputs)
        return response(*([[1.0, 0.0]] * len(inputs)))

    budgeted = config(
        target_min_tokens=1,
        target_max_tokens=2,
        overlap_tokens=0,
        max_input_tokens=3,
        max_batch_tokens=3,
    )
    result = embedding_client(handler, client_config=budgeted).embed(
        ["one two three four five", "ok"]
    )

    assert requests == [["ok"]]
    assert len(result.vectors) == 1
    assert [(failure.index, failure.category) for failure in result.failures] == [
        (0, "input_too_large")
    ]


def test_batch_splitting_preserves_global_failure_indexes_and_dimensions() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        inputs = json.loads(request.content)["input"]
        if calls == 2:
            return response(status=400)
        return response(*([[1.0, 0.0]] * len(inputs)))

    client = embedding_client(handler, client_config=config(batch_size=2))
    result = client.embed(["a", "b", "c", "d", "e"])

    assert calls == 3
    assert len(result.vectors) == 3
    assert [failure.index for failure in result.failures] == [2, 3]
    assert result.dimensions == 2


def test_dimensions_must_match_across_batches() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response([1.0, 0.0] if calls == 1 else [1.0, 0.0, 0.0])

    result = embedding_client(handler, client_config=config(batch_size=1)).embed(["a", "b"])

    assert len(result.vectors) == 1
    assert result.failures[0].index == 1
    assert result.failures[0].category == "dimension_mismatch"
    assert result.dimensions == 2


def test_embed_query_raises_typed_bounded_error_on_failure() -> None:
    client = embedding_client(lambda _request: response(status=503))

    with pytest.raises(SemanticUnavailableError, match="query embedding unavailable") as error:
        client.embed_query("query", expected_dimensions=2)

    assert len(error.value.message) <= 1_000
    assert "secret" not in error.value.message


def test_embed_query_rejects_dimension_mismatch() -> None:
    client = embedding_client(lambda _request: response([1.0, 0.0, 0.0]))

    with pytest.raises(SemanticUnavailableError, match="dimensions") as error:
        client.embed_query("query", expected_dimensions=2)

    assert error.value.details == {"expected_dimensions": 2, "actual_dimensions": 3}


def test_empty_embed_does_not_make_a_request() -> None:
    client = embedding_client(lambda _request: pytest.fail("unexpected request"))

    assert client.embed([]).vectors == ()
    assert client.embed([]).failures == ()


def test_model_property_exposes_resolved_model_read_only() -> None:
    client = embedding_client(lambda _request: response([1.0, 0.0]))

    assert client.model == "embed-v1"
    assert EmbeddingClient.model.fset is None


def test_close_is_idempotent_and_closes_only_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_http_client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: response([1.0, 0.0]))
    )
    monkeypatch.setattr(
        "vault_rag.embedding.client.httpx.Client",
        lambda: owned_http_client,
    )

    with EmbeddingClient(
        config(),
        {"EMBEDDING_MODEL": "embed-v1", "LITELLM_API_KEY": "secret"},
    ) as client:
        assert client.model == "embed-v1"

    assert owned_http_client.is_closed is True
    client.close()


def test_close_leaves_injected_http_client_open() -> None:
    injected = httpx.Client(transport=httpx.MockTransport(lambda _request: response([1.0, 0.0])))
    client = EmbeddingClient(
        config(),
        {"EMBEDDING_MODEL": "embed-v1", "LITELLM_API_KEY": "secret"},
        http_client=injected,
    )

    client.close()

    assert injected.is_closed is False
    injected.close()


def test_fingerprints_are_stable_non_secret_and_observe_dimensions() -> None:
    first = config(
        base_url="https://user:password@one.example/v1",
        api_key_env="FIRST_API_KEY",
    )
    second = config(base_url="https://two.example/v1", api_key_env="SECOND_API_KEY")
    payload = {
        "model": "embed-v1",
        "endpoint_class": "local",
        "tokenizer": "cl100k_base",
        "target_min_tokens": 500,
        "target_max_tokens": 900,
        "overlap_tokens": 80,
        "max_input_tokens": 8191,
        "revision": "1",
        "dimensions": None,
        "normalization": "l2-f32-v1",
    }
    expected = (
        "sha256:"
        + sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )

    first_fingerprint = configuration_fingerprint(first, "embed-v1")
    second_fingerprint = configuration_fingerprint(second, "embed-v1")
    changed_model_fingerprint = configuration_fingerprint(first, "embed-v2")

    assert first_fingerprint == expected
    assert second_fingerprint == expected
    assert "password" not in expected
    assert configuration_fingerprint(config(model_env="OTHER_MODEL"), "embed-v1") == expected
    assert (
        configuration_fingerprint(
            config(base_url="https://remote.example/v1", endpoint_class="remote"), "embed-v1"
        )
        != expected
    )
    assert configuration_fingerprint(config(max_batch_tokens=200_000), "embed-v1") == expected
    assert configuration_fingerprint(config(revision="2"), "embed-v1") != expected
    assert configuration_fingerprint(config(dimensions=1_024), "embed-v1") != expected
    assert changed_model_fingerprint != expected
    assert observed_fingerprint(expected, 2) == observed_fingerprint(expected, 2)
    assert observed_fingerprint(expected, 2) != observed_fingerprint(expected, 3)
    assert observed_fingerprint(changed_model_fingerprint, 2) != observed_fingerprint(expected, 2)


@pytest.mark.parametrize("resolved_model", ["", "   "])
def test_configuration_fingerprint_rejects_empty_model(resolved_model: str) -> None:
    with pytest.raises(ValueError, match="model"):
        configuration_fingerprint(config(), resolved_model)


@pytest.mark.parametrize("dimensions", [0, -1, True])
def test_observed_fingerprint_rejects_invalid_dimensions(dimensions: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        observed_fingerprint("sha256:abc", dimensions)
