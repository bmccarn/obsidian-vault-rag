"""Strict FastAPI transport for the shared retrieval service."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from time import perf_counter
from typing import Annotated, Any, Protocol, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from mcp.server.transport_security import TransportSecuritySettings
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vault_rag.domain import JsonValue, LineRange, SourceKind
from vault_rag.errors import ServiceBusyError, VaultRagError
from vault_rag.presentation import read_payload, search_payload
from vault_rag.retrieval import ReadRequest, SearchFilters, SearchMode, SearchRequest

from .config import MCPTransportConfig
from .facade import VaultService
from .mcp import build_mcp_server
from .observability import ServiceMetrics, bounded_http_endpoint

_MAX_PROFILE_CHARACTERS = 200
_MAX_PATH_CHARACTERS = 2_000
_MAX_QUERY_CHARACTERS = 10_000
_MAX_VAULT_IDS = 100
_MAX_FRONTMATTER_FIELDS = 100
_MAX_FRONTMATTER_BYTES = 20_000
_MAX_HEADING_PARTS = 50
_MAX_REQUEST_BYTES = 1024 * 1024

ProfileName = Annotated[str, Field(min_length=1, max_length=_MAX_PROFILE_CHARACTERS)]
PathText = Annotated[str, Field(min_length=1, max_length=_MAX_PATH_CHARACTERS)]
HeadingText = Annotated[str, Field(min_length=1, max_length=_MAX_PATH_CHARACTERS)]
HeadingParts = Annotated[tuple[HeadingText, ...], Field(max_length=_MAX_HEADING_PARTS)]


class EventPort(Protocol):
    """The bounded event surface used to correlate generic server failures."""

    def emit(self, event: str, **bounded_fields: JsonValue) -> None: ...


class RuntimePort(Protocol):
    """The lifecycle and public operations required by this transport."""

    @property
    def service(self) -> VaultService: ...

    @property
    def metrics(self) -> ServiceMetrics: ...

    @property
    def events(self) -> EventPort: ...

    def start(self) -> None: ...

    def close(self) -> None: ...


class _RequestTooLarge(HTTPException):
    """Signal that a streaming request crossed the raw-byte limit."""

    def __init__(self) -> None:
        super().__init__(status_code=413)


class _BoundedHttpMiddleware:
    """Reject oversized bodies and keep unexpected failures out of server logs."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        runtime: RuntimePort,
        max_request_bytes: int,
    ) -> None:
        self._app = app
        self._runtime = runtime
        self._max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        started = perf_counter()
        status_code = 500

        async def observed_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            for name, value in scope["headers"]:
                if name == b"content-length":
                    with suppress(ValueError):
                        if int(value) > self._max_request_bytes:
                            await self._request_too_large(scope, receive, observed_send)
                            return

            received_bytes = 0

            async def bounded_receive() -> Message:
                nonlocal received_bytes
                message = await receive()
                if message["type"] == "http.request":
                    received_bytes += len(message.get("body", b""))
                    if received_bytes > self._max_request_bytes:
                        raise _RequestTooLarge
                return message

            try:
                await self._app(scope, bounded_receive, observed_send)
            except _RequestTooLarge:
                await self._request_too_large(scope, receive, observed_send)
            except Exception:
                correlation_id = str(uuid4())
                with suppress(Exception):
                    self._runtime.events.emit(
                        "http_internal_error",
                        correlation_id=correlation_id,
                        outcome="failure",
                    )
                response = JSONResponse(
                    status_code=500,
                    content={"error": {"code": "internal_error", "correlation_id": correlation_id}},
                )
                await response(scope, receive, observed_send)
        finally:
            state = scope.get("state")
            mode = state.get("vault_rag_metric_mode", "none") if isinstance(state, dict) else "none"
            endpoint = scope["path"]
            outcome = "success" if 200 <= status_code < 400 else "failure"
            elapsed_ms = (perf_counter() - started) * 1000
            with suppress(Exception):
                self._runtime.metrics.observe_http(endpoint, mode, outcome, elapsed_ms)
            with suppress(Exception):
                self._runtime.events.emit(
                    "http_request",
                    endpoint=bounded_http_endpoint(endpoint),
                    mode=mode,
                    outcome=outcome,
                    elapsed_ms=elapsed_ms,
                )

    @staticmethod
    async def _request_too_large(scope: Scope, receive: Receive, send: Send) -> None:
        response = _error_response(413, "request_too_large")
        await response(scope, receive, send)


class SearchFiltersInput(BaseModel):
    """Bounded, immutable HTTP search filter input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vault_ids: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAX_PROFILE_CHARACTERS)], ...
    ] = Field(default=(), max_length=_MAX_VAULT_IDS)
    path_prefix: Annotated[str, Field(min_length=1, max_length=_MAX_PATH_CHARACTERS)] | None = None
    source_kind: SourceKind | None = None
    frontmatter: dict[Annotated[str, Field(min_length=1, max_length=200)], Any] = Field(
        default_factory=dict, max_length=_MAX_FRONTMATTER_FIELDS
    )

    @field_validator("frontmatter")
    @classmethod
    def validate_frontmatter(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Keep arbitrary JSON filter values within the retrieval-size bound."""
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("frontmatter must contain JSON values") from exc
        if len(encoded.encode()) > _MAX_FRONTMATTER_BYTES:
            raise ValueError("frontmatter is too large")
        return value


class SearchInput(BaseModel):
    """Bounded, immutable HTTP search request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: ProfileName
    query: Annotated[str, Field(min_length=1, max_length=_MAX_QUERY_CHARACTERS)]
    filters: SearchFiltersInput = Field(default_factory=SearchFiltersInput)
    limit: int = Field(default=10, ge=1, le=100)
    mode: SearchMode = SearchMode.HYBRID


class ReadInput(BaseModel):
    """Bounded, immutable HTTP source-read request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: ProfileName
    path: PathText
    vault_id: Annotated[str, Field(min_length=1, max_length=_MAX_PROFILE_CHARACTERS)] | None = None
    heading: HeadingText | HeadingParts | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    expected_source_hash: str | None = Field(default=None, alias="expected_source_hash")

    @model_validator(mode="after")
    def validate_range(self) -> ReadInput:
        """Require a complete line interval when callers select lines."""
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("start_line and end_line must be supplied together")
        return self


def _error_response(
    status_code: int, code: str, *, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code}}, headers=headers)


def _vault_rag_status(error: VaultRagError) -> int:
    if error.code == "security_error":
        return 403
    if error.code in {"rebuild_required", "stale_source"}:
        return 409
    if error.code == "storage_error":
        return 503
    return 400


def create_app(
    runtime: RuntimePort,
    *,
    mcp_config: MCPTransportConfig | None = None,
) -> FastAPI:
    """Create an app whose lifespan owns the supplied, already-built runtime."""

    effective_mcp = mcp_config
    if effective_mcp is None:
        candidate = getattr(runtime, "mcp_config", None)
        effective_mcp = (
            candidate if isinstance(candidate, MCPTransportConfig) else MCPTransportConfig()
        )
    mcp_server = build_mcp_server(runtime) if effective_mcp.enabled else None
    mcp_http_app = None
    if mcp_server is not None:
        mcp_http_app = mcp_server.streamable_http_app(
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=True,
            max_request_body_size=_MAX_REQUEST_BYTES,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=list(effective_mcp.allowed_hosts),
                allowed_origins=list(effective_mcp.allowed_origins),
            ),
            host="0.0.0.0",
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            runtime.start()
            if mcp_server is None:
                yield
            else:
                async with mcp_server.session_manager.run():
                    yield
        finally:
            runtime.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        _BoundedHttpMiddleware,
        runtime=runtime,
        max_request_bytes=_MAX_REQUEST_BYTES,
    )

    @app.exception_handler(_RequestTooLarge)
    def request_too_large(_: Request, __: _RequestTooLarge) -> JSONResponse:
        return _error_response(413, "request_too_large")

    @app.exception_handler(ServiceBusyError)
    def service_busy(_: Request, __: ServiceBusyError) -> JSONResponse:
        return _error_response(503, "service_busy", headers={"Retry-After": "1"})

    @app.exception_handler(VaultRagError)
    def vault_rag_error(_: Request, error: VaultRagError) -> JSONResponse:
        status = _vault_rag_status(error)
        headers = {"Retry-After": "1"} if status == 503 else None
        return _error_response(status, error.code, headers=headers)

    @app.exception_handler(ValueError)
    def invalid_request(_: Request, __: ValueError) -> JSONResponse:
        return _error_response(400, "invalid_request")

    @app.exception_handler(RequestValidationError)
    def invalid_body(_: Request, __: RequestValidationError) -> JSONResponse:
        return _error_response(422, "validation_error")

    @app.post("/v1/search")
    def search(input: SearchInput, request: Request) -> dict[str, Any]:
        request.scope.setdefault("state", {})["vault_rag_metric_mode"] = input.mode.value
        filters = SearchFilters(
            vault_ids=input.filters.vault_ids,
            path_prefix=input.filters.path_prefix,
            source_kind=input.filters.source_kind,
            frontmatter=cast(dict[str, JsonValue], input.filters.frontmatter),
        )
        result = runtime.service.search(
            input.profile,
            SearchRequest(
                query=input.query,
                filters=filters,
                limit=input.limit,
                mode=input.mode,
            ),
        )
        return search_payload(result.profile, result.response, result.commits)

    @app.post("/v1/read")
    def read(request: ReadInput) -> dict[str, Any]:
        if request.start_line is None:
            lines = None
        else:
            assert request.end_line is not None
            lines = LineRange(request.start_line, request.end_line)
        result = runtime.service.read(
            request.profile,
            ReadRequest(
                path=request.path,
                vault_id=request.vault_id,
                heading=request.heading,
                lines=lines,
                expected_source_hash=request.expected_source_hash,
            ),
        )
        return read_payload(result.profile, result.response, result.commits)

    @app.get("/v1/profiles")
    def profiles() -> dict[str, Any]:
        return runtime.service.profiles()

    @app.get("/v1/status")
    def status(
        profile: Annotated[str, Query(min_length=1, max_length=_MAX_PROFILE_CHARACTERS)],
    ) -> dict[str, Any]:
        return runtime.service.status(profile)

    @app.post("/v1/admin/sync", status_code=202)
    def sync() -> dict[str, bool]:
        runtime.service.request_sync()
        return {"accepted": True}

    @app.get("/health/live")
    def liveness() -> dict[str, bool]:
        return {"live": True}

    @app.get("/health/ready")
    def readiness() -> JSONResponse:
        ready = runtime.service.ready()
        return JSONResponse(
            status_code=200 if ready.ready else 503,
            content={"ready": ready.ready, "profiles": dict(ready.profiles)},
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(
            content=runtime.metrics.render(),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    if mcp_http_app is not None:

        @app.api_route("/mcp", methods=["GET", "HEAD"], include_in_schema=False)
        def mcp_receive_stream_disabled() -> JSONResponse:
            """Reject the optional stateful receive stream for this JSON-only server."""

            return _error_response(
                405,
                "method_not_allowed",
                headers={"Allow": "POST"},
            )

        # Keep every explicit FastAPI route ahead of the catch-all root mount while
        # preserving the SDK's canonical, exact `/mcp` Streamable HTTP path.
        app.mount("/", mcp_http_app)

    return app
