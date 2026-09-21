"""Stateless MCP 2026-07-28 adapter over the transport-neutral Vault RAG facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Annotated, Any, Protocol, cast
from uuid import uuid4

from mcp.server import CacheHint, MCPServer
from mcp.server.caching import CacheableMethod
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import Annotations, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from vault_rag.domain import JsonValue, LineRange, SourceKind
from vault_rag.errors import ServiceBusyError, VaultRagError
from vault_rag.presentation import read_payload, search_payload
from vault_rag.retrieval import ReadRequest, SearchFilters, SearchMode, SearchRequest

from .facade import VaultService
from .observability import ServiceMetrics

_MAX_PROFILE_CHARACTERS = 200
_MAX_PATH_CHARACTERS = 2_000
_MAX_QUERY_CHARACTERS = 10_000
_MAX_VAULT_IDS = 100
_MAX_FRONTMATTER_FIELDS = 100
_MAX_SEARCH_LIMIT = 25
_SUBSCRIPTIONS_LISTEN_METHOD = "subscriptions/listen"

ProfileName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=_MAX_PROFILE_CHARACTERS,
        description=(
            "Caller-selectable profile returned by vault_profiles. A profile is the complete "
            "server-enforced vault access boundary; never guess a profile or a vault outside it."
        ),
    ),
]
PathText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=_MAX_PATH_CHARACTERS,
        description=(
            "Vault-relative source path exactly as returned by vault_search. This is not a local "
            "filesystem path and cannot escape the configured vault."
        ),
    ),
]
SourceHash = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        description=(
            "Exact source_hash from vault_search. Required to fence the read to the searched "
            "source revision. If it is stale, search again instead of dropping the hash."
        ),
    ),
]


class _OutputModel(BaseModel):
    """Strict base for MCP structured output schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContinuationOutput(_OutputModel):
    """How to continue a bounded source excerpt."""

    remaining_characters: int = Field(ge=0)
    next_offset: int = Field(ge=0)
    next_line: int = Field(ge=1)
    next_character: int = Field(ge=0)


class ScoreOutput(_OutputModel):
    """Lexical, dense, and fused ranking evidence for one search hit."""

    lexical_rank: int | None = Field(default=None, ge=1)
    lexical_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    fused_score: float
    exact_identifier: bool


class RecommendedRead(_OutputModel):
    """Copy-safe arguments for the next hash-fenced vault_read call."""

    profile: str
    vault_id: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    expected_source_hash: str


class SearchHitOutput(_OutputModel):
    """One source-grounded search result and its safe follow-up read arguments."""

    chunk_id: str
    vault_id: str
    path: str
    heading: list[str]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source_hash: str
    citation: str
    text: str
    continuation: ContinuationOutput | None
    metadata: dict[str, Any]
    metadata_truncated: bool
    scores: ScoreOutput
    recommended_read: RecommendedRead


class DegradedOutput(_OutputModel):
    """Whether semantic retrieval fell back and why."""

    semantic_search: bool
    reason: str | None


class CommitOutput(_OutputModel):
    """Commit and immutable revision identity represented by the request snapshot."""

    checkout_sha: str | None
    reconciled_sha: str | None
    sync_degraded_reason: str | None
    fully_reconciled: bool
    revision_id: str | None
    commit_sha: str | None


def _disable_idle_subscription_stream(server: MCPServer[None]) -> None:
    """Remove the SDK's default subscription stream from this static server.

    MCP SDK 2.0 installs ``subscriptions/listen`` automatically and derives the
    modern ``listChanged``/``subscribe`` capability flags from that handler.
    Vault RAG's tools, prompts, and resources change only on a deployment, so a
    resident per-client event stream would consume resources without ever
    delivering a useful event. The SDK has no public opt-out in 2.0.0; keep the
    private compatibility seam isolated here and fail loudly if its contract
    changes during an SDK upgrade.
    """
    lowlevel_server = server._lowlevel_server
    if lowlevel_server.get_request_handler(_SUBSCRIPTIONS_LISTEN_METHOD) is None:
        raise RuntimeError("MCP SDK subscription handler contract changed")
    del lowlevel_server._request_handlers[_SUBSCRIPTIONS_LISTEN_METHOD]


class VaultSearchOutput(_OutputModel):
    """Structured, snapshot-labelled hybrid retrieval result."""

    profile: str
    hits: list[SearchHitOutput]
    degraded: DegradedOutput
    elapsed_ms: float = Field(ge=0)
    index_age_seconds: float | None = Field(default=None, ge=0)
    vault_commits: dict[str, CommitOutput]
    vault_commits_truncated: int = Field(ge=0)
    usage_guidance: list[str]


class VaultReadOutput(_OutputModel):
    """Hash-fenced source content with exact line citation and continuation."""

    profile: str
    vault_id: str
    path: str
    heading: list[str]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source_hash: str
    citation: str
    text: str
    continuation: ContinuationOutput | None
    vault_commits: dict[str, CommitOutput]
    vault_commits_truncated: int = Field(ge=0)
    usage_guidance: list[str]


class ProfileOutput(_OutputModel):
    """One server-defined access profile and its included vaults."""

    name: str
    vaults: list[str]
    vaults_truncated: int = Field(ge=0)


class VaultProfilesOutput(_OutputModel):
    """Every caller-selectable profile; private reconciliation profiles are excluded."""

    profiles: list[ProfileOutput]
    profiles_truncated: int = Field(ge=0)
    usage_guidance: list[str]


class VaultStatusOutput(_OutputModel):
    """Bounded operational status for one profile."""

    profile: str
    ready: bool
    details: dict[str, Any]
    usage_guidance: list[str]


class _EventPort(Protocol):
    """Bounded structured-event surface used by MCP calls."""

    def emit(self, event: str, **bounded_fields: JsonValue) -> None: ...


class MCPRuntimePort(Protocol):
    """Runtime fields used by the transport adapter."""

    @property
    def service(self) -> VaultService: ...

    @property
    def metrics(self) -> ServiceMetrics: ...

    @property
    def events(self) -> _EventPort: ...


_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_ASSISTANT_RESOURCE = Annotations(audience=["assistant"], priority=1.0)
_GUIDE = """# Vault RAG MCP usage guide

This is a private, read-only, server-scoped retrieval service. It exposes indexed source content,
not generated answers. The server enforces every profile's vault allowlist.

## Required workflow

1. Call `vault_profiles` when the intended profile is not already explicit.
2. Call `vault_search` with `mode="hybrid"` unless lexical-only or dense-only diagnosis is
   intentional. Use a precise identifier, title, system name, or question.
3. Inspect `degraded` and `vault_commits`. A semantic fallback can still return useful lexical
   results, but disclose the degraded state when it matters.
4. For source context, copy a hit's complete `recommended_read` object into `vault_read`.
5. Keep `expected_source_hash`. A `stale_source` error means the indexed source changed; call
   `vault_search` again and use the new hash. Never retry by removing the hash.
6. Cite only the returned `citation` string. Do not manufacture a path, line number, commit, or
   quote that the tools did not return.
7. Use `vault_status` for readiness, revision identity, or degradation diagnosis. There is no MCP
   sync, mutation, arbitrary file read, repository configuration, or credential tool.

## Search controls

- `profile`: complete retrieval scope, selected from `vault_profiles`.
- `mode`: `hybrid` is normal; `lexical` is useful for exact wording; `dense` is semantic-only.
- `vault_ids`: optional narrowing only; every ID must already belong to the profile.
- `path_prefix`: optional vault-relative subtree narrowing.
- `source_kind`: optional exact `markdown`, `text`, `log`, or `json` filter.
- `frontmatter`: exact typed equality filters. Strings, booleans, numbers, arrays, and objects are
  distinct; do not stringify typed values.
- `limit`: 1-25. Start with 5 and widen only when recall is insufficient.

## Read controls

`heading` and `start_line`/`end_line` are alternative selectors. Supply both line endpoints or
neither. Search results already provide safe line selectors. Long reads may include `continuation`;
continue from its line/character position rather than guessing an offset.

## Error recovery

- `stale_source`: re-search, then use the replacement hash.
- `service_busy`: retry with bounded backoff.
- `storage_error`: the service is temporarily unavailable; use `vault_status` later.
- `security_error`: the requested profile/vault/path is outside the server-defined boundary.
- `invalid_request` or validation error: correct the arguments using the tool schema.
- `internal_error` with a correlation ID: report that ID to the operator; no secret or raw
  exception is returned.

## Transport behavior

The endpoint is MCP 2026-07-28 Streamable HTTP in stateless JSON mode. It does not create sticky
client sessions or SSE streams. Each request carries its own capability metadata and can be routed
to either API replica. Discovery/list/read results include private cache hints where supported.
"""

_SERVER_INSTRUCTIONS = """Use Vault RAG as a source-grounded retrieval service, never as an answer
generator. Resolve a server-defined profile, search (hybrid by default), inspect degradation and
commit identity, then use each hit's recommended_read arguments for a hash-fenced read. Preserve
expected_source_hash; on stale_source, re-search instead of weakening the read. Cite only returned
citation strings. Never infer access outside the listed profiles. Use vault_status for diagnosis.
No MCP mutation or synchronization operation exists."""


def _package_version() -> str:
    try:
        return version("vault-rag")
    except PackageNotFoundError:  # pragma: no cover - editable installs provide metadata
        return "0.1.0"


def _error_message(error: Exception, correlation_id: str | None = None) -> str:
    if isinstance(error, ServiceBusyError):
        return "service_busy: retry with bounded backoff"
    if isinstance(error, VaultRagError):
        if error.code == "stale_source":
            return "stale_source: the source changed; call vault_search again and keep the new hash"
        if error.code == "storage_error":
            return "storage_error: retrieval storage is temporarily unavailable"
        if error.code == "security_error":
            return "security_error: the requested profile, vault, or path is outside the boundary"
        return f"{error.code}: the request could not be completed"
    if isinstance(error, (TypeError, ValueError)):
        return "invalid_request: correct the arguments using the published tool schema"
    assert correlation_id is not None
    return f"internal_error: report correlation_id={correlation_id} to the operator"


def _invoke[ResultT](
    runtime: MCPRuntimePort, tool: str, operation: Callable[[], ResultT]
) -> ResultT:
    started = perf_counter()
    outcome = "success"
    try:
        return operation()
    except Exception as error:
        outcome = "failure"
        correlation_id = (
            None if isinstance(error, (VaultRagError, TypeError, ValueError)) else str(uuid4())
        )
        if correlation_id is not None:
            runtime.events.emit(
                "mcp_internal_error",
                correlation_id=correlation_id,
                tool=tool,
                outcome="failure",
            )
        raise ToolError(_error_message(error, correlation_id)) from error
    finally:
        elapsed_ms = (perf_counter() - started) * 1000
        runtime.metrics.observe_mcp(tool, outcome, elapsed_ms)
        runtime.events.emit(
            "mcp_tool_call",
            tool=tool,
            outcome=outcome,
            elapsed_ms=elapsed_ms,
        )


def _search_output(
    runtime: MCPRuntimePort, request: SearchRequest, profile: str
) -> VaultSearchOutput:
    envelope = runtime.service.search(profile, request)
    payload = cast(
        dict[str, Any], search_payload(envelope.profile, envelope.response, envelope.commits)
    )
    for hit in payload["hits"]:
        hit["recommended_read"] = {
            "profile": payload["profile"],
            "vault_id": hit["vault_id"],
            "path": hit["path"],
            "start_line": hit["start_line"],
            "end_line": hit["end_line"],
            "expected_source_hash": hit["source_hash"],
        }
    payload["usage_guidance"] = [
        "Use a hit's recommended_read object verbatim when more context is needed.",
        "Preserve expected_source_hash; if stale_source is returned, search again.",
        "Use only returned citation strings in grounded answers.",
    ]
    return VaultSearchOutput.model_validate(payload)


def build_mcp_server(runtime: MCPRuntimePort) -> MCPServer[None]:
    """Register the read-only agent surface over the existing service facade."""
    cache_hints: dict[CacheableMethod, CacheHint] = {
        "server/discover": CacheHint(ttl_ms=3_600_000, scope="private"),
        "tools/list": CacheHint(ttl_ms=3_600_000, scope="private"),
        "resources/list": CacheHint(ttl_ms=3_600_000, scope="private"),
        "resources/templates/list": CacheHint(ttl_ms=3_600_000, scope="private"),
        "prompts/list": CacheHint(ttl_ms=3_600_000, scope="private"),
        "resources/read": CacheHint(ttl_ms=30_000, scope="private"),
    }
    server: MCPServer[None] = MCPServer(
        name="vault-rag",
        title="Vault RAG",
        description=(
            "Private, profile-scoped hybrid retrieval over Git-synchronized vaults with exact "
            "citations, immutable revision identity, and hash-fenced source reads."
        ),
        instructions=_SERVER_INSTRUCTIONS,
        website_url="https://github.com/bmccarn/obsidian-vault-rag",
        version=_package_version(),
        cache_hints=cache_hints,
    )
    _disable_idle_subscription_stream(server)

    @server.tool(
        name="vault_search",
        title="Search approved vaults",
        description=(
            "Search one server-defined profile using hybrid lexical+dense retrieval by default. "
            "Returns bounded source excerpts, exact citations, ranking evidence, degradation "
            "state, "
            "request-snapshot commit identities, and a recommended_read object for every hit. "
            "Call vault_profiles first if the profile is uncertain. Filters only narrow the chosen "
            "profile; they never grant access. Use exact identifiers when available."
        ),
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def vault_search(
        profile: ProfileName,
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=_MAX_QUERY_CHARACTERS,
                description=(
                    "Natural-language question, exact identifier, title, system name, or keywords. "
                    "Include distinctive identifiers verbatim when known."
                ),
            ),
        ],
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=_MAX_SEARCH_LIMIT,
                description="Maximum hits. Start at 5; widen only when recall is insufficient.",
            ),
        ] = 5,
        mode: Annotated[
            SearchMode,
            Field(
                description=(
                    "hybrid is the normal mode; lexical favors exact wording/identifiers; dense is "
                    "semantic-only and may fail closed or degrade when embeddings are unavailable."
                )
            ),
        ] = SearchMode.HYBRID,
        vault_ids: Annotated[
            list[str] | None,
            Field(
                max_length=_MAX_VAULT_IDS,
                description=(
                    "Optional vault IDs to narrow within the selected profile. Obtain IDs from "
                    "vault_profiles; an out-of-profile ID is rejected."
                ),
            ),
        ] = None,
        path_prefix: Annotated[
            str | None,
            Field(
                min_length=1,
                max_length=_MAX_PATH_CHARACTERS,
                description="Optional vault-relative path prefix used only to narrow results.",
            ),
        ] = None,
        source_kind: Annotated[
            SourceKind | None,
            Field(description=("Optional exact source-kind filter: markdown, text, log, or json.")),
        ] = None,
        frontmatter: Annotated[
            dict[str, Any] | None,
            Field(
                max_length=_MAX_FRONTMATTER_FIELDS,
                description=(
                    "Optional exact typed frontmatter equality filters. Preserve JSON types: a "
                    "boolean, number, array, object, and string are not interchangeable."
                ),
            ),
        ] = None,
    ) -> VaultSearchOutput:
        def operation() -> VaultSearchOutput:
            filters = SearchFilters(
                vault_ids=tuple(vault_ids or ()),
                path_prefix=path_prefix,
                source_kind=source_kind,
                frontmatter=cast(Mapping[str, JsonValue], frontmatter or {}),
            )
            return _search_output(
                runtime,
                SearchRequest(query=query, filters=filters, limit=limit, mode=mode),
                profile,
            )

        return _invoke(runtime, "vault_search", operation)

    @server.tool(
        name="vault_read",
        title="Read a hash-fenced vault source",
        description=(
            "Read authoritative source text from one selected profile and vault. Always copy the "
            "recommended_read object returned by vault_search. expected_source_hash is mandatory: "
            "it prevents a citation from silently crossing source revisions. A stale_source error "
            "must be resolved by searching again, never by omitting the hash. Use either heading "
            "or "
            "a complete start_line/end_line pair."
        ),
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def vault_read(
        profile: ProfileName,
        vault_id: Annotated[
            str,
            Field(
                min_length=1,
                max_length=_MAX_PROFILE_CHARACTERS,
                description="Exact vault_id from a search hit; it must belong to profile.",
            ),
        ],
        path: PathText,
        expected_source_hash: SourceHash,
        heading: Annotated[
            str | list[str] | None,
            Field(
                description=(
                    "Optional exact heading text or breadcrumb sequence. Do not combine with line "
                    "selectors. Search-provided line selectors are preferred."
                )
            ),
        ] = None,
        start_line: Annotated[
            int | None,
            Field(ge=1, description="Optional inclusive start line; requires end_line."),
        ] = None,
        end_line: Annotated[
            int | None,
            Field(ge=1, description="Optional inclusive end line; requires start_line."),
        ] = None,
    ) -> VaultReadOutput:
        def operation() -> VaultReadOutput:
            if (start_line is None) != (end_line is None):
                raise ValueError("line range must be complete")
            if heading is not None and start_line is not None:
                raise ValueError("heading and line range are alternatives")
            normalized_heading = tuple(heading) if isinstance(heading, list) else heading
            lines = None
            if start_line is not None:
                assert end_line is not None
                lines = LineRange(start_line, end_line)
            envelope = runtime.service.read(
                profile,
                ReadRequest(
                    path=path,
                    vault_id=vault_id,
                    heading=normalized_heading,
                    lines=lines,
                    expected_source_hash=expected_source_hash,
                ),
            )
            payload = cast(
                dict[str, Any],
                read_payload(envelope.profile, envelope.response, envelope.commits),
            )
            payload["usage_guidance"] = [
                "Quote only returned text and cite only the returned citation string.",
                "If continuation is present, continue from its returned position.",
                "If a later read reports stale_source, search again before citing.",
            ]
            return VaultReadOutput.model_validate(payload)

        return _invoke(runtime, "vault_read", operation)

    @server.tool(
        name="vault_profiles",
        title="List available retrieval profiles",
        description=(
            "List every caller-selectable profile and its server-enforced vault allowlist. Use "
            "this "
            "before search when the intended profile is uncertain. Returned profiles grant no new "
            "access; they describe the service's configured boundaries."
        ),
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def vault_profiles() -> VaultProfilesOutput:
        def operation() -> VaultProfilesOutput:
            payload = cast(dict[str, Any], runtime.service.profiles())
            payload["usage_guidance"] = [
                "Choose the narrowest profile that contains the needed vaults.",
                "Profiles scope retrieval; they are not per-user authorization.",
                "Do not invent profile or vault names not returned here.",
            ]
            return VaultProfilesOutput.model_validate(payload)

        return _invoke(runtime, "vault_profiles", operation)

    @server.tool(
        name="vault_status",
        title="Inspect profile readiness and revision identity",
        description=(
            "Return bounded readiness, index, commit, reconciliation, and degradation status for a "
            "profile. Use for diagnostics or to explain degraded retrieval; do not poll this tool "
            "on every search. Status is read-only and does not trigger synchronization."
        ),
        annotations=_READ_ONLY,
        structured_output=True,
    )
    def vault_status(profile: ProfileName) -> VaultStatusOutput:
        def operation() -> VaultStatusOutput:
            details = cast(dict[str, Any], runtime.service.status(profile))
            ready = bool(runtime.service.ready().profiles.get(profile, False))
            return VaultStatusOutput(
                profile=profile,
                ready=ready,
                details=details,
                usage_guidance=[
                    "A ready=false profile should not be treated as current authoritative context.",
                    "Inspect each vault's commit and semantic degradation before diagnosing "
                    "results.",
                    "This tool cannot sync or mutate a vault; contact the operator for "
                    "remediation.",
                ],
            )

        return _invoke(runtime, "vault_status", operation)

    @server.resource(
        "vault-rag://guide",
        name="vault-rag-guide",
        title="Vault RAG agent usage guide",
        description=(
            "Complete search/read/citation workflow, argument semantics, transport behavior, and "
            "bounded error recovery for agents using this MCP server."
        ),
        mime_type="text/markdown",
        annotations=_ASSISTANT_RESOURCE,
    )
    def guide() -> str:
        return _GUIDE

    @server.resource(
        "vault-rag://profiles",
        name="vault-rag-profiles",
        title="Current Vault RAG profiles",
        description="Live JSON inventory of caller-selectable profiles and included vault IDs.",
        mime_type="application/json",
        annotations=_ASSISTANT_RESOURCE,
    )
    def profiles_resource() -> dict[str, JsonValue]:
        return runtime.service.profiles()

    @server.prompt(
        name="grounded_vault_research",
        title="Grounded vault research",
        description=(
            "Reusable agent workflow for answering a question exclusively from one Vault RAG "
            "profile with hash-fenced reads and exact citations."
        ),
    )
    def grounded_vault_research(profile: str, question: str) -> str:
        return f"""Answer this question using only Vault RAG profile {profile!r}: {question}

Required process:
1. Call vault_search(profile={profile!r}, query={question!r}, mode='hybrid', limit=5).
2. Inspect degradation and commit identities. Widen the query or limit only if needed.
3. Use each relevant hit's recommended_read object verbatim with vault_read.
4. If stale_source occurs, re-search; never remove expected_source_hash.
5. Answer only from returned source text. Cite exact returned citation strings after each claim.
6. State when the indexed vaults do not support an answer; do not fill gaps from memory or the web.
"""

    return server
