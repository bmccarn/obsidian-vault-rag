# Retrieval and freshness

`vault-rag` is a retrieval system, not an answer generator. Search returns
bounded source excerpts and stable citations; read returns current source bytes
only after verifying the indexed hash.

## Sources and provenance

Discovery evaluates manifest include patterns relative to the canonical vault
root, then applies default and manifest excludes. Supported source suffixes are:

- `.md` as Markdown;
- `.txt` as text;
- `.log` as log text; and
- `.json` as line-oriented text.

Markdown parsing preserves YAML frontmatter, the note title, nested heading
breadcrumbs, structural section boundaries, wikilinks, aliases, tags, and
one-based source line ranges. Only manifest-selected frontmatter fields are
carried into indexed chunk metadata. YAML dates and datetimes are canonicalized
to ISO strings; malformed frontmatter remains a bounded parse failure.

Text, log, and JSON inputs are treated as text rather than deeply interpreted
structured documents. Unsupported binary and document formats are not
converted. Symlink handling and canonical path containment fail closed; a
source must remain beneath its registered root.

Excluded paths are skipped before any containment, extension, or decoding
check, so an excluded symlink or excluded binary is never inspected. A file that
an include pattern selects but that fails containment, carries an unsupported
extension, cannot be read, or is not valid UTF-8 receives a bounded per-file diagnostic
in the index report and is skipped; other files continue indexing,
and any previously indexed revision of that file is removed so it cannot be
returned. These per-file problems are reported by the index run, not by
`doctor`: `doctor` reports vault-level state, so a missing or symlinked vault
root and a case-folded path collision remain vault-level failures there.

## Chunks and compatibility

A normal heading section is kept separate from adjacent headings. Embedding
text consists of the vault ID, note title, heading breadcrumb, selected
frontmatter, and source text. Sections above the input limit are split at
structural boundaries where possible, with a normal target of 500–900 tokens,
at most 80 tokens of overlap, and a strict input ceiling of 8,191 tokens by
default. An oversized fenced code block is intentionally indivisible and fails
closed rather than being silently truncated.

Chunk identity is deterministic from vault ID, normalized relative path,
heading breadcrumb, structural ordinal, and the chunker schema version. Its
content hash is separate and determines whether fresh embedding is needed.
Every chunk stores exact one-based line ranges and its source kind. Line
numbers count physical CommonMark lines: a newline, a carriage return, and a
carriage return followed by a newline end a line, and no other Unicode
separator does. A cited range therefore matches what an editor, `sed`, or a
forge permalink shows for the same file. This alignment remains stable when
frontmatter uses mixed line endings: blank frontmatter does not shift later
headings or their citations.

Index compatibility is fingerprinted. Manifest policy, parser schema, chunker
schema and token settings, the resolved model identifier, endpoint class,
tokenizer, vector normalization, and observed vector dimensions protect the
index from silently mixing incompatible output. The embedding route and key are
not included in the embedding configuration fingerprint. A model or schema
mismatch disables dense use until a controlled `index --rebuild` is run.

## Incremental indexing and pending coverage

`index` discovers the complete selected source set and compares content hashes.
It parses and chunks added or changed files, atomically replaces each successful
source revision and its FTS/vector state, and removes deleted revisions. An
unchanged fully embedded scan makes no embedding requests.

A parse failure suppresses a changed source's old active revision so stale text
and invalid citations are not returned. An embedding failure records fresh
lexical chunks as pending without retaining vectors for older text. Pending
chunks remain searchable through FTS and are retried on later reconciliation;
status exposes pending coverage and search exposes a degraded semantic state.

The local SQLite index is disposable. A vault change after indexing is detected
on `read`: the current file hash must match active provenance; a changed hash is
reported as `stale_source`, while malformed or out-of-range reads are reported
separately. Reconcile before relying on a stale index.

## Ranking

Search applies the resolved profile's vault allowlist and any structured filters
before lexical or dense ranking. Filters support bound vault IDs, a POSIX
relative path prefix, source kind, and only frontmatter fields named by at
least one bound vault manifest. Filters are validated and compiled by the
application; they are never raw SQL. The CLI exposes repeatable `--vault`,
`--path-prefix`, `--source-kind`, and repeatable `--frontmatter KEY=JSON`
options.

### Retrieval modes

`search` and `evaluate` accept `--mode lexical`, `--mode dense`, or `--mode
hybrid`; hybrid is the default. Lexical-only mode runs exact recognized-
identifier and BM25 retrieval without making a query-embedding request.
Dense-only mode runs only query embedding and cosine ranking, deliberately
excluding BM25 and exact-identifier shortcuts so it measures semantic retrieval.
Hybrid mode runs both candidate generators and applies normal fusion and exact
priority.

Intentionally selecting lexical-only mode is not semantic degradation. Dense-
only and hybrid requests report degradation when semantic retrieval cannot run.

### Lexical retrieval

SQLite FTS5 supplies BM25-ranked candidates over titles, headings, aliases,
tags, paths, selected metadata, and source text. The query analyzer preserves
meaningful identifier punctuation by emitting each recognized token as one
quoted FTS5 phrase; it does not add normalized alternative spellings.

Recognized complete identifiers receive a priority tier above ordinary fused
results: Jira keys, context-qualified Git hashes, unqualified 7–40 character
hexadecimal Git hashes containing at least one letter, `repository #number` pull
request references, exact vault-relative paths, and configured metadata
identifiers. When the query contains one of those recognized complete
identifiers, an identifier-shaped exact title, path, selected frontmatter, or
alias value is placed in that same priority tier; there is no score boost.
An ordinary free-text title or alias remains under BM25/dense ranking. Bare
numbers and partial free-text tokens do not trigger the exact tier.

### Dense retrieval and fusion

For compatible coverage, the query is embedded with the configured model and
compared by exact cosine similarity against normalized float32 vectors for the
already filtered profile. The implementation loads that profile's compatible
vectors into memory; it does not use an approximate-nearest-neighbor service.

In hybrid mode, lexical and dense ranks are combined using equal-weight
Reciprocal Rank Fusion with `k = 60`. Exact identifier candidates remain above
non-exact fused results; within each tier ordering is deterministic. There is no
cross-encoder or LLM reranker in this phase.

If query embedding is unavailable, rejected by a local-only policy, pending,
or incompatible, lexical results can still be returned. JSON output includes a
machine-readable `degraded` state rather than hiding the missing semantic
coverage.

### Example dimension baseline

An example configuration is `text-embedding-3-large` with `dimensions = 1024`.
This is a starting point, not a published quality or latency guarantee. Keep
`revision` explicit and compare lexical-only, dense-only, and hybrid evaluation
on representative sources before changing the model or dimensions.

## Excerpts, reads, and citations

Search and read excerpts are bounded to 1,800 characters. A truncated result
includes continuation information: remaining characters, next text offset,
next line, and next character position. No result is silently
truncated without that continuation.

A citation has this stable form:

```text
vault://<vault-id>/<relative-path>#L<start>-L<end>
```

Paths are percent-encoded where required. Use `read` to retrieve authoritative
current text before acting on consequential information. For a single-vault
profile, `read` may infer the vault; a multi-vault profile requires `--vault`.
A heading selector and explicit line range are mutually exclusive. Reads reject
paths outside the profile and reject inactive, changed, missing, malformed, or
out-of-range source state.

## Evaluation and targets

`evaluate` runs a bounded profile-bound evaluation document. Before searching,
it verifies every expected source through a current hash-checked read. It then
records only case IDs, expected path identities, ranks, valid citation
identities, counts, and latency—never queries, source bodies, metadata, or
vectors in the report.

Reported metrics are retrieval mode, case count, recall@5, mean reciprocal
rank, identifier rank-1, invalid citation count, degraded cases and reasons, and
p50/p95 local search latency. Expected identities compare both `vault_id` and
path. Omitted vault IDs are accepted only for a single-vault profile.

A report passes only when all gates pass: at least 25 cases, configured recall
and identifier thresholds, zero invalid citations, zero degraded cases, and
local search p95 no slower than 500 ms after vectors load. A dense-only or
hybrid evaluation therefore cannot pass through lexical fallback during an
embedding outage. An unchanged warm reconciliation remains separately targeted
at no slower than five seconds. These are acceptance targets, not claims about
every corpus or embedding route.

The private deployment gate is an operator-owned 25-case corpus through the
centralized HTTP endpoint, with zero degraded cases and p95 below 500 ms. It
also includes two-machine private routing verification. Repository-local
synthetic tests exercise service behavior but cannot verify either private
external criterion.

### PostgreSQL exact-search scale runs

`postgres-scale` is an operator workload, not an application request. It builds
one deterministic synthetic immutable revision, promotes it once, and measures
only exact pgvector cosine searches. Its JSON report contains stable hashed
dataset, revision, query, and result identities plus aggregate latency,
throughput, CPU/RSS, pool, PostgreSQL, index-size, and promotion metrics. It
never contains a source body, query, vector, DSN, or credential. Keep each JSON
report as a deployment artifact for the retention period used by the release
evidence store.

`index_size_bytes` is the nonnegative physical PostgreSQL index-size delta:
all `vault_rag` index bytes immediately after the run minus the same total
immediately before synthetic seeding. It is not a per-vault index total, because
PostgreSQL B-tree pages are physically shared by the schema's index relations.
`pool_waits` is the cumulative Psycopg `requests_queued` delta sampled
immediately before query submission and immediately after the executor drains;
seeding and final aggregate measurement do not contribute to it.

Run it only against the intended PostgreSQL service after `vault-rag db check`;
the selected DSN environment variable is read locally and is never emitted.
The operator matrix is deliberately limited to these twelve commands:

```sh
vault-rag postgres-scale --vectors 10000 --concurrency 1 --json > scale-10000-c1.json
vault-rag postgres-scale --vectors 10000 --concurrency 5 --json > scale-10000-c5.json
vault-rag postgres-scale --vectors 10000 --concurrency 10 --json > scale-10000-c10.json
vault-rag postgres-scale --vectors 10000 --concurrency 25 --json > scale-10000-c25.json
vault-rag postgres-scale --vectors 50000 --concurrency 1 --json > scale-50000-c1.json
vault-rag postgres-scale --vectors 50000 --concurrency 5 --json > scale-50000-c5.json
vault-rag postgres-scale --vectors 50000 --concurrency 10 --json > scale-50000-c10.json
vault-rag postgres-scale --vectors 50000 --concurrency 25 --json > scale-50000-c25.json
vault-rag postgres-scale --vectors 100000 --concurrency 1 --json > scale-100000-c1.json
vault-rag postgres-scale --vectors 100000 --concurrency 5 --json > scale-100000-c5.json
vault-rag postgres-scale --vectors 100000 --concurrency 10 --json > scale-100000-c10.json
vault-rag postgres-scale --vectors 100000 --concurrency 25 --json > scale-100000-c25.json
```

Accept a run only when it reports zero errors, `no_hnsw: true`, and every query
returns its deterministic target as the first exact result with descending
scores. Compare p50/p95/p99, throughput, CPU/RSS, pool waits/connections,
PostgreSQL operation latency/rows, index size, and promotion duration against
the retained baseline. HNSW remains disabled: do not introduce an approximate
index without documented exact-versus-approximate quality and performance
evidence for this matrix.
