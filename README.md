# Obsidian Vault RAG

Search an Obsidian vault from your terminal and return note excerpts with file
and line citations. Keyword search runs locally with SQLite. Optional embeddings
add semantic search through an OpenAI-compatible provider.

This is a standalone CLI, not an Obsidian plugin or a chatbot. It reads your notes
without editing them. You do not need Docker, a database server, or an API key
for local keyword search.

## Install from the repo

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
These commands use a POSIX shell on macOS or Linux. On Windows, use WSL2;
native Windows is not tested. uv can download Python 3.12 if needed.

```bash
git clone https://github.com/bmccarn/obsidian-vault-rag.git
cd obsidian-vault-rag
uv python install 3.12
uv tool install --python 3.12 .
vault-rag version --json
```

The command is `vault-rag`, even though the repository is `obsidian-vault-rag`.
If your shell cannot find it, run `uv tool update-shell` and open a new terminal.
You can run the installed command from any directory. Installation and the first
index run may download dependencies and tokenizer data; subsequent lexical
indexing and search do not send note or query text to a provider.

To try synthetic notes before using your own vault, run this from the checkout:

```bash
bash examples/lexical-demo.sh
```

Expect a search hit for `note.md` with a `vault://demo/note.md#L...` citation,
followed by the note text. The script prints the temporary directory it keeps.

## Search your vault

### 1. Select files

Create `.vault-rag.toml` in your vault's root directory. If it already exists,
review it instead of replacing it. This example indexes Markdown and excludes
`private/` as well as Obsidian and Git internals:

```toml
schema_version = 1
id = "my-vault"
egress_policy = "local-only"
include = ["**/*.md"]
exclude = [".git/**", ".obsidian/**", "private/**"]
```

The manifest selects files. It does not contain your machine's path or secrets.
Review the include/exclude patterns before indexing personal notes.

### 2. Configure this machine

Create the configuration directory:

```bash
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/vault-rag"
```

Save the following as `config.toml` in that directory. Replace the example path
with your vault's absolute path. Keep the quotes if the path contains spaces.
If you already have a registry, merge the vault and profile entries instead of
replacing your configuration.

```toml
[vaults.my-vault]
path = "/absolute/path/to/your/vault"

[profiles.personal]
vaults = ["my-vault"]

[embedding]
base_url = "https://embedding.invalid/v1"
model_env = "VAULT_RAG_EMBEDDING_MODEL"
endpoint_class = "remote"
```

The current configuration schema requires an embedding section even for keyword
search. This placeholder is not a service to install. The `local-only` manifest
blocks the `remote` route, so indexing cannot call it. Do not change those two
settings unless you intend to enable embeddings.

```bash
chmod 600 "${XDG_CONFIG_HOME:-$HOME/.config}/vault-rag/config.toml"
export VAULT_RAG_EMBEDDING_MODEL=disabled
```

The model variable must be set even when embeddings are disabled. Add that
`export` to your shell startup file, such as `~/.zshrc` or `~/.bashrc`, or set it
again in each new terminal.

### 3. Index and search

```bash
vault-rag index --profile personal --json
vault-rag search --profile personal "deployment" --mode lexical --limit 5 --json
```

Replace `deployment` with a word in your notes. Results include matching text,
relative paths, line numbers, and citations. No matches returns an empty result
list. Use a returned path and line range to read the source:

```bash
# Replace the path and line range with values from your search result.
vault-rag read --profile personal "notes/example.md" --start-line 1 --end-line 10 --json
vault-rag status --profile personal --json
```

`personal` is the profile defined above. Commands require an explicit profile;
they never search all registered vaults. The lexical-only setup reports semantic
retrieval as disabled by policy. That is expected, not an embedding outage.

Run `index` again after editing, adding, or deleting notes. There is no file
watcher. Indexing updates changed files; it does not rewrite your vault.
The local index is at `${XDG_DATA_HOME:-$HOME/.local/share}/vault-rag/index.sqlite3`
and contains note text. Protect it like the original vault.

## Optional semantic search

Keyword search matches words and identifiers. Dense search matches embedding
vectors; hybrid search combines both. To enable them, follow the
[embedding setup](docs/configuration.md#enable-embeddings).
Remote providers receive selected note text and queries and may charge for use.
A local proxy can also forward text to a remote provider.

## Agents and shared service

An agent with terminal access can use the CLI workflow above. HTTP and MCP access
require a separate running service; installing the CLI does not start one.

- [Docker demo](docs/docker-demo.md): run the API and workers with synthetic notes.
- [MCP guide](docs/mcp.md): connect clients to an existing private service.
- [Kubernetes deployment](docs/postgresql-kubernetes-deployment.md): set up a
  PostgreSQL-backed service for Git-hosted vaults.

The service has no application authentication or per-user permissions. Every
client that can reach it can select its profiles. Keep it on a trusted private
network. See [SECURITY.md](SECURITY.md).

## Update or uninstall

From your checkout:

```bash
git pull --ff-only
uv tool install --python 3.12 --force .
vault-rag version --json
```

If a version reports `rebuild_required`, run `vault-rag index --profile personal
--rebuild --json`. With embeddings enabled, `--rebuild` re-embeds the entire corpus
and can incur provider charges. Review configuration changes before rebuilding.

To remove the executable:

```bash
uv tool uninstall vault-rag
```

Uninstalling leaves your vault, manifest, registry, and index in place. Remove
those configuration/index files separately only if you no longer need them.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `vault-rag: command not found` | Run `uv tool update-shell`, then open a new terminal. |
| Registry or manifest missing | Check the two file locations in the setup above. |
| Missing model variable | Run `export VAULT_RAG_EMBEDDING_MODEL=disabled` for the lexical setup. |
| Empty search results | Check include/exclude rules, run `index`, and try a word present in a selected file. |
| `stale_source` when reading | The file changed after indexing. Run `index`, search again, and use the new result. |
| Semantic retrieval disabled | Expected for this lexical setup. Use `--mode lexical`. |

`vault-rag doctor --profile personal --json` checks configuration and index health.
Use `vault-rag --help` or `vault-rag COMMAND --help` for command options.

## Exit codes

| Exit code | Meaning |
| --- | --- |
| `0` | Completed. |
| `1` | Internal or storage failure. |
| `2` | Invalid configuration/input, parse error, or rebuild required. |
| `3` | Path/profile boundary violation or stale source. |
| `4` | Embeddings pending or semantic retrieval unavailable. |
| `5` | Evaluation completed but did not meet its thresholds. |

For `index`, exit code 4 means lexical indexing completed but some embeddings are
pending. Inspect the JSON report; later indexing retries them. Policy-disabled
embeddings are not pending.

## Reference

- [Configuration](docs/configuration.md): profiles, file selection, egress, and service settings.
- [Retrieval](docs/retrieval.md): ranking, filters, citations, and evaluation.
- [Architecture](docs/architecture.md): local and shared data flows.
- [Contributing](CONTRIBUTING.md): development and test commands.
- [Changelog](CHANGELOG.md): release changes.

## License

[MIT](LICENSE). You can use, modify, and redistribute this software, including
commercially, under the license terms.
