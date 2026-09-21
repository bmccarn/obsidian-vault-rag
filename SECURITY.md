# Security policy

## Report privately

Use [GitHub private vulnerability reporting](https://github.com/bmccarn/obsidian-vault-rag/security/advisories/new).
Do not publish exploit details, credentials or private notes in an issue. Include
the affected commit/version, a synthetic reproducer, expected/actual behavior and
impact. There is no guaranteed response SLA. Never attach live vaults or tokens.

## Supported versions

The latest main revision and latest 0.1 release are supported on a best-effort
basis. This early release is not an Internet-facing multi-tenant service.

## Trust model

- No application authentication or per-user authorization is implemented. All
  reachable service clients are trusted and can choose configured profiles.
  Profiles enforce retrieval scope, not an identity-based ACL.
- Host/Origin validation mitigates DNS rebinding, not unauthorized users. Put the
  service behind a private trusted network and enforce network access policy.
- Do not publicly expose `/mcp`, the HTTP API, metrics or administrative sync.
- Remote embedding providers receive indexed text and query text. Classify the
  actual destination, including remote upstreams behind local proxies. Use vault
  egress policy and file exclusions deliberately.
- Indexes, vectors, managed checkouts, responses and backups can disclose source
  content. Apply source-equivalent permissions, retention and encryption controls.
- Retrieved notes are untrusted data. Agents must not treat instructions found in
  a note as authority to run commands, reveal secrets or change permissions.
- Use least-privilege Git/database credentials, keep secrets outside Git and
  rotate compromised values. Logs and issue attachments need review too.

The CLI reads notes and writes local configuration/index state. The service
fetches configured repositories and writes managed checkouts/database state; it
never pushes source changes. Read-only retrieval does not make returned content
non-sensitive.
