# Backups & Restore

archi ships with two small scripts for snapshotting and restoring the
user-facing rows of an archi deployment.  They are intentionally **focused
on data users care about** — conversations, feedback, tool-call history,
audit trails — not on the deploy-time configuration (`static_config`,
`dynamic_config`) or the vector index, which are re-seeded from your
deployment config and source corpus.

## Quick start

```bash
# Snapshot everything user-facing into a single tarball.
python scripts/archi_backup.py --out archi-2026-05-13.tar.gz

# Restore it into another deployment (or the same one).
python scripts/archi_restore.py --in archi-2026-05-13.tar.gz
```

Both scripts read database connection info from the standard `PGHOST`,
`PGPORT`, `PGDATABASE`, `PGUSER`, and `PG_PASSWORD` environment variables.

## What's in a backup

Each backup is a tar.gz (or directory, see below) containing:

| File | Contents |
|---|---|
| `manifest.json` | format version, creation timestamp, schema version from `migration_state`, and a per-table row count |
| `users.jsonl.gz` | one user per line |
| `conversation_metadata.jsonl.gz` | one conversation header per line |
| `conversations.jsonl.gz` | every message |
| `feedback.jsonl.gz` | every feedback row |
| `agent_traces.jsonl.gz` | every agent trace |
| `agent_tool_calls.jsonl.gz` | every tool call |
| `ab_comparisons.jsonl.gz` | every A/B vote |
| `config_audit.jsonl.gz` | dynamic-config change log |
| `user_actions.jsonl.gz` | (if present) user-action audit log |
| `tool_approvals.jsonl.gz` | (if present) MCP-approval decisions |

Encoding conventions:

- Binary columns (encrypted API keys, refresh tokens, ...) are
  base64-encoded under `{"__b64__": "..."}`.
- Timestamps are ISO-8601 under `{"__dt__": "..."}`.

**The backup tool never decrypts anything** — encrypted columns are dumped
as-is.  Your `BYOK_ENCRYPTION_KEY` does not need to leave the deployment to
take a backup, only to read the cleartext, which neither script does.

## Output formats

You can write either an archive (`--out path.tar.gz`) or an uncompressed
directory (`--out-dir path/`).  The directory form is useful for
poking at individual `*.jsonl.gz` files with `zcat | jq`.

## Partial backup / restore

```bash
# Back up only a couple of tables.
python scripts/archi_backup.py --out small.tar.gz \
    --table conversations --table conversation_metadata

# Restore only the user-action log from a full backup.
python scripts/archi_restore.py --in full.tar.gz --table user_actions
```

## Idempotency

`archi_restore.py` inserts each row with `ON CONFLICT (pk) DO NOTHING`.
Re-running the restore against a partially populated database converges to
"every row from the backup is present" without ever overwriting existing
rows.  This makes it safe to:

- Re-run a restore that was interrupted partway through.
- Layer multiple backups on top of each other.
- Restore a backup into a database that already has more recent data —
  newer rows are preserved.

If you want destructive overwrite behaviour (e.g. for disaster recovery onto
a known-empty database), drop the relevant tables first:

```bash
psql -c "TRUNCATE conversations, conversation_metadata, ... CASCADE;"
python scripts/archi_restore.py --in backup.tar.gz
```

## What's **not** backed up

These are reproducible from deployment config and source data; backing them
up would inflate the archive and risk restore-time conflicts:

- `static_config`, `dynamic_config`, `config_audit` (config is restored via
  the normal deployment seeding step)
- `documents`, `document_chunks`, vector embeddings (re-ingest from sources)
- `sessions`, `mcp_auth_codes`, short-lived OAuth/PKCE state

If you need a full pg-level snapshot (including these), use `pg_dump`
directly.  This tooling is for *user data* portability, not infrastructure
disaster recovery.

## Migration unification (future work)

Today archi has two parallel schema sources: `src/cli/templates/init.sql`
(fresh-install template) and `src/utils/config_service.py:_ensure_tables`
(in-process schema patcher for upgrades).  A separate change will collapse
these into a single OpenSpec-driven migration runner so backups stamped
with one schema version can be unambiguously replayed against a database
known to be on that version.  For now the manifest carries the version
returned by `migration_state`; restore is best-effort across versions.
