#!/usr/bin/env python3
"""archi_backup — snapshot user-facing tables to a gzipped JSONL archive.

A simple, dependency-light alternative to pg_dump for the rows users care
about: conversations, feedback, tool-call history, agent traces, A/B
comparisons, user preferences and audit trails.  Crucially **omits** the
deploy-time tables (``static_config``, ``dynamic_config``, ``documents``,
``document_chunks``, vector indexes) — those are re-seeded from the deployment
config and source corpus, so including them inflates the backup pointlessly
and risks restore-time conflicts.

Output layout::

    out/
      manifest.json          # schema_version, created_at, row counts per table
      users.jsonl.gz
      conversation_metadata.jsonl.gz
      conversations.jsonl.gz
      feedback.jsonl.gz
      agent_tool_calls.jsonl.gz
      agent_traces.jsonl.gz
      ab_comparisons.jsonl.gz
      config_audit.jsonl.gz
      user_actions.jsonl.gz                # if user_actions exists
      tool_approvals.jsonl.gz              # if tool_approvals exists

The companion ``archi_restore.py`` consumes the same layout.

Encrypted columns (BYOK API keys, refresh tokens, ...) are dumped **as bytes**
in their already-encrypted form.  ``BYOK_ENCRYPTION_KEY`` does *not* need to
leave the deployment to take a backup — only to read the cleartext, which
this tool does not do.

Usage:
    python scripts/archi_backup.py --out backup-2026-05-13.tar.gz
    python scripts/archi_backup.py --out-dir backup-2026-05-13/        # uncompressed dir
    python scripts/archi_backup.py --table conversations --table feedback ...
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    sys.stderr.write("psycopg2 is required.  Install psycopg2-binary.\n")
    raise


# Tables backed up by default — ordered to respect foreign-key constraints
# at restore time (users -> conversation_metadata -> conversations -> ...).
DEFAULT_TABLES: tuple[str, ...] = (
    "users",
    "conversation_metadata",
    "conversations",
    "feedback",
    "agent_traces",
    "agent_tool_calls",
    "ab_comparisons",
    "config_audit",
    "user_actions",
    "tool_approvals",
)

MANIFEST_NAME = "manifest.json"
JSONL_SUFFIX = ".jsonl.gz"
SCHEMA_VERSION_FALLBACK = "unknown"


def _pg_config_from_env() -> dict:
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": os.environ.get("PGDATABASE", "archi-db"),
        "user": os.environ.get("PGUSER", "archi"),
        "password": os.environ.get("PG_PASSWORD", ""),
    }


def _encode_row(row: dict) -> dict:
    """Make a row JSON-serialisable, preserving binary as base64."""
    out = {}
    for key, value in row.items():
        if isinstance(value, (bytes, bytearray, memoryview)):
            out[key] = {"__b64__": base64.b64encode(bytes(value)).decode("ascii")}
        elif isinstance(value, datetime):
            out[key] = {"__dt__": value.isoformat()}
        else:
            out[key] = value
    return out


def _table_exists(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return cur.fetchone() is not None


def _read_schema_version(cur) -> str:
    if not _table_exists(cur, "migration_state"):
        return SCHEMA_VERSION_FALLBACK
    try:
        cur.execute(
            """
            SELECT migration_name FROM migration_state
            WHERE status = 'completed'
            ORDER BY completed_at DESC NULLS LAST, started_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return row[0] if row else SCHEMA_VERSION_FALLBACK
    except Exception:
        return SCHEMA_VERSION_FALLBACK


def _dump_table(cur, table: str, fp: io.BufferedWriter, *, batch_size: int = 1000) -> int:
    """Stream rows from *table* to *fp* as JSONL.  Returns the row count."""
    cur.execute(f"SELECT * FROM {table}")
    count = 0
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            fp.write((json.dumps(_encode_row(dict(row))) + "\n").encode("utf-8"))
            count += 1
    return count


def _write_table_to_dir(conn, table: str, out_dir: Path) -> tuple[int, str]:
    out_path = out_dir / f"{table}{JSONL_SUFFIX}"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if not _table_exists(cur, table):
            return 0, ""
        with gzip.open(out_path, "wb") as gz:
            count = _dump_table(cur, table, gz)
    return count, str(out_path)


def _build_manifest(conn, tables_dumped: dict) -> dict:
    with conn.cursor() as cur:
        version = _read_schema_version(cur)
    return {
        "format_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": version,
        "tables": tables_dumped,
        "format_notes": [
            "Binary columns are base64-encoded under {'__b64__': '...'}.",
            "Timestamp columns are ISO-8601 strings under {'__dt__': '...'}.",
        ],
    }


def run_backup(
    *,
    tables: Iterable[str],
    out: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    pg_config: Optional[dict] = None,
) -> Path:
    if (out is None) == (out_dir is None):
        raise ValueError("Pass exactly one of --out or --out-dir.")

    pg_config = pg_config or _pg_config_from_env()
    conn = psycopg2.connect(**pg_config)
    try:
        # Stage everything to a working directory, even when --out (tar.gz) is requested.
        working = Path(out_dir) if out_dir is not None else Path(out).with_suffix("") / "_staging"
        working.mkdir(parents=True, exist_ok=True)

        tables_dumped = {}
        for table in tables:
            count, path = _write_table_to_dir(conn, table, working)
            if path:
                tables_dumped[table] = {"rows": count, "file": Path(path).name}
                print(f"  {table}: {count} row(s)", file=sys.stderr)
            else:
                print(f"  {table}: skipped (table not present)", file=sys.stderr)

        manifest = _build_manifest(conn, tables_dumped)
        (working / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        if out_dir is not None:
            return working

        # Tar+gz it.
        out_path = Path(out)
        with tarfile.open(out_path, "w:gz") as tar:
            for child in sorted(working.iterdir()):
                tar.add(child, arcname=child.name)
        # Clean up the staging dir.
        for child in working.iterdir():
            child.unlink()
        working.rmdir()
        # If the parent of working was created just for staging, drop it too.
        try:
            working.parent.rmdir()
        except OSError:
            pass
        return out_path
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--out", type=Path,
                           help="Path to .tar.gz archive to write.")
    out_group.add_argument("--out-dir", type=Path,
                           help="Write an uncompressed directory of jsonl.gz files.")
    parser.add_argument("--table", action="append",
                        help="Backup only this table (repeatable).  Default: all "
                             "user-facing tables.")
    args = parser.parse_args(argv)

    tables = args.table or list(DEFAULT_TABLES)
    print(f"Backing up {len(tables)} table(s) to "
          f"{args.out or args.out_dir}", file=sys.stderr)

    try:
        result = run_backup(tables=tables, out=args.out, out_dir=args.out_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Backup written to {result}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
