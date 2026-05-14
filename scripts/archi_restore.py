#!/usr/bin/env python3
"""archi_restore — replay an ``archi_backup`` archive into a target database.

Idempotent by default: each row is inserted with ``ON CONFLICT DO NOTHING`` on
the primary key, so re-running the restore against a partially populated
database is safe and converges to "every row from the backup is present".

Usage:
    python scripts/archi_restore.py --in backup-2026-05-13.tar.gz
    python scripts/archi_restore.py --in-dir backup-2026-05-13/
    python scripts/archi_restore.py --in backup.tar.gz --table conversations
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    sys.stderr.write("psycopg2 is required.  Install psycopg2-binary.\n")
    raise


MANIFEST_NAME = "manifest.json"
JSONL_SUFFIX = ".jsonl.gz"

# Primary-key column(s) per table — used for ON CONFLICT DO NOTHING.
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "users": ("id",),
    "conversation_metadata": ("conversation_id",),
    "conversations": ("message_id",),
    "feedback": ("id",),
    "agent_traces": ("trace_id",),
    "agent_tool_calls": ("id",),
    "ab_comparisons": ("comparison_id",),
    "config_audit": ("id",),
    "user_actions": ("action_id",),
    "tool_approvals": ("approval_id",),
}


def _pg_config_from_env() -> dict:
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": os.environ.get("PGDATABASE", "archi-db"),
        "user": os.environ.get("PGUSER", "archi"),
        "password": os.environ.get("PG_PASSWORD", ""),
    }


def _decode_value(value):
    """Inverse of _encode_row in archi_backup.py."""
    if isinstance(value, dict):
        if "__b64__" in value and len(value) == 1:
            return base64.b64decode(value["__b64__"])
        if "__dt__" in value and len(value) == 1:
            return datetime.fromisoformat(value["__dt__"])
    return value


def _decode_row(row: dict) -> dict:
    return {k: _decode_value(v) for k, v in row.items()}


def _ensure_dir(archive: Optional[Path], in_dir: Optional[Path]) -> Path:
    """Either accept an unpacked directory or unpack a tar.gz to a tempdir."""
    if in_dir is not None:
        return in_dir
    assert archive is not None
    tmp = Path(tempfile.mkdtemp(prefix="archi-restore-"))
    with tarfile.open(archive, "r:*") as tar:
        # Use `data` filter when available (py3.12+) for safe extraction.
        try:
            tar.extractall(tmp, filter="data")
        except TypeError:
            tar.extractall(tmp)
    return tmp


def _restore_one_table(conn, table: str, jsonl_path: Path) -> int:
    """Insert all rows from *jsonl_path* into *table*, idempotently."""
    pk = PRIMARY_KEYS.get(table)
    if pk is None:
        raise ValueError(
            f"Refusing to restore unknown table {table!r}; add it to PRIMARY_KEYS."
        )

    inserted = 0
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as fp:
        batch: list[dict] = []
        for raw_line in fp:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = _decode_row(json.loads(raw_line))
            batch.append(row)
            if len(batch) >= 500:
                inserted += _flush_batch(conn, table, pk, batch)
                batch = []
        if batch:
            inserted += _flush_batch(conn, table, pk, batch)
    return inserted


def _flush_batch(conn, table: str, pk: tuple[str, ...], batch: list[dict]) -> int:
    """Execute an INSERT ... ON CONFLICT (pk) DO NOTHING for *batch*."""
    columns = list(batch[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(f'"{c}"' for c in columns)
    pk_sql = ", ".join(f'"{c}"' for c in pk)
    sql = (
        f'INSERT INTO {table} ({column_sql}) '
        f'VALUES ({placeholders}) '
        f'ON CONFLICT ({pk_sql}) DO NOTHING'
    )
    inserted = 0
    with conn.cursor() as cur:
        for row in batch:
            cur.execute(sql, tuple(row.get(c) for c in columns))
            inserted += cur.rowcount
    conn.commit()
    return inserted


def run_restore(
    *,
    archive: Optional[Path],
    in_dir: Optional[Path],
    only_tables: Optional[Iterable[str]] = None,
    pg_config: Optional[dict] = None,
) -> dict:
    if (archive is None) == (in_dir is None):
        raise ValueError("Pass exactly one of --in or --in-dir.")

    work_dir = _ensure_dir(archive, in_dir)
    manifest_path = work_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Backup is missing {MANIFEST_NAME} at {work_dir}")
    manifest = json.loads(manifest_path.read_text())
    print(f"Manifest schema_version={manifest.get('schema_version')!r}, "
          f"created_at={manifest.get('created_at')!r}", file=sys.stderr)

    pg_config = pg_config or _pg_config_from_env()
    conn = psycopg2.connect(**pg_config)
    results: dict[str, int] = {}
    try:
        for table, entry in manifest.get("tables", {}).items():
            if only_tables and table not in only_tables:
                continue
            jsonl_path = work_dir / entry["file"]
            if not jsonl_path.exists():
                print(f"  {table}: SKIP (file {entry['file']} missing)", file=sys.stderr)
                continue
            inserted = _restore_one_table(conn, table, jsonl_path)
            results[table] = inserted
            print(f"  {table}: inserted {inserted} of {entry.get('rows', '?')} row(s)",
                  file=sys.stderr)
    finally:
        conn.close()
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="archive", type=Path,
                     help="Path to a tar.gz archive produced by archi_backup.")
    src.add_argument("--in-dir", type=Path,
                     help="Path to an unpacked backup directory.")
    parser.add_argument("--table", action="append",
                        help="Restore only this table (repeatable).")
    args = parser.parse_args(argv)

    try:
        results = run_restore(
            archive=args.archive, in_dir=args.in_dir, only_tables=args.table,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    total = sum(results.values())
    print(f"Restore complete: {total} row(s) inserted across {len(results)} table(s).",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
