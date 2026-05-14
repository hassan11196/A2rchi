#!/usr/bin/env python3
"""Backfill ``archi_service`` and ``source_ref`` on legacy Mattermost rows.

PR #543 introduced ``conversation_metadata.archi_service`` (default ``'chat'``)
and ``conversation_metadata.source_ref`` so Mattermost-originated rows can be
distinguished from web-chat rows and looked up by a stable external key.

Deployments that ran the Mattermost service *before* PR #543 — when
``mattermost.py`` used the in-memory single-turn flow — never wrote either
column.  Their pre-existing rows are therefore stuck with
``archi_service = 'chat'`` and ``source_ref IS NULL`` even though they were
in fact Mattermost conversations.

This script repairs those rows.  Heuristic: any ``conversation_metadata``
row whose ``client_id`` matches ``mm_user_%`` was created by the bridge
(see ``ThreadContextManager.mm_client_id`` in ``src/interfaces/mattermost.py``)
and should have ``archi_service = 'mattermost'``.  We leave ``source_ref``
alone if it can't be reconstructed cheaply.

Usage:
    python scripts/backfill_mattermost_archi_service.py            # dry-run
    python scripts/backfill_mattermost_archi_service.py --apply    # write
    python scripts/backfill_mattermost_archi_service.py --apply --batch 100
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

try:
    import psycopg2
except ImportError:  # pragma: no cover — diagnostic path
    sys.stderr.write("psycopg2 is required.  Install psycopg2-binary.\n")
    raise


def _pg_config_from_env() -> Dict[str, str]:
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": os.environ.get("PGDATABASE", "archi-db"),
        "user": os.environ.get("PGUSER", "archi"),
        "password": os.environ.get("PG_PASSWORD", ""),
    }


SQL_COUNT_AFFECTED = """
SELECT COUNT(*)
FROM conversation_metadata
WHERE client_id LIKE 'mm_user_%%'
  AND (archi_service IS NULL OR archi_service = 'chat');
"""

SQL_BACKFILL = """
UPDATE conversation_metadata
SET archi_service = 'mattermost'
WHERE client_id LIKE 'mm_user_%%'
  AND (archi_service IS NULL OR archi_service = 'chat');
"""

SQL_PEEK_SAMPLES = """
SELECT conversation_id, title, client_id, archi_service, source_ref, last_message_at
FROM conversation_metadata
WHERE client_id LIKE 'mm_user_%%'
  AND (archi_service IS NULL OR archi_service = 'chat')
ORDER BY last_message_at DESC NULLS LAST
LIMIT %s;
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually run the UPDATE.  Without this, prints a dry-run summary.",
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of sample rows to display in dry-run mode (default 10).",
    )
    args = parser.parse_args(argv)

    cfg = _pg_config_from_env()
    print(f"Connecting to {cfg['user']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}", file=sys.stderr)

    conn = psycopg2.connect(**cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(SQL_COUNT_AFFECTED)
            total = cur.fetchone()[0]
            print(f"Rows eligible for backfill: {total}")

            if total == 0:
                return 0

            cur.execute(SQL_PEEK_SAMPLES, (args.samples,))
            rows = cur.fetchall()
            print(f"\nSample of up to {args.samples} affected rows:")
            for row in rows:
                conv_id, title, client_id, archi_service, source_ref, last_at = row
                print(
                    f"  id={conv_id} client_id={client_id!r} "
                    f"archi_service={archi_service!r} source_ref={source_ref!r} "
                    f"last_message_at={last_at} title={title!r}"
                )

            if not args.apply:
                print("\nDry-run only.  Re-run with --apply to perform the update.")
                return 0

            cur.execute(SQL_BACKFILL)
            updated = cur.rowcount
        conn.commit()
        print(f"Backfilled {updated} row(s) to archi_service='mattermost'.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
