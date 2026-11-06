"""Create the FraudGuard database tables.

Safe by construction: uses ``Base.metadata.create_all``, which reflects the
schema first (``checkfirst=True``) and then issues a plain ``CREATE TABLE`` - plus
that table's ``CREATE INDEX`` statements - only for tables that are absent. It
never drops a table, never alters an existing one, and never touches rows.
Re-running it is a no-op because every declared table is already present.

Run once after setting ``DATABASE_URL`` in ``.env``::

    python scripts/init_db.py

Why not auto-create on application startup?
-------------------------------------------
Implicit DDL at boot means every replica races to run it, and it hides schema
drift. An explicit, auditable step is easier to reason about in deployment.

Why Alembic instead, at scale?
------------------------------
``create_all`` cannot express *changes*: rename a column, add a NOT NULL column
to a populated table, or backfill data, and it silently does nothing, leaving the
database out of sync with the models until something fails at runtime. Alembic
gives versioned, reviewable, reversible migrations that run as a deploy step, with
a recorded schema version in the database. It is the right tool as soon as more
than one person or environment shares the schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is sys.path[0]; add the project root so `import app...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import (  # noqa: E402
    DatabaseUnavailable,
    dispose_engine,
    get_engine,
    init_db,
    mask_database_url,
    safe_error_message,
)


def main() -> int:
    print("FraudGuard - database initialisation")
    print(f"Target: {mask_database_url(settings.database_url)}")

    if not settings.database_url:
        print(
            "\nDATABASE_URL is not set, so there is nothing to initialise.\n"
            "  1. Create a Neon project and copy the POOLED connection string.\n"
            "  2. Put it in .env as DATABASE_URL=postgresql+psycopg://...\n"
            "  3. Re-run:  python scripts/init_db.py\n"
        )
        return 1

    try:
        engine = get_engine()
        before = set(inspect(engine).get_table_names())
        print(f"Tables before: {sorted(before) or 'none'}")

        declared = init_db()

        after = sorted(inspect(engine).get_table_names())
        print(f"Tables declared by the models: {declared}")
        print(f"Tables after: {after}")

        created = [name for name in declared if name not in before]
        if created:
            print(f"\nCreated: {created}")
        else:
            print("\nNothing to create - the schema was already present.")
        print("No table was dropped or altered. Re-running this script is safe.")
        return 0

    except DatabaseUnavailable as exc:
        print(f"\nERROR: {exc}")
        return 1
    except SQLAlchemyError as exc:
        print(f"\nERROR: could not reach the database.\n  {safe_error_message(exc)}")
        print("  Check DATABASE_URL, the Neon project state, and your network.")
        return 1
    finally:
        dispose_engine()


if __name__ == "__main__":
    raise SystemExit(main())
