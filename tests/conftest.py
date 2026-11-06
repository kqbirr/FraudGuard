"""Shared pytest configuration.

Tests must never touch a real database. A developer's ``.env`` may hold a live
``DATABASE_URL`` (for example a Neon branch), and ``app.config`` calls
``load_dotenv()`` at import time -- which would silently wire these tests to that
database. Emptying the variable here, before any ``app`` module is imported,
keeps ``Settings`` on its documented "persistence disabled" default:
``load_dotenv`` does not override variables that are already present in the
environment, even when their value is the empty string.

Tests that exercise persistence opt in explicitly by building their own
file-backed SQLite engine and overriding the session dependency
(see ``tests/test_persistence.py``).
"""

from __future__ import annotations

import os

os.environ["DATABASE_URL"] = ""
