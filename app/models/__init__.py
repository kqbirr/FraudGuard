"""ORM models for FraudGuard.

Importing this package registers every model on ``Base.metadata``, which is what
``init_db()`` relies on to create the schema.
"""

from app.models.flagged import FlaggedTransaction

__all__ = ["FlaggedTransaction"]
