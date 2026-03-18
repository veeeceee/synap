"""Storage backends for synap."""

from synap.backends.sqlite import SQLiteBackend
from synap.backends.kuzu import KuzuBackend

__all__ = ["KuzuBackend", "SQLiteBackend"]
