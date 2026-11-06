"""Persistent anonymous session credentials and field ownership for the HTTP API."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import secrets
import sqlite3


class SessionStore:
    def __init__(self, data_dir):
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "sessions.sqlite3"
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS field_owners (
                    field_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL REFERENCES sessions(token_hash)
                );
                CREATE INDEX IF NOT EXISTS fields_by_owner ON field_owners(token_hash);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _hash(token):
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def create(self):
        token = secrets.token_urlsafe(32)
        with self._connect() as db:
            db.execute("INSERT INTO sessions(token_hash) VALUES (?)", (self._hash(token),))
        return token

    def authenticate(self, token):
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            return None
        digest = self._hash(token)
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM sessions WHERE token_hash = ?", (digest,)).fetchone()
        return digest if exists else None

    def claim_field(self, field_id, owner):
        with self._connect() as db:
            db.execute("INSERT INTO field_owners(field_id, token_hash) VALUES (?, ?)", (field_id, owner))

    def owns(self, field_id, owner):
        with self._connect() as db:
            return db.execute("SELECT 1 FROM field_owners WHERE field_id = ? AND token_hash = ?",
                              (field_id, owner)).fetchone() is not None

    def fields(self, owner):
        with self._connect() as db:
            return [row[0] for row in db.execute(
                "SELECT field_id FROM field_owners WHERE token_hash = ? ORDER BY field_id", (owner,))]
