"""SQLite transactions commit decisions, positions and observations together."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from rl_stocks.core.contracts import canonical


@contextmanager
def transaction(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('''CREATE TABLE IF NOT EXISTS sessions
                      (session TEXT PRIMARY KEY, value TEXT NOT NULL)''')
        yield db
        db.execute('COMMIT')
    except BaseException:
        if db.in_transaction:
            db.execute('ROLLBACK')
        raise
    finally:
        db.close()


def get(db, key, default=None):
    row = db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_value(db, key, value):
    db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',
               (key, canonical(value).decode()))


def records(db):
    return [json.loads(r[0]) for r in db.execute('SELECT value FROM sessions ORDER BY session')]
