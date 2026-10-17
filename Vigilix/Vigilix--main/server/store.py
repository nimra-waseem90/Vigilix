"""
Lightweight SQLite persistence for Vigilix.

Starts empty on first run (a fresh vigilix.db is created with no rows) and
keeps growing from there - alerts and camera configs survive a backend
restart instead of living only in memory.
"""

import json
import os
import sqlite3
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "vigilix.db")

_lock = threading.Lock()


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for high concurrent throughput without locking
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


_conn = _connect()
with _lock:
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL
        )
    """)
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC)")
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_created_at ON cameras(created_at DESC)")
    _conn.commit()


def save_alert(alert):
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO alerts (id, created_at, data) VALUES (?, ?, ?)",
            (alert["id"], alert.get("created_at", ""), json.dumps(alert)),
        )
        _conn.commit()


def load_alerts(limit=200):
    with _lock:
        rows = _conn.execute(
            "SELECT data FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]


def save_camera(camera_id, name, source, created_at):
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO cameras (id, name, source, created_at) VALUES (?, ?, ?, ?)",
            (camera_id, name, source, created_at),
        )
        _conn.commit()


def delete_camera(camera_id):
    with _lock:
        _conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        _conn.commit()


def load_cameras():
    with _lock:
        rows = _conn.execute("SELECT id, name, source FROM cameras").fetchall()
    return [dict(r) for r in rows]
