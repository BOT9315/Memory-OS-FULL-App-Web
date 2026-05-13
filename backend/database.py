import sqlite3, os, json
from typing import List, Dict, Optional
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "memory_os.db")

class Database:
    def _conn(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self):
        with self._conn() as c:
            # Phase 1
            c.execute("""CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                ai_reply TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            # Phase 2
            c.execute("""CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            # Phase 3
            c.execute("""CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                deadline TEXT,
                status TEXT DEFAULT 'active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                sentiment TEXT DEFAULT 'neutral',
                last_mentioned TEXT,
                frequency INTEGER DEFAULT 1,
                UNIQUE(user_id, name)
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS mood_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                score INTEGER NOT NULL,
                note TEXT,
                timestamp TEXT NOT NULL
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_entries_user ON entries(user_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_mood_user ON mood_logs(user_id)")
            c.commit()
        print(f"DB ready: {DB_PATH}")

    def save_entry(self, user_id, user_message, ai_reply, timestamp):
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO entries (user_id,user_message,ai_reply,timestamp) VALUES (?,?,?,?)",
                (user_id, user_message, ai_reply, timestamp)
            )
            c.commit()
            return cur.lastrowid

    def get_recent_entries(self, user_id, limit=20):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM entries WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_all_entries(self, user_id, limit=200):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM entries WHERE user_id=? ORDER BY created_at ASC LIMIT ?",
                (user_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_entry_by_id(self, entry_id):
        with self._conn() as c:
            row = c.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return dict(row) if row else None

    def count_entries(self, user_id):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM entries WHERE user_id=?", (user_id,)).fetchone()[0]

    def clear_entries(self, user_id):
        with self._conn() as c:
            c.execute("DELETE FROM entries WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM patterns WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM goals WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM relationships WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM mood_logs WHERE user_id=?", (user_id,))
            c.commit()

    # Patterns
    def save_patterns(self, user_id, data):
        with self._conn() as c:
            c.execute("INSERT INTO patterns (user_id,data) VALUES (?,?)", (user_id, json.dumps(data)))
            c.commit()

    def get_latest_patterns(self, user_id):
        with self._conn() as c:
            row = c.execute(
                "SELECT data FROM patterns WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
                (user_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    # Goals
    def add_goal(self, user_id, goal, deadline=None):
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO goals (user_id,goal,deadline) VALUES (?,?,?)",
                (user_id, goal, deadline)
            )
            c.commit()
            return cur.lastrowid

    def get_active_goals(self, user_id):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM goals WHERE user_id=? AND status='active' ORDER BY created_at DESC",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def complete_goal(self, goal_id):
        with self._conn() as c:
            c.execute(
                "UPDATE goals SET status='completed', completed_at=? WHERE id=?",
                (datetime.now().isoformat(), goal_id)
            )
            c.commit()

    def upsert_detected_goal(self, user_id, goal, status):
        with self._conn() as c:
            existing = c.execute(
                "SELECT id FROM goals WHERE user_id=? AND goal LIKE ?",
                (user_id, f"%{goal[:30]}%")
            ).fetchone()
            if not existing:
                c.execute(
                    "INSERT INTO goals (user_id,goal,status) VALUES (?,?,?)",
                    (user_id, goal, status)
                )
            c.commit()

    # Relationships
    def upsert_relationship(self, user_id, name, sentiment, last_mentioned, frequency):
        with self._conn() as c:
            c.execute("""INSERT INTO relationships (user_id,name,sentiment,last_mentioned,frequency)
                VALUES (?,?,?,?,?)
                ON CONFLICT(user_id,name) DO UPDATE SET
                  sentiment=excluded.sentiment,
                  last_mentioned=excluded.last_mentioned,
                  frequency=excluded.frequency""",
                (user_id, name, sentiment, last_mentioned, frequency)
            )
            c.commit()

    def get_relationships(self, user_id):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM relationships WHERE user_id=? ORDER BY frequency DESC",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # Mood
    def log_mood(self, user_id, score, note=None):
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO mood_logs (user_id,score,note,timestamp) VALUES (?,?,?,?)",
                (user_id, score, note, datetime.now().isoformat())
            )
            c.commit()
            return cur.lastrowid

    def get_mood_history(self, user_id, limit=30):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM mood_logs WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

db = Database()
