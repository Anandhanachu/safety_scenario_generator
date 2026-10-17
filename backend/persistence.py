"""SQLite transactions, append-only snapshots and separate human review records."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    created TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, kind, payload, identity=None):
        identity = identity or uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO snapshots VALUES (?,?,?,?)", (identity, kind, now(), json.dumps(payload, allow_nan=False)))
        return identity

    def get(self, identity):
        with self.connect() as db:
            row = db.execute("SELECT * FROM snapshots WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise KeyError(identity)
        return {**dict(row), "payload": json.loads(row["payload"])}

    def list(self, kind, limit=200):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM snapshots WHERE kind=? ORDER BY created DESC, rowid DESC LIMIT ?", (kind, limit)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def iter_snapshots(self, kind):
        with self.connect() as db:
            cursor = db.execute("SELECT * FROM snapshots WHERE kind=? ORDER BY created DESC, rowid DESC", (kind,))
            while True:
                batch = cursor.fetchmany(100)
                if not batch:
                    break
                for row in batch:
                    yield {**dict(row), "payload": json.loads(row["payload"])}

    def latest_reviews(self, run_id=None):
        with self.connect() as db:
            query = "SELECT * FROM reviews" + (" WHERE run_id=?" if run_id else "") + " ORDER BY created, rowid"
            rows = db.execute(query, (run_id,) if run_id else ()).fetchall()
        return {(r["run_id"], r["scenario_id"]): {**dict(r), "payload": json.loads(r["payload"])} for r in rows}

    def setting(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        # This store is deliberately not a credential store.
        def check(item):
            if isinstance(item, dict):
                for k, v in item.items():
                    if any(term in k.lower() for term in ("token", "secret", "password", "api_key")) and k != "max_tokens":
                        raise ValueError("Secrets must be supplied through the environment")
                    check(v)
            elif isinstance(item, list):
                for v in item:
                    check(v)
        check(value)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))

    def review(self, run_id, scenario_id, decision, reviewer, notes):
        allowed = {"accepted", "revision_requested", "rejected", "evidence_insufficient", "additional_testing"}
        if decision not in allowed or not reviewer.strip():
            raise ValueError("A valid decision and named reviewer are required")
        run = self.get(run_id)["payload"]
        from backend.analytics import proposal_entries
        scenarios = [entry["scope"] for entry in proposal_entries(run)]
        if scenario_id not in scenarios:
            raise ValueError("Scenario not found in this run")
        if decision == "accepted" and not next(e for e in proposal_entries(run) if e["scope"] == scenario_id)["schema_valid"]:
            raise ValueError("Schema-invalid candidates require revision before acceptance")
        entry = next(e for e in proposal_entries(run) if e['scope'] == scenario_id)
        if decision == 'accepted' and entry['candidate'].get('needs_human_review'):
            raise ValueError('Resolve expert-evidence/risk/confidence flags and rerun before acceptance')
        payload = {"decision": decision, "reviewer": reviewer[:200], "notes": notes[:8000]}
        with self.connect() as db:
            db.execute("INSERT INTO reviews VALUES (?,?,?,?,?)", (uuid4().hex, run_id, scenario_id, now(), json.dumps(payload)))

    def reviews(self, run_id):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM reviews WHERE run_id=? ORDER BY created, rowid", (run_id,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def clear_history(self):
        with self.connect() as db:
            db.execute("DELETE FROM reviews")
            db.execute("DELETE FROM snapshots WHERE kind IN ('run','evaluation','simulation','risk_assessment','retrieval_benchmark')")
