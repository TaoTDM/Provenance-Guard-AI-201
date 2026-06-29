"""SQLite audit log + content store (planning.md "Audit Log" section).

Two tables:
  decisions — one row per submitted piece of content (the canonical record a
              reader/grader inspects; also holds the mutable `status` an appeal
              flips to "under_review").
  events    — append-only stream of things that happened to a content_id
              (classified | appeal | verified), so the full history is replayable.

Text is never stored raw: we keep a sha256 hash + a short preview (PII-safe,
per planning.md "Improvements"). Milestone 3 fills the LLM columns; the
stylometric/lexical columns exist now but stay NULL until Milestone 4.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH, LOG_DEFAULT_LIMIT


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso():
    """UTC timestamp with millisecond precision, e.g. 2026-06-28T22:32:10.123Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_preview(text, n=120):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                content_id        TEXT PRIMARY KEY,
                creator_id        TEXT NOT NULL,
                content_type      TEXT NOT NULL DEFAULT 'text',
                timestamp         TEXT NOT NULL,
                text_hash         TEXT,
                text_preview      TEXT,
                word_count        INTEGER,
                attribution       TEXT,        -- likely_ai | likely_human | uncertain
                confidence        REAL,
                p_ai              REAL,
                p_llm             REAL,
                p_style           REAL,
                p_lex             REAL,
                p_meta            REAL,        -- image_metadata signal (multi-modal stretch)
                disagreement      REAL,
                llm_rationale     TEXT,
                verified_creator  INTEGER DEFAULT 0,  -- had a Verified-Human cert at submit time
                degraded          INTEGER DEFAULT 0,
                status            TEXT NOT NULL DEFAULT 'classified',
                appeal_reasoning  TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS certificates (
                cert_id     TEXT PRIMARY KEY,
                creator_id  TEXT NOT NULL,
                method      TEXT NOT NULL,
                issued_at   TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id  TEXT NOT NULL,
                event_type  TEXT NOT NULL,      -- classified | appeal | verified
                timestamp   TEXT NOT NULL,
                detail_json TEXT,
                FOREIGN KEY (content_id) REFERENCES decisions(content_id)
            )
            """
        )


def insert_decision(record):
    """Insert one classification decision. `record` is a dict keyed by column name."""
    cols = [
        "content_id", "creator_id", "content_type", "timestamp", "text_hash",
        "text_preview", "word_count", "attribution", "confidence", "p_ai",
        "p_llm", "p_style", "p_lex", "p_meta", "disagreement", "llm_rationale",
        "verified_creator", "degraded", "status", "appeal_reasoning",
    ]
    placeholders = ", ".join("?" for _ in cols)
    values = [record.get(c) for c in cols]
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO decisions ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )


def add_event(content_id, event_type, detail):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (content_id, event_type, timestamp, detail_json) "
            "VALUES (?, ?, ?, ?)",
            (content_id, event_type, now_iso(), json.dumps(detail)),
        )


def get_decision(content_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM decisions WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def get_recent_decisions(limit=LOG_DEFAULT_LIMIT):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def mark_under_review(content_id, reasoning):
    """Flip a content's status to 'under_review' and store the appeal reasoning.
    Returns True if the content existed and was updated, else False."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE decisions SET status = 'under_review', appeal_reasoning = ? "
            "WHERE content_id = ?",
            (reasoning, content_id),
        )
        return cur.rowcount > 0


def get_appeals_queue():
    """All content currently awaiting human review (the reviewer queue)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE status = 'under_review' "
            "ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# --- Provenance certificates (Verified Human stretch, planning.md S2) ---------

def insert_certificate(cert_id, creator_id, method):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO certificates (cert_id, creator_id, method, issued_at) "
            "VALUES (?, ?, ?, ?)",
            (cert_id, creator_id, method, now_iso()),
        )


def get_certificate(creator_id):
    """Return the most recent certificate for a creator, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM certificates WHERE creator_id = ? "
            "ORDER BY issued_at DESC LIMIT 1",
            (creator_id,),
        ).fetchone()
    return dict(row) if row else None


# --- Analytics aggregates (dashboard stretch, planning.md S3) ------------------

def get_analytics():
    """Aggregate detection patterns for the dashboard."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

        verdicts = {
            row["attribution"]: row["n"]
            for row in conn.execute(
                "SELECT attribution, COUNT(*) AS n FROM decisions GROUP BY attribution"
            ).fetchall()
        }
        appeals = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE status = 'under_review'"
        ).fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM decisions WHERE confidence IS NOT NULL"
        ).fetchone()[0]
        # signal agreement: decisions where the signals broadly agreed
        agree = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE disagreement IS NOT NULL "
            "AND disagreement < 0.15"
        ).fetchone()[0]
        scored = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE disagreement IS NOT NULL"
        ).fetchone()[0]
        # extra metric: appeal rate on AI verdicts (a false-positive proxy)
        ai_total = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE attribution = 'likely_ai'"
        ).fetchone()[0]
        ai_appealed = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE attribution = 'likely_ai' "
            "AND status = 'under_review'"
        ).fetchone()[0]
        certs = conn.execute("SELECT COUNT(*) FROM certificates").fetchone()[0]

    def rate(num, den):
        return round(num / den, 3) if den else 0.0

    return {
        "total_submissions": total,
        "verdict_distribution": verdicts,
        "appeal_rate": rate(appeals, total),
        "average_confidence": round(avg_conf, 3) if avg_conf is not None else None,
        "signal_agreement_rate": rate(agree, scored),
        "ai_verdict_appeal_rate": rate(ai_appealed, ai_total),
        "verified_creators": certs,
    }
