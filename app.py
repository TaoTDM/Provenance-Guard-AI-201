"""Provenance Guard — Flask app (Milestone 3).

Endpoints live here:
  POST /submit   accept text + creator_id, run Signal 1, log, return a verdict
  GET  /log      return recent audit-log entries as JSON
  GET  /healthz  liveness + whether the Groq key is configured

Confidence scoring, the transparency label, /appeal, and rate limiting are added
in Milestones 4 and 5. The /submit response intentionally returns a PLACEHOLDER
confidence and label for now (see planning.md "AI Tool Plan").
"""

import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from config import LOG_DEFAULT_LIMIT
from db import (
    add_event,
    get_recent_decisions,
    init_db,
    insert_decision,
    now_iso,
    text_hash,
    text_preview,
)
from detection import (
    combine_signals,
    groq_signal,
    lexical_signal,
    stylometric_signal,
)

load_dotenv()  # read GROQ_API_KEY from .env

app = Flask(__name__)
init_db()

_PLACEHOLDER_LABEL = "(label generated in Milestone 5 — placeholder)"


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()

    if not text:
        return jsonify({"error": "field 'text' is required"}), 400
    if not creator_id:
        return jsonify({"error": "field 'creator_id' is required"}), 400

    content_id = str(uuid.uuid4())
    word_count = len(text.split())

    # --- Run the three detection signals (planning.md §1) ---
    sig1 = groq_signal(text)              # semantic
    sig2 = stylometric_signal(text)       # structural
    sig3 = lexical_signal(text)           # lexical
    p_llm, p_style, p_lex = sig1["p_llm"], sig2["p_style"], sig3["p_lex"]

    # --- Combine into a calibrated confidence + verdict (planning.md §2) ---
    score = combine_signals(p_llm, p_style, p_lex, word_count, sig1["degraded"])
    attribution = score["attribution"]
    confidence = score["confidence"]

    # --- Persist to audit log ---
    record = {
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": "text",
        "timestamp": now_iso(),
        "text_hash": text_hash(text),
        "text_preview": text_preview(text),
        "word_count": word_count,
        "attribution": attribution,
        "confidence": confidence,
        "p_ai": score["p_ai"],
        "p_llm": p_llm,
        "p_style": p_style,
        "p_lex": p_lex,
        "disagreement": score["disagreement"],
        "llm_rationale": sig1["rationale"],
        "degraded": 1 if sig1["degraded"] else 0,
        "status": "classified",
        "appeal_reasoning": None,
    }
    insert_decision(record)
    add_event(content_id, "classified", {
        "attribution": attribution,
        "confidence": confidence,
        "p_ai": score["p_ai"],
        "p_llm": p_llm,
        "p_style": p_style,
        "p_lex": p_lex,
        "disagreement": score["disagreement"],
        "degraded": sig1["degraded"],
    })

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence,
        "label": _PLACEHOLDER_LABEL,         # PLACEHOLDER until Milestone 5
        "p_ai": score["p_ai"],
        "signals": {
            "llm_score": p_llm,
            "stylometric_score": p_style,
            "lexical_score": p_lex,
            "disagreement": score["disagreement"],
            "llm_rationale": sig1["rationale"],
        },
        "low_evidence": score["low_evidence"],
        "degraded": sig1["degraded"],
        "status": "classified",
    }), 200


@app.route("/log", methods=["GET"])
def log():
    """Return recent audit entries. ?limit=N optional. Open for grading visibility;
    a real deployment would require auth here."""
    try:
        limit = int(request.args.get("limit", LOG_DEFAULT_LIMIT))
    except ValueError:
        limit = LOG_DEFAULT_LIMIT
    return jsonify({"entries": get_recent_decisions(limit)}), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    key = os.environ.get("GROQ_API_KEY")
    return jsonify({
        "status": "ok",
        "groq_key_configured": bool(key and key != "your_key_here"),
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
