"""Provenance Guard — Flask app.

Endpoints:
  POST /submit    accept text + creator_id, run 3 signals, score, label, log
  POST /appeal    creator contests a verdict -> status 'under_review' + logged
  GET  /log       return recent audit-log entries as JSON
  GET  /appeals   reviewer queue: content awaiting human review
  GET  /healthz   liveness + whether the Groq key is configured

Rate limiting (Flask-Limiter) protects /submit and /appeal — see planning.md
"Rate Limiting" for the chosen limits and reasoning.
"""

import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import LOG_DEFAULT_LIMIT
from db import (
    add_event,
    get_analytics,
    get_appeals_queue,
    get_certificate,
    get_decision,
    get_recent_decisions,
    init_db,
    insert_certificate,
    insert_decision,
    mark_under_review,
    now_iso,
    text_hash,
    text_preview,
)
from detection import (
    combine_metadata,
    combine_signals,
    groq_signal,
    lexical_signal,
    metadata_signal,
    stylometric_signal,
)
from labels import generate_label

load_dotenv()  # read GROQ_API_KEY from .env

app = Flask(__name__)
init_db()

# Rate limiting. Limits are documented in planning.md / README:
#   /submit  10/min, 100/day  — a real creator submits their own work rarely;
#                               bursts above this indicate scripted abuse.
#   /appeal  5/min            — appeals are human-paced and rare.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "rate limit exceeded",
        "detail": str(e.description),
    }), 429


@app.route("/", methods=["GET"])
def index():
    """Serve the Provenance Guard web UI (the attribution desk)."""
    ui_path = os.path.join(os.path.dirname(__file__), "webui.html")
    with open(ui_path, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api", methods=["GET"])
def api_index():
    """Machine-readable listing of the available endpoints."""
    return jsonify({
        "service": "Provenance Guard",
        "status": "running",
        "endpoints": {
            "POST /submit": "classify text or image metadata",
            "POST /appeal": "contest a verdict (content_id + creator_reasoning)",
            "POST /verify": "earn a Verified-Human certificate (creator_id + passage)",
            "GET /log": "recent audit-log entries",
            "GET /appeals": "reviewer queue (content under review)",
            "GET /analytics": "detection metrics (JSON)",
            "GET /dashboard": "detection metrics (HTML)",
            "GET /healthz": "liveness check",
        },
    }), 200


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute; 100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    creator_id = (data.get("creator_id") or "").strip()
    content_type = (data.get("content_type") or "text").strip()

    if not creator_id:
        return jsonify({"error": "field 'creator_id' is required"}), 400
    if content_type not in ("text", "image_metadata"):
        return jsonify({"error": "content_type must be 'text' or 'image_metadata'"}), 400

    content_id = str(uuid.uuid4())
    verified = get_certificate(creator_id) is not None

    if content_type == "image_metadata":
        result = _classify_image_metadata(data)
    else:
        result = _classify_text(data)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400

    score = result["score"]
    attribution = score["attribution"]
    confidence = score["confidence"]

    # --- Transparency label (+ Verified-Human badge if applicable) ---
    label = generate_label(attribution, score["p_ai"], confidence, verified_creator=verified)

    # --- Persist to audit log ---
    record = {
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "timestamp": now_iso(),
        "text_hash": result["text_hash"],
        "text_preview": result["text_preview"],
        "word_count": result["word_count"],
        "attribution": attribution,
        "confidence": confidence,
        "p_ai": score["p_ai"],
        "p_llm": result["p_llm"],
        "p_style": result["p_style"],
        "p_lex": result["p_lex"],
        "p_meta": result["p_meta"],
        "disagreement": score["disagreement"],
        "llm_rationale": result["rationale"],
        "verified_creator": 1 if verified else 0,
        "degraded": 1 if result["degraded"] else 0,
        "status": "classified",
        "appeal_reasoning": None,
    }
    insert_decision(record)
    add_event(content_id, "classified", {
        "content_type": content_type,
        "attribution": attribution,
        "confidence": confidence,
        "p_ai": score["p_ai"],
        "p_llm": result["p_llm"],
        "p_style": result["p_style"],
        "p_lex": result["p_lex"],
        "p_meta": result["p_meta"],
        "disagreement": score["disagreement"],
        "verified_creator": verified,
        "degraded": result["degraded"],
    })

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "attribution": attribution,
        "confidence": confidence,
        "label": label["text"],
        "label_variant": label["variant"],
        "verified_creator": verified,
        "p_ai": score["p_ai"],
        "signals": result["signals_out"],
        "low_evidence": score["low_evidence"],
        "degraded": result["degraded"],
        "status": "classified",
    }), 200


def _classify_text(data):
    """Run the 3 text signals + scoring. Returns a result dict or {'error':...}."""
    text = (data.get("text") or "").strip()
    if not text:
        return {"error": "field 'text' is required for content_type 'text'"}

    word_count = len(text.split())
    sig1 = groq_signal(text)              # semantic
    sig2 = stylometric_signal(text)       # structural
    sig3 = lexical_signal(text)           # lexical
    p_llm, p_style, p_lex = sig1["p_llm"], sig2["p_style"], sig3["p_lex"]
    score = combine_signals(p_llm, p_style, p_lex, word_count, sig1["degraded"])

    return {
        "score": score,
        "text_hash": text_hash(text),
        "text_preview": text_preview(text),
        "word_count": word_count,
        "p_llm": p_llm, "p_style": p_style, "p_lex": p_lex, "p_meta": None,
        "rationale": sig1["rationale"],
        "degraded": sig1["degraded"],
        "signals_out": {
            "llm_score": p_llm,
            "stylometric_score": p_style,
            "lexical_score": p_lex,
            "disagreement": score["disagreement"],
            "llm_rationale": sig1["rationale"],
        },
    }


def _classify_image_metadata(data):
    """Multi-modal path (stretch S4): metadata signal + text signals on description."""
    metadata = data.get("metadata") or {}
    description = (data.get("description") or "").strip()
    if not metadata:
        return {"error": "field 'metadata' (object) is required for content_type 'image_metadata'"}

    meta = metadata_signal(metadata)
    word_count = len(description.split())
    p_style = stylometric_signal(description)["p_style"] if description else None
    p_lex = lexical_signal(description)["p_lex"] if description else None
    score = combine_metadata(meta["p_meta"], p_style, p_lex, word_count)

    preview = description or " ".join(f"{k}={v}" for k, v in metadata.items())
    return {
        "score": score,
        "text_hash": text_hash(preview),
        "text_preview": text_preview(preview),
        "word_count": word_count,
        "p_llm": None, "p_style": p_style, "p_lex": p_lex, "p_meta": meta["p_meta"],
        "rationale": meta["reason"],
        "degraded": False,
        "signals_out": {
            "metadata_score": meta["p_meta"],
            "metadata_reason": meta["reason"],
            "stylometric_score": p_style,
            "lexical_score": p_lex,
            "disagreement": score["disagreement"],
        },
    }


@app.route("/appeal", methods=["POST"])
@limiter.limit("5 per minute")
def appeal():
    """A creator contests a classification (planning.md §4).

    Captures their reasoning, flips status to 'under_review', logs the appeal
    alongside the original decision snapshot, and confirms receipt. No automated
    re-classification — an appeal flags the item for a human reviewer.
    """
    data = request.get_json(silent=True) or {}
    content_id = (data.get("content_id") or "").strip()
    reasoning = (data.get("creator_reasoning") or "").strip()

    if not content_id:
        return jsonify({"error": "field 'content_id' is required"}), 400
    if not reasoning:
        return jsonify({"error": "field 'creator_reasoning' is required"}), 400

    original = get_decision(content_id)
    if original is None:
        return jsonify({"error": f"unknown content_id '{content_id}'"}), 404

    updated = mark_under_review(content_id, reasoning)
    if not updated:  # defensive; should not happen if original was found
        return jsonify({"error": "could not update content status"}), 500

    appeal_id = str(uuid.uuid4())
    original_decision = {
        "attribution": original["attribution"],
        "confidence": original["confidence"],
        "p_ai": original["p_ai"],
        "p_llm": original["p_llm"],
        "p_style": original["p_style"],
        "p_lex": original["p_lex"],
        "disagreement": original["disagreement"],
    }
    add_event(content_id, "appeal", {
        "appeal_id": appeal_id,
        "creator_reasoning": reasoning,
        "original_decision": original_decision,
    })

    return jsonify({
        "appeal_id": appeal_id,
        "content_id": content_id,
        "status": "under_review",
        "message": "Appeal received. This content has been flagged for human review.",
        "original_decision": original_decision,
    }), 200


@app.route("/appeals", methods=["GET"])
def appeals():
    """Reviewer queue: every content currently awaiting human review."""
    return jsonify({"queue": get_appeals_queue()}), 200


@app.route("/log", methods=["GET"])
def log():
    """Return recent audit entries. ?limit=N optional. Open for grading visibility;
    a real deployment would require auth here."""
    try:
        limit = int(request.args.get("limit", LOG_DEFAULT_LIMIT))
    except ValueError:
        limit = LOG_DEFAULT_LIMIT
    return jsonify({"entries": get_recent_decisions(limit)}), 200


@app.route("/verify", methods=["POST"])
@limiter.limit("5 per minute")
def verify():
    """Provenance certificate / Verified-Human credential (stretch S2).

    The creator completes a verification step: they submit an original passage
    written on the spot. We run it through detection; if it does NOT read as AI,
    we issue a certificate tied to their creator_id. Future submissions by that
    creator carry a Verified-Human badge on their label.
    """
    data = request.get_json(silent=True) or {}
    creator_id = (data.get("creator_id") or "").strip()
    passage = (data.get("passage") or "").strip()

    if not creator_id:
        return jsonify({"error": "field 'creator_id' is required"}), 400
    if len(passage.split()) < 25:
        return jsonify({"error": "field 'passage' must be at least 25 words"}), 400

    sig = groq_signal(passage)
    p_style = stylometric_signal(passage)["p_style"]
    p_lex = lexical_signal(passage)["p_lex"]
    score = combine_signals(sig["p_llm"], p_style, p_lex, len(passage.split()), sig["degraded"])

    if score["attribution"] == "likely_ai":
        return jsonify({
            "verified": False,
            "reason": "verification passage itself reads as AI-generated; not issuing a credential",
            "attribution": score["attribution"],
            "p_ai": score["p_ai"],
        }), 200

    cert_id = str(uuid.uuid4())
    insert_certificate(cert_id, creator_id, method="original_passage_challenge")
    add_event(cert_id, "verified", {
        "creator_id": creator_id,
        "method": "original_passage_challenge",
        "passage_attribution": score["attribution"],
        "passage_p_ai": score["p_ai"],
    })
    return jsonify({
        "verified": True,
        "certificate_id": cert_id,
        "creator_id": creator_id,
        "badge": "✅ Verified Human Creator",
        "message": "Verification passed. This creator's content will display a Verified-Human badge.",
    }), 200


@app.route("/analytics", methods=["GET"])
def analytics():
    """Detection-pattern metrics as JSON (stretch S3)."""
    return jsonify(get_analytics()), 200


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Minimal HTML view of the analytics (stretch S3)."""
    a = get_analytics()
    vd = a["verdict_distribution"]
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(vd.items())
    ) or "<tr><td colspan=2>no data yet</td></tr>"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Provenance Guard — Analytics</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;color:#1a1a1a}}
 h1{{font-size:1.3rem}} table{{border-collapse:collapse;margin:8px 0}}
 td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}
 .metric{{font-size:1.6rem;font-weight:700}} .card{{display:inline-block;margin:8px 16px 8px 0}}
 .muted{{color:#666;font-size:.85rem}}
</style></head><body>
<h1>📊 Provenance Guard — Detection Analytics</h1>
<div class="card"><div class="metric">{a['total_submissions']}</div><div class="muted">total submissions</div></div>
<div class="card"><div class="metric">{a['appeal_rate']:.0%}</div><div class="muted">appeal rate</div></div>
<div class="card"><div class="metric">{(a['average_confidence'] or 0):.2f}</div><div class="muted">avg confidence</div></div>
<div class="card"><div class="metric">{a['signal_agreement_rate']:.0%}</div><div class="muted">signal agreement</div></div>
<div class="card"><div class="metric">{a['ai_verdict_appeal_rate']:.0%}</div><div class="muted">AI-verdict appeal rate<br>(false-positive proxy)</div></div>
<div class="card"><div class="metric">{a['verified_creators']}</div><div class="muted">verified creators</div></div>
<h2 style="font-size:1.05rem">Verdict distribution</h2>
<table><tr><th>verdict</th><th>count</th></tr>{rows}</table>
<p class="muted">Generated by GET /dashboard · JSON at GET /analytics</p>
</body></html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/healthz", methods=["GET"])
def healthz():
    key = os.environ.get("GROQ_API_KEY")
    return jsonify({
        "status": "ok",
        "groq_key_configured": bool(key and key != "your_key_here"),
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
