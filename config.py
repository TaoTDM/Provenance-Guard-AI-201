"""Central configuration for Provenance Guard.

Every tunable lives here so weights/thresholds can be recalibrated in one place
(see planning.md §1, §2). Milestone 3 only uses GROQ_MODEL, DB_PATH, and the
verdict thresholds; the ensemble weights are defined now so M4 can wire them in
without touching this file again.
"""

import os

# --- Detection signal 1: Groq LLM ---
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Ensemble weights (planning.md §1) — used from Milestone 4 onward ---
WEIGHTS = {"llm": 0.50, "style": 0.30, "lex": 0.20}

# --- Confidence scoring / verdict thresholds (planning.md §2) ---
# Asymmetric on purpose: a higher bar to call AI (0.15 above the 0.50 midpoint)
# than to clear a human (0.12 below it) — protecting creators from false positives.
AI_THRESHOLD = 0.65        # p_ai must be >= this to return likely_ai
HUMAN_THRESHOLD = 0.38     # p_ai must be <= this to return likely_human
DISAGREEMENT_MAX = 0.33    # if signals disagree more than this -> forced "uncertain"
SHORT_TEXT_WORDS = 40      # below this word count -> forced "uncertain"

# --- Storage ---
DB_PATH = os.path.join(os.path.dirname(__file__), "provenance_guard.sqlite")

# --- Audit log ---
LOG_DEFAULT_LIMIT = 20     # how many recent entries GET /log returns by default
