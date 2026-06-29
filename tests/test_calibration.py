"""Calibration regression tests (planning.md §2 "How we'll validate").

These are OFFLINE and DETERMINISTIC — they exercise the stylometric signal,
the lexical signal, and the combine_signals() scoring logic WITHOUT calling Groq
(the LLM is non-deterministic and costs API quota). The semantic signal is
verified separately by the live curl tests in the README.

Run:  python -m tests.test_calibration      (from the project root)
Exits 0 and prints "ALL CALIBRATION CHECKS PASSED" on success.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AI_THRESHOLD, HUMAN_THRESHOLD  # noqa: E402
from detection import combine_signals, lexical_signal, stylometric_signal  # noqa: E402

# --- Calibration corpus (from the project spec) -----------------------------
CLEAR_AI = (
    "Artificial intelligence represents a transformative paradigm shift in modern society. "
    "It is important to note that while the benefits of AI are numerous, it is equally "
    "essential to consider the ethical implications. Furthermore, stakeholders across "
    "various sectors must collaborate to ensure responsible deployment."
)
CLEAR_HUMAN = (
    "ok so i finally tried that new ramen place downtown and honestly? underwhelming. "
    "the broth was fine but they put WAY too much sodium in it and i was thirsty for like "
    "three hours after. my friend got the spicy version and said it was better. probably "
    "won't go back unless someone drags me there"
)
FORMAL_HUMAN = (
    "The relationship between monetary policy and asset price inflation has been extensively "
    "studied in the literature. Central banks face a fundamental tension between their mandate "
    "for price stability and the unintended consequences of prolonged low interest rates on "
    "equity and real estate valuations."
)

_passed = 0


def check(name, condition, detail=""):
    global _passed
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    assert condition, f"{name} failed: {detail}"
    _passed += 1


def test_stylometric_ordering():
    print("\nStylometric signal (higher = more AI-like):")
    ai = stylometric_signal(CLEAR_AI)["p_style"]
    human = stylometric_signal(CLEAR_HUMAN)["p_style"]
    print(f"    clear_ai={ai}  clear_human={human}")
    check("AI text scores more AI-like than casual human", ai > human, f"{ai} > {human}")


def test_lexical_ordering():
    print("\nLexical signal (higher = more AI-like):")
    ai = lexical_signal(CLEAR_AI)["p_lex"]
    human = lexical_signal(CLEAR_HUMAN)["p_lex"]
    print(f"    clear_ai={ai}  clear_human={human}")
    check("AI cliche/formal text scores more AI-like", ai > human, f"{ai} > {human}")


def test_combine_thresholds():
    print("\ncombine_signals() verdict thresholds:")
    # Unanimous high -> likely_ai
    hi = combine_signals(0.85, 0.85, 0.85, word_count=60, degraded=False)
    check("unanimous high -> likely_ai", hi["attribution"] == "likely_ai", str(hi))
    check("high case is confident", hi["confidence"] >= 0.5, str(hi["confidence"]))

    # Unanimous low -> likely_human
    lo = combine_signals(0.10, 0.10, 0.10, word_count=60, degraded=False)
    check("unanimous low -> likely_human", lo["attribution"] == "likely_human", str(lo))

    # Mid -> uncertain (the false-positive trap should NOT be called AI)
    mid = combine_signals(0.55, 0.60, 0.50, word_count=60, degraded=False)
    check("mid blend -> uncertain", mid["attribution"] == "uncertain", str(mid))

    # Signals at war -> forced uncertain even if the mean lands in a band
    war = combine_signals(0.95, 0.10, 0.95, word_count=60, degraded=False)
    check("high-disagreement -> uncertain", war["attribution"] == "uncertain", str(war))

    # Disagreement damps confidence vs. agreement AT THE SAME p_ai.
    # Both blends below have weighted mean exactly 0.80; only the spread differs.
    agree = combine_signals(0.8, 0.8, 0.8, word_count=60, degraded=False)["confidence"]
    disagree = combine_signals(0.8, 1.0, 0.5, word_count=60, degraded=False)["confidence"]
    check("disagreement lowers confidence", disagree < agree, f"{disagree} < {agree}")


def test_short_text_guard():
    print("\nShort-text guard:")
    short = combine_signals(0.95, 0.95, 0.95, word_count=10, degraded=False)
    check("<40 words forced uncertain", short["attribution"] == "uncertain", str(short))
    check("short text flagged low_evidence", short["low_evidence"] is True)


def test_degraded_caps_confidence():
    print("\nGraceful degradation (Groq unavailable):")
    deg = combine_signals(None, 0.9, 0.9, word_count=60, degraded=True)
    check("degraded confidence capped at 0.60", deg["confidence"] <= 0.60, str(deg["confidence"]))


def test_full_pipeline_offline():
    """Simulate the full blend using fixed LLM scores (what the live LLM returns
    for these inputs, per Milestone 3 testing) to confirm end-to-end buckets."""
    print("\nFull pipeline (fixed p_llm to keep it deterministic):")
    cases = [
        ("clear_ai", CLEAR_AI, 0.8, "likely_ai"),
        ("clear_human", CLEAR_HUMAN, 0.2, "likely_human"),
        ("formal_human", FORMAL_HUMAN, 0.5, "uncertain"),  # the false-positive trap
    ]
    for name, text, p_llm, expected in cases:
        p_style = stylometric_signal(text)["p_style"]
        p_lex = lexical_signal(text)["p_lex"]
        out = combine_signals(p_llm, p_style, p_lex, len(text.split()), degraded=False)
        print(f"    {name:13s} p_ai={out['p_ai']:.2f} conf={out['confidence']:.2f} "
              f"-> {out['attribution']} (expected {expected})")
        check(f"{name} -> {expected}", out["attribution"] == expected,
              f"got {out['attribution']}")


if __name__ == "__main__":
    test_stylometric_ordering()
    test_lexical_ordering()
    test_combine_thresholds()
    test_short_text_guard()
    test_degraded_caps_confidence()
    test_full_pipeline_offline()
    print(f"\nALL CALIBRATION CHECKS PASSED ({_passed} checks)")
