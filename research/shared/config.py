"""Shared constants for all research experiments."""

from pathlib import Path

# ── Infrastructure ─────────────────────────────────────────────────────────

LOCAL_LLM_BASE_URL = ""
LOCAL_LLM_MODEL = ""
LOCAL_EMBED_ENDPOINT = ""

# ── Evaluation settings ────────────────────────────────────────────────────

NUM_EVAL_QUESTIONS = 100
TOP_K = 5

# ── Dataset ────────────────────────────────────────────────────────────────

# HotpotQA dev distractor set — self-contained (10 passages per question)
HOTPOTQA_DEV_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)

# ── Paths ──────────────────────────────────────────────────────────────────

RESEARCH_DIR = Path(__file__).parent.parent
DATA_DIR = RESEARCH_DIR / "data"
RESULTS_DIR = RESEARCH_DIR / "results"
