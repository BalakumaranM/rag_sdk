"""Shared constants for all research experiments."""

from pathlib import Path

# ── Infrastructure ─────────────────────────────────────────────────────────

LOCAL_LLM_BASE_URL = ""
LOCAL_LLM_MODEL = ""
LOCAL_EMBED_ENDPOINT = ""

# ── Evaluation settings ────────────────────────────────────────────────────

NUM_EVAL_QUESTIONS = 100
NUM_QASPER_QUESTIONS = 100
TOP_K = 5

# ── Paths ──────────────────────────────────────────────────────────────────

RESEARCH_DIR = Path(__file__).parent.parent
DATA_DIR = RESEARCH_DIR / "data"
RESULTS_DIR = RESEARCH_DIR / "results"

# ── Dataset file locations ──────────────────────────────────────────────────
# Place these files here before running any experiment.
# The loaders raise FileNotFoundError with download instructions if missing.

HOTPOTQA_DATA_PATH = DATA_DIR / "hotpotqa_dev_distractor.json"
QASPER_DATA_PATH = DATA_DIR / "qasper_dev_v0.3.json"
