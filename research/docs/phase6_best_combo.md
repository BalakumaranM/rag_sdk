# Phase 6 — Best-of-Breed Combination

## Purpose

Combine the winning component from each phase into a single pipeline and measure
the total improvement over the Phase 1 baseline.

This is the payoff phase — it answers: **was all that optimisation worth it?**

---

## What Changes

Every other phase changed one variable and held the rest constant. Phase 6 changes
everything at once, deliberately:

```
Baseline (6a)                    Best-of-breed (6b)
─────────────────────────────    ──────────────────────────────────────────
Chunking:  TextSplitter(512,50)  Chunking:  Phase 2 winner
Retrieval: Dense                 Retrieval: Phase 3 winner
Reranking: None                  Reranking: Phase 4 winner (if it helps)
Generation:Standard              Generation:Phase 5 winner
```

---

## Why Re-run the Baseline?

Phase 1 used `per_question` mode (~20 chunks per question). Phases 3–5 used
`shared_corpus` mode (~600–1000 chunks total). Comparing Phase 1's result file
directly against Phase 6's best-of-breed would be comparing different corpus
sizes — an unfair comparison.

Phase 6 re-runs the Phase 1 configuration (6a) in `shared_corpus` mode so both
variants face exactly the same retrieval difficulty.

---

## How to Update the Winner Constants

After completing Phases 2–5, open `research/phase6_best_combo/run.py` and update
the six constants at the top:

```python
# Phase 2 chunking winner — check research/results/phase2_chunking_*.json
BEST_CHUNKING_STRATEGY = "recursive"   # e.g. "semantic", "proposition"
BEST_CHUNK_SIZE = 512                  # e.g. 256, 1024
BEST_CHUNK_OVERLAP = 50               # e.g. 25, 100

# Phase 3 retrieval winner — check research/results/phase3_retrieval_*.json
BEST_RETRIEVAL_STRATEGY = "dense"     # e.g. "hybrid", "multi_query"

# Phase 4 reranking winner — check research/results/phase4_reranking_*.json
BEST_RERANKING_PROVIDER = None        # e.g. "cross-encoder", "cohere"

# Phase 5 generation winner — check research/results/phase5_generation_*.json
BEST_GENERATION_STRATEGY = "standard" # e.g. "cove"
```

**Which metric to use when picking a winner:**

| Phase | Primary metric | Tiebreaker |
|-------|---------------|-----------|
| 2 (Chunking) | context_recall | F1 |
| 3 (Retrieval) | context_recall | F1 |
| 4 (Reranking) | context_precision | F1 |
| 5 (Generation) | F1 | exact_match |

If the Phase 4 reranking winner is "None" (no reranking), set
`BEST_RERANKING_PROVIDER = None` and the pipeline will skip the reranking step.

---

## How to Run

```bash
cd /path/to/rag_sdk

# Full comparison (runs 6a baseline + 6b best-of-breed):
.venv/bin/python research/phase6_best_combo/run.py

# Re-run only the best-of-breed (baseline already cached):
.venv/bin/python research/phase6_best_combo/run.py --variants 6b --force

# Re-run everything from scratch:
.venv/bin/python research/phase6_best_combo/run.py --force
```

Results saved to:
- `research/results/phase6_best_combo_6a.json`
- `research/results/phase6_best_combo_6b.json`

---

## How to Read the Output

```
======================================================================
Phase 6 — Best-of-Breed vs Baseline
======================================================================

  Pipeline components:
  ┌──────────────┬────────────────────────────┬────────────────────────────┐
  │ Component    │ Baseline (6a)              │ Best-of-breed (6b)         │
  ├──────────────┼────────────────────────────┼────────────────────────────┤
  │ Chunking     │ TextSplitter(512, 50)      │ SemanticSplitter           │
  │ Retrieval    │ Dense                      │ hybrid                     │
  │ Reranking    │ None                       │ cross-encoder              │
  │ Generation   │ Standard                   │ standard                   │
  └──────────────┴────────────────────────────┴────────────────────────────┘

  Metrics (shared_corpus mode, 100 questions):

  Metric                Baseline    Best     Delta
  ──────────────────────────────────────────────────────────
  Context Recall           0.643   0.731   +0.088 (+13.7%)
  Context Precision        0.389   0.467   +0.078 (+20.1%)
  Exact Match              0.281   0.341   +0.060 (+21.4%)
  F1                       0.451   0.523   +0.072 (+16.0%)

  Avg Latency              2.34s   3.12s   +0.78s (+33.3%)

======================================================================

  Verdict: Strong improvement — optimisation clearly paid off.
  F1 gain: +0.072  |  Recall gain: +0.088  |  Latency cost: +0.78s
```

(Numbers above are illustrative hypotheses, not actual results.)

**Interpreting the delta:**

| F1 gain | Interpretation |
|---------|---------------|
| > 0.05 | Strong — chunking/retrieval improvements have a real impact |
| 0.01–0.05 | Moderate — gains exist but are smaller than expected |
| ≈ 0 | Flat — HotpotQA distractor setup may be too constrained to show improvements |
| Negative | Regression — a "winner" from an earlier phase actually hurts the combination |

**If the result is flat or negative:**

A flat result doesn't mean the optimisations don't work — it may mean:
1. The per-question isolation in Phases 1–2 made the task too easy, hiding differences
   that only emerge in production
2. The Phase 2–5 winners were chosen by per-phase metrics that don't transfer to
   the combined pipeline (component interactions)
3. HotpotQA's distractor format creates an artificial ceiling on retrieval recall
   (gold docs always in the pool)

In that case, Phase 7 (real PDF corpus) will be more informative.

---

## What Phase 7 Will Change

Phase 7 takes the Phase 6 best-of-breed pipeline and runs it on a real domain
PDF corpus with synthetic Q&A pairs, testing whether the optimisations generalise
beyond the Wikipedia/HotpotQA distribution.
