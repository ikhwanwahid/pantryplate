# Deep-CF exploration (LightGCN + NeuMF)

**Status: EXPLORATORY. Not part of the sanctioned Stage 1 model menu** (see
`docs/data_decisions.md` locked decision #5). These models are *not* wired into
`src/models/`, the eval harness's model registry, the leaderboard, or the
Stage 1 → Stage 2 pipeline. They live here as a self-contained robustness
check, kept for the write-up.

Author: Anastasia Frederica. Reorganized into `experiments/` (from loose files
at the repo root) without changing the models or results.

## Why this exists

The headline Stage 1 finding is a **null result**: tuned CF (BPR, EASE, ALS)
does not significantly beat a popularity baseline on this sparse Food.com slice
(see `docs/stage1_leaderboard.md` and the significance tests). The obvious
follow-up question for the deck is *"did you only try shallow models?"* This
exploration answers it: **even modern deep CF — graph convolution (LightGCN)
and neural matrix factorization (NeuMF) — lands in the same band as
popularity.** It's the data, not the model class.

## Results (warm track, n_users=2000, seed=42, k=(10,100) — same convention as the leaderboard)

| Model | Recall@10 | Recall@100 | Cold@10 / @100 |
|---|---|---|---|
| Popularity (baseline) | 2.95 | 11.55 | 0 / 0 |
| BPR | 2.70 | 12.25 | 0 / 0 |
| EASE | 2.90 | 8.10 | 0 / 0 |
| ALS | 2.05 | 7.95 | 0 / 0 |
| **LightGCN** *(exploratory)* | **3.05** | **11.3** | 0 / 0 |
| **NeuMF** *(exploratory)* | **2.05** | **9.85** | 0 / 0 |

(Recall ×100.) LightGCN is within noise of popularity/EASE/BPR; NeuMF is
slightly worse. Both score 0 on the cold track by construction (pure CF, no
item content) — expected, same as the other CF models.

Hyperparameter sweeps confirm there's no hidden win:
- `lightgcn_dim_sweep.csv` — embedding width 8→128: Recall@10 stays 0.030–0.0315.
- `lightgcn_layers_sweep.csv` — 1→4 propagation layers: Recall@10 stays 0.030–0.0315.
- `lightgcn_epoch_curve` — recall saturates by ~epoch 10–20; more training doesn't help.

**Takeaway for the deck:** deep CF doesn't move the needle here. Reinforces the
"sparse interactions cap CF; content is what unlocks the cold track" story.

## Files

| File | What |
|---|---|
| `lightgcn.py` | LightGCN recommender (follows the `.fit()/.recommend()` contract) |
| `neumf.py` | NeuMF recommender (same contract) |
| `deep_cf_compare.py` | Runs LightGCN/NeuMF vs Popularity/EASE/BPR on the warm track |
| `lightgcn_dim_sweep.py` / `.csv` / `.png` | Embedding-width sweep |
| `lightgcn_layers_sweep.py` / `.csv` / `.png` | Propagation-depth sweep |
| `lightgcn_epoch_curve.py` / `.png` | Training-length saturation curve |

## Running

Needs `torch` (already a project dependency). Run from the project root:

```bash
uv run python experiments/deep_cf/deep_cf_compare.py
```

The sweep scripts (`lightgcn_dim_sweep.py`, etc.) additionally write their CSV +
PNG into this directory. Each is GPU-optional (CPU works; slower).
