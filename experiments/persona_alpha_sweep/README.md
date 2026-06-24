# Per-persona α-sweep (Hypothesis 2 backup analysis)

Tests the proposal's **Hypothesis 2** — *does the optimal (αₜ, αₚ, αₙ) shift with
the kind of user?* The main α-sweep (`src/eval/alpha_sweep.py`,
`notebooks/alpha_sweep.ipynb`) derives each user's constraints from their own
history and averages over 2000 users, so it can't isolate the effect of the
*constraint profile*. This re-runs the same 2000 users / same held-out items,
but swaps every user's pantry/macros/restrictions for one persona's fixed
profile at a time (fitness_focused / vegan_busy / family_friendly).

Author: Anastasia Frederica. Kept as **backup analysis** — the Hypothesis 2
result was inconclusive — not part of the headline α-sweep deliverable.

## Files
| File | What |
|---|---|
| `persona_alpha_sweep.py` | Runs the 3 per-persona sweeps (uses BPR Stage 1, same eval convention as the global sweep) |
| `persona_sweep_plot.py` | Ternary comparison plot (feasible_rate@10 across the 3 personas) |

## Outputs (written to `data/processed/`)
`alpha_sweep_persona_{fitness_focused,vegan_busy,family_friendly}.csv`,
`alpha_sweep_persona_summary.csv`, and `alpha_sweep_persona_ternary.png`.

## Running
From the project root (needs the Food.com data in `data/raw/`):

```bash
uv run python experiments/persona_alpha_sweep/persona_alpha_sweep.py   # produces the CSVs
uv run python experiments/persona_alpha_sweep/persona_sweep_plot.py    # then the ternary plot
```
