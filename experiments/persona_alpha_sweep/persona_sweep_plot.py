"""Ternary comparison plot for the per-persona α-sweep (see persona_alpha_sweep.py).

One row of 3 ternary panels — feasible_rate@10 (the clean, non-degenerate
metric) for fitness_focused / vegan_busy / family_friendly — so the deck can
show that the optimal α and the achievable ceiling both shift by persona.
Run from the project root after persona_alpha_sweep.py has produced the CSVs:

    python experiments/persona_alpha_sweep/persona_sweep_plot.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

PERSONAS = [
    ("fitness_focused", "Fitness-focused"),
    ("vegan_busy", "Vegan & busy"),
    ("family_friendly", "Family-friendly"),
]

_A = np.array([0.5, np.sqrt(3) / 2])
_B = np.array([0.0, 0.0])
_C = np.array([1.0, 0.0])


def _xy(at, ap, an):
    at, ap, an = np.asarray(at), np.asarray(ap), np.asarray(an)
    return at * _A[0] + ap * _B[0] + an * _C[0], at * _A[1] + ap * _B[1] + an * _C[1]


def ternary(ax, df, col, title, cmap):
    x, y = _xy(df.alpha_taste, df.alpha_pantry, df.alpha_nutrition)
    z = df[col].to_numpy() * 100
    tcf = ax.tricontourf(mtri.Triangulation(x, y), z, levels=14, cmap=cmap)
    ax.plot([_A[0], _B[0], _C[0], _A[0]], [_A[1], _B[1], _C[1], _A[1]], color="#2D3142", lw=1.1)
    bi = df[col].idxmax()
    bx, by = _xy(df.loc[bi].alpha_taste, df.loc[bi].alpha_pantry, df.loc[bi].alpha_nutrition)
    ax.scatter([bx], [by], marker="*", s=300, color="#C75D2C", edgecolor="white", zorder=5)
    ax.text(_A[0], _A[1] + 0.05, "taste", ha="center", fontsize=9)
    ax.text(_B[0] - 0.02, _B[1] - 0.03, "pantry", ha="right", va="top", fontsize=9)
    ax.text(_C[0] + 0.02, _C[1] - 0.03, "nutrition", ha="left", va="top", fontsize=9)
    ax.set_title(f"{title}\nmax {z.max():.1f}%  @ ({df.loc[bi].alpha_taste:.2f},{df.loc[bi].alpha_pantry:.2f},{df.loc[bi].alpha_nutrition:.2f})", fontsize=10)
    ax.axis("off")
    ax.set_aspect("equal")
    plt.colorbar(tcf, ax=ax, shrink=0.7)


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (pid, label) in zip(axes, PERSONAS):
    df = pd.read_csv(f"data/processed/alpha_sweep_persona_{pid}.csv")
    ternary(ax, df, "feasible_rate@10", f"{label}\nPANTRY-FEASIBLE rate · top-10", "Greens")
fig.suptitle("Per-persona α-sweep — same 2000 users/held-out items, fixed persona constraints (BPR/warm)", y=1.03, fontsize=12)
plt.tight_layout()
plt.savefig("data/processed/alpha_sweep_persona_ternary.png", dpi=150, bbox_inches="tight")
print("saved data/processed/alpha_sweep_persona_ternary.png")
