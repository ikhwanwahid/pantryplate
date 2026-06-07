"""Stage 2 multi-constraint reranker.

Public API:
    from src.reranker import Stage2Reranker             # the combiner
    from src.reranker import (                          # individual scorers
        pantry_score, nutrition_score, diet_score, diet_compliant,
    )

The reranker takes a Stage 1 candidate list + a persona and produces a
constraint-aware top-K ranking via:

    final(u, r) = s_diet × (αₜ·s_taste + αₚ·s_pantry + αₙ·s_nutrition)

See `docs/data_decisions.md` §§7-9 for the constraint design rationale
and `src/reranker/scores.py` for each scorer.
"""

from src.reranker.scores import (
    INGREDIENT_BLOCKLIST,
    STAPLES,
    TAG_FOR_RESTRICTION,
    diet_compliant,
    diet_score,
    get_staples_for_persona,
    missing_count,
    nutrition_score,
    pantry_score,
)
from src.reranker.combiner import Stage2Reranker

__all__ = [
    "Stage2Reranker",
    "STAPLES",
    "INGREDIENT_BLOCKLIST",
    "TAG_FOR_RESTRICTION",
    "get_staples_for_persona",
    "missing_count",
    "pantry_score",
    "nutrition_score",
    "diet_score",
    "diet_compliant",
]
