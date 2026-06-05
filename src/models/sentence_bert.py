"""Sentence-BERT content recommender — Week 4 content baseline.

Embeds each recipe's text (name + ingredients + tags) into a 384-dim
sentence-transformers vector, builds per-user profiles by averaging the
L2-normalized embeddings of items the user rated positively, and ranks
candidates by cosine similarity to the user vector.

Why this model: cold-track Recall on the authors' test set is exactly 0 for
any pure-CF model (cold items have no rater history). A content-aware model
that uses recipe text can produce non-zero numbers on cold and that is the
"content paradigm pays off" win for the proposal's headline plot.

Convention (locked decision 11):
    .fit(train_df)              -> returns self
    .recommend(user_id, k, exclude_seen=True) -> list[int]

Notes:
- Embeddings are cached to data/processed/recipe_sbert_<model>.npy keyed
  by row order matching `self.recipe_ids`. First fit takes ~5-10 min;
  subsequent fits load the cache in seconds.
- Cold users (no positives in train_df) fall back to popularity ranking.
- L2-normalized embeddings → cosine similarity is a dot product.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.data.loader import POSITIVE_THRESHOLD, load_recipes


SBERT_CACHE_DIR = Path("data/processed")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_recipe_text(row: pd.Series) -> str:
    """name | ingredients | tags — one string per recipe.

    Uses parsed list columns when available; falls back to raw string parse.
    """
    name = str(row.get("name") or "").strip()

    ings = row.get("ingredients_parsed")
    if isinstance(ings, list):
        ings_text = ", ".join(ings)
    else:
        ings_text = re.sub(r"[\[\]'\"]", "", str(row.get("ingredients") or "")).strip()

    tags = row.get("tags_parsed")
    if isinstance(tags, list):
        tags_text = ", ".join(tags)
    else:
        tags_text = re.sub(r"[\[\]'\"]", "", str(row.get("tags") or "")).strip()

    return f"{name} | {ings_text} | {tags_text}"


def _cache_path_for_model(model_name: str, cache_dir: Path) -> Path:
    safe = model_name.replace("/", "__")
    return cache_dir / f"recipe_sbert_{safe}.npy"


class SentenceBERTRecommender:
    """Cosine-similarity recommender over sentence-transformers recipe text embeddings.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model identifier. Default: MiniLM-L6-v2
        (384-dim, ~80MB, runs fast on CPU/MPS).
    positive_threshold : int
        Rating ≥ this counts as a positive for user-profile construction.
        Defaults to the project-wide POSITIVE_THRESHOLD (= 4).
    cache_dir : Path
        Where recipe embeddings get cached (and reloaded across fits).
    batch_size : int
        Encoder batch size. 128 is safe on most laptops; bump higher on GPU.
    device : str | None
        Torch device override. None lets sentence-transformers auto-detect
        (uses MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU).
    force_rebuild : bool
        If True, re-encode all recipes even if a cache file exists.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        positive_threshold: int = POSITIVE_THRESHOLD,
        cache_dir: Path = SBERT_CACHE_DIR,
        batch_size: int = 128,
        device: str | None = None,
        force_rebuild: bool = False,
    ):
        self.model_name = model_name
        self.positive_threshold = positive_threshold
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.device = device
        self.force_rebuild = force_rebuild

        # Populated by fit()
        self.recipe_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._recipe_id_to_row: dict[int, int] = {}
        self._recipe_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)  # (n_recipes, dim), L2-normalized
        self._user_vectors: dict[int, np.ndarray] = {}                         # user_id -> (dim,) L2-normalized
        self._popularity_rank: np.ndarray = np.empty(0, dtype=np.int64)        # cold-user fallback
        self._user_seen: dict[int, set[int]] = {}

    # ---------------------------------------------------------------------
    # fit
    # ---------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "SentenceBERTRecommender":
        if not {"user_id", "recipe_id", "rating"}.issubset(train_df.columns):
            raise ValueError("train_df must have 'user_id', 'recipe_id', 'rating' columns")

        # 1. Build / load recipe embeddings for the FULL catalogue (cold-track requires this)
        recipes = load_recipes()
        recipes["id"] = recipes["id"].astype(np.int64)
        self.recipe_ids = recipes["id"].to_numpy(copy=False)
        self._recipe_id_to_row = {int(rid): i for i, rid in enumerate(self.recipe_ids)}

        cache_path = _cache_path_for_model(self.model_name, self.cache_dir)
        if cache_path.exists() and not self.force_rebuild:
            cached = np.load(cache_path)
            if cached.shape[0] != len(self.recipe_ids):
                raise RuntimeError(
                    f"Cache at {cache_path} has {cached.shape[0]} rows but the "
                    f"recipe catalogue has {len(self.recipe_ids)}. Delete the cache "
                    f"or pass force_rebuild=True."
                )
            self._recipe_matrix = cached.astype(np.float32, copy=False)
        else:
            self._recipe_matrix = self._encode_recipes(recipes)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self._recipe_matrix)

        # 2. Per-user profile vectors from positives in train
        positives = train_df[train_df["rating"] >= self.positive_threshold]
        self._user_vectors = self._build_user_profiles(positives)

        # 3. Popularity rank as cold-user fallback (same logic as PopularityRecommender)
        popularity = train_df.groupby("recipe_id")["user_id"].nunique()
        # Some popular recipe_ids may not be in the recipe catalogue (rare, but possible)
        popularity = popularity[popularity.index.isin(self._recipe_id_to_row)]
        self._popularity_rank = popularity.sort_values(ascending=False).index.to_numpy()

        # 4. Seen-set per user, for exclude_seen handling
        self._user_seen = (
            train_df.groupby("user_id")["recipe_id"]
            .agg(set)
            .to_dict()
        )
        return self

    # ---------------------------------------------------------------------
    # recommend
    # ---------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        if self._recipe_matrix.size == 0:
            raise RuntimeError("Call .fit(train_df) before .recommend(...)")

        user_vec = self._user_vectors.get(int(user_id))
        if user_vec is None:
            return self._popularity_fallback(user_id, k, exclude_seen)

        scores = self._recipe_matrix @ user_vec  # (n_recipes,) cosine similarity (both L2-normed)

        if exclude_seen:
            seen = self._user_seen.get(int(user_id), set())
            if seen:
                seen_rows = [self._recipe_id_to_row[r] for r in seen if r in self._recipe_id_to_row]
                if seen_rows:
                    scores = scores.copy()
                    scores[seen_rows] = -np.inf

        # argpartition for top-k is O(n) vs O(n log n) for full sort
        k_eff = min(k, scores.size)
        top_unsorted = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top_sorted = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [int(self.recipe_ids[i]) for i in top_sorted]

    def recommend_many(
        self,
        user_ids: Iterable[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, List[int]]:
        return {int(uid): self.recommend(int(uid), k=k, exclude_seen=exclude_seen) for uid in user_ids}

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _encode_recipes(self, recipes: pd.DataFrame) -> np.ndarray:
        """Encode all recipe texts to (n_recipes, dim), L2-normalized."""
        # Import lazily so importing this module is cheap when the cache is warm
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(self.model_name, device=self.device)
        texts = [_build_recipe_text(row) for _, row in recipes.iterrows()]
        emb = encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize so dot product == cosine sim
        )
        return emb.astype(np.float32, copy=False)

    def _build_user_profiles(self, positives: pd.DataFrame) -> dict[int, np.ndarray]:
        """Average L2-normalized recipe embeddings per user, then re-L2 the mean."""
        profiles: dict[int, np.ndarray] = {}
        # Group by user, average their positive items' vectors
        for user_id, group in positives.groupby("user_id"):
            rows = [
                self._recipe_id_to_row[int(rid)]
                for rid in group["recipe_id"].to_numpy()
                if int(rid) in self._recipe_id_to_row
            ]
            if not rows:
                continue
            mean_vec = self._recipe_matrix[rows].mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm == 0:
                continue
            profiles[int(user_id)] = (mean_vec / norm).astype(np.float32, copy=False)
        return profiles

    def _popularity_fallback(self, user_id: int, k: int, exclude_seen: bool) -> List[int]:
        if exclude_seen:
            seen = self._user_seen.get(int(user_id), set())
            out: list[int] = []
            for item in self._popularity_rank:
                if int(item) not in seen:
                    out.append(int(item))
                    if len(out) >= k:
                        break
            return out
        return [int(i) for i in self._popularity_rank[:k]]
