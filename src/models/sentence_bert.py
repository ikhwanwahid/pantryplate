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


VALID_PROFILE_STRATEGIES = ("mean", "rating_weighted", "recency_weighted")


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
    profile_strategy : {"mean", "rating_weighted", "recency_weighted"}
        How to aggregate a user's positive recipes into a single profile vector.
        - "mean" : simple mean of L2-normalized item vectors (baseline).
        - "rating_weighted" : weight by max(rating - 3, 0) so 5★ counts 2x more
          than 4★. Captures intensity of preference.
        - "recency_weighted" : exponential decay by days from user's most recent
          rating (half-life = `recency_half_life_days`). Captures evolving taste.
        All strategies re-L2 the resulting mean so cosine sim is well-defined.
    recency_half_life_days : int
        Half-life of the recency weight (in days). Only used by recency_weighted.
        Default 180 (≈6 months): a 1-year-old rating counts ~25%; a 3-year-old
        rating counts ~3%.
    tag_feature_weight : float in [0, 1]
        If > 0, concatenate the 107-dim Tag SVD + nutrition vector from
        `src/data/features.py` onto the SBERT embedding (per-block L2-norm,
        sqrt-weighted, then re-L2). Algebraically this makes cosine in the
        combined space equal to:
            (1 - tag_feature_weight) * cos(sbert_parts) +
            tag_feature_weight       * cos(tag_parts)
        - tag_feature_weight=0.0 (default): pure SBERT (current behavior)
        - tag_feature_weight=1.0: pure Tag SVD/nutrition
        - intermediate: weighted blend of two CONTENT spaces (unlike layer 3
          which blends content vs popularity)
        This is fit-time because the recipe matrix dimensionality changes.
    content_weight : float in [0, 1]
        Stage-1-internal blend between content and popularity signals:
            score = content_weight * cosine + (1 - content_weight) * popularity_score
        Both terms are pre-normalized into [0, 1] so the weight is interpretable.
        - content_weight=1.0 (default): pure SBERT cosine. Optimal for cold-track.
        - content_weight=0.0: pure popularity. Equivalent to PopularityRecommender
          on warm-track (content acts as a 0 tiebreaker).
        - content_weight~0.5: balanced blend. Useful when popularity has signal
          (warm) but content adds taste personalization.
        Cold recipes have popularity_score=0 by construction, so content_weight<1
        always hurts cold-track Recall@K — this is correct, not a bug.

        NOTE: this α is INTERNAL to Stage 1. It is distinct from the deck's
        Stage 2 (αₜ, αₚ, αₙ) constraint-weight simplex. This knob produces
        a better s_taste for Stage 2 to consume; it does not replace the
        Stage 2 reranker.
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
        profile_strategy: str = "mean",
        recency_half_life_days: int = 180,
        tag_feature_weight: float = 0.0,
        content_weight: float = 1.0,
        cache_dir: Path = SBERT_CACHE_DIR,
        batch_size: int = 128,
        device: str | None = None,
        force_rebuild: bool = False,
    ):
        if profile_strategy not in VALID_PROFILE_STRATEGIES:
            raise ValueError(
                f"profile_strategy must be one of {VALID_PROFILE_STRATEGIES}, "
                f"got {profile_strategy!r}"
            )
        if not 0.0 <= content_weight <= 1.0:
            raise ValueError(
                f"content_weight must be in [0, 1], got {content_weight}"
            )
        if not 0.0 <= tag_feature_weight <= 1.0:
            raise ValueError(
                f"tag_feature_weight must be in [0, 1], got {tag_feature_weight}"
            )
        self.model_name = model_name
        self.positive_threshold = positive_threshold
        self.profile_strategy = profile_strategy
        self.recency_half_life_days = recency_half_life_days
        self.tag_feature_weight = tag_feature_weight
        self.content_weight = content_weight
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
        self._popularity_score: np.ndarray = np.empty(0, dtype=np.float32)     # per-recipe pop in [0, 1]
        self._user_seen: dict[int, set[int]] = {}

    # ---------------------------------------------------------------------
    # fit
    # ---------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        recipes_df: pd.DataFrame | None = None,
    ) -> "SentenceBERTRecommender":
        """Fit the recommender.

        Parameters
        ----------
        train_df : pd.DataFrame
            Interactions with user_id / recipe_id / rating columns.
        recipes_df : pd.DataFrame, optional
            Pre-loaded recipe catalogue (from `load_recipes()`). Pass this to
            avoid re-parsing the 230K-row CSV when the caller already has it
            in memory (e.g. the Streamlit app) — roughly halves cold start.
            Must contain an "id" column. If None, recipes are loaded internally.
            Only used when the embedding cache needs (re)building or for the
            recipe-id index; the cached embedding matrix path is unchanged.
        """
        if not {"user_id", "recipe_id", "rating"}.issubset(train_df.columns):
            raise ValueError("train_df must have 'user_id', 'recipe_id', 'rating' columns")

        # 1. Build / load recipe embeddings for the FULL catalogue (cold-track requires this)
        recipes = load_recipes() if recipes_df is None else recipes_df.copy()
        if "id" not in recipes.columns:
            # Allow a recipe_id-indexed frame (what the Streamlit app holds)
            if recipes.index.name in ("id", "recipe_id"):
                recipes = recipes.reset_index().rename(columns={"recipe_id": "id"})
            else:
                raise ValueError("recipes_df must have an 'id' column or be indexed by recipe id")
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

        # 1b. Layer 4: optional concat with Tag SVD + nutrition features.
        # Each block is L2-normed individually, sqrt-weighted, then concatenated;
        # the result is re-L2-normed so cosine in the combined space equals
        # (1-w) * cos(sbert) + w * cos(tag).
        if self.tag_feature_weight > 0:
            self._recipe_matrix = self._concat_tag_features(self._recipe_matrix)

        # 2. Per-user profile vectors from positives in train
        positives = train_df[train_df["rating"] >= self.positive_threshold]
        self._user_vectors = self._build_user_profiles(positives)

        # 3. Popularity rank (for cold-user fallback) + per-recipe popularity score
        #    (for the blend with cosine similarity).
        popularity = train_df.groupby("recipe_id")["user_id"].nunique()
        popularity = popularity[popularity.index.isin(self._recipe_id_to_row)]
        self._popularity_rank = popularity.sort_values(ascending=False).index.to_numpy()

        # Per-recipe popularity score in [0, 1]: rank-based normalization makes it
        # robust to the heavy popularity tail. Recipes with zero train ratings
        # (i.e. cold) stay at 0, which is what we want for blending.
        pop_score = np.zeros(len(self.recipe_ids), dtype=np.float32)
        n_with_signal = len(self._popularity_rank)
        if n_with_signal > 0:
            # rank 0 (most popular) → 1.0, rank n-1 (least popular) → ~0
            for rank, rid in enumerate(self._popularity_rank):
                row = self._recipe_id_to_row[int(rid)]
                pop_score[row] = 1.0 - (rank / n_with_signal)
        self._popularity_score = pop_score

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

        # Cosine similarity (both sides L2-normalized, so dot product == cosine).
        # Cosine values are in [-1, 1] but for normalized text embeddings they're
        # almost always in [0, 1], so blending with [0, 1] popularity is fine.
        scores = self._recipe_matrix @ user_vec  # (n_recipes,)

        w = self.content_weight
        if w < 1.0:
            scores = w * scores + (1.0 - w) * self._popularity_score

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
    # Seed-based recommend — for personas + walk-in users (no user_id needed)
    # ---------------------------------------------------------------------
    def recommend_for_seeds(
        self,
        seed_recipe_ids: Iterable[int],
        k: int = 100,
        exclude_seeds: bool = True,
    ) -> List[int]:
        """Recommend recipes similar to a list of seed recipe IDs.

        For personas (synthetic users with taste_seeds in their JSON) or for
        the cold-start onboarding flow where a new user picks a few starter
        recipes from a Spotify-style onboarding survey.

        The seed embeddings are averaged and re-L2-normalized to form a
        "persona vector"; recipes are ranked by cosine similarity to it.
        Seeds themselves are excluded from the result by default.

        Parameters
        ----------
        seed_recipe_ids : iterable of int
            Recipe IDs the persona/user has indicated as exemplary.
        k : int
            Number of recommendations to return.
        exclude_seeds : bool
            If True (default), drop the seed recipes from the result so
            the recommendation list is novel.

        Returns
        -------
        list[int] — top-k recipe_ids, ordered by cosine similarity desc.
        Falls back to popularity if NONE of the seeds are in the catalogue.
        """
        if self._recipe_matrix.size == 0:
            raise RuntimeError("Call .fit(train_df) before .recommend_for_seeds(...)")

        seed_rows = [
            self._recipe_id_to_row[int(rid)]
            for rid in seed_recipe_ids
            if int(rid) in self._recipe_id_to_row
        ]
        if not seed_rows:
            return self._popularity_fallback(user_id=-1, k=k, exclude_seen=False)

        seed_vec = self._recipe_matrix[seed_rows].mean(axis=0)
        norm = float(np.linalg.norm(seed_vec))
        if norm == 0:
            return self._popularity_fallback(user_id=-1, k=k, exclude_seen=False)
        seed_vec = (seed_vec / norm).astype(np.float32, copy=False)

        scores = self._recipe_matrix @ seed_vec  # (n_recipes,) cosine sim

        w = self.content_weight
        if w < 1.0:
            scores = w * scores + (1.0 - w) * self._popularity_score

        if exclude_seeds and seed_rows:
            scores = scores.copy()
            scores[seed_rows] = -np.inf

        k_eff = min(k, scores.size)
        top_unsorted = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top_sorted = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [int(self.recipe_ids[i]) for i in top_sorted]

    def recommend_for_text(
        self,
        seed_texts: Iterable[str],
        k: int = 100,
    ) -> List[int]:
        """Recommend recipes similar to a list of arbitrary text seeds.

        For walk-in users who provide a pantry / ingredient list but no
        recipe IDs. The texts are encoded with the same sentence-transformer
        model used for recipe embeddings; the resulting vectors are averaged
        and used as a query in the same cosine-similarity space.

        Use cases:
            - Walk-in pantry: ["chicken breast", "rice", "broccoli", ...]
            - Free-form taste description: ["spicy thai noodles",
              "vegetarian curries", "quick weeknight meals"]

        Parameters
        ----------
        seed_texts : iterable of str
            Strings to embed. Pantry items, dish names, free text — anything
            the encoder can produce a vector for.
        k : int
            Number of recommendations to return.

        Returns
        -------
        list[int] — top-k recipe_ids, ordered by cosine similarity desc.
        Falls back to popularity if seed_texts is empty.

        Performance note: this loads the encoder lazily and runs a fresh
        encode pass per call (typically <1s for ≤20 short strings on MPS).
        Cache the encoded vector externally if you need to call this in a
        hot path.
        """
        if self._recipe_matrix.size == 0:
            raise RuntimeError("Call .fit(train_df) before .recommend_for_text(...)")

        seeds = [str(t).strip() for t in seed_texts if str(t).strip()]
        if not seeds:
            return self._popularity_fallback(user_id=-1, k=k, exclude_seen=False)

        # Lazy-load the encoder. The recipe-matrix cache means we may never
        # have instantiated SentenceTransformer at fit time.
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(self.model_name, device=self.device)

        # IMPORTANT: when tag_feature_weight>0, the recipe matrix is the
        # SBERT block sqrt(1-w)-scaled and concat with the tag block. We
        # only have a text encoder, so we encode the seeds into the SBERT
        # subspace and zero-pad the tag dimensions to match. This is a
        # principled choice because the seeds are pure text — they have
        # no tag features.
        seed_vecs = encoder.encode(
            seeds,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        seed_vec_text = seed_vecs.mean(axis=0)
        norm = float(np.linalg.norm(seed_vec_text))
        if norm == 0:
            return self._popularity_fallback(user_id=-1, k=k, exclude_seen=False)
        seed_vec_text = seed_vec_text / norm

        recipe_dim = self._recipe_matrix.shape[1]
        text_dim = seed_vec_text.shape[0]
        if text_dim == recipe_dim:
            seed_vec = seed_vec_text
        elif text_dim < recipe_dim:
            # tag_feature_weight>0 case: recipe matrix has extra tag dims.
            # Construct a vector with the SBERT block populated (sqrt-weighted
            # to match the recipe matrix's per-block scaling) and the tag
            # block zeroed, then re-L2.
            w = self.tag_feature_weight
            padded = np.zeros(recipe_dim, dtype=np.float32)
            padded[:text_dim] = float(np.sqrt(1.0 - w)) * seed_vec_text
            n2 = float(np.linalg.norm(padded))
            if n2 == 0:
                return self._popularity_fallback(user_id=-1, k=k, exclude_seen=False)
            seed_vec = (padded / n2).astype(np.float32, copy=False)
        else:
            raise RuntimeError(
                f"seed text dim {text_dim} larger than recipe matrix dim {recipe_dim} — "
                "encoder must match the one used at fit time"
            )

        scores = self._recipe_matrix @ seed_vec

        w = self.content_weight
        if w < 1.0:
            scores = w * scores + (1.0 - w) * self._popularity_score

        k_eff = min(k, scores.size)
        top_unsorted = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top_sorted = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [int(self.recipe_ids[i]) for i in top_sorted]

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

    def _concat_tag_features(self, sbert_matrix: np.ndarray) -> np.ndarray:
        """Layer 4: concatenate Tag SVD + nutrition features onto SBERT embeddings.

        Each block is L2-normed individually, sqrt-weighted, then concatenated.
        Re-L2 on the combined vector so cosine in the new space equals:
            (1 - w) * cos(sbert_parts) + w * cos(tag_parts)
        """
        # Lazy import so the SBERT model doesn't depend on features.py unless needed
        from src.data.features import build_recipe_feature_matrix

        tag_features = build_recipe_feature_matrix()
        # Align Tag SVD rows to self.recipe_ids order (defensive — they should match)
        tag_aligned = (
            tag_features.reindex(self.recipe_ids)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        # SBERT block is already L2-normed (the encoder writes normalized vectors).
        # Normalize the tag block per-row to put both blocks on the same scale.
        tag_norms = np.linalg.norm(tag_aligned, axis=1, keepdims=True)
        tag_norms[tag_norms == 0] = 1.0
        tag_normalized = tag_aligned / tag_norms

        w = self.tag_feature_weight
        combined = np.hstack([
            np.sqrt(1.0 - w) * sbert_matrix,
            np.sqrt(w)       * tag_normalized,
        ]).astype(np.float32, copy=False)

        # Re-L2 the combined vectors so cosine is well-defined.
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (combined / norms).astype(np.float32, copy=False)

    def _build_user_profiles(self, positives: pd.DataFrame) -> dict[int, np.ndarray]:
        """Aggregate per-user positive recipe embeddings into a single profile vector.

        Strategy is selected by self.profile_strategy. All strategies produce
        an L2-normalized 1D vector so cosine sim at recommend time is a dot product.
        """
        profiles: dict[int, np.ndarray] = {}
        strategy = self.profile_strategy

        for user_id, group in positives.groupby("user_id"):
            recipe_ids = group["recipe_id"].to_numpy()
            mask = np.array([int(rid) in self._recipe_id_to_row for rid in recipe_ids])
            if not mask.any():
                continue

            rows = np.array(
                [self._recipe_id_to_row[int(rid)] for rid in recipe_ids[mask]],
                dtype=np.int64,
            )
            vectors = self._recipe_matrix[rows]  # (n_pos, dim)

            if strategy == "mean":
                weights = None
            elif strategy == "rating_weighted":
                ratings = group["rating"].to_numpy()[mask].astype(np.float32)
                # 4★ → 1, 5★ → 2 (subtract 3, clip at 0). All positives are ≥4 by construction
                # but clip defensively in case the caller passes the threshold differently.
                weights = np.clip(ratings - 3.0, 0.0, None)
            elif strategy == "recency_weighted":
                dates = pd.to_datetime(group["date"].to_numpy()[mask], errors="coerce")
                if dates.isna().all():
                    weights = None  # date column missing/corrupt — fall back to mean
                else:
                    ref = dates.max()  # the user's most recent rating
                    days_old = np.asarray((ref - dates).total_seconds()) / 86400.0
                    # exp(-ln(2) * d / half_life) → 1.0 at d=0, 0.5 at d=half_life
                    weights = np.exp(-np.log(2.0) * days_old / self.recency_half_life_days).astype(np.float32)
            else:
                raise RuntimeError(f"unknown profile_strategy: {strategy}")

            if weights is not None and weights.sum() > 0:
                profile_vec = (vectors * weights[:, None]).sum(axis=0) / weights.sum()
            else:
                profile_vec = vectors.mean(axis=0)

            norm = np.linalg.norm(profile_vec)
            if norm == 0:
                continue
            profiles[int(user_id)] = (profile_vec / norm).astype(np.float32, copy=False)

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
