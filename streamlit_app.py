"""PantryPlate — Streamlit demo widget.

Wraps Stage 1 (SBERT / Popularity routing) and Stage 2 (constraint reranker)
into an interactive UI. Two modes:

    Persona mode   — pre-built characters with taste_seeds + pantry + macros
    Walk-in mode   — audience volunteer types pantry + dietary preferences

Both flow through the same Stage 2 reranker. Three α sliders let the user
explore the (αₜ, αₚ, αₙ) simplex live; the recommendations update on rerun.

Run:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.reranker import Stage2Reranker, filter_by_diet


# =============================================================================
# Page config + theme polish
# =============================================================================

st.set_page_config(
    page_title="PantryPlate — recipe recommender",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
/* --- typography --- */
html, body, [class*="css"] {
    font-family: 'Georgia', 'Times New Roman', serif;
}
h1, h2, h3 { font-family: 'Georgia', serif; font-weight: 700; }

/* --- recipe card --- */
.recipe-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(45, 49, 66, 0.06);
    border-left: 5px solid #C75D2C;
}
.recipe-card.rank-1 { border-left-color: #D4A24C; background: linear-gradient(135deg, #FFFEF8 0%, #FFFAEC 100%); }
.recipe-card.rank-2 { border-left-color: #B89456; }
.recipe-card.rank-3 { border-left-color: #A5825A; }
.recipe-card.diet-fail {
    background: #F4ECE9;
    border-left-color: #B85F4A;
    opacity: 0.7;
}

.recipe-rank {
    font-family: 'Georgia', serif;
    font-size: 1.4em;
    font-weight: 700;
    color: #C75D2C;
    margin-right: 8px;
}
.recipe-rank.rank-1 { color: #D4A24C; }

.recipe-title {
    font-size: 1.25em;
    font-weight: 700;
    color: #2D3142;
    margin-bottom: 6px;
    text-transform: capitalize;
}

.recipe-meta {
    color: #6E6F75;
    font-size: 0.92em;
    margin-bottom: 12px;
    font-style: italic;
}

/* --- score chips --- */
.score-row { margin-top: 10px; }
.score-chip {
    display: inline-block;
    padding: 4px 12px;
    margin: 3px 6px 3px 0;
    border-radius: 999px;
    font-size: 0.85em;
    font-weight: 600;
    background: #F1ECE0;
    color: #2D3142;
}
.score-chip.taste   { background: #FCE9D9; color: #8C3E1A; }
.score-chip.pantry  { background: #E4EAD9; color: #4A6034; }
.score-chip.macros  { background: #DDE9EA; color: #2A4D52; }
.score-chip.diet-ok { background: #DBE6D2; color: #3C5E26; }
.score-chip.diet-no { background: #F2D8D2; color: #8B3826; }
.score-chip.final   { background: #2D3142; color: #FAF8F3; }

/* novelty: highlights cold/novel items SBERT surfaces (vs popularity defaults) */
.score-chip.novelty-novel   { background: #E4EFD9; color: #3C5E26; border: 1px dashed #7A9A6E; }
.score-chip.novelty-popular { background: #F1ECE0; color: #6E6F75; }

/* --- sidebar polish --- */
[data-testid="stSidebar"] {
    background: #F1ECE0;
    border-right: 1px solid #E3DCC8;
}
[data-testid="stSidebar"] h2 {
    color: #C75D2C;
    border-bottom: 2px solid #C75D2C;
    padding-bottom: 6px;
    margin-bottom: 18px;
}

/* --- alpha summary --- */
.alpha-summary {
    background: #ffffff;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    text-align: center;
    font-family: 'Georgia', serif;
    color: #2D3142;
    border: 1px dashed #C75D2C;
}

/* --- header --- */
.app-header {
    padding: 8px 0 18px 0;
    border-bottom: 2px solid #E3DCC8;
    margin-bottom: 24px;
}
.app-header h1 {
    color: #C75D2C;
    font-size: 2.4em;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header .tagline {
    color: #6E6F75;
    font-style: italic;
    font-size: 1.05em;
    margin-top: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Cached loaders — load expensive things once per session
# =============================================================================

@st.cache_resource(show_spinner="🍳 Loading PantryPlate models — first launch takes about 30 seconds...")
def load_models_and_data():
    """Load recipes, fit Popularity + SBERT, load personas. Cached for the session."""
    from src.data.loader import load_recipes, load_train_interactions
    from src.models.popularity import PopularityRecommender
    from src.models.sentence_bert import SentenceBERTRecommender

    train = load_train_interactions()
    recipes_raw = load_recipes()
    recipes_raw["id"] = recipes_raw["id"].astype(np.int64)
    recipes = recipes_raw.set_index(recipes_raw["id"].rename("recipe_id"))

    pop = PopularityRecommender().fit(train)
    # Pass the already-parsed catalogue so fit() doesn't re-parse the 230K-row
    # recipe CSV a second time — roughly halves cold start.
    sbert = SentenceBERTRecommender(batch_size=256).fit(train, recipes_df=recipes_raw)

    personas = {}
    for p_path in sorted(Path("data/personas").glob("*.json")):
        with open(p_path) as f:
            p = json.load(f)
            personas[p["id"]] = p

    # Per-recipe unique-rater count — drives the novelty badge / filter.
    # A recipe with 0 raters in train is what the leaderboard calls a "cold item":
    # exactly the case SBERT's cold-track training was meant to serve.
    rater_counts = train.groupby("recipe_id")["user_id"].nunique().to_dict()

    return {
        "recipes": recipes,
        "pop": pop,
        "sbert": sbert,
        "personas": personas,
        "rater_counts": rater_counts,
    }


@st.cache_resource(show_spinner="Loading text encoder for walk-in mode...")
def get_text_encoder():
    """Standalone encoder for walk-in pantry text — cached so we don't re-instantiate per click."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =============================================================================
# Helpers
# =============================================================================

def normalize_alphas(at: float, ap: float, an: float) -> tuple[float, float, float]:
    """Normalize three alphas to sum to 1.0. If all zero, default to (1/3, 1/3, 1/3)."""
    total = at + ap + an
    if total < 1e-9:
        return (1 / 3, 1 / 3, 1 / 3)
    return (at / total, ap / total, an / total)


def pantry_pool_for_walkin(sbert, pantry_text: list[str], encoder, k: int) -> tuple[list[int], dict]:
    """Walk-in candidate pool: encode the pantry text and rank recipes by cosine.

    Bypasses the model's recommend_for_text() so we can reuse a cached encoder
    instance instead of re-instantiating SentenceTransformer per call.
    """
    seeds = [t.strip() for t in pantry_text if t.strip()]
    if not seeds:
        ids = sbert._popularity_fallback(user_id=-1, k=k, exclude_seen=False)
        scores = {rid: 1.0 / (rank + 1) for rank, rid in enumerate(ids)}
        return ids, scores

    seed_vecs = encoder.encode(seeds, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    seed_vec = seed_vecs.mean(axis=0)
    norm = float(np.linalg.norm(seed_vec))
    if norm == 0:
        ids = sbert._popularity_fallback(user_id=-1, k=k, exclude_seen=False)
        scores = {rid: 1.0 / (rank + 1) for rank, rid in enumerate(ids)}
        return ids, scores
    seed_vec /= norm

    recipe_dim = sbert._recipe_matrix.shape[1]
    text_dim = seed_vec.shape[0]
    if text_dim == recipe_dim:
        query_vec = seed_vec
    else:
        # tag_feature_weight > 0 — zero-pad the tag block
        padded = np.zeros(recipe_dim, dtype=np.float32)
        w = sbert.tag_feature_weight
        padded[:text_dim] = float(np.sqrt(1.0 - w)) * seed_vec
        n2 = float(np.linalg.norm(padded))
        query_vec = (padded / n2).astype(np.float32, copy=False)

    raw_scores = sbert._recipe_matrix @ query_vec
    k_eff = min(k, raw_scores.size)
    top_unsorted = np.argpartition(-raw_scores, k_eff - 1)[:k_eff]
    top_sorted = top_unsorted[np.argsort(-raw_scores[top_unsorted])]
    ids = [int(sbert.recipe_ids[i]) for i in top_sorted]
    scores = {rid: 1.0 / (rank + 1) for rank, rid in enumerate(ids)}
    return ids, scores


def categorize_ingredients(
    recipe_ings: list[str],
    user_pantry: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split a recipe's ingredients into (in_pantry, staples, missing).

    - in_pantry: non-staple ingredients the user has → ✓ green
    - staples : universal pantry items → ➖ grey ("assumed")
    - missing : non-staple items the user lacks  → ✗ red ("need to buy")

    Uses substring matching against the pantry items so "chicken breast"
    matches when user has "chicken" in their pantry.
    """
    from src.reranker import STAPLES

    pantry_lower = {p.lower().strip() for p in user_pantry}
    in_pantry: list[str] = []
    staples: list[str] = []
    missing: list[str] = []

    for ing in recipe_ings:
        if not isinstance(ing, str):
            continue
        ing_low = ing.lower().strip()
        if ing_low in STAPLES:
            staples.append(ing)
            continue
        # Substring match either way: user pantry may be "chicken" while
        # recipe ingredient is "chicken breast", or vice versa
        matched = any(p in ing_low or ing_low in p for p in pantry_lower)
        if matched:
            in_pantry.append(ing)
        else:
            missing.append(ing)
    return in_pantry, staples, missing


def render_recipe_card(rank: int, recipe_id: int, name: str, scored_row: pd.Series) -> str:
    """Render a single recipe as a styled HTML card."""
    rank_class = f"rank-{rank}" if rank <= 3 else ""
    diet_class = "diet-fail" if scored_row["s_diet"] == 0 else ""
    card_class = " ".join(c for c in ["recipe-card", rank_class, diet_class] if c)

    diet_chip = (
        '<span class="score-chip diet-ok">✓ diet OK</span>'
        if scored_row["s_diet"] == 1
        else '<span class="score-chip diet-no">✗ diet filter</span>'
    )

    novelty_cls, novelty_label = novelty_for(recipe_id)
    novelty_chip = f'<span class="score-chip {novelty_cls}">{novelty_label}</span>'

    return f"""
    <div class="{card_class}">
        <span class="recipe-rank rank-{rank}">#{rank}</span>
        <span class="recipe-title">{name}</span>
        <div class="recipe-meta">recipe id: {recipe_id}</div>
        <div class="score-row">
            <span class="score-chip taste">🍴 taste {scored_row['s_taste']:.2f}</span>
            <span class="score-chip pantry">🥕 pantry {scored_row['s_pantry']:.2f}</span>
            <span class="score-chip macros">🥗 macros {scored_row['s_nutrition']:.2f}</span>
            {diet_chip}
            {novelty_chip}
            <span class="score-chip final">final {scored_row['final']:.3f}</span>
        </div>
    </div>
    """


# =============================================================================
# Header
# =============================================================================

st.markdown(
    """
    <div class="app-header">
        <h1>🍳 PantryPlate</h1>
        <div class="tagline">Recipe recommendations under your constraints — pantry, macros, dietary needs.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Load everything (cached)
# =============================================================================

data = load_models_and_data()
recipes = data["recipes"]
pop = data["pop"]
sbert = data["sbert"]
personas = data["personas"]
rater_counts = data["rater_counts"]

# Threshold for the novelty badge — "novel" = has < this many raters in train.
# 10 is a nice midpoint: every cold (zero-rater) recipe plus emerging ones with
# small followings get badged. The leaderboard's "cold track" uses zero-rater
# strictly; we relax slightly here so the badge marks more candidates.
NOVELTY_THRESHOLD = 10


def novelty_for(recipe_id: int) -> tuple[str, str]:
    """Return (css_class, label) for a recipe's novelty badge.

    Returns one of:
        ("novelty-novel",   "🌱 novel  (Nr raters)")
        ("novelty-popular", "📈 popular (Nr raters)")
    """
    n = int(rater_counts.get(int(recipe_id), 0))
    if n < NOVELTY_THRESHOLD:
        if n == 0:
            return ("novelty-novel", "🌱 cold item · 0 raters")
        return ("novelty-novel", f"🌱 novel · {n} rater{'s' if n != 1 else ''}")
    return ("novelty-popular", f"📈 popular · {n:,} raters")


# =============================================================================
# Sidebar — input controls
# =============================================================================

with st.sidebar:
    st.markdown("## Who's cooking?")

    mode = st.radio(
        "Mode",
        ["Persona", "Walk-in"],
        index=0,
        help="**Persona**: choose a pre-built character (with taste_seeds).  \n"
             "**Walk-in**: type a pantry & preferences live — the demo audience mode.",
    )

    # Common pantry vocabulary — used as the multiselect options pool.
    # Personas' own pantry items are unioned in so they always appear as selectable.
    SUGGESTED_PANTRY = sorted({
        # proteins
        "chicken breast", "chicken thigh", "ground beef", "ground turkey",
        "salmon", "tuna", "shrimp", "bacon", "ham",
        "tofu", "tempeh", "eggs", "greek yogurt", "cottage cheese",
        "chickpeas", "black beans", "lentils", "edamame",
        # grains / starches
        "rice", "brown rice", "pasta", "quinoa", "couscous", "oats",
        "potato", "sweet potato", "bread", "tortilla", "noodles",
        # vegetables
        "broccoli", "spinach", "kale", "tomato", "carrot", "bell pepper",
        "onion", "garlic", "mushrooms", "asparagus", "zucchini", "lettuce",
        "corn", "peas", "celery", "cucumber",
        # dairy / fats
        "cheddar", "mozzarella", "parmesan cheese", "cream cheese",
        "butter", "milk", "yogurt", "olive oil", "coconut milk",
        # aromatics + condiments
        "ginger", "lemon", "lime", "soy sauce", "vinegar", "honey",
        "peanut butter", "ketchup", "mayonnaise", "mustard",
        # plant-based
        "oat milk", "almond milk", "nutritional yeast", "tahini",
        # snacks / fruits
        "almonds", "walnuts", "banana", "berries", "avocado", "apple",
    })
    RESTRICTION_OPTIONS = ["vegetarian", "vegan", "gluten-free", "dairy-free",
                           "nut-free", "egg-free", "low-carb", "low-fat",
                           "low-sodium", "kosher"]

    # ----- mode-specific defaults --------------------------------------
    if mode == "Persona":
        persona_id = st.selectbox(
            "Start from persona template",
            options=list(personas.keys()),
            format_func=lambda pid: personas[pid].get("label", pid),
            help="Pick a starting profile. You can edit any field below — the persona's "
                 "taste seeds drive Stage 1, but everything else is yours to tweak.",
        )
        base_persona = personas[persona_id]
        with st.expander("Why this persona?"):
            st.write(base_persona.get("description", ""))
            st.caption(f"Taste seeds: {len(base_persona['taste_seeds'])} recipes — "
                       f"these define this persona's content profile and are fixed.")

        default_pantry = base_persona["pantry"]
        default_restrictions = base_persona["restrictions"]
        default_macros = base_persona["macro_targets"]
        taste_seeds = base_persona["taste_seeds"]  # hidden from UI; the persona's identity
    else:
        st.markdown("_No persona — Stage 1 will use SBERT on your pantry text._")
        default_pantry = ["chicken breast", "rice", "broccoli", "garlic", "olive oil"]
        default_restrictions = []
        default_macros = {"calories": 600}
        taste_seeds = []  # walk-in has no seeds

    # ----- optional: detect pantry from a fridge photo (CV inference) -----
    # Available in BOTH modes — it's just another way to fill the pantry.
    # Detected items are MERGED with the mode's default pantry (persona keeps
    # its usual items + whatever's in the photo); user can deselect any below.
    with st.expander("📷 Detect pantry from a fridge photo"):
        uploaded = st.file_uploader(
            "Fridge photo", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Gemini Vision detects ingredients from the image and adds "
                 "them to your pantry below. Needs GEMINI_API_KEY in .env.",
        )
        if uploaded is not None and st.button("Detect ingredients", use_container_width=True):
            try:
                from src.vision.cv_inference import detect_ingredients_from_image
                from src.vision.ingredient_normalizer import normalize

                suffix = ".png" if uploaded.type == "image/png" else ".jpg"
                tmp_path = Path("data") / f"_tmp_fridge{suffix}"
                tmp_path.write_bytes(uploaded.getvalue())
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                detected = normalize(
                    detect_ingredients_from_image(str(tmp_path), mime_type=mime)
                )
                tmp_path.unlink(missing_ok=True)
                if detected:
                    st.session_state["cv_pantry"] = detected
                    st.success(f"Detected {len(detected)}: {', '.join(detected)}")
                else:
                    st.warning("No ingredients detected — try another photo.")
            except RuntimeError as e:
                st.error(str(e))  # e.g. missing GEMINI_API_KEY
            except Exception as e:  # noqa: BLE001 — surface any CV failure to the user
                st.error(f"Detection failed: {e}")

        if st.session_state.get("cv_pantry"):
            st.caption(f"📷 detected: {', '.join(st.session_state['cv_pantry'])}")
            if st.button("Clear detected", use_container_width=True):
                st.session_state.pop("cv_pantry", None)

    # Merge photo-detected ingredients into the pantry default (union, order:
    # mode defaults first, then any new detected items).
    if st.session_state.get("cv_pantry"):
        merged = list(default_pantry)
        for item in st.session_state["cv_pantry"]:
            if item not in merged:
                merged.append(item)
        default_pantry = merged

    # ----- editable inputs (same controls in both modes) ----------------
    st.markdown("**🥕 Pantry**")
    pantry_options = sorted(set(SUGGESTED_PANTRY) | set(default_pantry))
    active_pantry = st.multiselect(
        "What's in your kitchen?",
        options=pantry_options,
        default=default_pantry,
        label_visibility="collapsed",
        help="Multi-select. Add or remove items — the persona's defaults are pre-loaded.",
    )

    st.markdown("**🚫 Dietary restrictions**")
    active_restrictions = st.multiselect(
        "Any restrictions?",
        options=RESTRICTION_OPTIONS,
        default=default_restrictions,
        label_visibility="collapsed",
        help="Diet is a hard filter — recipes failing ANY restriction score 0.",
    )

    st.markdown("**🥗 Macro targets (optional)**")
    use_macros = st.checkbox(
        "Apply macro targets",
        value=bool(default_macros),
        help="Uncheck to ignore macros entirely (αₙ will have no effect).",
    )
    active_macros: dict = {}
    if use_macros:
        c1, c2 = st.columns(2)
        with c1:
            cals = st.number_input("Calories",   0, 2000, int(default_macros.get("calories", 500)), 50)
            prot = st.number_input("Protein PDV", 0, 200, int(default_macros.get("protein_pdv", 30)), 5)
        with c2:
            carb = st.number_input("Carbs PDV",   0, 200, int(default_macros.get("carbs_pdv", 30)), 5)
            fat  = st.number_input("Fat PDV",     0, 200, int(default_macros.get("fat_pdv", 25)), 5)
        if cals > 0: active_macros["calories"]    = cals
        if prot > 0: active_macros["protein_pdv"] = prot
        if carb > 0: active_macros["carbs_pdv"]   = carb
        if fat  > 0: active_macros["fat_pdv"]     = fat

    # ----- assemble the active persona used for Stage 2 -----------------
    if mode == "Persona":
        active_persona = {
            "id": base_persona["id"],
            "label": base_persona.get("label", base_persona["id"]),
            "pantry": active_pantry,
            "macro_targets": active_macros,
            "restrictions": active_restrictions,
            "exclude_from_staples": base_persona.get("exclude_from_staples", []),
            "taste_seeds": taste_seeds,
        }
    else:
        active_persona = {
            "id": "walkin_demo",
            "label": "Walk-in volunteer",
            "pantry": active_pantry,
            "macro_targets": active_macros,
            "restrictions": active_restrictions,
            "exclude_from_staples": [],
            "taste_seeds": [],
        }

    st.markdown("---")
    st.markdown("## What matters to you?")
    st.caption("Move one slider and the other two rebalance automatically — the three always sum to 1.")

    # Initialize the simplex with defaults (0.5, 0.3, 0.2)
    if "alpha_taste" not in st.session_state:
        st.session_state["alpha_taste"]     = 0.50
        st.session_state["alpha_pantry"]    = 0.30
        st.session_state["alpha_nutrition"] = 0.20

    def _rebalance(changed: str) -> None:
        """When one slider moves, redistribute the remaining 1 - x across the
        other two in proportion to their *previous* ratio. Preserves user
        intent — if you had pantry > nutrition, you keep that ordering."""
        keys = ["alpha_taste", "alpha_pantry", "alpha_nutrition"]
        others = [k for k in keys if k != changed]
        x = float(st.session_state[changed])
        x = max(0.0, min(1.0, x))
        remainder = 1.0 - x
        a, b = float(st.session_state[others[0]]), float(st.session_state[others[1]])
        total_other = a + b
        if remainder <= 0:
            st.session_state[others[0]] = 0.0
            st.session_state[others[1]] = 0.0
        elif total_other < 1e-9:
            st.session_state[others[0]] = remainder / 2
            st.session_state[others[1]] = remainder / 2
        else:
            st.session_state[others[0]] = remainder * a / total_other
            st.session_state[others[1]] = remainder * b / total_other

    st.slider(
        "🍴 Taste", 0.0, 1.0, step=0.01, key="alpha_taste",
        on_change=_rebalance, args=("alpha_taste",),
        help="Stage 1's content-based ranking weight.",
    )
    st.slider(
        "🥕 Pantry match", 0.0, 1.0, step=0.01, key="alpha_pantry",
        on_change=_rebalance, args=("alpha_pantry",),
        help="Favor recipes you can mostly cook from your pantry.",
    )
    st.slider(
        "🥗 Macros", 0.0, 1.0, step=0.01, key="alpha_nutrition",
        on_change=_rebalance, args=("alpha_nutrition",),
        help="Favor recipes near your macro targets.",
    )

    alpha_taste     = float(st.session_state["alpha_taste"])
    alpha_pantry    = float(st.session_state["alpha_pantry"])
    alpha_nutrition = float(st.session_state["alpha_nutrition"])

    # Sanity / display — values now always sum to ~1
    total_show = alpha_taste + alpha_pantry + alpha_nutrition
    st.markdown(
        f"""
        <div class='alpha-summary'>
            current mix (sum = {total_show:.2f})<br>
            🍴 {alpha_taste:.2f} &nbsp; 🥕 {alpha_pantry:.2f} &nbsp; 🥗 {alpha_nutrition:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick presets for live demo — one click jumps to a simplex corner
    pc1, pc2, pc3, pc4 = st.columns(4)
    def _set_alpha(t: float, p: float, n: float) -> None:
        st.session_state["alpha_taste"]     = t
        st.session_state["alpha_pantry"]    = p
        st.session_state["alpha_nutrition"] = n
    if pc1.button("⚖️ Balanced",  use_container_width=True): _set_alpha(0.50, 0.30, 0.20)
    if pc2.button("🍴 Taste",      use_container_width=True): _set_alpha(1.00, 0.00, 0.00)
    if pc3.button("🥕 Pantry",     use_container_width=True): _set_alpha(0.00, 1.00, 0.00)
    if pc4.button("🥗 Macros",     use_container_width=True): _set_alpha(0.00, 0.00, 1.00)

    st.markdown("---")
    st.markdown("## Discovery options")
    only_novel = st.toggle(
        "🌱 Surface novel recipes only",
        value=False,
        help=f"Filter the candidate pool to recipes with fewer than {NOVELTY_THRESHOLD} "
             "raters in the training data. These are 'cold items' — the use case the "
             "content model (SBERT) was specifically designed to handle. Off by default.",
    )

    st.markdown("---")
    k_show = st.number_input("How many recipes to show?", min_value=3, max_value=20, value=10, step=1)


# =============================================================================
# Main panel — generate + display recommendations
# =============================================================================

POOL_SIZE = 100
# Overgenerate factor: with dietary restrictions, Stage 1 may return mostly
# non-compliant candidates (e.g., chicken-heavy pantry + vegan filter). Get
# 5x and post-filter by diet so the reranker has compliant candidates to
# work with. Standard "expand-then-filter" candidate-generation pattern.
OVERGEN_FACTOR = 5
INITIAL_POOL = POOL_SIZE * OVERGEN_FACTOR if active_persona["restrictions"] else POOL_SIZE

# Stage 1 — route to the right model for this mode
if mode == "Persona":
    initial_ids = sbert.recommend_for_seeds(active_persona["taste_seeds"], k=INITIAL_POOL)
    stage1_label = "SBERT(taste_seeds)"
else:
    if not active_persona["pantry"]:
        initial_ids = pop.recommend(user_id=-1, k=INITIAL_POOL, exclude_seen=False)
        stage1_label = "Popularity (no pantry provided)"
    else:
        encoder = get_text_encoder()
        initial_ids, _ = pantry_pool_for_walkin(
            sbert, active_persona["pantry"], encoder, INITIAL_POOL
        )
        stage1_label = "SBERT(pantry text)"

# Diet pre-filter — keeps Stage 1's order but drops non-compliant candidates
candidate_ids = filter_by_diet(
    initial_ids,
    active_persona["restrictions"],
    recipes,
    target_k=POOL_SIZE if not only_novel else POOL_SIZE * 4,
)

# Optional novelty filter — keep only candidates with < NOVELTY_THRESHOLD raters
# in train. This is what SBERT's cold-track training was for: surfacing recipes
# nobody has rated yet. Off by default; on for the "show me discoveries" demo path.
n_pre_novelty = len(candidate_ids)
if only_novel:
    candidate_ids = [
        rid for rid in candidate_ids
        if int(rater_counts.get(int(rid), 0)) < NOVELTY_THRESHOLD
    ][:POOL_SIZE]

taste_scores = {rid: 1.0 / (rank + 1) for rank, rid in enumerate(candidate_ids)}

# Surface what happened during filtering — helps the audience understand
# why "vegan + chicken pantry" might show fewer recipes than expected
n_initial = len(initial_ids)
n_compliant = len(candidate_ids)
if active_persona["restrictions"] and n_compliant < POOL_SIZE:
    st.info(
        f"🍃 Diet filter applied. Looked at {n_initial} Stage 1 candidates; "
        f"{n_compliant} satisfy "
        f"{', '.join(active_persona['restrictions'])}. "
        f"{'Try widening your pantry or relaxing the diet filter for more variety.' if n_compliant < 10 else ''}"
    )

if only_novel:
    n_after_novelty = len(candidate_ids)
    st.info(
        f"🌱 Novelty filter applied. {n_after_novelty}/{n_pre_novelty} candidates "
        f"have fewer than {NOVELTY_THRESHOLD} raters in train — these are the "
        f"recipes content models (SBERT) were specifically trained to surface."
        f"{' Try toggling off to compare with the full pool.' if n_after_novelty < 10 else ''}"
    )

# Stage 2 — rerank with the current alphas
reranker = Stage2Reranker(
    alpha_taste=alpha_taste,
    alpha_pantry=alpha_pantry,
    alpha_nutrition=alpha_nutrition,
)
scored = reranker.rerank(active_persona, candidate_ids, taste_scores, recipes, k=k_show, return_scores=True)

# Layout: results on the left, "what's happening" on the right
col_main, col_side = st.columns([3, 1], gap="large")

with col_main:
    st.markdown(f"### Your top {k_show} recipes")
    st.caption(f"Stage 1 candidates from **{stage1_label}** · Stage 2 reranked with the α-weights")

    if scored.empty:
        st.warning("No candidates to show. Try a different mode or input.")
    else:
        for i, row in scored.iterrows():
            rid = int(row["recipe_id"])
            recipe_row = recipes.loc[rid] if rid in recipes.index else None
            name = recipe_row["name"] if recipe_row is not None else "(unknown)"
            st.markdown(
                render_recipe_card(rank=i + 1, recipe_id=rid, name=name, scored_row=row),
                unsafe_allow_html=True,
            )

            # Expandable breakdown: ingredients (with pantry overlap visible)
            # + nutrition + tags. Makes the "pantry match" claim concrete for
            # live demo audience.
            if recipe_row is not None:
                with st.expander("📋 Show ingredients & nutrition"):
                    ings = recipe_row.get("ingredients_parsed") or []
                    tags = recipe_row.get("tags_parsed") or []
                    nutrition = recipe_row.get("nutrition_parsed") or {}
                    in_p, staples, missing = categorize_ingredients(
                        ings, active_persona.get("pantry", [])
                    )

                    icol1, icol2 = st.columns([2, 1])
                    with icol1:
                        st.markdown("**🥕 Ingredients**")
                        if in_p:
                            st.markdown(
                                "<span style='color:#3C5E26'>✓ in your pantry</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<ul style='margin-top:0;color:#3C5E26'>"
                                + "".join(f"<li>{ing}</li>" for ing in in_p)
                                + "</ul>",
                                unsafe_allow_html=True,
                            )
                        if staples:
                            st.markdown(
                                "<span style='color:#6E6F75'>➖ assumed staples</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<ul style='margin-top:0;color:#6E6F75'>"
                                + "".join(f"<li>{ing}</li>" for ing in staples)
                                + "</ul>",
                                unsafe_allow_html=True,
                            )
                        if missing:
                            st.markdown(
                                f"<span style='color:#8B3826'>✗ need to buy "
                                f"({len(missing)} item{'s' if len(missing) != 1 else ''})</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<ul style='margin-top:0;color:#8B3826'>"
                                + "".join(f"<li>{ing}</li>" for ing in missing)
                                + "</ul>",
                                unsafe_allow_html=True,
                            )
                        if not (in_p or staples or missing):
                            st.caption("(no ingredients parsed)")

                    with icol2:
                        st.markdown("**🥗 Nutrition per serving**")
                        if nutrition:
                            for field, label in [
                                ("calories",    "Calories"),
                                ("protein_pdv", "Protein"),
                                ("carbs_pdv",   "Carbs"),
                                ("fat_pdv",     "Fat"),
                                ("sodium_pdv",  "Sodium"),
                            ]:
                                val = nutrition.get(field)
                                if val is None:
                                    continue
                                unit = "" if field == "calories" else "% PDV"
                                target = active_persona.get("macro_targets", {}).get(field)
                                if target:
                                    delta_pct = (val - target) / target * 100
                                    delta_str = f" ({delta_pct:+.0f}% vs target)"
                                else:
                                    delta_str = ""
                                st.markdown(
                                    f"- **{label}**: {val:.0f}{unit}{delta_str}"
                                )
                        else:
                            st.caption("(no nutrition data)")

                        if tags:
                            st.markdown("**🏷️ Tags**")
                            shown = [t for t in tags if isinstance(t, str)][:8]
                            st.markdown(
                                " ".join(f"`{t}`" for t in shown)
                            )

with col_side:
    st.markdown("### What's happening?")
    st.markdown(
        f"""
        **Stage 1** → 100 candidates
        Method: `{stage1_label}`

        **Stage 2** → top-{k_show}
        Formula:
        `final = s_diet × (αₜ·s_taste + αₚ·s_pantry + αₙ·s_nutrition)`
        """
    )

    if not scored.empty:
        st.markdown("### Score statistics")
        st.metric("Mean s_taste",     f"{scored['s_taste'].mean():.2f}")
        st.metric("Mean s_pantry",    f"{scored['s_pantry'].mean():.2f}")
        st.metric("Mean s_nutrition", f"{scored['s_nutrition'].mean():.2f}")
        st.metric("Diet pass rate",   f"{(scored['s_diet'] == 1).mean():.0%}")

    with st.expander("ℹ️ About PantryPlate"):
        st.markdown(
            """
            **Stage 1** generates candidate recipes — content-similar to a user's seeds
            (persona mode) or pantry text (walk-in mode), or popular baseline if no input.

            **Stage 2** reranks those candidates by your weighted constraints. Diet is a
            **hard filter** (0 or 1); pantry, nutrition, and taste are **continuous** signals
            on the simplex.

            **The α sliders** are the *X-factor* — they let you explore the trade-off
            between taste, pantry feasibility, and macro fit. Watch the top-10 reorder
            as you change them.
            """
        )
