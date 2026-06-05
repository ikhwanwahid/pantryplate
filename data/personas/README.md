# Personas

Persona JSONs in this directory define the user contexts the Stage 2 reranker is evaluated against. Each persona is a triple of constraint inputs (pantry, macro targets, dietary restrictions) that the reranker uses to compute `s_pantry`, `s_nutrition`, and `s_diet`.

---

## What personas are used for

**1. Evaluation — Useful Recall@K and α-sweep**

The signature metric `Useful Recall@K` requires constraint inputs per user. Real users in the test set don't tell us their pantry or macro goals, so we use personas as a proxy. The evaluation routes the real test users' rating histories through Stage 1 (taste signal), then applies the persona's constraints in Stage 2.

This means a single test user × one persona = one Useful Recall@K observation. We report Useful Recall@K **per persona** so we can study Hypothesis 2 from the proposal: *"optimal α-weighting differs by user type."*

**2. Demo**

The interactive demo widget loads a persona on persona-switcher dropdown, then drives the α-sliders. Demo users see the system in action against a realistic user profile.

For the demo only, personas may also have a `taste_seeds` field (list of Food.com recipe IDs the persona would rate 5 stars). This bootstraps Stage 1 for a "cold" demo user with no prior rating history. Evaluation personas don't need this field — evaluation uses real test users' actual histories.

---

## Schema

```json
{
  "id": "persona_id",
  "label": "Display name for the demo",
  "description": "Why this persona exists, what they represent.",
  "macro_targets": {
    "calories":     500,
    "protein_pdv":  50,
    "carbs_pdv":    30,
    "fat_pdv":      25,
    "sodium_pdv":   30
  },
  "restrictions": ["vegan", "gluten-free"],
  "exclude_from_staples": [],
  "pantry": [
    "ingredient_1", "ingredient_2", ...
  ],
  "taste_seeds": [recipe_id, recipe_id, ...]
}
```

### Field definitions

| Field | Required for eval | Required for demo | Notes |
|---|---|---|---|
| `id` | ✓ | ✓ | Snake_case unique identifier; filename = `<id>.json` |
| `label` | ✓ | ✓ | Human-readable display name |
| `description` | optional | optional | Why this persona exists; what user type it represents |
| `macro_targets` | ✓ | ✓ | Macro nutrition the persona is targeting. Used by `s_nutrition`. `calories` is absolute kcal; the others are PDV percentages |
| `restrictions` | ✓ | ✓ | List of dietary restriction strings. Hard filter — `s_diet = 0` if any is violated. Examples: `"vegan"`, `"vegetarian"`, `"gluten-free"`, `"keto"`, `"paleo"`, `"low-sodium"`. See [data_decisions.md §8](../../docs/data_decisions.md) for the supported list. |
| `exclude_from_staples` | optional | optional | List of staple items to remove from the project-wide staples for THIS persona. E.g., a gluten-free persona might exclude `"flour"`. Vegan personas auto-drop dairy/eggs via `get_staples_for_persona()` — you don't need to list them here. |
| `pantry` | ✓ | ✓ | 25-35 user-specific ingredients the persona keeps on hand. **Do not include staples** (salt, oil, flour, eggs, milk, butter, garlic, onion, etc.) — those are project-wide via `src/utils/staples.py`. |
| `taste_seeds` | optional | demo only | List of Food.com recipe IDs the persona would rate 5★. Demo cold-start bootstrap. Skip for evaluation-only personas. |

---

## The existing 3 personas

### `fitness_focused.json`

- High-protein, low-carb tendencies; no formal diet restriction
- Macro targets drive ranking (calories ≈ 500, protein PDV 50, carbs PDV 30)
- Pantry: lean proteins (chicken, salmon, tofu), green vegetables, complex carbs, nuts
- **Tests**: the nutrition score with tight macro targets; how the reranker handles a persona with no hard diet filter

### `vegan_busy.json`

- Vegan diet restriction; moderate balanced macros
- Pantry: plant-based proteins (tofu, tempeh, lentils), vegetables, grains, dairy alternatives
- **Tests**:
  - The diet hard-filter (`s_diet = 0` for non-vegan recipes)
  - The vegan auto-exclusion of dairy/eggs from STAPLES via `get_staples_for_persona()`. Recipes needing milk or eggs become unmatchable from the staples side.

### `family_friendly.json`

- No diet restriction; higher calorie target (700, larger portions); balanced macros
- Pantry: kid-friendly versatile items (chicken thigh, ground beef, pasta, rice, cheeses, common vegetables)
- **Tests**: the "loose constraints" case where mostly `s_taste` drives ranking with mild nutrition and pantry signals. Used as a reference / baseline persona.

### Why these three together

The α-sweep study tests **Hypothesis 2**: *"optimal α-weighting differs by user type."* For that hypothesis to be testable, we need personas that should reasonably prefer different α-positions:

- `fitness_focused` should benefit from higher `αn` (nutrition weight) — they care more about hitting macros
- `family_friendly` should benefit from higher `αp` (pantry weight) — they keep a lot of common items, so leveraging them matters
- `vegan_busy` should be insensitive to `αp` because the diet hard filter dominates first, then `αt` matters

If the α-sweep shows different optimum α per persona, Hypothesis 2 is supported. If they all peak at the same α, Hypothesis 2 fails. Either result is a meaningful empirical finding.

---

## Pantry conventions to follow when adding new personas

1. **25-35 items, NOT including staples.** Staples are project-wide in `src/utils/staples.py` and assumed available for every persona by default.
2. **Use lowercase, singular forms** (`broccoli`, not `Broccolis`) so they match the recipe ingredient canonicalization.
3. **Use the same vocabulary as `ingr_map.pkl`** where possible. Common canonical forms: `chicken breast`, `ground beef`, `bell pepper`, `cheddar`, `mozzarella`, `parmesan cheese`. Run `normalize_ingredient()` on questionable items to check.
4. **Items absent from `ingr_map.pkl` still work** — they just won't match any recipe ingredients. So including obscure items (e.g., a specific spice) is harmless but pointless.
5. **Don't over-think it.** A persona pantry isn't a complete kitchen inventory. It's 25-35 items that capture what's *distinctive* about this user.

---

## Loading personas in code

```python
from src.data.pantry import load_persona

persona = load_persona("fitness_focused")
# returns the dict from fitness_focused.json

print(persona["pantry"])       # list of 25 ingredients
print(persona["macro_targets"])  # dict of macro PDVs
print(persona["restrictions"])  # ["vegan"], etc.

# For staples-aware pantry handling
from src.utils.staples import get_staples_for_persona
staples_for_persona = get_staples_for_persona(persona)
# vegan personas → dairy/eggs auto-removed
# personas with exclude_from_staples → those items removed
```

The Stage 2 reranker accepts a persona dict and computes the 4 scores against it.

---

## Adding a new persona

1. Decide what user type this persona represents (in 1 sentence)
2. Pick macro targets that reflect that type
3. Pick restrictions if any apply
4. List 25-35 user-specific (non-staple) pantry items
5. Save as `data/personas/<id>.json` matching the schema above
6. Optionally add `taste_seeds` if this persona will be used in the demo
7. Add to the table in the README ("The existing N personas" section above)

That's it. The reranker and harness pick up the new persona automatically — no code changes needed.

---

## Stretch goal: team self-profiles

For the final demo, each team member can author their own persona using their actual preferences:
- Pick 20-30 real Food.com recipes you'd rate 5★ (this becomes your `taste_seeds`)
- List ~25 ingredients you actually keep
- Set your macro targets (or use rough defaults)
- List any actual dietary restrictions

Save as `team_<name>.json`. Used in the demo presentation for the audience-engagement moment (*"let me show you what the system recommends to ME based on my taste"*).

This is a Week 5-6 polish task — not required for evaluation. The 3 generic personas above are enough for the headline empirical study.
