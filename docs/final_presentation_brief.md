# PantryPlate — Final Presentation Deck Brief

**Hand this document to Claude.ai (or any AI assistant) and ask it to build `PantryPlate_Final.pptx`.**

This is the final-presentation counterpart to `docs/proposal_deck_rebuild_brief.md` (which built the proposal deck). The proposal made promises; this brief turns the now-finished project — Stage 1 leaderboard, Stage 2 reranker, the α-sweep headline result, the live demo — into the final 19-slide deck. Presentation date: **24 June 2026**.

---

## 0. Use instructions for Claude

You are being asked to build an ~19-slide final-presentation deck for a graduate Recommender Systems course project (PantryPlate). All factual content below comes from two sources: `PantryPlate_Proposal.pptx` (what was already promised/presented) and `notebooks/pantryplate_e2e.ipynb` (the authoritative final-results notebook — read it directly for exact figures, tables, and plots, don't re-derive numbers). Do not invent numbers not present in either source.

**Suggested output**: a Python script using `python-pptx` that generates `PantryPlate_Final.pptx`. Match the proposal deck's visual identity (Section 3 of `proposal_deck_rebuild_brief.md`): section label top-left, "N / 19" slide number top-right, bold headline + italic subhead, pull-out emphasis boxes (caps labels), italic footer naming the section role.

**Don't re-debate**: the five required agenda items below come directly from the professor's rubric — every slide must roll up into one of them. If a slide doesn't, cut it.

---

## 1. Professor's rubric — every slide must map to one of these five

1. Finalized dataset, problem, algorithms, experimental results
2. Discussion on the applicability and significance of the project
3. Proposed or attempted extensions, with evidence
4. Demonstration of a working recommender system
5. Future work

---

## 2. What changed since the proposal (context for whoever builds this)

- **Dataset numbers**: the proposal quoted *raw* row counts (698,901 train / 7,023 validation / 10,393 test). The notebook's numbers (681K train / 5,900 validation / 10,393 test) are *after* dropping 0-star "review without rating" entries — both are correct, just pre/post-cleaning. **Use the notebook's post-cleaning numbers** and label them as final.
- **Architecture evolved**: the proposal pitched a single hybrid model as "best of both" (warm + cold). The finished project found the opposite — **routing beats hybrids** (BPR wins warm, SBERT wins cold; every linear hybrid tried loses to just routing). This is a real finding, not a simplification — present it as one.
- **Reach Goal 1 (CV fridge-photo input)**: built. `src/vision/cv_inference.py` sends a fridge photo to Gemini 2.5 Flash, which returns a comma-separated ingredient list; that list becomes the user's pantry input to Stage 2. It's wired into `streamlit_app.py` as an optional input path (alongside persona mode and manual walk-in entry). Show this as the extension-with-evidence slide.
- **Reach Goal 2 (shelf-life prioritization)**: never started — zero code or doc references anywhere in the repo. Don't give it a dedicated slide. One line in Future Work only.
- **Proposal Hypothesis 2** ("optimal α-weighting differs by user type") was investigated post-proposal via a per-persona α-sweep (`persona_alpha_sweep.py`, `data/processed/alpha_sweep_persona_*.csv`) but the result was inconclusive — relevance still peaks at the taste corner regardless of persona, and the useful/near-rate degenerates to ~0% for all three personas (fixed external macro targets are much harder to satisfy than the self-consistent history-derived targets used in the headline sweep). **Don't present this as a result.** State Hypothesis 2 as explored-but-inconclusive in Future Work.
- **Two exploratory deep-CF models** (LightGCN, NeuMF) were added as personal side experiments — explicitly labeled in code as "NOT part of the official Stage 1 model menu" (`docs/data_decisions.md` decision #5), unreviewed by the team. Included in the merged algorithms/results table (Slide 6) with that labeling kept intact, not folded silently into the official 7-model menu.

---

## 3. Slide-by-slide specification (19 slides)

---

### Slide 1 — Title

```
[Header label]  FINAL PRESENTATION
[Title]         PantryPlate
[Subtitle]      Multi-constraint recipe recommendation balancing
                taste, pantry, nutrition, and dietary restrictions
[Tagline]       Two-stage recommender + a constraint reranker
[Footer]        Recommender Systems · Final Presentation · 24 June 2026
```

---

### Slide 2 — Motivation (NEW — replaces the proposal's competitor-gap framing)

**Section label**: MOTIVATION (2 / 19)

**Headline**: You have five minutes and an open fridge
**Subhead**: Everyone in this room has done this math under time pressure — but the right answer depends on who's asking.

**Body, part 1** (the relatable hook — lead with the scene, not a feature comparison):

> It's 7pm. You just got home. You're hungry, and you have maybe five minutes before you need to actually start cooking. In that window you're running four calculations at once, without writing any of them down: *what's actually in my fridge right now, does this fit what I'm trying to eat today, can I even eat this given my restrictions, and do I actually want it.* Nobody has the patience to check four separate apps for that. Most days, "what's for dinner" isn't a discovery problem — it's a five-minute constraint-satisfaction problem, solved badly, standing up.

**Body, part 2** (the pivot — same scene, three different people, three different stakes):

> But "solve it well" doesn't mean the same thing for everyone. Picture three people standing in front of that same open fridge tonight:

- **A fitness-focused cook** doesn't want to manually check macros against five different recipes before picking one — they want their goals already factored into the ranking, every time, with no extra effort.
- **A busy vegan** doesn't have time to second-guess every suggestion for hidden dairy or eggs — they need to trust, instantly, that what's recommended is actually safe to eat.
- **A family feeding kids on a budget** doesn't want another grocery run for one missing ingredient — they want tonight's dinner to come from what's already in the kitchen.

> Same fridge, same five minutes, three completely different versions of "the right recommendation." You'll meet these three people again later — they're not hypothetical, they're the personas PantryPlate is built and tested against.

**Pull-out box "THE GAP"**: "Existing recommenders optimize one of these at a time — a recipe finder ranks by taste, a macro tracker filters by nutrition, a pantry app filters by what's on hand. None of them resolve the trade-off between all four *for* you — and none of them resolve it *differently* depending on who's asking."

**Footer**: Motivation — why this is a real, everyday problem, for more than one kind of person

---

### Slide 3 — Recap: the contribution & architecture (condensed "since the proposal" slide)

**Section label**: RECAP (3 / 19)

**Headline**: Recap on Architecture

**Three compressed panels**:
1. **The 4 constraints** — Taste · Pantry · Nutrition · Diet (one line each, reuse proposal Slide 2 wording)
2. **Headline hypothesis** (proposal Slide 3): "Balanced multi-objective ranking produces qualitatively better recommendations than single-objective optimization on a composite usefulness metric. The slider between objectives isn't a UI feature — it's the empirical finding."
3. **Architecture diagram** (proposal Slide 8 / notebook §2):
```
USER CONTEXT (history · pantry · macros · diet · α-weights)
        ↓
STAGE 1 — candidate generation (real recsys, routed per user context)
        ↓  (diet hard filter applied here)
STAGE 2 — constraint rerank: αt·taste + αp·pantry + αn·nutrition
        ↓
Top-10 recommendations
```

**Footer**: Recap — what we already told you, condensed

---

### Slide 4 — Finalized dataset

**Section label**: FINALIZED DATASET (4 / 19)

**Headline**: Food.com (Majumder et al., EMNLP 2019)
**Subhead**: Authors' pre-split train/validation/test — used as published, for comparability.

**Stats panel** (use the notebook's post-cleaning numbers, not the proposal's raw ones):
- **231,637 recipes** · ~25,000 active users
- **Train**: 681K interactions (24,961 users) — after dropping 0-star "review without rating" entries
- **Validation**: 5,900 held-out positives — hyperparameter tuning (cold-by-construction)
- **Test**: 10,393 held-out positives — all cold items (0 raters in train)
- 100% nutrition coverage · 8,023 canonical ingredients · time range 2000–2018

**Small note**: "Train is NOT pre-filtered to ≥5 ratings — wide activity spread (~41% low / 38% medium / 21% high activity)."

**Footer**: Dataset — finalized, post-cleaning numbers

---

### Slide 5 — Evaluation design: dual-track + the signature metric

**Section label**: EVALUATION DESIGN (5 / 19)

**Headline**: Two tracks, because real recipe apps face both
**Subhead**: Warm-item (standard CF) and cold-item (novel recipes) — same architecture, different question.

**Two-column**:
- **Warm** — our own time-based LOO on train (hold out each user's most-recent 4★+ rating). Tests: "what's the user's next pick?"
- **Cold** — authors' pre-split test, 10,393 items with zero raters in train. Tests: "can we recommend a recipe nobody has rated yet?" CF models score exactly 0 here **by construction** — correct, not a bug.

**Metric**: Recall@K at @10 (standard) and @100 (the candidate-pool size handed to Stage 2 — the pipeline-relevant number).

**Pull-out box "SIGNATURE METRIC — Useful Recall"**: "Of what we recommend, how much is actually usable — pantry-feasible (≤3 missing non-staple ingredients) AND macro-near (±20%)? Diet is a hard filter upstream, not part of this score."

**Footer**: Evaluation — dual-track design + Useful Recall

---

### Slide 6 — Algorithms & Stage 1 results (merged — training method + leaderboard in one table)

**Section label**: ALGORITHMS & RESULTS (6 / 19)

**Headline**: Seven models, one shared contract — and how they land
**Subhead**: `model.fit(train_df)` → `model.recommend(user_id, k, exclude_seen)`. Every model is swappable and evaluated identically; results follow directly from how each is trained.

**Table** (merges notebook §3c training matrix with the §4 leaderboard, `data/processed/stage1_leaderboard.csv` — keep the exploratory tag in the model name, don't drop it):

| Model | Family | How it's trained | Warm@10 | Warm@100 | Cold@10 | Cold@100 |
|---|---|---|---|---|---|---|
| Popularity | baseline | rank by # distinct raters in train | 2.95 | 11.55 | 0.000 | 0.000 |
| MF / ALS | CF (implicit) | `implicit` ALS, binary matrix, confidence-scaled (factors=64, reg=0.05) | 2.05 | 7.95 | 0.000 | 0.000 |
| EASE | CF (shallow AE) | closed-form B = −(P/diag P), one matrix inverse, no SGD (λ=250) | 2.90 | 8.10 | 0.000 | 0.000 |
| BPR | CF (pairwise) | Cornac BPR — SGD on pairwise ranking loss (k=100, 500 iters) | 2.70 | **12.25** | 0.000 | 0.000 |
| Tag SVD content | content | 107-dim recipe vectors (100 tag-SVD + 7 nutrition); cosine rank | 0.00 | 0.50 | 0.010 | 0.164 |
| SBERT content | content | frozen `all-MiniLM-L6-v2` on name\|ingredients\|tags, 384-dim; cosine rank | 0.15 | 0.70 | 0.087 | **0.452** |
| Hybrid linear | CF × content | rank-normalize + blend `α·CF + (1−α)·content` | 1.75 | 4.45 | 0.019 | 0.087 |
| LightGCN *(exploratory, unreviewed — off locked menu)* | CF (graph) | He et al. 2020 — strips feature transforms/nonlinearities from a GCN; embeddings propagate over the symmetric-normalized user-item bipartite graph, final embedding = layer-wise mean; trained with BPR pairwise loss | 3.05 | 11.30 | 0.000 | 0.000 |
| NeuMF *(exploratory, unreviewed — off locked menu)* | CF (neural) | He et al. 2017 — fuses GMF (element-wise user/item embedding product) with an MLP tower; BCE loss + uniform negative sampling on implicit feedback | 2.05 | 9.85 | 0.000 | 0.000 |

**Headline numbers** (notebook §4 findings):
- **BPR**: Warm Recall@100 = 12.25% — the best @100 pool coverage, the number that feeds Stage 2
- **SBERT**: Cold Recall@100 = 0.452% (~10× random; the only family with real cold signal)
- **CF = 0 on cold** for every CF model, including the exploratory LightGCN/NeuMF — by construction
- LightGCN edges out everyone on Warm@10 (3.05%) but loses to BPR at @100 (11.30 vs 12.25) — the pool-coverage number is what actually matters for the pipeline, so BPR stays the routed warm specialist
- ALS is the weakest official CF (rounds out the menu, doesn't change picks)

**Small note**: "EASE/BPR/ALS train on items with ≥10 ratings (~20K) to bound the matrices; SBERT/Tag SVD cover the full 231K catalogue — needed for the cold track. LightGCN/NeuMF are personal side experiments, not on the team's locked model menu (`docs/data_decisions.md` decision #5) — included for completeness, not as vetted results."

**Footer**: Algorithms & experimental results — Stage 1

---

### Slide 7 — Finding: routing beats hybrids

**Section label**: STAGE 1 FINDING (7 / 19)

**Headline**: We tried one model for both. None beat just routing.
**Subhead**: A negative result that justifies the architecture.

**Table** (notebook §5):

| Approach | Warm @100 | Cold @100 | Verdict |
|---|---|---|---|
| BPR (CF) | 12.25 | 0 | warm specialist |
| SBERT (content) | 0.70 | 0.45 | cold specialist |
| EASE+SBERT hybrid | 4.45 | 0.087 | better than either alone, but loses to BPR on warm AND SBERT on cold |

**Footer**: Experimental results — routing vs hybrids

---

### Slide 8 — Stage 2 reranker: the formula

**Section label**: STAGE 2 RERANKER (8 / 19)

**Headline**: The constraint blend
**Subhead**: No training — a deterministic reweight of the Stage 1 pool.

**Formula (centered, prominent)**:
`final(u,r) = αt·s_taste + αp·s_pantry + αn·s_nutrition`     (αt+αp+αn = 1)

**Score definitions**:
- **s_taste** — Stage 1's relevance score, min-max normalized within the pool
- **s_pantry** — fraction of non-staple ingredients the user has, [0,1]
- **s_nutrition** — Gaussian proximity of recipe macros to user targets, [0,1]
- **diet** — hard filter applied at Stage 1's exit, before Stage 2 ever sees the candidate; Stage 2 still reports a diet badge for transparency but doesn't score on it

**Footer**: Algorithms — Stage 2 reranker

---

### Slide 9 — X-factor: the α-sweep (headline result, part 1)

**Section label**: X-FACTOR (9 / 19)

**Headline**: No corner wins everything
**Subhead**: Sweep the whole (αt, αp, αn) simplex; watch relevance and usefulness trade off.

**Visual**: the three ternary heatmaps from `data/processed/alpha_sweep_ternary.png` (notebook §7) — Recall@10 (relevance), feasible_rate@10 (cookable), useful_rate@10.

**Small note (state this before showing the plots — pantry/macros aren't hand-typed here)**: "For this sweep, each of the 2,000 users' pantry and macro targets are derived from their OWN real rating history — the non-staple ingredients and average macros of recipes they personally rated 4★+ in the past. Not a persona, not manual entry — inferred from what they've actually cooked before."

**Headline finding**: "Cookability jumps **33% → 76%** as pantry weight rises — the strongest, clearest dial. Relevance peaks at the pure-taste corner. They move in opposite directions; the optimal α is a deliberate choice, not something the data resolves for you."

**Footer**: X-factor — the headline empirical result, part 1

---

### Slide 10 — X-factor: significance & the corner table

**Section label**: X-FACTOR (10 / 19)

**Headline**: The trade-off is real, not noise
**Subhead**: Paired Wilcoxon signed-rank on the same 2,000 users — only α changes.

**Corner table** (notebook §7):

| weighting | Recall@10 % | cookable % | useful % |
|---|---|---|---|
| taste (1,0,0) | 2.70 | 33.1 | 0.38 |
| pantry (0,1,0) | 1.50 | 75.9 | 1.11 |
| nutrition (0,0,1) | 1.25 | 36.6 | 1.24 |
| balanced (.4,.3,.3) | 1.90 | 59.1 | 1.18 |

**Significance table**:

| Contrast | Δ | p-value |
|---|---|---|
| Cookable: pantry vs taste | +42.75pp | 1.5×10⁻³¹⁰ |
| Relevance: taste vs pantry | +1.20pp | 0.0047 |
| Relevance: taste vs nutrition | +1.45pp | 0.0002 |
| Useful-rate: nutrition vs taste | +0.86pp | 3.6×10⁻²⁰ |

**Closing line**: "Contrast this with Stage 1, where no CF model significantly beats popularity. Model choice barely matters here; constraint weighting matters a lot."

**Footer**: X-factor — statistical significance

---

### Slide 11 — Applicability & significance (explicit discussion slide, per rubric item 2)

**Section label**: APPLICABILITY & SIGNIFICANCE (11 / 19)

**Headline**: Back to the fridge, with numbers
**Subhead**: Tie the X-factor back to the five-minute decision from Slide 2.

**Body**: "The cookable-rate jump (33% → 76%) is the empirical version of the motivation scene: turning up the pantry dial means three out of every four top-10 recommendations are things you could actually make tonight with what's already in the kitchen, instead of one in three. That's a direct answer to 'what should I cook tonight' — not 'what would you enjoy in the abstract.'"

**Significance points**:
- Recipe apps today optimize one constraint and call it personalization; PantryPlate makes the trade-off explicit and user-controlled
- The routing finding (Slide 7) generalizes beyond recipes: when warm and cold regimes need fundamentally different signal, don't force one model — route
- The honest caveat (macro near-rate stays low) shows where the applicability ceiling currently is — useful for anyone building on this

**Footer**: Discussion — applicability and significance

---

### Slide 12 — Extension attempted: CV fridge-photo input, with evidence

**Section label**: EXTENSION ATTEMPTED (12 / 19)

**Headline**: Reach Goal 1 — vision-based pantry input (built)
**Subhead**: A fridge photo becomes a pantry, no manual typing.

**Body**: "`src/vision/cv_inference.py` sends a fridge photo to Gemini 2.5 Flash, which returns a comma-separated ingredient list. That list feeds directly into Stage 2 as the user's pantry — same `s_pantry` scoring path as manual entry or a persona's preset pantry. It's wired into the Streamlit demo as a third input mode, alongside persona mode and manual walk-in entry."

**Evidence to show**: screenshot of the Streamlit app's photo-upload mode + the resulting detected-ingredient list (capture this live or from a saved session before the talk).

**Note on the other reach goal**: "Shelf-life prioritization (Reach Goal 2) was scoped in the proposal but never started — see Future Work."

**Footer**: Extensions — what we attempted, with evidence

---

### Slide 13 — Meet the personas & walk-in mode (NEW — intro before the demo)

**Section label**: DEMO SETUP (13 / 19)

**Headline**: Four ways to start a session
**Subhead**: Three pre-built personas, or type your own pantry live.

**Table** (`data/personas/*.json`):

| Mode | Diet | Macro target | Pantry | Taste signal |
|---|---|---|---|---|
| **Fitness-focused** | none | 500 kcal, 50 PDV protein — high-protein, low-carb lean | 25 lean-protein/whole-food items (chicken breast, salmon, quinoa, Greek yogurt...) | 8 taste seeds — high-protein, lean-prep mains |
| **Busy vegan** | vegan (hard filter) | 550 kcal, balanced — quick weeknight meals | 25 vegan-friendly items | 8 taste seeds — quick vegan meals |
| **Family with kids** | none — the neutral/loose-constraints reference persona | 700 kcal, balanced — feeding multiple people | 25 versatile, kid-friendly items | 8 taste seeds |
| **Walk-in (live)** | audience-typed, optional | audience-typed, optional | audience types it live, on the spot | none — no taste seeds exist for a walk-in identity |

**Body**: "The three personas are documented, repeatable test cases — every macro target, pantry item, and taste seed is checked into `data/personas/`, so the same persona always produces the same Stage 1 pool. Walk-in mode proves the system isn't hardcoded to them: anyone in this room can type a pantry and a restriction and get a real recommendation, routed the same way (Stage 1 → diet filter → Stage 2 rerank)."

**Small note**: "If a walk-in user gives no pantry at all, Stage 1 falls back to Popularity (no content signal to rank against); otherwise it's SBERT ranked against the typed pantry text."

**Footer**: Demonstration — meet the personas, before the live walkthrough

---

### Slide 14 — Demo walkthrough: one persona, start to finish

**Section label**: DEMONSTRATION (14 / 19)

**Headline**: vegan_busy, end to end
**Subhead**: SBERT(taste seeds) → diet hard-filter → Stage 2 rerank → top-5.

**Table** (notebook §8, real output — `e2e-17-demo`):

| recipe | s_taste | s_pantry | s_nutrition | final |
|---|---|---|---|---|
| greens and garlic | 1.000 | 0.200 | 0.005 | 0.561 |
| spinach with chickpeas | 0.495 | 0.333 | 0.450 | 0.438 |
| open faced falafel burgers | 0.158 | 0.154 | 0.421 | 0.209 |
| chickpea salad | 0.327 | 0.083 | 0.037 | 0.196 |
| veggie medley | 0.074 | 0.300 | 0.327 | 0.192 |

**Closing line**: "Every recipe shown is vegan (hard filter already applied) — what's visible here is purely the blend of taste + pantry + nutrition. Moving the α-sliders in the live app changes this ordering in real time."

**Footer**: Demonstration — a concrete, real persona run

---

### Slide 15 — Live demo (Streamlit, in-person)

**Section label**: LIVE DEMO (15 / 19)

**Headline**: Now, live
**Subhead**: `streamlit run streamlit_app.py`

**Bullets** (notebook §9):
- Persona mode — pick fitness / vegan / family; pantry, macros, restrictions pre-load and are editable
- Walk-in mode — type a pantry + restriction live, on the spot
- Fridge photo mode — Gemini Vision fills the pantry from a picture
- Three α-sliders, auto-normalized — recommendations + per-recipe score breakdown update live; cards show ingredient color-coding (✓ have / staple / ✗ need) and a novelty badge for cold items

**Demo arc**: pick vegan_busy → show top-10 → expand a card → slide pantry up → watch cookable recipes rise.

**Footer**: Demonstration — the working system, live

---

### Slide 16 — Limitations

**Section label**: LIMITATIONS (16 / 19)

**Headline**: Say these before they're asked
**Subhead**: Honest framing, from the notebook (§10).

**Bullets**:
- Relevance numbers look small (single-digit % Recall@10) — normal for LOO on a 231K-item catalogue; report @100 (pool coverage) alongside @10
- The macro constraint is hard — all-5-macros-±20% rarely all hold; cookability is the cleaner constraint story
- CV (fridge photo) is an optional input path, not a core model — it just fills the pantry
- Hybrids underperform routing — a negative result, but a useful one
- No live deployment — course deliverable is the presentation + a cooked recipe, not a service

**Footer**: Honest framing — limitations

---

### Slide 17 — Future work (per rubric item 5 — required, do not cut)

**Section label**: FUTURE WORK (17 / 19)

**Headline**: What we'd do with another sprint
**Subhead**: Three categories — descoped reach goals, an inconclusive hypothesis, and harder modeling.

**Bullets**:
- **Shelf-life prioritization** (Reach Goal 2, proposal Slide 18) — never started; would add a 5th constraint score (`s_shelf`) to Stage 2
- **Hypothesis 2** (optimal α differs by user type, proposal Slide 11) — explored post-proposal with a per-persona α-sweep; inconclusive. Relevance still peaks at pure-taste for every persona; the useful/near-rate collapses to ~0% because fixed external macro targets (e.g. a fitness persona's 50 PDV protein target) are far harder to satisfy than the self-consistent, history-derived targets used in the headline sweep. Needs either looser macro tolerance or fewer simultaneous macro constraints to produce a clean result
- **Macro near-rate** — stays at 2.5–3.9% across the whole simplex; hitting all 5 macros within ±20% at once is hard. Worth exploring per-macro weighting or a relaxed "hit-3-of-5" definition
- **Sequential models** (SASRec/GRU4Rec, proposal stretch) — not attempted; would test whether order-of-cooking signal beats the static CF/content split
- **LightGCN/NeuMF** (Slide 6) — promising on Warm@10 but unreviewed; would need to go through the team's locked-menu process before counting as a real contender

**Footer**: Future work

---

### Slide 18 — Closing / takeaways

**Section label**: CLOSING (18 / 19)

**Headline**: The five things to remember
**Subhead**: (notebook's own TL;DR, §0 — use verbatim, it's already tight)

**Bullets**:
1. The task: recommend recipes under competing constraints, not just "what will you like"
2. Two-stage architecture: Stage 1 swappable, Stage 2 is the contribution
3. Dual-track evaluation: different models win warm vs cold — which motivates routing
4. Model routing beats one model: BPR for warm, SBERT for cold, every hybrid loses to routing
5. The X-factor: the (αt, αp, αn) trade-off — no setting wins everything; the optimal balance is a deliberate choice

**Footer**: Closing — the headline takeaways

---

### Slide 19 — Q&A / backup

**Section label**: Q&A (19 / 19)

**Headline**: Questions?
**Footer**: Thank you · PantryPlate · Final Presentation

*(Keep this slide light — backup material, if needed as an appendix, lives in `notebooks/pantryplate_e2e.ipynb` §12 "Pointers / reproduce.")*

---

## 4. Sanity checks before declaring done

- [ ] All 19 slides present, numbered 1/19 through 19/19
- [ ] Slide 4 uses the notebook's **post-cleaning** dataset numbers (681K/5,900/10,393), not the proposal's raw numbers (698,901/7,023/10,393)
- [ ] Slide 6's merged table keeps the LightGCN/NeuMF exploratory labeling intact — don't let them blend into the official 7-model menu
- [ ] Slide 7 frames hybrids losing to routing as a **finding**, not a failure
- [ ] Slide 9/10 ternary plots + corner table pulled directly from `data/processed/alpha_sweep_ternary.png` and notebook §7 — no invented numbers
- [ ] Slide 12 has an actual screenshot of the CV fridge-photo flow, not just a description
- [ ] Slide 13 (personas + walk-in intro) comes before Slide 14's single-persona walkthrough, not after
- [ ] Slide 17 (Future Work) is present and not cut — it's a required rubric item
- [ ] Slide 17 states Hypothesis 2 as **inconclusive**, not as a positive result — `data/processed/alpha_sweep_persona_*.csv` exists but is not cited as evidence anywhere else in the deck
- [ ] No dedicated slide for shelf-life prioritization (Reach Goal 2) — one line in Future Work only
- [ ] Every slide maps to one of the five rubric items in Section 1

---

## End of brief
