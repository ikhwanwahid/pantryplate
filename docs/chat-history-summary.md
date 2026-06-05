# Discussion History Summary

Context distilled from the planning discussion that led to this project.
Useful background for Claude Code so you don't have to re-explain decisions.

---

## The domain exploration journey

We considered and rejected several domains before settling on recipes:

### LEGO (rejected)
**Idea**: Recommend LEGO sets a user can build from their existing parts.
**Killed by**: Rebrickable API doesn't expose other users' collections — only the authenticated user's own. Scraping public profiles is slow, gray-area, and yields ~few thousand users at best. Without bulk user data, the project would have been content-based filtering with no real CF signal. Documented six workarounds; concluded none of them addressed the structural issue cleanly enough.

### 3D printing models (Thingiverse, Printables)
**Idea**: Recommend printable models with a "graded implicit feedback" angle (likes < downloads < makes).
**Strong candidate**, would have worked. Lost out to recipes because recipes hit broader audience accessibility.

### Image-to-music (SoundFrame)
**Idea**: Given an image, recommend music that matches its mood.
**Killed by**: (1) Spotify Million Playlist Dataset no longer publicly downloadable. (2) Spotify Audio Features API deprecated for new apps in Nov 2024. (3) Project was leaning toward representation learning rather than recsys — even with workarounds, the recsys content was thin compared to multimodal ML work. Identified by the team as falling outside what a recsys class is grading.

### Fitness tracking (briefly explored)
**Verdict**: Same user-data accessibility issue as LEGO. Strava/Fitbit/MyFitnessPal all closed. Catalogs are accessible (Wger, Free Exercise DB) but real user behavior at scale is not. Same structural problem.

### Chess (Lichess) — considered, rejected
**Strong recsys candidate**, real per-user data at massive scale. Rejected because audience accessibility was a concern — most students don't play chess, demo would need significant domain explanation.

### KHU "Smart Refrigerator" project (suggested by groupmate, rejected)
**The KHU GitHub project**: trains a YOLOv5 model on fridge images and matches detected ingredients against a JSON of recipes. Looks impressive but the "recommender" is a dict lookup — not a recommender system in any meaningful sense. Adopting this as-is would have meant doing a CV project with thin recsys, exactly the failure mode the rubric penalizes. We extracted the useful idea (fridge-photo input as a UX feature) and re-positioned it as a reach goal layered on top of real recsys.

---

## Why recipes (Food.com) won

1. **Real user behavior data**: ~1.1M user-recipe ratings, ~226K active users — comparable to MovieLens-25M scale. CF, MF, BPR, sequential modeling all become natively applicable.

2. **Audience universal**: everyone cooks and eats; the problem framing needs zero setup.

3. **Constraint angle preserved**: the multi-constraint (pantry, nutrition, diet, taste) structure preserves what made LEGO appealing (constraint-aware recsys) without the data accessibility wall.

4. **No scraping required**: Kaggle dataset is a 1.5 GB direct download. Catalog + interactions + nutrition + tags all in clean CSVs.

5. **Strong demo physicality**: you can literally cook the recommended recipe and bring it to class.

---

## The structural argument (worth re-stating)

A common failure mode in coursework recsys projects: no real user-behavior data, so the project becomes "ask the user their preferences, filter the catalog" — which is search, not recommendation.

PantryPlate avoids this through the two-stage architecture:
- **Stage 1 (real recsys)** — learns user taste from observed Food.com ratings via MF, BPR, hybrid models. The taste signal is *learned from behavior*, not declared.
- **Stage 2 (X-factor)** — multi-constraint reranking layers context (pantry, macros, diet) on top. Constraints refine learned preferences; they don't replace them.

If you ever find yourself collapsing the stages and using constraints as the primary lens with taste as a secondary scoring, you've drifted toward the filtering trap. Stay disciplined.

---

## Why the CV reach-goal scoping

The team was initially tempted to make computer vision (fridge-photo input) a core component. We explicitly de-scoped it for these reasons:

1. The course is recsys, not computer vision. Spending project time on YOLO training competes with recsys work.
2. Pretrained vision-LLMs (Claude, GPT-4V, Gemini) do ingredient detection out of the box with zero training. The "CV work" is API integration.
3. CV scope creep is a known risk — "let's improve the model accuracy" can absorb arbitrary time.

Resolution: CV is a reach goal added at the Week 4 decision gate, using a vision-LLM (no custom training), framed as an input-modality enhancement rather than a core contribution. Always-on fallback is manual ingredient typing.

---

## Key technical decisions already made (don't re-debate)

1. **Domain**: recipes (Food.com), not LEGO/music/fitness/chess
2. **Dataset**: Food.com Kaggle dataset (~230K recipes, ~1.1M reviews)
3. **Architecture**: two-stage (Stage 1 candidate generation + Stage 2 reranking)
4. **Headline metric**: Useful Recall@K (joint of predicted-positive + constraints-satisfied)
5. **Evaluation protocol**: leave-one-out by date, positive = 4 stars or higher
6. **Active user threshold**: ≥5 ratings (cold-start users evaluated separately)
7. **CV**: reach goal only, pretrained vision-LLM, decision at Week 4
8. **Demo**: web widget + physical prop (cooked recipe)

---

## What was nearly chosen but rejected

These are documented in case the team wants to revisit:

- **LEGO with Workaround 5** (content-only, no real users): rejected for filtering-vs-learning concern
- **LEGO with Workaround 6** (combined sources + light scraping): rejected for data engineering cost (1-2 weeks before any recsys work)
- **Image-to-music** (SoundFrame): rejected because center of gravity was in representation learning, not recsys
- **Movies/books/music with a twist**: rejected because the twists would need to be substantial to differentiate, and the team wanted distance from textbook recsys domains

---

## What the proposal slides cover

17-slide deck (`PantryPlate_Proposal.pptx`):

1. Title
2. Problem ("What should I cook tonight?")
3. Contribution (trade-off as research object)
4. Dataset (Food.com Kaggle)
5. Architecture (two-stage diagram)
6. Stage 1 models (8 models mapped to course weeks)
7. Stage 2 reranker (four constraint scores + formula)
8. X-factor (α-sweep study, 3 hypotheses)
9-13. Evaluation deep-dive (5 slides covering metrics, ground truth, protocol, signature metric, expected results table)
14. Reach goal — vision-based pantry input
15. Reach goal considerations (what it adds, what it costs, risks, fallback)
16. Timeline
17. Demo & close

---

## Open decisions for the team

A few things were left for the team to decide collectively:

1. **Which Stage 1 models to actually build**. Eight are listed; realistically 4-5 well-built. Pick in Week 1.
2. **Whether the CV reach goal happens**. Decided at end of Week 4 based on recsys progress.
3. **Specific persona definitions**. Five to eight personas; design them collectively.
4. **Pantry threshold for "satisfies constraint"**. Default ≥90% but could tune.
5. **Whether to add BrickLink price data** (just kidding — this isn't LEGO. Carryover from earlier discussion.).

---

## What this project is NOT

To prevent scope drift, also documented what we're NOT doing:

- NOT a meal-planning system (no multi-day planning, just per-meal recommendation)
- NOT a calorie/macro tracker (we use targets, we don't track daily intake)
- NOT a recipe generator (we retrieve from a fixed catalog, we don't synthesize)
- NOT a CV-driven kitchen assistant (CV is one input mode, not the centerpiece)
- NOT a substitution engine (ingredient substitution would be a separate project)

These are all interesting but explicitly out of scope.
