PantryPlate Streamlit demo — quick start
=========================================

1. Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/

2. From this folder, install dependencies:
       uv sync

3. Run the app:
       uv run streamlit run streamlit_app.py

   First launch is slow (a minute or so) — it's fitting Popularity, SBERT, and
   BPR models in the background. Subsequent reruns in the same session are
   fast (cached).

4. Three modes in the sidebar:
   - Persona      — pick a pre-built character
   - Walk-in      — type your own pantry + restrictions live
   - Returning user — pick a real user from the dataset (or type any user_id);
                       this is the only mode where BPR (collaborative
                       filtering) runs live

Optional: the "detect pantry from a fridge photo" feature needs a free
Gemini API key (https://aistudio.google.com) saved as GEMINI_API_KEY in a
.env file in this folder. Everything else works without it.

What's included in this zip:
   streamlit_app.py        — the app
   src/                     — all model/reranker/eval code
   data/raw/RAW_recipes.csv, interactions_train.csv — the dataset (the only
                              two files this app actually reads)
   data/personas/*.json     — the 3 pre-built personas
   pyproject.toml, uv.lock  — exact dependency versions
