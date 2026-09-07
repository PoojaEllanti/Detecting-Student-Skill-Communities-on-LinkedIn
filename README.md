# Student Skill Community Analyzer

**Detecting Student Skill Communities on LinkedIn Using NLP and Louvain Community Detection**

A working Streamlit application that takes LinkedIn-style student/professional
profile data, extracts technical skills using NLP, builds a weighted student
similarity graph, and applies the **Louvain community detection algorithm**
to automatically discover clusters of students with similar skill sets
(AI/ML, Data Science, Web Development, Cloud/DevOps, Cybersecurity, etc.).

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser tab that opens:
1. Click **"Load Demo Dataset"** in the sidebar (works instantly, no upload needed).
2. Click **"🚀 Detect Communities"**.
3. Explore the Dashboard, Communities, Network, Students, Skills and
   Recommendations tabs.

No API keys, no login, no internet access required for the core demo.

## Pipeline

```
CSV Dataset
  → NLP preprocessing (clean_text)
  → Skill extraction (dictionary + multi-word phrase matching)
  → Student representation (TF-IDF over skill sets)
  → Similarity calculation (cosine similarity)
  → Weighted student graph (NetworkX, threshold or k-NN edges)
  → Louvain community detection (python-louvain)
  → Community labeling (dominant-skill → category dictionary, post-hoc)
  → Visualization (Plotly network graph, KPIs, charts)
  → Recommendations (rule/similarity based)
```

## Project Structure

```
student-skill-community/
├── app.py                  # Main Streamlit application (all pages/tabs)
├── requirements.txt
├── README.md
├── data/
│   └── demo_students.csv   # 90 synthetic student profiles
├── src/
│   ├── skill_dictionary.py # Curated skill list + category mapping
│   ├── preprocessing.py    # NLP cleaning pipeline (NLTK w/ safe fallback)
│   ├── skill_extraction.py # Column detection + hybrid skill extraction
│   ├── similarity.py       # TF-IDF + cosine similarity
│   ├── graph.py             # Weighted graph construction (threshold/k-NN)
│   ├── community.py        # Louvain algorithm wrapper (+ NetworkX fallback)
│   ├── labeling.py         # Post-hoc community labeling from dominant skills
│   ├── recommendations.py  # Skill/peer/community recommendations
│   └── data_generator.py   # Synthetic demo dataset generator
└── assets/
```

## File-by-file explanation

- **`app.py`** — the entire UI: sidebar (data input, graph settings, Louvain
  resolution), and 7 pages (Dashboard, NLP & Skills, Communities, Network,
  Students, Skills, Recommendations, About Project). Uses
  `st.cache_data` to avoid recomputing skill extraction / similarity on
  every rerun.
- **`src/skill_dictionary.py`** — ~90 curated technical skills, each mapped
  to one or more of 5 categories (AI/ML, Data Science, Web Development,
  Cloud/DevOps, Cybersecurity). Skills like `python` or `sql` are
  deliberately shared across categories — this is what creates realistic
  overlap for Louvain to resolve.
- **`src/preprocessing.py`** — lowercase → strip punctuation (while
  preserving tokens like `c++`, `c#`, `node.js`) → tokenize → remove
  stopwords → lemmatize. Tries NLTK's stopword list + WordNet lemmatizer
  first; if NLTK data can't be downloaded (offline), it transparently falls
  back to a built-in stopword list and a simple suffix-stripper, so the
  demo never crashes for lack of internet.
- **`src/skill_extraction.py`** — `detect_columns()` fuzzy-matches whatever
  column names your CSV actually has (`bio`, `about`, `bio_text`, etc. all
  map correctly) to canonical fields. `extract_all_skills_for_row()` unions
  skills parsed from a structured `skills` column (if present) with skills
  found via regex phrase-matching in free-text fields.
- **`src/similarity.py`** — turns each student's skill set into a
  space-joined "document" and fits TF-IDF across all students, then computes
  pairwise cosine similarity.
- **`src/graph.py`** — builds a `networkx.Graph` with either a similarity
  **threshold** rule or a **k-NN** rule (both configurable in the sidebar),
  so students aren't all connected to everyone (avoids a meaningless
  complete graph).
- **`src/community.py`** — runs the actual Louvain algorithm
  (`community_louvain.best_partition`) on the weighted graph and computes
  modularity. Falls back to NetworkX's built-in `louvain_communities` if
  `python-louvain` isn't installed — same algorithm either way.
- **`src/labeling.py`** — **after** Louvain returns a partition, this module
  looks at the dominant skills inside each discovered community and scores
  them against the category dictionary to produce a label like "AI/ML".
  Louvain itself never sees these categories.
- **`src/recommendations.py`** — simple, explainable rule/similarity based
  recommendations: related skills (co-occurrence within a community),
  similar students (cosine similarity ranking), relevant communities
  (skill-overlap ranking).
- **`src/data_generator.py`** — generates the 90-row synthetic demo dataset
  with 5 overlapping skill clusters (core skills + occasional
  cross-cluster "borrowed" skills), so Louvain has genuine structure to
  discover rather than a hardcoded label being echoed back.

## How to test the demo

1. `streamlit run app.py`
2. Sidebar → **Load Demo Dataset** → **Detect Communities** (default
   threshold 0.25 works well and reproduces ~5 clean communities with
   modularity ≈ 0.6–0.7).
3. Dashboard tab: check KPIs, network graph, community table, top-skills
   chart.
4. NLP & Skills tab: see the before/after cleaning example and try typing
   your own sentence.
5. Communities tab: pick each community, check its label matches its top
   skills sensibly.
6. Network tab: zoom/pan the graph, check the hub list.
7. Students tab: pick a student, confirm "similar students" makes sense
   (same-community students should dominate).
8. Skills tab: type `python` — should show it's spread across AI/ML, Data
   Science, Cloud/DevOps.
9. Recommendations tab: try both "Existing student" and "Custom skill list"
   modes.
10. Try your own CSV upload with different column names to confirm the
    column auto-detection is robust.

## 5-minute demo script

1. **(30s)** Open the app, explain the pipeline diagram (see About Project
   tab).
2. **(30s)** Click Load Demo Dataset, show the raw data preview.
3. **(45s)** Go to NLP & Skills — show the before/after cleaning, then the
   extracted-skills table (prove skills come from real text, not labels).
4. **(30s)** Back to sidebar — explain the similarity threshold / k-NN
   choice, click Detect Communities.
5. **(60s)** Dashboard — walk through KPIs, the network graph (point out
   distinct colored clusters), the community table.
6. **(45s)** Communities tab — open "AI/ML", show its top skills genuinely
   match the label.
7. **(30s)** Students tab — pick a student, show their top-3 similar
   students and explain why (shared skills).
8. **(30s)** Skills tab — search "python", show it spans multiple
   communities (an honest, non-trivial result).
9. **(30s)** Recommendations tab — show a skill recommendation for a
   student.
10. **(20s)** Close on modularity score + "communities were discovered, not
    assigned" as the key academic claim.

## Common errors and fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'community'` | `pip install python-louvain` (NOT `pip install community`, that's a different package). The app also auto-falls-back to NetworkX's built-in Louvain if this is missing. |
| `LookupError` from NLTK about missing corpora | Harmless — the app auto-downloads `stopwords`/`wordnet` on first run if internet is available, and silently falls back to a built-in stopword list if not. No action needed. |
| Blank / all-isolated network graph | Your similarity threshold is too high for your dataset. Lower it in the sidebar, or switch to "K-Nearest Neighbours" mode. |
| Only 1 giant community detected | Threshold too low (graph too dense) — raise it, or lower Louvain resolution. |
| CSV upload shows no skills extracted | Your CSV's skill/text columns don't share vocabulary with the built-in dictionary. Check the "Detected column mapping" JSON on the NLP & Skills page — if a field wasn't detected, rename that column to something like `skills`, `profile_summary`, etc. |
| `streamlit: command not found` | Use `python -m streamlit run app.py` instead, or ensure your virtualenv's `bin`/`Scripts` folder is on PATH. |
| App is slow on first run | The very first run downloads NLTK data and computes TF-IDF; subsequent reruns are cached via `st.cache_data`. |

## Notes on running as a public demo link

This project is a local Streamlit app (`streamlit run app.py`) — it isn't
hosted anywhere by default. To get a shareable link:
- **Streamlit Community Cloud** (free): push this folder to a GitHub repo,
  then deploy it at share.streamlit.io pointing at `app.py`.
- Any other host that runs `pip install -r requirements.txt && streamlit
  run app.py` will work identically (Render, Railway, a VM, etc.).
