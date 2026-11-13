Name: Shreyas Shinde
Student ID: 26104692
This is my part of the code. (Interactive UI and REST API - one presentation-ready web app with every feature of the project (fast transform, group pruning, class-stratified thresholds, adaptive multi-pooling, explanations, early classification, reproduction results) and an HTTP API, packaged for deployment)

# MiniRocket Explorer: UI and API

The web app and API live in **this one folder** (`ui/`).

* It **only reads** the library code (`code/`), the benchmark helpers (`benchmarks/`), the data
  loader in `experiments/multipool.py` and the published results (`results/`); it changes nothing
  outside this folder.
* **Don't want the UI?** Delete this folder; nothing else depends on it.

---

## 1. What is in this folder

| File | What it is |
|---|---|
| `app.py` | the web app's frame: page set-up, menu, dataset chooser (Streamlit) |
| `views/` | one file per page (10 pages, listed in section 4) |
| `common.py` | shared by all pages: feature list, cached data/models, plotting helpers |
| `transform_service.py` | logic for the Fast transform and Reproduction pages |
| `warm_up.py` | compiles every page's code once (used by the Dockerfile; optional locally) |
| `api.py` | REST API (FastAPI) for other programs / a future web frontend |
| `service.py` | shared logic used by both: load data, train/load models, predict, explain, early timeline |
| `train_and_save.py` | trains the models once and saves them to `artifacts/`, so the app starts instantly |
| `pruning_service.py` | logic for the Group pruning and Class-stratified thresholds pages (calls `code/minirocket_pruned.py`, `code/minirocket_stratified.py`) |
| `test_service.py`, `test_transform_service.py`, `test_pruning_service.py` | 4 + 3 + 5 tests for the logic behind the pages |
| `requirements.txt` | Python packages needed |
| `Dockerfile`, `docker-compose.yml` | container packaging for deployment |
| `DEPLOYMENT.md` | step-by-step deployment plan (cloud, CI/CD, MLOps, security) |
| `README.md` | this file |
| `artifacts/` | saved models (created automatically, not in Git) |

```
 app.py + views/ (Streamlit UI, 10 pages)                  api.py (REST API)
      │                    │                   │                 │
 transform_service.py   service.py ────────────┴─────────────────┘   pruning_service.py
      │                    │ imports (read-only)                          │
 code/minirocket.py     code/minirocket_adaptive.py                  code/minirocket_pruned.py
 code/minirocket_joint_ppv_simd.py   code/minirocket_explain.py      code/minirocket_stratified.py
                        code/minirocket_early.py
```

---

## 2. Install (once)

From the repository root (Python 3.10 or newer):

```bash
pip install -r ui/requirements.txt
```

---

## 3. Start the app

```bash
streamlit run ui/app.py            # from the repository root
```

The browser opens at **http://localhost:8501**. Stop it with `Ctrl + C` in the terminal.

**First time a dataset is opened:** its three models are trained and saved (roughly 10–60 s,
shown with a spinner). After that it loads instantly. To prepare everything before a
presentation:

```bash
python ui/train_and_save.py                      # all 9 demo datasets
python ui/train_and_save.py --datasets GunPoint ECG200 Trace
```

---

## 4. What the app shows

**Sidebar:** the menu with every page, then the dataset chooser (GunPoint, ECG200, Coffee,
ItalyPowerDemand, ArrowHead, Trace, SyntheticControl, CBF, Wine) and the random seed.
**Every page works on the chosen dataset**, so you can move between features at any time; models
are trained once per dataset and shared by all pages.

| Menu section | Page | What you can do |
|---|---|---|
| Overview | **Home** | key numbers for the dataset and a card for every feature with a link to its page |
| Classify & explain | **Adaptive multi-pooling** | accuracy, the feature set chosen by cross-validation (PPV or PPV+MPV), benchmark of original vs. fast transform vs. adaptive model |
| | **Explain a prediction** | pick a test series (or **upload your own CSV**); prediction, confidence and the **heat map**; choose which class to explain |
| | **Model insights** | importance of pattern widths and kernels; evidence map per class; what separates the classes |
| Real time | **Early classification** | drag "points received so far": the plot fills in as if data were arriving live; a green line marks the moment the model decides |
| Speed & size | **Fast transform (joint PPV)** | live timing of the original vs. the joint-PPV transform on the dataset, with a bit-for-bit identity check; the 113-dataset benchmark |
| | **Group pruning** | prune a model live (ranking `ridge`/`filter`, threshold strategy, shares to keep): accuracy, speed and size per share; map of surviving groups; the 113-dataset benchmark |
| | **Class-stratified thresholds** | which class supplies each group's thresholds (random vs. stratified); 3-seed comparison of the strategies; the 113 × 20-seed result |
| Evidence | **Reproduction of the paper** | our accuracy vs. the published accuracy on 109 datasets, with the largest differences |
| | **About** | the project in plain words, and full model details |

The app shows feature names only (no author names). If `code/minirocket_pruned.py` is missing,
the Group pruning and Class-stratified thresholds pages show a short notice; all other pages work.

**Uploading your own series:** a `.csv` or `.txt` file with one series, either one row
(`1.2,1.5,1.9,...`) or one column (one number per line). A header line is fine. If its
length differs from the dataset's, it is resampled automatically (the app says so).

**Suggested demo script (5 minutes):**

1. *Home*: "every improvement of the project, on one dataset".
2. *Reproduction of the paper*: "our MiniRocket matches the published accuracy (85.69% vs. 85.67%)".
3. *Fast transform*: "same features, bit for bit, in less time".
4. *Group pruning*: "keep half the groups, lose almost nothing, transform much faster".
5. *Explain a prediction* (GunPoint): "red is where the model found its evidence: the moment the hand leaves the hip".
6. *Early classification*: drag the slider; the model decides at 30% of the movement.

---

## 5. The REST API

```bash
uvicorn api:app --app-dir ui --port 8000
```

Open **http://localhost:8000/docs** for interactive documentation where every endpoint can be tried.

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | — | status, models in memory |
| GET | `/datasets` | — | available datasets and whether a saved model exists |
| GET | `/models/{dataset}` | — | model metadata (accuracy, chosen features, early metrics, …) |
| POST | `/predict` | `{"dataset": "GunPoint", "series": [...]}` | predicted class, score per class, confidence margin |
| POST | `/explain` | same (+ optional `"target_class"`) | heat map (one value per time step), strongest window, summary |
| POST | `/early` | same | prediction and confidence at every prefix length, and the decision point |

Example from Python:

```python
import json, urllib.request
body = json.dumps({"dataset": "GunPoint", "series": [-1.1, -1.1, -0.9, 0.2, 1.1, 1.2, 1.2, 0.3, -0.8, -1.0]}).encode()
request = urllib.request.Request("http://localhost:8000/predict", data=body,
                                 headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(request)))
```

Errors are clear: unknown dataset → 404, invalid class or bad series → 422.

---

## 6. Tests

```bash
python -B -m unittest discover -s ui -p "test_*.py" -v
```

They check saving and loading a model (identical predictions), the content of predictions,
explanations and early timelines (the heat map adds up to the model's score), and input handling
(resampling, missing values, CSV formats), that the fast transform gives identical features, and
that the pruning layer gives exactly the numbers `experiments/group_pruning.py` gives. Every page
of the app was also run end to end in a test harness.

---

## 7. Docker and deployment

```bash
docker build -f ui/Dockerfile -t minirocket-explorer .          # from the repository root
docker run -p 8501:8501 minirocket-explorer
docker compose -f ui/docker-compose.yml up --build                # UI + API
```

The image trains the demo models and compiles the numba code of every page (`warm_up.py`)
while it is built, so the first user does not wait. See **`DEPLOYMENT.md`** for the full plan: architecture, free public demo
(Hugging Face Spaces / Streamlit Community Cloud), production on Google Cloud Run, CI/CD with
GitHub Actions, model versioning, monitoring, drift detection and security.

---

## 8. Removing the UI

```bash
rm -rf ui
```

Nothing else needs to change: the library code, experiments and tests do not import the UI.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | `pip install streamlit` (same environment), or run `python -m streamlit run ui/app.py` |
| `No module named 'fastapi'` | `pip install fastapi uvicorn` |
| The first dataset takes long to open | it is training once; run `train_and_save.py` beforehand |
| A dataset fails to load | internet needed for the first download (except GunPoint, ItalyPowerDemand, ArrowHead) |
| Port already in use | `streamlit run ... --server.port 8502` or `uvicorn ... --port 8001` |
| Saved models look outdated | delete `ui/artifacts/` and they are rebuilt |
