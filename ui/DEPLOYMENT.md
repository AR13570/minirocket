Name: Shreyas Shinde
Student ID: 26104692
This is my part of the code. (Deployment Plan - step-by-step architecture for taking the MiniRocket Explorer from a laptop demo to a containerised, monitored cloud service with CI/CD and model versioning)

# Deployment Plan: MiniRocket Explorer

This document describes how the UI and API in this folder go from a local demo to a
production-style deployment. Stages 0-2 can be done with what is in this folder today;
stages 3-5 describe the production setup step by step.

---

## 1. Goals and requirements

| Requirement | Target | How it is met |
|---|---|---|
| Fast predictions | < 1 s per request | models are trained **before** deployment and kept in memory; numba code is compiled at image build time |
| Reproducible | same input, same output, any machine | fixed seeds, versioned model artifacts with metadata, pinned dependencies, one Docker image |
| Explainable | every prediction can be explained | `/explain` endpoint + heat maps in the UI |
| Cheap to run | near zero when idle | serverless containers that scale to zero (or one warm instance) |
| Safe | no secrets in code, minimal attack surface | non-root container, HTTPS, input validation, secret manager |
| Maintainable | changes are tested before they go live | CI runs the 28 tests on every push; staging before production |

---

## 2. Target architecture

```
                         ┌──────────────────────────────────────────────┐
   Users (browser) ──►   │  HTTPS load balancer (managed by the cloud)  │
                         └──────────────┬───────────────────────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
   ┌──────────────────────────┐                 ┌──────────────────────────────┐
   │ Frontend                 │   HTTP/JSON     │ Inference API (FastAPI)      │
   │ now: Streamlit (app.py)  │ ──────────────► │ api.py, stateless container, │
   │ later: React / Next.js   │                 │ autoscaled 0..N instances    │
   └──────────────────────────┘                 └──────────────┬───────────────┘
                                                               │ loads at start-up
                                                               ▼
   ┌──────────────────────────┐   writes        ┌──────────────────────────────┐
   │ Training pipeline        │ ──────────────► │ Model registry / artifacts   │
   │ train_and_save.py        │                 │ object storage, versioned:   │
   │ (CI job or scheduled)    │                 │ <dataset>_seed<s>.joblib     │
   └────────────┬─────────────┘                 │ + metadata .json             │
                │ reads                         └──────────────────────────────┘
                ▼
   ┌──────────────────────────┐                 ┌──────────────────────────────┐
   │ Data (UCR archive /      │                 │ Monitoring & logging         │
   │ customer data)           │                 │ latency, errors, prediction  │
   └──────────────────────────┘                 │ and confidence drift, alerts │
                                                └──────────────────────────────┘
```

**Key design decisions**

1. **Training is separated from serving.** Models are trained by `train_and_save.py` (the
   training pipeline) and saved as artifacts. The API only loads them. A user never waits
   for training, and the same artifact can be tested before it is released.
2. **A stateless API.** Each API instance holds models in memory but no user state, so the
   platform can add or remove instances freely (horizontal scaling).
3. **One shared service layer** (`service.py`). The UI and API cannot drift apart, and the
   logic is tested without any web framework.
4. **Frontend and API are separate services.** The Streamlit demo calls `service.py` directly
   (simplest for a demo); a production frontend calls the API over HTTP, so it can be
   rewritten (e.g. React) without touching the models.

---

## 3. Technology choices

| Layer | Choice | Why | Alternatives |
|---|---|---|---|
| Model code | NumPy + numba + scikit-learn | already used by MiniRocket; fast on CPU, no GPU needed | — |
| Demo UI | Streamlit | interactive Python UI in one file, ideal for demos | Gradio, Dash |
| Production UI | React / Next.js | full control over design, runs in any browser, calls the API | Vue, Svelte |
| API | FastAPI + Uvicorn | automatic validation (pydantic) and interactive docs (`/docs`), async-ready | Flask, Django REST |
| Packaging | Docker | identical environment on laptop, CI and cloud | Conda env + VM |
| Hosting (demo) | Hugging Face Spaces or Streamlit Community Cloud | free, public URL in minutes | — |
| Hosting (prod) | Google Cloud Run | serverless containers, HTTPS included, scales to zero, pay per use | AWS App Runner / ECS Fargate, Azure Container Apps |
| Image registry | Google Artifact Registry | private, close to Cloud Run, vulnerability scanning | GitHub Container Registry, AWS ECR |
| Artifacts | Cloud Storage bucket with object versioning | cheap, versioned, any service can read it | MLflow Model Registry, AWS S3 |
| Experiment tracking | MLflow | logs parameters, metrics and artifacts per training run | Weights & Biases |
| CI/CD | GitHub Actions | runs tests, builds and deploys on every push | GitLab CI, Cloud Build |
| Monitoring | Cloud Logging + Cloud Monitoring | built into Cloud Run; dashboards and alerts | Prometheus + Grafana |

---

## 4. Step-by-step deployment

### Stage 0: run locally (available now)

```bash
cd minirocket                     # the repository root
pip install streamlit fastapi uvicorn                      # once
python ui/train_and_save.py --datasets GunPoint ECG200   # optional: pre-train
streamlit run ui/app.py                  # UI  -> http://localhost:8501
uvicorn api:app --app-dir ui --port 8000 # API -> http://localhost:8000/docs
```

### Stage 1: containerise (available now; needs Docker Desktop)

```bash
cd minirocket                     # the repository root
docker build -f ui/Dockerfile -t minirocket-explorer .
docker run -p 8501:8501 minirocket-explorer                # UI
docker compose -f ui/docker-compose.yml up --build    # UI + API together
```

What the Dockerfile does (best practices):

* **slim base image** (`python:3.11-slim`) to keep the image small;
* **dependencies in their own layer**, so rebuilding after a code change is fast;
* **models and numba cache built during `docker build`**, so there is no cold start;
* **non-root user** at run time;
* **port from `$PORT`**, as cloud platforms expect.

### Stage 2: free public demo (for a presentation or CV link)

**Option A: Hugging Face Spaces (Docker)**

1. Create a free account at huggingface.co → *New Space* → SDK **Docker**.
2. Push the project (or a copy containing `code/`, `benchmarks/`, `experiments/`, `results/`, `ui/`) to the Space's Git repository.
3. Put the Dockerfile at the repository root (copy of `ui/Dockerfile`) and set the Space's
   app port to 8501 in its README settings.
4. The Space builds the image and serves the UI at a public URL.

**Option B: Streamlit Community Cloud**

1. Push the project to a GitHub repository.
2. On share.streamlit.io choose the repository and set the main file to `ui/app.py`.
3. It installs the packages from `requirements.txt` and starts the app.

### Stage 3: production on Google Cloud Run

```bash
# one-time setup
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create minirocket --repository-format=docker --location=europe-west1

# build in the cloud and push the image (tag = Git commit, so every release is traceable)
TAG=$(git rev-parse --short HEAD)
IMAGE=europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/minirocket/explorer:$TAG
gcloud builds submit --tag $IMAGE .            # uses the Dockerfile copied to the project root

# deploy the API (2 GiB memory: models + numba; 1 warm instance avoids cold starts)
gcloud run deploy minirocket-api --image $IMAGE --region europe-west1 \
  --command uvicorn --args "api:app,--app-dir,ui,--host,0.0.0.0,--port,8080" \
  --port 8080 --memory 2Gi --cpu 2 --min-instances 1 --max-instances 5 --concurrency 20

# deploy the UI from the same image
gcloud run deploy minirocket-ui --image $IMAGE --region europe-west1 \
  --port 8080 --memory 2Gi --min-instances 0 --max-instances 3 --allow-unauthenticated
```

(`gcloud builds submit` looks for a `Dockerfile` in the build folder: copy
`ui/Dockerfile` to the repository root in the deployment repository, or use a
`cloudbuild.yaml` that passes `-f ui/Dockerfile`.)

Cloud Run gives each service an HTTPS URL. Use `--no-allow-unauthenticated` on the API and
put an API gateway or identity-aware proxy in front of it when real users or data are involved.

*AWS equivalent:* push the image to ECR and run it with App Runner (simplest) or ECS on Fargate
behind an Application Load Balancer.

### Stage 4: CI/CD with GitHub Actions

Every push runs the tests; every merge to `main` builds, deploys to **staging**, and after a
manual approval deploys to **production**. Save as `.github/workflows/deploy.yml` in the
deployment repository:

```yaml
name: test-build-deploy
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r ui/requirements.txt
      - run: PYTHONPATH=code python -B -m unittest discover -s tests
      - run: PYTHONPATH=code python -B -m unittest discover -s ui -p "test_*.py"

  deploy-staging:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions: {contents: read, id-token: write}     # keyless login (Workload Identity Federation)
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_DEPLOY_SA }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: |
          IMAGE=europe-west1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/minirocket/explorer:${GITHUB_SHA::7}
          gcloud builds submit --tag $IMAGE .
          gcloud run deploy minirocket-ui-staging --image $IMAGE --region europe-west1 --port 8080

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production          # GitHub "environment" with required reviewers = manual approval
    permissions: {contents: read, id-token: write}
    steps:
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_DEPLOY_SA }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: |
          IMAGE=europe-west1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/minirocket/explorer:${GITHUB_SHA::7}
          gcloud run deploy minirocket-ui --image $IMAGE --region europe-west1 --port 8080
```

Best practices used: tests gate every deployment; **keyless authentication** (no long-lived
cloud keys stored in GitHub); images tagged with the commit SHA; staging before production;
manual approval for production; easy **rollback** (`gcloud run services update-traffic
minirocket-ui --to-revisions=PREVIOUS_REVISION=100`).

### Stage 5: MLOps (operating the models)

| Practice | How |
|---|---|
| **Model versioning** | every artifact has a `.json` with dataset, seed, accuracy, chosen features, library versions and date (already produced by `train_and_save.py`); store artifacts in a versioned bucket, path `models/<dataset>/<version>/` |
| **Experiment tracking** | log each training run to MLflow (parameters, test accuracy, early-classification metrics, artifact) to compare versions |
| **Release gate** | a new model is only promoted if its accuracy is not lower than the current one (compare the `.json` files in CI) |
| **Model card** | short document per model: data, intended use, accuracy, limits (univariate, equal length), explanation method |
| **Input validation** | already in place: length resampling, rejection of missing values, pydantic request checks |
| **Monitoring** | request count, latency, error rate (built into Cloud Run); plus custom logs of predicted class, confidence margin and early-stopping point |
| **Drift detection** | weekly job compares recent inputs and predictions with the training data (e.g. mean/std of series, class frequencies, average margin); alert on large shifts |
| **Retraining** | when drift is detected or new labelled data arrives: run the training pipeline, compare, promote via the release gate |
| **Logging** | structured JSON logs (no raw user data stored unless consented) |

---

## 5. Security checklist

* HTTPS only (default on Cloud Run).
* API not public in production: API gateway with API keys, or identity-aware proxy.
* Rate limiting and a maximum series length to prevent abuse.
* Container runs as non-root (done in the Dockerfile); minimal base image.
* Secrets (API keys, cloud credentials) in a secret manager, never in code or images.
* Dependency and image scanning: `pip-audit` in CI, Dependabot alerts, Artifact Registry vulnerability scanning.
* Only load model artifacts from your own bucket (joblib/pickle files can execute code when loaded).

---

## 6. Performance notes (from the project's measurements)

* MiniRocket transform: milliseconds per dataset with numba (adaptive-pooling measurements on a Mac).
* The first numba call compiles code (seconds): avoided by compiling during `docker build`.
* Training per dataset (model + explainer + early classifier): seconds, done offline.
* An explanation recomputes ~2,700 convolutions in NumPy: about 0.1-0.3 s; cache explanations of
  repeated requests if traffic grows.
* Scale: the API is CPU-bound and stateless, so more traffic means more instances (Cloud Run
  `--max-instances`), not a bigger machine.

---

## 7. Frontend roadmap (production UI)

| Page | Content | API used |
|---|---|---|
| Upload & classify | drag-and-drop a series, prediction with confidence | `POST /predict` |
| Explanation | interactive heat map (hover for evidence per time step) | `POST /explain` |
| Live monitor | series streaming in, confidence bar, "decision" moment | `POST /early` |
| Model card | accuracy, chosen features, limitations | `GET /models/{dataset}` |

Built with React/Next.js and a charting library (e.g. Plotly or Recharts), deployed as a
static site (Cloud Storage + CDN, Vercel, or Netlify) that calls the API.

---

## 8. Honest summary for a CV

> Designed and implemented an explainable time-series classification service: a Streamlit
> UI and a FastAPI REST API over an adaptive MiniRocket model with exact per-time-step
> explanations and early classification; containerised with Docker (models and JIT cache
> built at image build time), with a documented cloud deployment plan (Cloud Run, CI/CD with
> GitHub Actions, versioned model artifacts, monitoring and drift detection).

Only list as "deployed" what you have actually deployed. Stages 0-1 are implemented in this folder;
stages 2-5 are the plan.
