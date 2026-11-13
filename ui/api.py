"""
api.py - REST API for the models (FastAPI)

Author : Shreyas Shinde (Student ID 26104692)

Start it from the repository root:
    uvicorn api:app --app-dir ui --port 8000
Interactive documentation (try every endpoint in the browser):
    http://localhost:8000/docs

Endpoints
    GET  /health                    is the service up?
    GET  /datasets                  datasets that can be used + whether a model is ready
    GET  /models/{dataset}          metadata of the model (accuracy, chosen features, ...)
    POST /predict                   {"dataset": "GunPoint", "series": [...]} -> class + scores
    POST /explain                   same input (+ optional "target_class") -> heat map
    POST /early                     same input -> confidence at every prefix + decision point

Why an API: any frontend (a React web app, a mobile app, another service) can use the
models over HTTP without Python. The Streamlit demo uses service.py directly; in the
production design (DEPLOYMENT.md) the frontend talks to this API instead.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import service                                   # first: sets up paths and caches

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MiniRocket Explorer API", version="1.0.0",
              description="Adaptive multi-pooling MiniRocket with explanations and early classification.")

# Models stay in memory after their first use (loading/training is the slow part).
_bundles = {}


def get_bundle(dataset, seed=0):
    if dataset not in service.DEMO_DATASETS:
        raise HTTPException(status_code=404, detail=f"unknown dataset '{dataset}'")
    key = (dataset, seed)
    if key not in _bundles:
        _bundles[key] = service.load_bundle(dataset, seed=seed)
    return _bundles[key]


# ---------------------------------------------------------------------------
# Request / response formats (validated automatically by FastAPI + pydantic)
# ---------------------------------------------------------------------------
class SeriesRequest(BaseModel):
    dataset: str = Field(..., examples=["GunPoint"])
    series: List[float] = Field(..., min_length=2, description="one time series (any length; resampled if needed)")
    seed: int = 0
    target_class: Optional[str] = Field(None, description="for /explain: class to explain (default: predicted)")


class PredictResponse(BaseModel):
    predicted_class: str
    scores: Dict[str, float]
    margin: float
    feature_set: str
    note: str


class ExplainResponse(BaseModel):
    predicted_class: str
    explained_class: str
    heat: List[float]
    decision_score: float
    strongest_window: List[int]
    summary: str
    note: str


class EarlyStep(BaseModel):
    points_seen: int
    fraction_seen: float
    prediction: str
    margin: float
    threshold: Optional[float]
    confident: bool


class EarlyResponse(BaseModel):
    steps: List[EarlyStep]
    decision_index: int
    decision: EarlyStep
    note: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": [f"{d} (seed {s})" for d, s in _bundles]}


@app.get("/datasets")
def datasets():
    return [{"name": name, "description": text,
             "model_ready": service.artifact_path(name).exists()}
            for name, text in service.DEMO_DATASETS.items()]


@app.get("/models/{dataset}")
def model_info(dataset: str, seed: int = 0):
    return get_bundle(dataset, seed).metadata


@app.post("/predict", response_model=PredictResponse)
def predict(request: SeriesRequest):
    try:
        return service.predict(get_bundle(request.dataset, request.seed), request.series)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/explain", response_model=ExplainResponse)
def explain(request: SeriesRequest):
    bundle = get_bundle(request.dataset, request.seed)
    if request.target_class is not None and request.target_class not in bundle.classes:
        raise HTTPException(status_code=422, detail=f"target_class must be one of {bundle.classes}")
    try:
        return service.explain(bundle, request.series, request.target_class)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/early", response_model=EarlyResponse)
def early(request: SeriesRequest):
    try:
        return service.early_timeline(get_bundle(request.dataset, request.seed), request.series)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
