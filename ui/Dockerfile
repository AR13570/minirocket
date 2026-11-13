# Container image for the MiniRocket Explorer (Streamlit UI + REST API in one image)
#
# Build from the REPOSITORY ROOT, because the app imports code/, benchmarks/ and experiments/:
#     docker build -f ui/Dockerfile -t minirocket-explorer .
# Run the UI:   docker run -p 8501:8501 minirocket-explorer
# Run the API:  docker run -p 8000:8000 minirocket-explorer \
#                   uvicorn api:app --app-dir ui --host 0.0.0.0 --port 8000

FROM python:3.11-slim

# Smaller, predictable Python behaviour inside containers
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/code \
    NUMBA_CACHE_DIR=/app/.numba_cache
# (one compile cache for every module, built below and reused by the running app)

WORKDIR /app

# 1. Dependencies first (this layer is cached while only the code changes)
COPY ui/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 2. Only what the app needs: the library code, the benchmark helpers, the experiment
#    loaders, the published results it displays, and the app itself
COPY code/ code/
COPY benchmarks/ benchmarks/
COPY experiments/ experiments/
COPY results/ results/
COPY README_GROUP_PRUNING.md ./
COPY ui/ ui/

# 3. Build model artifacts AND numba's compile cache at build time,
#    so the first user never waits for training or compilation (no "cold start").
#    These three datasets ship with aeon, so no internet is needed here.
RUN python ui/train_and_save.py --datasets GunPoint ItalyPowerDemand ArrowHead
#    ... and compile the remaining pages' code (fast transform, group pruning, stratified thresholds)
RUN python ui/warm_up.py --dataset GunPoint

# 4. Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000
# Cloud platforms (e.g. Cloud Run) pass the port in $PORT; default 8501 locally.
CMD ["sh", "-c", "streamlit run ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
