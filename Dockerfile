FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MODEL_PATH=/app/artifacts/model_bundle.joblib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ddos_pipeline.py train_model.py app.py streamlit_app.py model_loader.py pkl_backend.py ./
COPY xg_boost_best_model.pkl xg_boost_best_model.meta.json ./
COPY .streamlit/ /app/.streamlit/
RUN mkdir -p /app/artifacts
COPY artifacts/ /app/artifacts/

EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8080"]
