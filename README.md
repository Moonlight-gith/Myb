# Efficiency Platform Starter

A ready-to-run full-stack starter you can extend into a platform that boosts other software’s efficiency across finance, industrial, conglomerate, and general company contexts.

## What you get

- Backend: FastAPI (Python) with endpoints for metric ingestion, summaries, anomaly detection, and optimization suggestions (sector-aware).
- Frontend: React (CDN) + Tailwind + Chart.js single-file dashboard for posting metrics, viewing trends, and getting suggestions.

## Run locally

### 1) Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

This will create a local SQLite database `metrics.db` in `backend/`.

### 2) Frontend

Open `frontend/index.html` in your browser (double-click). If your API runs elsewhere, edit `API Base URL` at the top of the page or change the `api_base` in localStorage.

## API

- POST `/api/v1/ingest` — send a data point `{ sector, metric_name, value }`
- GET `/api/v1/metrics` — fetch time series per sector/metric
- GET `/api/v1/summary` — min/max/avg + simple moving average trend
- GET `/api/v1/anomalies` — z-score anomaly detection
- POST `/api/v1/optimize` — heuristic, sector-aware recommendations

CORS is enabled for development. Modify as needed for production.

