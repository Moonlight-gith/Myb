from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_PATH = "metrics.db"


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    connection = get_db_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                sector TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL
            );
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_sector_metric_ts
            ON metrics (sector, metric_name, ts);
            """
        )
        connection.commit()
    finally:
        connection.close()


class IngestRequest(BaseModel):
    sector: str = Field(..., description="Sector name, e.g., finance, industrial, conglomerate, general")
    metric_name: str = Field(..., description="Metric identifier, e.g., latency_ms, throughput_rps, error_rate")
    value: float = Field(..., description="Numeric value of the metric sample")
    timestamp: Optional[datetime] = Field(
        default=None, description="Optional ISO 8601 timestamp; defaults to now (UTC)"
    )


class OptimizeRequest(BaseModel):
    sector: str
    metric_name: Optional[str] = None
    context: Optional[str] = None


class Point(BaseModel):
    ts: str
    value: float


class Series(BaseModel):
    sector: str
    metric_name: str
    points: List[Point]


class MetricsResponse(BaseModel):
    series: List[Series]


class SummaryResponse(BaseModel):
    sector: str
    metric_name: str
    count: int
    minimum: float
    maximum: float
    average: float
    moving_average_window: int
    moving_average: List[float]
    trend_slope: float


class Anomaly(BaseModel):
    index: int
    ts: str
    value: float
    z_score: float


class AnomaliesResponse(BaseModel):
    sector: str
    metric_name: str
    threshold_z: float
    anomalies: List[Anomaly]


app = FastAPI(title="Efficiency Platform Starter API", version="0.1.0")

# Allow browser access from file:// and localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _now_epoch_seconds() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _epoch_to_iso(ts_seconds: int) -> str:
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).isoformat()


def _fetch_series(
    sector: Optional[str],
    metric_name: Optional[str],
    limit: Optional[int],
    since: Optional[int],
    until: Optional[int],
) -> List[Series]:
    connection = get_db_connection()
    try:
        where_clauses: List[str] = []
        params: List[Any] = []

        if sector is not None:
            where_clauses.append("sector = ?")
            params.append(sector)
        if metric_name is not None:
            where_clauses.append("metric_name = ?")
            params.append(metric_name)
        if since is not None:
            where_clauses.append("ts >= ?")
            params.append(since)
        if until is not None:
            where_clauses.append("ts <= ?")
            params.append(until)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""

        # Query and group
        query = f"""
            SELECT sector, metric_name, ts, value
            FROM metrics
            {where_sql}
            ORDER BY sector, metric_name, ts ASC
            {limit_sql}
        """
        rows = connection.execute(query, params).fetchall()

        series_map: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
        for row in rows:
            key = (row["sector"], row["metric_name"])
            series_map.setdefault(key, []).append((row["ts"], float(row["value"])) )

        result: List[Series] = []
        for (sec, met), points in series_map.items():
            result.append(
                Series(
                    sector=sec,
                    metric_name=met,
                    points=[Point(ts=_epoch_to_iso(ts), value=val) for ts, val in points],
                )
            )
        return result
    finally:
        connection.close()


def _compute_summary(values: List[float], window: int) -> Tuple[float, float, float, List[float], float]:
    if not values:
        raise ValueError("No values to summarize")

    array = np.array(values, dtype=float)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    average = float(np.mean(array))

    # Simple moving average
    if window <= 1:
        moving_average = array.tolist()
    else:
        kernel = np.ones(window) / window
        moving_average = np.convolve(array, kernel, mode="valid").tolist()

    # Trend slope via linear regression on latest window or entire series if short
    consider = array[-max(window, min(len(array), window)) :]
    if consider.size >= 2:
        x = np.arange(consider.size)
        slope = float(np.polyfit(x, consider, 1)[0])
    else:
        slope = 0.0

    return minimum, maximum, average, moving_average, slope


def _detect_anomalies(values: List[float], threshold_z: float) -> List[Tuple[int, float]]:
    if len(values) < 2:
        return []
    array = np.array(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array))
    if std == 0:
        return []
    z_scores = (array - mean) / std
    indices = np.where(np.abs(z_scores) >= threshold_z)[0]
    return [(int(i), float(z_scores[int(i)])) for i in indices]


def _sector_suggestions(
    sector: str, metric_name: Optional[str], values: List[float]
) -> List[Dict[str, Any]]:
    sector_lower = sector.strip().lower()

    def suggestion(title: str, description: str, impact: str) -> Dict[str, Any]:
        return {"title": title, "description": description, "estimated_impact": impact}

    suggestions: List[Dict[str, Any]] = []

    # Generic baselines
    suggestions.append(
        suggestion(
            "Instrument and trace",
            "Ensure distributed tracing and structured logging are enabled across services to localize latency and errors quickly.",
            "Medium",
        )
    )
    suggestions.append(
        suggestion(
            "Right-size concurrency",
            "Tune worker pool sizes and connection pools based on throughput and tail latencies observed.",
            "Medium",
        )
    )

    if sector_lower in {"finance", "financial", "fintech"}:
        suggestions.extend(
            [
                suggestion(
                    "Idempotent payment APIs",
                    "Use idempotency keys and request deduplication to reduce double-processing and chargebacks.",
                    "High",
                ),
                suggestion(
                    "Transaction tracing",
                    "Correlate ledger, payment, and notification paths to identify reconciliation delays.",
                    "High",
                ),
                suggestion(
                    "Hot path caching",
                    "Cache FX rates and risk lookups with bounded TTL; precompute aggregates.",
                    "Medium",
                ),
            ]
        )
    elif sector_lower in {"industrial", "manufacturing"}:
        suggestions.extend(
            [
                suggestion(
                    "Condition-based maintenance",
                    "Use vibration/temperature thresholds and rolling windows to schedule maintenance before failures.",
                    "High",
                ),
                suggestion(
                    "Edge buffering",
                    "Buffer telemetry locally during link loss; batch-upload with backpressure handling.",
                    "Medium",
                ),
                suggestion(
                    "PLC/SCADA optimization",
                    "Segment networks and prioritize critical control traffic to reduce latency spikes.",
                    "Medium",
                ),
            ]
        )
    elif sector_lower in {"conglomerate", "enterprise", "general"}:
        suggestions.extend(
            [
                suggestion(
                    "Portfolio standardization",
                    "Harmonize telemetry schemas across business units to enable cross-domain analytics.",
                    "High",
                ),
                suggestion(
                    "Shared platform services",
                    "Centralize identity, observability, and messaging to reduce duplicated effort.",
                    "Medium",
                ),
            ]
        )

    # Metric-specific heuristics
    metric_lower = (metric_name or "").lower()
    if metric_lower.endswith("latency") or metric_lower.endswith("latency_ms"):
        suggestions.append(
            suggestion(
                "Latency budget split",
                "Adopt per-tier latency budgets and enforce with SLO-based alerts.",
                "Medium",
            )
        )
    if metric_lower in {"error_rate", "errors", "failure_rate"}:
        suggestions.append(
            suggestion(
                "Circuit breakers and retries",
                "Introduce exponential backoff and egress circuit breakers to avoid cascading failures.",
                "High",
            )
        )

    # Data-driven hint based on recent trend
    if values:
        last_values = values[-5:]
        if len(last_values) >= 2 and np.mean(last_values[-2:]) > np.mean(last_values[:2]) * 1.25:
            suggestions.append(
                suggestion(
                    "Recent regression detected",
                    "Recent samples are notably worse than earlier ones; bisect recent changes and roll back risky deployments.",
                    "High",
                )
            )

    return suggestions


@app.post("/api/v1/ingest")
def ingest(request: IngestRequest) -> Dict[str, Any]:
    ts_seconds = (
        int(request.timestamp.replace(tzinfo=timezone.utc).timestamp())
        if request.timestamp is not None
        else _now_epoch_seconds()
    )

    connection = get_db_connection()
    try:
        cursor = connection.execute(
            "INSERT INTO metrics (ts, sector, metric_name, value) VALUES (?, ?, ?, ?)",
            (ts_seconds, request.sector, request.metric_name, float(request.value)),
        )
        connection.commit()
        inserted_id = int(cursor.lastrowid)
    finally:
        connection.close()

    return {
        "id": inserted_id,
        "ts": _epoch_to_iso(ts_seconds),
        "sector": request.sector,
        "metric_name": request.metric_name,
        "value": float(request.value),
    }


@app.get("/api/v1/metrics", response_model=MetricsResponse)
def get_metrics(
    sector: Optional[str] = Query(default=None),
    metric_name: Optional[str] = Query(default=None),
    limit: Optional[int] = Query(default=None, ge=1),
    since: Optional[int] = Query(default=None, description="UNIX epoch seconds inclusive"),
    until: Optional[int] = Query(default=None, description="UNIX epoch seconds inclusive"),
) -> MetricsResponse:
    series = _fetch_series(sector=sector, metric_name=metric_name, limit=limit, since=since, until=until)
    return MetricsResponse(series=series)


@app.get("/api/v1/summary", response_model=SummaryResponse)
def get_summary(
    sector: str = Query(...),
    metric_name: str = Query(...),
    window: int = Query(5, ge=1, le=200),
    limit: Optional[int] = Query(default=None, ge=1),
    since: Optional[int] = Query(default=None),
    until: Optional[int] = Query(default=None),
) -> SummaryResponse:
    series = _fetch_series(sector=sector, metric_name=metric_name, limit=limit, since=since, until=until)
    if not series:
        raise HTTPException(status_code=404, detail="No data for given filters")

    points = series[0].points
    values = [p.value for p in points]
    minimum, maximum, average, moving_average, slope = _compute_summary(values, window)

    return SummaryResponse(
        sector=sector,
        metric_name=metric_name,
        count=len(values),
        minimum=minimum,
        maximum=maximum,
        average=average,
        moving_average_window=window,
        moving_average=moving_average,
        trend_slope=slope,
    )


@app.get("/api/v1/anomalies", response_model=AnomaliesResponse)
def get_anomalies(
    sector: str = Query(...),
    metric_name: str = Query(...),
    threshold_z: float = Query(3.0, gt=0),
    limit: Optional[int] = Query(default=None, ge=1),
    since: Optional[int] = Query(default=None),
    until: Optional[int] = Query(default=None),
) -> AnomaliesResponse:
    series = _fetch_series(sector=sector, metric_name=metric_name, limit=limit, since=since, until=until)
    if not series:
        raise HTTPException(status_code=404, detail="No data for given filters")

    points = series[0].points
    values = [p.value for p in points]
    indices_with_z = _detect_anomalies(values, threshold_z=threshold_z)

    anomalies: List[Anomaly] = []
    for idx, z in indices_with_z:
        p = points[idx]
        anomalies.append(Anomaly(index=idx, ts=p.ts, value=p.value, z_score=z))

    return AnomaliesResponse(
        sector=series[0].sector,
        metric_name=series[0].metric_name,
        threshold_z=threshold_z,
        anomalies=anomalies,
    )


@app.post("/api/v1/optimize")
def post_optimize(request: OptimizeRequest) -> Dict[str, Any]:
    # Pull recent values for context
    series = _fetch_series(
        sector=request.sector, metric_name=request.metric_name, limit=None, since=None, until=None
    )
    values: List[float] = []
    if series:
        values = [p.value for p in series[0].points]

    suggestions = _sector_suggestions(request.sector, request.metric_name, values)

    return {
        "sector": request.sector,
        "metric_name": request.metric_name,
        "suggestions": suggestions,
    }