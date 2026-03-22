from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_PATH = "metrics.db"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                address TEXT NOT NULL,
                city TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                price REAL NOT NULL,
                bedrooms INTEGER DEFAULT 0,
                bathrooms INTEGER DEFAULT 0,
                area_sqft REAL NOT NULL,
                listed_at INTEGER NOT NULL,
                image_url TEXT
            );

            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                source TEXT NOT NULL,
                property_id INTEGER,
                status TEXT NOT NULL,
                budget REAL,
                notes TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (property_id) REFERENCES properties(id)
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER NOT NULL,
                lead_id INTEGER,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                commission REAL NOT NULL,
                closed_at INTEGER NOT NULL,
                agent TEXT NOT NULL,
                FOREIGN KEY (property_id) REFERENCES properties(id)
            );

            CREATE TABLE IF NOT EXISTS market_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                city TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL
            );
            """
        )
        conn.commit()
        _seed_demo_data(conn)
    finally:
        conn.close()


def _seed_demo_data(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0] > 0:
        return

    now = int(datetime.now(tz=timezone.utc).timestamp())
    day = 86400

    properties = [
        ("Luxury Penthouse Manhattan", "432 Park Ave", "New York", "apartment", "available", 4_800_000, 3, 3, 3200),
        ("Suburban Family Villa", "78 Maple Lane", "Los Angeles", "villa", "sold", 1_250_000, 5, 4, 4500),
        ("Downtown Studio Loft", "215 W 10th St", "Chicago", "apartment", "rented", 320_000, 1, 1, 680),
        ("Beachfront Bungalow", "9 Ocean Drive", "Miami", "villa", "available", 2_100_000, 4, 3, 3100),
        ("Tech Hub Office Space", "1200 Market St", "San Francisco", "commercial", "available", 6_500_000, 0, 0, 8400),
        ("Mountain Retreat Cabin", "55 Alpine Rd", "Denver", "villa", "pending", 890_000, 3, 2, 2200),
        ("Urban Micro Apartment", "99 Broad St", "Boston", "apartment", "available", 480_000, 1, 1, 520),
        ("Industrial Warehouse", "300 Commerce Blvd", "Dallas", "commercial", "sold", 1_900_000, 0, 0, 12000),
        ("Golf Course Estate", "7 Fairway Dr", "Scottsdale", "villa", "available", 3_400_000, 6, 5, 6800),
        ("Historic Brownstone", "44 Beacon St", "Boston", "apartment", "available", 1_100_000, 3, 2, 2100),
        ("Waterfront Condo", "1 Harbor View", "Seattle", "apartment", "pending", 750_000, 2, 2, 1400),
        ("Desert Land Plot", "Lot 12 Sand Dune Rd", "Phoenix", "land", "available", 220_000, 0, 0, 43560),
    ]

    images = [
        "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=400",
        "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=400",
        "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?w=400",
        "https://images.unsplash.com/photo-1493809842364-78817add7ffb?w=400",
        "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400",
        "https://images.unsplash.com/photo-1449844908441-8829872d2607?w=400",
        "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=400",
        "https://images.unsplash.com/photo-1486325212027-8081e485255e?w=400",
        "https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=400",
        "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400",
        "https://images.unsplash.com/photo-1554995207-c18c203602cb?w=400",
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400",
    ]

    for i, (title, address, city, ptype, status, price, beds, baths, area) in enumerate(properties):
        conn.execute(
            "INSERT INTO properties (title, address, city, type, status, price, bedrooms, bathrooms, area_sqft, listed_at, image_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (title, address, city, ptype, status, price, beds, baths, area, now - random.randint(0, 90) * day, images[i % len(images)])
        )

    leads_data = [
        ("James Anderson", "james@email.com", "+1-555-0101", "website", 1, "qualified", 5_000_000),
        ("Sarah Chen", "sarah.chen@email.com", "+1-555-0102", "referral", 4, "converted", 2_200_000),
        ("Michael Torres", "m.torres@email.com", "+1-555-0103", "social", None, "new", 400_000),
        ("Emma Williams", "emma.w@email.com", "+1-555-0104", "walk-in", 7, "contacted", 500_000),
        ("David Kim", "d.kim@email.com", "+1-555-0105", "website", 5, "qualified", 7_000_000),
        ("Olivia Brown", "olivia.b@email.com", "+1-555-0106", "referral", 9, "contacted", 3_500_000),
        ("Liam Johnson", "liam.j@email.com", "+1-555-0107", "social", 3, "new", 350_000),
        ("Ava Martinez", "ava.m@email.com", "+1-555-0108", "website", 10, "qualified", 1_200_000),
        ("Noah Davis", "noah.d@email.com", "+1-555-0109", "referral", None, "lost", 800_000),
        ("Isabella Wilson", "i.wilson@email.com", "+1-555-0110", "walk-in", 6, "converted", 1_000_000),
        ("Ethan Taylor", "e.taylor@email.com", "+1-555-0111", "website", 2, "new", 1_300_000),
        ("Mia Garcia", "mia.g@email.com", "+1-555-0112", "social", 11, "contacted", 250_000),
    ]

    for name, email, phone, source, prop_id, status, budget in leads_data:
        conn.execute(
            "INSERT INTO leads (name, email, phone, source, property_id, status, budget, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (name, email, phone, source, prop_id, status, budget, now - random.randint(0, 60) * day)
        )

    transactions_data = [
        (2, 2, "sale", 1_250_000, 37_500, "James Porter"),
        (3, 10, "rental", 3_800, 228, "Lisa Monroe"),
        (8, None, "sale", 1_900_000, 57_000, "James Porter"),
        (6, None, "sale", 870_000, 26_100, "Rachel Stone"),
        (4, 2, "sale", 2_050_000, 61_500, "Lisa Monroe"),
    ]

    for prop_id, lead_id, ttype, amount, commission, agent in transactions_data:
        conn.execute(
            "INSERT INTO transactions (property_id, lead_id, type, amount, commission, closed_at, agent) VALUES (?,?,?,?,?,?,?)",
            (prop_id, lead_id, ttype, amount, commission, now - random.randint(0, 30) * day, agent)
        )

    cities = ["New York", "Los Angeles", "Miami", "Chicago", "San Francisco"]
    metrics = ["avg_price_sqft", "days_on_market", "listing_volume", "sold_volume"]
    base_values = {
        "avg_price_sqft": {"New York": 1850, "Los Angeles": 920, "Miami": 680, "Chicago": 340, "San Francisco": 1100},
        "days_on_market": {"New York": 28, "Los Angeles": 22, "Miami": 18, "Chicago": 35, "San Francisco": 15},
        "listing_volume": {"New York": 420, "Los Angeles": 380, "Miami": 290, "Chicago": 310, "San Francisco": 210},
        "sold_volume": {"New York": 310, "Los Angeles": 270, "Miami": 220, "Chicago": 220, "San Francisco": 165},
    }
    rng = random.Random(42)
    for metric in metrics:
        for city in cities:
            base = base_values[metric][city]
            for week in range(12):
                ts = now - (11 - week) * 7 * day
                noise = rng.uniform(-0.05, 0.05)
                val = base * (1 + noise + week * 0.005)
                conn.execute(
                    "INSERT INTO market_metrics (ts, city, metric, value) VALUES (?,?,?,?)",
                    (ts, city, metric, round(val, 2))
                )

    conn.commit()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PropertyCreate(BaseModel):
    title: str
    address: str
    city: str
    type: str = Field(..., description="apartment, villa, commercial, land")
    status: str = Field(..., description="available, sold, pending, rented")
    price: float
    bedrooms: int = 0
    bathrooms: int = 0
    area_sqft: float
    image_url: Optional[str] = None


class LeadCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    source: str = Field(..., description="website, referral, walk-in, social")
    property_id: Optional[int] = None
    status: str = Field(default="new")
    budget: Optional[float] = None
    notes: Optional[str] = None


class ValuationRequest(BaseModel):
    city: str
    type: str
    area_sqft: float
    bedrooms: int = 0
    bathrooms: int = 0
    condition: str = Field(default="good")


class MarketInsightRequest(BaseModel):
    city: str
    property_type: Optional[str] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RealEstate AI Platform", version="1.0.0")

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


# ---------------------------------------------------------------------------
# Dashboard KPIs
# ---------------------------------------------------------------------------

@app.get("/api/v1/dashboard")
def get_dashboard() -> Dict[str, Any]:
    conn = get_db()
    try:
        total_props = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
        available = conn.execute("SELECT COUNT(*) FROM properties WHERE status='available'").fetchone()[0]
        sold = conn.execute("SELECT COUNT(*) FROM properties WHERE status='sold'").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM properties WHERE status='pending'").fetchone()[0]
        rented = conn.execute("SELECT COUNT(*) FROM properties WHERE status='rented'").fetchone()[0]

        total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        new_leads = conn.execute("SELECT COUNT(*) FROM leads WHERE status='new'").fetchone()[0]
        qualified_leads = conn.execute("SELECT COUNT(*) FROM leads WHERE status='qualified'").fetchone()[0]
        converted_leads = conn.execute("SELECT COUNT(*) FROM leads WHERE status='converted'").fetchone()[0]

        revenue_row = conn.execute("SELECT SUM(commission) FROM transactions").fetchone()
        total_revenue = float(revenue_row[0] or 0)

        sales_volume_row = conn.execute("SELECT SUM(amount) FROM transactions WHERE type='sale'").fetchone()
        total_sales_volume = float(sales_volume_row[0] or 0)

        avg_price_row = conn.execute("SELECT AVG(price) FROM properties").fetchone()
        avg_price = float(avg_price_row[0] or 0)

        recent_transactions = conn.execute(
            """SELECT t.id, t.type, t.amount, t.commission, t.agent, t.closed_at,
                      p.title, p.city
               FROM transactions t JOIN properties p ON t.property_id = p.id
               ORDER BY t.closed_at DESC LIMIT 5"""
        ).fetchall()

        top_agents_rows = conn.execute(
            """SELECT agent, SUM(amount) as volume, SUM(commission) as earnings, COUNT(*) as deals
               FROM transactions GROUP BY agent ORDER BY volume DESC LIMIT 5"""
        ).fetchall()

        pipeline = {
            "new": new_leads,
            "contacted": conn.execute("SELECT COUNT(*) FROM leads WHERE status='contacted'").fetchone()[0],
            "qualified": qualified_leads,
            "converted": converted_leads,
            "lost": conn.execute("SELECT COUNT(*) FROM leads WHERE status='lost'").fetchone()[0],
        }

        return {
            "properties": {
                "total": total_props,
                "available": available,
                "sold": sold,
                "pending": pending,
                "rented": rented,
                "avg_price": round(avg_price, 2),
            },
            "leads": {
                "total": total_leads,
                "new": new_leads,
                "qualified": qualified_leads,
                "converted": converted_leads,
                "conversion_rate": round(converted_leads / total_leads * 100, 1) if total_leads else 0,
                "pipeline": pipeline,
            },
            "revenue": {
                "total_commission": round(total_revenue, 2),
                "total_sales_volume": round(total_sales_volume, 2),
                "avg_commission_rate": round(total_revenue / total_sales_volume * 100, 2) if total_sales_volume else 0,
            },
            "recent_transactions": [
                {
                    "id": r["id"],
                    "type": r["type"],
                    "amount": r["amount"],
                    "commission": r["commission"],
                    "agent": r["agent"],
                    "property": r["title"],
                    "city": r["city"],
                    "closed_at": datetime.fromtimestamp(r["closed_at"], tz=timezone.utc).isoformat(),
                }
                for r in recent_transactions
            ],
            "top_agents": [
                {
                    "agent": r["agent"],
                    "volume": r["volume"],
                    "earnings": r["earnings"],
                    "deals": r["deals"],
                }
                for r in top_agents_rows
            ],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@app.get("/api/v1/properties")
def list_properties(
    status: Optional[str] = Query(default=None),
    city: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None),
    min_price: Optional[float] = Query(default=None),
    max_price: Optional[float] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    conn = get_db()
    try:
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status=?"); params.append(status)
        if city:
            clauses.append("city LIKE ?"); params.append(f"%{city}%")
        if type:
            clauses.append("type=?"); params.append(type)
        if min_price is not None:
            clauses.append("price>=?"); params.append(min_price)
        if max_price is not None:
            clauses.append("price<=?"); params.append(max_price)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM properties {where} ORDER BY listed_at DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        return {"properties": [dict(r) for r in rows], "total": len(rows)}
    finally:
        conn.close()


@app.post("/api/v1/properties")
def create_property(body: PropertyCreate) -> Dict[str, Any]:
    conn = get_db()
    try:
        now = int(datetime.now(tz=timezone.utc).timestamp())
        cur = conn.execute(
            "INSERT INTO properties (title, address, city, type, status, price, bedrooms, bathrooms, area_sqft, listed_at, image_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (body.title, body.address, body.city, body.type, body.status, body.price,
             body.bedrooms, body.bathrooms, body.area_sqft, now, body.image_url)
        )
        conn.commit()
        return {"id": cur.lastrowid, **body.model_dump(), "listed_at": now}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------

@app.get("/api/v1/leads")
def list_leads(
    status: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    conn = get_db()
    try:
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status=?"); params.append(status)
        if source:
            clauses.append("source=?"); params.append(source)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM leads {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        return {"leads": [dict(r) for r in rows], "total": len(rows)}
    finally:
        conn.close()


@app.post("/api/v1/leads")
def create_lead(body: LeadCreate) -> Dict[str, Any]:
    conn = get_db()
    try:
        now = int(datetime.now(tz=timezone.utc).timestamp())
        cur = conn.execute(
            "INSERT INTO leads (name, email, phone, source, property_id, status, budget, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (body.name, body.email, body.phone, body.source, body.property_id,
             body.status, body.budget, body.notes, now)
        )
        conn.commit()
        return {"id": cur.lastrowid, **body.model_dump(), "created_at": now}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Market metrics
# ---------------------------------------------------------------------------

@app.get("/api/v1/market")
def get_market(
    city: Optional[str] = Query(default=None),
    metric: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    conn = get_db()
    try:
        clauses: List[str] = []
        params: List[Any] = []
        if city:
            clauses.append("city=?"); params.append(city)
        if metric:
            clauses.append("metric=?"); params.append(metric)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT city, metric, ts, value FROM market_metrics {where} ORDER BY city, metric, ts ASC",
            params
        ).fetchall()
        grouped: Dict[str, Dict[str, List]] = {}
        for r in rows:
            grouped.setdefault(r["city"], {}).setdefault(r["metric"], []).append({
                "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc).isoformat(),
                "value": r["value"]
            })
        return {"market": grouped}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AI: Property Valuation
# ---------------------------------------------------------------------------

@app.post("/api/v1/ai/valuate")
def ai_valuate(req: ValuationRequest) -> Dict[str, Any]:
    base_ppsf: Dict[str, float] = {
        "new york": 1850, "los angeles": 920, "miami": 680,
        "chicago": 340, "san francisco": 1100, "boston": 810,
        "seattle": 740, "dallas": 280, "denver": 420, "phoenix": 260,
        "scottsdale": 480,
    }
    city_key = req.city.lower()
    ppsf = base_ppsf.get(city_key, 450)

    type_mult = {"villa": 1.15, "apartment": 1.0, "commercial": 1.2, "land": 0.4}.get(req.type, 1.0)
    cond_mult = {"poor": 0.75, "fair": 0.88, "good": 1.0, "excellent": 1.15}.get(req.condition, 1.0)
    bed_premium = min(req.bedrooms * 0.04, 0.20)
    bath_premium = min(req.bathrooms * 0.02, 0.10)

    estimated = req.area_sqft * ppsf * type_mult * cond_mult * (1 + bed_premium + bath_premium)

    return {
        "estimated_value": round(estimated, 0),
        "range_low": round(estimated * 0.88, 0),
        "range_high": round(estimated * 1.12, 0),
        "confidence": 0.82 if city_key in base_ppsf else 0.65,
        "price_per_sqft": round(ppsf * type_mult * cond_mult, 2),
        "factors": [
            {"factor": "Location premium", "impact": f"{round((ppsf / 450 - 1) * 100, 1):+.1f}%"},
            {"factor": "Property type", "impact": f"{round((type_mult - 1) * 100, 1):+.1f}%"},
            {"factor": "Condition", "impact": f"{round((cond_mult - 1) * 100, 1):+.1f}%"},
            {"factor": "Bedroom premium", "impact": f"+{round(bed_premium * 100, 1)}%"},
            {"factor": "Bathroom premium", "impact": f"+{round(bath_premium * 100, 1)}%"},
        ],
        "recommendation": (
            "Strong buy — below market median" if estimated < ppsf * req.area_sqft * 0.9
            else "Fairly priced relative to local comparables"
        ),
    }


# ---------------------------------------------------------------------------
# AI: Lead Scoring
# ---------------------------------------------------------------------------

@app.get("/api/v1/ai/lead-score/{lead_id}")
def ai_lead_score(lead_id: int) -> Dict[str, Any]:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Lead not found")
        lead = dict(row)

        score = 50
        score += {"new": 0, "contacted": 10, "qualified": 25, "converted": 40, "lost": -20}.get(lead["status"], 0)
        score += {"referral": 15, "website": 8, "walk-in": 12, "social": 5}.get(lead["source"], 0)
        if lead["phone"]:
            score += 8
        if lead["property_id"]:
            score += 10
        if lead["budget"] and lead["budget"] > 500_000:
            score += 7
        score = max(0, min(100, score))

        tier = "Hot" if score >= 75 else "Warm" if score >= 45 else "Cold"
        actions = []
        if lead["status"] == "new":
            actions.append("Schedule introductory call within 24h")
        if lead["status"] == "contacted":
            actions.append("Send property matches based on budget")
        if lead["status"] == "qualified":
            actions.append("Book property viewings this week")
        if not lead["phone"]:
            actions.append("Collect phone number to improve reachability")

        return {
            "lead_id": lead_id,
            "name": lead["name"],
            "score": score,
            "tier": tier,
            "next_actions": actions,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AI: Market Insights
# ---------------------------------------------------------------------------

@app.post("/api/v1/ai/market-insights")
def ai_market_insights(req: MarketInsightRequest) -> Dict[str, Any]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT metric, value FROM market_metrics WHERE city=? ORDER BY ts DESC",
            (req.city,)
        ).fetchall()

        latest: Dict[str, float] = {}
        seen: set = set()
        for r in rows:
            if r["metric"] not in seen:
                latest[r["metric"]] = r["value"]
                seen.add(r["metric"])

        insights = []
        score = 70

        dom = latest.get("days_on_market", 30)
        if dom < 20:
            insights.append({"type": "bullish", "text": f"Properties selling fast — avg {dom:.0f} days on market signals high demand."})
            score += 10
        elif dom > 40:
            insights.append({"type": "bearish", "text": f"Slow market — avg {dom:.0f} days on market. Buyers have leverage."})
            score -= 10
        else:
            insights.append({"type": "neutral", "text": f"Balanced market — avg {dom:.0f} days on market."})

        ppsf = latest.get("avg_price_sqft", 500)
        if ppsf > 1000:
            insights.append({"type": "bullish", "text": f"Premium market at ${ppsf:,.0f}/sqft — strong appreciation potential."})
            score += 8
        elif ppsf < 400:
            insights.append({"type": "opportunity", "text": f"Affordable entry at ${ppsf:,.0f}/sqft — value play for investors."})
            score += 5

        lv = latest.get("listing_volume", 300)
        sv = latest.get("sold_volume", 200)
        if lv > 0:
            absorption = sv / lv
            if absorption > 0.75:
                insights.append({"type": "bullish", "text": f"High absorption rate ({absorption:.0%}) — inventory tightening."})
                score += 8
            elif absorption < 0.5:
                insights.append({"type": "bearish", "text": f"Low absorption rate ({absorption:.0%}) — oversupply risk."})
                score -= 8

        health = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 55 else "Weak"

        return {
            "city": req.city,
            "market_health_score": min(100, max(0, score)),
            "market_health": health,
            "insights": insights,
            "key_metrics": {
                "avg_price_per_sqft": round(ppsf, 2),
                "avg_days_on_market": round(dom, 1),
                "listing_volume": round(lv),
                "sold_volume": round(sv),
                "absorption_rate": round(sv / lv * 100, 1) if lv else 0,
            },
            "recommendation": (
                "Strong market — ideal for sellers and long-term investors." if health in ("Excellent", "Good")
                else "Cautious outlook — buyers should negotiate; investors await clarity."
            ),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AI: Portfolio Optimizer
# ---------------------------------------------------------------------------

@app.get("/api/v1/ai/portfolio-optimize")
def ai_portfolio_optimize() -> Dict[str, Any]:
    conn = get_db()
    try:
        props = [dict(p) for p in conn.execute("SELECT * FROM properties").fetchall()]
        recs = []

        available = [p for p in props if p["status"] == "available"]
        if available:
            oldest = sorted(available, key=lambda x: x["listed_at"])[0]
            days = (int(datetime.now(tz=timezone.utc).timestamp()) - oldest["listed_at"]) // 86400
            if days > 45:
                recs.append({
                    "type": "price_reduction", "priority": "high",
                    "property_id": oldest["id"], "property": oldest["title"],
                    "insight": f"Listed {days} days with no sale. Consider a 5-8% price reduction or staging upgrade.",
                })

        by_city: Dict[str, int] = {}
        for p in props:
            by_city[p["city"]] = by_city.get(p["city"], 0) + 1
        top_city = max(by_city, key=lambda c: by_city[c])
        if by_city[top_city] > len(props) * 0.4:
            recs.append({
                "type": "diversification", "priority": "medium", "property": None,
                "insight": f"{round(by_city[top_city]/len(props)*100)}% of listings in {top_city}. Geographic diversification reduces concentration risk.",
            })

        commercial = [p for p in props if p["type"] == "commercial" and p["status"] == "available"]
        if commercial:
            recs.append({
                "type": "yield_play", "priority": "medium",
                "property_id": commercial[0]["id"], "property": commercial[0]["title"],
                "insight": "Commercial asset idle. Consider leasing to generate yield while awaiting sale.",
            })

        recs.append({
            "type": "market_timing", "priority": "low", "property": None,
            "insight": "Q1 historically sees 12-18% higher buyer activity. Schedule open houses and digital campaigns now.",
        })

        total_value = sum(p["price"] for p in props if p["status"] == "available")
        return {
            "portfolio_size": len(props),
            "total_available_value": round(total_value, 2),
            "recommendations": recs,
            "by_status": {s: sum(1 for p in props if p["status"] == s) for s in ["available", "sold", "pending", "rented"]},
            "by_type": {t: sum(1 for p in props if p["type"] == t) for t in ["apartment", "villa", "commercial", "land"]},
        }
    finally:
        conn.close()
