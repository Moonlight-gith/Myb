const express = require('express');
const cors = require('cors');
const Database = require('better-sqlite3');
const path = require('path');

const DB_PATH = path.join(__dirname, 'realestate.db');
const app = express();
app.use(cors());
app.use(express.json());

// ─── DB Init ────────────────────────────────────────────────────────────────

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');

db.exec(`
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
    status TEXT NOT NULL DEFAULT 'new',
    budget REAL,
    notes TEXT,
    created_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id INTEGER NOT NULL,
    lead_id INTEGER,
    type TEXT NOT NULL,
    amount REAL NOT NULL,
    commission REAL NOT NULL,
    closed_at INTEGER NOT NULL,
    agent TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS market_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    city TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL
  );
`);

// ─── Seed ────────────────────────────────────────────────────────────────────

function seed() {
  const count = db.prepare('SELECT COUNT(*) as c FROM properties').get().c;
  if (count > 0) return;

  const now = Math.floor(Date.now() / 1000);
  const day = 86400;
  const rand = (n) => Math.floor(Math.random() * n);

  const properties = [
    ["Luxury Penthouse Manhattan","432 Park Ave","New York","apartment","available",4800000,3,3,3200,"https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=400"],
    ["Suburban Family Villa","78 Maple Lane","Los Angeles","villa","sold",1250000,5,4,4500,"https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=400"],
    ["Downtown Studio Loft","215 W 10th St","Chicago","apartment","rented",320000,1,1,680,"https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?w=400"],
    ["Beachfront Bungalow","9 Ocean Drive","Miami","villa","available",2100000,4,3,3100,"https://images.unsplash.com/photo-1493809842364-78817add7ffb?w=400"],
    ["Tech Hub Office Space","1200 Market St","San Francisco","commercial","available",6500000,0,0,8400,"https://images.unsplash.com/photo-1497366216548-37526070297c?w=400"],
    ["Mountain Retreat Cabin","55 Alpine Rd","Denver","villa","pending",890000,3,2,2200,"https://images.unsplash.com/photo-1449844908441-8829872d2607?w=400"],
    ["Urban Micro Apartment","99 Broad St","Boston","apartment","available",480000,1,1,520,"https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=400"],
    ["Industrial Warehouse","300 Commerce Blvd","Dallas","commercial","sold",1900000,0,0,12000,"https://images.unsplash.com/photo-1486325212027-8081e485255e?w=400"],
    ["Golf Course Estate","7 Fairway Dr","Scottsdale","villa","available",3400000,6,5,6800,"https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=400"],
    ["Historic Brownstone","44 Beacon St","Boston","apartment","available",1100000,3,2,2100,"https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400"],
    ["Waterfront Condo","1 Harbor View","Seattle","apartment","pending",750000,2,2,1400,"https://images.unsplash.com/photo-1554995207-c18c203602cb?w=400"],
    ["Desert Land Plot","Lot 12 Sand Dune Rd","Phoenix","land","available",220000,0,0,43560,"https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400"],
  ];

  const insertProp = db.prepare('INSERT INTO properties (title,address,city,type,status,price,bedrooms,bathrooms,area_sqft,listed_at,image_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)');
  for (const p of properties) {
    // p = [title, address, city, type, status, price, bedrooms, bathrooms, area_sqft, image_url]  (9 fields + image = 10)
    insertProp.run(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8], now - rand(90)*day, p[9]);
  }

  const leads = [
    ["James Anderson","james@email.com","+1-555-0101","website",1,"qualified",5000000],
    ["Sarah Chen","sarah.chen@email.com","+1-555-0102","referral",4,"converted",2200000],
    ["Michael Torres","m.torres@email.com","+1-555-0103","social",null,"new",400000],
    ["Emma Williams","emma.w@email.com","+1-555-0104","walk-in",7,"contacted",500000],
    ["David Kim","d.kim@email.com","+1-555-0105","website",5,"qualified",7000000],
    ["Olivia Brown","olivia.b@email.com","+1-555-0106","referral",9,"contacted",3500000],
    ["Liam Johnson","liam.j@email.com","+1-555-0107","social",3,"new",350000],
    ["Ava Martinez","ava.m@email.com","+1-555-0108","website",10,"qualified",1200000],
    ["Noah Davis","noah.d@email.com","+1-555-0109","referral",null,"lost",800000],
    ["Isabella Wilson","i.wilson@email.com","+1-555-0110","walk-in",6,"converted",1000000],
    ["Ethan Taylor","e.taylor@email.com","+1-555-0111","website",2,"new",1300000],
    ["Mia Garcia","mia.g@email.com","+1-555-0112","social",11,"contacted",250000],
  ];
  const insertLead = db.prepare('INSERT INTO leads (name,email,phone,source,property_id,status,budget,created_at) VALUES (?,?,?,?,?,?,?,?)');
  for (const l of leads) insertLead.run(l[0],l[1],l[2],l[3],l[4],l[5],l[6], now - rand(60)*day);

  const txns = [
    [2,2,"sale",1250000,37500,"James Porter"],
    [3,10,"rental",3800,228,"Lisa Monroe"],
    [8,null,"sale",1900000,57000,"James Porter"],
    [6,null,"sale",870000,26100,"Rachel Stone"],
    [4,2,"sale",2050000,61500,"Lisa Monroe"],
  ];
  const insertTxn = db.prepare('INSERT INTO transactions (property_id,lead_id,type,amount,commission,closed_at,agent) VALUES (?,?,?,?,?,?,?)');
  for (const t of txns) insertTxn.run(t[0],t[1],t[2],t[3],t[4], now - rand(30)*day, t[5]);

  const cities = ["New York","Los Angeles","Miami","Chicago","San Francisco"];
  const metrics = ["avg_price_sqft","days_on_market","listing_volume","sold_volume"];
  const baseVals = {
    avg_price_sqft:  {"New York":1850,"Los Angeles":920,"Miami":680,"Chicago":340,"San Francisco":1100},
    days_on_market:  {"New York":28,"Los Angeles":22,"Miami":18,"Chicago":35,"San Francisco":15},
    listing_volume:  {"New York":420,"Los Angeles":380,"Miami":290,"Chicago":310,"San Francisco":210},
    sold_volume:     {"New York":310,"Los Angeles":270,"Miami":220,"Chicago":220,"San Francisco":165},
  };
  const insertMkt = db.prepare('INSERT INTO market_metrics (ts,city,metric,value) VALUES (?,?,?,?)');
  for (const metric of metrics) {
    for (const city of cities) {
      const base = baseVals[metric][city];
      for (let w = 0; w < 12; w++) {
        const ts = now - (11-w)*7*day;
        const noise = (Math.random()-0.5)*0.1;
        const val = base * (1 + noise + w*0.005);
        insertMkt.run(ts, city, metric, Math.round(val*100)/100);
      }
    }
  }

  console.log('Demo data seeded.');
}
seed();

// ─── Helpers ─────────────────────────────────────────────────────────────────

const now = () => Math.floor(Date.now()/1000);
const iso = (ts) => new Date(ts*1000).toISOString();

// ─── Dashboard ───────────────────────────────────────────────────────────────

app.get('/api/v1/dashboard', (req, res) => {
  const total_props = db.prepare('SELECT COUNT(*) as c FROM properties').get().c;
  const available   = db.prepare("SELECT COUNT(*) as c FROM properties WHERE status='available'").get().c;
  const sold        = db.prepare("SELECT COUNT(*) as c FROM properties WHERE status='sold'").get().c;
  const pending     = db.prepare("SELECT COUNT(*) as c FROM properties WHERE status='pending'").get().c;
  const rented      = db.prepare("SELECT COUNT(*) as c FROM properties WHERE status='rented'").get().c;
  const avg_price   = db.prepare('SELECT AVG(price) as v FROM properties').get().v || 0;

  const total_leads    = db.prepare('SELECT COUNT(*) as c FROM leads').get().c;
  const new_leads      = db.prepare("SELECT COUNT(*) as c FROM leads WHERE status='new'").get().c;
  const contacted      = db.prepare("SELECT COUNT(*) as c FROM leads WHERE status='contacted'").get().c;
  const qualified      = db.prepare("SELECT COUNT(*) as c FROM leads WHERE status='qualified'").get().c;
  const converted      = db.prepare("SELECT COUNT(*) as c FROM leads WHERE status='converted'").get().c;
  const lost           = db.prepare("SELECT COUNT(*) as c FROM leads WHERE status='lost'").get().c;

  const rev_row   = db.prepare('SELECT SUM(commission) as v FROM transactions').get();
  const total_rev = rev_row.v || 0;
  const vol_row   = db.prepare("SELECT SUM(amount) as v FROM transactions WHERE type='sale'").get();
  const total_vol = vol_row.v || 0;

  const recent_txns = db.prepare(`
    SELECT t.id,t.type,t.amount,t.commission,t.agent,t.closed_at,p.title,p.city
    FROM transactions t JOIN properties p ON t.property_id=p.id
    ORDER BY t.closed_at DESC LIMIT 5`).all();

  const top_agents = db.prepare(`
    SELECT agent, SUM(amount) as volume, SUM(commission) as earnings, COUNT(*) as deals
    FROM transactions GROUP BY agent ORDER BY volume DESC LIMIT 5`).all();

  res.json({
    properties: { total:total_props, available, sold, pending, rented, avg_price: Math.round(avg_price) },
    leads: {
      total: total_leads, new: new_leads, qualified, converted,
      conversion_rate: total_leads ? +(converted/total_leads*100).toFixed(1) : 0,
      pipeline: { new: new_leads, contacted, qualified, converted, lost },
    },
    revenue: {
      total_commission: Math.round(total_rev),
      total_sales_volume: Math.round(total_vol),
      avg_commission_rate: total_vol ? +(total_rev/total_vol*100).toFixed(2) : 0,
    },
    recent_transactions: recent_txns.map(t => ({
      id:t.id, type:t.type, amount:t.amount, commission:t.commission,
      agent:t.agent, property:t.title, city:t.city, closed_at:iso(t.closed_at)
    })),
    top_agents: top_agents.map(a => ({ agent:a.agent, volume:a.volume, earnings:a.earnings, deals:a.deals })),
  });
});

// ─── Properties ──────────────────────────────────────────────────────────────

app.get('/api/v1/properties', (req, res) => {
  const { status, city, type, min_price, max_price, limit=20 } = req.query;
  let q = 'SELECT * FROM properties WHERE 1=1';
  const p = [];
  if (status)    { q += ' AND status=?';      p.push(status); }
  if (city)      { q += ' AND city LIKE ?';   p.push(`%${city}%`); }
  if (type)      { q += ' AND type=?';        p.push(type); }
  if (min_price) { q += ' AND price>=?';      p.push(+min_price); }
  if (max_price) { q += ' AND price<=?';      p.push(+max_price); }
  q += ' ORDER BY listed_at DESC LIMIT ?';    p.push(+limit);
  const rows = db.prepare(q).all(...p);
  res.json({ properties: rows, total: rows.length });
});

app.post('/api/v1/properties', (req, res) => {
  const { title,address,city,type,status,price,bedrooms=0,bathrooms=0,area_sqft,image_url=null } = req.body;
  const r = db.prepare('INSERT INTO properties (title,address,city,type,status,price,bedrooms,bathrooms,area_sqft,listed_at,image_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)')
    .run(title,address,city,type,status,+price,+bedrooms,+bathrooms,+area_sqft,now(),image_url);
  res.json({ id:r.lastInsertRowid, ...req.body });
});

// ─── Leads ───────────────────────────────────────────────────────────────────

app.get('/api/v1/leads', (req, res) => {
  const { status, source, limit=20 } = req.query;
  let q = 'SELECT * FROM leads WHERE 1=1';
  const p = [];
  if (status) { q += ' AND status=?'; p.push(status); }
  if (source) { q += ' AND source=?'; p.push(source); }
  q += ' ORDER BY created_at DESC LIMIT ?'; p.push(+limit);
  const rows = db.prepare(q).all(...p);
  res.json({ leads: rows, total: rows.length });
});

app.post('/api/v1/leads', (req, res) => {
  const { name,email,phone=null,source,property_id=null,status='new',budget=null,notes=null } = req.body;
  const r = db.prepare('INSERT INTO leads (name,email,phone,source,property_id,status,budget,notes,created_at) VALUES (?,?,?,?,?,?,?,?,?)')
    .run(name,email,phone,source,property_id,status,budget,notes,now());
  res.json({ id:r.lastInsertRowid, ...req.body });
});

// ─── Market ──────────────────────────────────────────────────────────────────

app.get('/api/v1/market', (req, res) => {
  const { city, metric } = req.query;
  let q = 'SELECT city,metric,ts,value FROM market_metrics WHERE 1=1';
  const p = [];
  if (city)   { q += ' AND city=?';   p.push(city); }
  if (metric) { q += ' AND metric=?'; p.push(metric); }
  q += ' ORDER BY city,metric,ts ASC';
  const rows = db.prepare(q).all(...p);
  const grouped = {};
  for (const r of rows) {
    grouped[r.city] = grouped[r.city] || {};
    grouped[r.city][r.metric] = grouped[r.city][r.metric] || [];
    grouped[r.city][r.metric].push({ ts:iso(r.ts), value:r.value });
  }
  res.json({ market: grouped });
});

// ─── AI Valuation ─────────────────────────────────────────────────────────────

app.post('/api/v1/ai/valuate', (req, res) => {
  const { city, type, area_sqft, bedrooms=0, bathrooms=0, condition='good' } = req.body;
  const ppsf_map = {
    'new york':1850,'los angeles':920,'miami':680,'chicago':340,'san francisco':1100,
    'boston':810,'seattle':740,'dallas':280,'denver':420,'phoenix':260,'scottsdale':480
  };
  const ppsf = ppsf_map[city.toLowerCase()] || 450;
  const type_mult = { villa:1.15, apartment:1.0, commercial:1.2, land:0.4 }[type] || 1.0;
  const cond_mult = { poor:0.75, fair:0.88, good:1.0, excellent:1.15 }[condition] || 1.0;
  const bed_prem  = Math.min(bedrooms*0.04, 0.20);
  const bath_prem = Math.min(bathrooms*0.02, 0.10);
  const est = area_sqft * ppsf * type_mult * cond_mult * (1 + bed_prem + bath_prem);

  res.json({
    estimated_value: Math.round(est),
    range_low: Math.round(est*0.88),
    range_high: Math.round(est*1.12),
    confidence: ppsf_map[city.toLowerCase()] ? 0.82 : 0.65,
    price_per_sqft: +(ppsf*type_mult*cond_mult).toFixed(2),
    factors: [
      { factor:'Location premium',  impact:`${((ppsf/450-1)*100).toFixed(1).replace(/^(?!-)/, '+')}%` },
      { factor:'Property type',     impact:`${((type_mult-1)*100).toFixed(1).replace(/^(?!-)/, '+')}%` },
      { factor:'Condition',         impact:`${((cond_mult-1)*100).toFixed(1).replace(/^(?!-)/, '+')}%` },
      { factor:'Bedroom premium',   impact:`+${(bed_prem*100).toFixed(1)}%` },
      { factor:'Bathroom premium',  impact:`+${(bath_prem*100).toFixed(1)}%` },
    ],
    recommendation: est < ppsf*area_sqft*0.9
      ? 'Strong buy — below market median'
      : 'Fairly priced relative to local comparables',
  });
});

// ─── AI Lead Score ────────────────────────────────────────────────────────────

app.get('/api/v1/ai/lead-score/:id', (req, res) => {
  const lead = db.prepare('SELECT * FROM leads WHERE id=?').get(req.params.id);
  if (!lead) return res.status(404).json({ detail:'Lead not found' });

  let score = 50;
  score += { new:0, contacted:10, qualified:25, converted:40, lost:-20 }[lead.status] || 0;
  score += { referral:15, website:8, 'walk-in':12, social:5 }[lead.source] || 0;
  if (lead.phone) score += 8;
  if (lead.property_id) score += 10;
  if (lead.budget && lead.budget > 500000) score += 7;
  score = Math.max(0, Math.min(100, score));

  const tier = score >= 75 ? 'Hot' : score >= 45 ? 'Warm' : 'Cold';
  const actions = [];
  if (lead.status === 'new')       actions.push('Schedule introductory call within 24h');
  if (lead.status === 'contacted') actions.push('Send property matches based on budget');
  if (lead.status === 'qualified') actions.push('Book property viewings this week');
  if (!lead.phone)                 actions.push('Collect phone number to improve reachability');

  res.json({ lead_id:+req.params.id, name:lead.name, score, tier, next_actions:actions });
});

// ─── AI Market Insights ───────────────────────────────────────────────────────

app.post('/api/v1/ai/market-insights', (req, res) => {
  const { city } = req.body;
  const rows = db.prepare('SELECT metric,value FROM market_metrics WHERE city=? ORDER BY ts DESC').all(city);

  const latest = {};
  const seen = new Set();
  for (const r of rows) {
    if (!seen.has(r.metric)) { latest[r.metric] = r.value; seen.add(r.metric); }
  }

  let score = 70;
  const insights = [];
  const dom  = latest['days_on_market'] || 30;
  const ppsf = latest['avg_price_sqft'] || 500;
  const lv   = latest['listing_volume'] || 300;
  const sv   = latest['sold_volume']    || 200;

  if (dom < 20)       { insights.push({type:'bullish',    text:`Properties selling fast — avg ${Math.round(dom)} days signals high demand.`}); score+=10; }
  else if (dom > 40)  { insights.push({type:'bearish',    text:`Slow market — avg ${Math.round(dom)} days on market. Buyers have leverage.`}); score-=10; }
  else                { insights.push({type:'neutral',    text:`Balanced market — avg ${Math.round(dom)} days on market.`}); }

  if (ppsf > 1000)    { insights.push({type:'bullish',    text:`Premium market at $${Math.round(ppsf).toLocaleString()}/sqft — strong appreciation potential.`}); score+=8; }
  else if (ppsf < 400){ insights.push({type:'opportunity',text:`Affordable entry at $${Math.round(ppsf)}/sqft — value play for investors.`}); score+=5; }

  if (lv > 0) {
    const absorption = sv/lv;
    if (absorption > 0.75)      { insights.push({type:'bullish',text:`High absorption rate (${Math.round(absorption*100)}%) — inventory tightening.`}); score+=8; }
    else if (absorption < 0.5)  { insights.push({type:'bearish',text:`Low absorption rate (${Math.round(absorption*100)}%) — oversupply risk.`}); score-=8; }
  }

  score = Math.max(0, Math.min(100, score));
  const health = score >= 85 ? 'Excellent' : score >= 70 ? 'Good' : score >= 55 ? 'Fair' : 'Weak';

  res.json({
    city, market_health_score:score, market_health:health, insights,
    key_metrics: {
      avg_price_per_sqft: Math.round(ppsf*100)/100,
      avg_days_on_market: Math.round(dom*10)/10,
      listing_volume: Math.round(lv),
      sold_volume: Math.round(sv),
      absorption_rate: lv ? Math.round(sv/lv*1000)/10 : 0,
    },
    recommendation: ['Excellent','Good'].includes(health)
      ? 'Strong market — ideal for sellers and long-term investors.'
      : 'Cautious outlook — buyers should negotiate; investors await clarity.',
  });
});

// ─── AI Portfolio Optimizer ───────────────────────────────────────────────────

app.get('/api/v1/ai/portfolio-optimize', (req, res) => {
  const props = db.prepare('SELECT * FROM properties').all();
  const recs = [];
  const n = now();

  const available = props.filter(p => p.status === 'available');
  if (available.length) {
    const oldest = available.sort((a,b) => a.listed_at-b.listed_at)[0];
    const days = Math.floor((n - oldest.listed_at) / 86400);
    if (days > 45) {
      recs.push({ type:'price_reduction', priority:'high', property_id:oldest.id, property:oldest.title,
        insight:`Listed ${days} days with no sale. Consider a 5-8% price reduction or staging upgrade.` });
    }
  }

  const byCityCount = {};
  for (const p of props) byCityCount[p.city] = (byCityCount[p.city]||0)+1;
  const topCity = Object.entries(byCityCount).sort((a,b)=>b[1]-a[1])[0];
  if (topCity && topCity[1] > props.length*0.4) {
    recs.push({ type:'diversification', priority:'medium', property:null,
      insight:`${Math.round(topCity[1]/props.length*100)}% of listings in ${topCity[0]}. Geographic diversification reduces concentration risk.` });
  }

  const commercial = props.filter(p => p.type === 'commercial' && p.status === 'available');
  if (commercial.length) {
    recs.push({ type:'yield_play', priority:'medium', property_id:commercial[0].id, property:commercial[0].title,
      insight:'Commercial asset idle. Consider leasing to generate yield while awaiting sale.' });
  }

  recs.push({ type:'market_timing', priority:'low', property:null,
    insight:'Q1 historically sees 12-18% higher buyer activity. Schedule open houses and digital campaigns now.' });

  const total_value = available.reduce((sum,p)=>sum+p.price, 0);
  const by_status = {};
  const by_type = {};
  for (const p of props) {
    by_status[p.status] = (by_status[p.status]||0)+1;
    by_type[p.type]     = (by_type[p.type]||0)+1;
  }

  res.json({ portfolio_size:props.length, total_available_value:total_value, recommendations:recs, by_status, by_type });
});

// ─── Start ────────────────────────────────────────────────────────────────────

const PORT = 8000;
app.listen(PORT, () => console.log(`EstateAI API running on http://localhost:${PORT}`));
