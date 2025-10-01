# ===================== main.py (debug build) =====================
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import pandas as pd
import json, os, re, datetime
import httpx
from fastapi.middleware.cors import CORSMiddleware
from pyairtable import Api

# --- Existing long explainers (unchanged) ---
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story

# --- New conversational explainer (human-advisor tone) ---
from recommendation_explainer import (
    build_recommendation_parts,
    parse_portfolio as rx_parse_portfolio,
)

__API_BUILD__ = "debug-story-write-2025-09-30"
app = FastAPI(title="SIPY Investment Engine API (debug)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------- Env & Airtable ----------
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE   = os.getenv("AIRTABLE_TABLE", "Investor_Inputs")

if not (AIRTABLE_API_KEY and AIRTABLE_BASE_ID and AIRTABLE_TABLE):
    print("⚠️  Missing Airtable env vars; /trigger_processing will fail until they are set.")

api = Api(AIRTABLE_API_KEY) if AIRTABLE_API_KEY else None
airtable = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE) if api else None

def _hdr():
    return {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

async def _airtable_get_record(record_id: str):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}/{record_id}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_hdr())
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ---------------- Helpers: parsing & math ----------------
# --- PATCH START: make years parser accept floats like 0.25; keep default as float ---
def _to_years_from_any(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else default
# --- PATCH END ---

def _to_float(x, default=0.0):
    if x is None: return default
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",", "").replace("₹", "").strip()
    try:
        return float(s)
    except:
        m = re.findall(r"[-+]?\d*\.?\d+", s)
        return float(m[0]) if m else default

def expected_return_from_profile(years_to_goal: int, risk_profile: str) -> float:
    rp = (risk_profile or "").strip().lower()
    if years_to_goal <= 3:
        base = {"conservative": 0.06, "moderate": 0.075, "aggressive": 0.085}
    elif years_to_goal <= 5:
        base = {"conservative": 0.07, "moderate": 0.09, "aggressive": 0.105}
    elif years_to_goal <= 10:
        base = {"conservative": 0.08, "moderate": 0.10, "aggressive": 0.12}
    else:
        base = {"conservative": 0.09, "moderate": 0.11, "aggressive": 0.13}
    return base.get(rp, 0.10)

# ------------- Load CSVs once -------------
def _load_csvs():
    active_eq  = pd.read_csv("data/active_equity_portfolio.csv")
    passive_eq = pd.read_csv("data/passive_equity_portfolio.csv")
    hybrid_eq  = pd.read_csv("data/hybrid_equity_portfolio.csv")
    tmf        = pd.read_csv("data/tmf_selected.csv")
    duration_d = pd.read_csv("data/debt_duration_selected.csv")
    print("✅ CSVs loaded.")
    return active_eq, passive_eq, hybrid_eq, tmf, duration_d

active_eq, passive_eq, hybrid_eq, tmf, duration_debt = _load_csvs()

# ------------- Pydantic -------------
# --- PATCH START: accept fractional horizons from Airtable (e.g., 0.25 = 3 months) ---
class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: float   # was int
    risk_profile: str
    current_corpus: Optional[float] = 0.0
# --- PATCH END ---

# ------------- Core engine -------------

# --- PATCH START: precise fractional-years funding ratio + min 1 month guard ---
def calculate_funding_ratio(monthly_investment, current_corpus, target_corpus, years, annual_expected_return):
    years = max(0.0, float(years))   # allow fractional years
    r_m = (1 + annual_expected_return) ** (1/12) - 1
    n = max(1, int(round(years * 12)))   # months, min 1 to avoid downstream zero-length issues
    fv_lumpsum = (current_corpus or 0.0) * ((1 + annual_expected_return) ** years)
    fv_sip = monthly_investment * (((1 + r_m) ** n - 1) / r_m) if r_m != 0 else monthly_investment * n
    fv_total = fv_lumpsum + fv_sip
    return fv_total, (fv_total / target_corpus) if target_corpus else 0.0
# --- PATCH END ---

def choose_strategy(time_to_goal, risk_profile, funding_ratio):
    if time_to_goal <= 3: return "Passive"
    if funding_ratio >= 1.3: funding_status = 'overfunded'
    elif funding_ratio < 0.7: funding_status = 'underfunded'
    else: funding_status = 'balanced'
    rp = (risk_profile or "").strip().lower()
    if time_to_goal <= 5:
        if funding_status == 'overfunded': return "Passive"
        elif rp == 'aggressive' and funding_status == 'underfunded': return "Hybrid"
        else: return "Passive"
    elif 6 <= time_to_goal <= 10:
        if rp == 'conservative': return "Passive"
        elif rp == 'moderate': return "Hybrid" if funding_status == 'balanced' else "Passive"
        elif rp == 'aggressive': return "Active" if funding_status == 'underfunded' else "Hybrid"
    else:
        if rp == 'conservative': return "Passive"
        elif funding_status == 'underfunded': return "Active"
        elif funding_status == 'overfunded': return "Hybrid" if rp != 'conservative' else "Passive"
        else: return "Active" if rp == 'aggressive' else "Hybrid"

def _cap_bucket_from_name(name: str) -> str:
    n = (name or "").lower()
    if re.search(r"(nifty\s*50|sensex|large\s*cap|blue\s*chip|bluechip|top\s*100|nifty\s*100|bse\s*100)", n): return "large"
    if re.search(r"(mid\s*cap|midcap|nifty\s*midcap)", n): return "mid"
    if re.search(r"(small\s*cap|smallcap|nifty\s*smallcap)", n): return "small"
    return "other"

def standardize_fund_col(df):
    if 'Fund' in df.columns: return df
    cand = [c for c in df.columns if any(k in c.lower() for k in ['fund', 'scheme', 'name'])]
    return df.rename(columns={cand[0]: 'Fund'}) if cand else df.assign(Fund=df.index.astype(str))

def filter_equity_by_horizon(eq_df: pd.DataFrame, years_to_goal: int) -> pd.DataFrame:
    if years_to_goal > 3: return eq_df
    df = standardize_fund_col(eq_df.copy())
    df['__cap__'] = df['Fund'].map(_cap_bucket_from_name)
    large = df[df['__cap__'] == 'large'].copy()
    return (large.drop(columns='__cap__') if not large.empty else df.drop(columns='__cap__'))

def enforce_min_allocation(df, total_pct, min_pct=10, step=5):
    df = df.copy()
    df['Weight (%)'] = df['Weight (%)'].apply(lambda x: round(x / step) * step)
    df['Weight (%)'] = df['Weight (%)'].apply(lambda x: max(min_pct, x))
    s = df['Weight (%)'].sum()
    if s == 0: raise ValueError("Zero weights after min-enforcement.")
    df['Weight (%)'] = df['Weight (%)'] / s * total_pct
    df['Weight (%)'] = df['Weight (%)'].apply(lambda x: round(x / step) * step)
    diff = total_pct - df['Weight (%)'].sum()
    if diff != 0:
        idx = df['Weight (%)'].idxmax()
        df.at[idx, 'Weight (%)'] += diff
    return df

def trim_funds_to_min(df, total_pct, min_pct=10, sort_col=None):
    max_funds = max(1, int(total_pct // min_pct))
    if len(df) > max_funds:
        if sort_col and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False).head(max_funds)
        else:
            df = df.head(max_funds)
    return df.copy()

def get_debt_allocation(duration_debt_df, time_to_goal, debt_pct):
    allowed = ['Liquid'] if time_to_goal > 3 else ['Liquid', 'Ultra Short']
    df = duration_debt_df[duration_debt_df['Sub-category'].str.strip().isin(allowed)].copy()
    if df.empty: raise ValueError("No suitable debt funds found for the selected duration.")
    df = standardize_fund_col(df)
    df['Category'] = 'Debt'
    df = trim_funds_to_min(df, debt_pct, min_pct=10, sort_col='Weight' if 'Weight' in df.columns else None)
    df['Weight (%)'] = debt_pct / len(df)
    return df[['Fund', 'Category', 'Sub-category', 'Weight (%)']]

# --- PATCH START: make glide path robust for sub-1-year inputs (never empty) ---
def generate_step_down_glide_path(time_to_goal, funding_ratio, risk_profile):
    # Coerce to at least 1 year and int for loop safety
    time_to_goal = max(1, int(round(float(time_to_goal))))

    funding_ratio = float(funding_ratio)
    glide_path = []

    short_cap = 30 if time_to_goal <= 3 else 100
    if funding_ratio > 2.0: base_equity, start = 70, int(time_to_goal * 0.4)
    elif funding_ratio > 1.0: base_equity, start = 80, int(time_to_goal * 0.5)
    else: base_equity, start = 90, int(time_to_goal * 0.6)

    rp = (risk_profile or "").strip().lower()
    if rp == 'conservative': base_equity -= 10
    elif rp == 'aggressive': base_equity += 5
    base_equity = min(base_equity, short_cap)

    final_equity = 10
    derisk_years = max(1, time_to_goal - start)
    step = (base_equity - final_equity) / derisk_years if derisk_years > 0 else 0

    for y in range(1, time_to_goal + 1):
        e = base_equity if y <= start else base_equity - step * (y - start)
        e = int(round(max(min(e, short_cap), 0) / 5) * 5); e = 10 if 0 < e < 10 else e
        d = 100 - e; d = 10 if 0 < d < 10 else d; e = 100 - d
        glide_path.append({'Year': y, 'Equity Allocation (%)': int(e), 'Debt Allocation (%)': int(d)})
    return pd.DataFrame(glide_path)
# --- PATCH END ---

# ---------------- API: generate ----------------
@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    try:
        # --- PATCH START: normalize years for functions that expect ints; keep fractional for funding ratio ---
        safe_years_int = max(1, int(round(float(user_input.years_to_goal))))
        # --- PATCH END ---

        # 1) math & strategy
        annual_er = expected_return_from_profile(safe_years_int, user_input.risk_profile)
        _, funding_ratio = calculate_funding_ratio(
            monthly_investment=user_input.monthly_investment,
            current_corpus=user_input.current_corpus or 0.0,
            target_corpus=user_input.target_corpus,
            years=user_input.years_to_goal,               # keep fractional precision here
            annual_expected_return=annual_er
        )
        strategy = choose_strategy(safe_years_int, user_input.risk_profile, funding_ratio)
        glide_path = generate_step_down_glide_path(user_input.years_to_goal, funding_ratio, user_input.risk_profile)

        # --- PATCH START: ensure glide_path is never empty before iloc[0] ---
        if isinstance(glide_path, pd.DataFrame) and glide_path.empty:
            glide_path = pd.DataFrame([{'Year': 1, 'Equity Allocation (%)': 10, 'Debt Allocation (%)': 90}])
        # --- PATCH END ---

        y1_equity = int(glide_path.iloc[0]['Equity Allocation (%)'])
        y1_debt   = 100 - y1_equity

        # 2) equity pool by strategy
        if strategy == "Active":
            eq_df = active_eq.copy(); eq_df['Sub-category'] = 'Active'
        elif strategy == "Passive":
            eq_df = passive_eq.copy(); eq_df['Sub-category'] = 'Passive'
        elif strategy == "Hybrid":
            eq_df = hybrid_eq.copy(); eq_df = eq_df.rename(columns={'Type': 'Sub-category'})
        else:
            raise ValueError("Invalid strategy")

        eq_df = standardize_fund_col(eq_df)
        eq_df = filter_equity_by_horizon(eq_df, safe_years_int)  # use normalized int horizon
        eq_df['Category'] = 'Equity'
        if 'Weight' not in eq_df.columns: eq_df['Weight'] = 1.0

        eq_df = trim_funds_to_min(eq_df, y1_equity, min_pct=10, sort_col='Weight')
        eq_df['Weight'] = eq_df['Weight'] / eq_df['Weight'].sum()
        eq_df['Weight (%)'] = eq_df['Weight'] * y1_equity
        eq_df = enforce_min_allocation(eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']],
                                       y1_equity, min_pct=10, step=5)

        # 3) debt
        debt_df = get_debt_allocation(duration_debt, safe_years_int, y1_debt)  # use normalized int horizon
        debt_df = enforce_min_allocation(debt_df, y1_debt, min_pct=10, step=5)

        # 4) combine → final (keep Sub-category here)
        combined = pd.concat([eq_df, debt_df], ignore_index=True)
        final_portfolio = enforce_min_allocation(combined, 100, min_pct=10, step=5)

        # 5) build 4-column display copy
        display_portfolio = final_portfolio.rename(columns={"Sub-category": "Type"})[
            ["Fund", "Category", "Type", "Weight (%)"]
        ]

        # 6) long explainers (legacy)
        glide_block = explain_glide_story({
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(funding_ratio),
            "glide_path": glide_path.to_dict(orient="records"),
        })
        strategy_block = explain_strategy_story({
            "strategy": strategy,
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(funding_ratio),
        })
        # IMPORTANT: use final_portfolio (has Sub-category) to avoid 500
        portfolio_block = explain_portfolio_story(final_portfolio)

        return {
            "strategy": strategy,
            "funding_ratio": round(float(funding_ratio), 4),
            "glide_path": glide_path.to_dict(orient="records"),
            "portfolio": display_portfolio.to_dict(orient="records"),  # 4 columns
            "glide_explainer": glide_block,
            "strategy_explainer": strategy_block,
            "portfolio_explainer": portfolio_block,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Airtable webhook ----------------
class AirtableWebhookPayload(BaseModel):
    record_id: str

@app.post("/trigger_processing/")
async def trigger_processing(payload: AirtableWebhookPayload):
    if airtable is None:
        raise HTTPException(status_code=500, detail="Airtable not configured.")

    try:
        record_id = payload.record_id
        record = airtable.get(record_id)
        fields = record.get("fields", {})

        user_input_data = {
            "monthly_investment": _to_float(fields.get("Monthly Investments"), 0.0),
            "target_corpus": _to_float(fields.get("Target Corpus"), 0.0),
            # --- PATCH START: accept fractional "Time to goal" values from Airtable ---
            "years_to_goal": _to_years_from_any(fields.get("Time to goal", fields.get("Time Horizon (years)")), 0.0),
            # --- PATCH END ---
            "risk_profile": fields.get("Risk Preference") or "",
            "current_corpus": _to_float(fields.get("Current Corpus"), 0.0),
        }

        # Run the engine locally (no network roundtrip)
        processed = generate_portfolio(PortfolioInput(**user_input_data))

        # Format glide path for Airtable (UNCHANGED)
        formatted_glide = ", ".join([
            f"{{{row['Year']}, {row['Equity Allocation (%)']}, {row['Debt Allocation (%)']}}}"
            for row in processed["glide_path"]
        ])

        # ---------- Build ctx for new human-advisor story ----------
        try:
            fr_center_pct = float(processed.get("funding_ratio", 0.0)) * 100.0
        except:
            fr_center_pct = 0.0

        try:
            y1_equity = int(processed["glide_path"][0]["Equity Allocation (%)"])
        except Exception:
            y1_equity = 0

        # Parse portfolio buckets (tolerant to Type/Sub-category via builder)
        display_port = processed.get("portfolio") or []
        port_agg = rx_parse_portfolio(display_port) or {}
        # Guard against None → 0
        port_agg = {k: (v or 0) for k, v in port_agg.items()}

 # --- PATCH: add horizon so explainer can switch to short-term mode ---
years_horizon = _to_years_from_any(
    fields.get("Time to goal", fields.get("Time Horizon (years)")), 
    0.0
)
months_horizon = (
    max(1, int(round(float(years_horizon) * 12)))
    if years_horizon is not None 
    else None
)

ctx = {
    "name": fields.get("User Name") or fields.get("Name") or "Investor",
    "target_corpus": fields.get("Target Corpus"),
    "funding_ratio_pct": fr_center_pct,
    "strategy": processed.get("strategy"),
    "risk_profile": fields.get("Risk Preference"),
    "equity_start_pct": y1_equity,
    "large_cap_pct": port_agg.get("large_cap_pct"),
    "mid_cap_pct": port_agg.get("mid_cap_pct"),
    "small_cap_pct": port_agg.get("small_cap_pct"),
    "debt_pct": port_agg.get("debt_pct"),
    # ✅ NEW: enable scenario-aware storytelling
    "years_to_goal": years_horizon,
    "months_to_goal": months_horizon,
}


        # **Debug: print ctx**
        print("[ctx]", ctx)

        parts = build_recommendation_parts(ctx)

        # **Debug: preview explainers**
        print("[explainer previews]")
        for k in ("var1", "var2", "var3", "var4"):
            txt = (parts.get(k) or "").replace("\n", " ")
            print(f"{k}: {txt[:140]}...")

        # Funding ratio (ratio form)
        try:
            fr_ratio = float(processed.get("funding_ratio", 0.0))
        except:
            fr_ratio = 0.0

        update_data = {
            "strategy": processed["strategy"],
            "funding_ratio": round(fr_ratio, 2),
            "glide_path": formatted_glide,
            "portfolio": json.dumps(processed["portfolio"]),  # 4-column JSON
            "glide_explainer_story": processed["glide_explainer"]["story"],
            "strategy_explainer_story": processed["strategy_explainer"]["story"],
            "portfolio_explainer_story": processed["portfolio_explainer"]["story"],
            # NEW conversational blocks
            "explainer 1": parts["var1"],
            "explainer 2": parts["var2"],
            "explainer 3": parts["var3"],
            "explainer 4": parts["var4"],
        }

        # **Debug: show which fields we are writing**
        print("[airtable update fields]", list(update_data.keys()))

        airtable.update(record_id, fields=update_data)
        return {"status": "success", "record_id": record_id, "build": __API_BUILD__}

    except Exception as e:
        # surface more detail for debugging
        raise HTTPException(status_code=500, detail=f"Failed to process record {payload.record_id}: {e}")

@app.get("/portfolio")
def get_portfolio(record_id: str = Query(..., alias="record_id")):
    try:
        record = airtable.get(record_id)
        fields = record.get("fields", {})
        return {
            "record_id": record_id,
            "strategy": fields.get("strategy"),
            "funding_ratio": fields.get("funding_ratio"),
            "glide_path": fields.get("glide_path"),
            "explainer_1": fields.get("explainer 1"),
            "explainer_2": fields.get("explainer 2"),
            "explainer_3": fields.get("explainer 3"),
            "explainer_4": fields.get("explainer 4"),
            "portfolio": json.loads(fields.get("portfolio") or "[]"),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Record not found: {e}")


# ---------------- Debug endpoint (no Airtable) ----------------
@app.post("/_debug/story")
def debug_story(
    monthly_investment: float = 15000,
    target_corpus: float = 1e7,
    years_to_goal: int = 12,
    risk_profile: str = "moderate",
    current_corpus: float = 0.0
):
    annual_er = expected_return_from_profile(years_to_goal, risk_profile)
    _, fr = calculate_funding_ratio(monthly_investment, current_corpus, target_corpus, years_to_goal, annual_er)
    glide = generate_step_down_glide_path(years_to_goal, fr, risk_profile)
    y1_equity = int(glide.iloc[0]['Equity Allocation (%)'])

    demo_port = [
        {"Fund":"UTI Nifty 50 Index", "Category":"Equity", "Type":"Passive", "Weight (%)":40},
        {"Fund":"Motilal Oswal Midcap 150 Index", "Category":"Equity", "Type":"Passive", "Weight (%)":30},
        {"Fund":"Bandhan Small Cap Fund", "Category":"Equity", "Type":"Active", "Weight (%)":10},
        {"Fund":"ICICI Pru Liquid", "Category":"Debt", "Type":"Liquid", "Weight (%)":20},
    ]
    port_agg = rx_parse_portfolio(demo_port) or {}
    port_agg = {k: (v or 0) for k, v in port_agg.items()}

    ctx = {
        "name": "Investor",
        "target_corpus": target_corpus,
        "funding_ratio_pct": fr * 100.0,
        "strategy": choose_strategy(years_to_goal, risk_profile, fr),
        "risk_profile": risk_profile,
        "equity_start_pct": y1_equity,
        **port_agg,
    }
    parts = build_recommendation_parts(ctx)
    return {"ctx": ctx, **parts}

# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"ok": True, "build": __API_BUILD__, "ts": datetime.datetime.utcnow().isoformat() + "Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
