# ======================= main.py =======================
# (CHANGES: added HTTPException, Request, CORS, and the /explain/glide endpoint)

from fastapi import FastAPI, HTTPException, Request   # === NEW: added HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware    # === NEW: CORS (optional, safe to keep)
from typing import Any, Dict, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

# === NEW: import the explainer (same folder as main.py)
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
# === NEW: import the explainer (same folder as main.py)
from portfolio_explainer import explain_portfolio_story

app = FastAPI()

# === NEW (optional): CORS if your frontend is on a different domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domains if you prefer
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load data once at app startup ===
def _load_csvs():
    try:
        # You said the latest CSVs are in data/
        active_eq = pd.read_csv("data/active_equity_portfolio.csv")
        passive_eq = pd.read_csv("data/passive_equity_portfolio.csv")
        hybrid_eq = pd.read_csv("data/hybrid_equity_portfolio.csv")
        tmf = pd.read_csv("data/tmf_selected.csv")
        duration_debt = pd.read_csv("data/debt_duration_selected.csv")
        print("✅ All CSVs loaded successfully.")
        return active_eq, passive_eq, hybrid_eq, tmf, duration_debt
    except Exception as e:
        print("❌ Failed to load CSVs:", e)
        raise

active_eq, passive_eq, hybrid_eq, tmf, duration_debt = _load_csvs()

# === Input Schema ===
class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str  # 'conservative', 'moderate', 'aggressive'

# ----------------- Core Utility Functions -----------------
def calculate_funding_ratio(monthly_investment, target_corpus, years):
    expected_return = 0.13  # fixed expected return (annual, 13%)
    r = (1 + expected_return) ** (1/12) - 1  # monthly return
    n = years * 12
    fv = monthly_investment * (((1 + r) ** n - 1) / r)
    return fv, fv / target_corpus

def generate_step_down_glide_path(time_to_goal, funding_ratio, risk_profile):
    """
    Same logic as before, BUT we floor any non-zero bucket to >=10%
    so later per-fund min-10% is always satisfiable after trimming.
    """
    funding_ratio = float(funding_ratio)
    glide_path = []

    short_goal_equity_cap = 30 if time_to_goal <= 3 else 100

    if funding_ratio > 2.0:
        base_equity = 70
        derisk_start = int(time_to_goal * 0.4)
    elif funding_ratio > 1.0:
        base_equity = 80
        derisk_start = int(time_to_goal * 0.5)
    else:
        base_equity = 90
        derisk_start = int(time_to_goal * 0.6)

    if risk_profile == 'conservative':
        base_equity -= 10
    elif risk_profile == 'aggressive':
        base_equity += 5

    base_equity = min(base_equity, short_goal_equity_cap)
    final_equity = 10
    derisk_years = time_to_goal - derisk_start
    equity_step = (base_equity - final_equity) / derisk_years if derisk_years > 0 else 0

    for year in range(1, time_to_goal + 1):
        if year <= derisk_start:
            equity = base_equity
        else:
            equity = base_equity - equity_step * (year - derisk_start)

        equity = int(round(max(min(equity, short_goal_equity_cap), 0) / 5) * 5)

        # >>> bucket floors so we never have a non-zero bucket < 10%
        if equity > 0 and equity < 10:
            equity = 10

        debt = 100 - equity
        if debt > 0 and debt < 10:
            debt = 10
            equity = 100 - debt

        glide_path.append({'Year': year, 'Equity Allocation (%)': int(equity), 'Debt Allocation (%)': int(debt)})

    return pd.DataFrame(glide_path)

def choose_strategy(time_to_goal, risk_profile, funding_ratio):

 # normalize risk_profile to lowercase for consistent checks

    risk_profile = (risk_profile or "").lower()

    if time_to_goal <= 3:
        return "Passive"
    if funding_ratio >= 1.3:
        funding_status = 'overfunded'
    elif funding_ratio < 0.7:
        funding_status = 'underfunded'
    else:
        funding_status = 'balanced'

    if time_to_goal <= 5:
        if funding_status == 'overfunded':
            return "Passive"
        elif risk_profile == 'aggressive' and funding_status == 'underfunded':
            return "Hybrid"
        else:
            return "Passive"
    elif 6 <= time_to_goal <= 10:
        if risk_profile == 'conservative':
            return "Passive"
        elif risk_profile == 'moderate':
            return "Hybrid" if funding_status == 'balanced' else "Passive"
        elif risk_profile == 'aggressive':
            return "Active" if funding_status == 'underfunded' else "Hybrid"
    else:
        if risk_profile == 'conservative':
            return "Passive"
        elif funding_status == 'underfunded':
            return "Active"
        elif funding_status == 'overfunded':
            return "Hybrid" if risk_profile != 'conservative' else "Passive"
        else:
            return "Active" if risk_profile == 'aggressive' else "Hybrid"

# ---------- Min-10% helpers: trim + enforce ----------
def trim_funds_to_min(df, total_pct, min_pct=10, sort_col=None):
    """
    Reduce number of funds so that len(df) * min_pct <= total_pct.
    Keeps the best 'sort_col' funds if provided; otherwise current order.
    """
    max_funds = max(1, int(total_pct // min_pct))
    if len(df) > max_funds:
        if sort_col and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False).head(max_funds)
        else:
            df = df.head(max_funds)
    return df.copy()

def enforce_min_allocation(df, total_pct, min_pct=10, step=5):
    """
    Enforce:
      - rounding to 'step' (default 5%)
      - minimum of 'min_pct' per row
      - re-normalize to total_pct
      - final rounding + mismatch fix
    """
    df = df.copy()
    df['Weight (%)'] = df['Weight (%)'].apply(lambda x: round(x / step) * step)
    df['Weight (%)'] = df['Weight (%)'].apply(lambda x: max(min_pct, x))
    s = df['Weight (%)'].sum()
    if s == 0:
        raise ValueError("Zero weights after min-enforcement.")
    df['Weight (%)'] = df['Weight (%)'] / s * total_pct
    df['Weight (%)'] = df['Weight (%)'].apply(lambda x: round(x / step) * step)
    diff = total_pct - df['Weight (%)'].sum()
    if diff != 0:
        idx = df['Weight (%)'].idxmax()
        df.at[idx, 'Weight (%)'] += diff
    return df

def standardize_fund_col(df):
    if 'Fund' in df.columns:
        return df
    cand = [c for c in df.columns if 'fund' in c.lower() or 'scheme' in c.lower() or 'name' in c.lower()]
    if cand:
        return df.rename(columns={cand[0]: 'Fund'})
    # fallback
    df = df.copy()
    df['Fund'] = df.index.astype(str)
    return df

def get_debt_allocation(duration_debt_df, time_to_goal, debt_pct):
    duration_debt_df = duration_debt_df.copy()

    # If longer horizon, stick to Liquid; for <=3 yrs, allow Liquid + Ultra Short
    allowed = ['Liquid'] if time_to_goal > 3 else ['Liquid', 'Ultra Short']
    debt_filtered = duration_debt_df[duration_debt_df['Sub-category'].str.strip().isin(allowed)].copy()
    if debt_filtered.empty:
        raise ValueError("⚠️ No suitable debt funds found for the selected duration.")

    debt_filtered = standardize_fund_col(debt_filtered)
    debt_filtered['Category'] = 'Debt'

    # Trim to satisfy min-10 rule given the bucket
    debt_filtered = trim_funds_to_min(debt_filtered, debt_pct, min_pct=10, sort_col='Weight' if 'Weight' in debt_filtered.columns else None)

    # Equal split to start within the bucket
    debt_filtered['Weight (%)'] = debt_pct / len(debt_filtered)

    return debt_filtered[['Fund', 'Category', 'Sub-category', 'Weight (%)']]

# ================= API Endpoint: Generate portfolio =================
@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    # 1) Funding + Strategy + Glide
    fv, funding_ratio = calculate_funding_ratio(
        user_input.monthly_investment,
        user_input.target_corpus,
        user_input.years_to_goal
    )
    strategy = choose_strategy(user_input.years_to_goal, user_input.risk_profile, funding_ratio)
    glide_path = generate_step_down_glide_path(user_input.years_to_goal, funding_ratio, user_input.risk_profile)

    # Use Year-1 buckets for construction; ensure non-zero bucket floors (defensive)
    glide_equity_pct = int(glide_path.iloc[0]['Equity Allocation (%)'])
    if glide_equity_pct > 0 and glide_equity_pct < 10:
        glide_equity_pct = 10
    glide_debt_pct = 100 - glide_equity_pct
    if glide_debt_pct > 0 and glide_debt_pct < 10:
        glide_debt_pct = 10
        glide_equity_pct = 100 - glide_debt_pct

    # 2) Equity selection (by chosen strategy)
    if strategy == "Active":
        eq_df = active_eq.copy()
        eq_df['Sub-category'] = 'Active'
    elif strategy == "Passive":
        eq_df = passive_eq.copy()
        eq_df['Sub-category'] = 'Passive'
    elif strategy == "Hybrid":
        eq_df = hybrid_eq.copy()
        eq_df = eq_df.rename(columns={'Type': 'Sub-category'})
    else:
        raise ValueError("Invalid strategy")

    eq_df = standardize_fund_col(eq_df)
    eq_df['Category'] = 'Equity'

    # If a 'Weight' column exists, keep it as a relative signal; otherwise equal weights
    if 'Weight' not in eq_df.columns:
        eq_df['Weight'] = 1.0

    # Trim equity funds so each can be >=10% within the equity bucket
    eq_df = trim_funds_to_min(eq_df, glide_equity_pct, min_pct=10, sort_col='Weight')

    # Start with proportional weights inside the equity bucket
    eq_df['Weight'] = eq_df['Weight'] / eq_df['Weight'].sum()
    eq_df['Weight (%)'] = eq_df['Weight'] * glide_equity_pct

    # Enforce per-fund min 10% within equity bucket
    eq_df = enforce_min_allocation(eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']], glide_equity_pct, min_pct=10, step=5)

    # 3) Debt selection
    selected_debt = get_debt_allocation(duration_debt, user_input.years_to_goal, glide_debt_pct)
    selected_debt = enforce_min_allocation(selected_debt, glide_debt_pct, min_pct=10, step=5)

    # 4) Combine & final enforcement for whole portfolio
    combined = pd.concat([
        eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']],
        selected_debt[['Fund', 'Category', 'Sub-category', 'Weight (%)']]
    ], ignore_index=True)

    final_portfolio = enforce_min_allocation(combined, 100, min_pct=10, step=5)

    # Safety checks
    if (final_portfolio['Weight (%)'] < 10).any():
        raise ValueError("Found a fund below 10% after enforcement.")
    if final_portfolio['Weight (%)'].sum() != 100:
        raise ValueError("Final portfolio does not sum to 100%.")

    return {
        "strategy": strategy,
        "funding_ratio": round(float(funding_ratio), 4),
        "glide_path": glide_path.to_dict(orient="records"),
        "portfolio": final_portfolio.to_dict(orient="records")
    }

# ================= API Endpoint: Explain Glide (NEW) =================
@app.post("/explain/glide")
async def explain_glide(request: Request):
    """
    Expects body:
    {
      "common": { "user": {...}, "goal": {"years_to_goal": <int>} },
      "glide_path": {
        "bands": [ { "Year": 1, "Equity Allocation (%)": 85, "Debt Allocation (%)": 15 }, ... ]
        // OR: "glide_path": [ ... ]  (alternate key also tolerated)
      }
    }
    """
    try:
        payload = await request.json()
        common = payload.get("common", {}) or {}
        glide_path = payload.get("glide_path", {}) or {}

        # Tolerate either "bands" or "glide_path" inside glide_path
        if "bands" not in glide_path and "glide_path" in glide_path:
            glide_path = {"bands": glide_path.get("glide_path", [])}

        out = explain_glide_story(common, glide_path)
        return out
    except Exception as e:
        # Surface a helpful error to the caller
        raise HTTPException(status_code=400, detail=str(e))


# ================= API Endpoint: Explain Strategy (NEW) =================
class StrategyExplainRequest(BaseModel):
    strategy: str                  # "Active" | "Passive" | "Hybrid"
    years_to_goal: int
    risk_preference: str           # "conservative" | "moderate" | "aggressive"
    funding_ratio: float = None

@app.post("/explain/strategy")
def explain_strategy(req: StrategyExplainRequest):
    """
    Returns a layman-friendly 3-line explanation of why this strategy was chosen.
    """
    try:
        return explain_strategy_story(
            strategy=req.strategy,
            years_to_goal=req.years_to_goal,
            risk_preference=req.risk_preference,
            funding_ratio=req.funding_ratio,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ================= API Endpoint: Explain Portfolio (NEW) =================
class FundItem(BaseModel):
    Fund: str
    Category: str
    Sub_category: str
    Weight: float

class PortfolioExplainRequest(BaseModel):
    portfolio: list[FundItem]

@app.post("/explain/portfolio")
def explain_portfolio(req: PortfolioExplainRequest):
    """
    Returns a plain-language story about the portfolio funds.
    Each fund should include Fund, Category, Sub_category, and Weight.
    """
    try:
        df = pd.DataFrame([f.dict() for f in req.portfolio])
        return explain_portfolio_story(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ================= API Endpoint: One-call Orchestrator (NEW) =================
@app.post("/plan/full")
def generate_full_plan(user_input: PortfolioInput):
    """
    One call that returns:
    - Glide path + Glide explanation
    - Strategy choice + Strategy explanation
    - Portfolio (fund list) + Portfolio explanation
    Grouped and ordered for the UI.
    """
    # ---------- 1) Funding + Strategy + Glide ----------
    fv, funding_ratio = calculate_funding_ratio(
        user_input.monthly_investment,
        user_input.target_corpus,
        user_input.years_to_goal
    )

    # choose strategy
    strategy = choose_strategy(user_input.years_to_goal, user_input.risk_profile, funding_ratio)

    # build glide path (DataFrame -> list[dict])
    glide_df = generate_step_down_glide_path(user_input.years_to_goal, funding_ratio, user_input.risk_profile)
    glide_records = glide_df.to_dict(orient="records")

    # ---------- 2) Portfolio construction ----------
    # Use Year-1 buckets for construction; ensure non-zero bucket floors (defensive)
    glide_equity_pct = int(glide_df.iloc[0]['Equity Allocation (%)'])
    if glide_equity_pct > 0 and glide_equity_pct < 10:
        glide_equity_pct = 10
    glide_debt_pct = 100 - glide_equity_pct
    if glide_debt_pct > 0 and glide_debt_pct < 10:
        glide_debt_pct = 10
        glide_equity_pct = 100 - glide_debt_pct

    # Equity universe by strategy
    if strategy == "Active":
        eq_df = active_eq.copy()
        eq_df['Sub-category'] = 'Active'
    elif strategy == "Passive":
        eq_df = passive_eq.copy()
        eq_df['Sub-category'] = 'Passive'
    elif strategy == "Hybrid":
        eq_df = hybrid_eq.copy()
        eq_df = eq_df.rename(columns={'Type': 'Sub-category'})
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    eq_df = standardize_fund_col(eq_df)
    eq_df['Category'] = 'Equity'
    if 'Weight' not in eq_df.columns:
        eq_df['Weight'] = 1.0

    # Trim to satisfy min-10 inside equity bucket
    eq_df = trim_funds_to_min(eq_df, glide_equity_pct, min_pct=10, sort_col='Weight')
    eq_df['Weight'] = eq_df['Weight'] / eq_df['Weight'].sum()
    eq_df['Weight (%)'] = eq_df['Weight'] * glide_equity_pct
    eq_df = enforce_min_allocation(eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']], glide_equity_pct, min_pct=10, step=5)

    # Debt selection
    selected_debt = get_debt_allocation(duration_debt, user_input.years_to_goal, glide_debt_pct)
    selected_debt = enforce_min_allocation(selected_debt, glide_debt_pct, min_pct=10, step=5)

    # Combine and final enforcement to 100%
    combined = pd.concat([
        eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']],
        selected_debt[['Fund', 'Category', 'Sub-category', 'Weight (%)']]
    ], ignore_index=True)
    final_portfolio = enforce_min_allocation(combined, 100, min_pct=10, step=5)

    if (final_portfolio['Weight (%)'] < 10).any():
        raise HTTPException(status_code=400, detail="Found a fund below 10% after enforcement.")
    if final_portfolio['Weight (%)'].sum() != 100:
        raise HTTPException(status_code=400, detail="Final portfolio does not sum to 100%.")

    # ---------- 3) EXPLANATIONS ----------
    # Glide explainer expects {common, glide_path:{bands:[...]}}
    glide_block = explain_glide_story(
        {"user": {"risk_preference": user_input.risk_profile},
         "goal": {"years_to_goal": user_input.years_to_goal}},
        {"bands": glide_records}
    )

    # Strategy explainer
    strategy_block = explain_strategy_story(
        strategy=strategy,
        years_to_goal=user_input.years_to_goal,
        risk_preference=user_input.risk_profile,
        funding_ratio=funding_ratio
    )

    # Portfolio explainer expects Weight (or Weight (%)); we already have "Weight (%)"
    # Convert to "Weight" for cleaner schema
    portfolio_for_story = final_portfolio.rename(columns={"Weight (%)": "Weight"})
    portfolio_block = explain_portfolio_story(portfolio_for_story)

    # ---------- 4) Assemble ordered sections ----------
    response = {
        "meta": {
            "funding_ratio": round(float(funding_ratio), 4),
            "future_value": round(float(fv), 2),
            "years_to_goal": int(user_input.years_to_goal),
            "risk_profile": user_input.risk_profile,
            "strategy": strategy
        },
        "sections": [
            {
                "id": "glide",
                "output": {"glide_path": glide_records},
                "explanation": {
                    "story": glide_block["story"],
                    "data_points": glide_block["data_points"]
                }
            },
            {
                "id": "strategy",
                "output": {"strategy": strategy},
                "explanation": {
                    "story": strategy_block["story"],
                    "data_points": strategy_block["data_points"]
                }
            },
            {
                "id": "portfolio",
                "output": {"portfolio": final_portfolio.to_dict(orient="records")},
                "explanation": {
                    "story": portfolio_block["story"],
                    "data_points": portfolio_block["data_points"]
                }
            }
        ]
    }
    return response



# ================= Entrypoint =================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
# ===================== end main.py ======================
