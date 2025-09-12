# ===================== main.py =====================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

__API_BUILD__ = "glide-fix-002"  # bump this whenever you deploy

def _call_glide_explainer(glide_path_df, years_to_goal, risk_profile, funding_ratio):
    """Works with both legacy(list) and new(dict) glide_explainer signatures."""
    rows = glide_path_df.to_dict(orient="records")
    try:
        # Legacy: function expects just the list
        return explain_glide_story(rows)
    except TypeError as e:
        # Try new-style payload with full context
        try:
            return explain_glide_story({
                "years_to_goal": years_to_goal,
                "risk_profile": risk_profile,
                "funding_ratio": float(funding_ratio),
                "glide_path": rows,
            })
        except TypeError:
            # Re-raise original for a clean error if both fail
            raise e


# === Explainers (stories layer) ===
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story

app = FastAPI(title="SIPY Investment Engine API")

# === Load data once at app startup ===
def _load_csvs():
    try:
        # Your latest CSVs live under data/
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
    Same logic as before, with floors so any non-zero bucket >= 10%.
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

        # bucket floors so we never have a non-zero bucket < 10%
        if 0 < equity < 10:
            equity = 10

        debt = 100 - equity
        if 0 < debt < 10:
            debt = 10
            equity = 100 - debt

        glide_path.append({
            'Year': year,
            'Equity Allocation (%)': int(equity),
            'Debt Allocation (%)': int(debt)
        })

    return pd.DataFrame(glide_path)

def choose_strategy(time_to_goal, risk_profile, funding_ratio):
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
    max_funds = max(1, int(total_pct // min_pct))
    if len(df) > max_funds:
        if sort_col and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False).head(max_funds)
        else:
            df = df.head(max_funds)
    return df.copy()

def enforce_min_allocation(df, total_pct, min_pct=10, step=5):
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
    df = df.copy()
    df['Fund'] = df.index.astype(str)
    return df

def get_debt_allocation(duration_debt_df, time_to_goal, debt_pct):
    duration_debt_df = duration_debt_df.copy()
    allowed = ['Liquid'] if time_to_goal > 3 else ['Liquid', 'Ultra Short']
    debt_filtered = duration_debt_df[duration_debt_df['Sub-category'].str.strip().isin(allowed)].copy()
    if debt_filtered.empty:
        raise ValueError("⚠️ No suitable debt funds found for the selected duration.")
    debt_filtered = standardize_fund_col(debt_filtered)
    debt_filtered['Category'] = 'Debt'
    debt_filtered = trim_funds_to_min(
        debt_filtered, debt_pct, min_pct=10,
        sort_col='Weight' if 'Weight' in debt_filtered.columns else None
    )
    debt_filtered['Weight (%)'] = debt_pct / len(debt_filtered)
    return debt_filtered[['Fund', 'Category', 'Sub-category', 'Weight (%)']]

# ================= API Endpoint =================
@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    try:
        #normalize risk profile
        risk = (user_input.risk_profile or "").strip().lower()
        # 1) Funding + Strategy + Glide (REAL logic)
        fv, funding_ratio = calculate_funding_ratio(
            user_input.monthly_investment,
            user_input.target_corpus,
            user_input.years_to_goal
        )
        strategy = choose_strategy(user_input.years_to_goal, risk, funding_ratio)
        glide_path = generate_step_down_glide_path(
            user_input.years_to_goal, funding_ratio, risk
        )

        # Use Year-1 buckets for construction; ensure non-zero bucket floors (defensive)
        glide_equity_pct = int(glide_path.iloc[0]['Equity Allocation (%)'])
        if 0 < glide_equity_pct < 10:
            glide_equity_pct = 10
        glide_debt_pct = 100 - glide_equity_pct
        if 0 < glide_debt_pct < 10:
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
        if 'Weight' not in eq_df.columns:
            eq_df['Weight'] = 1.0

        eq_df = trim_funds_to_min(eq_df, glide_equity_pct, min_pct=10, sort_col='Weight')
        eq_df['Weight'] = eq_df['Weight'] / eq_df['Weight'].sum()
        eq_df['Weight (%)'] = eq_df['Weight'] * glide_equity_pct
        eq_df = enforce_min_allocation(
            eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']],
            glide_equity_pct, min_pct=10, step=5
        )

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

        # ---------- EXPLAINER BLOCKS ----------
        glide_rows = glide_path.to_dict(orient="records")

        # Your glide_explainer currently expects only the list (legacy signature)
        glide_block = _call_glide_explainer(
    glide_path, user_input.years_to_goal, user_input.risk_profile, funding_ratio
)

        # Strategy explainer: give richer context
        equity_summary = {
            "bucket_pct": int(glide_equity_pct),
            "fund_count": int((final_portfolio['Category'] == 'Equity').sum()),
            "subcats": (final_portfolio[final_portfolio['Category'] == 'Equity']['Sub-category']
                        .value_counts().to_dict()),
        }
        debt_summary = {
            "bucket_pct": int(glide_debt_pct),
            "fund_count": int((final_portfolio['Category'] == 'Debt').sum()),
            "subcats": (final_portfolio[final_portfolio['Category'] == 'Debt']['Sub-category']
                        .value_counts().to_dict()),
        }
        strategy_ctx = {
            "strategy": strategy,
            "funding_ratio": round(float(funding_ratio), 4),
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "glide_path": glide_rows,
            "equity_summary": equity_summary,
            "debt_summary": debt_summary,
        }
        try:
            strategy_block = explain_strategy_story(strategy_ctx)
        except TypeError as e:
            msg = str(e).lower()
            if "unexpected" in msg or "positional" in msg or "keyword" in msg:
                strategy_block = explain_strategy_story({
                    "strategy": strategy,
                    "funding_ratio": float(funding_ratio)
                })
            else:
                raise

        # Portfolio explainer: pass the final DF
        portfolio_block = explain_portfolio_story(final_portfolio)

        # ---------- RESPONSE ----------
        return {
            # Raw engine outputs (backward-compatible)
            "strategy": strategy,
            "funding_ratio": round(float(funding_ratio), 4),
            "glide_path": glide_rows,
            "portfolio": final_portfolio.to_dict(orient="records"),

            # Explainer blocks (new)
            "glide_explainer": glide_block,
            "strategy_explainer": strategy_block,
            "portfolio_explainer": portfolio_block,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
def health():
    return {"ok": True, "build": __API_BUILD__}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
