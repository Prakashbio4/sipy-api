# ===================== main.py =====================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import pandas as pd
import json
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from pyairtable import Api
import re

from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story

__API_BUILD__ = "final-fix-003"
app = FastAPI(title="SIPY Investment Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# === Load data once ===
def _load_csvs():
    try:
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

# === Pydantic Input ===
class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str

# ----------------- Utility functions -----------------
def calculate_funding_ratio(monthly_investment, target_corpus, years):
    expected_return = 0.13
    r = (1 + expected_return) ** (1/12) - 1
    n = years * 12
    fv = monthly_investment * (((1 + r) ** n - 1) / r)
    return fv, fv / target_corpus

def generate_step_down_glide_path(time_to_goal, funding_ratio, risk_profile):
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

    rp = (risk_profile or "").strip().lower()
    if rp == 'conservative':
        base_equity -= 10
    elif rp == 'aggressive':
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

    rp = (risk_profile or "").strip().lower()

    if time_to_goal <= 5:
        if funding_status == 'overfunded':
            return "Passive"
        elif rp == 'aggressive' and funding_status == 'underfunded':
            return "Hybrid"
        else:
            return "Passive"
    elif 6 <= time_to_goal <= 10:
        if rp == 'conservative':
            return "Passive"
        elif rp == 'moderate':
            return "Hybrid" if funding_status == 'balanced' else "Passive"
        elif rp == 'aggressive':
            return "Active" if funding_status == 'underfunded' else "Hybrid"
    else:
        if rp == 'conservative':
            return "Passive"
        elif funding_status == 'underfunded':
            return "Active"
        elif funding_status == 'overfunded':
            return "Hybrid" if rp != 'conservative' else "Passive"
        else:
            return "Active" if rp == 'aggressive' else "Hybrid"

# === Portfolio Engine ===
@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    try:
        risk = (user_input.risk_profile or "").strip().lower()

        fv, funding_ratio = calculate_funding_ratio(
            user_input.monthly_investment,
            user_input.target_corpus,
            user_input.years_to_goal
        )
        strategy = choose_strategy(user_input.years_to_goal, risk, funding_ratio)
        glide_path = generate_step_down_glide_path(
            user_input.years_to_goal, funding_ratio, risk
        )

        glide_equity_pct = int(glide_path.iloc[0]['Equity Allocation (%)'])
        if 0 < glide_equity_pct < 10:
            glide_equity_pct = 10
        glide_debt_pct = 100 - glide_equity_pct
        if 0 < glide_debt_pct < 10:
            glide_debt_pct = 10
            glide_equity_pct = 100 - glide_debt_pct

        # 2) Equity selection
        if strategy == "Active":
            eq_df = active_eq.copy(); eq_df['Sub-category'] = 'Active'
        elif strategy == "Passive":
            eq_df = passive_eq.copy(); eq_df['Sub-category'] = 'Passive'
        elif strategy == "Hybrid":
            eq_df = hybrid_eq.copy(); eq_df = eq_df.rename(columns={'Type': 'Sub-category'})
        else:
            raise ValueError("Invalid strategy")

        eq_df['Category'] = 'Equity'
        if 'Weight' not in eq_df.columns:
            eq_df['Weight'] = 1.0

        eq_df['Weight'] = eq_df['Weight'] / eq_df['Weight'].sum()
        eq_df['Weight (%)'] = eq_df['Weight'] * glide_equity_pct

        combined = pd.concat([eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']]], ignore_index=True)
        final_portfolio = combined

        glide_context = {"years_to_goal": user_input.years_to_goal, "risk_profile": user_input.risk_profile,
                         "funding_ratio": float(funding_ratio), "glide_path": glide_path.to_dict(orient="records")}
        glide_block = explain_glide_story(glide_context)

        strategy_context = {"strategy": strategy, "years_to_goal": user_input.years_to_goal,
                            "risk_profile": user_input.risk_profile, "funding_ratio": float(funding_ratio)}
        strategy_block = explain_strategy_story(strategy_context)

        portfolio_block = explain_portfolio_story(final_portfolio)

        return {
            "strategy": strategy,
            "funding_ratio": round(float(funding_ratio), 4),
            "glide_path": glide_path.to_dict(orient="records"),
            "portfolio": final_portfolio.to_dict(orient="records"),
            "glide_explainer": glide_block,
            "strategy_explainer": strategy_block,
            "portfolio_explainer": portfolio_block,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Airtable ===
class AirtableWebhookPayload(BaseModel):
    record_id: str

AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "Investor_inputs"

api = Api(AIRTABLE_API_KEY)
airtable = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

@app.post("/trigger_processing/")
async def trigger_processing(payload: AirtableWebhookPayload):
    try:
        record_id = payload.record_id
        record = airtable.get(record_id)
        inputs = record.get('fields', {})

        user_input_data = {
            "monthly_investment": inputs.get("Monthly Investments"),
            "target_corpus": inputs.get("Target Corpus"),
            "years_to_goal": inputs.get("Time Horizon (years)"),
            "risk_profile": inputs.get("Risk Preference")
        }

        processed_output = generate_portfolio(PortfolioInput(**user_input_data))

        # ✅ FIXED: Format glide_path for Airtable
        formatted_glide_path = ", ".join(
            [
                f"{{{row['Year']}, {row['Equity Allocation (%)']}, {row['Debt Allocation (%)']}}}"
                for row in processed_output["glide_path"]
            ]
        )

        update_data = {
            "strategy": processed_output["strategy"],
            "funding_ratio": processed_output["funding_ratio"],
            "glide_path": formatted_glide_path,  # ✅ fixed line
            "portfolio": json.dumps(processed_output["portfolio"]),
            "glide_explainer_story": processed_output["glide_explainer"]["story"],
            "strategy_explainer_story": processed_output["strategy_explainer"]["story"],
            "portfolio_explainer_story": processed_output["portfolio_explainer"]["story"],
        }

        airtable.update(record_id, fields=update_data)
        return {"status": "success", "record_id": record_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process: {e}")

@app.get("/health")
def health():
    return {"ok": True, "build": __API_BUILD__}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
