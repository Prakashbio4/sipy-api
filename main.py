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


# === Explainers (stories layer) ===
# NOTE: These are local files, not remote libraries
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story


__API_BUILD__ = "final-fix-001"
app = FastAPI(title="SIPY Investment Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# === Load data once at app startup ===
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
        print(f"❌ Error loading CSVs: {e}")
        return None, None, None, None, None

active_equity, passive_equity, hybrid_equity, tmf, duration_debt = _load_csvs()

# ==================== Airtable Setup ===================
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME")
# Use a default table name if not provided
AIRTABLE_TABLE_NAME = AIRTABLE_TABLE_NAME or "Investor_inputs"

airtable = Api(AIRTABLE_API_KEY)
investor_inputs_table = airtable.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)


# ==================== Core Logic ====================
class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str


def get_risk_score(risk_profile: str) -> float:
    risk_profiles = {
        "conservative": 0.2,
        "moderate": 0.5,
        "aggressive": 0.8,
    }
    return risk_profiles.get(risk_profile.lower(), 0.5)


def generate_glide_path(
    time_to_goal: int,
    risk_score: float,
    max_equity: int = 90,
    min_equity: int = 10,
    smoothness: float = 0.5,
) -> List[Dict[str, Any]]:
    path = []
    for year in range(1, time_to_goal + 1):
        if time_to_goal == 1:
            equity = int(risk_score * (max_equity - min_equity) + min_equity)
        else:
            t_norm = (year - 1) / (time_to_goal - 1)
            factor = np.exp(-smoothness * t_norm)
            equity = int(min_equity + (max_equity - min_equity) * factor * risk_score)
        debt = 100 - equity
        path.append({
            'Year': year,
            'Equity Allocation (%)': equity,
            'Debt Allocation (%)': debt,
        })
    return path


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

    return glide_path


def generate_portfolio(
    input: PortfolioInput,
):
    # Determine the strategy based on the risk profile and time horizon
    years = input.years_to_goal
    risk = input.risk_profile.lower()

    if years <= 3:
        strategy = "short_term_debt"
    elif years > 3 and risk in ["conservative", "moderate"]:
        strategy = "balanced_hybrid"
    else:
        strategy = "aggressive_equity"

    # Get portfolio composition based on the determined strategy
    if strategy == "short_term_debt":
        portfolio_composition = duration_debt.to_dict('records')
    elif strategy == "balanced_hybrid":
        portfolio_composition = hybrid_eq.to_dict('records')
    else:
        portfolio_composition = active_equity.to_dict('records')

    # Generate the glide path (equity/debt allocation over time)
    # NOTE: Using a simplified logic for now
    glide_path = generate_step_down_glide_path(
        input.years_to_goal,
        input.monthly_investment * input.years_to_goal * 12 / input.target_corpus,
        input.risk_profile,
    )

    # Generate explainer stories
    glide_explainer = explain_glide_story(glide_path, input.years_to_goal)
    strategy_explainer = explain_strategy_story(strategy, input.risk_profile, input.years_to_goal)
    portfolio_explainer = explain_portfolio_story(portfolio_composition)

    output = {
        "strategy": strategy,
        "funding_ratio": input.monthly_investment * input.years_to_goal * 12 / input.target_corpus,
        "glide_path": glide_path,
        "portfolio": portfolio_composition,
        "glide_explainer": glide_explainer,
        "strategy_explainer": strategy_explainer,
        "portfolio_explainer": portfolio_explainer,
    }

    return output


# ==================== API Endpoints ====================
class AirtableWebhookPayload(BaseModel):
    record_id: str


@app.post("/trigger_processing/")
async def trigger_processing(payload: AirtableWebhookPayload):
    try:
        record_id = payload.record_id

        # 1. Fetch the data from Airtable using the record_id
        record = airtable.get(record_id)
        inputs = record.get('fields', {})

        # Map Airtable fields to the PortfolioInput pydantic model
        user_input_data = {
            "monthly_investment": inputs.get("Monthly Investments"),
            "target_corpus": inputs.get("Target Corpus"),
            "years_to_goal": inputs.get("Time Horizon (years)"),
            "risk_profile": inputs.get("Risk Preference")
        }

        # 2. Call the existing portfolio generation engine
        try:
            processed_output = generate_portfolio(PortfolioInput(**user_input_data))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Engine error: {e}")

        # Convert the glide_path list to a clean, readable string
        glide_path_str = ", ".join(
            f"{{{record['Year']}, {record['Equity Allocation (%)']}, {record['Debt Allocation (%)']}}}"
            for record in processed_output["glide_path"]
        )

        # 3. Write the output back to Airtable against the same record_id
        update_data = {
            "strategy": processed_output["strategy"],
            "funding_ratio": processed_output["funding_ratio"],
            "glide_path": glide_path_str,
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)