import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pyairtable import Api

# === Explainers (imported locally) ===
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story

__API_BUILD__ = "sipy-prod-fix-strategy-funding-glide-001"
app = FastAPI(title="SIPY Investment Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def compact_glide_verbose_to_array(glide_verbose: List[Dict[str, Any]]) -> Dict[str, Any]:
    arr = []
    for item in glide_verbose:
        y = int(item.get("Year"))
        e = int(round(float(item.get("Equity Allocation (%)", 0))))
        d = int(round(float(item.get("Debt Allocation (%)", 0))))
        arr.append([y, e, d])
    return {"v": 1, "g": arr}

def minified_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def safe_store(text: str, max_len: int = 45000) -> str:
    if text is None:
        return ""
    return text if len(text) <= max_len else text[: max_len - 12] + "...[TRUNC]"

# === Load data ===
def _load_csvs():
    try:
        active_eq = pd.read_csv("data/active_equity_portfolio.csv")
        passive_eq = pd.read_csv("data/passive_equity_portfolio.csv")
        hybrid_eq = pd.read_csv("data/hybrid_equity_portfolio.csv")
        tmf = pd.read_csv("data/tmf_selected.csv")
        duration_debt = pd.read_csv("data/debt_duration_selected.csv")
        print("✅ CSVs loaded successfully.")
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
    risk_profile: str

# === Funding & Glide Path Logic ===
def calculate_funding_ratio(monthly_investment, target_corpus, years):
    expected_return = 0.13
    r = (1 + expected_return) ** (1/12) - 1
    n = years * 12
    fv = monthly_investment * (((1 + r) ** n - 1) / r)
    return fv, fv / target_corpus

def generate_step_down_glide_path(time_to_goal, funding_ratio, risk_profile):
    glide_path = []
    base_equity = 75
    final_equity = 10
    step = (base_equity - final_equity) / max(1, time_to_goal - 1)
    for year in range(1, time_to_goal + 1):
        eq = max(final_equity, int(round(base_equity - (year - 1) * step)))
        de = 100 - eq
        glide_path.append({
            "Year": year,
            "Equity Allocation (%)": eq,
            "Debt Allocation (%)": de
        })
    return pd.DataFrame(glide_path)

def choose_strategy(time_to_goal, risk_profile, funding_ratio):
    if time_to_goal <= 3:
        return "Passive"
    if funding_ratio < 0.8 and risk_profile == "aggressive":
        return "Active"
    if risk_profile == "conservative":
        return "Passive"
    return "Hybrid"

# === API: Portfolio Generator ===
@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    try:
        fv, funding_ratio = calculate_funding_ratio(
            user_input.monthly_investment,
            user_input.target_corpus,
            user_input.years_to_goal
        )

        strategy = choose_strategy(
            user_input.years_to_goal,
            user_input.risk_profile,
            funding_ratio
        )

        glide_path = generate_step_down_glide_path(
            user_input.years_to_goal,
            funding_ratio,
            user_input.risk_profile
        )

        # ✅ Always trim to years_to_goal
        glide_path = glide_path.iloc[: user_input.years_to_goal].reset_index(drop=True)

        glide_context = {
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(funding_ratio),
            "glide_path": glide_path.to_dict(orient="records")
        }
        glide_block = explain_glide_story(glide_context)

        strategy_block = explain_strategy_story({
            "strategy": strategy,
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(funding_ratio)
        })

        portfolio_block = explain_portfolio_story([])  # pass real portfolio later

        return {
            "strategy": strategy,
            "funding_ratio": round(float(funding_ratio), 4),
            "glide_path": glide_path.to_dict(orient="records"),
            "portfolio": [],  # placeholder
            "glide_explainer": glide_block,
            "strategy_explainer": strategy_block,
            "portfolio_explainer": portfolio_block,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Airtable Integration ===
class AirtableWebhookPayload(BaseModel):
    record_id: str

AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "Investor_inputs"

api = Api(AIRTABLE_API_KEY)
airtable = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

@app.post("/trigger_processing/")
async def trigger_processing(payload: AirtableWebhookPayload, authorization: Optional[str] = Header(None)):
    try:
        record_id = payload.record_id
        record = airtable.get(record_id)
        fields = record.get("fields", {})

        monthly = float(fields.get("Monthly Investment", 0))
        target = float(fields.get("Target Corpus", 0))
        years = int(fields.get("Time to Goal (years)", 10))
        risk = fields.get("Risk Preference", "Moderate")

        user_input = PortfolioInput(
            monthly_investment=monthly,
            target_corpus=target,
            years_to_goal=years,
            risk_profile=risk
        )

        processed = generate_portfolio(user_input)

        # ✅ Trim glide path again
        glide_verbose = processed["glide_path"][:years]
        compact_glide = compact_glide_verbose_to_array(glide_verbose)
        glide_json = minified_json(compact_glide)
        portfolio_json = minified_json(processed["portfolio"])

        update_data = {
            "glide_path": safe_store(glide_json),
            "portfolio": safe_store(portfolio_json),
            "glide_explainer_story": processed["glide_explainer"]["story"],
            "strategy_explainer_story": processed["strategy_explainer"]["story"],
            "portfolio_explainer_story": processed["portfolio_explainer"]["story"],
            "funding_ratio": processed["funding_ratio"],   # ✅ now written
            "strategy": processed["strategy"],             # ✅ now written
        }
        airtable.update(record_id, fields=update_data)

        return {"status": "success", "record_id": record_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process: {e}")

@app.get("/health")
def health():
    return {"ok": True, "build": __API_BUILD__}
