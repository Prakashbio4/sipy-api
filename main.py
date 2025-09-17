# ===================== main.py =====================
import os
import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyairtable import Api

# ---------- Local explainers (imported from your project) ----------
# Make sure these modules exist alongside main.py
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story

__API_BUILD__ = "sipy-fix-glide-cap-portfolio-df-writeback-2025-09-17"

app = FastAPI(title="SIPY Investment Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =================== Helper utilities ===================

def _to_number(v) -> float:
    """Convert strings like '25k', '25,000' -> 25000.0"""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").lower()
    if s.endswith("k"):
        return float(s[:-1]) * 1000.0
    return float(s)

def _req_int(name: str, v) -> int:
    try:
        return int(float(_to_number(v)))
    except Exception:
        raise HTTPException(status_code=422, detail=f"Missing/invalid integer: {name}")

def _req_float(name: str, v) -> float:
    try:
        return float(_to_number(v))
    except Exception:
        raise HTTPException(status_code=422, detail=f"Missing/invalid number: {name}")

def compact_glide_verbose_to_array(glide_verbose: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Input: [{"Year":1,"Equity Allocation (%)":70,"Debt Allocation (%)":30}, ...]
    Output (compact): {"v":1,"g":[[1,70,30],[2,70,30],...]}
    """
    arr = []
    for item in glide_verbose:
        y = int(item.get("Year"))
        e = int(round(float(item.get("Equity Allocation (%)", 0))))
        d = int(round(float(item.get("Debt Allocation (%)", 0))))
        arr.append([y, e, d])
    return {"v": 1, "g": arr}

def minified_json(obj: Any) -> str:
    """Dump JSON without whitespace (smaller payloads)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def safe_store(text: str, max_len: int = 45000) -> str:
    """
    Trim text if it exceeds a conservative Long text comfort zone; add marker if truncated.
    """
    if text is None:
        return ""
    return text if len(text) <= max_len else text[: max_len - 12] + "...[TRUNC]"

# =================== Data loading (optional) ===================
# If you rely on CSVs for fund lists, load them here. Keep optional to avoid boot failure.
def _try_load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

ACTIVE_EQ = _try_load_csv("data/active_equity_portfolio.csv")
PASSIVE_EQ = _try_load_csv("data/passive_equity_portfolio.csv")
HYBRID_EQ = _try_load_csv("data/hybrid_equity_portfolio.csv")
DURATION_DEBT = _try_load_csv("data/debt_duration_selected.csv")

# =================== Core investment logic ===================

class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str  # 'conservative' | 'moderate' | 'aggressive' (free text tolerated)

def calculate_funding_ratio(monthly_investment: float, target_corpus: float, years: int):
    """
    Simple annuity FV to estimate funding; tweak expected_return as you see fit.
    """
    expected_return = 0.13  # 13% annual
    r = (1 + expected_return) ** (1/12) - 1
    n = years * 12
    fv = monthly_investment * (((1 + r) ** n - 1) / r)
    return fv, (fv / target_corpus) if target_corpus > 0 else 0.0

def choose_strategy(time_to_goal: int, risk_profile: str, funding_ratio: float) -> str:
    rp = (risk_profile or "").strip().lower()
    if time_to_goal <= 3:
        return "Passive"
    if funding_ratio < 0.8 and rp == "aggressive":
        return "Active"
    if rp == "conservative":
        return "Passive"
    # Default middle ground
    return "Hybrid"

def generate_step_down_glide_path(time_to_goal: int, funding_ratio: float, risk_profile: str) -> pd.DataFrame:
    """
    Yearly glide path: simple linear step-down from base_equity to 10%.
    You can replace with your existing, richer logic.
    """
    # base equity chosen loosely by funding/risk; tweak as needed
    rp = (risk_profile or "").strip().lower()
    if funding_ratio >= 1.2:
        base_equity = 75
    elif funding_ratio >= 0.9:
        base_equity = 70
    else:
        base_equity = 80  # a tad higher if underfunded

    if rp == "conservative":
        base_equity -= 10
    elif rp == "aggressive":
        base_equity += 5

    base_equity = max(10, min(90, base_equity))
    final_equity = 10
    steps = max(1, time_to_goal - 1)
    step_down = (base_equity - final_equity) / steps

    rows = []
    for y in range(1, time_to_goal + 1):
        eq = int(round(base_equity - (y - 1) * step_down))
        eq = max(0, min(100, eq))
        de = 100 - eq
        rows.append({"Year": y, "Equity Allocation (%)": eq, "Debt Allocation (%)": de})
    df = pd.DataFrame(rows)
    # ✅ Hard-cap length (safety)
    return df.iloc[: time_to_goal].reset_index(drop=True)

def build_simple_portfolio_df(equity_pct: int) -> pd.DataFrame:
    """
    Minimal portfolio frame that the explainer can consume.
    You can replace this with your real equity/debt fund selection.
    """
    debt_pct = 100 - equity_pct
    data = [
        {"Fund": "Equity Allocation (Basket)", "Category": "Equity", "Sub-category": "Blend", "Weight (%)": equity_pct},
        {"Fund": "Debt Allocation (Basket)",   "Category": "Debt",   "Sub-category": "Duration", "Weight (%)": debt_pct},
    ]
    return pd.DataFrame(data, columns=["Fund", "Category", "Sub-category", "Weight (%)"])

def run_engine(user_input: PortfolioInput) -> Dict[str, Any]:
    """
    Full pipeline: funding -> strategy -> glide -> portfolio -> explainers.
    Returns a dict that is JSON-serializable.
    """
    # 1) funding + strategy
    fv, fr = calculate_funding_ratio(user_input.monthly_investment, user_input.target_corpus, user_input.years_to_goal)
    strategy = choose_strategy(user_input.years_to_goal, user_input.risk_profile, fr)

    # 2) glide path (DataFrame) and ensure correct length
    glide_df = generate_step_down_glide_path(user_input.years_to_goal, fr, user_input.risk_profile)
    glide_records = glide_df.to_dict(orient="records")

    # 3) portfolio DataFrame (ensure it's a DF before explainer)
    #    Use Year-1 allocation as the base portfolio split for illustration
    y1_equity = int(glide_df.iloc[0]["Equity Allocation (%)"]) if not glide_df.empty else 60
    portfolio_df = build_simple_portfolio_df(y1_equity)

    # 4) explainers (guard against unexpected errors to keep API resilient)
    glide_block = {}
    strategy_block = {}
    portfolio_block = {}
    try:
        glide_block = explain_glide_story({
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(fr),
            "glide_path": glide_records,
        })
    except Exception as e:
        glide_block = {"story": f"(glide explainer unavailable) {e}"}

    try:
        strategy_block = explain_strategy_story({
            "strategy": strategy,
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(fr),
        })
    except Exception as e:
        strategy_block = {"story": f"(strategy explainer unavailable) {e}"}

    try:
        portfolio_block = explain_portfolio_story(portfolio_df)
    except Exception as e:
        portfolio_block = {"story": f"(portfolio explainer unavailable) {e}"}

    return {
        "strategy": strategy,
        "funding_ratio": round(float(fr), 4),
        "glide_path": glide_records,                                  # list[dict]
        "portfolio": portfolio_df.to_dict(orient="records"),          # list[dict]
        "glide_explainer": glide_block,
        "strategy_explainer": strategy_block,
        "portfolio_explainer": portfolio_block,
    }

# =================== Public endpoints ===================

@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    try:
        output = run_engine(user_input)
        # ✅ Safety cap in case anything upstream over-produces
        years = user_input.years_to_goal
        output["glide_path"] = output["glide_path"][:years]
        return output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================== Airtable integration ===================

class AirtableWebhookPayload(BaseModel):
    record_id: str

AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")  # PAT with data.records:read/write
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "Investor_inputs"  # you can also use the table ID (tbl...)

_api = Api(AIRTABLE_API_KEY)
_airtable = _api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

@app.post("/trigger_processing/")
async def trigger_processing(payload: AirtableWebhookPayload, authorization: Optional[str] = Header(None)):
    """
    Called by Airtable Automation with { record_id }.
    Reads inputs from the record, runs engine, writes outputs back to the same record.
    """
    try:
        rec_id = payload.record_id
        rec = _airtable.get(rec_id)
        fields = rec.get("fields", {})

        # Pull inputs (tolerant to earlier naming variants)
        monthly_raw = fields.get("Monthly Investment", fields.get("Monthly Investments"))
        target_raw = fields.get("Target Corpus")
        years_raw = fields.get("Time to Goal (years)", fields.get("Time Horizon (years)"))
        risk_raw = fields.get("Risk Preference", "Moderate")

        user_input = PortfolioInput(
            monthly_investment=_req_float("Monthly Investment", monthly_raw),
            target_corpus=_req_float("Target Corpus", target_raw),
            years_to_goal=_req_int("Time to Goal (years)", years_raw),
            risk_profile=(risk_raw or "Moderate"),
        )

        processed = run_engine(user_input)

        # ✅ Enforce exact N-year glide
        N = user_input.years_to_goal
        glide_verbose = processed["glide_path"][:N]

        # Compact + minify for Long text fields
        compact_glide = compact_glide_verbose_to_array(glide_verbose)
        glide_json = minified_json(compact_glide)

        portfolio_json = minified_json(processed["portfolio"])

        # Write back to your exact Long text columns
        update_fields = {
            "glide_path":               safe_store(glide_json),
            "portfolio":                safe_store(portfolio_json),
            "glide_explainer_story":    processed["glide_explainer"]["story"],
            "strategy_explainer_story": processed["strategy_explainer"]["story"],
            "portfolio_explainer_story":processed["portfolio_explainer"]["story"],
            # ✅ also persist these two as requested
            "funding_ratio":            processed["funding_ratio"],
            "strategy":                 processed["strategy"],
        }
        _airtable.update(rec_id, fields=update_fields)

        return {"status": "success", "record_id": rec_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process: {e}")

@app.get("/health")
def health():
    return {"ok": True, "build": __API_BUILD__}

# ============= Local dev entrypoint (optional) =============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
