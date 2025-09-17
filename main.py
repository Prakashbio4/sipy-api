# ===================== main.py =====================
import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyairtable import Api

# ---------- Local explainers (must exist next to this file) ----------
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story

__API_BUILD__ = "sipy-fix-fields-fr2-glidecap-funds-2025-09-17"

app = FastAPI(title="SIPY Investment Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =================== Helpers ===================

INR_UNITS = [
    (re.compile(r"(crore|cr)\b", re.I), 10_000_000.0),
    (re.compile(r"(lakh|lac|l)\b", re.I), 100_000.0),
    (re.compile(r"k\b", re.I), 1_000.0),
]

def _to_number_india(v) -> float:
    """
    Parse Airtable text like: '₹25,00,000', '25L', '1.5 Cr', '25k', '2,50,000'
    -> float rupees.
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    # Drop currency symbols and spaces/commas/underscores
    s_clean = re.sub(r"[₹,\s_]", "", s, flags=re.U)

    # Unit multiplier (Cr, Lakh/Lac/L, k)
    mult = 1.0
    for pat, m in INR_UNITS:
        if pat.search(s):
            mult = m
            s_clean = pat.sub("", s_clean)

    # Keep only digits and decimal point
    s_num = re.sub(r"[^0-9.]", "", s_clean)
    if s_num in ("", "."):
        return 0.0

    try:
        return float(s_num) * mult
    except Exception:
        return 0.0

def _req_float(name: str, v) -> float:
    try:
        return float(_to_number_india(v))
    except Exception:
        raise HTTPException(status_code=422, detail=f"Missing/invalid number: {name}")

def _req_int(name: str, v) -> int:
    if v is None:
        raise HTTPException(status_code=422, detail=f"Missing/invalid integer: {name}")
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    s_digits = re.sub(r"[^\d]", "", s)
    if s_digits == "":
        raise HTTPException(status_code=422, detail=f"Missing/invalid integer: {name}")
    return int(s_digits)

def compact_glide_verbose_to_array(glide_verbose: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Input: [{"Year":1,"Equity Allocation (%)":70,"Debt Allocation (%)":30}, ...]
    Output: {"v":1,"g":[[1,70,30],[2,70,30],...]}
    """
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

# =================== CSV loading ===================
def _load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

ACTIVE_EQ    = _load_csv("data/active_equity_portfolio.csv")
PASSIVE_EQ   = _load_csv("data/passive_equity_portfolio.csv")
HYBRID_EQ    = _load_csv("data/hybrid_equity_portfolio.csv")
DURATION_DEBT= _load_csv("data/debt_duration_selected.csv")

def _standardize_fund_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if "Fund" in df.columns:
        return df
    cand = [c for c in df.columns if re.search(r"(fund|scheme|name)", c, re.I)]
    if cand:
        return df.rename(columns={cand[0]: "Fund"})
    out = df.copy()
    out["Fund"] = out.index.astype(str)
    return out

def _trim_rank_and_scale(df: pd.DataFrame, total_pct: int, min_pct: int = 10, sort_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Pick a handful of top rows, enforce min % per fund, normalize to total_pct, round to 1% then to 5%.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Fund","Category","Sub-category","Weight (%)"])

    df = df.copy()
    df = _standardize_fund_col(df)
    # Prefer explicit ranking/score columns if present
    sort_cols = sort_cols or [c for c in ["Score","Rank","AUM","Sharpe","Return"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols))
    # choose up to total_pct/min_pct funds (e.g., 60%/10% -> max 6 funds)
    max_funds = max(1, total_pct // min_pct)
    df = df.head(max_funds).copy()

    # Initial equal weighting
    if "Weight (%)" not in df.columns:
        df["Weight (%)"] = total_pct / len(df)
    # Round to 1, enforce minimum
    df["Weight (%)"] = df["Weight (%)"].apply(lambda x: max(min_pct, round(float(x))))
    # Normalize back to total_pct
    s = df["Weight (%)"].sum()
    df["Weight (%)"] = df["Weight (%)"] / (s if s else 1) * total_pct
    # Round to nearest 5 for cleaner numbers
    df["Weight (%)"] = df["Weight (%)"].apply(lambda x: int(round(float(x)/5.0)*5))
    # Fix rounding drift
    diff = total_pct - int(df["Weight (%)"].sum())
    if diff != 0:
        idx = df["Weight (%)"].idxmax()
        df.at[idx, "Weight (%)"] = int(df.at[idx, "Weight (%)"]) + diff
    return df[["Fund","Weight (%)"]]

# =================== Core investment logic ===================

class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str  # 'Conservative' | 'Moderate' | 'Aggressive' (case-insensitive)

def calculate_funding_ratio(monthly_investment: float, target_corpus: float, years: int):
    """
    Simple annuity FV to estimate funding ratio.
    """
    expected_return = 0.13  # 13% annual
    r = (1 + expected_return) ** (1/12) - 1
    n = max(0, years) * 12
    fv = monthly_investment * (((1 + r) ** n - 1) / r) if r > 0 else monthly_investment * n
    fr = fv / target_corpus if target_corpus > 0 else 0.0
    return fv, fr

def choose_strategy(time_to_goal: int, risk_profile: str, funding_ratio: float) -> str:
    rp = (risk_profile or "").strip().lower()
    if time_to_goal <= 3:
        return "Passive"
    if funding_ratio < 0.8 and rp == "aggressive":
        return "Active"
    if rp == "conservative":
        return "Passive"
    return "Hybrid"

def generate_step_down_glide_path(time_to_goal: int, funding_ratio: float, risk_profile: str) -> pd.DataFrame:
    """
    Yearly glide path: linear step-down from base equity to 10%.
    Always returns exactly `time_to_goal` rows (1..N).
    """
    rp = (risk_profile or "").strip().lower()

    # Base equity heuristic
    if funding_ratio >= 1.2:
        base_equity = 75
    elif funding_ratio >= 0.9:
        base_equity = 70
    else:
        base_equity = 80  # slightly higher if underfunded

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
    return pd.DataFrame(rows).iloc[: time_to_goal].reset_index(drop=True)

def build_portfolio_from_csvs(strategy: str, equity_pct: int) -> pd.DataFrame:
    """
    Select *real fund names* from your CSVs.
    Equity table is chosen by strategy (Active/Passive/Hybrid).
    Debt uses DURATION_DEBT.
    Weights are sized to equity_pct / (100 - equity_pct).
    """
    # ---- Equity side ----
    if strategy == "Active" and ACTIVE_EQ is not None and not ACTIVE_EQ.empty:
        eq_src = ACTIVE_EQ.copy()
        subcat = "Active"
    elif strategy == "Passive" and PASSIVE_EQ is not None and not PASSIVE_EQ.empty:
        eq_src = PASSIVE_EQ.copy()
        subcat = "Passive"
    elif HYBRID_EQ is not None and not HYBRID_EQ.empty:
        eq_src = HYBRID_EQ.copy()
        subcat = "Hybrid"
    else:
        eq_src = None
        subcat = "Blend"

    if eq_src is not None and not eq_src.empty:
        eq_pick = _trim_rank_and_scale(eq_src, total_pct=equity_pct, min_pct=10)
        eq_pick["Category"] = "Equity"
        eq_pick["Sub-category"] = subcat
    else:
        eq_pick = pd.DataFrame([{"Fund":"Equity Allocation (Basket)","Weight (%)":equity_pct,"Category":"Equity","Sub-category":"Blend"}])

    # ---- Debt side ----
    debt_pct = 100 - equity_pct
    if debt_pct > 0:
        if DURATION_DEBT is not None and not DURATION_DEBT.empty:
            dt_pick = _trim_rank_and_scale(DURATION_DEBT, total_pct=debt_pct, min_pct=10)
            dt_pick["Category"] = "Debt"
            dt_pick["Sub-category"] = "Duration"
        else:
            dt_pick = pd.DataFrame([{"Fund":"Debt Allocation (Basket)","Weight (%)":debt_pct,"Category":"Debt","Sub-category":"Duration"}])
        portfolio = pd.concat([eq_pick, dt_pick], ignore_index=True)
    else:
        portfolio = eq_pick

    # Final tidy/columns
    portfolio = portfolio[["Fund","Category","Sub-category","Weight (%)"]].copy()
    # Final rounding hygiene (sum == 100)
    s = int(portfolio["Weight (%)"].sum())
    if s != 100:
        idx = portfolio["Weight (%)"].idxmax()
        portfolio.at[idx, "Weight (%)"] = int(portfolio.at[idx, "Weight (%)"]) + (100 - s)
    return portfolio

def run_engine(user_input: PortfolioInput) -> Dict[str, Any]:
    """
    Full pipeline: funding -> strategy -> glide -> portfolio -> explainers.
    """
    # 1) Funding & Strategy
    fv, fr = calculate_funding_ratio(user_input.monthly_investment, user_input.target_corpus, user_input.years_to_goal)
    strategy = choose_strategy(user_input.years_to_goal, user_input.risk_profile, fr)

    # 2) Glide path (DF) capped to N
    glide_df = generate_step_down_glide_path(user_input.years_to_goal, fr, user_input.risk_profile)
    glide_records = glide_df.to_dict(orient="records")

    # 3) Portfolio from CSVs using Year-1 equity split
    y1_equity = int(glide_df.iloc[0]["Equity Allocation (%)"]) if not glide_df.empty else 60
    portfolio_df = build_portfolio_from_csvs(strategy, y1_equity)

    # 4) Explainers (pass proper shapes)
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
        "funding_ratio": float(fr),  # keep full precision internally; round on write/output
        "glide_path": glide_records,                                  # list[dict]
        "portfolio": portfolio_df.to_dict(orient="records"),          # list[dict]
        "glide_explainer": glide_block,
        "strategy_explainer": strategy_block,
        "portfolio_explainer": portfolio_block,
    }

# =================== Public endpoints ===================

class GenerateRequest(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str

@app.post("/generate_portfolio/")
def generate_portfolio(user_input: GenerateRequest):
    try:
        ui = PortfolioInput(
            monthly_investment=user_input.monthly_investment,
            target_corpus=user_input.target_corpus,
            years_to_goal=int(user_input.years_to_goal),
            risk_profile=user_input.risk_profile,
        )
        output = run_engine(ui)
        # Cap glide to N (safety)
        output["glide_path"] = output["glide_path"][: ui.years_to_goal]
        # Round funding ratio for API consumers too
        output["funding_ratio"] = round(float(output["funding_ratio"]), 2)
        return output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================== Airtable integration ===================

class AirtableWebhookPayload(BaseModel):
    record_id: str

AIRTABLE_API_KEY   = os.environ.get("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID   = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME= "Investor_inputs"

_api      = Api(AIRTABLE_API_KEY)
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

        # ---- Read EXACT Airtable fields you provided ----
        monthly_raw = fields.get("Monthly Investments")   # ✅ exact
        target_raw  = fields.get("Target Corpus")         # ✅ exact
        years_raw   = fields.get("Time to goal")          # ✅ exact (duration years)
        risk_raw    = fields.get("Risk Preference")       # ✅ exact

        if monthly_raw is None or target_raw is None or years_raw is None:
            raise HTTPException(
                status_code=422,
                detail="Missing required fields: 'Monthly Investments', 'Target Corpus', or 'Time to goal'."
            )

        monthly = _req_float("Monthly Investments", monthly_raw)
        target  = _req_float("Target Corpus", target_raw)
        years   = int(_req_int("Time to goal", years_raw))  # already a duration
        risk    = (risk_raw or "Moderate")

        user_input = PortfolioInput(
            monthly_investment=monthly,
            target_corpus=target,
            years_to_goal=years,
            risk_profile=risk
        )

        processed = run_engine(user_input)

        # ✅ Enforce exact N-year glide and compact
        N = years
        glide_verbose = processed["glide_path"][:N]
        compact_glide = compact_glide_verbose_to_array(glide_verbose)
        glide_json = minified_json(compact_glide)

        # ✅ Portfolio JSON (actual fund names)
        portfolio_json = minified_json(processed["portfolio"])

        # ✅ Round funding ratio to 2 decimals on write
        fr_2dp = round(float(processed["funding_ratio"]), 2)

        update_fields = {
            "glide_path":                 safe_store(glide_json),
            "portfolio":                  safe_store(portfolio_json),
            "glide_explainer_story":      processed["glide_explainer"]["story"],
            "strategy_explainer_story":   processed["strategy_explainer"]["story"],
            "portfolio_explainer_story":  processed["portfolio_explainer"]["story"],
            "funding_ratio":              fr_2dp,
            "strategy":                   processed["strategy"],
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
