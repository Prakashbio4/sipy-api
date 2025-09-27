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

# === Explainers (stories layer) — existing ===
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
from portfolio_explainer import explain_portfolio_story

__API_BUILD__ = "final-fix-current-corpus-4part-002"
app = FastAPI(title="SIPY Investment Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------------------- Helpers (parsing & math) ---------------------
def _to_years_from_any(x, default=0):
    """Parse an integer year-count from messy strings like 'in 15 years', '54 (user is currently 37)', or 15."""
    if x is None:
        return default
    if isinstance(x, int):
        return x
    s = str(x)
    nums = re.findall(r"\d+", s)
    return int(nums[0]) if nums else default

def _to_float(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("₹", "").strip()
    try:
        return float(s)
    except:
        m = re.findall(r"[-+]?\d*\.?\d+", s)
        return float(m[0]) if m else default

def expected_return_from_profile(years_to_goal: int, risk_profile: str) -> float:
    """
    Return ANNUAL expected return (decimal) based on time-to-goal & risk profile.
    More conservative for shorter horizons; differentiates by risk.
    """
    rp = (risk_profile or "").strip().lower()
    if years_to_goal <= 3:
        base = {"conservative": 0.06, "moderate": 0.075, "aggressive": 0.085}
    elif years_to_goal <= 5:
        base = {"conservative": 0.07, "moderate": 0.09,  "aggressive": 0.105}
    elif years_to_goal <= 10:
        base = {"conservative": 0.08, "moderate": 0.10,  "aggressive": 0.12}
    else:
        base = {"conservative": 0.09, "moderate": 0.11,  "aggressive": 0.13}
    return base.get(rp, 0.10)  # default 10% if unknown

# ----------- Load CSVs once at app startup ------------
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

# -------------------- Pydantic Input --------------------
class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str  # 'conservative', 'moderate', 'aggressive'
    current_corpus: Optional[float] = 0.0  # include current corpus

# ----------------- Engine Utility Functions -----------------
def calculate_funding_ratio(monthly_investment: float,
                            current_corpus: float,
                            target_corpus: float,
                            years: int,
                            annual_expected_return: float):
    """
    FV_total = FV(current_corpus) + FV(SIP)
    """
    years = max(0, int(years))
    r_m = (1 + annual_expected_return) ** (1/12) - 1
    n = years * 12

    fv_lumpsum = current_corpus * ((1 + annual_expected_return) ** years)
    fv_sip = monthly_investment * (((1 + r_m) ** n - 1) / r_m) if r_m != 0 else monthly_investment * n
    fv_total = fv_lumpsum + fv_sip
    funding_ratio = (fv_total / target_corpus) if target_corpus else 0.0
    return fv_total, funding_ratio

def generate_step_down_glide_path(time_to_goal, funding_ratio, risk_profile):
    funding_ratio = float(funding_ratio)
    glide_path = []
    short_goal_equity_cap = 30 if time_to_goal <= 3 else 100

    if funding_ratio > 2.0:
        base_equity = 70; derisk_start = int(time_to_goal * 0.4)
    elif funding_ratio > 1.0:
        base_equity = 80; derisk_start = int(time_to_goal * 0.5)
    else:
        base_equity = 90; derisk_start = int(time_to_goal * 0.6)

    rp = (risk_profile or "").strip().lower()
    if rp == 'conservative': base_equity -= 10
    elif rp == 'aggressive': base_equity += 5

    base_equity = min(base_equity, short_goal_equity_cap)
    final_equity = 10
    derisk_years = time_to_goal - derisk_start
    equity_step = (base_equity - final_equity) / derisk_years if derisk_years > 0 else 0

    for year in range(1, time_to_goal + 1):
        equity = base_equity if year <= derisk_start else base_equity - equity_step * (year - derisk_start)
        equity = int(round(max(min(equity, short_goal_equity_cap), 0) / 5) * 5)
        if 0 < equity < 10: equity = 10
        debt = 100 - equity
        if 0 < debt < 10: debt = 10; equity = 100 - debt
        glide_path.append({'Year': year, 'Equity Allocation (%)': int(equity), 'Debt Allocation (%)': int(debt)})
    return pd.DataFrame(glide_path)

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

# ---------- Min-10% helpers ----------
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
    if s == 0: raise ValueError("Zero weights after min-enforcement.")
    df['Weight (%)'] = df['Weight (%)'] / s * total_pct
    df['Weight (%)'] = df['Weight (%)'].apply(lambda x: round(x / step) * step)
    diff = total_pct - df['Weight (%)'].sum()
    if diff != 0:
        idx = df['Weight (%)'].idxmax()
        df.at[idx, 'Weight (%)'] += diff
    return df

def standardize_fund_col(df):
    if 'Fund' in df.columns: return df
    cand = [c for c in df.columns if 'fund' in c.lower() or 'scheme' in c.lower() or 'name' in c.lower()]
    if cand: return df.rename(columns={cand[0]: 'Fund'})
    df = df.copy(); df['Fund'] = df.index.astype(str); return df

def get_debt_allocation(duration_debt_df, time_to_goal, debt_pct):
    duration_debt_df = duration_debt_df.copy()
    allowed = ['Liquid'] if time_to_goal > 3 else ['Liquid', 'Ultra Short']
    debt_filtered = duration_debt_df[duration_debt_df['Sub-category'].str.strip().isin(allowed)].copy()
    if debt_filtered.empty: raise ValueError("⚠️ No suitable debt funds found for the selected duration.")
    debt_filtered = standardize_fund_col(debt_filtered)
    debt_filtered['Category'] = 'Debt'
    debt_filtered = trim_funds_to_min(debt_filtered, debt_pct, min_pct=10,
                                      sort_col='Weight' if 'Weight' in debt_filtered.columns else None)
    debt_filtered['Weight (%)'] = debt_pct / len(debt_filtered)
    return debt_filtered[['Fund', 'Category', 'Sub-category', 'Weight (%)']]

# ---------- NEW: Short-horizon equity filter (large-cap bias) ----------
def _cap_bucket_from_name(name: str) -> str:
    n = (name or "").lower()
    if re.search(r"(nifty\s*50|sensex|large\s*cap|blue\s*chip|bluechip|top\s*100|nifty\s*100|s&p\s*bse\s*100)", n):
        return "large"
    if re.search(r"(mid\s*cap|midcap|nifty\s*midcap)", n):
        return "mid"
    if re.search(r"(small\s*cap|smallcap|nifty\s*smallcap)", n):
        return "small"
    return "other"

def filter_equity_by_horizon(eq_df: pd.DataFrame, years_to_goal: int) -> pd.DataFrame:
    if years_to_goal > 3:
        return eq_df
    df = eq_df.copy()
    if 'Fund' not in df.columns:
        cands = [c for c in df.columns if any(k in c.lower() for k in ['fund', 'scheme', 'name'])]
        if cands:
            df = df.rename(columns={cands[0]: 'Fund'})
        else:
            df['Fund'] = df.index.astype(str)
    df['__cap_bucket__'] = df['Fund'].map(_cap_bucket_from_name)
    large_only = df[df['__cap_bucket__'] == 'large'].copy()
    if not large_only.empty:
        return large_only.drop(columns='__cap_bucket__')
    if 'Sub-category' in df.columns:
        broad = df[df['Sub-category'].str.contains(r"(large|blue)", case=False, na=False)].copy()
        if not broad.empty:
            return broad.drop(columns='__cap_bucket__')
    return df.drop(columns='__cap_bucket__')

# ============= 4-part explainer helpers =============
def _pct(val) -> int:
    try: return max(0, min(100, int(round(float(val)))))
    except: return 0

def _money(val) -> str:
    try: v = float(val)
    except: return str(val)
    if v >= 1e7: return f"₹{v/1e7:.2f}Cr"
    if v >= 1e5: return f"₹{v/1e5:.2f}L"
    return f"₹{v:,.0f}"

def _range_pm5(center_pct: float) -> str:
    c = _pct(center_pct); lo = max(0, c - 5); hi = min(100, c + 5)
    return f"{lo}–{hi}%"

def _bucket_from_fund_name(name: str) -> str:
    n = (name or "").lower()
    if "smallcap" in n or "small cap" in n: return "small"
    if "midcap" in n or "mid cap" in n: return "mid"
    if "nifty 50" in n or "nifty50" in n or "next 50" in n or "nifty next 50" in n or "bluechip" in n or "large" in n:
        return "large"
    return "other"

def aggregate_portfolio_buckets(portfolio_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    large = mid = small = debt = other = 0.0
    for row in portfolio_rows or []:
        try: w = float(row.get("Weight (%)", 0))
        except: w = 0.0
        cat = (row.get("Category") or "").lower(); fund = row.get("Fund") or ""
        if "debt" in cat or "liquid" in (row.get("Sub-category") or "").lower(): debt += w
        elif "equity" in cat:
            b = _bucket_from_fund_name(fund)
            if b == "large": large += w
            elif b == "mid": mid += w
            elif b == "small": small += w
            else: other += w
    total = large + mid + small + debt + other
    if total > 0:
        scale = 100.0 / total
        large, mid, small, debt, other = [round(x * scale) for x in (large, mid, small, debt, other)]
    return {"large_cap_pct": _pct(large), "mid_cap_pct": _pct(mid), "small_cap_pct": _pct(small), "debt_pct": _pct(debt)}

def _strategy_sentence(strategy: str, risk_profile: Optional[str]) -> str:
    s = (strategy or "").strip().title() or "Active"
    rp = (risk_profile or "").strip().lower()
    if s == "Passive":
        if rp.startswith("conserv"):
            return ("Given your conservative preference, we recommend a Passive strategy — "
                    "low-cost index funds that mirror the market. It keeps costs low and reduces surprises "
                    "while you stay invested for growth.")
        return ("We recommend a Passive strategy — low-cost index funds that mirror the market, "
                "keeping things simple and cost-efficient while you stay invested.")
    if s == "Hybrid":
        return ("To balance growth and steadiness, we recommend a Hybrid strategy — "
                "a mix of index funds and focused active ideas.")
    if rp.startswith("conserv"):
        return ("We recommend an Active strategy with care — expert-managed funds aiming to add value, "
                "used thoughtfully given your conservative preference.")
    return ("To help close the gap faster (with more ups and downs), we recommend an Active strategy — "
            "expert-managed funds that aim to add value over the market.")

def generate_four_part_explainer(
    name: str,
    target_corpus: Any,
    funding_ratio_pct_center: float,
    strategy: str,
    risk_profile: Optional[str],
    year1_equity_pct: int,
    portfolio_rows: List[Dict[str, Any]],
    years_to_goal: int,
) -> Dict[str, str]:
    target_str = _money(target_corpus)
    fr_display = _range_pm5(funding_ratio_pct_center)

    var1 = (
        f"Hi {name}, here’s where you stand. If you continue as you are, "
        f"you’ll likely reach only {fr_display} of your goal of {target_str}.\n\n"
        "Why the range? Because markets don’t return the same number every time. "
        "Over the past 5 years, equities have averaged between 10–15% per year. "
        "That’s what we use to calculate your funding gap."
    )

    if years_to_goal <= 3:
        debt_pct = max(0, 100 - int(year1_equity_pct))
        var2 = (
            "With your goal so close, protecting your savings matters more than chasing returns. "
            "We recommend a conservative, capital-first approach — mostly debt, with a small equity slice for gentle growth."
        )
        var3 = (
            f"In Year 1, your money starts with {int(year1_equity_pct)}% in equities and {debt_pct}% in debt. "
            "This mix is designed to keep your corpus safe while still giving it a small boost before your goal."
        )
    else:
        buckets = aggregate_portfolio_buckets(portfolio_rows)
        large = buckets.get("large_cap_pct", 0)
        mid   = buckets.get("mid_cap_pct", 0)
        small = buckets.get("small_cap_pct", 0)
        debt  = buckets.get("debt_pct", 0)

        var2 = _strategy_sentence(strategy, risk_profile)

        parts = []
        if large > 0: parts.append(f"{large}% in large companies for stability")
        if mid   > 0: parts.append(f"{mid}% in mid-sized companies for growth")
        if small > 0: parts.append(f"{small}% in smaller companies for extra growth potential")
        split_sentence = "; ".join(parts) if parts else "equities"

        var3 = (
            f"In Year 1, your money starts with {int(year1_equity_pct)}% in equities — "
            f"split into {split_sentence}. The rest is in debt ({debt}%) for balance, "
            "and over time more will shift into debt to protect your savings."
        )

    var4 = (
        "This isn’t guesswork. It’s a disciplined plan built only for you. "
        "And we’ll review and rebalance regularly to keep you on track."
    )

    return {"explainer 1": var1, "explainer 2": var2, "explainer 3": var3, "explainer 4": var4}

# ================= API Endpoint =================
@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    try:
        annual_er = expected_return_from_profile(user_input.years_to_goal, user_input.risk_profile)

        fv, funding_ratio = calculate_funding_ratio(
            monthly_investment=user_input.monthly_investment,
            current_corpus=user_input.current_corpus or 0.0,
            target_corpus=user_input.target_corpus,
            years=user_input.years_to_goal,
            annual_expected_return=annual_er
        )

        strategy = choose_strategy(user_input.years_to_goal, user_input.risk_profile, funding_ratio)
        glide_path = generate_step_down_glide_path(user_input.years_to_goal, funding_ratio, user_input.risk_profile)

        glide_equity_pct = int(glide_path.iloc[0]['Equity Allocation (%)'])
        if 0 < glide_equity_pct < 10: glide_equity_pct = 10
        glide_debt_pct = 100 - glide_equity_pct
        if 0 < glide_debt_pct < 10: glide_debt_pct = 10; glide_equity_pct = 100 - glide_debt_pct

        # 2) Equity selection (by chosen strategy)
        if strategy == "Active":
            eq_df = active_eq.copy(); eq_df['Sub-category'] = 'Active'
        elif strategy == "Passive":
            eq_df = passive_eq.copy(); eq_df['Sub-category'] = 'Passive'
        elif strategy == "Hybrid":
            eq_df = hybrid_eq.copy(); eq_df = eq_df.rename(columns={'Type': 'Sub-category'})
        else:
            raise ValueError("Invalid strategy")

        eq_df = standardize_fund_col(eq_df)

        # ✅ short-horizon: keep only large-cap style equity funds
        eq_df = filter_equity_by_horizon(eq_df, user_input.years_to_goal)

        eq_df['Category'] = 'Equity'
        if 'Weight' not in eq_df.columns: eq_df['Weight'] = 1.0

        eq_df = trim_funds_to_min(eq_df, glide_equity_pct, min_pct=10, sort_col='Weight')
        eq_df['Weight'] = eq_df['Weight'] / eq_df['Weight'].sum()
        eq_df['Weight (%)'] = eq_df['Weight'] * glide_equity_pct
        eq_df = enforce_min_allocation(eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']],
                                       glide_equity_pct, min_pct=10, step=5)

        # 3) Debt selection
        selected_debt = get_debt_allocation(duration_debt, user_input.years_to_goal, glide_debt_pct)
        selected_debt = enforce_min_allocation(selected_debt, glide_debt_pct, min_pct=10, step=5)

        # 4) Combine & enforce whole portfolio
        combined = pd.concat([
            eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']],
            selected_debt[['Fund', 'Category', 'Sub-category', 'Weight (%)']]
        ], ignore_index=True)
        final_portfolio = enforce_min_allocation(combined, 100, min_pct=10, step=5)

        if (final_portfolio['Weight (%)'] < 10).any():
            raise ValueError("Found a fund below 10% after enforcement.")
        if final_portfolio['Weight (%)'].sum() != 100:
            raise ValueError("Final portfolio does not sum to 100%.")

        glide_context = {
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(funding_ratio),
            "glide_path": glide_path.to_dict(orient="records")
        }
        glide_block = explain_glide_story(glide_context)

        strategy_context = {
            "strategy": strategy,
            "years_to_goal": user_input.years_to_goal,
            "risk_profile": user_input.risk_profile,
            "funding_ratio": float(funding_ratio)
        }
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
            "monthly_investment": _to_float(inputs.get("Monthly Investments"), 0.0),
            "target_corpus": _to_float(inputs.get("Target Corpus"), 0.0),
            "years_to_goal": _to_years_from_any(inputs.get("Time to goal", inputs.get("Time Horizon (years)")), 0),
            "risk_profile": inputs.get("Risk Preference") or "",
            "current_corpus": _to_float(inputs.get("Current Corpus"), 0.0),
        }

        processed_output = generate_portfolio(PortfolioInput(**user_input_data))

        formatted_glide_path = ", ".join(
            [
                f"{{{row['Year']}, {row['Equity Allocation (%)']}, {row['Debt Allocation (%)']}}}"
                for row in processed_output["glide_path"]
            ]
        )

        # === New 4-part explainer generation ===
        name = inputs.get("User Name") or inputs.get("Name") or "Investor"
        target_corpus = inputs.get("Target Corpus")

        try:
            fr_center_pct = float(processed_output.get("funding_ratio", 0)) * 100.0
        except:
            fr_center_pct = 0.0

        try:
            y1_equity = int(processed_output["glide_path"][0]['Equity Allocation (%)'])
        except Exception:
            y1_equity = 0

        years_to_goal_raw = inputs.get("Time to goal") or inputs.get("Time Horizon (years)")
        years_to_goal = _to_years_from_any(years_to_goal_raw)

        four_part = generate_four_part_explainer(
            name=name,
            target_corpus=target_corpus,
            funding_ratio_pct_center=fr_center_pct,
            strategy=processed_output.get("strategy"),
            risk_profile=inputs.get("Risk Preference"),
            year1_equity_pct=y1_equity,
            portfolio_rows=processed_output.get("portfolio") or [],
            years_to_goal=years_to_goal,
        )

        try:
            fr_val = float(processed_output.get("funding_ratio", 0.0))
        except:
            fr_val = 0.0

        update_data = {
            "strategy": processed_output["strategy"],
            "funding_ratio": round(fr_val, 2),  # ratio form, 2dp
            "glide_path": formatted_glide_path,
            "portfolio": json.dumps(processed_output["portfolio"]),
            "glide_explainer_story": processed_output["glide_explainer"]["story"],
            "strategy_explainer_story": processed_output["strategy_explainer"]["story"],
            "portfolio_explainer_story": processed_output["portfolio_explainer"]["story"],
            "explainer 1": four_part["explainer 1"],
            "explainer 2": four_part["explainer 2"],
            "explainer 3": four_part["explainer 3"],
            "explainer 4": four_part["explainer 4"],
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
