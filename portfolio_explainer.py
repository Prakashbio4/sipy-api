# ===================== portfolio_explainer.py =====================
from typing import Dict, Any, List
import pandas as pd
import re

# ---------- basic normalizer ----------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Weight (%)" in df.columns and "Weight" not in df.columns:
        df = df.rename(columns={"Weight (%)": "Weight"})
    need = {"Fund", "Category", "Sub-category", "Weight"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Portfolio explainer: missing columns: {', '.join(miss)}")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
    return df

# ---------- pattern helpers ----------
def _has(text: str, *keys: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keys)

def _pick(df: pd.DataFrame, *keys: str) -> float:
    mask = df["Fund"].str.lower().apply(lambda x: any(k in x for k in keys))
    return float(df.loc[mask, "Weight"].sum())

def _sum(df: pd.DataFrame, cat: str) -> float:
    return float(df.loc[df["Category"].str.lower() == cat.lower(), "Weight"].sum())

# ---------- main ----------
def explain_portfolio_story(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a layman, ≤3-sentence story about the portfolio composition.
    """
    df = _normalize(df)

    if df.empty:
        return {
            "block_id": "block_3_portfolio",
            "story": "No portfolio data available to explain.",
            "data_points": {}
        }

    fund_count = len(df)
    equity = df[df["Category"].str.lower() == "equity"]
    debt = df[df["Category"].str.lower() == "debt"]

    eq_total = _sum(df, "equity")
    debt_total = _sum(df, "debt")

    # Equity bucket weights
    large_cap_total = _pick(equity, "nifty 50", "nifty next 50")
    mid_cap_total = _pick(equity, "midcap 150")
    small_cap_total = _pick(equity, "smallcap 250")

    # Debt bucket weights
    debt_liquid = _pick(debt, "liquid")
    debt_ultra = _pick(debt, "ultra-short")

    parts: List[str] = []

    # 1. Start with the high-level split and fund count
    if eq_total + debt_total > 0:
        parts.append(
            f"Your portfolio holds **~{int(round(eq_total))}% equity** and **~{int(round(debt_total))}% debt** across {fund_count} funds."
        )

    # 2. Break down the equity portion by role
    equity_details: List[str] = []
    if large_cap_total > 0:
        equity_details.append(f"a core position in **Large Cap (~{int(large_cap_total)}%)**, which invests in the top 100 companies in the country for stability and long-term growth")
    if mid_cap_total > 0:
        equity_details.append(f"a **Midcap engine (~{int(mid_cap_total)}%)**, as these funds are the growth engine of the portfolio and can deliver strong returns")
    if small_cap_total > 0:
        equity_details.append(f"a **Smallcap kicker (~{int(small_cap_total)}%)** to give your portfolio an extra upside")

    if equity_details:
        equity_sentence = "For your equity portion, you have " + ", and ".join(equity_details) + "."
        parts.append(equity_sentence)
    
    # 3. Explain the debt portion
    if debt_total > 0:
        debt_purpose = "providing stability and easy access to cash"
        if debt_liquid > 0:
            parts.append(f"Your debt portion is anchored in **liquid funds (~{int(debt_liquid)}%)**, {debt_purpose}.")
        else:
            parts.append(f"Your debt portion is designed for {debt_purpose}.")

    # 4. Conclude with a powerful, reassuring statement
    parts.append("Each fund is selected by SIPY's investment engine with one purpose: what’s best for you.")
    
    story = " ".join(parts)
    
    data_points = {
        "equity_total_pct": int(round(eq_total)),
        "debt_total_pct": int(round(debt_total)),
        "fund_count": fund_count,
        "equity_funds": {
            "Large Cap": int(large_cap_total),
            "Midcap": int(mid_cap_total),
            "Smallcap": int(small_cap_total),
        },
        "debt_funds": {
            "liquid": int(debt_liquid),
            "ultra-short": int(debt_ultra),
        },
    }

    return {
        "block_id": "block_3_portfolio",
        "story": story,
        "data_points": data_points
    }