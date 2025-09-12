# portfolio_explainer.py
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
def explain_portfolio_story(final_portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Uses actual weights; speaks in plain English.
    If common Indian indices are detected, we call them out by name.
    """
    df = _normalize(final_portfolio)

    equity_total = int(round(_sum(df, "Equity")))
    debt_total   = int(round(_sum(df, "Debt")))
    fund_count   = int(df.shape[0])

    # Detect common buckets by fund name (so wording matches exactly)
    large_core   = int(round(_pick(df, "nifty 50", "sensex")))
    next50       = int(round(_pick(df, "nifty next 50")))
    mid_engine   = int(round(_pick(df, "midcap 150", "mid cap 150")))
    small_kicker = int(round(_pick(df, "smallcap 250", "small cap 250")))

    debt_liquid  = int(round(_pick(df[df["Category"].str.lower()=="debt"], "liquid")))
    debt_ultra   = int(round(_pick(df[df["Category"].str.lower()=="debt"], "ultra short", "ultra-short")))

    # Build a faithful, concise story
    parts: List[str] = []
    parts.append(f"Your portfolio holds **~{equity_total}% equity** and **~{debt_total}% debt** across {fund_count} funds.")

    # Equity details (only mention chunks that exist)
    details: List[str] = []
    if large_core:   details.append(f"Large-cap core via **Nifty 50 (~{large_core}%)**")
    if next50:       details.append(f"**Nifty Next 50 (~{next50}%)** adds large-cap challengers")
    if mid_engine:   details.append(f"Mid-cap engine via **Midcap 150 (~{mid_engine}%)**")
    if small_kicker: details.append(f"Small-cap kicker via **Smallcap 250 (~{small_kicker}%)**")
    if details:
        parts.append("; ".join(details) + ".")

    # Debt line
    if debt_total:
        debt_bits = []
        if debt_liquid: debt_bits.append(f"liquid (~{debt_liquid}%)")
        if debt_ultra:  debt_bits.append(f"ultra-short (~{debt_ultra}%)")
        if debt_bits:
            parts.append("Debt anchors stability and access to cash via " + ", ".join(debt_bits) + ".")
        else:
            parts.append("Debt anchors stability and access to cash.")

    parts.append("Weights are kept meaningful (≥10%) so each position pulls its weight.")
    story = " ".join(parts)

    # Data points (for chips/labels)
    data_points = {
        "equity_total_pct": equity_total,
        "debt_total_pct": debt_total,
        "fund_count": fund_count,
        "by_index_family": {
            "Nifty 50": large_core,
            "Nifty Next 50": next50,
            "Midcap 150": mid_engine,
            "Smallcap 250": small_kicker,
        },
        "debt_breakdown": {
            "Liquid": debt_liquid,
            "Ultra Short": debt_ultra,
        }
    }

    return {"block_id": "block_3_portfolio", "story": story, "data_points": data_points}
