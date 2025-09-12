# portfolio_explainer.py
from typing import Dict, Any, List
import pandas as pd
import re

# ===================== Column Normalizers =====================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify column names and ensure numeric weights.
    Accepts:
      - Weight (%) OR Weight  -> Weight
      - Sub-category OR Sub_category -> Sub-category
    """
    df = df.copy()

    # Normalize "Weight"
    if "Weight" in df.columns and "Weight (%)" in df.columns:
        df = df.drop(columns=["Weight (%)"])
    elif "Weight (%)" in df.columns:
        df = df.rename(columns={"Weight (%)": "Weight"})
    if "Weight" not in df.columns:
        raise ValueError("Portfolio is missing a 'Weight' or 'Weight (%)' column.")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    # Normalize Sub-category spelling
    if "Sub_category" in df.columns and "Sub-category" not in df.columns:
        df = df.rename(columns={"Sub_category": "Sub-category"})

    required = {"Fund", "Category", "Sub-category", "Weight"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Portfolio is missing required column(s): {', '.join(missing)}")

    df["Category"] = df["Category"].astype(str).str.title()
    return df

# ===================== Detection Helpers =====================

def _is_index(subcat: str, name: str) -> bool:
    s = (subcat or "").strip().lower()
    if s == "passive":
        return True
    n = (name or "").lower()
    return any(k in n for k in [" index", "nifty", "sensex", " etf", "total market"])

def _equity_bucket(subcat: str, name: str) -> str:
    s = (subcat or "").strip().lower()
    if s in ["large cap", "large-cap", "large"]: return "Large Cap"
    if s in ["mid cap", "mid-cap", "mid"]:       return "Mid Cap"
    if s in ["small cap", "small-cap", "small"]: return "Small Cap"

    n = (name or "").lower()
    if any(k in n for k in ["smallcap", " small cap"]):                  return "Small Cap"
    if any(k in n for k in ["midcap", " mid cap", "nifty next 50"]):     return "Mid Cap"
    if any(k in n for k in ["largecap", " large cap", "bluechip", "nifty 50", "sensex"]): return "Large Cap"
    return "Other Equity"

def _debt_bucket(subcat: str, name: str) -> str:
    s = (subcat or "").strip().lower()
    if s in ["liquid", "ultra short", "ultra-short", "money market", "overnight", "short duration"]:
        return s.title()
    n = (name or "").lower()
    for k, label in [
        ("liquid", "Liquid"),
        ("ultra short", "Ultra Short"),
        ("money market", "Money Market"),
        ("overnight", "Overnight"),
        ("short duration", "Short Duration"),
    ]:
        if k in n: return label
    return "Debt (Other)"

def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Type"] = ""
    df["Bucket"] = ""
    for i, r in df.iterrows():
        cat = (r.get("Category") or "").strip().title()
        sub = r.get("Sub-category")
        name = r.get("Fund")
        if cat == "Equity":
            df.at[i, "Type"]   = "Index" if _is_index(sub, name) else "Active"
            df.at[i, "Bucket"] = _equity_bucket(sub, name)
        else:
            df.at[i, "Type"]   = "Debt"
            df.at[i, "Bucket"] = _debt_bucket(sub, name)
    return df

# ===================== Sentence Builders =====================

def _equity_bucket_sentence(bucket: str, idx_pct: float, act_pct: float) -> str:
    ip = int(round(idx_pct)); ap = int(round(act_pct)); tp = ip + ap
    if bucket == "Large Cap":
        if ip and ap: return (f"Large-cap (~{tp}%) splits between **index funds** covering the top 100 at low cost "
                              f"and **active funds** seeking selective alpha")
        if ip:        return  f"Large-cap (~{ip}%) comes via **index funds** covering the top 100 at low cost"
        if ap:        return  f"Large-cap (~{ap}%) uses **active funds** aiming to beat the large-cap index"
    if bucket == "Mid Cap":
        if ip and ap: return (f"Mid-cap (~{tp}%) blends **index funds** on the next 150 companies with **active** ideas")
        if ip:        return  f"Mid-cap (~{ip}%) tracks an **index** of mid-sized companies—the growth engine"
        if ap:        return  f"Mid-cap (~{ap}%) uses **active funds** in mid-sized companies for added punch"
    if bucket == "Small Cap":
        if ip and ap: return (f"Small-cap (~{tp}%) combines **index exposure** to 250 smaller companies with **active** picks")
        if ip:        return  f"Small-cap (~{ip}%) adds **index exposure** to smaller, early-stage companies"
        if ap:        return  f"Small-cap (~{ap}%) uses **active funds** for higher-beta growth"
    if tp:             return  f"Other equity (~{tp}%) adds diversification"
    return ""

def _debt_sentence(total_debt_pct: float, debt_break: Dict[str, float]) -> str:
    td = int(round(total_debt_pct))
    mentions = []
    for key in ["Liquid", "Ultra Short", "Money Market", "Overnight", "Short Duration"]:
        v = int(round(debt_break.get(key, 0)))
        if v > 0: mentions.append(f"{key.lower()} (~{v}%)")
    detail = ", ".join(mentions[:2]) if mentions else None
    return (f"Debt (~{td}%) anchors stability and liquidity"
            + (f", mostly in {detail}" if detail else ""))

# ===================== Main Explainer =====================

def explain_portfolio_story(final_portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Input DF must include:
      ['Fund', 'Category', 'Sub-category', 'Weight']  or  ['...','Weight (%)']
    """
    df = _normalize_columns(final_portfolio)
    df = _annotate(df)

    by_cat = df.groupby("Category")["Weight"].sum().to_dict()
    eq_total   = float(by_cat.get("Equity", 0.0))
    debt_total = float(by_cat.get("Debt",   0.0))
    fund_count = int(len(df))

    eq = df[df["Category"] == "Equity"].copy()
    bucket_type = eq.groupby(["Bucket", "Type"])["Weight"].sum().reset_index() if not eq.empty else pd.DataFrame(columns=["Bucket","Type","Weight"])

    buckets: Dict[str, Dict[str, float]] = {}
    for _, r in bucket_type.iterrows():
        b = str(r["Bucket"]); t = str(r["Type"]); w = float(r["Weight"])
        buckets.setdefault(b, {"Index": 0.0, "Active": 0.0, "Total": 0.0})
        buckets[b][t]   += w
        buckets[b]["Total"] += w

    debt = df[df["Category"] == "Debt"].copy()
    debt_break = debt.groupby("Bucket")["Weight"].sum().to_dict() if not debt.empty else {}

    parts: List[str] = []
    if eq_total + debt_total > 0:
        parts.append(
            f"Your portfolio holds **~{int(round(eq_total))}% equity** and **~{int(round(debt_total))}% debt** across {fund_count} funds."
        )

    for key in ["Large Cap", "Mid Cap", "Small Cap"]:
        if key in buckets and buckets[key]["Total"] > 0:
            s = _equity_bucket_sentence(key, buckets[key].get("Index", 0.0), buckets[key].get("Active", 0.0))
            if s: parts.append(s + ".")

    others_total = sum(v["Total"] for k, v in buckets.items() if k not in ["Large Cap", "Mid Cap", "Small Cap"])
    if others_total > 0:
        parts.append(_equity_bucket_sentence("Other Equity", 0.0, others_total) + ".")

    if debt_total > 0:
        parts.append(_debt_sentence(debt_total, debt_break) + ".")

    parts.append("Weights stay meaningful (≥10%) so each position contributes visibly to outcomes.")
    story = " ".join(parts)

    data_points = {
        "equity_total_pct": int(round(eq_total)),
        "debt_total_pct":   int(round(debt_total)),
        "fund_count":       fund_count,
        "equity_buckets":   {k: {kk: int(round(vv)) for kk, vv in v.items()} for k, v in buckets.items()},
        "debt_breakdown":   {k: int(round(v)) for k, v in debt_break.items()},
    }

    return {"block_id": "block_3_portfolio", "story": story, "data_points": data_points}
