# portfolio_explainer.py
from typing import Dict, Any, List
import pandas as pd

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
        # prefer explicit Weight
        df = df.drop(columns=["Weight (%)"])
    elif "Weight (%)" in df.columns:
        df = df.rename(columns={"Weight (%)": "Weight"})
    # Ensure Weight exists
    if "Weight" not in df.columns:
        raise ValueError("Portfolio is missing a 'Weight' or 'Weight (%)' column.")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    # Normalize Sub-category spelling
    if "Sub_category" in df.columns and "Sub-category" not in df.columns:
        df = df.rename(columns={"Sub_category": "Sub-category"})

    # Ensure required columns exist
    required = {"Fund", "Category", "Sub-category", "Weight"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Portfolio is missing required column(s): {', '.join(missing)}")

    # Tidy category string cases
    df["Category"] = df["Category"].astype(str).str.title()

    return df


# ===================== Detection Helpers =====================

def _is_index(subcat: str, name: str) -> bool:
    """
    Decide if an equity fund is Index/Passive using sub-category and name keywords.
    """
    s = (subcat or "").strip().lower()
    if s == "passive":
        return True
    n = (name or "").lower()
    # Include a leading space for " index" to avoid "indexation" false positives
    return any(k in n for k in [" index", "nifty", "sensex", " etf", "total market"])

def _equity_bucket(subcat: str, name: str) -> str:
    """
    Map to Large/Mid/Small (fallback to name keywords). Otherwise "Other Equity".
    """
    s = (subcat or "").strip().lower()
    if s in ["large cap", "large-cap", "large"]:
        return "Large Cap"
    if s in ["mid cap", "mid-cap", "mid"]:
        return "Mid Cap"
    if s in ["small cap", "small-cap", "small"]:
        return "Small Cap"

    # Name fallbacks
    n = (name or "").lower()
    if any(k in n for k in ["smallcap", " small cap"]):
        return "Small Cap"
    if any(k in n for k in ["midcap", " mid cap", "nifty next 50"]):
        return "Mid Cap"
    if any(k in n for k in ["largecap", " large cap", "bluechip", "nifty 50", "sensex"]):
        return "Large Cap"

    return "Other Equity"

def _debt_bucket(subcat: str, name: str) -> str:
    """
    Categorize debt to common subtypes (Liquid, Ultra Short, etc.), else 'Debt (Other)'.
    """
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
        if k in n:
            return label

    return "Debt (Other)"

def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - Type: 'Index' | 'Active' for Equity, 'Debt' for debt
      - Bucket: Large/Mid/Small or Debt subtype
    """
    df = df.copy()
    df["Type"] = ""
    df["Bucket"] = ""

    for i, r in df.iterrows():
        cat = (r.get("Category") or "").strip().title()
        sub = r.get("Sub-category")
        name = r.get("Fund")

        if cat == "Equity":
            df.at[i, "Type"] = "Index" if _is_index(sub, name) else "Active"
            df.at[i, "Bucket"] = _equity_bucket(sub, name)
        else:
            df.at[i, "Type"] = "Debt"
            df.at[i, "Bucket"] = _debt_bucket(sub, name)

    return df


# ===================== Sentence Builders =====================

def _equity_bucket_sentence(bucket: str, idx_pct: float, act_pct: float) -> str:
    """
    Hybrid narration: if both Index+Active exist in a bucket, merge into one sentence.
    """
    ip = int(round(idx_pct))
    ap = int(round(act_pct))
    tp = ip + ap

    if bucket == "Large Cap":
        if ip > 0 and ap > 0:
            return (f"Large-cap (~{tp}%) is split between **index funds** that cover the top 100 companies at low cost "
                    f"and **active funds** that try to beat that index with selective picks")
        if ip > 0:
            return f"Large-cap (~{ip}%) comes via **index funds** that cover the top 100 companies at low cost"
        if ap > 0:
            return f"Large-cap (~{ap}%) uses **active funds** aiming to beat the large-cap index while keeping risk moderate"

    if bucket == "Mid Cap":
        if ip > 0 and ap > 0:
            return (f"Mid-cap (~{tp}%) blends **index funds** on the next 150 mid-sized companies "
                    f"with **active funds** for extra growth")
        if ip > 0:
            return f"Mid-cap (~{ip}%) tracks an **index** of mid-sized companies—the portfolio’s growth engine"
        if ap > 0:
            return f"Mid-cap (~{ap}%) uses **active funds** in mid-sized companies to add punch to growth"

    if bucket == "Small Cap":
        if ip > 0 and ap > 0:
            return (f"Small-cap (~{tp}%) combines **index exposure** to 250 smaller companies "
                    f"with **active picks** for tactical opportunities")
        if ip > 0:
            return f"Small-cap (~{ip}%) adds **index exposure** to 250 smaller, early-stage companies"
        if ap > 0:
            return f"Small-cap (~{ap}%) is via **active funds** targeting emerging companies for higher growth"

    # Other equity styles collapsed to a simple sentence
    if tp > 0:
        if ap > 0 and ip > 0:
            return f"Other equity (~{tp}%) mixes **index** and **active** styles for diversification"
        return f"Other equity (~{tp}%) adds diversification"

    return ""  # nothing in this bucket

def _debt_sentence(total_debt_pct: float, debt_break: Dict[str, float]) -> str:
    td = int(round(total_debt_pct))
    mentions = []
    for key in ["Liquid", "Ultra Short", "Money Market", "Overnight", "Short Duration"]:
        v = int(round(debt_break.get(key, 0)))
        if v > 0:
            mentions.append(f"{key.lower()} (~{v}%)")
    if mentions:
        detail = ", ".join(mentions[:2])  # keep concise
        return f"Debt (~{td}%) **anchors** the plan for safety and liquidity, mostly in {detail}"
    return f"Debt (~{td}%) **anchors** the plan for safety and liquidity"


# ===================== Main Explainer =====================

def explain_portfolio_story(final_portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Input DF must include:
      ['Fund', 'Category', 'Sub-category', 'Weight']  or  ['...','Weight (%)']
    Returns:
      {
        "block_id": "block_3_portfolio",
        "story": "<coherent layman paragraph>",
        "data_points": { ... }   # useful aggregates for chips/labels
      }
    """
    # Normalize & annotate
    df = _normalize_columns(final_portfolio)
    df = _annotate(df)

    # Equity vs Debt split
    by_cat = df.groupby("Category")["Weight"].sum().to_dict()
    eq_total = float(by_cat.get("Equity", 0.0))
    debt_total = float(by_cat.get("Debt", 0.0))
    fund_count = len(df)

    # Equity: bucket × type weights
    eq = df[df["Category"] == "Equity"].copy()
    if not eq.empty:
        bucket_type = eq.groupby(["Bucket", "Type"])["Weight"].sum().reset_index()
    else:
        bucket_type = pd.DataFrame(columns=["Bucket", "Type", "Weight"])

    # Summarize into {bucket: {"Index": pct, "Active": pct, "Total": pct}}
    buckets: Dict[str, Dict[str, float]] = {}
    for _, r in bucket_type.iterrows():
        b = str(r["Bucket"])
        t = str(r["Type"])
        buckets.setdefault(b, {"Index": 0.0, "Active": 0.0, "Total": 0.0})
        buckets[b][t] += float(r["Weight"])
        buckets[b]["Total"] += float(r["Weight"])

    # Debt breakdown by subtype
    debt = df[df["Category"] == "Debt"].copy()
    debt_break = debt.groupby("Bucket")["Weight"].sum().to_dict() if not debt.empty else {}

    # Build concise story (≤3 sentences typical)
    parts: List[str] = []

    # Opening: overall split + fund count
    if eq_total + debt_total > 0:
        parts.append(
            f"Your portfolio is diversified across **~{int(round(eq_total))}% equity** and **~{int(round(debt_total))}% debt**, "
            f"spread over {fund_count} funds so each allocation has meaningful weight."
        )

    # Equity sentences in order: Large, Mid, Small; then any Other Equity
    for key in ["Large Cap", "Mid Cap", "Small Cap"]:
        if key in buckets and buckets[key]["Total"] > 0:
            s = _equity_bucket_sentence(key, buckets[key].get("Index", 0.0), buckets[key].get("Active", 0.0))
            if s:
                parts.append(s + ".")

    others_total = sum(v["Total"] for k, v in buckets.items() if k not in ["Large Cap", "Mid Cap", "Small Cap"])
    if others_total > 0:
        parts.append(_equity_bucket_sentence("Other Equity", 0.0, others_total) + ".")

    # Debt sentence (unified)
    if debt_total > 0:
        parts.append(_debt_sentence(debt_total, debt_break) + ".")

    # Close with differentiator
    parts.append("Each fund is chosen using our proprietary quantitative filter across time horizons and metrics, so every pick has a clear role.")

    story = " ".join(parts)

    # datapoints for optional UI chips
    data_points = {
        "equity_total_pct": int(round(eq_total)),
        "debt_total_pct": int(round(debt_total)),
        "fund_count": fund_count,
        "equity_buckets": {k: {kk: int(round(vv)) for kk, vv in v.items()} for k, v in buckets.items()},
        "debt_breakdown": {k: int(round(v)) for k, v in debt_break.items()},
    }

    return {
        "block_id": "block_3_portfolio",
        "story": story,
        "data_points": data_points
    }
