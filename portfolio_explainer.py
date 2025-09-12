# portfolio_explainer.py
from typing import Dict, Any, List
import pandas as pd

# ---------- detection helpers ----------
def _is_index(subcat: str, name: str) -> bool:
    s = (subcat or "").strip().lower()
    if s == "passive":
        return True
    n = (name or "").lower()
    return any(k in n for k in [" index", "nifty", "sensex", " etf", "total market"])

def _equity_bucket(subcat: str, name: str) -> str:
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
    s = (subcat or "").strip().lower()
    if s in ["liquid", "ultra short", "ultra-short", "money market", "overnight", "short duration"]:
        return s.title()
    n = (name or "").lower()
    for k, label in [
        ("liquid", "Liquid"),
        ("ultra short", "Ultra Short"),
        ("money market", "Money Market"),
        ("overnight", "Overnight"),
        ("short duration", "Short Duration")
    ]:
        if k in n:
            return label
    return "Debt (Other)"

def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Category"] = df["Category"].astype(str).str.title()
    df["Type"] = ""     # Index | Active for Equity; Debt for debt
    df["Bucket"] = ""   # Large/Mid/Small or Debt subtype

    for i, r in df.iterrows():
        cat = r.get("Category", "")
        sub = r.get("Sub-category", "")
        name = r.get("Fund", "")
        if cat == "Equity":
            df.at[i, "Type"] = "Index" if _is_index(sub, name) else "Active"
            df.at[i, "Bucket"] = _equity_bucket(sub, name)
        else:
            df.at[i, "Type"] = "Debt"
            df.at[i, "Bucket"] = _debt_bucket(sub, name)
    return df

# ---------- sentence builders ----------
def _equity_bucket_sentence(bucket: str, idx_pct: float, act_pct: float) -> str:
    # Round nicely
    ip = int(round(idx_pct))
    ap = int(round(act_pct))
    tp = ip + ap

    if bucket == "Large Cap":
        if ip > 0 and ap > 0:
            return f"Large-cap (~{tp}%) is split between **index funds** that cover the top 100 companies at low cost and **active funds** that try to beat that index with selective picks"
        if ip > 0:
            return f"Large-cap (~{ip}%) comes via **index funds** that cover the top 100 companies at low cost"
        return f"Large-cap (~{ap}%) uses **active funds** aiming to beat the large-cap index while keeping risk moderate"

    if bucket == "Mid Cap":
        if ip > 0 and ap > 0:
            return f"Mid-cap (~{tp}%) blends **index funds** on the next 150 mid-sized companies with **active funds** for extra growth"
        if ip > 0:
            return f"Mid-cap (~{ip}%) tracks an **index** of mid-sized companies—the portfolio’s growth engine"
        return f"Mid-cap (~{ap}%) uses **active funds** in mid-sized companies to add punch to growth"

    if bucket == "Small Cap":
        if ip > 0 and ap > 0:
            return f"Small-cap (~{tp}%) combines **index exposure** to 250 smaller companies with **active picks** for tactical opportunities"
        if ip > 0:
            return f"Small-cap (~{ip}%) adds **index exposure** to 250 smaller, early-stage companies"
        return f"Small-cap (~{ap}%) is via **active funds** targeting emerging companies for higher growth"

    # Other equity styles collapsed to a simple sentence
    if tp > 0:
        if ap > 0 and ip > 0:
            return f"Other equity (~{tp}%) mixes **index** and **active** styles for diversification"
        return f"Other equity (~{tp}%) adds diversification"

    return ""

def _debt_sentence(total_debt_pct: float, debt_break: Dict[str, float]) -> str:
    td = int(round(total_debt_pct))
    # Mention Liquid/Ultra-Short if present
    mentions = []
    for key in ["Liquid", "Ultra Short", "Money Market", "Overnight", "Short Duration"]:
        v = int(round(debt_break.get(key, 0)))
        if v > 0:
            mentions.append(f"{key.lower()} (~{v}%)")
    if mentions:
        detail = ", ".join(mentions[:2])  # keep it tight
        return f"Debt (~{td}%) **anchors** the plan for safety and liquidity, mostly in {detail}"
    return f"Debt (~{td}%) **anchors** the plan for safety and liquidity"

# ---------- main explainer ----------
def explain_portfolio_story(final_portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Input: DataFrame with columns ['Fund','Category','Sub-category','Weight (%)']
    Output: { block_id, story, data_points }
    """
    df = _annotate(final_portfolio)

    # Equity vs Debt split
    by_cat = df.groupby("Category")["Weight (%)"].sum().to_dict()
    eq_total = float(by_cat.get("Equity", 0.0))
    debt_total = float(by_cat.get("Debt", 0.0))
    fund_count = len(df)

    # Equity breakdown by bucket + type
    eq = df[df["Category"] == "Equity"].copy()
    bucket_type = (
        eq.groupby(["Bucket", "Type"])["Weight (%)"].sum().reset_index()
        if not eq.empty else pd.DataFrame(columns=["Bucket", "Type", "Weight (%)"])
    )
    # Simplify to dict: {bucket: {"Index": pct, "Active": pct, "Total": pct}}
    buckets = {}
    for _, r in bucket_type.iterrows():
        b = r["Bucket"]
        t = r["Type"]
        buckets.setdefault(b, {"Index": 0.0, "Active": 0.0, "Total": 0.0})
        buckets[b][t] += float(r["Weight (%)"])
        buckets[b]["Total"] += float(r["Weight (%)"])

    # Debt breakdown by subtype
    debt = df[df["Category"] == "Debt"].copy()
    debt_break = (
        debt.groupby("Bucket")["Weight (%)"].sum().to_dict()
        if not debt.empty else {}
    )

    # Build sentences (keep to <= 3 total lines when joined)
    parts: List[str] = []

    # Opening: big picture split
    if eq_total + debt_total > 0:
        parts.append(
            f"Your portfolio is diversified across **~{int(round(eq_total))}% equity** and **~{int(round(debt_total))}% debt**, spread over {fund_count} funds so each allocation has meaningful weight."
        )

    # Equity sentences: Large/Mid/Small first (in that order), then any Other Equity
    for key in ["Large Cap", "Mid Cap", "Small Cap"]:
        if key in buckets and buckets[key]["Total"] > 0:
            s = _equity_bucket_sentence(
                key, buckets[key].get("Index", 0.0), buckets[key].get("Active", 0.0)
            )
            if s:
                parts.append(s + ".")

    # Other Equity, if present
    others_total = sum(
        v["Total"] for k, v in buckets.items() if k not in ["Large Cap", "Mid Cap", "Small Cap"]
    )
    if others_total > 0:
        parts.append(_equity_bucket_sentence("Other Equity", 0.0, others_total) + ".")

    # Debt sentence (unified)
    if debt_total > 0:
        parts.append(_debt_sentence(debt_total, debt_break) + ".")

    # Close with selection differentiator (keep short)
    parts.append("Each fund is chosen using our proprietary quantitative filter across time horizons and metrics, so every pick has a clear role.")

    # Compose into ≤3 lines: join and then soft wrap by sentences
    # For MVP, keep as a single flowing paragraph.
    story = " ".join(parts)

    # datapoints for UI chips if needed
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
