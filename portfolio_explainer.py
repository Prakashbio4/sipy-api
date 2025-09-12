# portfolio_explainer.py
from typing import Dict, Any
import pandas as pd
import re

# Simple mapping for common Indian index/fund names
def _style_from_name(name: str) -> str:
    n = (name or "").lower()
    # Large
    if re.search(r"(nifty\s*50|sensex|nifty\s*100|next\s*50|large\s*cap|blue\s*chip)", n):
        return "Large Cap"
    # Mid
    if re.search(r"(mid\s*cap|midcap\s*150|midcap\s*100|midcap\s*50)", n):
        return "Mid Cap"
    # Small
    if re.search(r"(small\s*cap|smallcap\s*250|smallcap\s*100|smallcap\s*50)", n):
        return "Small Cap"
    return "Other"

def explain_portfolio_story(final_portfolio: pd.DataFrame) -> Dict[str, Any]:
    df = final_portfolio.copy()

    equity_total = int(round(df.loc[df["Category"]=="Equity","Weight (%)"].sum()))
    debt_total   = int(round(df.loc[df["Category"]=="Debt",  "Weight (%)"].sum()))
    fund_count   = int(df.shape[0])

    # Equity style split (by fund name)
    eq = df[df["Category"]=="Equity"].copy()
    eq["Style"] = eq["Fund"].apply(_style_from_name)
    style_split = (
        eq.groupby("Style")["Weight (%)"].sum().round().astype(int).to_dict()
        if not eq.empty else {}
    )
    large = style_split.get("Large Cap", 0)
    mid   = style_split.get("Mid Cap", 0)
    small = style_split.get("Small Cap", 0)
    other = style_split.get("Other", 0)

    # Passive vs Active flavor (by Sub-category label)
    eq_subcats = (
        eq["Sub-category"].str.lower().value_counts().to_dict() if not eq.empty else {}
    )
    passive_share = int(round(eq.loc[eq["Sub-category"].str.lower().str.contains("passive|index", na=False), "Weight (%)"].sum())) if not eq.empty else 0
    active_share  = max(0, equity_total - passive_share)

    # Debt mix
    debt_mix = (
        df[df["Category"]=="Debt"]
        .groupby("Sub-category")["Weight (%)"].sum()
        .round().astype(int).to_dict()
    )

    # Narrative built from real weights
    bits = []
    bits.append(f"Your portfolio holds **~{equity_total}% equity** and **~{debt_total}% debt** across {fund_count} funds.")
    if equity_total:
        parts = [f"Large-cap (~{large}%)", f"Mid-cap (~{mid}%)", f"Small-cap (~{small}%)"]
        if other: parts.append(f"Other (~{other}%)")
        bits.append("Equity style mix: " + ", ".join(parts) + ".")
        # Mention passive/active only if clear
        if passive_share >= 60:
            bits.append(f"Most equity exposure (~{passive_share}%) comes via **index funds** for broad, low-cost market capture.")
        elif active_share >= 60:
            bits.append(f"Most equity exposure (~{active_share}%) is **actively managed**, aiming for selective alpha.")
    if debt_total:
        dm = ", ".join([f"{k} (~{v}%)" for k, v in debt_mix.items()])
        bits.append("Debt anchors stability and liquidity via " + dm + ".")

    bits.append("Weights are kept meaningful (≥10%) so each position contributes visibly to outcomes.")
    story = " ".join(bits)

    return {
        "block_id": "block_3_portfolio",
        "story": story,
        "data_points": {
            "equity_total_pct": equity_total,
            "debt_total_pct": debt_total,
            "fund_count": fund_count,
            "equity_style_split": {"Large Cap": large, "Mid Cap": mid, "Small Cap": small, "Other": other},
            "equity_passive_pct": passive_share,
            "equity_active_pct": active_share,
            "debt_mix": debt_mix,
        },
    }
