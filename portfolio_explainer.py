# ===================== portfolio_explainer.py =====================
from typing import Dict, Any, List
import pandas as pd
import re


# ---------- Column Normalizer ----------
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

    eq_df = df[df["Category"].str.lower() == "equity"]
    debt_df = df[df["Category"].str.lower() == "debt"]

    equity_total = float(eq_df["Weight"].sum())
    debt_total = float(debt_df["Weight"].sum())
    fund_count = len(df)

    # Summarize sub-categories
    equity_sub_cats = eq_df.groupby("Sub-category")["Weight"].sum().sort_values(ascending=False).to_dict()
    debt_sub_cats = debt_df.groupby("Sub-category")["Weight"].sum().sort_values(ascending=False).to_dict()

    parts: List[str] = []

    # 1. High-level summary
    summary_sentence = (
        f"Your portfolio is built with a strong focus on growth, holding **~{int(round(equity_total))}% in equity** "
        f"and **~{int(round(debt_total))}% in debt**."
    )
    parts.append(summary_sentence)

    # 2. Equity mix explanation
    equity_parts = []
    if any("Large Cap" in k for k in equity_sub_cats.keys()):
        equity_parts.append("a core of **large-cap funds** for stability")
    if any("Mid Cap" in k for k in equity_sub_cats.keys()):
        equity_parts.append("a **mid-cap fund** to drive growth")
    if any("Small Cap" in k for k in equity_sub_cats.keys()):
        equity_parts.append("a **small-cap fund** for extra upside potential")
    
    if equity_parts:
        equity_mix_sentence = f"For equity, you have a mix of all three market sizes: {', '.join(equity_parts)}."
        parts.append(equity_mix_sentence)

    # 3. Debt purpose explanation
    if debt_total > 0:
        debt_purpose_sentence = f"Your debt portion is anchored in **liquid funds** to provide stability and easy access to cash."
        parts.append(debt_purpose_sentence)

    # 4. Final conclusion
    conclusion = f"This blend is designed to capture market growth while steadily protecting your capital as you get closer to your goal."
    parts.append(conclusion)
    
    story = " ".join(parts)

    data_points = {
        "equity_total_pct": int(round(equity_total)),
        "debt_total_pct": int(round(debt_total)),
        "fund_count": fund_count,
        "equity_sub_cats": {k: int(v) for k, v in equity_sub_cats.items()},
        "debt_sub_cats": {k: int(v) for k, v in debt_sub_cats.items()},
    }

    return {
        "block_id": "block_3_portfolio",
        "story": story,
        "data_points": data_points
    }