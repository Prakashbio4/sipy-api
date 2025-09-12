# ===================== glide_explainer.py =====================
from typing import Any, Dict, List, Union
import math

def _clamp_pct(x: float) -> int:
    try:
        return max(0, min(100, int(round(float(x)))))
    except Exception:
        return 0

def _yr(r: Dict[str, Any]) -> int:
    return int(r.get("Year") or r.get("year") or 0)

def _eq(r: Dict[str, Any]) -> float:
    return float(r.get("Equity Allocation (%)") or r.get("equity_pct") or r.get("equity") or 0.0)

def _coerce_ctx(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        raise TypeError("Input must be a dictionary.")
    return ctx

def _pick_points(bands: List[Dict[str, Any]], years_to_goal: int):
    b = sorted(bands, key=_yr)
    start_eq = _clamp_pct(_eq(b[0]))
    end_eq   = _clamp_pct(_eq(b[-1]))
    end_year = _yr(b[-1])

    mid_year = math.ceil(len(b) / 2)
    mid_eq   = _clamp_pct(_eq(b[mid_year - 1]))

    return start_eq, mid_eq, end_eq, end_year

def explain_glide_story(ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _coerce_ctx(ctx)
    path = ctx.get("glide_path") or []
    years = ctx.get("years_to_goal", len(path) or 0)

    if not path:
        return {
            "block_id": "block_1_glide_path",
            "story": "We’ll show your glide path once we have enough data.",
            "data_points": {}
        }

    start_eq, mid_eq, end_eq, end_year = _pick_points(path, years)
    end_debt = 100 - end_eq

    # 1. Start with a confident opening
    start_sentence = f"Your plan begins with a strong start, holding **~{start_eq}% in equity** to capture early growth."
    
    # 2. Describe the mid-journey transition dynamically
    if start_eq > mid_eq:
        mid_sentence = f"As you move closer to your goal, the portfolio will automatically de-risk, stepping down to a more balanced mix of **~{mid_eq}% equity** midway through the journey."
    else:
        mid_sentence = "The portfolio will maintain its high-growth allocation for a period before beginning to de-risk."

    # 3. Conclude with the end-game
    end_sentence = f"By the final year, the allocation will finish at **~{end_eq}% equity** and **~{end_debt}% debt**, so your capital is protected right when you need it."
    
    story = " ".join([start_sentence, mid_sentence, end_sentence])

    return {
        "block_id": "block_1_glide_path",
        "story": story,
        "data_points": {
            "equity_start_pct": start_eq,
            "equity_mid_pct": mid_eq,
            "equity_end_pct": end_eq,
            "end_debt_pct": end_debt,
            "years_to_goal": years,
            "end_year": end_year
        }
    }