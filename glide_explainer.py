# glide_explainer.py
from typing import Any, Dict, List, Union
import math

Row = Dict[str, Any]
Ctx = Dict[str, Any]

def _coerce_ctx(v: Union[Ctx, List[Row]]) -> Ctx:
    """Accept either full context dict or legacy list[rows]."""
    if isinstance(v, dict) and "glide_path" in v:
        return {
            "years_to_goal": int(v.get("years_to_goal") or len(v["glide_path"]) or 0),
            "risk_profile": (v.get("risk_profile") or "").strip().title(),
            "funding_ratio": float(v.get("funding_ratio") or 0.0),
            "glide_path": v["glide_path"],
        }
    if isinstance(v, list):
        return {
            "years_to_goal": len(v),
            "risk_profile": "",
            "funding_ratio": 0.0,
            "glide_path": v,
        }
    raise ValueError("glide_explainer: expected dict with 'glide_path' or list[rows].")

def _num(row: Row, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)

def explain_glide_story(ctx_or_rows: Union[Ctx, List[Row]]) -> Dict[str, Any]:
    ctx = _coerce_ctx(ctx_or_rows)
    path = ctx["glide_path"]
    if not path:
        raise ValueError("glide_explainer: empty glide_path")

    years = int(ctx["years_to_goal"])
    start_eq = _num(path[0], "Equity Allocation (%)")
    end_eq   = _num(path[-1], "Equity Allocation (%)")
    end_debt = 100 - end_eq
    mid_idx  = max(0, math.ceil(years / 2) - 1)
    mid_eq   = _num(path[mid_idx], "Equity Allocation (%)")

    # Clear, consistent story (short vs long horizon tone)
    parts = []
    if years <= 5:
        parts.append(
            f"We begin near **{int(start_eq)}% equity** to capture early growth, "
            f"then de-risk quickly (around year {mid_idx+1}: **~{int(mid_eq)}%**). "
            f"By the goal year, only **{int(end_eq)}%** is in equity and **{int(end_debt)}%** in debt to protect capital."
        )
    else:
        parts.append(
            f"Starting at **{int(start_eq)}% equity**, we step down gradually "
            f"(mid-journey **~{int(mid_eq)}%**) and finish at **{int(end_eq)}% equity** "
            f"(**{int(end_debt)}% debt**) close to the goal."
        )

    fr = float(ctx.get("funding_ratio") or 0.0)
    if fr:
        if fr >= 1.3: fr_txt = "ahead of pace (over-funded)"
        elif fr >= 0.7: fr_txt = "on track"
        else: fr_txt = "behind pace (under-funded)"
        parts.append(f"Funding ratio **{fr:.2f}** — {fr_txt} — guides this growth vs. safety balance.")

    risk = (ctx.get("risk_profile") or "").strip()
    if risk:
        parts.append(f"Risk preference noted: **{risk}**.")

    story = " ".join(parts)

    return {
        "block_id": "block_1_glide_path",
        "story": story,
        "data_points": {
            "years_to_goal": years,
            "equity_start_pct": int(start_eq),
            "equity_mid_pct": int(mid_eq),
            "equity_end_pct": int(end_eq),
            "debt_end_pct": int(end_debt),
        },
    }
