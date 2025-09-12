# glide_explainer.py
from typing import Any, Dict, List, Union
import math

# === Your original phase helpers (kept) ===
PHASE_HIGH = "high_growth"        # equity >= 70
PHASE_BAL  = "balanced_growth"    # 30 <= equity < 70
PHASE_SAFE = "safety_first"       # equity < 30

def _phase(eq: float) -> str:
    if eq >= 70: return PHASE_HIGH
    if eq >= 30: return PHASE_BAL
    return PHASE_SAFE

def _clamp_pct(x: float) -> int:
    return max(0, min(100, round(float(x))))

def _yr(d: Dict[str, Any]) -> int:
    return int(d.get("Year") or d.get("year") or 0)

def _eq(d: Dict[str, Any]) -> float:
    return float(d.get("Equity Allocation (%)") or d.get("equity_pct") or d.get("equity") or 0.0)

def _pick_points(bands: List[Dict[str, Any]], years_to_goal: int):
    """
    Return integer start/mid/end equity and end_year using your band format.
    """
    b = sorted(bands, key=_yr)
    start_eq = _clamp_pct(_eq(b[0]))
    end_eq   = _clamp_pct(_eq(b[-1]))
    start_y  = _yr(b[0]) or 1
    end_y    = _yr(b[-1]) or (years_to_goal or len(b))

    span = max(1, (end_y - start_y) if end_y and start_y else (years_to_goal or 1))
    target_mid_year = start_y + span / 2
    mid_band = min(b, key=lambda x: abs((_yr(x) or 0) - target_mid_year))
    mid_eq   = _clamp_pct(_eq(mid_band))

    return start_eq, mid_eq, end_eq, (end_y or years_to_goal or len(b))

def _coerce_ctx(ctx_or_rows: Union[Dict[str, Any], List[Dict[str, Any]], None], glide_path_kw: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    """
    Accept any of:
      - explain_glide_story(glide_path=[...])
      - explain_glide_story([...])
      - explain_glide_story({"years_to_goal":..., "risk_profile":..., "funding_ratio":..., "glide_path":[...]})
    """
    # 1) Keyword form
    if glide_path_kw is not None:
        return {"years_to_goal": len(glide_path_kw), "risk_profile": "", "funding_ratio": 0.0, "glide_path": glide_path_kw}

    # 2) Positional list
    if isinstance(ctx_or_rows, list):
        return {"years_to_goal": len(ctx_or_rows), "risk_profile": "", "funding_ratio": 0.0, "glide_path": ctx_or_rows}

    # 3) Context dict (new style)
    if isinstance(ctx_or_rows, dict):
        gp = ctx_or_rows.get("glide_path") or ctx_or_rows.get("bands") or []
        return {
            "years_to_goal": int(ctx_or_rows.get("years_to_goal") or ctx_or_rows.get("goal", {}).get("years_to_goal") or len(gp) or 0),
            "risk_profile": (ctx_or_rows.get("risk_profile") or "").strip().title(),
            "funding_ratio": float(ctx_or_rows.get("funding_ratio") or 0.0),
            "glide_path": gp,
        }

    raise ValueError("glide_explainer: expected dict with 'glide_path', or a list of bands.")

def explain_glide_story(ctx_or_rows: Union[Dict[str, Any], List[Dict[str, Any]], None] = None, *, glide_path: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """
    Returns:
      {
        "block_id": "block_1_glide_path",
        "story": "<paragraph>",
        "data_points": {...}
      }
    """
    ctx = _coerce_ctx(ctx_or_rows, glide_path)
    bands = ctx["glide_path"]
    years = int(ctx["years_to_goal"] or len(bands) or 0)

    if not bands or len(bands) < 2:
        return {
            "block_id": "block_1_glide_path",
            "story": "We’ll show your glide path once we have enough data.",
            "data_points": {}
        }

    start_eq, mid_eq, end_eq, end_year = _pick_points(bands, years)
    end_debt = max(0, 100 - end_eq)

    # Shape detection (kept from your logic, wording polished)
    start_ph, mid_ph, end_ph = _phase(start_eq), _phase(mid_eq), _phase(end_eq)
    NEAR = 5
    flat_then_shift = abs(mid_eq - start_eq) <= NEAR and abs(end_eq - start_eq) > NEAR
    smooth_taper    = (start_eq - mid_eq) > NEAR and (mid_eq - end_eq) > NEAR
    direct_drop     = (
        start_ph == PHASE_HIGH and (
            (mid_ph == PHASE_SAFE and end_ph == PHASE_SAFE) or
            (abs(mid_eq - end_eq) <= NEAR and (start_eq - mid_eq) > 2*NEAR and end_ph in (PHASE_BAL, PHASE_SAFE))
        )
    )

    # Story (avoid repeating the same % twice)
    if years <= 5:
        opening = f"We start near **{start_eq}% equity** to capture early growth"
    else:
        opening = f"Starting at **{start_eq}% equity**, we plan for growth first"

    if direct_drop:
        middle = f", then de-risk decisively to about **{mid_eq}%** when it matters"
    elif flat_then_shift:
        middle = f", hold roughly steady early on, then shift toward **~{mid_eq}%** for balance"
    elif smooth_taper:
        middle = f", then step down gradually to **~{mid_eq}%** as you move closer"
    else:
        middle = f", easing toward **~{mid_eq}%** over the journey"

    closing = f". By year **{end_year}**, we finish at **{end_eq}% equity** (**{end_debt}% debt**) to protect the goal."

    fr = float(ctx.get("funding_ratio") or 0.0)
    if fr:
        if fr >= 1.3: fr_txt = "ahead of pace (over-funded)"
        elif fr >= 0.7: fr_txt = "on track"
        else: fr_txt = "behind pace (under-funded)"
        closing += f" Funding ratio **{fr:.2f}** — {fr_txt} — guides this balance."

    story = opening + middle + closing

    return {
        "block_id": "block_1_glide_path",
        "story": story,
        "data_points": {
            "years_to_goal": years,
            "equity_start_pct": start_eq,
            "equity_mid_pct": mid_eq,
            "equity_end_pct": end_eq,
            "end_debt_pct": end_debt,
            "end_year": end_year
        }
    }
