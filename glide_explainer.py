# glide_explainer.py
from typing import Any, Dict, List, Union
import math

Row = Dict[str, Any]
Ctx = Dict[str, Any]

# -------- helpers --------
def _clamp_pct(x: float) -> int:
    try:
        return max(0, min(100, int(round(float(x)))))
    except Exception:
        return 0

def _yr(r: Row) -> int:
    return int(r.get("Year") or r.get("year") or 0)

def _eq(r: Row) -> float:
    return float(r.get("Equity Allocation (%)") or r.get("equity_pct") or r.get("equity") or 0.0)

def _coerce_ctx(ctx_or_rows: Union[Ctx, List[Row], None], glide_path_kw: List[Row] | None) -> Ctx:
    """
    Accept any of:
      - explain_glide_story(glide_path=[...])
      - explain_glide_story([...])
      - explain_glide_story({"years_to_goal":..., "risk_profile":..., "glide_path":[...]})
    """
    if glide_path_kw is not None:
        return {"years_to_goal": len(glide_path_kw), "risk_profile": "", "glide_path": glide_path_kw}
    if isinstance(ctx_or_rows, list):
        return {"years_to_goal": len(ctx_or_rows), "risk_profile": "", "glide_path": ctx_or_rows}
    if isinstance(ctx_or_rows, dict):
        gp = ctx_or_rows.get("glide_path") or []
        return {
            "years_to_goal": int(ctx_or_rows.get("years_to_goal") or len(gp) or 0),
            "risk_profile": (ctx_or_rows.get("risk_profile") or "").strip().title(),
            "glide_path": gp,
        }
    raise ValueError("glide_explainer: expected dict with 'glide_path' or a list of rows")

def _pick_points(bands: List[Row], years_to_goal: int):
    b = sorted(bands, key=_yr)
    start_eq = _clamp_pct(_eq(b[0]))
    end_eq   = _clamp_pct(_eq(b[-1]))
    end_y    = _yr(b[-1]) or (years_to_goal or len(b) or 1)

    # choose the band closest to mid-journey
    span = max(1, end_y - (_yr(b[0]) or 1))
    mid_target = (_yr(b[0]) or 1) + span / 2
    mid_row = min(b, key=lambda r: abs((_yr(r) or 0) - mid_target))
    mid_eq = _clamp_pct(_eq(mid_row))

    return start_eq, mid_eq, end_eq, end_y

# -------- main --------
def explain_glide_story(ctx_or_rows: Union[Ctx, List[Row], None] = None,
                        *, glide_path: List[Row] | None = None) -> Dict[str, Any]:
    """
    Return:
      { "block_id": "block_1_glide_path", "story": str, "data_points": {...} }
    """
    ctx = _coerce_ctx(ctx_or_rows, glide_path)
    path = ctx["glide_path"]
    years = int(ctx["years_to_goal"] or len(path) or 0)

    if not path:
        return {"block_id": "block_1_glide_path",
                "story": "We’ll show your glide path once we have enough data.",
                "data_points": {}}

    start_eq, mid_eq, end_eq, end_year = _pick_points(path, years)
    end_debt = 100 - end_eq

    # Layman, non-repetitive wording; short-horizon vs long-horizon tone
    if years <= 5:
        story = (
            f"We start high at **{start_eq}% equity** for early growth, hold steady initially, "
            f"then step down to **{mid_eq}%** midway and finish at **{end_eq}% equity** "
            f"(**{end_debt}% debt**) by year **{end_year}** to protect the goal."
        )
    else:
        story = (
            f"Starting around **{start_eq}% equity**, the mix eases down over time "
            f"(mid-journey **~{mid_eq}%**) and ends near **{end_eq}% equity** "
            f"(**{end_debt}% debt**) close to the goal."
        )

    rp = (ctx.get("risk_profile") or "").strip()
    if rp:
        story += f" This reflects your **{rp}** risk preference through the early-years equity level."

    return {
        "block_id": "block_1_glide_path",
        "story": story,
        "data_points": {
            "years_to_goal": years,
            "equity_start_pct": start_eq,
            "equity_mid_pct": mid_eq,
            "equity_end_pct": end_eq,
            "debt_end_pct": end_debt,
            "end_year": end_year
        },
    }
