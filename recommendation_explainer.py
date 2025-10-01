# ===================== recommendation_explainer.py =====================
# Conversational 4-part explainers for Funding Ratio x Strategy scenarios
# Tone: human advisor (plain, warm, confident), with simple bridges.

from typing import Dict, List, Optional
import re

# ---------------- Formatting helpers (preserved & extended) ----------------

def _pct(val) -> int:
    """Coerce to int percent and clamp to [0, 100]."""
    try:
        p = int(round(float(val)))
    except Exception:
        p = 0
    return max(0, min(100, p))

def _money(val) -> str:
    """Simple INR short-format for readability."""
    try:
        v = float(val)
    except Exception:
        return str(val)
    if v >= 1e7:   # crore
        return f"₹{v/1e7:.2f}Cr"
    if v >= 1e5:   # lakh
        return f"₹{v/1e5:.2f}L"
    return f"₹{v:,.0f}"

def _band_percent(center_pct: float, width: int = 5) -> tuple[int, int]:
    """
    Given a percent (e.g., 77.3), return an integer band +/- width, clipped to [0, 999].
    We use 999 upper clip to be safe for very high FR%.
    """
    try:
        c = float(center_pct)
    except Exception:
        c = 0.0
    lo = max(0, int(round(c)) - width)
    hi = min(999, int(round(c)) + width)
    return lo, hi

def _coerce_pct(val):
    if val is None:
        return None
    try:
        return max(0, min(100, int(round(float(val)))))
    except Exception:
        return None

# --- PATCH START: helpers for scenario detection -----------------------------

def _cap_bucket_from_name(name: str) -> str:
    """
    Heuristic bucketer for equity fund names.
    """
    n = (name or "").lower()
    if re.search(r"(nifty\s*50|sensex|large\s*cap|blue\s*chip|bluechip|top\s*100|nifty\s*100|bse\s*100)", n):
        return "large"
    if re.search(r"(mid\s*cap|midcap|nifty\s*midcap)", n):
        return "mid"
    if re.search(r"(small\s*cap|smallcap|nifty\s*smallcap)", n):
        return "small"
    return "other"

def _safe_name(ctx: Dict) -> str:
    return (ctx.get("name") or "Investor").strip()

def _detect_horizon_years(ctx: Dict) -> Optional[float]:
    """
    Prefer years_to_goal if present; else infer from months_to_goal if provided.
    Returns None if not available.
    """
    y = ctx.get("years_to_goal")
    if y is not None:
        try:
            return float(y)
        except Exception:
            pass
    m = ctx.get("months_to_goal")
    if m is not None:
        try:
            return float(m) / 12.0
        except Exception:
            pass
    return None

def _detect_mode(ctx: Dict) -> str:
    """
    Modes:
      - 'short_term' : years_to_goal < 1 (if available)
      - 'impossible' : funding_ratio < 5% and years_to_goal <= 3 (or unknown)
      - 'default'    : otherwise
    """
    fr_pct = ctx.get("funding_ratio_pct")
    try:
        fr_ratio = float(fr_pct) / 100.0 if fr_pct is not None else None
    except Exception:
        fr_ratio = None

    y = _detect_horizon_years(ctx)

    if y is not None and y < 1.0:
        return "short_term"
    if fr_ratio is not None and fr_ratio < 0.05 and (y is None or y <= 3.0):
        return "impossible"
    return "default"

# --- PATCH END ---------------------------------------------------------------


# ---------------- Public API: parse_portfolio (preserved signature) ----------

def parse_portfolio(display_port: List[Dict]) -> Dict[str, float]:
    """
    Input rows like: {"Fund","Category","Type","Weight (%)"}
    Output: aggregate percentages for large/mid/small/debt.
    """
    large = mid = small = debt = 0.0
    for row in (display_port or []):
        try:
            w = float(row.get("Weight (%)", 0.0) or 0.0)
        except Exception:
            w = 0.0

        category = (row.get("Category") or "").strip().lower()

        if category == "debt":
            debt += w
        else:
            # treat as equity family; try to infer cap from name
            fund_name = row.get("Fund") or ""
            bucket = _cap_bucket_from_name(fund_name)
            if bucket == "large":
                large += w
            elif bucket == "mid":
                mid += w
            elif bucket == "small":
                small += w
            else:
                # unknown equity bucket -> add to large by default (conservative display)
                large += w

    # Normalize to ~100 across equity + debt when used standalone
    total = large + mid + small + debt
    if total > 0:
        scale = 100.0 / total
        large *= scale; mid *= scale; small *= scale; debt *= scale

    return {
        "large_cap_pct": round(large, 2),
        "mid_cap_pct": round(mid, 2),
        "small_cap_pct": round(small, 2),
        "debt_pct": round(debt, 2),
    }


# --- PATCH START: Scenario-aware storytellers (short_term, impossible, default)

def _story_short_term(ctx: Dict) -> Dict[str, str]:
    """
    ≤ 12 months to goal: prioritize capital protection, liquidity,
    and month-based framing; keep portfolio/glide structure for consistency.
    """
    name = _safe_name(ctx)
    fr_center = ctx.get("funding_ratio_pct")
    lo, hi = _band_percent(fr_center, width=5)

    y = _detect_horizon_years(ctx)
    months = max(1, int(round(y * 12))) if y is not None else None

    eq0 = _pct(ctx.get("equity_start_pct"))
    debt = _pct(ctx.get("debt_pct"))
    strat = (ctx.get("strategy") or "").strip()

    # var1: where you stand + short timeline
    var1 = (
        f"Hi {name}, here’s where you stand. Your plan is roughly {lo}–{hi}% funded. "
        f"With only {months if months is not None else 'a few'} month(s) left, the focus shifts from chasing returns "
        f"to protecting what you’ve built and making sure the money is there when you need it."
    )

    # var2: stability & liquidity
    var2 = (
        "We’ll keep the plan stable and liquid. Most of your portfolio stays in short-duration debt and ultra-liquid funds "
        "to hold value and remain accessible. A small slice stays in equities only if it adds meaningful upside without "
        "risking the target."
    )

    # var3: strategy framing
    var3 = (
        f"Strategy: {strat or 'Passive'}. Given the short timeline, this keeps costs low, behavior steady, and decisions simple. "
        "We’ll monitor, but avoid unnecessary changes. The goal now is delivery with precision, not maximizing performance."
    )

    # var4: portfolio snapshot
    var4 = (
        f"Portfolio snapshot for the coming months: Equity ~{eq0}%, Debt ~{debt}%. "
        "Within equity, we’ll bias to larger companies for lower volatility. "
        "We’ll hold this posture and lean further into debt as the date approaches."
    )

    return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}


def _story_impossible(ctx: Dict) -> Dict[str, str]:
    """
    Extremely low funding ratio with short horizon: call out infeasibility
    calmly, offer levers (time, contribution, target).
    """
    name = _safe_name(ctx)
    fr_center = ctx.get("funding_ratio_pct")
    lo, hi = _band_percent(fr_center, width=5)

    y = _detect_horizon_years(ctx)
    if y is None:
        horizon_txt = "the current timeline"
    elif y < 1.0:
        horizon_txt = f"~{max(1, int(round(y*12)))} months"
    else:
        horizon_txt = f"~{_pct(y)} year(s)"

    strat = (ctx.get("strategy") or "").strip()
    eq0 = _pct(ctx.get("equity_start_pct"))
    debt = _pct(ctx.get("debt_pct"))

    var1 = (
        f"Hi {name}, here’s the honest view. Your plan is about {lo}–{hi}% funded. "
        f"Given {horizon_txt}, this target isn’t mathematically achievable with the current inputs."
    )

    var2 = (
        "This isn’t a reflection of effort — it’s how compounding works. "
        "We can still make this meaningful by adjusting one of three levers:\n"
        "• Extend the timeline so compounding can work for you\n"
        "• Increase the monthly contribution (even modestly helps)\n"
        "• Redefine the target into achievable milestones toward the bigger goal"
    )

    var3 = (
        f"Strategy for now: {strat or 'Passive'}. We’ll keep costs low and behavior steady while we recalibrate the plan. "
        "As we update the inputs, SIPY will rebuild the path automatically."
    )

    var4 = (
        f"With current inputs, the starting allocation is Equity ~{eq0}%, Debt ~{debt}%. "
        "This is a placeholder to keep you invested sensibly while we right-size the plan."
    )

    return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}


def _story_default(ctx: Dict) -> Dict[str, str]:
    """
    Standard tone for normal horizons and feasible targets.
    Mirrors your current structure but tighter, clearer.
    """
    name = _safe_name(ctx)
    fr_center = ctx.get("funding_ratio_pct")
    lo, hi = _band_percent(fr_center, width=5)

    strat = (ctx.get("strategy") or "").strip()

    eq0 = _pct(ctx.get("equity_start_pct"))
    debt = _pct(ctx.get("debt_pct"))
    lg = _pct(ctx.get("large_cap_pct"))
    md = _pct(ctx.get("mid_cap_pct"))
    sm = _pct(ctx.get("small_cap_pct"))

    var1 = (
        f"Hi {name}, here’s where you stand. Your plan is roughly {lo}–{hi}% funded — "
        "that’s in the zone we want given normal market ups and downs."
    )

    var2 = (
        "We’ll stay disciplined and keep improving steadily. As the goal gets closer, "
        "we’ll lean down risk in measured steps to lock in progress."
    )

    var3 = (
        f"Strategy: {strat or 'Passive'}. This keeps costs low, behavior consistent, "
        "and sticks to the evidence on what actually compounds well over time."
    )

    var4 = (
        f"Portfolio snapshot (starting point): Equity ~{eq0}% (Large ~{lg}%, Mid ~{md}%, Small ~{sm}%), "
        f"Debt ~{debt}%. We’ll rebalance as needed to keep the plan on track."
    )

    return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}

# --- PATCH END ----------------------------------------------------------------


# ---------------- Public API: build_recommendation_parts (preserved) ----------

def build_recommendation_parts(ctx: Dict) -> Dict[str, str]:
    """
    Construct 4 short conversational blocks (var1..var4) based on context.
    Expected ctx keys (some optional):
      - name (str)
      - funding_ratio_pct (float, e.g., 95.3)
      - strategy (str)
      - risk_profile (str)
      - equity_start_pct (float/int)
      - large_cap_pct, mid_cap_pct, small_cap_pct, debt_pct (floats)
      - years_to_goal (optional float) or months_to_goal (optional int/float)
    """
    mode = _detect_mode(ctx)
    if mode == "short_term":
        return _story_short_term(ctx)
    if mode == "impossible":
        return _story_impossible(ctx)
    return _story_default(ctx)
