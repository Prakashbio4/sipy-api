# =============== recommendation_explainer.py ===============
# Conversational 4-part explainers for all Funding Ratio x Strategy scenarios
# Tone: human advisor (plain, warm, confident), with simple bridges.
from typing import Dict, Any, List, Optional
import json
import re

# ---------------- Formatting helpers ----------------

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
    Given a *percent* (e.g., 77.3), return an integer band +/- width, clipped to [0, 999].
    We use 999 upper clip just to be safe for very high FR%.
    """
    if center_pct is None:
        center_pct = 0.0
    lo = max(0, int(round(center_pct)) - width)
    hi = min(999, int(round(center_pct)) + width)
    return lo, hi

def _coerce_pct(val):
    if val is None:
        return None
    try:
        return max(0, min(100, int(round(float(val)))))
    except Exception:
        return None

# --------- NEW: minimal scenario detection (non-breaking) ---------

def _detect_horizon_years(ctx: Dict) -> Optional[float]:
    """
    Prefer years_to_goal if present; else infer from months_to_goal.
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

# ---------------- Portfolio parsing ----------------

def _bucket_from_fund_name(name: str) -> str:
    n = (name or "").lower()
    if "smallcap" in n or "small cap" in n:
        return "small"
    if "midcap" in n or "mid cap" in n:
        return "mid"
    if ("nifty 50" in n or "nifty50" in n or "next 50" in n
        or "nifty next 50" in n or "bluechip" in n or "large" in n
        or "nifty 100" in n or "top 100" in n or "bse 100" in n):
        return "large"
    return "other"

def parse_portfolio(value):
    """
    Parse 'portfolio' (list of dicts: Fund, Category, Type|Sub-category, Weight (%))
    into aggregate buckets (approximate):
      - large_cap_pct, mid_cap_pct, small_cap_pct, debt_pct
    """
    out = {}

    def _agg(lst: List[dict]):
        large = mid = small = debt = other = 0.0
        for row in lst:
            try:
                w = float(row.get("Weight (%)", 0))
            except Exception:
                w = 0.0
            cat = (row.get("Category") or "").lower()
            t = (row.get("Type") or row.get("Sub-category") or "").lower()
            fund = row.get("Fund") or ""

            if "debt" in cat or "liquid" in t:
                debt += w
            elif "equity" in cat:
                b = _bucket_from_fund_name(fund)
                if b == "large": large += w
                elif b == "mid": mid += w
                elif b == "small": small += w
                else: other += w

        total = large + mid + small + debt + other
        if total > 0:
            scale = 100.0 / total
            large, mid, small, debt, other = [round(x * scale) for x in (large, mid, small, debt, other)]
        return {
            "large_cap_pct": _coerce_pct(large),
            "mid_cap_pct": _coerce_pct(mid),
            "small_cap_pct": _coerce_pct(small),
            "debt_pct": _coerce_pct(debt),
        }

    if isinstance(value, list):
        return _agg(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return _agg(obj)
        except Exception:
            pass
    return out

# ---------------- Mirror moment (Explainer 1) ----------------

def _mirror_sentence(name: str,
                     goal_amount_str: str,
                     funding_ratio_pct: float,
                     funding_ratio_dec: float,
                     years_to_goal: Optional[int],
                     status: str) -> str:
    """
    status in {"underfunded", "balanced", "overfunded"}
    We show a +/-5% band (rounded ints) around funding_ratio_pct.
    """
    lo, hi = _band_percent(funding_ratio_pct, width=5)
    yr_line = f"\n\nWe have about {years_to_goal} year(s) ahead of us, which gives us room to steer this right." if years_to_goal else ""

    if status == "underfunded":
        return (
            f"Hi {name}, here’s where you stand. As of now, your plan is about {lo}–{hi}% funded. "
            "We’re not at the finish line yet - and that’s okay. This is fixable.\n\n"
            "It’s more common than you think to start in this range. We’ll make a few steady changes and close the gap."
            f"{yr_line}"
        )
    if status == "balanced":
        return (
            f"Hi {name}, here’s where you stand. Your plan looks on track - roughly {lo}–{hi}% funded. "
            "That’s right where we want to be, with normal market ups and downs in mind.\n\n"
            "We’ll keep the momentum, make small improvements, and stay disciplined."
            f"{yr_line}"
        )
    # overfunded
    return (
        f"Hi {name}, here’s where you stand. You’re ahead - roughly {lo}–{hi}% funded against your goal. "
        "That gives you options: reach the goal sooner, aim higher, or even ease the monthly amount.\n\n"
        "We’ll choose the calmest path that still gets you the outcome you want."
        f"{yr_line}"
    )

# ---------------- Strategy sentence (Explainer 2 core) ----------------

def _strategy_sentence(strategy: str, risk_profile: Optional[str], status: str) -> str:
    """
    Plain-language description of the chosen approach.
    """
    s = (strategy or "").strip().title() or "Active"
    rp = (risk_profile or "").strip().lower()

    if s == "Passive":
        # Lower cost, steady compounding; especially sensible if overfunded or short horizon
        return ("We’ll keep it simple with low-cost index funds. "
                "They track the market, keep fees low, and let your money grow quietly.")

    if s == "Hybrid":
        # Indices as base + limited active for selective edge
        return ("We’ll use a Hybrid plan - index funds for a steady base, "
                "and a small set of carefully chosen active funds to try and add a bit extra.")

    # Active
    if rp.startswith("conserv"):
        return ("We’ll go Active but stay careful - a few expert-managed funds, "
                "kept within sensible limits so risk doesn’t run ahead of you.")
    return ("We’ll go Active - a few expert-managed funds aiming for better-than-market returns. "
            "It can go up and down, so I’ll keep it within sensible limits.")

# ---------------- Main story builder ----------------

def build_recommendation_parts(ctx: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns the conversational story split into 4 variables:
      var1: Mirror moment (status)
      var2: Strategy (with bridge)
      var3: Glide path / evolution (with bridge)
      var4: Portfolio setup / close (with bridge)
    Expects ctx keys:
      - name, target_corpus, funding_ratio_pct (0..100*X), strategy, risk_profile
      - equity_start_pct, large_cap_pct, mid_cap_pct, small_cap_pct, debt_pct
      - years_to_goal (optional)
    """
    name        = (ctx.get("name") or "Investor").strip()
    target_str  = _money(ctx.get("target_corpus"))
    fr_pct      = float(ctx.get("funding_ratio_pct") or 0.0)             # e.g., 77.2
    fr_dec      = fr_pct / 100.0                                         # e.g., 0.772
    years       = ctx.get("years_to_goal") or ctx.get("time_to_goal") or ctx.get("horizon_years")

    # Funding status per your thresholds
    if fr_dec < 0.7:
        status = "underfunded"
    elif fr_dec >= 1.3:
        status = "overfunded"
    else:
        status = "balanced"

    strategy    = (ctx.get("strategy") or "").strip().title() or "Active"
    risk_prof   = (ctx.get("risk_profile") or "").strip()

    eq_start    = _pct(ctx.get("equity_start_pct"))
    large       = _pct(ctx.get("large_cap_pct"))
    mid         = _pct(ctx.get("mid_cap_pct"))
    small       = _pct(ctx.get("small_cap_pct"))
    provided_sum = sum(v for v in [large, mid, small] if v)
    debt        = _pct(ctx.get("debt_pct") if ctx.get("debt_pct") is not None else (100 - provided_sum))

    # -------- Bridges (kept exactly as in your current copy)
    b12 = "So what do we do now?"
    b23 = "Here’s how your plan changes over the years."
    b34 = "Now, let me show you the funds and weights."

    # -------- Mode detection (NEW, non-breaking)
    mode = _detect_mode(ctx)
    lo, hi = _band_percent(fr_pct, width=5)

    # ===================== SHORT-TERM MODE (≤ 12 months) =====================
    if mode == "short_term":
        y = _detect_horizon_years(ctx)
        months = max(1, int(round(y * 12))) if y is not None else 6

        # 1) Mirror (month framing)
        var1 = (
            f"Hi {name}, here’s where you stand. Your plan is roughly {lo}–{hi}% funded. "
            f"With only {months} month(s) left, the focus shifts from chasing returns "
            f"to protecting what you’ve built and making sure the money is there when you need it."
        )

        # 2) Strategy (bridge kept)
        var2_core = (
            "We’ll keep the plan stable and liquid. Most of your portfolio stays in short-duration debt and ultra-liquid funds "
            "to hold value and remain accessible. A small slice stays in equities only if it adds meaningful upside without "
            "risking the target."
        )
        var2 = f"{b12}\n\n{var2_core}"

        # 3) Evolution (bridge kept, month framing)
        if large > 0 or mid > 0 or small > 0:
            parts = []
            if large > 0: parts.append(f"{large}% in large companies for stability")
            if mid > 0:   parts.append(f"{mid}% in mid-sized companies for growth")
            if small > 0: parts.append(f"{small}% in smaller companies for extra growth")
            split_sentence = "; ".join(parts)
            var3_body = (
                f"In the coming months, we’ll hold about {eq_start}% in equities — split into {split_sentence}. "
                f"The rest sits in debt ({debt}%) for stability and liquidity. As the date approaches, "
                "we’ll lean further into debt to lock in what you’ve earned."
            )
        else:
            var3_body = (
                f"In the coming months, we’ll hold about {eq_start}% in equities and keep the balance in debt for stability and liquidity. "
                "As the date approaches, we’ll lean further into debt to lock in what you’ve earned."
            )
        var3 = f"{b23}\n\n{var3_body}"

        # 4) Close (bridge kept)
        closing_line = (
            "This isn’t guesswork. It’s a disciplined plan built for precision at a short horizon. "
            "I’ll keep watch and make small adjustments only if they genuinely improve your odds of hitting the date."
        )
        var4 = f"{b34}\n\n{closing_line}"

        return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}

    # ===================== IMPOSSIBLE MODE (very low FR + short horizon) =====================
    if mode == "impossible":
        y = _detect_horizon_years(ctx)
        if y is None:
            horizon_txt = "the current timeline"
        elif y < 1.0:
            horizon_txt = f"~{max(1, int(round(y*12)))} months"
        else:
            horizon_txt = f"~{_pct(y)} year(s)"

        # 1) Mirror (honest but empathetic)
        var1 = (
            f"Hi {name}, here’s the honest view. Your plan is about {lo}–{hi}% funded. "
            f"Given {horizon_txt}, this target isn’t mathematically achievable with the current inputs."
        )

        # 2) Strategy (bridge kept)
        var2_core = (
            "This isn’t a reflection of effort — it’s how compounding works. "
            "We can still make this meaningful by adjusting one of three levers:\n"
            "• Extend the timeline so compounding can work for you\n"
            "• Increase the monthly contribution (even modestly helps)\n"
            "• Redefine the target into achievable milestones toward the bigger goal"
        )
        var2 = f"{b12}\n\n{var2_core}"

        # 3) Evolution (bridge kept)
        var3_body = (
            "For now, we’ll keep costs low and behavior steady while we recalibrate the plan. "
            "As we update the inputs, SIPY will rebuild the path automatically."
        )
        var3 = f"{b23}\n\n{var3_body}"

        # 4) Close (bridge kept)
        closing_line = (
            "This isn’t guesswork. It’s a disciplined approach that avoids false promises and focuses on steps that actually work."
        )
        var4 = f"{b34}\n\n{closing_line}"

        return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}

    # ===================== DEFAULT MODE (your current copy) =====================

    # -------- Explainer 1 (mirror moment with +/-5% band)
    var1 = _mirror_sentence(
        name=name,
        goal_amount_str=target_str,
        funding_ratio_pct=fr_pct,
        funding_ratio_dec=fr_dec,
        years_to_goal=years,
        status=status
    )

    # -------- Explainer 2 (strategy)
    var2_core = _strategy_sentence(strategy, risk_prof, status)
    var2 = f"{b12}\n\n{var2_core}"

    # -------- Explainer 3 (evolution; unchanged tone & structure)
    if status == "underfunded":
        tone_prefix = ""
    elif status == "balanced":
        tone_prefix = "We’ll keep the pace sensible and steady. "
    else:  # overfunded
        tone_prefix = "You’ve already done a lot of the hard work, so we’ll be conservative by design. "

    if large > 0 or mid > 0 or small > 0:
        parts = []
        if large > 0: parts.append(f"{large}% in large companies for stability")
        if mid > 0:   parts.append(f"{mid}% in mid-sized companies for growth")
        if small > 0: parts.append(f"{small}% in smaller companies for extra growth")
        split_sentence = "; ".join(parts)

        var3_body = (
            f"{tone_prefix}"
            f"In the first year, we start with {eq_start}% in equities - split into {split_sentence}. "
            f"The rest sits in debt ({debt}%) to keep things steady. "
            "As the goal gets closer, we’ll move more into debt to lock in what you’ve earned."
        )
    else:
        var3_body = (
            f"{tone_prefix}"
            f"In the first year, we start with {eq_start}% in equities. "
            "The rest sits in debt to keep things steady. "
            "As the goal gets closer, we’ll move more into debt to lock in what you’ve earned."
        )
    var3 = f"{b23}\n\n{var3_body}"

    # -------- Explainer 4 (portfolio setup / close)
    closing_line = (
        "This isn’t guesswork. It’s a disciplined plan built for you. "
        "I’ll review and rebalance regularly so you stay on track."
    )
    var4 = f"{b34}\n\n{closing_line}"

    return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}

def build_recommendation_story(ctx: Dict[str, Any]) -> str:
    """Optional: full story in one string with plain paragraph breaks."""
    parts = build_recommendation_parts(ctx)
    return "\n\n".join([parts["var1"], parts["var2"], parts["var3"], parts["var4"]])


# ---------------- Minimal demo ----------------
if __name__ == "__main__":
    sample_portfolio = [
        {"Fund":"UTI Nifty 50 Index Fund", "Category":"Equity", "Type":"Passive", "Weight (%)":40},
        {"Fund":"Motilal Oswal Midcap 150 Index", "Category":"Equity", "Type":"Passive", "Weight (%)":30},
        {"Fund":"Bandhan Small Cap Fund", "Category":"Equity", "Type":"Active", "Weight (%)":10},
        {"Fund":"ICICI Pru Liquid", "Category":"Debt", "Type":"Liquid", "Weight (%)":20},
    ]
    ctx = {
        "name": "Investor",
        "target_corpus": 1e7,
        "funding_ratio_pct": 77.0,     # will display 72–82%
        "strategy": "Hybrid",
        "risk_profile": "moderate",
        "equity_start_pct": 85,
        "years_to_goal": 12,
        **parse_portfolio(sample_portfolio),
    }
    print(build_recommendation_story(ctx))
