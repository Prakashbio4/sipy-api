# ================= recommendation_explainer.py =================
# Generates a short, layman-friendly 4-part explainer (human-advisor tone).
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

# ---------------- Free-text / JSON parsers ----------------

def _parse_percent_from_text(text, keys):
    if not isinstance(text, str):
        return None
    t = text.lower()
    for k in keys:
        m = re.search(rf"{re.escape(k)}\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*%?", t)
        if m:
            try:
                return int(round(float(m.group(1))))
            except Exception:
                pass
    return None

def _coerce_pct(val):
    if val is None:
        return None
    try:
        return max(0, min(100, int(round(float(val)))))
    except Exception:
        return None

def parse_glide_path(value):
    """
    Parse 'glide_path' into:
      - equity_start_pct
      - equity_end_pct (optional)
    """
    out = {}
    if isinstance(value, dict):
        out["equity_start_pct"] = _coerce_pct(value.get("equity_start_pct"))
        out["equity_end_pct"]   = _coerce_pct(value.get("equity_end_pct"))
        return out

    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            start = first.get("Equity Allocation (%)")
            if start is not None:
                out["equity_start_pct"] = _coerce_pct(start)
        return out

    if isinstance(value, str):
        s = value.strip()
        # Try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                out["equity_start_pct"] = _coerce_pct(obj.get("equity_start_pct"))
                out["equity_end_pct"]   = _coerce_pct(obj.get("equity_end_pct"))
                return out
            if isinstance(obj, list) and obj:
                first = obj[0]
                if isinstance(first, dict):
                    start = first.get("Equity Allocation (%)")
                    if start is not None:
                        out["equity_start_pct"] = _coerce_pct(start)
                        return out
        except Exception:
            pass

        # Pattern like "{1, 80, 20}, {2, 80, 20}, ..."
        m = re.search(r"\{\s*\d+\s*,\s*(\d+)\s*,\s*\d+\s*\}", s)
        if m:
            out["equity_start_pct"] = _coerce_pct(m.group(1))
            return out

        # Keyed fallback
        start = _parse_percent_from_text(s, ["start", "equity_start", "year 1", "y1", "equity"])
        end   = _parse_percent_from_text(s, ["end", "equity_end", "near goal", "final"])
        out["equity_start_pct"] = start
        out["equity_end_pct"]   = end
        if start is not None or end is not None:
            return out

        # Fallback: first two numbers
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*%?", s)
        if nums:
            out["equity_start_pct"] = _coerce_pct(nums[0])
            if len(nums) > 1:
                out["equity_end_pct"] = _coerce_pct(nums[1])

    return out

def _bucket_from_fund_name(name: str) -> str:
    n = (name or "").lower()
    if "smallcap" in n or "small cap" in n:
        return "small"
    if "midcap" in n or "mid cap" in n:
        return "mid"
    if "nifty 50" in n or "nifty50" in n or "next 50" in n or "nifty next 50" in n or "bluechip" in n or "large" in n:
        return "large"
    return "other"

def parse_portfolio(value):
    """
    Parse 'portfolio' (list of dicts: Fund, Category, Type, Weight (%)) into aggregate buckets:
      - large_cap_pct
      - mid_cap_pct
      - small_cap_pct (optional)
      - debt_pct (optional)
    """
    out = {}

    def _agg_from_list(lst: List[dict]):
        large = mid = small = debt = other = 0.0
        for row in lst:
            try:
                w = float(row.get("Weight (%)", 0))
            except Exception:
                w = 0.0
            cat = (row.get("Category") or "").lower()
            fund = row.get("Fund") or ""
            type_str = (row.get("Type") or "").lower()

            if "debt" in cat or "liquid" in type_str:
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
        return _agg_from_list(value)

    if isinstance(value, str):
        s = value.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return _agg_from_list(obj)
        except Exception:
            pass

    return out

# ---------------- Story helpers for Explainer 1 ----------------

def _band_percent(center_pct: float, width: int = 5) -> tuple[int, int]:
    """
    Given a center percent (e.g., 77.0, 100.0, 320.0) return a clipped [lo, hi] band.
    We keep it simple and integer (e.g., 95–105).
    """
    if center_pct is None:
        center_pct = 0.0
    lo = max(0, int(round(center_pct)) - width)
    hi = int(round(center_pct)) + width
    return lo, hi

def _band_multiple(center_pct: float, width: int = 5) -> tuple[float, float]:
    """
    Convert a percent band to an X-multiple band, rounded to one decimal (e.g., 3.0–3.2×).
    """
    lo_pct, hi_pct = _band_percent(center_pct, width)
    lo_x = round(lo_pct / 100.0, 1)
    hi_x = round(hi_pct / 100.0, 1)
    return lo_x, hi_x

def _explainer1_story(name: str, goal_amount_str: str, funding_ratio_center_pct: float) -> str:
    """
    Explainer 1 with three cases:
      - Underfunded (FR < 0.90)
      - On track   (0.90 <= FR <= 1.10)
      - Overfunded (FR > 1.10)
    We never mention 'funding ratio' to users.
    """
    fr_dec = (funding_ratio_center_pct or 0.0) / 100.0
    lo_pct, hi_pct = _band_percent(funding_ratio_center_pct, width=5)

    if fr_dec < 0.90:
        # Underfunded — “Not there yet”
        return (
            f"Hi {name}, here’s where you stand. If you continue as you are, you’re likely to reach about "
            f"{lo_pct}–{hi_pct}% of your goal of {goal_amount_str} by the time you need it.\n\n"
            "That means we still have some work to do — but this is fixable. Small changes in how much you invest "
            "or how your money is allocated can close the gap and put you on track."
        )

    if fr_dec <= 1.10:
        # On track
        return (
            f"Hi {name}, here’s where you stand. If you continue as you are, you’re on track to reach your "
            f"goal of {goal_amount_str}, likely landing around {lo_pct}–{hi_pct}% by the time you need it.\n\n"
            "Why a range? Markets don’t move the same every year. Your plan is built to adapt within normal ups and downs."
        )

    # Overfunded — “Ahead of the curve”
    lo_x, hi_x = _band_multiple(funding_ratio_center_pct, width=5)
    return (
        f"Hi {name}, here’s where you stand. If you continue as you are, you’re on track to exceed your goal "
        f"of {goal_amount_str} — potentially reaching ~{lo_x}–{hi_x}× that amount by the time you need it.\n\n"
        "That gives you options: reach the goal sooner, aim bigger, or reduce the monthly amount and still get there comfortably."
    )

# ---------------- Tone: strategy & evolution ----------------

def _strategy_sentence(strategy: str, risk_profile: Optional[str]) -> str:
    s = (strategy or "").strip().title() or "Active"
    rp = (risk_profile or "").strip().lower()

    if s == "Passive":
        return ("We’ll keep it simple with low-cost index funds. "
                "They track the market, keep fees low, and let your money grow quietly.")

    if s == "Hybrid":
        return ("We’ll use a Hybrid plan — index funds for the steady base, "
                "and a small set of carefully chosen active funds to try and add a bit extra.")

    # Active
    if rp.startswith("conserv"):
        return ("We’ll go Active but stay careful — a few expert-managed funds, "
                "kept within sensible limits so risk doesn’t run ahead of you.")
    return ("We’ll go Active — a few expert-managed funds aiming for better-than-market returns. "
            "It can move more, so I’ll keep it within sensible limits.")

# ---------------- Story builders ----------------

def build_recommendation_parts(ctx: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns story split into 4 variables (human-advisor tone).
    """
    name        = (ctx.get("name") or "Investor").strip()
    target_str  = _money(ctx.get("target_corpus"))
    fr_center   = ctx.get("funding_ratio_pct", 0)

    strategy    = (ctx.get("strategy") or "").strip().title() or "Active"
    risk_prof   = (ctx.get("risk_profile") or "").strip()

    eq_start    = _pct(ctx.get("equity_start_pct"))
    large       = _pct(ctx.get("large_cap_pct"))
    mid         = _pct(ctx.get("mid_cap_pct"))
    small       = _pct(ctx.get("small_cap_pct"))
    provided_sum = sum(v for v in [large, mid, small] if v)
    debt        = _pct(ctx.get("debt_pct") if ctx.get("debt_pct") is not None else (100 - provided_sum))

    # -------- Explainer 1 (mirror moment)
    var1 = _explainer1_story(name=name, goal_amount_str=target_str, funding_ratio_center_pct=fr_center)

    # -------- Explainer 2 (plain language strategy)
    var2 = _strategy_sentence(strategy, risk_prof)

    # -------- Explainer 3 (bridge + direct evolution)
    if large > 0 or mid > 0 or small > 0:
        parts = []
        if large > 0: parts.append(f"{large}% in large companies for stability")
        if mid > 0:   parts.append(f"{mid}% in mid-sized companies for growth")
        if small > 0: parts.append(f"{small}% in smaller companies for extra growth")
        split_sentence = "; ".join(parts)

        var3 = (
            "Here’s how your plan changes over the years.\n\n"
            f"In the first year, we start with {eq_start}% in equities — split into {split_sentence}. "
            f"The rest sits in debt ({debt}%) to keep things steady. "
            "As the goal gets closer, we’ll move more into debt to lock in what you’ve earned."
        )
    else:
        var3 = (
            "Here’s how your plan changes over the years.\n\n"
            f"In the first year, we start with {eq_start}% in equities. "
            "The rest sits in debt to keep things steady. "
            "As the goal gets closer, we’ll move more into debt to lock in what you’ve earned."
        )

    # -------- Explainer 4 (simple, confident close)
    var4 = (
        "This isn’t guesswork. It’s a disciplined plan built for you. "
        "I’ll review and rebalance regularly so you stay on track."
    )

    return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}

def build_recommendation_story(ctx: Dict[str, Any]) -> str:
    """Optional: full story in one string with plain paragraph breaks."""
    parts = build_recommendation_parts(ctx)
    return "\n\n".join([parts["var1"], parts["var2"], parts["var3"], parts["var4"]])

# ---------------- Demo ----------------
if __name__ == "__main__":
    # Quick demo with a sample portfolio (4-column)
    sample_portfolio = [
        {"Fund":"Motilal Oswal Nifty Midcap 150 Index Fund(G)-Direct Plan","Category":"Equity","Type":"Passive","Weight (%)":30},
        {"Fund":"UTI Nifty 50 Index Fund(G)-Direct Plan","Category":"Equity","Type":"Passive","Weight (%)":25},
        {"Fund":"Nippon India Nifty Smallcap 250 Index Fund(G)-Direct Plan","Category":"Equity","Type":"Passive","Weight (%)":15},
        {"Fund":"ICICI Pru Liquid Fund(G)-Direct Plan","Category":"Debt","Type":"Liquid","Weight (%)":20},
        {"Fund":"ICICI Pru Nifty Next 50 Index Fund(G)-Direct Plan","Category":"Equity","Type":"Passive","Weight (%)":10},
    ]
    ctx = {
        "name": "GANESH KUMAR R",
        "target_corpus": 5e7,  # ₹5 Cr
        "funding_ratio_pct": 21,  # shows as ~16–26% band internally
        "strategy": "Passive",
        "risk_profile": "Conservative",
        "equity_start_pct": 80,
        **parse_portfolio(sample_portfolio),
    }
    print(build_recommendation_story(ctx))
