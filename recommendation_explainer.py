# recommendation_explainer.py
# Generates a short, layman-friendly 4-part explainer and provides Airtable helpers.
from typing import Dict, Any, List
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

def _range_plus_minus_5(center_pct: float) -> str:
    """Return 'lo–hi%' where lo/hi are center ± 5, clamped to [0,100]."""
    c = _pct(center_pct)
    lo = max(0, c - 5)
    hi = min(100, c + 5)
    return f"{lo}–{hi}%"

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

    Accepts:
      - dict: {"equity_start_pct":80, "equity_end_pct":10}
      - list[dict]: first item may have "Equity Allocation (%)"
      - JSON string of either form
      - string like "{1, 80, 20}, {2, 80, 20}, ..."  -> picks first tuple's middle number (80)
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
    if "smallcap" in n:
        return "small"
    if "midcap" in n:
        return "mid"
    if "nifty 50" in n or "nifty50" in n or "next 50" in n or "nifty next 50" in n or "bluechip" in n or "large" in n:
        return "large"
    return "other"

def parse_portfolio(value):
    """
    Parse 'portfolio' into aggregate buckets:
      - large_cap_pct
      - mid_cap_pct
      - small_cap_pct (optional)
      - debt_pct (optional)

    Accepts:
      - dict (simple %s)
      - JSON string of dict or list[dict]
      - list[dict] like your input with keys: Fund, Category, Sub-category, Weight (%)
    """
    out = {}
    # Direct dict
    if isinstance(value, dict):
        out["large_cap_pct"] = _coerce_pct(value.get("large_cap_pct"))
        out["mid_cap_pct"]   = _coerce_pct(value.get("mid_cap_pct"))
        out["small_cap_pct"] = _coerce_pct(value.get("small_cap_pct"))
        out["debt_pct"]      = _coerce_pct(value.get("debt_pct"))
        return out

    # List of funds
    def _agg_from_list(lst: List[dict]):
        large = mid = small = debt = other = 0
        for row in lst:
            try:
                w = float(row.get("Weight (%)", 0))
            except Exception:
                w = 0
            cat = (row.get("Category") or "").lower()
            fund = row.get("Fund") or ""

            if "debt" in cat or "liquid" in (row.get("Sub-category") or "").lower():
                debt += w
            elif "equity" in cat:
                b = _bucket_from_fund_name(fund)
                if b == "large": large += w
                elif b == "mid": mid += w
                elif b == "small": small += w
                else: other += w
        # Normalize if total != 100
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

    # JSON string
    if isinstance(value, str):
        s = value.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {
                    "large_cap_pct": _coerce_pct(obj.get("large_cap_pct")),
                    "mid_cap_pct": _coerce_pct(obj.get("mid_cap_pct")),
                    "small_cap_pct": _coerce_pct(obj.get("small_cap_pct")),
                    "debt_pct": _coerce_pct(obj.get("debt_pct")),
                }
            if isinstance(obj, list):
                return _agg_from_list(obj)
        except Exception:
            pass

        # Keyed extraction (fallback)
        large = _parse_percent_from_text(s, ["large", "large cap", "lcap"])
        mid   = _parse_percent_from_text(s, ["mid", "mid cap", "mcap"])
        debt  = _parse_percent_from_text(s, ["debt", "bonds", "fixed"])
        return {"large_cap_pct": _coerce_pct(large), "mid_cap_pct": _coerce_pct(mid), "debt_pct": _coerce_pct(debt)}

    return out

# ---------------- Airtable mapping ----------------

def airtable_record_to_ctx(record: dict) -> dict:
    """
    Map Investor_Inputs fields to the context expected by the story builders.
    """
    f = record.get("fields", record)

    name = f.get("User Name") or f.get("Name") or "Investor"
    target = f.get("Target Corpus")

    funding_ratio = f.get("funding_ratio")
    try:
        fr_center_pct = float(funding_ratio)
        # Accept either decimal or percent center
        if fr_center_pct <= 1.0:
            fr_center_pct *= 100.0
    except Exception:
        fr_center_pct = 0.0

    strategy = f.get("strategy")
    risk_profile = f.get("Risk Preference") or f.get("risk_preference")

    glide = parse_glide_path(f.get("glide_path"))
    port  = parse_portfolio(f.get("portfolio"))

    ctx = {
        "name": name,
        "target_corpus": target,
        "funding_ratio_pct": fr_center_pct,
        "strategy": strategy,
        "risk_profile": risk_profile,
        "equity_start_pct": glide.get("equity_start_pct"),
        "equity_end_pct": glide.get("equity_end_pct"),
        "large_cap_pct": port.get("large_cap_pct"),
        "mid_cap_pct": port.get("mid_cap_pct"),
        "small_cap_pct": port.get("small_cap_pct"),
        "debt_pct": port.get("debt_pct"),
    }
    return ctx

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
            "That means we still have some work to do — but the good news is, small changes in how much you invest "
            "or how your money is allocated can close that gap and put you firmly on track."
        )

    if fr_dec <= 1.10:
        # Fully funded — “On track”
        return (
            f"Hi {name}, here’s where you stand. If you continue as you are, you’re on track to reach your "
            f"goal of {goal_amount_str}, likely landing around {lo_pct}–{hi_pct}% by the time you need it.\n\n"
            "Why a range? Because markets don’t deliver the same return every year. Over the past 5 years, equities "
            "have averaged around 10–15% annually, and your plan is built to adapt within that range."
        )

    # Overfunded — “Ahead of the curve”
    lo_x, hi_x = _band_multiple(funding_ratio_center_pct, width=5)
    return (
        f"Hi {name}, here’s where you stand. If you continue as you are, you’re on track to exceed your goal "
        f"of {goal_amount_str} — potentially reaching ~{lo_x}–{hi_x}× that amount by the time you need it.\n\n"
        "That’s a great place to be. It means you’ll have options: you could achieve the goal sooner, aim for a bigger dream, "
        "or even reduce your monthly investment and still get there comfortably."
    )

# ---------------- Story builders ----------------

def _strategy_sentence(strategy: str, risk_profile: str | None) -> str:
    s = (strategy or "").strip().title() or "Active"
    rp = (risk_profile or "").strip().lower()
    if s == "Passive":
        if rp.startswith("conserv"):
            return ("Given your conservative preference, we recommend a Passive strategy — "
                    "low-cost index funds that mirror the market. It keeps costs low and reduces surprises "
                    "while you stay invested for growth.")
        return ("We recommend a Passive strategy — low-cost index funds that mirror the market, "
                "keeping things simple and cost-efficient while you stay invested.")
    if s == "Hybrid":
        return ("To balance growth and steadiness, we recommend a Hybrid strategy — "
                "a mix of index funds and focused active ideas.")
    # Active
    if rp.startswith("conserv"):
        # edge case: conservative + active (rare)
        return ("We recommend an Active strategy with care — expert-managed funds aiming to add value, "
                "used thoughtfully given your conservative preference.")
    return ("To help close the gap faster (with more ups and downs), we recommend an Active strategy — "
            "expert-managed funds that aim to add value over the market.")

def build_recommendation_parts(ctx: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns story split into 4 variables.
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
    # Debt: if not provided, deduce from remaining
    provided_sum = sum(v for v in [large, mid, small] if v)
    debt        = _pct(ctx.get("debt_pct") if ctx.get("debt_pct") is not None else (100 - provided_sum))

    # -------- Explainer 1 (new story-first logic)
    var1 = _explainer1_story(name=name, goal_amount_str=target_str, funding_ratio_center_pct=fr_center)

    # -------- Explainer 2 (unchanged)
    var2 = _strategy_sentence(strategy, risk_prof)

    # -------- Explainer 3 (unchanged; include small if present)
    if large > 0 or mid > 0 or small > 0:
        parts = []
        if large > 0: parts.append(f"{large}% in large companies for stability")
        if mid > 0:   parts.append(f"{mid}% in mid-sized companies for growth")
        if small > 0: parts.append(f"{small}% in smaller companies for extra growth potential")
        split_sentence = "; ".join(parts)
        var3 = (
            f"In Year 1, your money starts with {eq_start}% in equities — split into {split_sentence}. "
            f"The rest is in debt ({debt}%) for balance, and over time more will shift into debt to protect your savings."
        )
    else:
        var3 = (
            f"In Year 1, your money starts with {eq_start}% in equities. "
            f"The rest is in debt for balance, and over time more will shift into debt to protect your savings."
        )

    # -------- Explainer 4 (unchanged)
    var4 = (
        "This isn’t guesswork. It’s a disciplined plan built only for you. "
        "And we’ll review and rebalance regularly to keep you on track."
    )

    return {"var1": var1, "var2": var2, "var3": var3, "var4": var4}

def build_airtable_fields_for_story(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns an Airtable 'fields' dict keyed exactly as requested:
      - 'explainer 1', 'explainer 2', 'explainer 3', 'explainer 4'
    """
    parts = build_recommendation_parts(ctx)
    return {
        "explainer 1": parts["var1"],
        "explainer 2": parts["var2"],
        "explainer 3": parts["var3"],
        "explainer 4": parts["var4"],
    }

def build_recommendation_story(ctx: Dict[str, Any]) -> str:
    """Optional: full story in one string."""
    parts = build_recommendation_parts(ctx)
    return "\n\n".join([parts["var1"], parts["var2"], parts["var3"], parts["var4"]])

# ---------------- Demo ----------------
if __name__ == "__main__":
    # Quick demo with your sample portfolio & conservative risk
    sample_portfolio = [
        {"Fund":"Motilal Oswal Nifty Midcap 150 Index Fund(G)-Direct Plan","Category":"Equity","Sub-category":"Passive","Weight (%)":30},
        {"Fund":"UTI Nifty 50 Index Fund(G)-Direct Plan","Category":"Equity","Sub-category":"Passive","Weight (%)":25},
        {"Fund":"Nippon India Nifty Smallcap 250 Index Fund(G)-Direct Plan","Category":"Equity","Sub-category":"Passive","Weight (%)":15},
        {"Fund":"ICICI Pru Nifty Next 50 Index Fund(G)-Direct Plan","Category":"Equity","Sub-category":"Passive","Weight (%)":10},
        {"Fund":"ICICI Pru Liquid Fund(G)-Direct Plan","Category":"Debt","Sub-category":"Liquid","Weight (%)":20},
    ]
    ctx = {
        "name": "GANESH KUMAR R",
        "target_corpus": 5e7,  # ₹5 Cr
        "funding_ratio_pct": 21,  # shows as ~16–26% band internally
        "strategy": "Passive",
        "risk_profile": "Conservative",
        "equity_start_pct": 80,
        # derive large/mid/small/debt from the list
        "portfolio": sample_portfolio,
        **parse_portfolio(sample_portfolio),
    }
    print(build_recommendation_story(ctx))
