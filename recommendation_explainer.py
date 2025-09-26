# recommendation_explainer.py
# Generates a short, layman-friendly 4-part explainer and provides Airtable helpers.

from typing import Dict, Any
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
    Accepts dicts or JSON strings like:
      {"equity_start_pct":80, "equity_end_pct":10}
    Falls back to extracting numbers from free text.
    """
    out = {}
    if isinstance(value, dict):
        out["equity_start_pct"] = _coerce_pct(value.get("equity_start_pct"))
        out["equity_end_pct"]   = _coerce_pct(value.get("equity_end_pct"))
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
        except Exception:
            pass

        # Keyed extraction
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

def parse_portfolio(value):
    """
    Parse 'portfolio' into:
      - large_cap_pct
      - mid_cap_pct
      - debt_pct (optional)
    Accepts dicts or JSON strings like:
      {"large_cap_pct":35,"mid_cap_pct":25,"debt_pct":40}
    Also accepts free text e.g. "large:35%, mid:25, debt:40%".
    """
    out = {}
    if isinstance(value, dict):
        out["large_cap_pct"] = _coerce_pct(value.get("large_cap_pct"))
        out["mid_cap_pct"]   = _coerce_pct(value.get("mid_cap_pct"))
        out["debt_pct"]      = _coerce_pct(value.get("debt_pct"))
        return out

    if isinstance(value, str):
        s = value.strip()
        # Try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                out["large_cap_pct"] = _coerce_pct(obj.get("large_cap_pct"))
                out["mid_cap_pct"]   = _coerce_pct(obj.get("mid_cap_pct"))
                out["debt_pct"]      = _coerce_pct(obj.get("debt_pct"))
                return out
        except Exception:
            pass

        # Keyed extraction
        large = _parse_percent_from_text(s, ["large", "large cap", "lcap"])
        mid   = _parse_percent_from_text(s, ["mid", "mid cap", "mcap"])
        debt  = _parse_percent_from_text(s, ["debt", "bonds", "fixed"])
        out["large_cap_pct"] = large
        out["mid_cap_pct"]   = mid
        out["debt_pct"]      = debt
        if any(v is not None for v in out.values()):
            return out

    return out

# ---------------- Airtable mapping ----------------

def airtable_record_to_ctx(record: dict) -> dict:
    """
    Map Investor_Inputs fields to the context expected by the story builders.
      - "User Name"        -> name
      - "Target Corpus"    -> target_corpus
      - "funding_ratio"    -> funding_ratio_pct  (percent center or decimal)
      - "strategy"         -> strategy
      - "glide_path"       -> parsed equity_start_pct / equity_end_pct
      - "portfolio"        -> parsed large/mid/debt
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
    glide = parse_glide_path(f.get("glide_path"))
    port  = parse_portfolio(f.get("portfolio"))

    ctx = {
        "name": name,
        "target_corpus": target,
        "funding_ratio_pct": fr_center_pct,
        "strategy": strategy,
        "equity_start_pct": glide.get("equity_start_pct"),
        "equity_end_pct": glide.get("equity_end_pct"),
        "large_cap_pct": port.get("large_cap_pct"),
        "mid_cap_pct": port.get("mid_cap_pct"),
        "debt_pct": port.get("debt_pct"),
    }
    return ctx

# ---------------- Story builders ----------------

def build_recommendation_parts(ctx: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns story split into 4 variables:
      var1: Gap + market-range context (10–15% hardcoded)
      var2: Strategy sentence (Active/Passive/Hybrid, layman)
      var3: Year-1 allocation + shift-to-safety; omits split if large/mid not provided
      var4: Strong close + rebalance nudge
    """
    name        = (ctx.get("name") or "Investor").strip()
    target_str  = _money(ctx.get("target_corpus"))
    fr_center   = ctx.get("funding_ratio_pct", 0)
    fr_display  = _range_plus_minus_5(fr_center)

    strategy    = (ctx.get("strategy") or "").strip().title() or "Active"
    if strategy == "Passive":
        strat_line = "Passive strategy — low-cost index funds that mirror the market."
    elif strategy == "Hybrid":
        strat_line = "Hybrid strategy — a balanced mix of index funds and select active funds."
    else:
        strat_line = "Active strategy — carefully managed funds that aim to deliver better than the market."

    eq_start    = _pct(ctx.get("equity_start_pct"))
    large       = _pct(ctx.get("large_cap_pct"))
    mid         = _pct(ctx.get("mid_cap_pct"))
    debt        = _pct(ctx.get("debt_pct") if ctx.get("debt_pct") is not None else (100 - large - mid))

    var1 = (
        f"Hi {name}, here’s where you stand. If you continue as you are, "
        f"you’ll likely reach only {fr_display} of your goal of {target_str}.\n\n"
        "Why the range? Because markets don’t return the same number every time. "
        "Over the past 5 years, equities have averaged between 10–15% per year. "
        "That’s what we use to calculate your funding gap."
    )

    var2 = f"To bridge it, we recommend an {strat_line}"

    # If large/mid missing, keep it graceful.
    if (large > 0 or mid > 0):
        var3 = (
            f"In Year 1, your money starts with {eq_start}% in equities — split into "
            f"{large}% in large companies for stability and {mid}% in mid-sized companies for growth. "
            f"The rest is in debt for balance, and over time more will shift into debt to protect your savings."
        )
    else:
        var3 = (
            f"In Year 1, your money starts with {eq_start}% in equities. "
            f"The rest is in debt for balance, and over time more will shift into debt to protect your savings."
        )

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
    """Optional: full story in one string (if you need it anywhere)."""
    parts = build_recommendation_parts(ctx)
    return "\n\n".join([parts["var1"], parts["var2"], parts["var3"], parts["var4"]])

# ---------------- Demo (optional) ----------------
if __name__ == "__main__":
    demo_ctx = {
        "name": "Ramesh",
        "target_corpus": 2000000,   # ₹20 lakhs
        "funding_ratio_pct": 35,    # shows as 30–40%
        "strategy": "Active",
        "equity_start_pct": 80,
        "large_cap_pct": 35,
        "mid_cap_pct": 25,
        # debt omitted -> computed as remainder (40)
    }
    print(build_recommendation_story(demo_ctx))
