from typing import Dict, List, Optional
import json
from pathlib import Path

# === 1) Explainer ===

PHASE_HIGH = "high_growth"        # equity >= 70
PHASE_BAL = "balanced_growth"     # 30 <= equity < 70
PHASE_SAFE = "safety_first"       # equity < 30

def _phase(eq: float) -> str:
    if eq >= 70:
        return PHASE_HIGH
    if eq >= 30:
        return PHASE_BAL
    return PHASE_SAFE

def _pick_points(
    bands: Optional[List[Dict]] = None,
    equity_start_pct: Optional[float] = None,
    equity_end_pct: Optional[float] = None,
    years_to_goal: Optional[int] = None,
):
    """
    Return integer start/mid/end equity and end_year.
    Prefers bands with keys like {"Year", "Equity Allocation (%)"} (your prod format).
    """
    def _clamp_pct(x: float) -> int:
        return max(0, min(100, round(float(x))))

    if bands and len(bands) >= 2:
        # Sort by year
        def _yr(d): return int(d.get("Year") or d.get("year") or 0)
        b = sorted(bands, key=_yr)

        # Equity extractor
        def _eq(d):
            return float(d.get("Equity Allocation (%)") or d.get("equity_pct") or d.get("equity") or 0.0)

        start_eq = _eq(b[0])
        end_eq = _eq(b[-1])
        start_year = _yr(b[0])
        end_year = _yr(b[-1]) if _yr(b[-1]) else (years_to_goal or 0)

        # Midpoint year
        span = max(1, (end_year - start_year) if end_year and start_year is not None else (years_to_goal or 1))
        target_mid_year = start_year + span / 2
        mid_band = min(b, key=lambda x: abs(_yr(x) - target_mid_year) if _yr(x) else float("inf"))
        mid_eq = _eq(mid_band)

        # --- TWEAK #1: clamp all pcts to [0, 100] right before returning
        start_eq = _clamp_pct(start_eq)
        mid_eq   = _clamp_pct(mid_eq)
        end_eq   = _clamp_pct(end_eq)

        return start_eq, mid_eq, end_eq, (end_year if end_year else (years_to_goal or 0))

    # Fallback if only start/end provided
    if equity_start_pct is not None and equity_end_pct is not None:
        start_eq = float(equity_start_pct)
        end_eq = float(equity_end_pct)
        mid_eq = (start_eq + end_eq) / 2.0

        # --- TWEAK #1 also applied for fallback path
        start_eq = _clamp_pct(start_eq)
        mid_eq   = _clamp_pct(mid_eq)
        end_eq   = _clamp_pct(end_eq)

        return start_eq, mid_eq, end_eq, int(years_to_goal or 0)

    # Safe defaults
    return 55, 45, 35, int(years_to_goal or 0)


def explain_glide_story(common: Dict, glide_path: Dict) -> Dict:
    """
    Produce a single flowing paragraph that adapts wording to:
      - Smooth taper
      - Flat-then-shift
      - Direct drop
    Returns: {"block_id", "story", "data_points": {...}}
    """
    years = int(common.get("goal", {}).get("years_to_goal", 0) or 0)

    bands = glide_path.get("bands") or glide_path.get("glide_path")

    # --- TWEAK #2: early guard for missing/insufficient bands
    if not bands or len(bands) < 2:
        return {
            "block_id": "block_1_glide_path",
            "story": "We’ll show your glide path once we have enough data.",
            "data_points": {}
        }

    start_eq, mid_eq, end_eq, end_year = _pick_points(
        bands=bands,
        equity_start_pct=glide_path.get("equity_start_pct"),
        equity_end_pct=glide_path.get("equity_end_pct"),
        years_to_goal=years,
    )
    end_year = end_year or years
    end_debt = max(0, 100 - end_eq)

    # Classify phases
    start_ph, mid_ph, end_ph = _phase(start_eq), _phase(mid_eq), _phase(end_eq)
    NEAR = 5

    # Shape detection
    flat_then_shift = abs(mid_eq - start_eq) <= NEAR and abs(end_eq - start_eq) > NEAR
    smooth_taper = (start_eq - mid_eq) > NEAR and (mid_eq - end_eq) > NEAR
    direct_drop = (
        start_ph == PHASE_HIGH and (
            (mid_ph == PHASE_SAFE and end_ph == PHASE_SAFE) or
            (abs(mid_eq - end_eq) <= NEAR and (start_eq - mid_eq) > 2*NEAR and end_ph in (PHASE_BAL, PHASE_SAFE))
        )
    )

    # Narrative
    start_sentence = f"At first, your plan kicks off with {start_eq}% equity powering strong growth"

    if direct_drop:
        middle = f". Then, at the right point, equity shifts directly to around {mid_eq}% to prioritize safety"
        end_sentence = f". By year {end_year}, about {end_debt}% rests in debt, keeping your corpus protected"
    elif flat_then_shift:
        middle = f". It holds steady for the early years, then shifts to around {mid_eq}% for a more balanced mix"
        end_sentence = f". By year {end_year}, nearly {end_debt}% is in debt, shielding your corpus when you need it most"
    elif smooth_taper:
        middle = f". Through the middle years, equity moves to about {mid_eq}%, balancing growth with stability"
        end_sentence = f". By year {end_year}, nearly {end_debt}% rests in debt so your goal stays protected"
    else:
        middle = f". Over time, the allocation adjusts toward about {mid_eq}% equity to steady the ride"
        end_sentence = f". By year {end_year}, roughly {end_debt}% sits in debt, adding a safety cushion for your goal"

    story = (start_sentence + middle + end_sentence + ".")

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


# === 2) Demo helper ===
def demo(common: Dict, bands: List[Dict], label: str):
    out = explain_glide_story(common, {"bands": bands})
    print(f"\n[{label}]")
    print(out["story"])
    print("data_points:", out["data_points"])


# === 3) Run with your production JSON if present ===
prod_path = Path("/mnt/data/response_1757567760749.json")
if prod_path.exists():
    with open(prod_path, "r", encoding="utf-8") as f:
        sample = json.load(f)
    bands = sample.get("glide_path", [])
    years_to_goal = bands[-1]["Year"] if bands else 0
    common = {"user": {"risk_preference": "Moderate"}, "goal": {"years_to_goal": years_to_goal}}
    demo(common, bands, "Production sample")
