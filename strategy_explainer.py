# strategy_explainer.py
from typing import Dict, Any, Optional, Union

def explain_strategy_story(ctx_or_strategy: Union[Dict[str, Any], str],
                           years_to_goal: Optional[int] = None,
                           risk_preference: Optional[str] = None,
                           funding_ratio: Optional[float] = None) -> Dict[str, Any]:
    """
    Compatible with:
      - explain_strategy_story({"strategy":..., "years_to_goal":..., "risk_profile":...})
      - explain_strategy_story(strategy, years_to_goal, risk_preference, funding_ratio)
    (We ignore funding_ratio in the user-facing text.)
    """
    if isinstance(ctx_or_strategy, dict):
        s  = (ctx_or_strategy.get("strategy") or "").strip().title()
        yt = int(ctx_or_strategy.get("years_to_goal") or 0)
        rp = (ctx_or_strategy.get("risk_profile") or ctx_or_strategy.get("risk_preference") or "").strip().title() or "Moderate"
    else:
        s  = (ctx_or_strategy or "").strip().title()
        yt = int(years_to_goal or 0)
        rp = (risk_preference or "").strip().title() or "Moderate"

    # Plain-English reasons (no jargon, no funding-ratio mention)
    if s == "Passive":
        if yt <= 6:
            story = (
                f"**Passive** keeps things steady and low-cost for a {yt}-year goal. "
                f"Your **{rp}** preference shows up in the higher equity level early on, "
                f"while broad index funds keep the ride smoother as you get closer to the goal."
            )
        else:
            story = (
                f"**Passive** focuses on broad market growth at low cost. "
                f"It avoids manager guesswork and lets the glide path handle how much equity you carry over time."
            )
    elif s == "Hybrid":
        story = (
            f"**Hybrid** mixes index funds with select active ideas. "
            f"It aims for a little extra growth without making the journey complicated."
        )
    else:  # Active
        story = (
            f"**Active** leans on carefully chosen managers and factors to try for extra growth, "
            f"while the glide path still reduces risk as you near the goal."
        )

    return {
        "block_id": "block_2_strategy",
        "story": story,
        "data_points": {
            "strategy": s,
            "years_to_goal": yt,
            "risk_preference": rp,
        },
    }
