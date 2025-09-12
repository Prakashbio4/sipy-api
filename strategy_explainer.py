# strategy_explainer.py
from typing import Dict, Any, Optional, Union

def explain_strategy_story(ctx_or_strategy: Union[Dict[str, Any], str],
                           years_to_goal: Optional[int] = None,
                           risk_preference: Optional[str] = None,
                           funding_ratio: Optional[float] = None) -> Dict[str, Any]:
    """
    Compatible with:
      - explain_strategy_story({"strategy":..., "years_to_goal":..., "risk_profile":..., "funding_ratio":...})
      - explain_strategy_story(strategy, years_to_goal, risk_preference, funding_ratio)
    """
    if isinstance(ctx_or_strategy, dict):
        s  = (ctx_or_strategy.get("strategy") or "").strip().title()
        yt = int(ctx_or_strategy.get("years_to_goal") or 0)
        rp = (ctx_or_strategy.get("risk_profile") or ctx_or_strategy.get("risk_preference") or "").strip().title() or "Moderate"
        fr = ctx_or_strategy.get("funding_ratio")
        fr = None if fr is None else float(fr)
    else:
        s  = (ctx_or_strategy or "").strip().title()
        yt = int(years_to_goal or 0)
        rp = (risk_preference or "").strip().title() or "Moderate"
        fr = None if funding_ratio is None else float(funding_ratio)

    # Funding status label
    if fr is None:
        fr_label, fr_phrase = "unknown", "not yet available"
    elif fr < 0.7:
        fr_label, fr_phrase = "behind_pace", "behind pace"
    elif fr >= 1.3:
        fr_label, fr_phrase = "ahead_of_pace", "ahead of pace"
    else:
        fr_label, fr_phrase = "on_track", "on track"

    # Polished narratives (no "Aggressive + Passive" contradiction)
    if s == "Passive":
        if yt <= 6:
            story = (
                f"**Passive** keeps sequence risk and costs low with a {yt}-year horizon. "
                f"Your **{rp}** preference is honored via the higher equity level early in the glide path, "
                f"while broad index exposure avoids single-fund surprises as withdrawals get closer. "
                f"Given you’re **{fr_phrase}** (funding ratio {fr:.2f} if available), this maximizes the chance of meeting the goal on time."
            )
        else:
            story = (
                f"**Passive** fits because you’re **{fr_phrase}** (funding ratio {fr:.2f} if available) and broad, low-cost market capture "
                f"beats manager risk for this plan. Your **{rp}** preference is reflected in the glide-path equity, not extra strategy risk."
            )
    elif s == "Hybrid":
        story = (
            f"**Hybrid** blends low-cost index exposure with selective active ideas. "
            f"With a {yt}-year horizon and funding ratio {('%.2f' % fr) if fr is not None else 'n/a'} ({fr_phrase}), "
            f"it seeks added alpha where durable while keeping overall risk and cost in check."
        )
    else:  # Active
        story = (
            f"**Active** is chosen because alpha potential matters more in your case "
            f"(horizon {yt} years, funding ratio {('%.2f' % fr) if fr is not None else 'n/a'} — {fr_phrase}). "
            f"The portfolio targets persistent factors/managers to push outcomes toward the goal."
        )

    return {
        "block_id": "block_2_strategy",
        "story": story,
        "data_points": {
            "strategy": s,
            "years_to_goal": yt,
            "risk_preference": rp,
            "funding_ratio": fr,
            "funding_status": fr_label,
            "horizon_label": "short" if yt <= 5 else ("mid_term" if yt <= 10 else "long_term"),
        },
    }
