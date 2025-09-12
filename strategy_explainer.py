# strategy_explainer.py
from typing import Dict, Any

def explain_strategy_story(ctx: Dict[str, Any]) -> Dict[str, Any]:
    strategy      = (ctx.get("strategy") or "").strip().title()
    years         = int(ctx.get("years_to_goal") or 0)
    risk_raw      = (ctx.get("risk_profile") or "").strip().title()
    funding_ratio = float(ctx.get("funding_ratio") or 0.0)

    if funding_ratio >= 1.3:
        fr_label, fr_phrase = "ahead_of_pace", "ahead of pace"
    elif funding_ratio >= 0.7:
        fr_label, fr_phrase = "on_track", "on track"
    else:
        fr_label, fr_phrase = "behind_pace", "behind pace"

    if strategy == "Passive":
        # Explain why Passive even for Aggressive users: glide path handles aggression, Passive controls sequence/cost risk
        if years <= 6:
            story = (
                f"**Passive** is chosen to control sequence risk with a short horizon ({years} years) "
                f"and keep costs low. Your **{risk_raw}** preference is reflected in the equity level "
                f"of the glide path early on, while broad index exposure reduces single-fund surprises "
                f"as withdrawals approach. With funding ratio **{funding_ratio:.2f}** ({fr_phrase}), "
                f"this mix maximizes the chance of meeting the goal on time."
            )
        else:
            story = (
                f"**Passive** fits because you’re **{fr_phrase}** (funding ratio {funding_ratio:.2f}) "
                f"and broad market capture at low cost beats manager risk for your plan. "
                f"The **{risk_raw}** preference is honored via equity levels in the glide path."
            )
    elif strategy == "Hybrid":
        story = (
            f"**Hybrid** blends low-cost index exposure with selective active bets. "
            f"Given a {years}-year horizon and funding ratio {funding_ratio:.2f} ({fr_phrase}), "
            f"it seeks added alpha where durable while keeping overall risk and cost in check."
        )
    else:  # Active
        story = (
            f"**Active** is selected because alpha potential matters more in your case "
            f"(horizon {years} years, funding ratio {funding_ratio:.2f} — {fr_phrase}). "
            f"The design targets persistent factors/managers to push outcomes toward the goal."
        )

    return {
        "block_id": "block_2_strategy",
        "story": story,
        "data_points": {
            "strategy": strategy,
            "years_to_goal": years,
            "risk_preference": risk_raw,
            "funding_ratio": funding_ratio,
            "funding_status": fr_label,
        },
    }
