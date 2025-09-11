# strategy_explainer.py
from typing import Dict, Any, Optional

def explain_strategy_story(strategy: str,
                           years_to_goal: int,
                           risk_preference: str,
                           funding_ratio: Optional[float]) -> Dict[str, Any]:
    """
    Returns a layman, ≤3-sentence story about how the money will be managed.
    """
    s = (strategy or "").strip().lower()
    rp = (risk_preference or "").strip().capitalize() or "Moderate"
    horizon_label = "short" if years_to_goal <= 5 else ("mid-term" if years_to_goal <= 10 else "long-term")

    if funding_ratio is None:
        funding_status = "unknown"
    elif funding_ratio < 0.7:
        funding_status = "catch_up"
    elif funding_ratio >= 1.3:
        funding_status = "ahead"
    else:
        funding_status = "on_track"

    active_base = (
        "You’re in an active plan because right now your investments are falling short of the goal, "
        "but you still have time to build it up. "
        "This means strategically allocating more to high-potential opportunities to aggressively pursue growth, "
        "even if it comes with extra risk and cost. "
        f"That matches your {rp} profile and gives your plan the best shot to catch up."
    )
    passive_base = (
        "You’re in a passive plan because your goal is on track and there’s no need to take unnecessary risks. "
        "Most of your money is parked in index funds, riding the whole market at low cost so steady compounding does the work. "
        f"This fits your {rp} profile and keeps the journey simple and predictable."
    )
    hybrid_base = (
        "You’re in a hybrid plan because you need both reliability and a little extra growth and risk. "
        "A big part of your money compounds steadily in index funds, while another part is guided into active picks for added growth. "
        f"This balance reflects your {rp} and adapts as you move closer to the goal."
    )

    if s == "active":
        story = active_base
        if funding_status != "catch_up":
            story = (
                "You’re in an active plan because you want to pursue faster growth and you’ve got time to work with. "
                "This means strategically allocating more to high-potential opportunities to aggressively pursue growth, "
                "even if it comes with extra risk and cost. "
                f"That matches your {rp} profile and gives your plan the best shot to reach your goal."
            )
    elif s == "passive":
        if funding_status == "on_track":
            story = passive_base
        else:
            story = (
                "You’re in a passive plan to keep risk measured and costs low. "
                "Most of your money is parked in index funds, riding the whole market so steady compounding does the work without taking unnecessary risks. "
                f"This fits your {rp} profile and keeps the journey simple and predictable."
            )
        if years_to_goal <= 3:
            story = (
                "With your goal close, a passive plan keeps things steady and low-cost. "
                "Most of your money is in index funds, riding the whole market so compounding does the work without taking unnecessary risks. "
                f"This fits your {rp} profile and keeps the path predictable."
            )
    else:  # hybrid (default)
        story = hybrid_base
        if funding_status == "catch_up":
            story = story[:-1] + "—adding a little extra push where it helps most."
        elif funding_status == "ahead":
            story = story[:-1] + "—preserving efficiency while keeping options open."

    return {
        "block_id": "block_2_strategy",
        "story": story,
        "data_points": {
            "strategy": strategy,
            "years_to_goal": int(years_to_goal),
            "risk_preference": rp,
            "funding_ratio": None if funding_ratio is None else float(funding_ratio),
            "funding_status": funding_status,
            "horizon_label": horizon_label
        }
    }
