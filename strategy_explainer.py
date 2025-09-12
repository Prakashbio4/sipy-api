# ===================== strategy_explainer.py =====================
from typing import Dict, Any, Optional, Union


def explain_strategy_story(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a layman, ≤3-sentence story about how the money will be managed.
    """
    s = (ctx.get("strategy") or "").strip().title()
    yt = int(ctx.get("years_to_goal") or 0)
    rp = (ctx.get("risk_profile") or "Moderate").strip().capitalize() or "Moderate"
    
    story = ""
    if s == "Passive":
        story = (
            f"Your plan uses a **Passive** strategy. For a goal with a **{yt}-year horizon**, "
            "this approach helps keep costs low and avoids the risk of poor fund manager decisions. "
            f"This passive approach works well with your **{rp}** risk preference because it relies "
            "on the market's long-term growth, while the glide path handles the necessary risk "
            "reduction as your goal approaches."
        )
    elif s == "Hybrid":
        story = (
            f"Your plan uses a **Hybrid** strategy. This approach combines the stability and low cost of passive index funds "
            "with a small, targeted portion of active funds. This blend aims to capture the market's growth while "
            "still allowing for a little extra upside potential."
        )
    elif s == "Active":
        story = (
            f"Your plan uses an **Active** strategy. For a goal with a **{yt}-year horizon** that is currently "
            f"under-funded, this strategy strategically allocates more to high-potential funds to aggressively pursue growth. "
            f"This matches your **{rp}** risk preference and gives your plan the best shot to meet your goal."
        )
    else:
        story = "The investment strategy for your plan could not be determined."

    return {
        "block_id": "block_2_strategy",
        "story": story,
        "data_points": {
            "strategy": s,
            "years_to_goal": yt,
            "risk_preference": rp,
        }
    }