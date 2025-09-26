import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyairtable import Api

from recommendation_explainer import (
    build_airtable_fields_for_story,
    parse_portfolio,
    parse_glide_path,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Existing models -----------

class PortfolioInput(BaseModel):
    name: str
    time_horizon: int
    risk_preference: str
    target_corpus: float
    current_corpus: float

# ----------- Existing endpoint: trigger_processing -----------

@app.post("/trigger_processing/")
async def trigger_processing(request: Request):
    try:
        data = await request.json()
        record_id = data.get("record_id")
        if not record_id:
            raise HTTPException(status_code=400, detail="record_id is required")

        api_key = os.getenv("AIRTABLE_API_KEY")
        base_id = os.getenv("AIRTABLE_BASE_ID")
        table_name = os.getenv("AIRTABLE_TABLE_NAME", "Investor_Inputs")
        if not api_key or not base_id:
            raise HTTPException(status_code=500, detail="Missing Airtable credentials")

        api = Api(api_key)
        table = api.table(base_id, table_name)
        record = table.get(record_id)
        inputs = record.get("fields", {})

        # ----------------- Existing portfolio generation -----------------
        user_input_data = {
            "name": inputs.get("User Name") or inputs.get("Name"),
            "time_horizon": inputs.get("Time Horizon (yrs)"),
            "risk_preference": inputs.get("Risk Preference"),
            "target_corpus": inputs.get("Target Corpus"),
            "current_corpus": inputs.get("Current Corpus"),
        }
        from orchestrator import generate_portfolio
        processed_output = generate_portfolio(PortfolioInput(**user_input_data))

        # ---------- Build 4-part explainer fields (short story) ----------
        user_name = inputs.get("User Name") or inputs.get("Name") or "Investor"
        target_corpus = inputs.get("Target Corpus")

        # funding_ratio may be decimal (0.xx) — convert to percent
        try:
            fr_center_pct = float(processed_output.get("funding_ratio", 0))
            if fr_center_pct <= 1.0:
                fr_center_pct *= 100.0
        except Exception:
            fr_center_pct = 0.0

        # Year 1 equity: try engine → fallback to parsing Glide Path field → fallback to portfolio equity sum
        equity_start_pct = 0
        try:
            first_year = processed_output.get("glide_path", [])[0]
            equity_start_pct = int(first_year.get("Equity Allocation (%)", 0))
        except Exception:
            pass
        if equity_start_pct == 0:
            gp_parsed = parse_glide_path(inputs.get("glide_path"))
            equity_start_pct = int(gp_parsed.get("equity_start_pct") or 0)
        if equity_start_pct == 0:
            # derive from portfolio if it’s a list of funds (equity weights sum)
            try:
                portfolio_list = inputs.get("portfolio")
                if isinstance(portfolio_list, str):
                    portfolio_list = json.loads(portfolio_list)
                if isinstance(portfolio_list, list):
                    eq = sum(float(r.get("Weight (%)", 0)) for r in portfolio_list if (r.get("Category") or "").lower() == "equity")
                    equity_start_pct = int(round(eq))
            except Exception:
                pass

        # Parse Large/Mid/Small/Debt from Airtable 'portfolio' (list or json)
        portfolio_raw = inputs.get("portfolio")
        try:
            parsed_port = parse_portfolio(portfolio_raw)
        except Exception:
            parsed_port = {}
        large_cap_pct = parsed_port.get("large_cap_pct")
        mid_cap_pct   = parsed_port.get("mid_cap_pct")
        small_cap_pct = parsed_port.get("small_cap_pct")
        debt_pct      = parsed_port.get("debt_pct")

        ctx_for_story = {
            "name": user_name,
            "target_corpus": target_corpus,
            "funding_ratio_pct": fr_center_pct,
            "strategy": processed_output.get("strategy"),
            "risk_profile": inputs.get("Risk Preference"),
            "equity_start_pct": equity_start_pct,
            "large_cap_pct": large_cap_pct,
            "mid_cap_pct": mid_cap_pct,
            "small_cap_pct": small_cap_pct,
            "debt_pct": debt_pct,
        }
        explainer_fields = build_airtable_fields_for_story(ctx_for_story)
        # -----------------------------------------------------------------

        glide_path_str = json.dumps(processed_output.get("glide_path", []))

        update_data = {
            "strategy": processed_output["strategy"],
            "funding_ratio": processed_output["funding_ratio"],
            "glide_path": glide_path_str,
            "portfolio": json.dumps(processed_output["portfolio"]),
            "glide_explainer_story": processed_output["glide_explainer"]["story"],
            "strategy_explainer_story": processed_output["strategy_explainer"]["story"],
            "portfolio_explainer_story": processed_output["portfolio_explainer"]["story"],

            # Short explainer (4 parts)
            "explainer 1": explainer_fields.get("explainer 1"),
            "explainer 2": explainer_fields.get("explainer 2"),
            "explainer 3": explainer_fields.get("explainer 3"),
            "explainer 4": explainer_fields.get("explainer 4"),
        }

        table.update(record_id, update_data)
        return {"status": "success", "record_id": record_id, "updated_fields": list(update_data.keys())}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- New endpoint: generate explainer only ----------

@app.post("/generate_explainer/{record_id}")
def generate_explainer_only(record_id: str):
    """
    Fetch a record from Airtable, build 4-part explainer, and write it back
    without rerunning the portfolio engine.
    """
    try:
        api_key = os.getenv("AIRTABLE_API_KEY")
        base_id = os.getenv("AIRTABLE_BASE_ID")
        table_name = os.getenv("AIRTABLE_TABLE_NAME", "Investor_Inputs")
        if not api_key or not base_id:
            raise HTTPException(status_code=500, detail="Missing Airtable credentials")

        api = Api(api_key)
        table = api.table(base_id, table_name)
        rec = table.get(record_id)
        inputs = rec.get("fields", {})

        user_name = inputs.get("User Name") or inputs.get("Name") or "Investor"
        target_corpus = inputs.get("Target Corpus")

        funding_ratio = inputs.get("funding_ratio")
        try:
            fr_center_pct = float(funding_ratio)
            if fr_center_pct <= 1.0:
                fr_center_pct *= 100.0
        except Exception:
            fr_center_pct = 0.0

        gp_parsed = parse_glide_path(inputs.get("glide_path"))
        equity_start_pct = int(gp_parsed.get("equity_start_pct") or 0)

        parsed_port = parse_portfolio(inputs.get("portfolio"))
        large_cap_pct = parsed_port.get("large_cap_pct")
        mid_cap_pct   = parsed_port.get("mid_cap_pct")
        small_cap_pct = parsed_port.get("small_cap_pct")
        debt_pct      = parsed_port.get("debt_pct")

        ctx_for_story = {
            "name": user_name,
            "target_corpus": target_corpus,
            "funding_ratio_pct": fr_center_pct,
            "strategy": inputs.get("strategy"),
            "risk_profile": inputs.get("Risk Preference"),
            "equity_start_pct": equity_start_pct,
            "large_cap_pct": large_cap_pct,
            "mid_cap_pct": mid_cap_pct,
            "small_cap_pct": small_cap_pct,
            "debt_pct": debt_pct,
        }

        fields = build_airtable_fields_for_story(ctx_for_story)
        table.update(record_id, fields)
        return {"status": "ok", "record_id": record_id, "updated_fields": list(fields.keys())}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explainer: {e}")
