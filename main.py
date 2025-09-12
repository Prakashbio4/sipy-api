# ===================== main.py =====================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import importlib

# === Local imports ===
from glide_explainer import explain_glide_story
from strategy_explainer import explain_strategy_story
# Robust import for portfolio explainer
pe = importlib.import_module("portfolio_explainer")
assert hasattr(pe, "explain_portfolio_story"), "portfolio_explainer.explain_portfolio_story missing"

# === FastAPI app ===
app = FastAPI(title="SIPY Single Point API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Schemas =====================
class GlideExplainRequest(BaseModel):
    years_to_goal: int
    risk_pref: Optional[str] = None
    funding_gap: Optional[float] = None

class StrategyExplainRequest(BaseModel):
    strategy: str
    rationale: Optional[str] = None

class Fund(BaseModel):
    Fund: str
    Category: str
    Sub_category: Optional[str] = None
    Sub_category2: Optional[str] = None
    Weight: Optional[float] = None
    Weight_pct: Optional[float] = None

class PortfolioExplainRequest(BaseModel):
    portfolio: List[Fund]

class PlanInput(BaseModel):
    age: Optional[int] = None
    goal_years: Optional[int] = None
    risk_pref: Optional[str] = None
    monthly_investment: Optional[float] = None
    funding_gap: Optional[float] = None
    strategy: Optional[str] = None
    rationale: Optional[str] = None
    portfolio: Optional[List[Fund]] = None

# ===================== Debug & Health =====================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug/env")
def debug_env():
    import sys, pathlib
    return {
        "cwd": os.getcwd(),
        "files": sorted(os.listdir("."))[:200],
        "sys_path": sys.path,
        "code_dir": str(pathlib.Path(__file__).parent.resolve()),
    }

# ===================== Explainers =====================
@app.post("/explain/glide")
def explain_glide(req: GlideExplainRequest):
    try:
        return explain_glide_story(req.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/strategy")
def explain_strategy(req: StrategyExplainRequest):
    try:
        return explain_strategy_story(req.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/portfolio")
def explain_portfolio(req: PortfolioExplainRequest):
    """
    Returns a plain-language story about the portfolio funds.
    """
    try:
        df = pd.DataFrame([f.dict() for f in req.portfolio])

        # Normalize column names
        if "Sub-category" not in df.columns and "Sub_category" in df.columns:
            df = df.rename(columns={"Sub_category": "Sub-category"})
        if "Weight (%)" not in df.columns:
            if "Weight" in df.columns:
                df["Weight (%)"] = df["Weight"]
            elif "Weight_pct" in df.columns:
                df["Weight (%)"] = df["Weight_pct"]

        return pe.explain_portfolio_story(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===================== Single Point API =====================
@app.post("/plan/full")
def plan(req: PlanInput, request: Request):
    try:
        # Block 1: Glide Path
        glide_block = explain_glide_story(req.dict())

        # Block 2: Strategy
        strategy_block = explain_strategy_story(req.dict())

        # Block 3: Portfolio
        if req.portfolio:
            portfolio_for_story = pd.DataFrame([f.dict() for f in req.portfolio])
            if "Sub-category" not in portfolio_for_story.columns and "Sub_category" in portfolio_for_story.columns:
                portfolio_for_story = portfolio_for_story.rename(columns={"Sub_category": "Sub-category"})
            if "Weight (%)" not in portfolio_for_story.columns:
                if "Weight" in portfolio_for_story.columns:
                    portfolio_for_story["Weight (%)"] = portfolio_for_story["Weight"]
                elif "Weight_pct" in portfolio_for_story.columns:
                    portfolio_for_story["Weight (%)"] = portfolio_for_story["Weight_pct"]

            portfolio_block = pe.explain_portfolio_story(portfolio_for_story)
        else:
            portfolio_block = {"block_id": "block_3_portfolio", "story": "No portfolio provided."}

        return {
            "glide_explainer": glide_block,
            "strategy_explainer": strategy_block,
            "portfolio_explainer": portfolio_block,
        }

    except Exception as e:
        import traceback
        print("ERROR in /plan/full:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
