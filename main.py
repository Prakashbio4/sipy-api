from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# === Load data once at app startup ===
active_eq = pd.read_csv("data/active_equity_portfolio.csv")
passive_eq = pd.read_csv("data/passive_equity_portfolio.csv")
hybrid_eq = pd.read_csv("data/hybrid_equity_portfolio.csv")
tmf = pd.read_csv("data/tmf_selected.csv")
duration_debt = pd.read_csv("data/debt_duration_selected.csv")

# === Input Schema ===
class PortfolioInput(BaseModel):
    monthly_investment: float
    target_corpus: float
    years_to_goal: int
    risk_profile: str  # 'conservative', 'moderate', 'aggressive'

# --- Utility Functions (from orchestration engine) ---

def calculate_funding_ratio(monthly_investment, target_corpus,years):
    expected_return = 0.13  # Fixed expected return
    r = (1 + expected_return) ** (1/12) - 1
    n = years * 12
    fv = monthly_investment * (((1 + r) ** n - 1) / r)
    return fv, fv / target_corpus

def generate_step_down_glide_path(time_to_goal, funding_ratio, risk_profile):
    funding_ratio = float(funding_ratio)
    glide_path = []
    short_goal_equity_cap = 30 if time_to_goal <= 3 else 100

    if funding_ratio > 2.0:
        base_equity = 70
        derisk_start = int(time_to_goal * 0.4)
    elif funding_ratio > 1.0:
        base_equity = 80
        derisk_start = int(time_to_goal * 0.5)
    else:
        base_equity = 90
        derisk_start = int(time_to_goal * 0.6)

    if risk_profile == 'conservative':
        base_equity -= 10
    elif risk_profile == 'aggressive':
        base_equity += 5

    base_equity = min(base_equity, short_goal_equity_cap)
    final_equity = 10
    derisk_years = time_to_goal - derisk_start
    equity_step = (base_equity - final_equity) / derisk_years if derisk_years > 0 else 0

    for year in range(1, time_to_goal + 1):
        if year <= derisk_start:
            equity = base_equity
        else:
            equity = base_equity - equity_step * (year - derisk_start)
        equity = int(round(max(min(equity, short_goal_equity_cap), 0) / 5) * 5)
        debt = 100 - equity
        glide_path.append({'Year': year, 'Equity Allocation (%)': equity, 'Debt Allocation (%)': debt})

    return pd.DataFrame(glide_path)

def choose_strategy(time_to_goal, risk_profile, funding_ratio):
    if time_to_goal <= 3:
        return "Passive"
    if funding_ratio >= 1.3:
        funding_status = 'overfunded'
    elif funding_ratio < 0.7:
        funding_status = 'underfunded'
    else:
        funding_status = 'balanced'

    if time_to_goal <= 5:
        if funding_status == 'overfunded':
            return "Passive"
        elif risk_profile == 'aggressive' and funding_status == 'underfunded':
            return "Hybrid"
        else:
            return "Passive"
    elif 6 <= time_to_goal <= 10:
        if risk_profile == 'conservative':
            return "Passive"
        elif risk_profile == 'moderate':
            return "Hybrid" if funding_status == 'balanced' else "Passive"
        elif risk_profile == 'aggressive':
            return "Active" if funding_status == 'underfunded' else "Hybrid"
    else:
        if risk_profile == 'conservative':
            return "Passive"
        elif funding_status == 'underfunded':
            return "Active"
        elif funding_status == 'overfunded':
            return "Hybrid" if risk_profile != 'conservative' else "Passive"
        else:
            return "Active" if risk_profile == 'aggressive' else "Hybrid"

def get_debt_allocation(duration_debt_df, time_to_goal, debt_pct):
    duration_debt_df = duration_debt_df.copy()
    if time_to_goal > 3:
        allowed_categories = ['Liquid']
    else:
        allowed_categories = ['Liquid', 'Ultra Short']
    debt_filtered = duration_debt_df[duration_debt_df['Sub-category'].str.strip().isin(allowed_categories)].copy()
    if debt_filtered.empty:
        raise ValueError("⚠️ No suitable debt funds found for the selected duration.")
    debt_filtered['Fund'] = debt_filtered['Scheme Name']
    debt_filtered['Category'] = 'Debt'
    debt_filtered['Weight (%)'] = debt_pct / len(debt_filtered)
    debt_filtered['Weight (%)'] = debt_filtered['Weight (%)'].apply(lambda x: max(10, round(x / 5) * 5))
    weight_sum = debt_filtered['Weight (%)'].sum()
    debt_filtered['Weight (%)'] = debt_filtered['Weight (%)'] / weight_sum * debt_pct
    debt_filtered['Weight (%)'] = debt_filtered['Weight (%)'].apply(lambda x: round(x / 5) * 5)
    diff = debt_pct - debt_filtered['Weight (%)'].sum()
    if abs(diff) >= 5:
        idx = debt_filtered['Weight (%)'].idxmax()
        debt_filtered.loc[idx, 'Weight (%)'] += diff
    return debt_filtered[['Fund', 'Category', 'Sub-category', 'Weight (%)']]

# === API Endpoint ===
@app.post("/generate_portfolio/")
def generate_portfolio(user_input: PortfolioInput):
    fv, funding_ratio = calculate_funding_ratio(
        user_input.monthly_investment,
        user_input.target_corpus,
        user_input.years_to_goal
    )
    strategy = choose_strategy(user_input.years_to_goal, user_input.risk_profile, funding_ratio)
    glide_path = generate_step_down_glide_path(user_input.years_to_goal, funding_ratio, user_input.risk_profile)
    glide_equity_pct = int(glide_path.iloc[0]['Equity Allocation (%)'])

    # Equity Selection
    if strategy == "Active":
        eq_df = active_eq.copy()
        eq_df['Sub-category'] = 'Active'
    elif strategy == "Passive":
        eq_df = passive_eq.copy()
        eq_df['Sub-category'] = 'Passive'
    elif strategy == "Hybrid":
        eq_df = hybrid_eq.copy()
        eq_df.rename(columns={'Type': 'Sub-category'}, inplace=True)
    eq_df['Category'] = 'Equity'
    eq_df['Weight'] = 1 / len(eq_df)
    eq_df['Weight (%)'] = eq_df['Weight'] * glide_equity_pct / 100
    eq_df.rename(columns={col: 'Fund' for col in eq_df.columns if 'fund' in col.lower()}, inplace=True)
    eq_df['Weight (%)'] = eq_df['Weight (%)'].apply(lambda x: max(10, round(x / 5) * 5))
    eq_df['Weight (%)'] = eq_df['Weight (%)'] / eq_df['Weight (%)'].sum() * glide_equity_pct
    eq_df['Weight (%)'] = eq_df['Weight (%)'].apply(lambda x: round(x / 5) * 5)

    # Debt Selection
    debt_pct = 100 - glide_equity_pct
    selected_debt = get_debt_allocation(duration_debt, user_input.years_to_goal, debt_pct)

    # Combine
    combined = pd.concat([
        eq_df[['Fund', 'Category', 'Sub-category', 'Weight (%)']],
        selected_debt[['Fund', 'Category', 'Sub-category', 'Weight (%)']]
    ], ignore_index=True)
    combined['Weight (%)'] = combined['Weight (%)'].apply(lambda x: round(x / 5) * 5)
    diff = 100 - combined['Weight (%)'].sum()
    if diff != 0:
        idx = combined['Weight (%)'].idxmax()
        combined.at[idx, 'Weight (%)'] += diff

    return {
        "strategy": strategy,
        "glide_path": glide_path.to_dict(orient="records"),
        "portfolio": combined.to_dict(orient="records")
    }
