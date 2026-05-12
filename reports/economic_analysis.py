from __future__ import annotations

import pandas as pd

from project_config import BUDGET_LINES, MONTHLY_ENERGY_COST_TL, PROJECT_BUDGET_TL, WORK_PACKAGES


def build_economic_scenarios() -> pd.DataFrame:
    scenarios = [
        ("Muhafazakar", 0.02),
        ("Beklenen", 0.035),
        ("Iyimser", 0.05),
    ]
    rows = []
    for name, rate in scenarios:
        monthly = MONTHLY_ENERGY_COST_TL * rate
        rows.append(
            {
                "scenario": name,
                "saving_rate_pct": rate * 100,
                "monthly_saving_tl": monthly,
                "annual_saving_tl": monthly * 12,
                "payback_months": PROJECT_BUDGET_TL / monthly,
            }
        )
    return pd.DataFrame(rows).round(2)


def build_budget_summary() -> pd.DataFrame:
    return pd.DataFrame(BUDGET_LINES, columns=["budget_line", "amount_tl"])


def build_work_package_table() -> pd.DataFrame:
    return pd.DataFrame(WORK_PACKAGES, columns=["code", "name", "period", "person_hours"])
