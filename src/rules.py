# src/rules.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

# =========================================================
# 🧮 RULES ENGINE (No-ML) — Fixed policy version
# =========================================================
def check_loan_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the loan eligibility for each applicant based on fixed criteria.
    Adds a new column 'Eligibility' (list) or ['Ineligible'] if none match.

    NOTE:
    - 504 rule has NO 'avg NOI (2y)' condition
    - 504 does NOT require any specific purpose (no 'must include one of')
    - 504 still requires collateral
    - All loans require a valid purpose
    """
    required_bool = [
        "For Profit", "Fast Approval", "Collateral Availability",
        "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
        "Inventory Purchase", "Real Estate Acquisition or Improvement",
        "Business Acquisition or Buyout", "Refinancing Existing Debt",
        "Emergency Funds", "Franchise Financing", "Contract Financing",
        "Licensing or Permits", "Line of Credit Establishment"
    ]
    required_num = [
        "Personal Credit Score", "Business Credit Score",
        "DSCR (latest year)", "Annual Revenue (latest year)", "Loan Amount",
        "Years in Business", "Net Profit Margin",
        "NOI (1 year ago)", "NOI (2 years ago)",
        "Industry Experience", "Managerial Experience"
    ]

    df = df.copy()
    for c in required_bool:
        if c not in df.columns:
            df[c] = False
    for c in required_num:
        if c not in df.columns:
            df[c] = 0

    df[required_bool] = df[required_bool].fillna(False).astype(bool)
    df[required_num] = df[required_num].fillna(0)

    purposes: List[str] = [
        "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
        "Inventory Purchase", "Real Estate Acquisition or Improvement",
        "Business Acquisition or Buyout", "Refinancing Existing Debt",
        "Emergency Funds", "Franchise Financing", "Contract Financing",
        "Licensing or Permits", "Line of Credit Establishment"
    ]

    def determine_eligibility(row: pd.Series) -> List[str]:
        eligible: List[str] = []

        pcs = float(row["Personal Credit Score"])
        bcs = float(row["Business Credit Score"])
        dscr = float(row["DSCR (latest year)"])
        yib = float(row["Years in Business"])
        collateral = bool(row["Collateral Availability"])
        loan_amt = float(row["Loan Amount"])
        fast_approval = bool(row["Fast Approval"])
        npm = float(row["Net Profit Margin"])
        ind_exp = float(row["Industry Experience"])
        mng_exp = float(row["Managerial Experience"])
        for_profit = bool(row["For Profit"])

        valid_purpose = any(bool(row.get(p, False)) for p in purposes)

        # 7(a)
        if (
            for_profit
            and pcs >= 680
            and bcs >= 160
            and dscr >= 1.15
            and yib >= 2
            and 500001 <= loan_amt <= 5_000_000
            and valid_purpose
            and not bool(row["Real Estate Acquisition or Improvement"])
            and not bool(row["Emergency Funds"])
        ):
            eligible.append("7(a)")

        # 8(a)
        if (
            (not for_profit)
            and (not fast_approval)
            and yib >= 2
            and ind_exp >= 2
            and mng_exp >= 2
            and valid_purpose
            and not bool(row["Franchise Financing"])
            and not bool(row["Line of Credit Establishment"])
        ):
            eligible.append("8(a)")

        # 504  (no 'must include one of', no avg NOI)
        if (
            for_profit
            and pcs >= 680
            and dscr >= 1.15
            and npm > 0
            and yib >= 2
            and loan_amt <= 5_500_000
            and valid_purpose
            and collateral
            and not bool(row["Working Capital"])
            and not bool(row["Refinancing Existing Debt"])
            and not bool(row["Emergency Funds"])
        ):
            eligible.append("504")

        # Express
        if (
            for_profit
            and fast_approval
            and pcs >= 680
            and bcs >= 160
            and dscr >= 1.15
            and loan_amt <= 500_000
            and valid_purpose
            and not bool(row["Real Estate Acquisition or Improvement"])
            and not bool(row["Business Acquisition or Buyout"])
        ):
            eligible.append("Express")

        return eligible if eligible else ["Ineligible"]

    df["Eligibility"] = df.apply(determine_eligibility, axis=1)
    return df


# =========================================================
# 🔧 CONFIGURABLE RULES (loan-specific keys)
# =========================================================
def default_rules() -> Dict:
    """
    Defaults for UI configuration — loan-specific keys.
    (no global min thresholds; each loan section includes its own)
    """
    return {
        # Purpose universe
        "all_purposes": [
            "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
            "Inventory Purchase", "Real Estate Acquisition or Improvement",
            "Business Acquisition or Buyout", "Refinancing Existing Debt",
            "Emergency Funds", "Franchise Financing", "Contract Financing",
            "Licensing or Permits", "Line of Credit Establishment"
        ],

        # ---------- 7(a) ----------
        "7a_requires_for_profit": True,
        "7a_min_personal_credit": 680,
        "7a_min_business_credit": 160,
        "7a_min_dscr": 1.15,
        "7a_min_years_in_business": 2,
        "7a_loan_min": 500_001,
        "7a_loan_max": 5_000_000,
        "7a_exclude_purposes": ["Real Estate Acquisition or Improvement", "Emergency Funds"],

        # ---------- 8(a) ----------
        "enable_8a": False,
        "8a_requires_for_profit": False,
        "8a_requires_fast_approval": False,  # NOT fast approval
        "8a_min_years_in_business": 2,
        "8a_min_industry_exp": 2,
        "8a_min_managerial_exp": 2,
        "8a_exclude_purposes": ["Franchise Financing", "Line of Credit Establishment"],

        # ---------- 504 ----------
        "504_requires_for_profit": True,
        "504_requires_collateral": True,
        "504_min_personal_credit": 680,
        "504_min_dscr": 1.15,
        "504_min_net_profit_margin": 0.0,
        "504_min_years_in_business": 2,
        "504_max_loan": 5_500_000,
        "504_exclude_purposes": ["Working Capital", "Refinancing Existing Debt", "Emergency Funds"],

        # ---------- Express ----------
        "express_requires_for_profit": True,
        "express_requires_fast_approval": True,
        "express_min_personal_credit": 680,
        "express_min_business_credit": 160,
        "express_min_dscr": 1.15,
        "express_max_loan": 500_000,
        "express_exclude_purposes": ["Real Estate Acquisition or Improvement", "Business Acquisition or Buyout"],
    }


def _merge_rules(rules: Dict | None) -> Dict:
    base = default_rules()
    if rules:
        base.update(rules)
    return base


def check_loan_eligibility_configurable(df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """
    Loan eligibility with per-loan threshold keys.
    No 'must include one of' and no avg NOI for 504.
    """
    rules = _merge_rules(rules)

    def has_any(row, names): return any(bool(row.get(p, False)) for p in names)
    def has_none(row, names): return not any(bool(row.get(p, False)) for p in names)

    required_bool = [
        "For Profit", "Fast Approval", "Collateral Availability",
        "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
        "Inventory Purchase", "Real Estate Acquisition or Improvement",
        "Business Acquisition or Buyout", "Refinancing Existing Debt",
        "Emergency Funds", "Franchise Financing", "Contract Financing",
        "Licensing or Permits", "Line of Credit Establishment"
    ]
    required_num = [
        "Personal Credit Score", "Business Credit Score",
        "DSCR (latest year)", "Loan Amount",
        "Years in Business", "Net Profit Margin",
        "Industry Experience", "Managerial Experience"
    ]
    df = df.copy()
    for c in required_bool:
        if c not in df.columns: df[c] = False
    for c in required_num:
        if c not in df.columns: df[c] = 0
    df[required_bool] = df[required_bool].fillna(False).astype(bool)
    df[required_num] = df[required_num].fillna(0)

    purposes = rules["all_purposes"]

    def determine_eligibility(row: pd.Series) -> List[str]:
        eligible: List[str] = []

        pcs = float(row.get("Personal Credit Score", 0) or 0)
        bcs = float(row.get("Business Credit Score", 0) or 0)
        dscr = float(row.get("DSCR (latest year)", 0) or 0)
        yib = float(row.get("Years in Business", 0) or 0)
        loan_amt = float(row.get("Loan Amount", 0) or 0)
        net_margin = float(row.get("Net Profit Margin", 0) or 0)
        for_profit = bool(row.get("For Profit", False))
        fast_approval = bool(row.get("Fast Approval", False))
        collateral = bool(row.get("Collateral Availability", False))

        valid_purpose = has_any(row, purposes)

        # 7(a)
        if (
            (not rules["7a_requires_for_profit"] or for_profit) and
            pcs >= rules["7a_min_personal_credit"] and
            bcs >= rules["7a_min_business_credit"] and
            dscr >= rules["7a_min_dscr"] and
            yib >= rules["7a_min_years_in_business"] and
            (rules["7a_loan_min"] <= loan_amt <= rules["7a_loan_max"]) and
            valid_purpose and
            has_none(row, rules["7a_exclude_purposes"])
        ):
            eligible.append("7(a)")

        # 8(a) (toggleable)
        if rules.get("enable_8a", False):
            if (
                (not rules["8a_requires_for_profit"] or (not for_profit)) and
                (not rules["8a_requires_fast_approval"] or (not fast_approval)) and
                yib >= rules["8a_min_years_in_business"] and
                float(row.get("Industry Experience", 0) or 0) >= rules["8a_min_industry_exp"] and
                float(row.get("Managerial Experience", 0) or 0) >= rules["8a_min_managerial_exp"] and
                valid_purpose and
                has_none(row, rules["8a_exclude_purposes"])
            ):
                eligible.append("8(a)")

        # 504 (no 'must include' and no avg NOI)
        if (
            (not rules["504_requires_for_profit"] or for_profit) and
            pcs >= rules["504_min_personal_credit"] and
            dscr >= rules["504_min_dscr"] and
            net_margin > rules["504_min_net_profit_margin"] and
            yib >= rules["504_min_years_in_business"] and
            loan_amt <= rules["504_max_loan"] and
            valid_purpose and
            (not rules["504_requires_collateral"] or collateral) and
            has_none(row, rules["504_exclude_purposes"])
        ):
            eligible.append("504")

        # Express
        if (
            (not rules["express_requires_for_profit"] or for_profit) and
            (not rules["express_requires_fast_approval"] or fast_approval) and
            pcs >= rules["express_min_personal_credit"] and
            bcs >= rules["express_min_business_credit"] and
            dscr >= rules["express_min_dscr"] and
            loan_amt <= rules["express_max_loan"] and
            valid_purpose and
            has_none(row, rules["express_exclude_purposes"])
        ):
            eligible.append("Express")

        return eligible if eligible else ["Ineligible"]

    df["Eligibility"] = df.apply(determine_eligibility, axis=1)
    return df


# =========================================================
# 🧾 Narrative (unchanged logic)
# =========================================================
def generate_applicant_summary(inputs: dict, loans: list) -> dict:
    def fget(key, default=0.0):
        v = inputs.get(key, default)
        try:
            if isinstance(v, bool):
                return v
            return float(v)
        except Exception:
            return default

    def bget(key):
        v = inputs.get(key, False)
        if isinstance(v, (bool, int, float)): 
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "y", "t")
        return False

    pcs = fget("Personal Credit Score"); bcs = fget("Business Credit Score")
    dscr = fget("DSCR (latest year)"); dscr_1y = fget("DSCR (1 year ago)"); dscr_2y = fget("DSCR (2 years ago)")
    yib = fget("Years in Business"); npm = fget("Net Profit Margin")
    ann_rev = fget("Annual Revenue (latest year)"); ann_rev_1 = fget("Annual Revenue (1 year ago)"); ann_rev_2 = fget("Annual Revenue (2 years ago)")
    debt = fget("Business Debt (latest year)"); debt_1 = fget("Business Debt (1 year ago)")
    noi_1 = fget("NOI (1 year ago)"); loan_amt = fget("Loan Amount")
    ind_exp = fget("Industry Experience"); mng_exp = fget("Managerial Experience")

    has_collateral = bget("Collateral Availability")
    fast_approval  = bget("Fast Approval")
    intent_re   = bget("Real Estate Acquisition or Improvement")
    intent_equip= bget("Equipment Purchase or Leasing")
    intent_wc   = bget("Working Capital")
    intent_acq  = bget("Business Acquisition or Buyout")
    intent_refi = bget("Refinancing Existing Debt")

    loan_to_income = inputs.get("Loan_to_Income_Ratio", (loan_amt / ann_rev) if ann_rev > 0 else 0.0)
    debt_to_income = inputs.get("Debt_to_Income_Ratio", (debt / ann_rev) if ann_rev > 0 else 0.0)
    rev_g1 = inputs.get("Revenue_Growth_1y", ((ann_rev - ann_rev_1) / ann_rev_1) if ann_rev_1 > 0 else 0.0)
    rev_g2 = inputs.get("Revenue_Growth_2y", ((ann_rev_1 - ann_rev_2) / ann_rev_2) if ann_rev_2 > 0 else 0.0)
    noi_chg_1y = inputs.get("NOI_Change_1y", ((fget("NOI (latest year)") - noi_1) / noi_1) if noi_1 > 0 else 0.0)
    avg_dscr = inputs.get("Avg_DSCR", np.nanmean([dscr, dscr_1y, dscr_2y]) if any([dscr, dscr_1y, dscr_2y]) else dscr)
    dscr_var = inputs.get("DSCR_Variability", np.nanstd([dscr, dscr_1y, dscr_2y]) if any([dscr, dscr_1y, dscr_2y]) else 0.0)
    exp_index = inputs.get("Experience_Index", (ind_exp + mng_exp) / 2.0)
    has_collateral_flag = bool(inputs.get("Has_Collateral", has_collateral))
    has_wc_flag = bool(inputs.get("Has_WorkingCapital", intent_wc))
    has_equip_flag = bool(inputs.get("Has_EquipmentNeed", intent_equip))
    has_re_flag = bool(inputs.get("Has_RealEstateIntent", intent_re))

    score = 0; strengths = []; risks = []

    if pcs >= 720: score += 9; strengths.append("Strong personal credit (≥ 720).")
    elif pcs >= 680: score += 5
    else: risks.append("Personal credit below 680 baseline.")

    if bcs >= 180: score += 7; strengths.append("Healthy business credit (≥ 180).")
    elif bcs >= 160: score += 4
    else: risks.append("Business credit below 160 baseline.")

    if dscr >= 1.50: score += 10; strengths.append("Robust DSCR (≥ 1.50).")
    elif dscr >= 1.25: score += 7
    elif dscr >= 1.15: score += 4
    else: score -= 10; risks.append("DSCR below 1.15 minimum.")

    if avg_dscr >= 1.35: score += 4; strengths.append("Multi-year DSCR average is healthy.")
    if dscr_var > 0.25: risks.append("High DSCR variability across years (volatility)."); score -= 3

    if npm > 10: score += 6; strengths.append("Solid profitability (NPM > 10%).")
    elif npm >= 1: score += 3
    else: risks.append("Non-profitable or negative margins."); score -= 5

    if noi_chg_1y > 0.05: strengths.append("NOI trending up year-over-year."); score += 3
    elif noi_chg_1y < -0.05: risks.append("NOI declined year-over-year."); score -= 2

    if rev_g1 > 0.10: strengths.append("Revenue grew > 10% last year."); score += 4
    elif rev_g1 < -0.10: risks.append("Revenue declined > 10% last year."); score -= 4

    if rev_g2 > 0.10: score += 2
    elif rev_g2 < -0.10: score -= 2

    if loan_to_income <= 0.30: strengths.append("Conservative leverage (Loan/Revenue ≤ 30%)."); score += 8
    elif loan_to_income <= 0.50: score += 5
    elif loan_to_income > 0.80: risks.append("High leverage (Loan/Revenue > 80%)."); score -= 8

    if debt_to_income <= 0.35: score += 4
    elif debt_to_income >= 0.80: risks.append("High debt load vs revenue (Debt/Revenue ≥ 80%)."); score -= 4

    if yib >= 5: strengths.append("Established operating history (≥ 5 years)."); score += 6
    elif yib >= 2: score += 3
    else: risks.append("Limited operating history (< 2 years)."); score -= 5

    if exp_index >= 8: strengths.append("Strong leadership/industry experience."); score += 4
    elif exp_index >= 3: score += 2

    if has_collateral_flag: strengths.append("Collateral available (improves 504/7(a) profile)."); score += 6
    else:
        if has_re_flag or has_equip_flag:
            risks.append("No collateral signaled for asset-heavy use (504 typically requires it).")

    if "Express" in loans and fast_approval: score += 2
    if "504" in loans and (has_re_flag or has_equip_flag): score += 3
    if "7(a)" in loans and (has_wc_flag or has_equip_flag or intent_acq or intent_refi): score += 2

    if "Express" in loans and loan_amt > 500_000:
        risks.append("Requested amount exceeds typical SBA Express cap (>$500K).")
    if "7(a)" in loans and (loan_amt < 500_001 or loan_amt > 5_000_000):
        risks.append("7(a) request is outside 500,001–5,000,000 window used in rules.")
    if "504" in loans and loan_amt > 5_500_000:
        risks.append("504 request exceeds common max threshold (~$5.5M).")

    if not loans or loans == ["Ineligible"]:
        score = max(0, score - 20)

    score = int(min(100, max(0, score)))
    if not loans or loans == ["Ineligible"]:
        label = "Ineligible"
    elif score >= 70:
        label = "Strong Fit"
    elif score >= 50:
        label = "Good Fit"
    elif score >= 35:
        label = "Borderline"
    else:
        label = "Needs Review"

    if loans and loans != ["Ineligible"]:
        details = []
        if "504" in loans: details.append("verify collateral and asset purpose docs for 504")
        if "7(a)" in loans: details.append("confirm DSCR calculations and working capital/equipment use for 7(a)")
        if "Express" in loans: details.append("ensure request ≤ $500K for Express")
        if not details: details.append("validate financials and purpose documentation")
        rec = "Proceed with underwriting; " + "; ".join(details) + "."
    else:
        rec = ("Not eligible under current inputs. Improve DSCR/credit, reduce request, "
               "or align purpose/collateral with target program.")

    loans_txt = ", ".join(loans) if loans else "None"
    narrative = (
        f"Overall **{label}** (score **{score}/100**). "
        f"Eligible programs: **{loans_txt}**. "
        f"Assessment considers credit quality, DSCR level/stability, profitability, growth momentum, "
        f"leverage and debt load, operating history, collateral, and purpose alignment."
    )

    return {
        "score": score,
        "label": label,
        "strengths": strengths,
        "risks": risks,
        "recommendation": rec,
        "narrative": narrative,
    }


# =========================================================
# 🧾 UI Helper — Summary of rules by loan (optional)
# =========================================================
def rules_summary_by_loan(rules: Dict) -> Dict[str, List[str]]:
    """
    Build a per-loan human-readable summary of the rules for UI display in Tab 5 & Tab 9.
    """
    r = _merge_rules(rules)
    by_loan = {}

    by_loan["7(a)"] = [
        f"For-Profit required: {r['7a_requires_for_profit']}",
        f"Min Personal Credit: {r['7a_min_personal_credit']}",
        f"Min Business Credit: {r['7a_min_business_credit']}",
        f"Min DSCR: {r['7a_min_dscr']}",
        f"Min Years in Business: {r['7a_min_years_in_business']}",
        f"Loan amount: {r['7a_loan_min']} – {r['7a_loan_max']}",
        f"Excluded purposes: {', '.join(r['7a_exclude_purposes']) or 'None'}",
    ]

    if r.get("enable_8a", False):
        by_loan["8(a)"] = [
            f"For-Profit required: {r['8a_requires_for_profit']} (False = Non-profit eligible)",
            f"Requires NOT Fast Approval: {r['8a_requires_fast_approval']}",
            f"Min Years in Business: {r['8a_min_years_in_business']}",
            f"Min Industry Exp: {r['8a_min_industry_exp']}",
            f"Min Managerial Exp: {r['8a_min_managerial_exp']}",
            f"Excluded purposes: {', '.join(r['8a_exclude_purposes']) or 'None'}",
        ]

    by_loan["504"] = [
        f"For-Profit required: {r['504_requires_for_profit']}",
        f"Collateral required: {r['504_requires_collateral']}",
        f"Min Personal Credit: {r['504_min_personal_credit']}",
        f"Min DSCR: {r['504_min_dscr']}",
        f"Min Net Profit Margin: {r['504_min_net_profit_margin']}",
        f"Min Years in Business: {r['504_min_years_in_business']}",
        f"Max Loan: {r['504_max_loan']}",
        f"Excluded purposes: {', '.join(r['504_exclude_purposes']) or 'None'}",
    ]

    by_loan["Express"] = [
        f"For-Profit required: {r['express_requires_for_profit']}",
        f"Requires Fast Approval: {r['express_requires_fast_approval']}",
        f"Min Personal Credit: {r['express_min_personal_credit']}",
        f"Min Business Credit: {r['express_min_business_credit']}",
        f"Min DSCR: {r['express_min_dscr']}",
        f"Max Loan: {r['express_max_loan']}",
        f"Excluded purposes: {', '.join(r['express_exclude_purposes']) or 'None'}",
    ]
    return by_loan
