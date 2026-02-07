# src/ml_features.py
import numpy as np
import pandas as pd

PURPOSE_COLS = [
    "Working Capital","Business Expansion","Equipment Purchase or Leasing",
    "Inventory Purchase","Real Estate Acquisition or Improvement",
    "Business Acquisition or Buyout","Refinancing Existing Debt",
    "Emergency Funds","Franchise Financing","Contract Financing",
    "Licensing or Permits","Line of Credit Establishment"
]

def engineer(d0: pd.DataFrame) -> pd.DataFrame:
    d = d0.copy()
    d["Annual Revenue (1 year ago)"] = d.get("Annual Revenue (1 year ago)", 0)
    d["Business Debt (1 year ago)"] = d.get("Business Debt (1 year ago)", 0)
    d["NOI (1 year ago)"] = d.get("NOI (1 year ago)", 0)

    d["Revenue_Growth"]   = (d["Annual Revenue (latest year)"] - d["Annual Revenue (1 year ago)"]) / (d["Annual Revenue (1 year ago)"] + 1e-9)
    d["Debt_Growth"]      = (d["Business Debt (latest year)"] - d["Business Debt (1 year ago)"]) / (d["Business Debt (1 year ago)"] + 1e-9)
    d["NOI_Growth"]       = (d["NOI (latest year)"] - d["NOI (1 year ago)"]) / (d["NOI (1 year ago)"] + 1e-9)
    d["Debt_to_Revenue"]  = d["Business Debt (latest year)"] / (d["Annual Revenue (latest year)"] + 1e-9)
    d["Loan_to_Revenue"]  = d["Loan Amount"] / (d["Annual Revenue (latest year)"] + 1e-9)
    d["Profitability"]    = d.get("Net Profit Margin", 0) * d.get("DSCR (latest year)", 0)
    d["Experience_Score"] = 0.6*d.get("Industry Experience", 0) + 0.4*d.get("Managerial Experience", 0)
    d["Maturity"]         = np.log1p(d.get("Years in Business", 0)) * d.get("DSCR (latest year)", 0)
    pres = [c for c in PURPOSE_COLS if c in d.columns]
    d["Purpose_Count"] = d[pres].sum(axis=1).astype(float) if pres else 0.0
    d["Short_Term_Focus"] = d[[c for c in ["Working Capital","Emergency Funds","Line of Credit Establishment"] if c in d.columns]].sum(axis=1).astype(float)
    d["Capital_Intensive"] = d[[c for c in ["Real Estate Acquisition or Improvement","Equipment Purchase or Leasing","Business Acquisition or Buyout"] if c in d.columns]].sum(axis=1).astype(float)

    keep = [
        "Personal Credit Score","Business Credit Score","DSCR (latest year)","Years in Business","Loan Amount",
        "Collateral Availability","Fast Approval","For Profit"
    ] + PURPOSE_COLS + [
        "Annual Revenue (latest year)","Business Debt (latest year)","NOI (latest year)",
        "Revenue_Growth","Debt_Growth","NOI_Growth","Debt_to_Revenue","Loan_to_Revenue",
        "Profitability","Experience_Score","Maturity","Purpose_Count","Short_Term_Focus","Capital_Intensive"
    ]
    use = [c for c in keep if c in d.columns]
    d = d[use].copy()
    for c in d.columns:
        if d[c].dtype == bool: d[c] = d[c].astype(float)
    return d.fillna(0)

def add_cross_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc["Fast_LowLoan"] = ((Xc.get("Fast Approval",0)==1) & (Xc["Loan Amount"]<=500000)).astype(float)
    Xc["RE_Blocks_7a"] = (Xc.get("Real Estate Acquisition or Improvement",0)==1).astype(float)
    Xc["Buyout_Blocks_Express"] = (Xc.get("Business Acquisition or Buyout",0)==1).astype(float)
    Xc["CapitalIntensive_Collateral"] = (
        (Xc.get("Collateral Availability",0)==1) &
        ((Xc.get("Real Estate Acquisition or Improvement",0)==1) |
         (Xc.get("Equipment Purchase or Leasing",0)==1) |
         (Xc.get("Business Acquisition or Buyout",0)==1))
    ).astype(float)
    Xc["ShortTerm_Blocks_504"] = (
        (Xc.get("Working Capital",0)==1) |
        (Xc.get("Refinancing Existing Debt",0)==1) |
        (Xc.get("Emergency Funds",0)==1)
    ).astype(float)
    Xc["HighCredit_HighDSCR"] = (
        (Xc["Personal Credit Score"]>=700) & (Xc["DSCR (latest year)"]>=1.25)
    ).astype(float)
    Xc["DSCR_margin"] = Xc["DSCR (latest year)"] - 1.15
    Xc["Loan_over_500k"] = (Xc["Loan Amount"]>500000).astype(float)
    Xc["Loan_over_5m"]   = (Xc["Loan Amount"]>5000000).astype(float)
    return Xc
