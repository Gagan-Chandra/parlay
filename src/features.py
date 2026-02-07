# loan_app/src/features.py
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Eligibility" in df and df["Eligibility"].dtype == object:
        def _norm(x):
            if isinstance(x, list): return x
            s = str(x)
            if s.startswith("["):
                try: return ast.literal_eval(s)
                except Exception: return [s]
            return [s]
        df["Eligibility"] = df["Eligibility"].apply(_norm)
    return df

def ensure_list_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_eligibility(df)

def filter_by_loan(df: pd.DataFrame, loan: str) -> pd.DataFrame:
    if loan == "All": return df
    return df[df["Eligibility"].apply(lambda xs: loan in xs)]

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def sd(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return np.where(b == 0, 0, a/b)

    # Ratios
    if "Business Debt (latest year)" in df and "Annual Revenue (latest year)" in df:
        df["Debt_to_Income_Ratio"] = sd(df["Business Debt (latest year)"], df["Annual Revenue (latest year)"])
    if "Loan Amount" in df and "Annual Revenue (latest year)" in df:
        df["Loan_to_Income_Ratio"] = sd(df["Loan Amount"], df["Annual Revenue (latest year)"])
    if "NOI (latest year)" in df and "Annual Revenue (latest year)" in df:
        df["NOI_to_Revenue_Ratio"] = sd(df["NOI (latest year)"], df["Annual Revenue (latest year)"])

    # Trends
    if all(c in df for c in ["Annual Revenue (latest year)", "Annual Revenue (1 year ago)"]):
        df["Revenue_Growth_1y"] = sd(
            df["Annual Revenue (latest year)"] - df["Annual Revenue (1 year ago)"],
            df["Annual Revenue (1 year ago)"]
        )
    if all(c in df for c in ["Annual Revenue (1 year ago)", "Annual Revenue (2 years ago)"]):
        df["Revenue_Growth_2y"] = sd(
            df["Annual Revenue (1 year ago)"] - df["Annual Revenue (2 years ago)"],
            df["Annual Revenue (2 years ago)"]
        )

    # DSCR aggregates
    dscr_cols = [c for c in ["DSCR (latest year)", "DSCR (1 year ago)", "DSCR (2 years ago)"] if c in df]
    if dscr_cols:
        df["Avg_DSCR"] = df[dscr_cols].mean(axis=1)

    # Experience
    if all(c in df for c in ["Industry Experience", "Managerial Experience"]):
        df["Experience_Index"] = (df["Industry Experience"] + df["Managerial Experience"]) / 2

    # Purpose flags
    for col in ["Collateral Availability", "Working Capital", "Business Expansion",
                "Equipment Purchase or Leasing", "Real Estate Acquisition or Improvement"]:
        if col not in df:
            df[col] = 0
    df["Has_Collateral"] = df["Collateral Availability"].astype(int, errors="ignore")
    df["Has_WorkingCapital"] = df["Working Capital"].astype(int, errors="ignore")
    df["Has_ExpansionIntent"] = df["Business Expansion"].astype(int, errors="ignore")
    df["Has_EquipmentNeed"] = df["Equipment Purchase or Leasing"].astype(int, errors="ignore")
    df["Has_RealEstateIntent"] = df["Real Estate Acquisition or Improvement"].astype(int, errors="ignore")

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

def numeric_feature_list(df: pd.DataFrame):
    candidates = [
        "Personal Credit Score","Business Credit Score","DSCR (latest year)",
        "Annual Revenue (latest year)","Years in Business","Loan Amount",
        "Industry Experience","Managerial Experience","Avg_DSCR",
        "Loan_to_Income_Ratio","Debt_to_Income_Ratio","Revenue_Growth_1y","Revenue_Growth_2y"
    ]
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def stratified_train_test_from(df: pd.DataFrame, test_size=0.2, seed=42):
    df = normalize_eligibility(df)
    df = df.copy()
    label = df["Eligibility"].apply(lambda xs: xs[0] if isinstance(xs, list) and xs else "Ineligible")
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=label)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
