# loan_app/src/tabs/tab2_feature_eng.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from typing import Dict, List

# New model + features pipeline
from src.model_loader import load_model_bundle
from src.ml_features import engineer, add_cross_features


# ----------------------------
# Metadata for documentation
# ----------------------------
BASE_ENGINEERED_FEATURES = [
    # Rule-critical inputs (should be present in uploaded file)
    "Personal Credit Score", "Business Credit Score", "DSCR (latest year)",
    "Years in Business", "Loan Amount", "For Profit", "Fast Approval",
    "Collateral Availability",

    # Derived ratios/trends (engineer)
    "Revenue_Growth", "Debt_Growth", "NOI_Growth",
    "Debt_to_Revenue", "Loan_to_Revenue",
    "Profitability", "Experience_Score", "Maturity",

    # Purpose mix indicators
    "Purpose_Count", "Short_Term_Focus", "Capital_Intensive",

    # Stabilizers / raw financials used by the pipeline
    "Annual Revenue (latest year)", "Business Debt (latest year)", "NOI (latest year)",
]

CROSS_FEATURES = [
    # Cross interactions from add_cross_features()
    "Fast_LowLoan",              # (Fast Approval == 1) & (Loan Amount <= 500k)
    "RE_Blocks_7a",              # Real Estate use blocks 7(a)
    "Buyout_Blocks_Express",     # Business Acquisition blocks Express
    "CapitalIntensive_Collateral",  # Capital-intensive purpose + Collateral
    "ShortTerm_Blocks_504",      # Working capital/refi/emergency blocks 504
    "HighCredit_HighDSCR",       # Good performance combo
    "DSCR_margin",               # DSCR - 1.15 threshold margin
    "Loan_over_500k", "Loan_over_5m"
]

PURPOSE_FLAGS = [
    "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
    "Inventory Purchase", "Real Estate Acquisition or Improvement",
    "Business Acquisition or Buyout", "Refinancing Existing Debt",
    "Emergency Funds", "Franchise Financing", "Contract Financing",
    "Licensing or Permits", "Line of Credit Establishment",
]


def _avg_importances(models: List, feat_names: List[str]) -> pd.Series:
    """
    Average LightGBM feature importances (gain-based) across a list of CV models.
    Returns a Series indexed by feature name.
    """
    if not models:
        return pd.Series(dtype=float)

    # LightGBM sklearn wrapper stores splitter as .booster_.feature_importance()
    imps = [m.booster_.feature_importance(importance_type="gain") for m in models]
    arr = np.vstack(imps).astype(float)
    avg = arr.mean(axis=0)
    s = pd.Series(avg, index=feat_names)
    # Normalize to sum to 1 for readability
    if s.sum() > 0:
        s = s / s.sum()
    return s.sort_values(ascending=False)


def _global_importance(bundle: Dict, mode: str = "aggregate") -> pd.DataFrame:
    """
    Build a dataframe of feature importance either:
      - per loan (7(a),504,Express) or
      - aggregated across these loans.
    """
    models_cv = bundle["models_cv"]
    feat_cols = bundle["feature_cols"]
    labels = [lab for lab in bundle["labels"] if lab in ("7(a)", "504", "Express")]

    if mode == "aggregate":
        # mean importance over all labels
        per_label = [_avg_importances(models_cv[lab], feat_cols) for lab in labels]
        if not per_label:
            return pd.DataFrame()
        agg = pd.concat(per_label, axis=1).fillna(0.0)
        agg["mean_importance"] = agg.mean(axis=1)
        out = agg["mean_importance"].sort_values(ascending=False).to_frame("importance")
        return out
    else:
        # return multi-column df: each column corresponds to one loan label
        frames = {}
        for lab in labels:
            frames[lab] = _avg_importances(models_cv.get(lab, []), feat_cols)
        df = pd.DataFrame(frames).fillna(0.0).sort_values(by=labels, ascending=False)
        return df


def render():
    st.header("🧩 Feature Engineering & Insights (New Pipeline)")

    st.markdown(
        """
This tab reflects the **new feature engineering pipeline** used by your **multi-label LightGBM CV model**.
It combines:
- **Rule-critical raw fields** (credit scores, DSCR, loan amount, tenure, flags),
- **Engineered numerical features** (growth, leverage, profitability, experience),
- **Purpose mix features** (short-term vs capital-intensive),
- **Cross features** designed to reflect **policy logic** (e.g., real estate blocks 7(a)).

Use this tab to:  
1. Understand which features the model considers most important.  
2. Upload a dataset to preview engineered features.  
"""
    )

    # Load current (active) bundle
    try:
        bundle = load_model_bundle()
    except Exception as e:
        st.error(f"Could not load model bundle: {e}")
        return

    st.subheader("📦 Engineered Feature Families")
    with st.expander("🔍 See engineered & cross features", expanded=False):
        st.markdown("**Base engineered features (from `engineer()`):**")
        st.write(", ".join(BASE_ENGINEERED_FEATURES))
        st.markdown("---")
        st.markdown("**Cross features (from `add_cross_features()`):**")
        st.write(", ".join(CROSS_FEATURES))
        st.markdown("---")
        st.markdown("**Purpose flags:**")
        st.write(", ".join(PURPOSE_FLAGS))

    # ----------------------------
    # Global feature importance (from bundle)
    # ----------------------------
    st.subheader("🏅 Global Feature Importance (CV LightGBM)")
    mode = st.radio(
        "Show importances:",
        ["Aggregate across loans", "Per loan (separate columns)"],
        horizontal=True,
        index=0,
        key="t2_imp_mode"
    )

    if mode.startswith("Aggregate"):
        imp_df = _global_importance(bundle, mode="aggregate")
        if imp_df.empty:
            st.warning("No importances available.")
        else:
            s = imp_df["importance"].head(20)[::-1]  # top 20
            fig = px.bar(
                s,
                x=s.values,
                y=s.index,
                orientation="h",
                color=s.values,
                color_continuous_scale="Bluered_r",
                labels={"x": "Importance (normalized)", "y": "Feature"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        imp_df = _global_importance(bundle, mode="per_loan")
        if imp_df.empty:
            st.warning("No importances available.")
        else:
            st.caption("Higher = more gain-based contribution; columns denote each loan label.")
            # show top by sum across labels
            top = (imp_df.sum(axis=1).sort_values(ascending=False)).head(25).index
            st.dataframe(imp_df.loc[top], use_container_width=True)

    # ----------------------------
    # Upload to preview engineered features
    # ----------------------------
    st.markdown("---")
    st.subheader("📤 Preview Engineered Features (Your Data)")
    up = st.file_uploader("Upload dataset (xlsx/csv) — optional", type=["xlsx", "csv"], key="t2_upload")
    if up:
        try:
            raw = pd.read_excel(up) if up.name.lower().endswith(".xlsx") else pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        st.success(f"Loaded {raw.shape[0]} rows.")
        try:
            X = engineer(raw).copy()
            X = add_cross_features(X).fillna(0)
            st.markdown("**Engineered preview (first 10 rows):**")
            st.dataframe(X.head(10), use_container_width=True)

            # Show which engineered features are new (that were not in raw)
            new_cols = [c for c in X.columns if c not in raw.columns]
            st.markdown(f"**Newly-created columns:** {', '.join(new_cols[:30])}" + (" ..." if len(new_cols) > 30 else ""))

        except Exception as e:
            st.error(f"Could not engineer features for the uploaded dataset: {e}")
