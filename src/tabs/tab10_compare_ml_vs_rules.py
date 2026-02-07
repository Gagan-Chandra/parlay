# src/tabs/tab10_compare_ml_vs_rules.py
from __future__ import annotations
import ast
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any

from src.rules import (
    check_loan_eligibility_configurable,
    default_rules,
    generate_applicant_summary
)

# ---------- Utilities ----------
def _to_list(x):
    """Normalize to list[str]."""
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        s = x.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(i).strip() for i in v]
        except Exception:
            pass
        return [s]
    return ["Ineligible"]

def _compare_lists(rule_list: List[str], ml_list: List[str]) -> bool:
    return set(rule_list) == set(ml_list)

def _suggest_decision(rule_list: List[str], ml_list: List[str], row: pd.Series) -> str:
    """
    Basic decision assist:
    - If match -> Accept both.
    - If ML adds loans -> If high prob (>= 0.65) suggest ML, else Review.
    - If ML removes loans -> Favor Rules (policy alignment).
    """
    rule_set = set(rule_list)
    ml_set = set(ml_list)

    if rule_set == ml_set:
        if rule_set == {"Ineligible"}:
            return "Both: Ineligible. Suggest: Ineligible."
        return f"Both agree on: {', '.join(rule_list)}. Suggest: Accept."

    # read proba columns if available
    probas = {}
    for lab in ["7(a)", "504", "Express"]:
        col = f"proba_{lab}"
        if col in row and pd.notna(row[col]):
            try:
                probas[lab] = float(row[col])
            except Exception:
                pass

    # If rules have something ML doesn't include
    if rule_set - ml_set:
        missing = ", ".join(sorted(rule_set - ml_set))
        return f"ML missing: {missing} (in Rules). Suggest: **Rules**."

    # If ML has extras not in Rules
    if ml_set - rule_set:
        extra = list(ml_set - rule_set)
        high_conf = [l for l in extra if probas.get(l, 0.0) >= 0.65]
        if high_conf:
            return f"ML adds {', '.join(extra)} with high probability. Suggest: **ML**."
        else:
            return f"ML adds {', '.join(extra)} but confidence unclear. Suggest: **Review**."

    return "Disagreement detected. Suggest: Review."

# ---------- Tab Renderer ----------
def render():
    st.header("🔍 ML vs Rules — Side-by-Side Comparison")

    st.markdown("""
This tab compares:
- **ML Eligibility**: from Tab 4 (`Eligibility_Pred`).
- **Rules Eligibility**: exactly as computed in **Tab 5** (not recomputed here).

Below we show:
- A full comparison table with these two columns.
- A list of **differences only** with **Fit Score** and **suggested decision**.
""")

    # 1) Load ML results from Tab 4
    ml_df = st.session_state.get("pred_results_df")
    if ml_df is None or ml_df.empty:
        st.warning("No ML results found. Please run **Tab 4: Live Predictions** first.")
        return

    if "Applicant ID" not in ml_df.columns:
        st.error("`Applicant ID` is required in ML results.")
        return
    if "Eligibility_Pred" not in ml_df.columns:
        st.error("`Eligibility_Pred` not found in ML results from Tab 4.")
        return

    # Full input context saved by Tab 4
    full_input = st.session_state.get("t4_input_df")
    if full_input is None or full_input.empty:
        st.warning("Full applicant input not found from Tab 4; fit scores may be degraded.")
        full_input = pd.DataFrame()

    # 2) Load Rule results from Tab 5
    rules_out = st.session_state.get("tab5_rules_output_df")
    if rules_out is None or rules_out.empty:
        # fallback: compute using current rules if Tab 5 hasn't been run
        st.info("Tab 5 rules output not available. Computing rule-based eligibility using current rules...")
        try:
            active_rules = st.session_state.get("eligibility_rules", default_rules())
            temp = check_loan_eligibility_configurable(ml_df.copy(), active_rules)
            rules_out = temp[["Applicant ID", "Eligibility"]].copy()
            st.caption("Computed rules because Tab 5 output was not available.")
        except Exception as e:
            st.error(f"Cannot compute rules-based outputs: {e}")
            return

    # 3) Normalize lists
    ml_use = ml_df.copy()
    ml_use["Eligibility_Pred"] = ml_use["Eligibility_Pred"].apply(_to_list)
    rules_use = rules_out.copy()
    rules_use["Eligibility"] = rules_use["Eligibility"].apply(_to_list)

    # 4) Display columns
    id_cols = ["Applicant ID"]
    context_cols = [c for c in ["Business Name"] if c in ml_use.columns]
    proba_cols = [c for c in ml_use.columns if c.startswith("proba_")]

    ml_display = ml_use[id_cols + context_cols + ["Eligibility_Pred"] + proba_cols].copy()
    rules_display = rules_use[id_cols + ["Eligibility"]].copy()

    # 5) Merge
    merged = ml_display.merge(rules_display, on="Applicant ID", how="left")

    # 6) Comparison flags & suggestion
    merged["Equal?"] = merged.apply(lambda r: _compare_lists(r["Eligibility"], r["Eligibility_Pred"]), axis=1)
    merged["Suggested Decision"] = merged.apply(
        lambda r: _suggest_decision(r["Eligibility"], r["Eligibility_Pred"], r), axis=1
    )

    # 7) Full table
    st.subheader("📊 Full Comparison (ML vs Rules)")
    full_cols = id_cols + context_cols + ["Eligibility_Pred", "Eligibility", "Equal?", "Suggested Decision"]
    st.dataframe(merged[full_cols], use_container_width=True)

    # 8) Differences only
    st.subheader("❗ Differences Only")
    diffs = merged[~merged["Equal?"]].copy()
    if diffs.empty:
        st.success("No differences: ML and Rule-based predictions match for all applicants.")
        return
    st.dataframe(diffs[full_cols], use_container_width=True)

    # 9) Fit scores and notes for differences
    st.markdown("### 📋 Fit Score & Notes for Differences")
    for _, row in diffs.iterrows():
        loans = row["Eligibility_Pred"] if isinstance(row["Eligibility_Pred"], list) else _to_list(row["Eligibility_Pred"])

        # Pull the full raw row by Applicant ID for narrative
        app_id = int(row.get("Applicant ID", -1))
        if not full_input.empty and "Applicant ID" in full_input.columns:
            try:
                raw_row = full_input.loc[full_input["Applicant ID"] == app_id].iloc[0].to_dict()
            except Exception:
                raw_row = {}
        else:
            raw_row = {}

        # combine raw fields + ML/rules row (ML probas, etc.)
        combined = {**raw_row, **row.to_dict()}

        try:
            summary = generate_applicant_summary(combined, loans)
            st.markdown(
                f"**Applicant {app_id}** — "
                f"**Fit:** {summary['label']} (Score {summary['score']}/100) "
                f"| **ML:** {', '.join(loans)} | **Rules:** {', '.join(row['Eligibility'])}"
            )
            st.caption(f"Recommendation: {summary['recommendation']} — {row['Suggested Decision']}")
        except Exception:
            st.markdown(
                f"**Applicant {app_id}** — ML: {', '.join(loans)} | Rules: {', '.join(row['Eligibility'])} "
                f"| Suggestion: {row['Suggested Decision']}"
            )
