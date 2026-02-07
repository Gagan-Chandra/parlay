# loan_app/src/tabs/tab8_ai_assistant.py
from __future__ import annotations
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from typing import List, Dict, Any

from src.rules import (
    default_rules,
    check_loan_eligibility_configurable,  # Mirror Tab 5 rules (configurable)
    check_loan_eligibility,               # Baseline (fixed logic — your main function)
    generate_applicant_summary
)

# ----------------------------------------------
# Rerun helper (compatible across Streamlit versions)
# ----------------------------------------------
def _safe_rerun():
    """Use st.rerun() if available; fall back to st.experimental_rerun()."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ----------------------------------------------
# Constants & helpers
# ----------------------------------------------
BOOLEAN_FIELDS = [
    "For Profit", "Fast Approval", "Collateral Availability",
    "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
    "Inventory Purchase", "Real Estate Acquisition or Improvement",
    "Business Acquisition or Buyout", "Refinancing Existing Debt",
    "Emergency Funds", "Franchise Financing", "Contract Financing",
    "Licensing or Permits", "Line of Credit Establishment",
]

NUMERIC_FIELDS = [
    "Personal Credit Score", "Business Credit Score",
    "DSCR (latest year)", "Annual Revenue (latest year)", "Loan Amount",
    "Years in Business", "Net Profit Margin",
    "NOI (1 year ago)", "NOI (2 years ago)",
    "Industry Experience", "Managerial Experience",
]

CONTEXT_FIELDS = ["Applicant ID", "Business Name", "location", "Location"]

NUMERIC_DISPLAY = [
    "Personal Credit Score", "Business Credit Score", "DSCR (latest year)",
    "Years in Business", "Loan Amount", "Net Profit Margin",
    "Annual Revenue (latest year)"
]

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])

def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all required columns exist and types match what Tab 5 expects.
    """
    out = df.copy()
    for c in BOOLEAN_FIELDS:
        if c not in out.columns:
            out[c] = False
        out[c] = _to_bool_series(out[c])

    for c in NUMERIC_FIELDS:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    if "Applicant ID" not in out.columns:
        out["Applicant ID"] = np.arange(1, len(out) + 1)

    return out

def _compare_to_median(app_row: pd.Series, df: pd.DataFrame) -> str:
    """
    Compare applicant values to dataset medians for top numeric fields.
    """
    numeric = df.select_dtypes(include=[np.number])
    med = numeric.median(numeric_only=True)
    lines = []
    for c in NUMERIC_DISPLAY:
        if c in app_row and c in med.index:
            try:
                delta = float(app_row[c]) - float(med[c])
                lines.append(f"- {c}: {app_row[c]:,.2f} vs median {med[c]:,.2f} (Δ {delta:+,.2f})")
            except Exception:
                pass
    return "\n".join(lines) if lines else "No numeric fields available to compare."

def _recommend_loan(loans: List[str], row: Dict[str, Any]) -> str:
    """
    Simple heuristic to recommend among eligible loans.
    """
    if not loans or loans == ["Ineligible"]:
        return "No loans are eligible under the current inputs."

    fast = bool(row.get("Fast Approval", False))
    collateral = bool(row.get("Collateral Availability", False))
    npm = float(row.get("Net Profit Margin", 0) or 0)

    # Some simple logic
    if "Express" in loans and fast:
        return "Recommendation: **SBA Express** — aligns with fast approval & eligibility."
    if "504" in loans and collateral and npm > 0:
        return "Recommendation: **SBA 504** — asset-oriented and positive net margin with collateral."
    if "7(a)" in loans:
        return "Recommendation: **SBA 7(a)** — general-purpose flexibility and multi-purpose coverage."
    return f"Eligible: {', '.join(loans)} — consider DSCR/credit alignment against loan type."

def _why_not_loan_mirror(app_row: pd.Series, rules: Dict, loan: str) -> List[str]:
    """
    Explain 'why not' for Mirror Tab 5 mode using keys from config rules.
    """
    def v(field, default=0.0):
        try: return float(app_row.get(field, default) or default)
        except: return default

    pcs = v("Personal Credit Score")
    bcs = v("Business Credit Score")
    dscr = v("DSCR (latest year)")
    yib = v("Years in Business")
    loan_amt = v("Loan Amount")
    npm = v("Net Profit Margin")

    for_profit = bool(app_row.get("For Profit", False))
    fast = bool(app_row.get("Fast Approval", False))
    collateral = bool(app_row.get("Collateral Availability", False))

    purposes = rules.get("all_purposes", [])
    valid_purpose = any(bool(app_row.get(p, False)) for p in purposes)

    fail = []
    if loan == "7(a)":
        if rules.get("7a_requires_for_profit", True) and not for_profit:
            fail.append("7(a) requires for-profit.")
        if pcs < rules.get("7a_min_personal_credit", rules.get('min_personal_credit', 680)):
            fail.append(f"Personal Credit < {rules.get('7a_min_personal_credit', rules.get('min_personal_credit', 680))}.")
        if bcs < rules.get("7a_min_business_credit", rules.get('min_business_credit', 160)):
            fail.append(f"Business Credit < {rules.get('7a_min_business_credit', rules.get('min_business_credit', 160))}.")
        if dscr < rules.get("7a_min_dscr", rules.get('min_dscr', 1.15)):
            fail.append(f"DSCR < {rules.get('7a_min_dscr', rules.get('min_dscr', 1.15))}.")
        if yib < rules.get("7a_min_years_in_business", rules.get('min_years_in_business', 2)):
            fail.append(f"Years in Business < {rules.get('7a_min_years_in_business', rules.get('min_years_in_business', 2))}.")
        if not (rules.get("7a_loan_min", 500001) <= loan_amt <= rules.get("7a_loan_max", 5000000)):
            fail.append("Loan outside 7(a) min/max.")
        if not valid_purpose:
            fail.append("No valid purpose selected.")
        if any(bool(app_row.get(p, False)) for p in rules.get("7a_exclude_purposes", [])):
            fail.append("Contains excluded purpose for 7(a).")

    elif loan == "8(a)":
        if not rules.get("enable_8a", False):
            fail.append("8(a) program disabled.")
        if rules.get("8a_requires_for_profit", False) and for_profit:
            fail.append("8(a) requires NOT for-profit.")
        if rules.get("8a_requires_fast_approval", False) and fast:
            fail.append("8(a) requires NOT fast approval.")
        if yib < rules.get("8a_min_years_in_business", rules.get('min_years_in_business', 2)):
            fail.append(f"Years in Business < {rules.get('8a_min_years_in_business', rules.get('min_years_in_business', 2))}.")
        if v("Industry Experience") < rules.get("8a_min_industry_exp", 2):
            fail.append("Industry experience too low for 8(a).")
        if v("Managerial Experience") < rules.get("8a_min_managerial_exp", 2):
            fail.append("Managerial experience too low for 8(a).")
        if not valid_purpose:
            fail.append("No valid purpose selected.")
        if any(bool(app_row.get(p, False)) for p in rules.get("8a_exclude_purposes", [])):
            fail.append("Contains excluded purpose (8a).")

    elif loan == "504":
        if rules.get("504_requires_for_profit", True) and not for_profit:
            fail.append("504 requires for-profit.")
        if pcs < rules.get("504_min_personal_credit", rules.get('min_personal_credit', 680)):
            fail.append(f"Personal Credit < {rules.get('504_min_personal_credit', rules.get('min_personal_credit', 680))}.")
        if dscr < rules.get("504_min_dscr", rules.get('min_dscr', 1.15)):
            fail.append(f"DSCR < {rules.get('504_min_dscr', rules.get('min_dscr', 1.15))}.")
        if npm <= rules.get("504_min_net_profit_margin", 0.0):
            fail.append(f"Net Profit Margin ≤ {rules.get('504_min_net_profit_margin', 0.0)}.")
        if yib < rules.get("504_min_years_in_business", rules.get('min_years_in_business', 2)):
            fail.append(f"Years in Business < {rules.get('504_min_years_in_business', rules.get('min_years_in_business', 2))}.")
        if loan_amt > rules.get("504_max_loan", 5_500_000):
            fail.append("Loan exceeds 504 maximum.")
        if rules.get("504_requires_collateral", True) and not collateral:
            fail.append("Collateral required for 504.")
        if not valid_purpose:
            fail.append("No valid purpose selected.")
        if any(bool(app_row.get(p, False)) for p in rules.get("504_exclude_purposes", [])):
            fail.append("Contains excluded purpose (504).")

    elif loan == "Express":
        if rules.get("express_requires_for_profit", True) and not for_profit:
            fail.append("Express requires for-profit.")
        if rules.get("express_requires_fast_approval", True) and not fast:
            fail.append("Express requires 'Fast Approval'.")
        if pcs < rules.get("express_min_personal_credit", rules.get('min_personal_credit', 680)):
            fail.append(f"Personal Credit < {rules.get('express_min_personal_credit', rules.get('min_personal_credit', 680))}.")
        if bcs < rules.get("express_min_business_credit", rules.get('min_business_credit', 160)):
            fail.append(f"Business Credit < {rules.get('express_min_business_credit', rules.get('min_business_credit', 160))}.")
        if dscr < rules.get("express_min_dscr", rules.get('min_dscr', 1.15)):
            fail.append(f"DSCR < {rules.get('express_min_dscr', rules.get('min_dscr', 1.15))}.")
        if loan_amt > rules.get("express_max_loan", 500_000):
            fail.append("Loan exceeds Express maximum.")
        if not valid_purpose:
            fail.append("No valid purpose selected.")
        if any(bool(app_row.get(p, False)) for p in rules.get("express_exclude_purposes", [])):
            fail.append("Contains excluded purpose (Express).")
    return fail

def _why_not_loan_fixed(app_row: pd.Series, loan: str) -> List[str]:
    """
    Explain 'why not' for baseline local fixed logic (check_loan_eligibility).
    """
    def v(field, default=0.0):
        try: return float(app_row.get(field, default) or default)
        except: return default

    pcs = v("Personal Credit Score")
    bcs = v("Business Credit Score")
    dscr = v("DSCR (latest year)")
    yib = v("Years in Business")
    loan_amt = v("Loan Amount")
    npm = v("Net Profit Margin")
    for_profit = bool(app_row.get("For Profit", False))
    fast = bool(app_row.get("Fast Approval", False))
    collateral = bool(app_row.get("Collateral Availability", False))

    purposes = [
        "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
        "Inventory Purchase", "Real Estate Acquisition or Improvement",
        "Business Acquisition or Buyout", "Refinancing Existing Debt",
        "Emergency Funds", "Franchise Financing", "Contract Financing",
        "Licensing or Permits", "Line of Credit Establishment",
    ]
    valid_purpose = any(bool(app_row.get(p, False)) for p in purposes)

    fail = []
    if loan == "7(a)":
        if not for_profit: fail.append("7(a) requires for-profit.")
        if pcs < 680: fail.append("Personal Credit < 680.")
        if bcs < 160: fail.append("Business Credit < 160.")
        if dscr < 1.15: fail.append("DSCR < 1.15.")
        if yib < 2: fail.append("Years in Business < 2.")
        if not (500001 <= loan_amt <= 5_000_000):
            fail.append("Loan outside 500,001–5,000,000.")
        if not valid_purpose: fail.append("No valid purpose.")
        if bool(app_row.get("Real Estate Acquisition or Improvement", False)):
            fail.append("7(a) excludes Real Estate Acquisition or Improvement.")
        if bool(app_row.get("Emergency Funds", False)):
            fail.append("7(a) excludes Emergency Funds.")

    elif loan == "8(a)":
        if for_profit: fail.append("8(a) requires Not For-Profit.")
        if fast: fail.append("8(a) requires NOT fast approval.")
        if yib < 2: fail.append("Years in Business < 2.")
        if v("Industry Experience") < 2: fail.append("Industry Experience < 2.")
        if v("Managerial Experience") < 2: fail.append("Managerial Experience < 2.")
        if not valid_purpose: fail.append("No valid purpose.")
        if bool(app_row.get("Franchise Financing", False)):
            fail.append("8(a) excludes Franchise Financing.")
        if bool(app_row.get("Line of Credit Establishment", False)):
            fail.append("8(a) excludes Line of Credit Establishment.")

    elif loan == "504":
        if not for_profit: fail.append("504 requires for-profit.")
        if pcs < 680: fail.append("Personal Credit < 680.")
        if dscr < 1.15: fail.append("DSCR < 1.15.")
        if npm <= 0: fail.append("Net Profit Margin ≤ 0.")
        if yib < 2: fail.append("Years in Business < 2.")
        if loan_amt > 5_500_000: fail.append("Loan exceeds 5,500,000.")
        if not collateral: fail.append("Collateral required.")
        if bool(app_row.get("Working Capital", False)):
            fail.append("504 excludes Working Capital.")
        if bool(app_row.get("Refinancing Existing Debt", False)):
            fail.append("504 excludes Refinancing Existing Debt.")
        if bool(app_row.get("Emergency Funds", False)):
            fail.append("504 excludes Emergency Funds.")

    elif loan == "Express":
        if not for_profit: fail.append("Express requires for-profit.")
        if not fast: fail.append("Express requires Fast Approval.")
        if pcs < 680: fail.append("Personal Credit < 680.")
        if bcs < 160: fail.append("Business Credit < 160.")
        if dscr < 1.15: fail.append("DSCR < 1.15.")
        if loan_amt > 500_000: fail.append("Loan exceeds 500,000.")
        if not valid_purpose: fail.append("No valid purpose.")
        if bool(app_row.get("Real Estate Acquisition or Improvement", False)):
            fail.append("Express excludes Real Estate Acquisition or Improvement.")
        if bool(app_row.get("Business Acquisition or Buyout", False)):
            fail.append("Express excludes Business Acquisition or Buyout.")
    return fail

# ----------------------------------------------
# Main Tab
# ----------------------------------------------
def render():
    st.header("🤖 AI Assistant — Rule-based Eligibility + Q&A (No LLM)")

    st.markdown("""
Upload applicants, review **rule-based eligibility** for each row,  
then ask natural questions like:
- *"Why not 504?"*, *"Top strengths?"*, *"Compare with 14"*, *"Compare to medians"*, *"recommend a loan"*, *"show visuals"*.


**You can choose between**:
- **Mirror Tab 5 (Configurable Rules)** — uses the same rules dict modified in Tab 5.
- **Baseline Fixed Logic** — uses your `check_loan_eligibility` (multi-loan supported).
""")

    rule_mode = st.radio(
        "Eligibility Logic for this Tab:",
        ["Mirror Tab 5 (Configurable Rules)", "Baseline Fixed Logic"],
        index=0,
        horizontal=True,
        key="t8_rule_mode"
    )

    if rule_mode.startswith("Mirror"):
        rules = st.session_state.get("eligibility_rules", default_rules())
        st.info("Using **Tab 5 rules** (configurable). Edit them in Tab 5; results here will match.")
    else:
        rules = None
        st.info("Using **Baseline Fixed Logic** (via `check_loan_eligibility`).")

    with st.expander("📜 Show current rule parameters"):
        if rules is not None:
            st.json(rules)
        else:
            st.caption("Baseline fixed logic (no adjustable settings here).")

    # ----------------------------
    # Upload Applicants
    # ----------------------------
    up = st.file_uploader("📤 Upload Applicants (xlsx/csv)", type=["xlsx", "csv"], key="t8_upload")
    if not up:
        st.info("Upload a file to begin.")
        return

    try:
        if up.name.lower().endswith(".xlsx"):
            raw_df = pd.read_excel(up)
        else:
            raw_df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    st.success(f"Loaded {raw_df.shape[0]} applicants.")
    data_df = _normalize_schema(raw_df)

    # Compute eligibility
    if rules is not None:
        elg_df = check_loan_eligibility_configurable(data_df.copy(), rules)
    else:
        elg_df = check_loan_eligibility(data_df.copy())

    # Save for Q&A later
    st.session_state["t8_elg_df"] = elg_df.copy()

    # Show results
    show_cols = [c for c in CONTEXT_FIELDS if c in elg_df.columns] + ["Eligibility"]
    st.subheader("📋 Eligibility (Multi-Loan Lists Preserved)")
    st.dataframe(elg_df[show_cols], use_container_width=True)

    # Quick summary/filters
    with st.expander("🔎 Quick filters / stats"):
        loan_filter = st.multiselect(
            "Filter by loan type(s):",
            options=["7(a)", "504", "Express", "8(a)", "Ineligible"],
            default=[],
            key="t8_filter"
        )
        if loan_filter:
            mask = elg_df["Eligibility"].apply(lambda xs: any(l in xs for l in loan_filter))
            st.dataframe(elg_df.loc[mask, show_cols], use_container_width=True)
        st.markdown("**Counts by loan membership** (multi-loan allowed per row):")
        counts = {l: int(elg_df["Eligibility"].apply(lambda xs, l=l: l in xs).sum())
                  for l in ["7(a)", "504", "Express", "8(a)", "Ineligible"]}
        st.json(counts)

    # ----------------------------
    # Q&A Assistant (Manual) + Visual Holder
    # ----------------------------
    st.markdown("---")
    st.subheader("💬 Ask about an Applicant (Manual Assistant)")

    if "t8_chat" not in st.session_state:
        st.session_state["t8_chat"] = []
    if "t8_viz_id" not in st.session_state:
        st.session_state["t8_viz_id"] = None

    # This container will persist visuals after rerun
    viz_area = st.container()

    colq1, colq2 = st.columns([2,1])
    with colq1:
        q = st.text_input("Ask a question (e.g., 'Why not 504?' or 'Compare with 14' or 'Show visuals')", key="t8_q_text")
    with colq2:
        sel_id = st.selectbox("Applicant ID", elg_df["Applicant ID"].tolist(), key="t8_sel_id")

    ask = st.button("Ask", key="t8_ask_btn")
    clear = st.button("Clear Chat", key="t8_clear_btn")
    if clear:
        st.session_state["t8_chat"] = []
        st.session_state["t8_viz_id"] = None
        _safe_rerun()

    def _summary_for(app_id: int) -> Dict[str, Any]:
        row = elg_df.loc[elg_df["Applicant ID"] == app_id].iloc[0].to_dict()
        loans = row.get("Eligibility", ["Ineligible"])
        return generate_applicant_summary(row, loans)

    def _render_visuals(app_id: int):
        """
        Radar + KPI-style metrics for selected applicant.
        """
        row = elg_df.loc[elg_df["Applicant ID"] == app_id].iloc[0]
        loans = row["Eligibility"]
        radar_cols = [
            "Personal Credit Score", "Business Credit Score", "DSCR (latest year)",
            "Years in Business", "Net Profit Margin"
        ]
        vals = [float(row.get(c, 0) or 0) for c in radar_cols]
        bounds = {
            "Personal Credit Score": (300, 850),
            "Business Credit Score": (0, 300),
            "DSCR (latest year)": (0.0, 3.0),
            "Years in Business": (0, 20),
            "Net Profit Margin": (-50.0, 50.0),
        }
        scaled = []
        for c, v in zip(radar_cols, vals):
            lo, hi = bounds[c]
            v = max(lo, min(hi, v))
            scaled.append((v - lo) / (hi - lo + 1e-9))

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(theta=radar_cols, r=scaled, fill='toself', name=f"Applicant {app_id}"))
        fig.update_layout(showlegend=False, title=f"Radar — Applicant {app_id}")
        st.plotly_chart(fig, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        for label, container in zip(["Personal Credit Score", "Business Credit Score", "DSCR (latest year)"], [k1, k2, k3]):
            val = float(row.get(label, 0) or 0)
            lo, hi = bounds[label]
            pct = (max(lo, min(hi, val)) - lo) / (hi - lo + 1e-9) * 100
            container.metric(label=label, value=f"{val:.2f}", delta=f"{pct:.0f} pctile approx")

        st.markdown("**Eligible loan(s):** " + " ".join(
            f"<span style='background:#e7f5ff;padding:4px 8px;border-radius:8px;margin-right:6px'>{l if isinstance(l, str) else 'None'}</span>"
            for l in (loans if isinstance(loans, list) else [loans])
        ), unsafe_allow_html=True)

    def _offline_answer(user_q: str, selected_id: int) -> tuple[str, bool]:
        """
        Deterministic Q&A handling (no LLM).
        Returns (answer_text, show_visuals_flag)
        """
        try:
            row = elg_df.loc[elg_df["Applicant ID"] == selected_id].iloc[0]
        except Exception:
            return "Applicant not found.", False

        ql = user_q.strip().lower()
        loans = row["Eligibility"] if isinstance(row["Eligibility"], list) else [row["Eligibility"]]
        summary = _summary_for(selected_id)

        # Why not {loan}?
        if "why not " in ql:
            loan = None
            if "504" in ql: loan = "504"
            elif "7(" in ql or "7a" in ql: loan = "7(a)"
            elif "express" in ql: loan = "Express"
            elif "8(" in ql or "8a" in ql: loan = "8(a)"
            if loan:
                if loan in loans:
                    return f"Applicant {selected_id} **is eligible** for {loan}.", False
                else:
                    if rules is not None:  # Mirror mode
                        fails = _why_not_loan_mirror(row, rules, loan)
                    else:
                        fails = _why_not_loan_fixed(row, loan)
                    return ("Not eligible for **" + loan + "** because:\n" +
                            ("\n".join(f"- {f}" for f in fails) if fails else "no specific failing conditions were triggered.")), False
            return "Specify a loan: try 'Why not 504?' or 'Why not 7(a)?' or 'Why not Express?'.", False

        # Top strengths / top risks
        if "strength" in ql:
            if summary["strengths"]:
                return "Top strengths:\n" + "\n".join([f"- {s}" for s in summary["strengths"][:8]]), False
            return "No major strengths highlighted.", False
        if "risk" in ql or "weak" in ql or "concern" in ql:
            if summary["risks"]:
                return "Top risks:\n" + "\n".join([f"- {r}" for r in summary["risks"][:8]]), False
            return "No significant risks identified.", False

        # Compare with ID or medians
        if "compare" in ql:
            m = re.search(r"compare with\s+(\d+)", ql)
            if m and "Applicant ID" in elg_df.columns:
                try:
                    other_id = int(m.group(1))
                    other_row = elg_df[elg_df["Applicant ID"] == other_id].iloc[0]
                    lines = []
                    for c in NUMERIC_DISPLAY:
                        if c in row and c in other_row:
                            delta = float(row[c]) - float(other_row[c])
                            lines.append(f"- {c}: {float(row[c]):,.2f} vs Applicant {other_id} = {float(other_row[c]):,.2f} (Δ {delta:+,.2f})")
                    return ("\n".join(lines) if lines else f"Cannot find numeric fields to compare with Applicant {other_id}."), False
                except Exception:
                    return f"Applicant {other_id} not found.", False
            return ("Comparison to dataset median:\n" + _compare_to_median(row, elg_df)), False

        # Eligible loans?
        if "loan" in ql or "eligible" in ql:
            return f"Eligible loans: {', '.join(loans)}", False

        # Recommendation
        if "recommend" in ql or "best" in ql:
            return _recommend_loan(loans, row.to_dict()), False

        # Visuals suggestion or show
        if any(kw in ql for kw in ["visual", "plot", "chart", "graph", "show"]):
            # We will set a flag to render visuals after rerun / app pass
            return "Rendering visuals below (radar & KPIs)...", True

        # generic fallback
        return (f"**Summary:** {summary['narrative']}\n\nRecommendation: {summary['recommendation']}"), False

    # Show chat history
    for who, msg in st.session_state["t8_chat"]:
        if who == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    # Ask button behavior
    if ask and q.strip():
        st.session_state["t8_chat"].append(("user", q))
        reply, show_viz = _offline_answer(q, sel_id)
        st.session_state["t8_chat"].append(("assistant", reply))
        if show_viz:
            st.session_state["t8_viz_id"] = sel_id
        _safe_rerun()

    # If a visual is pending, render it now in the dedicated area
    if st.session_state.get("t8_viz_id"):
        with viz_area:
            _render_visuals(st.session_state["t8_viz_id"])
        # Clear after rendering once
        st.session_state["t8_viz_id"] = None

    if not st.session_state["t8_chat"]:
        st.caption("Try: 'Why not 504?', 'Top strengths', 'Compare with 3', 'Show visuals', or 'Recommend a loan'.")
