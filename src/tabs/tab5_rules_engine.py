# src/tabs/tab5_rules_engine.py
import pandas as pd
import streamlit as st

from src.rules import (
    check_loan_eligibility_configurable,
    default_rules,
    generate_applicant_summary,
    rules_summary_by_loan,  # optional: to display summarized rule set per loan
)

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def _clear_form_state(prev_id):
    """Clear all form widget state associated with the previous applicant to force reload."""
    if prev_id is None:
        return
    # Keys to purge start with these prefixes and the previous id
    prefixes = ["t5_num_", "t5_bool_"]
    keys_to_delete = []
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) and k.endswith(f"_{prev_id}") for p in prefixes):
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del st.session_state[k]

def render():
    st.header("✅ Rules Engine (No-ML) — Upload, Review & Decide")

    st.markdown(
        """
Use the **policy-based eligibility** (no ML) to see which SBA products an applicant qualifies for.  
1) Configure the **rules** if needed (per-loan).  
2) Upload your file (`.xlsx` or `.csv`).  
3) Pick an applicant and review the **auto-filled form**.  
4) Submit to see the **eligible loans** instantly. (Multi-loan supported.)
        """
    )

    # ---------- Rules Config UI ----------
    st.subheader("⚙️ Configure Rules — Per Loan")
    with st.expander("Show / Edit Rules", expanded=False):
        defaults = default_rules()
        rules_state = st.session_state.get("eligibility_rules", defaults.copy())

        # 7(a)
        st.markdown("### **7(a) Parameters**")
        c1, c2, c3, c4 = st.columns(4)
        rules_state["7a_requires_for_profit"] = c1.checkbox(
            "Requires For-Profit", value=rules_state.get("7a_requires_for_profit", defaults["7a_requires_for_profit"]), key="t5_r7a_fp"
        )
        rules_state["7a_min_personal_credit"] = c2.number_input(
            "Min Personal Credit", 300, 850, int(rules_state.get("7a_min_personal_credit", defaults["7a_min_personal_credit"])), key="t5_r7a_min_pcs"
        )
        rules_state["7a_min_business_credit"] = c3.number_input(
            "Min Business Credit", 0, 300, int(rules_state.get("7a_min_business_credit", defaults["7a_min_business_credit"])), key="t5_r7a_min_bcs"
        )
        rules_state["7a_min_dscr"] = c4.number_input(
            "Min DSCR", 0.0, 5.0, float(rules_state.get("7a_min_dscr", defaults["7a_min_dscr"])), step=0.01, format="%.2f", key="t5_r7a_min_dscr"
        )

        c5, c6, c7 = st.columns(3)
        rules_state["7a_min_years_in_business"] = c5.number_input(
            "Min Years in Business", 0, 100, int(rules_state.get("7a_min_years_in_business", defaults["7a_min_years_in_business"])), key="t5_r7a_min_yib"
        )
        rules_state["7a_loan_min"] = c6.number_input(
            "Min Loan Amount", 0, 10_000_000, int(rules_state.get("7a_loan_min", defaults["7a_loan_min"])), step=1000, key="t5_r7a_loan_min"
        )
        rules_state["7a_loan_max"] = c7.number_input(
            "Max Loan Amount", 0, 10_000_000, int(rules_state.get("7a_loan_max", defaults["7a_loan_max"])), step=1000, key="t5_r7a_loan_max"
        )
        rules_state["7a_exclude_purposes"] = st.multiselect(
            "Excluded Purposes (7a)",
            options=rules_state.get("all_purposes", defaults["all_purposes"]),
            default=rules_state.get("7a_exclude_purposes", defaults["7a_exclude_purposes"]),
            key="t5_r7a_excl_purp"
        )

        # 8(a)
        st.markdown("### **8(a) Parameters**")
        enable_8a_col = st.columns(1)[0]
        rules_state["enable_8a"] = enable_8a_col.checkbox(
            "Enable 8(a) Rule",
            value=rules_state.get("enable_8a", defaults["enable_8a"]),
            key="t5_r8a_enable"
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        rules_state["8a_requires_for_profit"] = c1.checkbox(
            "Requires For-Profit?", value=rules_state.get("8a_requires_for_profit", defaults["8a_requires_for_profit"]), key="t5_r8a_forprofit"
        )
        rules_state["8a_requires_fast_approval"] = c2.checkbox(
            "Requires NOT Fast Approval", value=rules_state.get("8a_requires_fast_approval", defaults["8a_requires_fast_approval"]), key="t5_r8a_notfast"
        )
        rules_state["8a_min_years_in_business"] = c3.number_input(
            "Min Years in Business", 0, 50, int(rules_state.get("8a_min_years_in_business", defaults["8a_min_years_in_business"])), key="t5_r8a_yib"
        )
        rules_state["8a_min_industry_exp"] = c4.number_input(
            "Min Industry Exp (yrs)", 0, 50, int(rules_state.get("8a_min_industry_exp", defaults["8a_min_industry_exp"])), key="t5_r8a_indexp"
        )
        rules_state["8a_min_managerial_exp"] = c5.number_input(
            "Min Managerial Exp (yrs)", 0, 50, int(rules_state.get("8a_min_managerial_exp", defaults["8a_min_managerial_exp"])), key="t5_r8a_mngexp"
        )
        rules_state["8a_exclude_purposes"] = st.multiselect(
            "Excluded Purposes (8a)",
            options=rules_state.get("all_purposes", defaults["all_purposes"]),
            default=rules_state.get("8a_exclude_purposes", defaults["8a_exclude_purposes"]),
            key="t5_r8a_excl_purp"
        )

        # 504
        st.markdown("### **504 Parameters**")
        c1, c2, c3, c4 = st.columns(4)
        rules_state["504_requires_for_profit"] = c1.checkbox(
            "Requires For-Profit", value=rules_state.get("504_requires_for_profit", defaults["504_requires_for_profit"]), key="t5_r504_fp"
        )
        rules_state["504_requires_collateral"] = c2.checkbox(
            "Requires Collateral", value=rules_state.get("504_requires_collateral", defaults["504_requires_collateral"]), key="t5_r504_coll"
        )
        rules_state["504_min_personal_credit"] = c3.number_input(
            "Min Personal Credit", 300, 850, int(rules_state.get("504_min_personal_credit", defaults["504_min_personal_credit"])), key="t5_r504_minpcs"
        )
        rules_state["504_min_dscr"] = c4.number_input(
            "Min DSCR", 0.0, 5.0, float(rules_state.get("504_min_dscr", defaults["504_min_dscr"])), step=0.01, format="%.2f", key="t5_r504_mindscr"
        )

        c5, c6, c7 = st.columns(3)
        rules_state["504_min_net_profit_margin"] = c5.number_input(
            "Min Net Profit Margin", -100.0, 100.0, float(rules_state.get("504_min_net_profit_margin", defaults["504_min_net_profit_margin"])),
            step=0.1, format="%.1f", key="t5_r504_minnpm"
        )
        rules_state["504_min_years_in_business"] = c6.number_input(
            "Min Years in Business", 0, 100, int(rules_state.get("504_min_years_in_business", defaults["504_min_years_in_business"])), key="t5_r504_minyib"
        )
        rules_state["504_max_loan"] = c7.number_input(
            "Max Loan Amount", 0, 10_000_000, int(rules_state.get("504_max_loan", defaults["504_max_loan"])),
            step=1000, key="t5_r504_maxloan"
        )
        rules_state["504_exclude_purposes"] = st.multiselect(
            "Excluded Purposes (504)",
            options=rules_state.get("all_purposes", defaults["all_purposes"]),
            default=rules_state.get("504_exclude_purposes", defaults["504_exclude_purposes"]),
            key="t5_r504_excl_purp"
        )

        # Express
        st.markdown("### **Express Parameters**")
        e1, e2, e3, e4 = st.columns(4)
        rules_state["express_requires_for_profit"] = e1.checkbox(
            "Requires For-Profit", value=rules_state.get("express_requires_for_profit", defaults["express_requires_for_profit"]), key="t5_rexp_fp"
        )
        rules_state["express_requires_fast_approval"] = e2.checkbox(
            "Requires Fast Approval", value=rules_state.get("express_requires_fast_approval", defaults["express_requires_fast_approval"]), key="t5_rexp_fast"
        )
        rules_state["express_min_personal_credit"] = e3.number_input(
            "Min Personal Credit", 300, 850, int(rules_state.get("express_min_personal_credit", defaults["express_min_personal_credit"])), key="t5_rexp_pcs"
        )
        rules_state["express_min_business_credit"] = e4.number_input(
            "Min Business Credit", 0, 300, int(rules_state.get("express_min_business_credit", defaults["express_min_business_credit"])), key="t5_rexp_bcs"
        )

        e5, e6 = st.columns(2)
        rules_state["express_min_dscr"] = e5.number_input(
            "Min DSCR", 0.0, 5.0, float(rules_state.get("express_min_dscr", defaults["express_min_dscr"])),
            step=0.01, format="%.2f", key="t5_rexp_dscr"
        )
        rules_state["express_max_loan"] = e6.number_input(
            "Max Loan Amount", 0, 10_000_000, int(rules_state.get("express_max_loan", defaults["express_max_loan"])),
            step=1000, key="t5_rexp_maxloan"
        )
        rules_state["express_exclude_purposes"] = st.multiselect(
            "Excluded Purposes (Express)",
            options=rules_state.get("all_purposes", defaults["all_purposes"]),
            default=rules_state.get("express_exclude_purposes", defaults["express_exclude_purposes"]),
            key="t5_rexp_excl_purp"
        )

        st.session_state["eligibility_rules"] = rules_state
        
        st.caption("These settings affect only this tab.")

        with st.expander("🔎 View Summary of Rules by Loan"):
            summary = rules_summary_by_loan(rules_state)
            for loan, lines in summary.items():
                st.markdown(f"**{loan}**")
                for ln in lines:
                    st.markdown(f"- {ln}")

    # ---------- Upload ----------
    up = st.file_uploader("📤 Upload Applicants File", type=["xlsx", "csv"], key="t5_rules_upload")

    numeric_fields = [
        ("Personal Credit Score", 0, 850, 1.0),
        ("Business Credit Score", 0, 300, 1.0),
        ("DSCR (latest year)", 0.0, 5.0, 0.01),
        ("Annual Revenue (latest year)", 0, 1_000_000_000, 1000.0),
        ("Loan Amount", 0, 6_000_000, 1000.0),
        ("Years in Business", 0, 100, 1.0),
        ("Net Profit Margin", -100.0, 100.0, 0.1),
        ("NOI (1 year ago)", 0, 10_000_000_000, 1000.0),
        ("NOI (2 years ago)", 0, 10_000_000_000, 1000.0),
        ("Industry Experience", 0, 100, 1.0),
        ("Managerial Experience", 0, 100, 1.0),
    ]

    boolean_fields = [
        "For Profit", "Fast Approval", "Collateral Availability",
        "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
        "Inventory Purchase", "Real Estate Acquisition or Improvement",
        "Business Acquisition or Buyout", "Refinancing Existing Debt",
        "Emergency Funds", "Franchise Financing", "Contract Financing",
        "Licensing or Permits", "Line of Credit Establishment"
    ]

    if not up:
        st.info("📂 Upload a `.xlsx` or `.csv` file to start.")
        return

    try:
        if up.name.lower().endswith(".xlsx"):
            rules_df = pd.read_excel(up)
        else:
            rules_df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return
    

    if "Applicant ID" not in rules_df.columns:
        rules_df["Applicant ID"] = range(1, len(rules_df) + 1)

    for c, *_ in numeric_fields:
        if c not in rules_df.columns:
            rules_df[c] = 0
    for c in boolean_fields:
        if c not in rules_df.columns:
            rules_df[c] = False
        rules_df[c] = _to_bool_series(rules_df[c])

    rules = st.session_state["eligibility_rules"]
    try:
        preview_df = check_loan_eligibility_configurable(rules_df.copy(), rules)
    except Exception as e:
        st.error(f"Eligibility rule error: {e}")
        return

    show_cols = [c for c in ["Applicant ID", "Business Name", "Eligibility"] if c in preview_df.columns]
    st.subheader("📄 Preview (Rule Output)")
    st.dataframe(preview_df[show_cols] if show_cols else preview_df[["Applicant ID", "Eligibility"]],
                 use_container_width=True)
    # ✅ Make rule-based outputs (Applicant ID + Eligibility) available to other tabs
    st.session_state["tab5_rules_output_df"] = preview_df[["Applicant ID", "Eligibility"]].copy()


    st.subheader("👤 Pick an Applicant")

    # maintain last picked to clear state when switching applicants
    last_picked = st.session_state.get("t5_last_picked_id", None)
    picked_id = st.selectbox("Applicant ID",
                             options=preview_df["Applicant ID"].tolist(),
                             index=0,
                             key="t5_rules_pick_id")

    if last_picked is None or picked_id != last_picked:
        _clear_form_state(last_picked)
        st.session_state["t5_last_picked_id"] = picked_id  # track current
        _safe_rerun()
        return

    # Locate selected row by Applicant ID
    try:
        sel_idx = preview_df.index[preview_df["Applicant ID"] == picked_id][0]
    except Exception:
        st.error("Could not find selected applicant in the preview.")
        return

    row = rules_df.loc[sel_idx]

    st.subheader("📝 Review / Edit Applicant Inputs")
    with st.form("t5_rules_form"):
        colA, colB = st.columns(2)
        num_values, bool_values = {}, {}

        # Use dynamic keys tied to applicant to ensure correct default values apply
        for i, (fname, vmin, vmax, step) in enumerate(numeric_fields):
            default_val = float(row.get(fname, 0) or 0)
            container = colA if i % 2 == 0 else colB
            num_values[fname] = container.number_input(
                fname,
                min_value=float(vmin), max_value=float(vmax),
                value=float(default_val), step=float(step),
                format="%.4f" if step < 1 else "%.0f",
                key=f"t5_num_{fname}_{picked_id}"
            )

        st.markdown("**Flags & Purposes**")
        bcol1, bcol2 = st.columns(2)
        for j, bname in enumerate(boolean_fields):
            default_bool = bool(row.get(bname, False))
            container = bcol1 if j % 2 == 0 else bcol2
            bool_values[bname] = container.checkbox(
                bname,
                value=default_bool,
                key=f"t5_bool_{bname}_{picked_id}"
            )

        submitted = st.form_submit_button("Run Rules", use_container_width=True)

    if not submitted:
        st.info("Adjust fields if needed, then click **Run Rules** to generate eligibility and assessment.")
        return

    # Build full context dict: start from the selected row, then apply form edits
    single = row.to_dict()
    single["Applicant ID"] = picked_id
    single.update(num_values)
    single.update({k: bool(v) for k, v in bool_values.items()})

    # Evaluate rules for this single-row application
    try:
        out = check_loan_eligibility_configurable(pd.DataFrame([single]), rules)
        loans = out.loc[0, "Eligibility"]
        if isinstance(loans, str):
            loans = [loans]
    except Exception as e:
        st.error(f"Rule evaluation failed: {e}")
        loans = ["Ineligible"]

    st.markdown("### 📌 Eligible Loan(s)")
    if loans and isinstance(loans, list) and loans != ["Ineligible"]:
        st.markdown(
            " ".join([
                f"<span style='background:#e7f5ff;padding:4px 8px;border-radius:8px;margin-right:6px'>{l}</span>"
                for l in loans
            ]),
            unsafe_allow_html=True
        )
    else:
        st.warning("Ineligible based on current inputs.")

    with st.expander("🔎 Show evaluated inputs"):
        st.json(single)

    # Generate the narrative summary using full-context dict
    try:
        summary = generate_applicant_summary(single, loans)
    except Exception:
        summary = {
            "score": 0, "label": "Needs Review",
            "strengths": [], "risks": ["Summary unavailable."],
            "recommendation": "Review inputs.", "narrative": "Unable to generate detailed summary."
        }

    st.subheader("🧾 Overall Assessment")
    badge_color = {
        "Strong Fit": "#2b8a3e", "Good Fit": "#1971c2", "Borderline": "#e67700",
        "Needs Review": "#c92a2a", "Ineligible": "#5f3dc4"
    }.get(summary["label"], "#444")

    st.markdown(
        f"<div style='padding:10px 12px;border-radius:10px;background:{badge_color}1A;border:1px solid {badge_color};'>"
        f"<b>Fit:</b> <span style='color:{badge_color}'>{summary['label']}</span> &nbsp; "
        f"<b>Score:</b> {summary['score']}/100</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"**Summary:** {summary['narrative']}")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**✅ Strengths**")
        if summary["strengths"]:
            for s in summary["strengths"][:8]:
                st.markdown(f"- {s}")
        else:
            st.markdown("- None highlighted.")

    with cols[1]:
        st.markdown("**⚠️ Risks / Further Review**")
        if summary["risks"]:
            for r in summary["risks"][:8]:
                st.markdown(f"- {r}")
        else:
            st.markdown("- No material flags from current inputs.")

    st.markdown("**📌 Recommendation**")
    st.markdown(f"- {summary['recommendation']}")
