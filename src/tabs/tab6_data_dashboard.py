# loan_app/src/tabs/tab6_data_dashboard.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.features import ensure_list_eligibility
from src.io_utils import parse_state_from_location

def render(new_df: pd.DataFrame):
    st.header("📊 Portfolio Dashboard — Combined View")
    st.markdown("Analyze your **entire portfolio** with filters for State, Loan type, and key ranges.")

    # prefer cached train/test, else full synthetic
    df_tr = st.session_state.get("cohort_train")
    df_te = st.session_state.get("cohort_test")
    base = pd.concat([df_tr, df_te], axis=0, ignore_index=True) if (df_tr is not None and df_te is not None) else new_df.copy()

    base = ensure_list_eligibility(base)
    base = parse_state_from_location(base)
    base["EligibilityLabel"] = base["Eligibility"].apply(lambda xs: "|".join(xs) if isinstance(xs, list) else str(xs))

    all_atomic = sorted(set([l for sub in base["Eligibility"] for l in (sub if isinstance(sub, list) else [str(sub)])]))
    states = sorted([s for s in base["State"].dropna().unique().tolist() if isinstance(s, str) and len(s) > 0])

    st.markdown("### 🔎 Filters")
    c1, c2, c3 = st.columns([1.5, 1.5, 2])
    with c1:
        state_sel = st.multiselect("State (parsed from `location`)", options=states, default=[])
    with c2:
        loan_sel = st.multiselect("Loan Types", options=all_atomic, default=[])
    with c3:
        amt_min = float(base["Loan Amount"].min()) if "Loan Amount" in base else 0.0
        amt_max = float(base["Loan Amount"].max()) if "Loan Amount" in base else 0.0
        pcs_min = float(base["Personal Credit Score"].min()) if "Personal Credit Score" in base else 0.0
        pcs_max = float(base["Personal Credit Score"].max()) if "Personal Credit Score" in base else 0.0
        dscr_min = float(base["DSCR (latest year)"].min()) if "DSCR (latest year)" in base else 0.0
        dscr_max = float(base["DSCR (latest year)"].max()) if "DSCR (latest year)" in base else 0.0

    r1, r2, r3 = st.columns(3)
    with r1:
        sel_amt = st.slider("Loan Amount range", min_value=amt_min, max_value=amt_max, value=(amt_min, amt_max), step=1000.0) if amt_max>amt_min else (amt_min, amt_max)
    with r2:
        sel_pcs = st.slider("Personal Credit Score", min_value=pcs_min, max_value=pcs_max, value=(pcs_min, pcs_max), step=1.0) if pcs_max>pcs_min else (pcs_min, pcs_max)
    with r3:
        sel_dscr = st.slider("DSCR (latest year)", min_value=dscr_min, max_value=dscr_max, value=(dscr_min, dscr_max), step=0.01) if dscr_max>dscr_min else (dscr_min, dscr_max)

    filt = base.copy()
    if state_sel: filt = filt[filt["State"].isin(state_sel)]
    if loan_sel:  filt = filt[filt["Eligibility"].apply(lambda xs: any(l in xs for l in loan_sel))]
    if "Loan Amount" in filt: filt = filt[(filt["Loan Amount"]>=sel_amt[0]) & (filt["Loan Amount"]<=sel_amt[1])]
    if "Personal Credit Score" in filt: filt = filt[(filt["Personal Credit Score"]>=sel_pcs[0]) & (filt["Personal Credit Score"]<=sel_pcs[1])]
    if "DSCR (latest year)" in filt: filt = filt[(filt["DSCR (latest year)"]>=sel_dscr[0]) & (filt["DSCR (latest year)"]<=sel_dscr[1])]

    st.markdown("### 📌 KPIs")
    k1, k2, k3, k4 = st.columns(4)
    total = len(filt)
    avg_amt = float(filt["Loan Amount"].mean()) if "Loan Amount" in filt and total else 0.0
    med_pcs = float(filt["Personal Credit Score"].median()) if "Personal Credit Score" in filt and total else 0.0
    inel = int(filt["Eligibility"].apply(lambda xs: xs==["Ineligible"]).sum()) if "Eligibility" in filt else 0
    elig_rate = ((total - inel)/total*100.0) if total else 0.0
    k1.metric("Applicants (Filtered)", f"{total:,}")
    k2.metric("Avg Loan Amount", f"${avg_amt:,.0f}")
    k3.metric("Median PCS", f"{med_pcs:.0f}")
    k4.metric("Eligibility Rate", f"{elig_rate:.1f}%")

    st.markdown("### 📈 Dashboard")
    g1, g2 = st.columns(2)
    with g1:
        if "Loan Amount" in filt:
            st.plotly_chart(px.histogram(filt, x="Loan Amount", nbins=40, title="Loan Amount Distribution", marginal="box"), use_container_width=True)
    with g2:
        if "EligibilityLabel" in filt:
            pie_df = filt["EligibilityLabel"].value_counts().reset_index()
            pie_df.columns = ["EligibilityLabel","Count"]
            st.plotly_chart(px.pie(pie_df, names="EligibilityLabel", values="Count", title="Loan Type Mix"), use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        if "State" in filt and "Loan Amount" in filt and filt["State"].notna().any():
            top_states = filt["State"].value_counts().index[:12].tolist()
            sub = filt[filt["State"].isin(top_states)]
            st.plotly_chart(px.box(sub, x="State", y="Loan Amount", title="Loan Amount by State (Top 12)"), use_container_width=True)
    with g4:
        if all(c in filt for c in ["Personal Credit Score","DSCR (latest year)","EligibilityLabel"]):
            st.plotly_chart(px.scatter(filt, x="Personal Credit Score", y="DSCR (latest year)", color="EligibilityLabel",
                                       title="Credit Score vs DSCR by Loan Label", opacity=0.7), use_container_width=True)

    st.markdown("### 🗺️ State Comparison")
    if "State" in filt and filt["State"].notna().any():
        comp_states = sorted(filt["State"].dropna().unique().tolist())
        cs1, cs2 = st.columns(2)
        with cs1:
            sA = st.selectbox("State A", comp_states, index=0, key="dash_state_a")
        with cs2:
            sB = st.selectbox("State B", comp_states, index=min(1, len(comp_states)-1), key="dash_state_b")
        subA, subB = filt[filt["State"]==sA], filt[filt["State"]==sB]
        ca, cb, cc = st.columns(3)
        ca.metric(f"{sA} • Applicants", f"{len(subA):,}")
        cb.metric(f"{sB} • Applicants", f"{len(subB):,}")
        if "Loan Amount" in filt:
            cc.metric(f"{sA} vs {sB} • Avg Loan", f"${subA['Loan Amount'].mean():,.0f} / ${subB['Loan Amount'].mean():,.0f}")
        if "EligibilityLabel" in filt:
            def share(df): return df["EligibilityLabel"].value_counts(normalize=True)
            sA_sh, sB_sh = share(subA), share(subB)
            cats = sorted(set(sA_sh.index).union(set(sB_sh.index)))
            comp = pd.DataFrame({"EligibilityLabel": cats, sA:[sA_sh.get(c,0) for c in cats], sB:[sB_sh.get(c,0) for c in cats]})
            st.plotly_chart(px.bar(comp.melt(id_vars="EligibilityLabel", var_name="State", value_name="Share"),
                                   x="EligibilityLabel", y="Share", color="State", barmode="group",
                                   title="Loan Type Share by State"), use_container_width=True)

    st.markdown("### ⬇️ Download Filtered View")
    st.download_button("Download CSV", data=filt.to_csv(index=False).encode("utf-8"),
                       file_name="portfolio_filtered.csv", mime="text/csv")
