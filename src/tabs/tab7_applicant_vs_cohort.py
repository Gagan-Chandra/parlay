# loan_app/src/tabs/tab7_applicant_vs_cohort.py
from __future__ import annotations
import ast
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors


def _normalize_pred_label(pred_label):
    """Turn '7(a)', "['7(a)', '504']", ['7(a)'] -> list[str]."""
    if isinstance(pred_label, list):
        return [str(x) for x in pred_label]

    if isinstance(pred_label, str):
        s = pred_label.strip()
        # stringified list
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]

    return [str(pred_label)]


def render(new_df: pd.DataFrame | None = None):
    """
    Applicant vs Cohort
    - uses ENTIRE dataset (no train/test split)
    - uses ONLY the single ML model output from Tab 4 (Eligibility_Pred)
    - keeps same visuals (z-score table, radar, NN, dists, boxplots)
    """
    st.header("👤 Applicant vs Cohort Comparison")

    # 1) get data from session (Tab 4 must have run)
    ml_df = st.session_state.get("pred_results_df")        # predictions with Eligibility_Pred + probas
    feat_df = st.session_state.get("pred_features_df")     # engineered features from Tab 4
    raw_input_df = st.session_state.get("t4_input_df")     # normalized original applicant input

    if ml_df is None or ml_df.empty or feat_df is None or feat_df.empty:
        st.warning("Run predictions in **Tab 4: Live Predictions** first, then come back here.")
        return

    # 2) build cohort = entire data
    #    priority: use Tab 4 normalized input (better columns), else fall back to ml_df
    if raw_input_df is not None and not raw_input_df.empty:
        cohort_df = raw_input_df.copy()
        if "Applicant ID" in ml_df.columns:
            cohort_df = cohort_df.merge(
                ml_df[["Applicant ID", "Eligibility_Pred"]],
                on="Applicant ID",
                how="left"
            )
        else:
            cohort_df["Eligibility_Pred"] = "Ineligible"
    else:
        # if Tab 4 didn't store raw input for some reason, use whatever ML produced
        cohort_df = ml_df.copy()

    # 3) pick applicant
    st.markdown("Pick an **applicant** and we'll compare them to others with the **same predicted loans**.")
    sel_app_id = st.selectbox(
        "Applicant ID",
        ml_df["Applicant ID"].tolist(),
        index=0,
        key="t7_applicant_id_select"
    )

    # locate applicant row in ML df
    try:
        app_idx = ml_df.index[ml_df["Applicant ID"] == sel_app_id][0]
    except Exception:
        st.error("Selected Applicant ID not found in ML results.")
        return

    # we only have one model → use Eligibility_Pred
    raw_label = ml_df.loc[app_idx, "Eligibility_Pred"]
    pred_loans = _normalize_pred_label(raw_label)
    st.markdown(f"**Predicted eligibility for this applicant:** `{pred_loans}`")

    # 4) cohort with same loan(s)
    def _has_overlap(row_loans):
        row_loans = _normalize_pred_label(row_loans)
        return any(l in row_loans for l in pred_loans)

    cohort_same = cohort_df[cohort_df["Eligibility_Pred"].apply(_has_overlap)]
    st.markdown(f"**Cohort size (same predicted loan):** {len(cohort_same)}")
    if len(cohort_same) == 0:
        st.warning("No cohort rows with the same predicted eligibility.")
        return

    # 5) get applicant row for numeric comparison
    if raw_input_df is not None and not raw_input_df.empty and "Applicant ID" in raw_input_df.columns:
        app_row = raw_input_df.loc[raw_input_df["Applicant ID"] == sel_app_id].iloc[[0]]
    else:
        # fallback to feature df if raw not available
        app_row = feat_df.iloc[[app_idx]].copy()

    # numeric columns common to both
    num_cols_cohort = cohort_same.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_app = app_row.select_dtypes(include=[np.number]).columns.tolist()
    comp_num_cols = [c for c in num_cols_cohort if c in num_cols_app]

    if not comp_num_cols:
        st.warning("No common numeric features to compare.")
        return

    # 6) stats + z-score table
    stats = cohort_same[comp_num_cols].describe().T
    means = stats["mean"]
    stds = stats["std"].replace(0, np.nan)
    med = stats["50%"]

    app_vals = app_row[comp_num_cols].iloc[0].astype(float)

    zscore = (app_vals - means) / stds
    pct = cohort_same[comp_num_cols].apply(lambda s, v=app_vals: (s <= v[s.name]).mean() * 100)

    comp_table = pd.DataFrame({
        "Applicant": app_vals,
        "Cohort Median": med,
        "Z-Score": zscore,
        "Percentile": pct.round(1)
    }).sort_values("Percentile", ascending=False)

    st.subheader("📋 Applicant vs Cohort (Numeric Features)")
    st.dataframe(comp_table, use_container_width=True)

    # 7) radar
    try:
        top_rad = comp_table.reindex(
            comp_table["Z-Score"].abs().sort_values(ascending=False).head(6).index
        )
        theta = top_rad.index.tolist()
        rel = (
            (top_rad["Applicant"] - top_rad["Cohort Median"])
            / (top_rad["Cohort Median"].replace(0, np.nan))
        ).fillna(0).values
        rel = np.clip(rel, -3, 3)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=rel,
            theta=theta,
            fill='toself',
            name='Applicant vs Cohort'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
            showlegend=False,
            title="Applicant tilt vs Cohort Median (relative)"
        )
        st.plotly_chart(fig, use_container_width=True, key="t7_polar_plot")
    except Exception as e:
        st.info(f"Polar chart unavailable: {e}")

    # 8) nearest neighbors (on SAME-LOAN cohort)
    try:
        nn = NearestNeighbors(
            n_neighbors=min(5, len(cohort_same)),
            metric="euclidean"
        )
        X_tr = cohort_same[comp_num_cols].fillna(0)
        X_app = app_row[comp_num_cols].fillna(0)
        nn.fit(X_tr)
        dists, idxs = nn.kneighbors(X_app)
        nn_rows = cohort_same.iloc[idxs[0]].copy()
        nn_rows.insert(0, "Distance", dists[0])
        show_nn_cols = [c for c in ["Applicant ID", "Business Name", "Eligibility_Pred"] if c in nn_rows.columns]

        st.subheader("🔎 Most Similar Applicants (Same Predicted Loan)")
        st.dataframe(
            nn_rows[["Distance"] + show_nn_cols + comp_num_cols[:5]].reset_index(drop=True),
            use_container_width=True
        )
    except Exception as e:
        st.info(f"Nearest-neighbor view unavailable: {e}")

    # 9) distribution overlays
    st.markdown("---")
    st.subheader("📊 Distribution: Cohort vs Applicant (pick up to 4)")
    default_feats = [c for c in comp_num_cols if c in [
        "Loan Amount",
        "Personal Credit Score",
        "DSCR (latest year)",
        "Annual Revenue (latest year)"
    ][:4]]
    choose_feats = st.multiselect(
        "Features",
        options=comp_num_cols,
        default=default_feats,
        max_selections=4,
        key="t7_dist_feats"
    )
    if choose_feats:
        cols = st.columns(2) if len(choose_feats) > 1 else [st]
        for i, feat in enumerate(choose_feats):
            pane = cols[i % len(cols)]
            try:
                fig = px.histogram(
                    cohort_same,
                    x=feat,
                    nbins=30,
                    title=f"{feat} — Cohort Distribution",
                    marginal="box",
                    opacity=0.85
                )
                fig.add_vline(
                    x=float(app_row[feat].iloc[0]),
                    line_width=3,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="Applicant",
                    annotation_position="top"
                )
                pane.plotly_chart(fig, use_container_width=True, key=f"t7_hist_{feat}")
            except Exception as e:
                pane.info(f"Could not build histogram for `{feat}`: {e}")
    else:
        st.info("Pick at least one numeric feature above to see distribution overlays.")

    # 10) boxplot
    st.markdown("---")
    st.subheader("📦 Multi-Feature Boxplot (Top |Z| Features)")
    try:
        top_box = comp_table.reindex(
            comp_table["Z-Score"].abs().sort_values(ascending=False).head(8).index
        )
        box_feats = top_box.index.tolist()
        if box_feats:
            long_df = cohort_same[box_feats].melt(var_name="Feature", value_name="Value")
            fig = px.box(long_df, x="Feature", y="Value", points=False, title="Top Deviations — Cohort Boxplots")
            for f in box_feats:
                fig.add_scatter(
                    x=[f],
                    y=[float(app_row[f].iloc[0])],
                    mode="markers",
                    marker=dict(size=12),
                    name=f"Applicant • {f}",
                    showlegend=False
                )
            st.plotly_chart(fig, use_container_width=True, key="t7_box_multi")
        else:
            st.info("No high-deviation features available for boxplot.")
    except Exception as e:
        st.info(f"Boxplot unavailable: {e}")
