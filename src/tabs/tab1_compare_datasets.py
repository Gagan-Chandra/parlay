# loan_app/src/tabs/tab1_compare_datasets.py
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp

def render(old_df: pd.DataFrame, new_df: pd.DataFrame):
    all_loans = sorted(
        set([l for sub in old_df["Eligibility"].tolist() for l in sub] +
            [l for sub in new_df["Eligibility"].tolist() for l in sub])
    )
    loan_types = [l for l in all_loans if l in ["7(a)", "504", "Express"]]
    selected_loan = st.sidebar.selectbox("🔹 Select Loan Type", loan_types)

    old_sub = old_df[old_df["Eligibility"].apply(lambda x: selected_loan in x)]
    new_sub = new_df[new_df["Eligibility"].apply(lambda x: selected_loan in x)]

    st.markdown(f"## 🔹 Loan Type: **{selected_loan}** Comparison")
    st.caption("Includes applicants eligible for this loan even if they qualify for multiple.")

    numeric_cols = [
        "Personal Credit Score", "Business Credit Score", "DSCR (latest year)",
        "Annual Revenue (latest year)", "Years in Business", "Loan Amount",
        "Industry Experience", "Managerial Experience"
    ]
    similarities = []
    for col in numeric_cols:
        if col in old_df.columns and col in new_df.columns:
            try:
                stat, _ = ks_2samp(old_sub[col].dropna(), new_sub[col].dropna())
                similarities.append({"Feature": col, "KS Similarity": round(1 - stat, 3)})
            except Exception:
                pass
    sim_df = pd.DataFrame(similarities).sort_values("KS Similarity", ascending=False)
    st.markdown("### 📈 Numeric Feature Similarity (KS Test)")
    st.dataframe(sim_df)

    fig = px.bar(sim_df, x="KS Similarity", y="Feature", orientation="h",
                 color="KS Similarity", color_continuous_scale="tealrose",
                 title=f"Numeric Feature Similarity for {selected_loan}")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Visual Distribution Comparison (Top 3 Numeric Features)")
    top_features = sim_df.head(3)["Feature"].tolist()
    for col in top_features:
        fig, ax = plt.subplots(figsize=(6, 3))
        try:
            sns.kdeplot(old_sub[col], label="Original", fill=True, alpha=0.4)
            sns.kdeplot(new_sub[col], label="Synthetic", fill=True, alpha=0.4)
            ax.set_title(f"{col} Distribution Comparison")
            ax.legend()
            st.pyplot(fig)
        except Exception:
            pass

    if not sim_df.empty:
        avg_sim = round(sim_df["KS Similarity"].mean(), 3)
        st.markdown(f"### ✅ Average Numeric Similarity: **{avg_sim}**")
