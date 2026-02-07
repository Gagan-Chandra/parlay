# loan_app/app.py
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


from src.silencer import silence_everything
from src.io_utils import load_local_datasets
from src.tabs.tab1_compare_datasets import render as tab1_render
from src.tabs.tab2_feature_eng import render as tab2_render
from src.tabs.tab3_model_reports import render as tab3_render
from src.tabs.tab4_live_predictions import render as tab4_render
from src.tabs.tab5_rules_engine import render as tab5_render
from src.tabs.tab6_data_dashboard import render as tab6_render
from src.tabs.tab7_applicant_vs_cohort import render as tab7_render
from src.tabs.tab8_ai_assistant import render as tab8_render
from src.tabs.tab9_ai_copilot import render as tab9_render
from src.tabs.tab10_compare_ml_vs_rules import render as tab10_render



st.set_page_config(page_title="Parlay Finance Dashboard", page_icon="📊", layout="wide")
st.title("📊 Parlay Finance: Real vs Synthetic Data Validation")

with silence_everything():
    old_df, new_df = load_local_datasets(
        given_csv_path=os.path.join("data", "given_data.xlsx"),
        synth_xlsx_path=os.path.join("data", "synthetic_data_generated.xlsx")
    )

# Overview
st.subheader("📄 Dataset Overview")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Original Data (sample)")
    st.dataframe(old_df.head())
    st.caption(f"Shape: {old_df.shape}")
with c2:
    st.markdown("### Synthetic Data (sample)")
    st.dataframe(new_df.head())
    st.caption(f"Shape: {new_df.shape}")

tabs = st.tabs([
    "Loan Type Comparison",
    "Feature Engineering",
    "ML Model Performance",
    "Live Predictions",
    "Rules Engine (No-ML)",
    "Portfolio Dashboard",
    "Applicant vs Cohort Comparison",
    "AI Assistant",
    "AI copilot(LLM)",
    "ML Model vs Rule Based"
])

with tabs[0]: tab1_render(old_df, new_df)
with tabs[1]: tab2_render()
with tabs[2]: tab3_render()
with tabs[3]: tab4_render()
with tabs[4]: tab5_render()
with tabs[5]: tab6_render(new_df)
with tabs[6]: tab7_render(new_df)
with tabs[7]: tab8_render()
with tabs[8]: tab9_render()
with tabs[9]: tab10_render()