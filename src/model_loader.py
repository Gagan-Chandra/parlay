# src/model_loader.py
import os, joblib, streamlit as st

@st.cache_resource(show_spinner=False)
def load_model_bundle():
    """Loads the trained LightGBM multi-label eligibility model bundle."""
    path_candidates = [
        "src/models/loan_eligibility_bundle.pkl",
        "models/loan_eligibility_bundle.pkl",
        "loan_eligibility_bundle.pkl",
    ]
    for p in path_candidates:
        if os.path.exists(p):
            bundle = joblib.load(p)
            print(f"✅ Loaded model bundle from {p}")
            return bundle
    raise FileNotFoundError("❌ Could not find loan_eligibility_bundle.pkl in any known path.")
