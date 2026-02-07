# loan_app/src/tabs/tab3_model_reports.py
from __future__ import annotations
import streamlit as st
import pandas as pd

# Optional: read some metadata from the saved bundle
from src.model_loader import load_model_bundle


# ---- Static classification summaries (your holdout reports) ----
REPORT_7A = """\
=== 7(a) (HOLDOUT) ===
              precision    recall  f1-score   support

           0     0.9999    0.9898    0.9948     24494
           1     0.9788    0.9997    0.9892     11521

    accuracy                         0.9930     36015
   macro avg     0.9894    0.9948    0.9920     36015
weighted avg     0.9931    0.9930    0.9930     36015
"""

REPORT_504 = """\
=== 504 (HOLDOUT) ===
              precision    recall  f1-score   support

           0     1.0000    0.9964    0.9982     19400
           1     0.9958    1.0000    0.9979     16615

    accuracy                         0.9981     36015
   macro avg     0.9979    0.9982    0.9980     36015
weighted avg     0.9981    0.9981    0.9981     36015
"""

REPORT_EXPRESS = """\
=== Express (HOLDOUT) ===
              precision    recall  f1-score   support

           0     0.9999    0.9990    0.9994     24867
           1     0.9977    0.9998    0.9987     11148

    accuracy                         0.9992     36015
   macro avg     0.9988    0.9994    0.9991     36015
weighted avg     0.9992    0.9992    0.9992     36015
"""

REPORT_INELIGIBLE = """\
=== Ineligible (HOLDOUT) ===
              precision    recall  f1-score   support

           0     0.9977    0.9999    0.9988     28359
           1     0.9996    0.9915    0.9955      7656

    accuracy                         0.9981     36015
   macro avg     0.9987    0.9957    0.9972     36015
weighted avg     0.9981    0.9981    0.9981     36015
"""

HAMMING_LOSS = 0.0029015687907816188
MICRO_F1 = 0.9955606533698677


def render():
    st.header("🤖 Loan Eligibility Prediction — Multi-label LightGBM (CV)")

    st.markdown("""
This tab summarizes the performance of the **single ML model** currently deployed:

### 🧠 What the model does
- It is a **multi-label classifier** that predicts, for each applicant, which SBA programs they may be eligible for:
  **7(a)**, **504**, **Express**; and a derived **Ineligible** flag.
- Under the hood, we train a **separate LightGBM binary classifier for each label** using **cross-validation (CV)**.  
  Their probabilities are averaged across folds to improve generalization.
- We **tune thresholds per label** using **F-β** on out-of-fold predictions — e.g. a slightly higher β for **7(a)** to favor recall.
- Features come from the **new engineered pipeline** (`engineer()` + `add_cross_features()`), combining rules-critical signals
  (credit, DSCR, collateral, tenure), profitability/leverage dynamics, and **policy-aware crosses** (e.g., real estate blocks 7(a)).

This approach fits your requirement for **multi-program eligibility** (an applicant can be eligible for more than one product).
""")

    st.subheader("📈 Holdout Classification Reports (per label)")
    st.caption("Precision/Recall/F1 for each label as a binary task on the holdout split.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**7(a)**")
        st.code(REPORT_7A, language="text")
    with col2:
        st.markdown("**504**")
        st.code(REPORT_504, language="text")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Express**")
        st.code(REPORT_EXPRESS, language="text")
    with col4:
        st.markdown("**Ineligible**")
        st.code(REPORT_INELIGIBLE, language="text")

    st.subheader("📊 Overall Multi-label Metrics (Holdout)")
    st.metric("Hamming Loss", f"{HAMMING_LOSS:.4f}")
    st.metric("Example-based Micro F1", f"{MICRO_F1:.4f}")

    # --- Optional transparency from the active bundle ---
    with st.expander("🔎 Model bundle info (thresholds, labels, features)"):
        try:
            bundle = load_model_bundle()
            st.markdown("**Labels:**")
            st.write(bundle.get("labels", []))

            st.markdown("**Thresholds (per label):**")
            st.json(bundle.get("thresholds", {}))

            st.markdown("**Number of features:**")
            st.write(len(bundle.get("feature_cols", [])))

            st.caption("Note: Feature importances are shown in the Feature Engineering tab.")
        except Exception as e:
            st.warning(f"Could not inspect bundle: {e}")
