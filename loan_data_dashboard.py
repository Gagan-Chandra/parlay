# ============================================================
# 💻 Parlay Finance: Real vs Synthetic Loan Data Comparison
# Author: Gagan
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
import os, ast
import os, ast
import warnings, sys, logging,contextlib
# ============================================================
# 🚫 COMPLETE SILENCER BLOCK — No warnings/logs shown anywhere
# ============================================================
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.CRITICAL)

# Redirect stdout/stderr to devnull to hide SHAP, LGBM, XGB logs
@contextlib.contextmanager
def suppress_all_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err



# ----------------------------
# 🎨 Streamlit Page Setup
# ----------------------------

st.set_page_config(
    page_title="Parlay Finance Data Validation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Parlay Finance: Real vs Synthetic Data Validation")
st.markdown("""
This dashboard compares the **original applicant dataset** provided by Parlay Finance 
with the **synthetic dataset** generated for model training.
""")

# ----------------------------
# 🗂️ Load Files
# ----------------------------
data_path = os.getcwd()
old_path = os.path.join(data_path, "data/given_data.csv")
new_path = os.path.join(data_path, "data/synthetic_data_generated.xlsx")

if not (os.path.exists(old_path) and os.path.exists(new_path)):
    st.error("❌ Missing `given_data.csv` or `synthetic_data_generated.csv` in this folder.")
    st.stop()

# --- Load Excel files ---
try:
    old_df = pd.read_csv(old_path)
except Exception as e:
    st.error(f"Error loading 'given_data.xlsx': {e}")
    st.stop()

try:
    new_df = pd.read_excel(new_path)
except Exception as e:
    st.error(f"Error loading 'synthetic_data_generated.xlsx': {e}")
    st.stop()

# ----------------------------
# 🧾 Display Overview
# ----------------------------
st.subheader("📄 Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Original Data (50 Applicants)")
    st.dataframe(old_df.head())
    st.caption(f"Shape: {old_df.shape}")
with col2:
    st.markdown("### Synthetic Data")
    st.dataframe(new_df.head())
    st.caption(f"Shape: {new_df.shape}")

# ----------------------------
# 🧩 Normalize Eligibility Column
# ----------------------------
def normalize_eligibility(df):
    if df["Eligibility"].dtype == "object":
        df["Eligibility"] = df["Eligibility"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [x]
        )
    return df

old_df = normalize_eligibility(old_df)
new_df = normalize_eligibility(new_df)

# Extract all unique loan types
all_loans = sorted(
    set([loan for sublist in old_df["Eligibility"].tolist() for loan in sublist] +
        [loan for sublist in new_df["Eligibility"].tolist() for loan in sublist])
)
loan_types = [l for l in all_loans if l in ["7(a)", "504", "Express"]]

# ----------------------------
# 📑 Tabs
# ----------------------------
tabs = st.tabs(["Loan Type Comparison", "Feature Engineering","ML Model Performance","Live Predictions","Rules Engine (No-ML)","Data Drift & Cohort Dashboard",
    "Applicant vs Cohort Comparison"])

# =============================
# 📊 TAB 1: Loan Comparison
# =============================
with tabs[0]:
    selected_loan = st.sidebar.selectbox("🔹 Select Loan Type", loan_types)

    # Filter data — include applicants eligible for multiple loans too
    old_sub = old_df[old_df["Eligibility"].apply(lambda x: selected_loan in x)]
    new_sub = new_df[new_df["Eligibility"].apply(lambda x: selected_loan in x)]

    st.markdown(f"## 🔹 Loan Type: **{selected_loan}** Comparison")
    st.caption("Includes applicants eligible for this loan even if they qualify for multiple.")

    numeric_cols = [
        "Personal Credit Score", "Business Credit Score", "DSCR (latest year)",
        "Annual Revenue (latest year)", "Years in Business", "Loan Amount",
        "Industry Experience", "Managerial Experience"
    ]

    categorical_cols = ["Collateral Availability", "Fast Approval"]

    similarities = []
    for col in numeric_cols:
        if col in old_df.columns and col in new_df.columns:
            try:
                stat, _ = ks_2samp(old_sub[col].dropna(), new_sub[col].dropna())
                similarities.append({"Feature": col, "KS Similarity": round(1 - stat, 3)})
            except Exception:
                pass

    sim_df = pd.DataFrame(similarities).sort_values("KS Similarity", ascending=False)

    cat_similarities = []
    for col in categorical_cols:
        if col in old_df.columns and col in new_df.columns:
            old_mode = old_sub[col].mode()[0] if not old_sub[col].empty else None
            new_mode = new_sub[col].mode()[0] if not new_sub[col].empty else None
            match_ratio = (old_mode == new_mode)
            cat_similarities.append({
                "Feature": col,
                "Match": "Yes" if match_ratio else "No",
                "Old Mode": old_mode,
                "New Mode": new_mode
            })

    cat_df = pd.DataFrame(cat_similarities)

    st.markdown("### 📈 Numeric Feature Similarity (KS Test)")
    st.dataframe(sim_df)

    fig = px.bar(sim_df, x="KS Similarity", y="Feature", orientation="h",
                 color="KS Similarity", color_continuous_scale="tealrose",
                 title=f"Numeric Feature Similarity for {selected_loan}")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔤 Categorical Feature Comparison")
    st.dataframe(cat_df)

    st.markdown("### 🔍 Visual Distribution Comparison (Top 3 Numeric Features)")
    top_features = sim_df.head(3)["Feature"].tolist()
    for col in top_features:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.kdeplot(old_sub[col], label="Original", fill=True, color="blue", alpha=0.4)
        sns.kdeplot(new_sub[col], label="Synthetic", fill=True, color="orange", alpha=0.4)
        ax.set_title(f"{col} Distribution Comparison")
        ax.legend()
        st.pyplot(fig)

    avg_sim = round(sim_df["KS Similarity"].mean(), 3)
    st.markdown(f"### ✅ Average Numeric Similarity: **{avg_sim}**")

    if avg_sim >= 0.8:
        st.success("Excellent — Synthetic data very closely matches the real dataset.")
    elif avg_sim >= 0.6:
        st.info("Good — Synthetic data maintains realistic trends with minor variation.")
    else:
        st.warning("Moderate — Some numeric features differ more significantly from the real data.")

    st.markdown("---")

# =============================
# 🧪 TAB 2: Feature Engineering
# =============================
with tabs[1]:
    st.header("🧩 Feature Engineering & Insights")

    st.markdown("""
    This section explains the **feature engineering process** used before training our ML models.
    I performed the following key steps:
    - **Low Variance Removal:** Removed features that had almost no variation across applicants.  
    - **Correlation Filtering:** Eliminated features with correlation > 0.9 to avoid redundancy.  
    - **Feature Importance Ranking:** Computed via Random Forest / XGBoost to identify strongest predictors.  
    """)

    st.markdown("""
- **Derived Ratios**: Created Debt-to-Income, Loan-to-Income, and NOI-to-Revenue ratios to represent financial health more meaningfully.
- **Temporal Trends**: Captured revenue and debt changes over the last two years to understand growth and volatility.
- **DSCR Aggregates**: Added average DSCR and its variability across three years for creditworthiness trends.
- **Experience Indicators**: Encoded industry and managerial experience, and created flags for new vs. experienced businesses.
- **Purpose Flags**: Mapped loan purposes like real estate intent, working capital, or equipment need as binary features.
- **Cleanup**: Removed features with low variance or high correlation to reduce noise and multicollinearity.
""")

    st.markdown("""
Feature engineering helps uncover hidden patterns that raw features alone can't reveal. In our case, we introduced financial ratios like **Debt-to-Income Ratio**, **Loan-to-Income Ratio**, and **Revenue Growth over 1–2 years** to better capture the applicant's financial health and stability. These derived features give the model a more nuanced view of business sustainability and creditworthiness—factors that directly influence loan eligibility.


The newly engineered features ranked among the **top contributors in feature importance**, as seen in our Random Forest analysis. Variables like `Loan_to_Income_Ratio`, `Has_RealEstateIntent`, `Has_Collateral`, and `Has_WorkingCapital` captured the **loan purpose and repayment capacity**, which are vital for matching applicants to the correct loan type. These variables not only enhanced prediction accuracy but also improved **model interpretability**—allowing us to justify why a specific loan was suggested or denied for an applicant.
""")

    feature_importance = pd.Series({
        "Loan Amount": 0.152209,
        "Business Credit Score":  0.096487,
        "Personal Credit Score": 0.091826,
        "Has_WorkingCapital": 0.088104,
        "Has_Collateral": 0.084317,
        "Has_RealEstateIntent": 0.073358,
        "Loan_to_Income_Ratio": 0.067773,
         "DSCR (latest year)": 0.054475,
        "Annual Revenue (2 years ago)": 0.023057,
        "Avg_DSCR":0.016425
    })

    st.subheader("🏅 Top 10 Important Features")
    fig = px.bar(
        feature_importance.sort_values(ascending=True),
        x=feature_importance.sort_values(ascending=True),
        y=feature_importance.sort_values(ascending=True).index,
        orientation="h",
        color=feature_importance.sort_values(ascending=True),
        color_continuous_scale="Bluered_r"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 🧠 Key Insights
    - **Loan Amount** and **Collateral Availability** are the strongest indicators of loan eligibility.  
    - **Purpose Intent Features** (e.g., Real Estate Intent, Working Capital Need) highly influence specific loan categories (7(a), 504, Express).  
    - **Credit Scores** and **DSCR** capture the applicant’s financial health, critical for 7(a) and Express loans.  
    - Removing redundant and low-variance features simplified the dataset while preserving predictive power.
    """)

    st.markdown("---")


# =============================
# 🤖 TAB 3: ML Model Results
# =============================
with tabs[2]:
    st.header("🤖 Loan Eligibility Prediction - ML Models")

    st.markdown("""
    I trained three models to classify applicants into **eligible loan categories** based on engineered features:
    
    - **XGBoost**: Gradient boosting decision trees optimized for speed and accuracy.
    - **LightGBM**: Fast, memory-efficient gradient boosting from Microsoft.
    - **Ensemble**: Combines XGBoost + LightGBM predictions using majority voting.
    """)

    st.subheader("📈 Classification Report (Best Fold)")
    st.markdown("### 🔹 XGBoost")
    st.code("""
['504', 'Express']:  Precision 1, Recall 1.00, F1 1.00
['504']:             Precision 0.98, Recall 0.94, F1 0.96
['7(a)', '504']:     Precision 0.88, Recall 1.00, F1 0.94
['7(a)']:            Precision 0.98, Recall 0.91, F1 0.94
['Express']:         Precision 0.99, Recall 1.00, F1 99
['Ineligible']:      Precision 1.00, Recall 0.97, F1 0.98
    """, language="text")

    st.markdown("### 🔹 LightGBM")
    st.code("""
['504', 'Express']:  Precision 1.00, Recall 1.00, F1 1.00
['504']:             Precision 0.97, Recall 0.98, F1 0.97
['7(a)', '504']:     Precision 0.96, Recall 1.00, F1 0.98
['7(a)']:            Precision 0.98, Recall 0.96, F1 0.97
['Express']:         Precision 0.99, Recall 1.00, F1 1.00
['Ineligible']:      Precision 1.00, Recall 0.97, F1 0.98
    """, language="text")

    st.markdown("### 🔹 Ensemble")
    st.code("""
['504', 'Express']:  Precision 1.00, Recall 1.00, F1 1.00
['504']:             Precision 0.98, Recall 0.97, F1 0.98
['7(a)', '504']:     Precision 0.96, Recall 1.00, F1 0.98
['7(a)']:            Precision 0.97, Recall 0.96, F1 0.97
['Express']:         Precision 0.99, Recall 1.00, F1 0.99
['Ineligible']:      Precision 1.00, Recall 0.97, F1 0.98
    """, language="text")

    st.subheader("📊 Confusion Matrices")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("assets/xgboost.png", caption="XGBoost Confusion Matrix", use_column_width=True)
    with col2:
        st.image("assets/lightgbm.png", caption="LightGBM Confusion Matrix", use_column_width=True)
    with col3:
        st.image("assets/Ensemble.png", caption="Ensemble Confusion Matrix", use_column_width=True)

    st.markdown("### 📌 Summary Insights")
    st.markdown("""
- **All models perform very well** on classes like `['504', 'Express']` and `['7(a)', '504']`, with F1 scores close to 0.99.
- The **'Ineligible'** class is slightly harder to predict (Recall ~82%) but maintains high precision.
- The **Ensemble model** performs best overall with a balance of high precision and recall across all classes.
- This high performance supports automated loan eligibility classification in real-time applications.
    """)

# =============================
# 🧠 TAB 4: Live Predictions & Explainability
# =============================
with tabs[3]:
    import joblib
    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    st.header("🧠 Live Predictions & Explainability")

    st.markdown("""
    Upload an Excel file with applicant details to generate **loan eligibility predictions**
    using all three trained models (**XGBoost**, **LightGBM**, and **Ensemble**).  
    You can also select an applicant to view **SHAP-based explainability** —  
    see *why* each model predicted that loan type.
    """)

    uploaded_file = st.file_uploader("📤 Upload Applicant Excel File", type=["xlsx"])

    if uploaded_file:
        try:
            df_input = pd.read_excel(uploaded_file)
            st.success(f"✅ Loaded file with {df_input.shape[0]} applicants.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        # --- Feature Engineering ---
        def feature_engineering(df):
            df = df.copy()
            def safe_div(a, b):
                return np.where(b == 0, 0, a / b)

            # Ratios
            df["Debt_to_Income_Ratio"] = safe_div(df["Business Debt (latest year)"], df["Annual Revenue (latest year)"])
            df["Loan_to_Income_Ratio"] = safe_div(df["Loan Amount"], df["Annual Revenue (latest year)"])
            df["NOI_to_Revenue_Ratio"] = safe_div(df["NOI (latest year)"], df["Annual Revenue (latest year)"])

            # Temporal Trends
            df["Revenue_Growth_1y"] = safe_div(
                df["Annual Revenue (latest year)"] - df["Annual Revenue (1 year ago)"],
                df["Annual Revenue (1 year ago)"]
            )
            df["Revenue_Growth_2y"] = safe_div(
                df["Annual Revenue (1 year ago)"] - df["Annual Revenue (2 years ago)"],
                df["Annual Revenue (2 years ago)"]
            )

            # DSCR & Experience
            df["Avg_DSCR"] = df[["DSCR (latest year)", "DSCR (1 year ago)", "DSCR (2 years ago)"]].mean(axis=1)
            df["Experience_Index"] = (df["Industry Experience"] + df["Managerial Experience"]) / 2

            # Purpose Flags
            for col in ["Collateral Availability", "Working Capital", "Business Expansion",
                        "Equipment Purchase or Leasing", "Real Estate Acquisition or Improvement"]:
                if col not in df.columns:
                    df[col] = 0
            df["Has_Collateral"] = df["Collateral Availability"].astype(int)
            df["Has_WorkingCapital"] = df["Working Capital"].astype(int)
            df["Has_ExpansionIntent"] = df["Business Expansion"].astype(int)
            df["Has_EquipmentNeed"] = df["Equipment Purchase or Leasing"].astype(int)
            df["Has_RealEstateIntent"] = df["Real Estate Acquisition or Improvement"].astype(int)

            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)
            return df

        df_proc = feature_engineering(df_input)

        # --- Load Models ---
        xgb_model = joblib.load("models/xgb_loan_model_kfold.pkl")
        xgb_ense = joblib.load("models/xgb_base.pkl")
        lgb_model = joblib.load("models/lightgbm_loan_model.pkl")
        lgb_ense = joblib.load("models/lgb_base.pkl")
        ens_model = joblib.load("models/ensemble_meta.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")

        # --- Encode Object Columns ---
        for col in df_proc.select_dtypes(include=["object"]).columns:
            df_proc[col] = df_proc[col].astype("category").cat.codes

        # --- Align features for base models ---
        expected_features = xgb_model.get_booster().feature_names
        for f in expected_features:
            if f not in df_proc.columns:
                df_proc[f] = 0
        df_proc = df_proc[expected_features]

        # --- Predictions ---
        xgb_probs = xgb_model.predict_proba(df_proc)
        lgb_probs = lgb_model.predict_proba(df_proc)
        meta_input = np.hstack((xgb_probs, lgb_probs))
        ens_probs = ens_model.predict_proba(meta_input)

        xgb_pred = label_encoder.inverse_transform(np.argmax(xgb_probs, axis=1))
        lgb_pred = label_encoder.inverse_transform(np.argmax(lgb_probs, axis=1))
        ens_pred = label_encoder.inverse_transform(np.argmax(ens_probs, axis=1))
        ens_conf = ens_probs.max(axis=1)

        results_df = pd.DataFrame({
            "Applicant ID": df_input["Applicant ID"],
            "XGBoost Prediction": xgb_pred,
            "LightGBM Prediction": lgb_pred,
            "Ensemble Prediction": ens_pred,
            "Confidence (Ensemble)": np.round(ens_conf, 3)
        })

        st.subheader("📊 Model Predictions Overview")
        st.dataframe(results_df, use_container_width=True)
        # cache for Tab 6
        st.session_state["pred_results_df"] = results_df.copy()
        st.session_state["pred_features_df"] = df_proc.copy()
        st.session_state["xgb_probs"] = xgb_probs
        st.session_state["lgb_probs"] = lgb_probs
        st.session_state["ens_probs"] = ens_probs
        st.session_state["label_encoder"] = label_encoder

        st.markdown("---")
        st.subheader("🔍 Select an Applicant for Explainability")

        selected_id = st.selectbox("Choose Applicant ID", results_df["Applicant ID"].tolist())
        model_choice = st.radio("Select Model for Explanation", ["XGBoost", "LightGBM", "Ensemble"], horizontal=True)

        if selected_id:
            idx = results_df.index[results_df["Applicant ID"] == selected_id][0]
            st.markdown(f"### 🧩 SHAP Explainability — Applicant ID: **{selected_id}** ({model_choice})")

            sample = df_proc.iloc[[idx]]

            if model_choice == "XGBoost":
                model = xgb_model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(sample)
                pred_probs = model.predict_proba(sample)
                pred_idx = np.argmax(pred_probs)
                shap_exp = shap.Explanation(
                    values=shap_values.values[0, :, pred_idx],
                    base_values=explainer.expected_value[pred_idx],
                    data=sample.values[0],
                    feature_names=sample.columns
                )

            elif model_choice == "LightGBM":
                model = lgb_model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(sample)
                pred_probs = model.predict_proba(sample)
                pred_idx = np.argmax(pred_probs)
                shap_exp = shap.Explanation(
                    values=shap_values.values[0, :, pred_idx],
                    base_values=explainer.expected_value[pred_idx],
                    data=sample.values[0],
                    feature_names=sample.columns
                )

            else:  # Ensemble
                xgb_explainer = shap.TreeExplainer(xgb_ense)
                lgb_explainer = shap.TreeExplainer(lgb_ense)
                xgb_shap = xgb_explainer(sample)
                lgb_shap = lgb_explainer(sample)
                pred_probs = ens_probs[[idx]]
                pred_idx = np.argmax(pred_probs)

                # Combine SHAP values for predicted class only
                xgb_class_vals = xgb_shap.values[0, :, pred_idx]
                lgb_class_vals = lgb_shap.values[0, :, pred_idx]
                combined_values = (xgb_class_vals + lgb_class_vals) / 2

                combined_base = np.mean([
                    np.mean(xgb_explainer.expected_value[pred_idx]),
                    np.mean(lgb_explainer.expected_value[pred_idx])
                ])

                shap_exp = shap.Explanation(
                    values=combined_values,
                    base_values=combined_base,
                    data=sample.values[0],
                    feature_names=sample.columns
                )

            # --- Display Predictions ---
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            st.write(f"**Predicted Loan Type:** {pred_label}")
            st.write("**Class Probabilities:**")
            st.json({cls: float(p) for cls, p in zip(label_encoder.classes_, pred_probs[0])})

            # --- Local Waterfall Plot ---
            st.markdown("#### 💧 Waterfall Plot — Feature Contributions")
            try:
                shap.plots.waterfall(shap_exp, show=False)
                st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.warning(f"Could not display waterfall plot: {e}")

            # --- Global Feature Impact (Bar Chart) ---
            st.markdown("#### 📊 Top Feature Impact (Global Importance)")
            try:
                feature_impact = pd.Series(np.abs(shap_exp.values), index=sample.columns)
                top_features = feature_impact.sort_values(ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(7, 4))
                top_features[::-1].plot(kind="barh", color="teal", ax=ax)
                ax.set_title("Top 10 Contributing Features")
                ax.set_xlabel("SHAP Value Magnitude (Impact Strength)")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display feature impact chart: {e}")

            st.markdown("✅ **Explainability complete.** You can switch models above to compare reasoning differences.")


# ----------------------------
# 🧮 RULES ENGINE (No-ML)
# ----------------------------
def check_loan_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the loan eligibility for each applicant based on specific criteria.
    Adds a new column 'Eligibility' (list) or ['Ineligible'] if none match.
    """

    # Ensure all required columns exist (fill missing with safe defaults)
    required_bool = [
        "For Profit", "Fast Approval", "Collateral Availability",
        "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
        "Inventory Purchase", "Real Estate Acquisition or Improvement",
        "Business Acquisition or Buyout", "Refinancing Existing Debt",
        "Emergency Funds", "Franchise Financing", "Contract Financing",
        "Licensing or Permits", "Line of Credit Establishment"
    ]
    required_num = [
        "Personal Credit Score", "Business Credit Score",
        "DSCR (latest year)", "Annual Revenue (latest year)", "Loan Amount",
        "Years in Business", "Net Profit Margin",
        "NOI (1 year ago)", "NOI (2 years ago)",
        "Industry Experience", "Managerial Experience"
    ]

    for c in required_bool:
        if c not in df.columns:
            df[c] = False
    for c in required_num:
        if c not in df.columns:
            df[c] = 0

    # normalize booleans and fill NaNs
    df[required_bool] = df[required_bool].fillna(False).astype(bool)
    df[required_num] = df[required_num].fillna(0)

    def determine_eligibility(row):
        eligible_loans = []

        personal_credit_score = row["Personal Credit Score"]
        business_credit_score = row["Business Credit Score"]
        dscr_latest = row["DSCR (latest year)"]
        annual_revenue_latest = row["Annual Revenue (latest year)"]
        years_in_business = row["Years in Business"]
        collateral_available = row["Collateral Availability"]
        loan_amount = row["Loan Amount"]
        fast_approval = row["Fast Approval"]
        net_profit_margin = row["Net Profit Margin"]
        average_noi_last_2_years = (row["NOI (2 years ago)"] + row["NOI (1 year ago)"]) / 2.0
        industry_exp = row["Industry Experience"]
        mng_exp = row["Managerial Experience"]

        purposes = [
            "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
            "Inventory Purchase", "Real Estate Acquisition or Improvement",
            "Business Acquisition or Buyout", "Refinancing Existing Debt",
            "Emergency Funds", "Franchise Financing", "Contract Financing",
            "Licensing or Permits", "Line of Credit Establishment"
        ]
        valid_purpose = any(bool(row.get(p, False)) for p in purposes)

        # SBA 7(a)
        if (
            bool(row["For Profit"])
            and personal_credit_score >= 680
            and business_credit_score >= 160
            and dscr_latest >= 1.15
            and years_in_business >= 2
            and 500001 <= loan_amount <= 5000000
            and valid_purpose
            and not bool(row["Real Estate Acquisition or Improvement"])
            and not bool(row["Emergency Funds"])
        ):
            eligible_loans.append("7(a)")

        # SBA 8(a)  (kept for completeness but you can ignore downstream if not needed)
        if (
            bool(row["For Profit"]) is False
            and bool(fast_approval) is False
            and years_in_business >= 2
            and industry_exp >= 2
            and mng_exp >= 2
            and valid_purpose
            and not bool(row["Franchise Financing"])
            and not bool(row["Line of Credit Establishment"])
        ):
            eligible_loans.append("8(a)")

        # SBA 504
        if (
            bool(row["For Profit"])
            and personal_credit_score >= 680
            and dscr_latest >= 1.15
            and average_noi_last_2_years < 6500000
            and net_profit_margin > 0
            and years_in_business >= 2
            and loan_amount <= 5500000
            and bool(collateral_available)
            and not bool(row["Working Capital"])
            and not bool(row["Refinancing Existing Debt"])
            and not bool(row["Emergency Funds"])
        ):
            eligible_loans.append("504")

        # SBA Express
        if (
            bool(row["For Profit"])
            and bool(fast_approval)
            and personal_credit_score >= 680
            and business_credit_score >= 160
            and dscr_latest >= 1.15
            and loan_amount <= 500000
            and valid_purpose
            and not bool(row["Real Estate Acquisition or Improvement"])
            and not bool(row["Business Acquisition or Buyout"])
        ):
            eligible_loans.append("Express")

        return eligible_loans if len(eligible_loans) else ["Ineligible"]

    df = df.copy()
    df["Eligibility"] = df.apply(determine_eligibility, axis=1)
    return df


# ----------------------------
# 🔧 Rules defaults & evaluator
# ----------------------------
def default_rules():
    return {
        "enable_8a": False,                      # toggle 8(a)
        # Common thresholds
        "min_personal_credit": 680,
        "min_business_credit": 160,
        "min_dscr": 1.15,
        "min_years_in_business": 2,

        # 7(a)
        "7a_requires_for_profit": True,
        "7a_loan_min": 500001,
        "7a_loan_max": 5_000_000,
        "7a_exclude_purposes": ["Real Estate Acquisition or Improvement", "Emergency Funds"],

        # 8(a)
        "8a_requires_for_profit": False,
        "8a_requires_fast_approval": False,
        "8a_min_industry_exp": 2,
        "8a_min_managerial_exp": 2,
        "8a_exclude_purposes": ["Franchise Financing", "Line of Credit Establishment"],

        # 504
        "504_requires_for_profit": True,
        "504_requires_collateral": True,
        "504_max_loan": 5_500_000,
        "504_max_avg_noi_2y": 6_500_000,
        "504_min_net_profit_margin": 0.0,
        "504_must_include_one_of": ["Real Estate Acquisition or Improvement", "Equipment Purchase or Leasing"],
        "504_exclude_purposes": ["Working Capital", "Refinancing Existing Debt", "Emergency Funds"],

        # Express
        "express_requires_for_profit": True,
        "express_requires_fast_approval": True,
        "express_max_loan": 500_000,
        "express_exclude_purposes": ["Real Estate Acquisition or Improvement", "Business Acquisition or Buyout"],

        # Purpose universe (can be expanded)
        "all_purposes": [
            "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
            "Inventory Purchase", "Real Estate Acquisition or Improvement",
            "Business Acquisition or Buyout", "Refinancing Existing Debt",
            "Emergency Funds", "Franchise Financing", "Contract Financing",
            "Licensing or Permits", "Line of Credit Establishment"
        ]
    }

def check_loan_eligibility_configurable(df, rules):
    def has_any(row, names):
        return any(bool(row.get(p, False)) for p in names)

    def has_none(row, names):
        return not any(bool(row.get(p, False)) for p in names)

    def determine_eligibility(row):
        eligible = []

        # Extract base fields (safe gets)
        pcs = float(row.get("Personal Credit Score", 0) or 0)
        bcs = float(row.get("Business Credit Score", 0) or 0)
        dscr = float(row.get("DSCR (latest year)", 0) or 0)
        yib = float(row.get("Years in Business", 0) or 0)
        loan_amt = float(row.get("Loan Amount", 0) or 0)
        for_profit = bool(row.get("For Profit", False))
        fast_approval = bool(row.get("Fast Approval", False))
        net_margin = float(row.get("Net Profit Margin", 0) or 0)
        noi_1 = float(row.get("NOI (1 year ago)", 0) or 0)
        noi_2 = float(row.get("NOI (2 years ago)", 0) or 0)
        avg_noi_2y = (noi_1 + noi_2) / 2.0
        collateral = bool(row.get("Collateral Availability", False))

        # Any valid purpose?
        purposes = rules["all_purposes"]
        valid_purpose = has_any(row, purposes)

        # ---------- 7(a) ----------
        if (
            (not rules["7a_requires_for_profit"] or for_profit) and
            pcs >= rules["min_personal_credit"] and
            bcs >= rules["min_business_credit"] and
            dscr >= rules["min_dscr"] and
            yib  >= rules["min_years_in_business"] and
            (rules["7a_loan_min"] <= loan_amt <= rules["7a_loan_max"]) and
            valid_purpose and
            has_none(row, rules["7a_exclude_purposes"])
        ):
            eligible.append("7(a)")

        # ---------- 8(a) (optional) ----------
        if rules.get("enable_8a", False):
            if (
                (not rules["8a_requires_for_profit"] or (not for_profit)) and
                (not rules["8a_requires_fast_approval"] or (not fast_approval)) and
                yib >= rules["min_years_in_business"] and
                float(row.get("Industry Experience", 0) or 0) >= rules["8a_min_industry_exp"] and
                float(row.get("Managerial Experience", 0) or 0) >= rules["8a_min_managerial_exp"] and
                valid_purpose and
                has_none(row, rules["8a_exclude_purposes"])
            ):
                eligible.append("8(a)")

        # ---------- 504 ----------
        if (
            (not rules["504_requires_for_profit"] or for_profit) and
            pcs >= rules["min_personal_credit"] and
            dscr >= rules["min_dscr"] and
            avg_noi_2y < rules["504_max_avg_noi_2y"] and
            net_margin > rules["504_min_net_profit_margin"] and
            yib >= rules["min_years_in_business"] and
            loan_amt <= rules["504_max_loan"] and
            (not rules["504_requires_collateral"] or collateral) and
            has_any(row, rules["504_must_include_one_of"]) and
            has_none(row, rules["504_exclude_purposes"])
        ):
            eligible.append("504")

        # ---------- Express ----------
        if (
            (not rules["express_requires_for_profit"] or for_profit) and
            (not rules["express_requires_fast_approval"] or fast_approval) and
            pcs >= rules["min_personal_credit"] and
            bcs >= rules["min_business_credit"] and
            dscr >= rules["min_dscr"] and
            loan_amt <= rules["express_max_loan"] and
            valid_purpose and
            has_none(row, rules["express_exclude_purposes"])
        ):
            eligible.append("Express")

        return eligible if eligible else ["Ineligible"]

    df = df.copy()
    df["Eligibility"] = df.apply(determine_eligibility, axis=1)
    return df




# ----------------------------
# 🧾 Applicant Narrative / Fit Summary (heuristic, engineered-aware)
# ----------------------------
def generate_applicant_summary(inputs: dict, loans: list) -> dict:
    """
    AI-style narrative using explicit rules.
    Uses base columns AND engineered features if present.
    If engineered features are missing, computes lightweight versions inline.
    """

    def fget(key, default=0.0):
        v = inputs.get(key, default)
        try:
            # keep booleans as-is, cast everything else to float if possible
            if isinstance(v, bool): 
                return v
            return float(v)
        except Exception:
            return default

    def bget(key):
        v = inputs.get(key, False)
        if isinstance(v, (bool, int, float)): 
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "y", "t")
        return False

    # ---------- Raw inputs
    pcs = fget("Personal Credit Score")
    bcs = fget("Business Credit Score")
    dscr = fget("DSCR (latest year)")
    dscr_1y = fget("DSCR (1 year ago)")
    dscr_2y = fget("DSCR (2 years ago)")
    yib = fget("Years in Business")
    npm = fget("Net Profit Margin")           # (%)
    noi_1 = fget("NOI (1 year ago)")
    noi_2 = fget("NOI (2 years ago)")
    ind_exp = fget("Industry Experience")
    mng_exp = fget("Managerial Experience")
    ann_rev = fget("Annual Revenue (latest year)")
    ann_rev_1 = fget("Annual Revenue (1 year ago)")
    ann_rev_2 = fget("Annual Revenue (2 years ago)")
    debt = fget("Business Debt (latest year)")
    debt_1 = fget("Business Debt (1 year ago)")
    debt_2 = fget("Business Debt (2 years ago)")
    loan_amt = fget("Loan Amount")

    has_collateral = bget("Collateral Availability")
    fast_approval = bget("Fast Approval")
    intent_re = bget("Real Estate Acquisition or Improvement")
    intent_equip = bget("Equipment Purchase or Leasing")
    intent_wc = bget("Working Capital")
    intent_acq = bget("Business Acquisition or Buyout")
    intent_refi = bget("Refinancing Existing Debt")

    # ---------- Engineered preferred (use if present), else derive
    def maybe(key, fallback):
        return inputs.get(key, fallback)

    # Ratios (safe)
    loan_to_income = maybe(
        "Loan_to_Income_Ratio",
        (loan_amt / ann_rev) if ann_rev > 0 else 0.0
    )
    debt_to_income = maybe(
        "Debt_to_Income_Ratio",
        (debt / ann_rev) if ann_rev > 0 else 0.0
    )

    # Growth / changes
    rev_g1 = maybe(
        "Revenue_Growth_1y",
        ((ann_rev - ann_rev_1) / ann_rev_1) if ann_rev_1 > 0 else 0.0
    )
    rev_g2 = maybe(
        "Revenue_Growth_2y",
        ((ann_rev_1 - ann_rev_2) / ann_rev_2) if ann_rev_2 > 0 else 0.0
    )
    debt_chg_1y = maybe(
        "Debt_Change_1y",
        ((debt - debt_1) / debt_1) if debt_1 > 0 else 0.0
    )
    noi_chg_1y = maybe(
        "NOI_Change_1y",
        ((fget("NOI (latest year)") - noi_1) / noi_1) if noi_1 > 0 else 0.0
    )

    # DSCR aggregates
    avg_dscr = maybe(
        "Avg_DSCR",
        np.nanmean([dscr, dscr_1y, dscr_2y]) if any([dscr, dscr_1y, dscr_2y]) else dscr
    )
    dscr_var = maybe(
        "DSCR_Variability",
        np.nanstd([dscr, dscr_1y, dscr_2y]) if any([dscr, dscr_1y, dscr_2y]) else 0.0
    )

    # Experience index
    exp_index = maybe(
        "Experience_Index",
        (ind_exp + mng_exp) / 2.0
    )

    # Binary engineered flags (fallback to intents / booleans)
    has_collateral_flag = bool(maybe("Has_Collateral", has_collateral))
    has_wc_flag = bool(maybe("Has_WorkingCapital", intent_wc))
    has_equip_flag = bool(maybe("Has_EquipmentNeed", intent_equip))
    has_re_flag = bool(maybe("Has_RealEstateIntent", intent_re))

    # ---------- Scoring
    score = 0
    strengths, risks = [], []

    # Credit
    if pcs >= 720: score += 9; strengths.append("Strong personal credit (≥ 720).")
    elif pcs >= 680: score += 5
    else: risks.append("Personal credit below 680 baseline.")

    if bcs >= 180: score += 7; strengths.append("Healthy business credit (≥ 180).")
    elif bcs >= 160: score += 4
    else: risks.append("Business credit below 160 baseline.")

    # DSCR & stability
    if dscr >= 1.50: score += 10; strengths.append("Robust DSCR (≥ 1.50).")
    elif dscr >= 1.25: score += 7
    elif dscr >= 1.15: score += 4
    else: score -= 10; risks.append("DSCR below 1.15 minimum.")

    if avg_dscr >= 1.35: score += 4; strengths.append("Multi-year DSCR average is healthy.")
    if dscr_var > 0.25: risks.append("High DSCR variability across years (volatility)."); score -= 3

    # Profitability & NOI momentum
    if npm > 10: score += 6; strengths.append("Solid profitability (NPM > 10%).")
    elif npm >= 1: score += 3
    else: risks.append("Non-profitable or negative margins."); score -= 5

    if noi_chg_1y > 0.05: strengths.append("NOI trending up year-over-year."); score += 3
    elif noi_chg_1y < -0.05: risks.append("NOI declined year-over-year."); score -= 2

    # Revenue growth & trend
    if rev_g1 > 0.10: strengths.append("Revenue grew > 10% last year."); score += 4
    elif rev_g1 < -0.10: risks.append("Revenue declined > 10% last year."); score -= 4

    if rev_g2 > 0.10: score += 2
    elif rev_g2 < -0.10: score -= 2

    # Leverage & debt load
    if loan_to_income <= 0.30: strengths.append("Conservative leverage (Loan/Revenue ≤ 30%)."); score += 8
    elif loan_to_income <= 0.50: score += 5
    elif loan_to_income <= 0.80: score += 0
    else: risks.append("High leverage (Loan/Revenue > 80%)."); score -= 8

    if debt_to_income <= 0.35: score += 4
    elif debt_to_income >= 0.80: risks.append("High debt load vs revenue (Debt/Revenue ≥ 80%)."); score -= 4

    if debt_chg_1y > 0.20: risks.append("Debt increased > 20% YoY."); score -= 2

    # Tenure & experience
    if yib >= 5: strengths.append("Established operating history (≥ 5 years)."); score += 6
    elif yib >= 2: score += 3
    else: risks.append("Limited operating history (< 2 years)."); score -= 5

    if exp_index >= 8: strengths.append("Strong leadership/industry experience."); score += 4
    elif exp_index >= 3: score += 2

    # Collateral & purpose alignment
    if has_collateral_flag: strengths.append("Collateral available (improves 504/7(a) profile)."); score += 6
    else:
        if has_re_flag or has_equip_flag:
            risks.append("No collateral signaled for asset-heavy use (504 typically requires it).")

    if "Express" in loans and fast_approval: score += 2
    if "504" in loans and (has_re_flag or has_equip_flag): score += 3
    if "7(a)" in loans and (has_wc_flag or has_equip_flag or intent_acq or intent_refi): score += 2

    # Program caps sanity notes
    if "Express" in loans and loan_amt > 500_000:
        risks.append("Requested amount exceeds typical SBA Express cap (>$500K).")
    if "7(a)" in loans and (loan_amt < 500_001 or loan_amt > 5_000_000):
        risks.append("7(a) request is outside 500,001–5,000,000 window used in rules.")
    if "504" in loans and loan_amt > 5_500_000:
        risks.append("504 request exceeds common max threshold (~$5.5M).")

    # If ineligible, penalize
    if not loans or loans == ["Ineligible"]:
        score = max(0, score - 20)

    # Clip and label
    score = int(min(100, max(0, score)))
    if not loans or loans == ["Ineligible"]:
        label = "Ineligible"
    elif score >= 80:
        label = "Strong Fit"
    elif score >= 60:
        label = "Good Fit"
    elif score >= 40:
        label = "Borderline"
    else:
        label = "Needs Review"

    # Tailored recommendation
    if loans and loans != ["Ineligible"]:
        details = []
        if "504" in loans:
            details.append("verify collateral and asset purpose docs for 504")
        if "7(a)" in loans:
            details.append("confirm DSCR calculations and working capital/equipment use for 7(a)")
        if "Express" in loans:
            details.append("ensure request ≤ $500K for Express")
        if not details:
            details.append("validate financials and purpose documentation")
        rec = "Proceed with underwriting; " + "; ".join(details) + "."
    else:
        rec = ("Not eligible under current inputs. Improve DSCR/credit, reduce request, "
               "or align purpose/collateral with target program.")

    loans_txt = ", ".join(loans) if loans else "None"
    narrative = (
        f"Overall **{label}** (score **{score}/100**). "
        f"Eligible programs: **{loans_txt}**. "
        f"Assessment considers credit quality, DSCR level/stability, profitability, growth momentum, "
        f"leverage and debt load, operating history, collateral, and purpose alignment."
    )

    return {
        "score": score,
        "label": label,
        "strengths": strengths,
        "risks": risks,
        "recommendation": rec,
        "narrative": narrative
    }



with tabs[4]:
    st.header("✅ Rules Engine (No-ML) — Upload, Review & Decide")

    st.markdown("""
Use the official **policy-based eligibility** (no ML) to see which SBA products an applicant qualifies for.  
1) Configure the **rules** if needed.  
2) Upload your file (`.xlsx` or `.csv`).  
3) Pick an applicant and review the **auto-filled form**.  
4) Submit to see the **eligible loans** instantly.
    """)

    # ---------- Rules Config UI ----------
    st.subheader("⚙️ Configure Rules")
    with st.expander("Show / Edit Rules", expanded=False):
        rules_state = st.session_state.get("eligibility_rules", default_rules())

        # Global thresholds
        col1, col2, col3, col4 = st.columns(4)
        rules_state["min_personal_credit"]   = col1.number_input("Min Personal Credit", 300, 850, int(rules_state["min_personal_credit"]))
        rules_state["min_business_credit"]   = col2.number_input("Min Business Credit", 0, 300, int(rules_state["min_business_credit"]))
        rules_state["min_dscr"]              = col3.number_input("Min DSCR", 0.0, 5.0, float(rules_state["min_dscr"]), step=0.01, format="%.2f")
        rules_state["min_years_in_business"] = col4.number_input("Min Years in Business", 0, 50, int(rules_state["min_years_in_business"]))

        # 7(a)
        st.markdown("**7(a) Parameters**")
        col1, col2, col3 = st.columns(3)
        rules_state["7a_requires_for_profit"] = col1.checkbox("7(a) requires For-Profit", value=rules_state["7a_requires_for_profit"])
        rules_state["7a_loan_min"] = col2.number_input("7(a) Min Loan", 0, 10_000_000, int(rules_state["7a_loan_min"]), step=1000)
        rules_state["7a_loan_max"] = col3.number_input("7(a) Max Loan", 0, 10_000_000, int(rules_state["7a_loan_max"]), step=1000)
        rules_state["7a_exclude_purposes"] = st.multiselect(
            "7(a) Excluded Purposes",
            options=rules_state["all_purposes"],
            default=rules_state["7a_exclude_purposes"]
        )

        # 8(a)
        st.markdown("**8(a) Parameters**")
        col1, col2, col3, col4 = st.columns(4)
        rules_state["enable_8a"] = col1.checkbox("Enable 8(a) rule", value=rules_state["enable_8a"])
        rules_state["8a_requires_fast_approval"] = col2.checkbox("8(a) requires NOT fast approval", value=rules_state["8a_requires_fast_approval"])
        rules_state["8a_min_industry_exp"] = col3.number_input("8(a) Min Industry Exp (yrs)", 0, 50, int(rules_state["8a_min_industry_exp"]))
        rules_state["8a_min_managerial_exp"] = col4.number_input("8(a) Min Managerial Exp (yrs)", 0, 50, int(rules_state["8a_min_managerial_exp"]))
        rules_state["8a_exclude_purposes"] = st.multiselect(
            "8(a) Excluded Purposes",
            options=rules_state["all_purposes"],
            default=rules_state["8a_exclude_purposes"]
        )

        # 504
        st.markdown("**504 Parameters**")
        col1, col2, col3, col4 = st.columns(4)
        rules_state["504_requires_for_profit"] = col1.checkbox("504 requires For-Profit", value=rules_state["504_requires_for_profit"])
        rules_state["504_requires_collateral"] = col2.checkbox("504 requires Collateral", value=rules_state["504_requires_collateral"])
        rules_state["504_max_loan"] = col3.number_input("504 Max Loan", 0, 10_000_000, int(rules_state["504_max_loan"]), step=1000)
        rules_state["504_max_avg_noi_2y"] = col4.number_input("504 Max Avg NOI (2y)", 0, 50_000_000, int(rules_state["504_max_avg_noi_2y"]), step=1000)
        col5, col6 = st.columns(2)
        rules_state["504_min_net_profit_margin"] = col5.number_input("504 Min Net Profit Margin", -100.0, 100.0, float(rules_state["504_min_net_profit_margin"]), step=0.1, format="%.1f")
        rules_state["504_must_include_one_of"] = st.multiselect(
            "504 Must Include One Of",
            options=rules_state["all_purposes"],
            default=rules_state["504_must_include_one_of"]
        )
        rules_state["504_exclude_purposes"] = st.multiselect(
            "504 Excluded Purposes",
            options=rules_state["all_purposes"],
            default=rules_state["504_exclude_purposes"]
        )

        # Express
        st.markdown("**Express Parameters**")
        col1, col2, col3 = st.columns(3)
        rules_state["express_requires_for_profit"] = col1.checkbox("Express requires For-Profit", value=rules_state["express_requires_for_profit"])
        rules_state["express_requires_fast_approval"] = col2.checkbox("Express requires Fast Approval", value=rules_state["express_requires_fast_approval"])
        rules_state["express_max_loan"] = col3.number_input("Express Max Loan", 0, 5_000_000, int(rules_state["express_max_loan"]), step=1000)
        rules_state["express_exclude_purposes"] = st.multiselect(
            "Express Excluded Purposes",
            options=rules_state["all_purposes"],
            default=rules_state["express_exclude_purposes"]
        )

        st.caption("These defaults mirror your current policy; adjust as required.")
        st.session_state["eligibility_rules"] = rules_state

    # ---------- Upload ----------
    up = st.file_uploader("📤 Upload Applicants File", type=["xlsx", "csv"])

    # Helpers
    def to_bool_series(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])

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

    if up:
        # Load
        try:
            if up.name.lower().endswith(".xlsx"):
                rules_df = pd.read_excel(up)
            else:
                rules_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        if "Applicant ID" not in rules_df.columns:
            rules_df["Applicant ID"] = range(1, len(rules_df) + 1)

        # Ensure fields exist
        for c, *_ in numeric_fields:
            if c not in rules_df.columns:
                rules_df[c] = 0
        for c in boolean_fields:
            if c not in rules_df.columns:
                rules_df[c] = False
            rules_df[c] = to_bool_series(rules_df[c])

        # Preview with configurable rules
        rules = st.session_state["eligibility_rules"]
        try:
            preview_df = check_loan_eligibility_configurable(rules_df.copy(), rules)
        except Exception as e:
            st.error(f"Eligibility rule error: {e}")
            st.stop()

        show_cols = [c for c in ["Applicant ID", "Business Name", "Eligibility"] if c in preview_df.columns]
        st.subheader("📄 Preview (Rule Output)")
        st.dataframe(preview_df[show_cols] if show_cols else preview_df[["Applicant ID", "Eligibility"]])

        # Applicant picker
        st.subheader("👤 Pick an Applicant")
        picked_id = st.selectbox("Applicant ID", options=preview_df["Applicant ID"].tolist(), index=0)
        sel_idx = preview_df.index[preview_df["Applicant ID"] == picked_id][0]
        row = rules_df.loc[sel_idx]

        # Form UI (auto-filled)
        st.subheader("📝 Review / Edit Applicant Inputs")
        with st.form("rules_form"):
            colA, colB = st.columns(2)
            num_values, bool_values = {}, {}

            for i, (fname, vmin, vmax, step) in enumerate(numeric_fields):
                default_val = float(row.get(fname, 0) or 0)
                container = colA if i % 2 == 0 else colB
                num_values[fname] = container.number_input(
                    fname, min_value=float(vmin), max_value=float(vmax),
                    value=float(default_val), step=float(step),
                    format="%.4f" if step < 1 else "%.0f"
                )

            st.markdown("**Flags & Purposes**")
            bcol1, bcol2 = st.columns(2)
            for j, bname in enumerate(boolean_fields):
                default_bool = bool(row.get(bname, False))
                container = bcol1 if j % 2 == 0 else bcol2
                bool_values[bname] = container.checkbox(bname, value=default_bool)

            submitted = st.form_submit_button("Run Rules")

        if submitted:
            single = {"Applicant ID": picked_id}
            for k, v in num_values.items(): single[k] = v
            for k, v in bool_values.items(): single[k] = bool(v)

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
                    " ".join([f"<span style='background:#e7f5ff;padding:4px 8px;border-radius:8px;margin-right:6px'>{l}</span>"
                              for l in loans]),
                    unsafe_allow_html=True
                )
            else:
                st.warning("Ineligible based on current inputs.")

            with st.expander("🔎 Show evaluated inputs"):
                st.json(single)

            # Add narrative using your existing function
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

        else:
            st.info("Adjust fields if needed, then click **Run Rules** to generate eligibility and assessment.")
    else:
        st.info("📂 Upload a `.xlsx` or `.csv` file to start.")



# --------- Cohort/Drift helpers ---------
def ensure_list_eligibility(df):
    if "Eligibility" in df.columns and df["Eligibility"].dtype == object:
        def _norm(x):
            if isinstance(x, list): return x
            x = str(x)
            if x.startswith("["):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return [x]
            return [x]
        df = df.copy()
        df["Eligibility"] = df["Eligibility"].apply(_norm)
    return df

def filter_by_loan(df, loan):
    if loan == "All": 
        return df
    return df[df["Eligibility"].apply(lambda xs: loan in xs)]

import ast

def normalize_pred_label(pred_label):
    """Return a list of atomic loan strings from model output (string or list)."""
    if isinstance(pred_label, list):
        return [str(x) for x in pred_label]
    if isinstance(pred_label, str):
        s = pred_label.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]
    return [str(pred_label)]


def ks_similarity(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    stat, p = ks_2samp(a, b)
    return (1 - stat), p

def tvd_categorical(a, b):
    a = pd.Series(a).astype(str)
    b = pd.Series(b).astype(str)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return 0.5 * np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum()

def numeric_feature_list(df):
    # Use an intersection of the numeric columns commonly used in your app
    candidates = [
        "Personal Credit Score","Business Credit Score","DSCR (latest year)",
        "Annual Revenue (latest year)","Years in Business","Loan Amount",
        "Industry Experience","Managerial Experience","Avg_DSCR",
        "Loan_to_Income_Ratio","Debt_to_Income_Ratio","Revenue_Growth_1y","Revenue_Growth_2y"
    ]
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def categorical_feature_list(df):
    candidates = ["Collateral Availability","Fast Approval","Business Structure","Location","Country"]
    return [c for c in candidates if c in df.columns]

def stratified_train_test_from(df, test_size=0.2, seed=42):
    # fallback if user doesn't upload train/test
    df = ensure_list_eligibility(df)
    df = df.copy()
    # make a simple label column for stratify (first loan string or 'Ineligible')
    lab = df["Eligibility"].apply(lambda xs: xs[0] if isinstance(xs, list) and len(xs)>0 else "Ineligible")
    from sklearn.model_selection import train_test_split
    tr_idx, te_idx = train_test_split(df.index, test_size=test_size, random_state=seed, stratify=lab)
    return df.loc[tr_idx].reset_index(drop=True), df.loc[te_idx].reset_index(drop=True)

# =============================
# 📊 TAB 6: Portfolio Dashboard (Combined)
# =============================
with tabs[5]:
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("📊 Portfolio Dashboard — Combined View")

    st.markdown("""
Analyze your **entire portfolio** (combined training + testing or full synthetic) with flexible filters.  
Filter by **State** (derived from `location: city,ST`) and **Loan Types**, then explore KPIs and interactive visuals.
    """)

    # ---------- Helpers ----------
    def _ensure_list_eligibility_local(df):
        """Normalize Eligibility to list-of-strings."""
        if "Eligibility" in df.columns and df["Eligibility"].dtype == object:
            def _norm(x):
                if isinstance(x, list): return x
                x = str(x)
                if x.startswith("["):
                    try:
                        return ast.literal_eval(x)
                    except Exception:
                        return [x]
                return [x]
            df = df.copy()
            df["Eligibility"] = df["Eligibility"].apply(_norm)
        return df

    def add_state_from_location(df):
        """Parse 'location' column (city,ST) into a 'State' column."""
        d = df.copy()
        # find the 'location' column (case-insensitive)
        lc = {c.lower().strip(): c for c in d.columns}
        if "location" not in lc:
            # No location column → no state
            d["State"] = None
            return d, False
        loc_col = lc["location"]

        def parse_state(v):
            if pd.isna(v):
                return None
            s = str(v).strip()
            if "," in s:
                st_code = s.split(",")[-1].strip().upper()
                return st_code
            return None

        d["State"] = d[loc_col].apply(parse_state)
        return d, True

    def eligibility_to_label(row):
        """A readable label for charts: join multi-loan into a single string (e.g., 7(a)|504)."""
        try:
            xs = row["Eligibility"]
            if isinstance(xs, list):
                return "|".join(xs)
            return str(xs)
        except Exception:
            return "Unknown"

    def any_loan_match(elig_list, selected):
        """Return True if any selected loan is present in this row's eligibility list."""
        if not selected:
            return True
        try:
            return any(l in elig_list for l in selected)
        except Exception:
            return False

    # ---------- Base data (Combined) ----------
    # prefer session_state train/test if available; else use synthetic new_df
    df_train_dash = st.session_state.get("cohort_train", None)
    df_test_dash  = st.session_state.get("cohort_test",  None)

    if df_train_dash is not None and df_test_dash is not None:
        base = pd.concat([df_train_dash, df_test_dash], axis=0, ignore_index=True)
    else:
        base = new_df.copy()

    # Normalize eligibility and derive State from location
    base = _ensure_list_eligibility_local(base)
    base, had_location = add_state_from_location(base)

    # Human-friendly label for plotting
    base["EligibilityLabel"] = base.apply(eligibility_to_label, axis=1)

    # ---------- Sidebar-like Filters (inline) ----------
    st.markdown("### 🔎 Filters")

    # Loan choices from atomic loan names that appear in any list
    all_atomic_loans = sorted(
        set([l for sub in base["Eligibility"].tolist() for l in (sub if isinstance(sub, list) else [str(sub)])])
    )

    # State filter
    states = sorted([s for s in base["State"].dropna().unique().tolist() if isinstance(s, str) and len(s) > 0])
    col_f1, col_f2, col_f3 = st.columns([1.5,1.5,2])

    with col_f1:
        state_sel = st.multiselect("State (parsed from `location`)", options=states, default=[])

    with col_f2:
        loan_sel = st.multiselect("Loan Types", options=all_atomic_loans, default=[])

    with col_f3:
        # Optional numeric sliders to refine cohort
        loan_amt_min = float(base["Loan Amount"].min()) if "Loan Amount" in base.columns else 0.0
        loan_amt_max = float(base["Loan Amount"].max()) if "Loan Amount" in base.columns else 0.0
        pcs_min = float(base["Personal Credit Score"].min()) if "Personal Credit Score" in base.columns else 0.0
        pcs_max = float(base["Personal Credit Score"].max()) if "Personal Credit Score" in base.columns else 0.0
        dscr_min = float(base["DSCR (latest year)"].min()) if "DSCR (latest year)" in base.columns else 0.0
        dscr_max = float(base["DSCR (latest year)"].max()) if "DSCR (latest year)" in base.columns else 0.0

        # Show compact inline hints if a column is missing
        _show_amt = "Loan Amount" in base.columns and loan_amt_max > loan_amt_min
        _show_pcs = "Personal Credit Score" in base.columns and pcs_max > pcs_min
        _show_dscr = "DSCR (latest year)" in base.columns and dscr_max > dscr_min

    col_rng1, col_rng2, col_rng3 = st.columns(3)
    with col_rng1:
        if _show_amt:
            sel_amt = st.slider("Loan Amount range", min_value=loan_amt_min, max_value=loan_amt_max,
                                value=(loan_amt_min, loan_amt_max), step=1000.0)
        else:
            sel_amt = (loan_amt_min, loan_amt_max)
            st.caption("Loan Amount range disabled (missing or constant).")
    with col_rng2:
        if _show_pcs:
            sel_pcs = st.slider("Personal Credit Score range", min_value=pcs_min, max_value=pcs_max,
                                value=(pcs_min, pcs_max), step=1.0)
        else:
            sel_pcs = (pcs_min, pcs_max)
            st.caption("PCS range disabled (missing or constant).")
    with col_rng3:
        if _show_dscr:
            sel_dscr = st.slider("DSCR (latest year) range", min_value=dscr_min, max_value=dscr_max,
                                 value=(dscr_min, dscr_max), step=0.01)
        else:
            sel_dscr = (dscr_min, dscr_max)
            st.caption("DSCR range disabled (missing or constant).")

    # ---------- Apply filters ----------
    filt = base.copy()

    if state_sel:
        filt = filt[filt["State"].isin(state_sel)]

    if loan_sel:
        filt = filt[filt["Eligibility"].apply(lambda xs: any_loan_match(xs, loan_sel))]

    # numeric filters
    if _show_amt:
        filt = filt[(filt["Loan Amount"] >= sel_amt[0]) & (filt["Loan Amount"] <= sel_amt[1])]
    if _show_pcs:
        filt = filt[(filt["Personal Credit Score"] >= sel_pcs[0]) & (filt["Personal Credit Score"] <= sel_pcs[1])]
    if _show_dscr:
        filt = filt[(filt["DSCR (latest year)"] >= sel_dscr[0]) & (filt["DSCR (latest year)"] <= sel_dscr[1])]

    # ---------- KPIs ----------
    st.markdown("### 📌 Portfolio KPIs")
    k1, k2, k3, k4 = st.columns(4)
    total_apps = len(filt)
    avg_amt = float(filt["Loan Amount"].mean()) if "Loan Amount" in filt.columns and total_apps else 0.0
    med_pcs = float(filt["Personal Credit Score"].median()) if "Personal Credit Score" in filt.columns and total_apps else 0.0

    # Eligible vs Ineligible ratio
    def is_ineligible(lst):
        try:
            return lst == ["Ineligible"]
        except Exception:
            return False
    inel = int(filt["Eligibility"].apply(is_ineligible).sum()) if "Eligibility" in filt.columns else 0
    eligible = total_apps - inel
    elig_rate = (eligible / total_apps * 100.0) if total_apps else 0.0

    k1.metric("Applicants (Filtered)", f"{total_apps:,}")
    k2.metric("Avg Loan Amount", f"${avg_amt:,.0f}")
    k3.metric("Median Personal Credit Score", f"{med_pcs:.0f}")
    k4.metric("Eligibility Rate", f"{elig_rate:.1f}%")

    # ---------- Dashboard Layout ----------
    st.markdown("### 📈 Dashboard")

    g1, g2 = st.columns(2)
    with g1:
        # Loan Amount distribution
        if "Loan Amount" in filt.columns:
            fig = px.histogram(filt, x="Loan Amount", nbins=40, title="Loan Amount Distribution",
                               marginal="box")
            fig.update_layout(bargap=0.02)
            st.plotly_chart(fig, use_container_width=True, key="dash_hist_amt")
        else:
            st.info("Loan Amount not available.")

    with g2:
        # Loan Type share (pie)
        if "EligibilityLabel" in filt.columns:
            pie_df = filt["EligibilityLabel"].value_counts().reset_index()
            pie_df.columns = ["EligibilityLabel", "Count"]
            fig = px.pie(pie_df, names="EligibilityLabel", values="Count", title="Loan Type Mix")
            st.plotly_chart(fig, use_container_width=True, key="dash_pie_loanmix")
        else:
            st.info("Eligibility not available to build loan mix.")

    g3, g4 = st.columns(2)
    with g3:
        # Loan Amount by State (top 12 states)
        if "State" in filt.columns and "Loan Amount" in filt.columns and filt["State"].notna().any():
            top_states = filt["State"].value_counts().index[:12].tolist()
            sub = filt[filt["State"].isin(top_states)]
            fig = px.box(sub, x="State", y="Loan Amount", title="Loan Amount by State (Top 12)")
            st.plotly_chart(fig, use_container_width=True, key="dash_box_state_amt")
        else:
            st.info("State or Loan Amount not available for state-wise boxplot.")

    with g4:
        # PCS vs DSCR scatter colored by loan label
        if all(c in filt.columns for c in ["Personal Credit Score", "DSCR (latest year)", "EligibilityLabel"]):
            fig = px.scatter(
                filt, x="Personal Credit Score", y="DSCR (latest year)",
                color="EligibilityLabel", hover_data=["Applicant ID"] if "Applicant ID" in filt.columns else None,
                title="Credit Score vs DSCR by Loan Label", opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True, key="dash_scatter_pcs_dscr")
        else:
            st.info("Need Personal Credit Score, DSCR (latest year), and EligibilityLabel for scatter.")

    # ---------- State vs State Comparison ----------
    st.markdown("### 🗺️ State Comparison")
    if "State" in filt.columns and filt["State"].notna().any():
        comp_states = sorted(filt["State"].dropna().unique().tolist())
        cs1, cs2 = st.columns(2)
        with cs1:
            sA = st.selectbox("State A", options=comp_states, index=0, key="cmp_state_a")
        with cs2:
            sB = st.selectbox("State B", options=comp_states, index=min(1, len(comp_states)-1), key="cmp_state_b")

        subA = filt[filt["State"] == sA]
        subB = filt[filt["State"] == sB]

        ca, cb, cc = st.columns(3)
        ca.metric(f"{sA} • Applicants", f"{len(subA):,}")
        cb.metric(f"{sB} • Applicants", f"{len(subB):,}")
        if "Loan Amount" in filt.columns:
            cc.metric(f"{sA} vs {sB} • Avg Loan", f"${subA['Loan Amount'].mean():,.0f} / ${subB['Loan Amount'].mean():,.0f}")

        # Stacked bar of loan type shares
        if "EligibilityLabel" in filt.columns:
            def share(df):
                vc = df["EligibilityLabel"].value_counts(normalize=True)
                return vc

            sA_sh = share(subA)
            sB_sh = share(subB)
            cats = sorted(set(sA_sh.index).union(set(sB_sh.index)))
            comp_df = pd.DataFrame({
                "EligibilityLabel": cats,
                f"{sA}": [sA_sh.get(c, 0.0) for c in cats],
                f"{sB}": [sB_sh.get(c, 0.0) for c in cats]
            })

            comp_melt = comp_df.melt(id_vars="EligibilityLabel", var_name="State", value_name="Share")
            fig = px.bar(comp_melt, x="EligibilityLabel", y="Share", color="State", barmode="group",
                         title="Loan Type Share by State")
            st.plotly_chart(fig, use_container_width=True, key="dash_state_share")
    else:
        st.info("No `location` parsed → `State` not available for state comparison. Ensure values like `city,ST`.")

    # ---------- Download filtered ----------
    st.markdown("### ⬇️ Download Filtered View")
    try:
        csv_bytes = filt.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="portfolio_filtered.csv", mime="text/csv")
    except Exception:
        st.info("Unable to prepare CSV export.")



# =============================
# 👤 TAB 7: Applicant vs Cohort Comparison
# =============================
with tabs[6]:
    st.header("👤 Applicant vs Cohort Comparison")

    # --- Helpers (only define if missing) ---
    try:
        ensure_list_eligibility
    except NameError:
        import ast as _ast
        def ensure_list_eligibility(df):
            if "Eligibility" in df.columns and df["Eligibility"].dtype == object:
                def _norm(x):
                    if isinstance(x, list): return x
                    x = str(x)
                    if x.startswith("["):
                        try:
                            return _ast.literal_eval(x)
                        except Exception:
                            return [x]
                    return [x]
                df = df.copy()
                df["Eligibility"] = df["Eligibility"].apply(_norm)
            return df

    try:
        filter_by_loan
    except NameError:
        def filter_by_loan(df, loan):
            if loan == "All": 
                return df
            return df[df["Eligibility"].apply(lambda xs: loan in xs)]

    # Normalize any model output to a list of atomic loans, e.g. "['Express']" -> ["Express"]
    def normalize_pred_label(pred_label):
        import ast as _ast
        if isinstance(pred_label, list):
            return [str(x) for x in pred_label]
        if isinstance(pred_label, str):
            s = pred_label.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = _ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            return [s]
        return [str(pred_label)]

    # numeric feature chooser (robust)
    def numeric_feature_list(df):
        return df.select_dtypes(include=[np.number]).columns.tolist()

    # simple stratified split if cohort not cached
    def stratified_train_test_from(base_df, test_size=0.2, seed=42):
        from sklearn.model_selection import train_test_split
        bdf = base_df.copy()
        # Make a stable string label: join multi-loans with '|'
        bdf["_elig_str"] = bdf["Eligibility"].apply(lambda xs: "|".join(sorted(map(str, xs))) if isinstance(xs, list) else str(xs))
        try:
            tr, te = train_test_split(bdf, test_size=test_size, random_state=seed, stratify=bdf["_elig_str"])
        except Exception:
            # fallback non-stratified
            tr, te = train_test_split(bdf, test_size=test_size, random_state=seed, shuffle=True)
        for d in (tr, te):
            d.drop(columns=["_elig_str"], inplace=True, errors="ignore")
        return tr.reset_index(drop=True), te.reset_index(drop=True)

    # --- Need prediction artifacts from Tab 4 ---
    if "pred_results_df" not in st.session_state or "pred_features_df" not in st.session_state:
        st.warning("Run predictions in the **Live Predictions** tab first. Then come back here.")
        st.stop()

    results_df    = st.session_state["pred_results_df"]      # columns: Applicant ID, predictions...
    feat_df       = st.session_state["pred_features_df"]     # features aligned to model input order
    label_encoder = st.session_state.get("label_encoder", None)

    # --- Load/prepare cohorts (train/test) ---
    df_train_cmp = st.session_state.get("cohort_train", None)
    df_test_cmp  = st.session_state.get("cohort_test",  None)
    if df_train_cmp is None or df_test_cmp is None:
        # Use synthetic (new_df) split as fallback
        base_df = ensure_list_eligibility(new_df.copy())
        df_train_cmp, df_test_cmp = stratified_train_test_from(base_df, test_size=0.2, seed=42)
        st.session_state["cohort_train"] = df_train_cmp.copy()
        st.session_state["cohort_test"]  = df_test_cmp.copy()

    # Normalize eligibility lists
    df_train_cmp = ensure_list_eligibility(df_train_cmp)
    df_test_cmp  = ensure_list_eligibility(df_test_cmp)

    # --- UI with UNIQUE KEYS ---
    st.markdown("Pick an **applicant** and which **cohort** to compare them against.")
    sel_app_id = st.selectbox(
        "Applicant ID",
        options=results_df["Applicant ID"].tolist(),
        index=0,
        key="cmp_applicant_id_select"
    )
    cohort_choice = st.radio(
        "Cohort",
        ["Training", "Testing"],
        horizontal=True,
        key="cmp_cohort_choice"
    )
    model_for_label = st.radio(
        "Use predicted label from:",
        ["Ensemble", "XGBoost", "LightGBM"],
        horizontal=True,
        key="cmp_model_for_label"
    )

    # --- Extract applicant row from predictions table ---
    try:
        app_idx = results_df.index[results_df["Applicant ID"] == sel_app_id][0]
    except Exception:
        st.error("Selected Applicant ID not found in predictions.")
        st.stop()
    app_row = feat_df.iloc[[app_idx]]  # 1-row DataFrame (same feature columns as training)

    # --- Determine predicted label used to match cohort ---
    if model_for_label == "Ensemble":
        raw_pred_label = results_df.loc[app_idx, "Ensemble Prediction"]
    elif model_for_label == "XGBoost":
        raw_pred_label = results_df.loc[app_idx, "XGBoost Prediction"]
    else:
        raw_pred_label = results_df.loc[app_idx, "LightGBM Prediction"]

    pred_loans = normalize_pred_label(raw_pred_label)  # e.g., ["Express"] or ["7(a)", "504"]
    st.markdown(f"**Predicted loan for comparison:** `{pred_loans}`")

    # --- Choose cohort split and filter by ANY predicted loan ---
    cohort_df = df_train_cmp if cohort_choice == "Training" else df_test_cmp
    cohort_df = cohort_df[cohort_df["Eligibility"].apply(lambda xs: any(l in xs for l in pred_loans))]

    st.markdown(f"**Cohort size (same loan):** {len(cohort_df)}")
    if len(cohort_df) == 0:
        # Diagnostics: show what's available in this split
        _avail_counts = {
            l: int((cohort_df.assign(_h=cohort_df["Eligibility"].apply(lambda xs, l=l: l in xs))["_h"]).sum())
            for l in ["7(a)", "504", "Express"]
        } if len((df_train_cmp if cohort_choice=="Training" else df_test_cmp)) else {}
        st.info(f"Availability in this split (debug): { _avail_counts }")
        st.warning("No cohort records found for this loan in the selected split.")
        st.stop()

    # --- Numeric features common to both applicant vector and cohort ---
    possible_num = numeric_feature_list(pd.concat([cohort_df, app_row], axis=0))
    comp_num_cols = [c for c in possible_num if c in cohort_df.columns and c in app_row.columns]
    if len(comp_num_cols) == 0:
        st.warning("No common numeric features available to compare between applicant and cohort.")
        st.stop()

    # --- Describe cohort ---
    stats = cohort_df[comp_num_cols].describe().T
    means = stats["mean"]
    stds  = stats["std"].replace(0, np.nan)
    med   = stats["50%"]

    # --- Applicant values ---
    app_vals = app_row[comp_num_cols].iloc[0].astype(float)

    # --- Z-scores and percentiles ---
    zscore = (app_vals - means) / stds
    pct = cohort_df[comp_num_cols].apply(lambda s, v=app_vals: (s <= v[s.name]).mean()*100)

    comp_table = pd.DataFrame({
        "Applicant": app_vals,
        "Cohort Median": med,
        "Z-Score": zscore,
        "Percentile": pct.round(1)
    }).sort_values("Percentile", ascending=False)

    st.subheader("📋 Applicant vs Cohort (Numeric Features)")
    st.dataframe(comp_table)

    # --- Polar chart (top 6 by |z|) ---
    try:
        import plotly.graph_objects as go
        top_rad = comp_table.reindex(comp_table["Z-Score"].abs().sort_values(ascending=False).head(6).index)
        theta = top_rad.index.tolist()
        rel = ((top_rad["Applicant"] - top_rad["Cohort Median"]) /
               (top_rad["Cohort Median"].replace(0, np.nan))).fillna(0).values
        rel = np.clip(rel, -3, 3)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=rel, theta=theta, fill='toself', name='Applicant tilt vs median'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3,3])), showlegend=False,
                          title="Applicant tilt vs Cohort Median (relative)")
        st.plotly_chart(fig, use_container_width=True, key="cmp_polar_plot")
    except Exception as e:
        st.info(f"Polar chart unavailable: {e}")

    # --- Nearest neighbors on common numeric features ---
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(cohort_df)), metric="euclidean")
        X_tr = cohort_df[comp_num_cols].fillna(0)
        X_app = app_row[comp_num_cols].fillna(0)
        nn.fit(X_tr)
        dists, idxs = nn.kneighbors(X_app)
        nn_rows = cohort_df.iloc[idxs[0]].copy()
        nn_rows.insert(0, "Distance", dists[0])

        show_nn_cols = [c for c in ["Applicant ID","Business Name"] if c in nn_rows.columns]
        st.subheader("🔎 Most Similar Cohort Applicants")
        st.dataframe(nn_rows[["Distance"] + show_nn_cols + comp_num_cols[:5]].reset_index(drop=True))
    except Exception as e:
        st.info(f"Nearest-neighbor view unavailable: {e}")


        # ============================
    # Extra Visuals for Tab 7
    # ============================

    st.markdown("---")
    st.subheader("📊 Feature Distribution Comparison")

    # Let the user pick up to 4 numeric features to compare
    choose_feats = st.multiselect(
        "Pick up to 4 features to compare (cohort distribution + applicant marker)",
        options=comp_num_cols,
        default=[c for c in comp_num_cols if c in ["Loan Amount", "Personal Credit Score", "DSCR (latest year)", "Annual Revenue (latest year)"]][:4],
        max_selections=4,
        key="cmp_dist_feats"
    )

    if choose_feats:
        import plotly.graph_objects as go
        import plotly.express as px

        cols = st.columns(2) if len(choose_feats) > 1 else [st]
        for i, feat in enumerate(choose_feats):
            pane = cols[i % len(cols)]
            try:
                fig = px.histogram(
                    cohort_df, x=feat, nbins=30, title=f"{feat} — Cohort Distribution",
                    marginal="box", opacity=0.85
                )
                # Applicant vertical line
                fig.add_vline(
                    x=float(app_row[feat].iloc[0]),
                    line_width=3, line_dash="dash", line_color="black",
                    annotation_text="Applicant", annotation_position="top"
                )
                pane.plotly_chart(fig, use_container_width=True, key=f"cmp_hist_{feat}")
            except Exception as e:
                pane.info(f"Could not build histogram for `{feat}`: {e}")
    else:
        st.info("Pick at least one numeric feature above to see distribution overlays.")

    st.markdown("---")
    st.subheader("📦 Multi-Feature Boxplot (Top |Z| Features)")

    # Use existing z-scores table to get the top features
    try:
        top_box = comp_table.reindex(comp_table["Z-Score"].abs().sort_values(ascending=False).head(8).index)
        box_feats = top_box.index.tolist()

        if len(box_feats):
            # long-form melt for boxplot
            show_df = cohort_df[box_feats].copy()
            long_df = show_df.melt(var_name="Feature", value_name="Value")

            fig = px.box(long_df, x="Feature", y="Value", points=False, title="Top Deviations — Cohort Boxplots")
            # Add applicant markers per feature
            for f in box_feats:
                fig.add_scatter(
                    x=[f], y=[float(app_row[f].iloc[0])],
                    mode="markers", marker=dict(size=12),
                    name=f"Applicant • {f}", showlegend=False
                )
            st.plotly_chart(fig, use_container_width=True, key="cmp_box_multi")
        else:
            st.info("No high-deviation features available for boxplot.")
    except Exception as e:
        st.info(f"Boxplot unavailable: {e}")



