# loan_app/src/tabs/tab4_live_predictions.py
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from typing import Dict, List, Tuple

# Your new model loader and features
from src.model_loader import load_model_bundle
from src.ml_features import engineer, add_cross_features

# Rules + narrative (used for relabel + summary)
from src.rules import (
    default_rules,
    check_loan_eligibility_configurable,  # used to relabel training corpus
    generate_applicant_summary,
)

# Optional: training data loader
from src.io_utils import load_training_corpus

# ----------------------------
# UI helpers
# ----------------------------
BOOLEAN_FIELDS = [
    "For Profit", "Fast Approval", "Collateral Availability",
    "Working Capital", "Business Expansion", "Equipment Purchase or Leasing",
    "Inventory Purchase", "Real Estate Acquisition or Improvement",
    "Business Acquisition or Buyout", "Refinancing Existing Debt",
    "Emergency Funds", "Franchise Financing", "Contract Financing",
    "Licensing or Permits", "Line of Credit Establishment",
]
NUMERIC_FIELDS = [
    "Personal Credit Score","Business Credit Score","DSCR (latest year)","Annual Revenue (latest year)","Loan Amount",
    "Years in Business","Net Profit Margin","NOI (latest year)","NOI (1 year ago)","NOI (2 years ago)",
    "Industry Experience","Managerial Experience","Business Debt (latest year)","Business Debt (1 year ago)",
    "Annual Revenue (1 year ago)","Annual Revenue (2 years ago)"
]

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().isin(["true","1","yes","y","t"])

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
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

# ----------------------------
# Prediction using bundle
# ----------------------------
def predict_on_new(df_new: pd.DataFrame, bundle: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      preds: df with [proba_7(a), proba_504, proba_Express, label flags, Eligibility_Pred, Main_Prediction, Confidence]
      Xn0: feature matrix BEFORE scaling (engineered + crosses)
    """
    models_cv   = bundle["models_cv"]
    thresholds  = bundle["thresholds"]
    scaler      = bundle["scaler"]
    feature_cols= bundle["feature_cols"]
    labels      = bundle["labels"]

    # 1) Engineer + crosses
    Xn0 = engineer(df_new).copy()
    Xn0 = add_cross_features(Xn0)
    for c in Xn0.columns:
        if Xn0[c].dtype == bool:
            Xn0[c] = Xn0[c].astype(float)
    Xn0 = Xn0.reindex(columns=feature_cols, fill_value=0.0)
    Xn  = scaler.transform(Xn0)

    # 2) per-label probs (average across CV folds)
    prob = {}
    for lab in labels:
        lab_probs = [m.predict_proba(Xn)[:,1] for m in models_cv[lab]]
        prob[lab] = np.mean(lab_probs, axis=0)

    preds = {lab: (prob[lab] >= thresholds[lab]).astype(int) for lab in labels}
    Yp = pd.DataFrame(preds, index=df_new.index)

    # 3) Ineligible exclusivity
    if "Ineligible" in labels:
        any_pos = Yp[[c for c in ["7(a)","504","Express"] if c in Yp.columns]].sum(axis=1) > 0
        Yp.loc[any_pos, "Ineligible"] = 0
        Yp.loc[~any_pos, "Ineligible"] = 1

    out = df_new.copy()
    # add proba columns + hard predictions
    for lab in ["7(a)","504","Express"]:
        if lab in labels:
            out[f"proba_{lab}"] = prob[lab]
            out[lab] = Yp[lab] if lab in Yp.columns else 0

    # predicted multi-label list
    def to_list(r):
        z=[]
        if "7(a)" in out.columns and r["7(a)"]==1: z.append("7(a)")
        if "504" in out.columns and r["504"]==1: z.append("504")
        if "Express" in out.columns and r["Express"]==1: z.append("Express")
        return z if z else ["Ineligible"]
    out["Eligibility_Pred"] = out.apply(to_list, axis=1)

    # derive "main" predicted label + confidence for this tab (like old UI)
    def main_pred(r):
        candidates = []
        for lab in ["7(a)","504","Express"]:
            col = f"proba_{lab}"
            if col in out.columns:
                candidates.append((lab, r[col]))
        if not candidates:
            return "Ineligible", 0.0
        # pick highest probability
        lab, p = max(candidates, key=lambda x: x[1])
        return lab, float(p)

    main, conf = [], []
    for idx, row in out.iterrows():
        m, c = main_pred(row)
        main.append(m if m else "Ineligible")
        conf.append(c)
    out["Main_Prediction"] = main
    out["Confidence"] = conf

    # 🔁 Compatibility for Tab 7
    out["Ensemble Prediction"] = out["Main_Prediction"]
    out["Confidence (Ensemble)"] = out["Confidence"]

    return out, Xn0

# ----------------------------
# SHAP for a label (first fold)
# ----------------------------
def _shap_for_label(app_features_row: pd.Series, bundle: dict, label: str):
    models_cv = bundle["models_cv"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    if label not in models_cv or not models_cv[label]:
        return None
    model = models_cv[label][0]  # use first fold
    x_row = app_features_row.reindex(feature_cols, fill_value=0.0).values.reshape(1, -1)
    x_row_scaled = scaler.transform(x_row)
    explainer = shap.TreeExplainer(model)
    try:
        sv = explainer(x_row_scaled)
    except Exception:
        return None
    return sv, feature_cols

# ----------------------------
# Model training (beta) - multi-label CV
# ----------------------------
def _train_multilabel_bundle(train_df: pd.DataFrame, rules: Dict, beta_map: Dict[str,float], progress=None) -> dict:
    """
    Re-train the CV-based LightGBM per-label classifiers on rule-labeled data.
    Uses F-beta tuning per label via beta_map.
    Returns a new bundle (models_cv, thresholds, scaler, feature_cols, labels).
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import precision_recall_curve
    import lightgbm as lgb

    def pbar(stage, pct, msg):
        if progress:
            progress.progress(min(max(pct, 0.0), 1.0))
            st.write(f"**{stage}** — {msg}")

    pbar("data", 0.05, "Engineering features...")
    # Build X (engineer + crosses)
    X0 = engineer(train_df).copy()
    X0 = add_cross_features(X0).fillna(0)

    # rules apply (multi-loan list) via configurables
    relabeled = check_loan_eligibility_configurable(train_df.copy(), rules)
    y_lists = relabeled["Eligibility"]
    labels = ["7(a)","504","Express","Ineligible"]
    Y = pd.DataFrame(0, index=X0.index, columns=labels)
    for i, val in enumerate(y_lists):
        if isinstance(val, list):
            for v in val:
                if v in labels:
                    Y.loc[i, v] = 1
            if not any(v in labels for v in val):
                Y.loc[i, "Ineligible"] = 1
        else:
            s = str(val)
            if s in labels:
                Y.loc[i, s] = 1
            else:
                Y.loc[i, "Ineligible"] = 1
    # exclusivity for Ineligible
    any_pos = Y[["7(a)","504","Express"]].sum(axis=1) > 0
    Y.loc[any_pos, "Ineligible"] = 0

    pbar("data", 0.15, "Scaling numeric features...")
    scaler = RobustScaler(with_centering=False).fit(X0.values)
    X_s = pd.DataFrame(scaler.transform(X0.values), columns=X0.columns, index=X0.index)

    mskf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # monotonic constraints approx (optional)
    mono = []
    inc_cols = [
        "Personal Credit Score","Business Credit Score","DSCR (latest year)","Years in Business",
        "Profitability","Experience_Score","Maturity","NOI (latest year)","Annual Revenue (latest year)"]
    dec_cols = ["Debt_to_Revenue","Loan_to_Revenue","Business Debt (latest year)","Debt_Growth"]
    for col in X_s.columns:
        if col in inc_cols:
            mono.append(1)
        elif col in dec_cols:
            mono.append(-1)
        else:
            mono.append(0)

    def lgb_params_for(label, spw=1.0):
        return dict(
            objective="binary",
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            reg_alpha=2.0,
            reg_lambda=6.0,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=100,
            monotone_constraints=mono,
            verbosity=-1,
            n_jobs=-1,
            scale_pos_weight=spw,
        )

    models_cv = {lab: [] for lab in labels}
    oof_probs = {lab: np.zeros(len(X_s)) for lab in labels}

    for fold, (tr, va) in enumerate(mskf.split(X_s, Y.values)):
        Xtr, Xva = X_s.iloc[tr], X_s.iloc[va]
        ytr, yva = Y.iloc[tr], Y.iloc[va]

        for lab in labels:
            pos = ytr[lab].sum()
            neg = len(ytr) - pos
            spw = float(neg / max(pos, 1))
            model = lgb.LGBMClassifier(**lgb_params_for(lab, spw))
            model.fit(
                Xtr, ytr[lab],
                eval_set=[(Xva, yva[lab])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(60, verbose=False)]
            )
            models_cv[lab].append(model)
            oof_probs[lab][va] = model.predict_proba(Xva)[:, 1]

        pbar("train", 0.4 + 0.2 * (fold+1) / 3.0, f"Fold {fold+1} done.")

    # threshold tuning per beta_map
    def tune_threshold_beta(y_true, y_prob, beta=1.0):
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(y_true, y_prob)
        if len(t) == 0:
            return 0.5
        fb = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-12)
        return float(np.clip(t[np.nanargmax(fb[:-1])], 0.1, 0.9))

    pbar("thresh", 0.85, "Tuning thresholds (F-beta)...")
    thresholds = {}
    for lab in labels:
        beta = beta_map.get(lab, 1.0)
        thresholds[lab] = tune_threshold_beta(Y[lab].values, oof_probs[lab], beta=beta)

    pbar("done", 1.0, "Training complete.")
    bundle = {
        "models_cv": models_cv,
        "thresholds": thresholds,
        "scaler": scaler,
        "feature_cols": list(X_s.columns),
        "labels": labels
    }
    return bundle

# ----------------------------
# Tab 4: Renderer
# ----------------------------
def render():
    st.header("🧠 Live Predictions & Explainability (Multi-label LGBM)")

    st.markdown("""
This tab uses your **LightGBM multi-label CV** model to predict **eligible loan types** per applicant.  
You can:  
- Edit **Tab-scoped rules** (per-loan).  
- **Relabel & Retrain** the model based on current rules.  
- Adjust **Threshold Auto-Tuning** (F-β) **per loan**.  
- Save retrained model as a **new .pkl** without overwriting the existing one.  
    """)

    # ----------------------------
    # Rules (Per-loan, tab-scoped)
    # ----------------------------
    st.subheader("⚙️ (Tab-scoped) Rules — Per Loan")
    with st.expander("Edit rules for this tab", expanded=False):
        rules_state = st.session_state.get("eligibility_rules", default_rules())

        # 7(a)
        st.markdown("### **7(a) Parameters**")
        c1, c2, c3, c4 = st.columns(4)
        rules_state["7a_requires_for_profit"] = c1.checkbox("Requires For-Profit", value=rules_state["7a_requires_for_profit"], key="t4_7a_fp")
        rules_state["7a_min_personal_credit"] = c2.number_input("Min Personal Credit", 300, 850, int(rules_state["7a_min_personal_credit"]), key="t4_7a_minpcs")
        rules_state["7a_min_business_credit"] = c3.number_input("Min Business Credit", 0, 300, int(rules_state["7a_min_business_credit"]), key="t4_7a_minbcs")
        rules_state["7a_min_dscr"] = c4.number_input("Min DSCR", 0.0, 5.0, float(rules_state["7a_min_dscr"]), step=0.01, format="%.2f", key="t4_7a_mindscr")

        c5, c6, c7 = st.columns(3)
        rules_state["7a_min_years_in_business"] = c5.number_input("Min Years in Business", 0, 100, int(rules_state["7a_min_years_in_business"]), key="t4_7a_yib")
        rules_state["7a_loan_min"] = c6.number_input("Min Loan Amount", 0, 10_000_000, int(rules_state["7a_loan_min"]), step=1000, key="t4_7a_minloan")
        rules_state["7a_loan_max"] = c7.number_input("Max Loan Amount", 0, 10_000_000, int(rules_state["7a_loan_max"]), step=1000, key="t4_7a_maxloan")
        rules_state["7a_exclude_purposes"] = st.multiselect("Excluded Purposes (7a)", options=rules_state["all_purposes"], default=rules_state["7a_exclude_purposes"], key="t4_7a_excl")

        # 8(a)
        st.markdown("### **8(a) Parameters**")
        rules_state["enable_8a"] = st.checkbox("Enable 8(a) Rule", value=rules_state["enable_8a"], key="t4_8a_enable")
        c1, c2, c3, c4, c5 = st.columns(5)
        rules_state["8a_requires_for_profit"] = c1.checkbox("Requires For-Profit?", value=rules_state["8a_requires_for_profit"], key="t4_8a_fp")
        rules_state["8a_requires_fast_approval"] = c2.checkbox("Requires NOT Fast Approval", value=rules_state["8a_requires_fast_approval"], key="t4_8a_notfast")
        rules_state["8a_min_years_in_business"] = c3.number_input("Min Years in Business", 0, 50, int(rules_state["8a_min_years_in_business"]), key="t4_8a_yib")
        rules_state["8a_min_industry_exp"] = c4.number_input("Min Industry Exp (yrs)", 0, 50, int(rules_state["8a_min_industry_exp"]), key="t4_8a_ind")
        rules_state["8a_min_managerial_exp"] = c5.number_input("Min Managerial Exp (yrs)", 0, 50, int(rules_state["8a_min_managerial_exp"]), key="t4_8a_mgr")
        rules_state["8a_exclude_purposes"] = st.multiselect("Excluded Purposes (8a)", options=rules_state["all_purposes"], default=rules_state["8a_exclude_purposes"], key="t4_8a_excl")

        # 504
        st.markdown("### **504 Parameters**")
        c1, c2, c3, c4 = st.columns(4)
        rules_state["504_requires_for_profit"] = c1.checkbox("Requires For-Profit", value=rules_state["504_requires_for_profit"], key="t4_504_fp")
        rules_state["504_requires_collateral"] = c2.checkbox("Requires Collateral", value=rules_state["504_requires_collateral"], key="t4_504_coll")
        rules_state["504_min_personal_credit"] = c3.number_input("Min Personal Credit", 300, 850, int(rules_state["504_min_personal_credit"]), key="t4_504_minpcs")
        rules_state["504_min_dscr"] = c4.number_input("Min DSCR", 0.0, 5.0, float(rules_state["504_min_dscr"]), step=0.01, format="%.2f", key="t4_504_mindscr")

        c5, c6, c7 = st.columns(3)
        rules_state["504_min_net_profit_margin"] = c5.number_input("Min Net Profit Margin", -100.0, 100.0, float(rules_state["504_min_net_profit_margin"]), step=0.1, format="%.1f", key="t4_504_minnpm")
        rules_state["504_min_years_in_business"] = c6.number_input("Min Years in Business", 0, 100, int(rules_state["504_min_years_in_business"]), key="t4_504_minyib")
        rules_state["504_max_loan"] = c7.number_input("Max Loan Amount", 0, 10_000_000, int(rules_state["504_max_loan"]), step=1000, key="t4_504_maxloan")
        rules_state["504_exclude_purposes"] = st.multiselect("Excluded Purposes (504)", options=rules_state["all_purposes"], default=rules_state["504_exclude_purposes"], key="t4_504_excl")

        # Express
        st.markdown("### **Express Parameters**")
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        rules_state["express_requires_for_profit"] = e1.checkbox("Requires For-Profit", value=rules_state["express_requires_for_profit"], key="t4_ex_fp")
        rules_state["express_requires_fast_approval"] = e2.checkbox("Requires Fast Approval", value=rules_state["express_requires_fast_approval"], key="t4_ex_fast")
        rules_state["express_min_personal_credit"] = e3.number_input("Min Personal Credit", 300, 850, int(rules_state["express_min_personal_credit"]), key="t4_ex_minpcs")
        rules_state["express_min_business_credit"] = e4.number_input("Min Business Credit", 0, 300, int(rules_state["express_min_business_credit"]), key="t4_ex_minbcs")
        rules_state["express_min_dscr"] = e5.number_input("Min DSCR", 0.0, 5.0, float(rules_state["express_min_dscr"]), step=0.01, format="%.2f", key="t4_ex_mindscr")
        rules_state["express_max_loan"] = e6.number_input("Max Loan Amount", 0, 10_000_000, int(rules_state["express_max_loan"]), step=1000, key="t4_ex_maxloan")
        rules_state["express_exclude_purposes"] = st.multiselect("Excluded Purposes (Express)", options=rules_state["all_purposes"], default=rules_state["express_exclude_purposes"], key="t4_ex_excl")

        st.session_state["eligibility_rules"] = rules_state
        st.caption("These settings affect only this tab (used when retraining).")

    # ----------------------------
    # Threshold tuning UI (per loan)
    # ----------------------------
    st.subheader("🎯 Threshold Auto-Tuning (F-β)")
    st.caption("Higher β increases recall preference. Common: 1.0 (balanced), 1.3–1.5 (recall-biased).")
    beta_7a = st.slider("β for 7(a)", 0.2, 3.0, 1.3, 0.1, key="t4_beta_7a")
    beta_504 = st.slider("β for 504", 0.2, 3.0, 1.0, 0.1, key="t4_beta_504")
    beta_ex = st.slider("β for Express", 0.2, 3.0, 1.0, 0.1, key="t4_beta_express")
    beta_map = {"7(a)": beta_7a, "504": beta_504, "Express": beta_ex, "Ineligible": 1.0}

    # ----------------------------
    # Upload & predict
    # ----------------------------
    uploaded = st.file_uploader("📤 Upload Applicants (xlsx/csv)", type=["xlsx", "csv"], key="t4_uploader")
    if not uploaded:
        st.info("Upload a file to begin.")
        return

    try:
        df_input = pd.read_excel(uploaded) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return
    st.success(f"Loaded {df_input.shape[0]} applicants.")

    # Normalize + predict using bundle (or override if retrained earlier)
    df_norm = _normalize(df_input)
    # After: df_norm = _normalize(df_input)
    st.session_state["t4_input_df"] = df_norm.copy()   # <-- add this line

    base_bundle = load_model_bundle()
    active_bundle = st.session_state.get("t4_bundle_override", base_bundle)
    preds, Xn0 = predict_on_new(df_norm, active_bundle)

    st.subheader("📊 Predictions")
    view_cols = [c for c in ["Applicant ID", "Business Name"] if c in preds.columns] + \
                ["Main_Prediction","Confidence","Eligibility_Pred","proba_7(a)","proba_504","proba_Express"]
    st.dataframe(preds[view_cols], use_container_width=True)

    # Save in session for explainability and other tabs
    st.session_state["lp_raw_df"] = df_input.copy()  
    st.session_state["pred_results_df"] = preds
    st.session_state["pred_features_df"] = Xn0
    st.session_state["ml_labels"] = active_bundle["labels"]

    # ----------------------------
    # Relabel & Retrain (beta)
    # ----------------------------
    st.markdown("---")
    st.subheader("🔁 Relabel & Retrain (beta)")

    st.caption("Retrains the CV per-label LGBM models using your **current rules** and F-β settings.")
    retrain_btn = st.button("Relabel data & retrain model", key="t4_retrain_btn")
    if retrain_btn:
        prog = st.progress(0.0)
        st.info("Loading training corpus...")
        base_df, loaded_files = load_training_corpus(files=("given_data.csv", "synthetic_data_generated.xlsx"), data_dir=None)
        if base_df is None or base_df.empty:
            st.error("Could not load training data.")
            return
        st.success(f"Loaded training corpus {base_df.shape[0]} rows from: {loaded_files}")

        # retrain
        try:
            new_bundle = _train_multilabel_bundle(
                base_df.copy(),
                st.session_state["eligibility_rules"],
                beta_map=beta_map,
                progress=prog
            )
            st.success("Retraining complete.")
            # re-run predictions on uploaded df
            new_preds, new_Xn0 = predict_on_new(df_norm, new_bundle)
            st.session_state["pred_results_df"] = new_preds
            st.session_state["pred_features_df"] = new_Xn0
            st.session_state["ml_labels"] = new_bundle["labels"]
            st.session_state["t4_bundle_override"] = new_bundle

            st.dataframe(new_preds[view_cols], use_container_width=True)

            # Save option for retrained bundle
            st.markdown("### 💾 Save retrained model bundle as .pkl")
            default_name = f"loan_eligibility_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            save_name = st.text_input("File name (will be saved under `models/trained_bundles/`)", value=default_name, key="t4_save_name")
            do_save = st.button("Save Bundle", key="t4_save_btn")
            if do_save:
                dest_dir = os.path.join("models", "trained_bundles")
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, save_name)
                try:
                    joblib.dump(new_bundle, dest_path)
                    st.success(f"Saved retrained bundle to `{dest_path}`")
                except Exception as e:
                    st.error(f"Could not save bundle: {e}")

        except Exception as e:
            st.error(f"Retraining failed: {e}")
            return

    # ----------------------------
    # Explainability & Fit summary
    # ----------------------------
    st.markdown("---")
    st.subheader("🔍 Explainability & Fit Summary")

    sel_id = st.selectbox("Choose Applicant ID", preds["Applicant ID"].tolist(), key="t4_explain_id")
    row_idx = preds.index[preds["Applicant ID"] == sel_id][0]
    row_preds = preds.loc[row_idx]
    loans = row_preds["Eligibility_Pred"]
    if isinstance(loans, str):
        loans = [loans]
    summary = generate_applicant_summary(row_preds.to_dict(), loans)

    st.markdown(f"**Loans:** {', '.join(loans)}")
    st.markdown(
        f"<div style='padding:10px 12px;border-radius:10px;background:#f6f9fc;border:1px solid #dbe4ff;'>"
        f"<b>Fit:</b> {summary['label']} &nbsp; | &nbsp; <b>Score:</b> {summary['score']}/100"
        f"</div>", unsafe_allow_html=True
    )

    # choose label to explain
    if loans and loans != ["Ineligible"]:
        explain_label = st.selectbox("Label to explain (SHAP)", options=loans, index=0, key="t4_label")
    else:
        explain_label = st.selectbox(
            "Pick label to explain",
            options=[lab for lab in ["7(a)","504","Express"] if f"proba_{lab}" in preds.columns],
            key="t4_label2"
        )

    # if user retrained in this session, use override; else base bundle
    active_bundle = st.session_state.get("t4_bundle_override", active_bundle)
    shap_out = _shap_for_label(st.session_state["pred_features_df"].loc[row_idx], active_bundle, explain_label)
    if shap_out is not None:
        sv, feat_cols = shap_out
        # Waterfall
        try:
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(bbox_inches="tight")
        except Exception as e:
            st.warning(f"Waterfall failed: {e}")
        # Top impact
        try:
            v = sv.values[0]
            vser = pd.Series(np.abs(v), index=feat_cols).sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(7,4))
            vser[::-1].plot(kind="barh", ax=ax)
            ax.set_title(f"Top Features ({explain_label})")
            ax.set_xlabel("SHAP |value|")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Top features chart failed: {e}")
    else:
        st.warning("Could not compute SHAP for this label/model.")

    # Strengths / Risks / Recommendation
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Strengths**")
        if summary["strengths"]:
            for s in summary["strengths"][:8]:
                st.markdown(f"- {s}")
        else:
            st.markdown("- None highlighted.")
    with c2:
        st.markdown("**⚠️ Risks**")
        if summary["risks"]:
            for r in summary["risks"][:8]:
                st.markdown(f"- {r}")
        else:
            st.markdown("- No material flags.")
    st.markdown(f"**📌 Recommendation:** {summary['recommendation']}")
