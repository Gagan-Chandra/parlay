# loan_app/src/config.py
MODEL_DIR = "models"
ASSET_DIR = "assets"

# Primary models used for prediction
XGB_MODEL = "xgb_loan_model_kfold.pkl"
LGB_MODEL = "lightgbm_loan_model.pkl"
ENS_MODEL = "ensemble_meta.pkl"
LBL_ENC   = "label_encoder.pkl"

# Optional “base” models (used for ensemble SHAP averaging if present)
XGB_BASE = "xgb_base.pkl"
LGB_BASE = "lgb_base.pkl"

TOP_FEATURE_IMPORTANCE = {
    "Loan Amount": 0.152209,
    "Business Credit Score": 0.096487,
    "Personal Credit Score": 0.091826,
    "Has_WorkingCapital": 0.088104,
    "Has_Collateral": 0.084317,
    "Has_RealEstateIntent": 0.073358,
    "Loan_to_Income_Ratio": 0.067773,
    "DSCR (latest year)": 0.054475,
    "Annual Revenue (2 years ago)": 0.023057,
    "Avg_DSCR": 0.016425
}

CONFUSION_MATS = {
    "xgb": "xgboost.png",
    "lgb": "lightgbm.png",
    "ens": "Ensemble.png",
}
