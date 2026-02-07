# loan_app/src/models.py
import os
import json
import hashlib
import joblib
import numpy as np
from typing import Callable, Dict, Optional, Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# xgboost / lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import (
    MODEL_DIR, XGB_MODEL, LGB_MODEL, ENS_MODEL, LBL_ENC,
    XGB_BASE, LGB_BASE
)

# ============================================================
# 🔧 Utilities
# ============================================================

def _safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def _safe_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def _noop_progress(stage: str, pct: float, msg: str):
    # Default progress callback
    pass

def rule_hash(rules: Dict) -> str:
    """
    Deterministic hash over the rules dict (sorted keys).
    Ensures we cache models per unique rule configuration.
    """
    try:
        s = json.dumps(rules, sort_keys=True)
    except TypeError:
        # If not fully JSON-serializable, convert sets/tuples/etc.
        def _default(o):
            if isinstance(o, set):
                return sorted(list(o))
            return str(o)
        s = json.dumps(rules, sort_keys=True, default=_default)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def _cache_dir() -> str:
    d = os.path.join(MODEL_DIR, "runtime_cache")
    os.makedirs(d, exist_ok=True)
    return d

def cached_paths(rulehash: str) -> Dict[str, str]:
    """
    Where we store/load models for a specific rule-hash.
    """
    base = _cache_dir()
    return {
        "xgb": os.path.join(base, f"xgb_{rulehash}.pkl"),
        "lgb": os.path.join(base, f"lgb_{rulehash}.pkl"),
        "ens": os.path.join(base, f"ens_{rulehash}.pkl"),
        "enc": os.path.join(base, f"label_enc_{rulehash}.pkl"),
    }

# ============================================================
# 📥 Loaders
# ============================================================

def load_all_models() -> Tuple[object, object, object, LabelEncoder, Optional[object], Optional[object]]:
    """
    Load primary models + optional base models (for SHAP or meta-only retrains).
    Baseline (non-cached) versions; used at cold start or as fallback.
    """
    xgb = _safe_load(os.path.join(MODEL_DIR, XGB_MODEL))
    lgb = _safe_load(os.path.join(MODEL_DIR, LGB_MODEL))
    ens = _safe_load(os.path.join(MODEL_DIR, ENS_MODEL))
    enc = _safe_load(os.path.join(MODEL_DIR, LBL_ENC))

    # Optional: base models for SHAP / meta-only usage
    xgb_base = _safe_load(os.path.join(MODEL_DIR, XGB_BASE))
    lgb_base = _safe_load(os.path.join(MODEL_DIR, LGB_BASE))

    if xgb is None or lgb is None or ens is None or enc is None:
        missing = [name for name, obj in {
            XGB_MODEL: xgb, LGB_MODEL: lgb, ENS_MODEL: ens, LBL_ENC: enc
        }.items() if obj is None]
        raise RuntimeError(f"Missing required model file(s): {', '.join(missing)}")

    return xgb, lgb, ens, enc, xgb_base, lgb_base

def load_cached_models(rhash: str) -> Optional[Tuple[object, object, object, LabelEncoder]]:
    """
    Load per-rule cached models; returns None if any file is missing.
    """
    paths = cached_paths(rhash)
    xgb = _safe_load(paths["xgb"])
    lgb = _safe_load(paths["lgb"])
    ens = _safe_load(paths["ens"])
    enc = _safe_load(paths["enc"])
    if xgb is None or lgb is None or ens is None or enc is None:
        return None
    return xgb, lgb, ens, enc

def save_cached_models(rhash: str, xgb, lgb, ens, enc: LabelEncoder):
    """
    Persist models for a rule-hash.
    """
    paths = cached_paths(rhash)
    _safe_dump(xgb, paths["xgb"])
    _safe_dump(lgb, paths["lgb"])
    _safe_dump(ens, paths["ens"])
    _safe_dump(enc, paths["enc"])

# ============================================================
# 🧮 Feature Alignment
# ============================================================

def align_features(df, model):
    """
    Align incoming df columns to the model’s expected feature order.
    NOTE: Uses XGBoost booster feature_names (works for LGBM input too).
    """
    expected = model.get_booster().feature_names
    for f in expected:
        if f not in df.columns:
            df[f] = 0
    return df[expected]

# ============================================================
# 🔮 Inference (unchanged)
# ============================================================

def predict_all(df_aligned, xgb, lgb, ens, label_encoder):
    xgb_probs = xgb.predict_proba(df_aligned)
    lgb_probs = lgb.predict_proba(df_aligned)
    meta_input = np.hstack((xgb_probs, lgb_probs))
    ens_probs = ens.predict_proba(meta_input)

    xgb_pred = label_encoder.inverse_transform(np.argmax(xgb_probs, axis=1))
    lgb_pred = label_encoder.inverse_transform(np.argmax(lgb_probs, axis=1))
    ens_pred = label_encoder.inverse_transform(np.argmax(ens_probs, axis=1))
    ens_conf = ens_probs.max(axis=1)
    return xgb_probs, lgb_probs, ens_probs, xgb_pred, lgb_pred, ens_pred, ens_conf

# ============================================================
# 🧪 Training Orchestration (Full / Meta-only / Fast preview)
# ============================================================

def _ensure_label_encoder(y) -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(y)
    return enc

def _split_train_valid(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def _train_xgb(X_tr, y_tr, X_va, y_va, progress: Callable = _noop_progress) -> XGBClassifier:
    progress("xgb", 0.0, "Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
        early_stopping_rounds=50
    )
    progress("xgb", 1.0, "XGBoost trained.")
    return xgb

def _train_lgb(X_tr, y_tr, X_va, y_va, progress: Callable = _noop_progress) -> LGBMClassifier:
    progress("lgb", 0.0, "Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        random_state=42,
        n_jobs=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        verbose=False,
        early_stopping_rounds=60
    )
    progress("lgb", 1.0, "LightGBM trained.")
    return lgb

def _train_meta_from_probs(xgb_probs_tr, lgb_probs_tr, y_tr,
                           xgb_probs_va, lgb_probs_va, y_va,
                           C=1.0, max_iter=200, progress: Callable = _noop_progress) -> LogisticRegression:
    progress("meta", 0.1, "Training Ensemble (meta)...")
    X_tr_meta = np.hstack((xgb_probs_tr, lgb_probs_tr))
    X_va_meta = np.hstack((xgb_probs_va, lgb_probs_va))

    meta = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        n_jobs=None
    )
    meta.fit(X_tr_meta, y_tr)  # We rely on validation for early stopping upstream
    progress("meta", 1.0, "Meta model trained.")
    return meta

def train_all_models(
    X,
    y,
    mode: str = "meta_only",
    base_xgb: Optional[object] = None,
    base_lgb: Optional[object] = None,
    rules: Optional[Dict] = None,
    progress: Callable = _noop_progress,
    sample_frac: float = 1.0,
    seed: int = 42
) -> Tuple[object, object, object, LabelEncoder, str]:
    """
    Train models under three modes:
      - "full":        retrain XGB & LGB from scratch, then meta.
      - "meta_only":   keep XGB & LGB (use provided or baseline), retrain meta on new labels.
      - "fast_preview":META-only on a subsample (quick), leave base models frozen.

    Returns: (xgb, lgb, ens, label_encoder, rulehash)
    Also saves a cached copy under rule-hash.
    """
    # Hash for caching
    rhash = rule_hash(rules or {"_no_rules": True})

    # Option: sampling for speed
    if 0 < sample_frac < 1.0:
        progress("data", 0.05, f"Sampling data for speed: frac={sample_frac:.2f}")
        rs = np.random.RandomState(seed)
        idx = rs.choice(len(X), size=int(len(X) * sample_frac), replace=False)
        X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        y = y[idx]

    # Train/valid split
    X_tr, X_va, y_tr, y_va = _split_train_valid(X, y, test_size=0.2, seed=seed)
    enc = _ensure_label_encoder(y)  # always re-encode for the new label space
    y_tr_enc = enc.transform(y_tr)
    y_va_enc = enc.transform(y_va)

    # Load baselines if needed
    if mode in ("meta_only", "fast_preview"):
        if base_xgb is None or base_lgb is None:
            bx, bl, _, _, _, _ = load_all_models()
            base_xgb = bx
            base_lgb = bl

    if mode == "full":
        # Fit both base models fresh
        progress("full", 0.05, "Beginning FULL retrain...")
        xgb = _train_xgb(X_tr, y_tr_enc, X_va, y_va_enc, progress)
        lgb = _train_lgb(X_tr, y_tr_enc, X_va, y_va_enc, progress)

        # Meta on probs
        xgb_probs_tr = xgb.predict_proba(X_tr)
        lgb_probs_tr = lgb.predict_proba(X_tr)
        xgb_probs_va = xgb.predict_proba(X_va)
        lgb_probs_va = lgb.predict_proba(X_va)
        ens = _train_meta_from_probs(xgb_probs_tr, lgb_probs_tr, y_tr_enc,
                                     xgb_probs_va, lgb_probs_va, y_va_enc,
                                     C=1.0, max_iter=200, progress=progress)

    elif mode == "meta_only":
        # Freeze base models, re-fit meta to align to new label rules
        progress("meta_only", 0.05, "Retraining meta model only...")
        xgb = base_xgb
        lgb = base_lgb

        xgb_probs_tr = xgb.predict_proba(align_features(X_tr.copy(), xgb))
        lgb_probs_tr = lgb.predict_proba(align_features(X_tr.copy(), xgb))
        xgb_probs_va = xgb.predict_proba(align_features(X_va.copy(), xgb))
        lgb_probs_va = lgb.predict_proba(align_features(X_va.copy(), xgb))

        ens = _train_meta_from_probs(xgb_probs_tr, lgb_probs_tr, y_tr_enc,
                                     xgb_probs_va, lgb_probs_va, y_va_enc,
                                     C=1.0, max_iter=150, progress=progress)

    else:  # "fast_preview"
        # Small subsample + quick meta fit
        progress("fast", 0.05, "FAST PREVIEW: quick meta refresh on subsample...")
        xgb = base_xgb
        lgb = base_lgb

        # reduce data further for speed (if large)
        if len(X_tr) > 20000:
            rs = np.random.RandomState(seed)
            idx_tr = rs.choice(len(X_tr), size=20000, replace=False)
            X_tr_s, y_tr_s = (X_tr.iloc[idx_tr] if hasattr(X_tr, "iloc") else X_tr[idx_tr]), y_tr_enc[idx_tr]
        else:
            X_tr_s, y_tr_s = X_tr, y_tr_enc

        if len(X_va) > 5000:
            rs = np.random.RandomState(seed + 1)
            idx_va = rs.choice(len(X_va), size=5000, replace=False)
            X_va_s, y_va_s = (X_va.iloc[idx_va] if hasattr(X_va, "iloc") else X_va[idx_va]), y_va_enc[idx_va]
        else:
            X_va_s, y_va_s = X_va, y_va_enc

        xgb_probs_tr = xgb.predict_proba(align_features(X_tr_s.copy(), xgb))
        lgb_probs_tr = lgb.predict_proba(align_features(X_tr_s.copy(), xgb))
        xgb_probs_va = xgb.predict_proba(align_features(X_va_s.copy(), xgb))
        lgb_probs_va = lgb.predict_proba(align_features(X_va_s.copy(), xgb))

        ens = _train_meta_from_probs(xgb_probs_tr, lgb_probs_tr, y_tr_s,
                                     xgb_probs_va, lgb_probs_va, y_va_s,
                                     C=0.8, max_iter=80, progress=progress)

    # Save cache
    save_cached_models(rhash, xgb, lgb, ens, enc)
    progress("done", 1.0, f"Training complete. Cached as hash={rhash}")
    return xgb, lgb, ens, enc, rhash

# ============================================================
# 🚚 Runtime Loader (rule-aware)
# ============================================================

def load_runtime_models(rules: Optional[Dict] = None):
    """
    Try to load cached models for the given rules; fallback to baseline models.
    Returns: (xgb, lgb, ens, enc, rulehash, from_cache: bool)
    """
    rhash = rule_hash(rules or {"_no_rules": True})
    cached = load_cached_models(rhash)
    if cached:
        xgb, lgb, ens, enc = cached
        return xgb, lgb, ens, enc, rhash, True

    # Fallback to baselines
    xgb, lgb, ens, enc, _, _ = load_all_models()
    return xgb, lgb, ens, enc, rhash, False
