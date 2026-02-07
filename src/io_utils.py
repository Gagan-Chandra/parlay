# loan_app/src/io_utils.py
import os
import ast
import pandas as pd
import streamlit as st

# Try to use config.DATA_DIR if available, otherwise fall back to project-root/data
try:
    from .config import DATA_DIR as _CONF_DATA_DIR
except Exception:
    _CONF_DATA_DIR = None

def _project_root() -> str:
    # src/ → project root is the parent of this file's dir's parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def _resolve_data_dir(data_dir: str | None) -> str:
    if data_dir and os.path.isabs(data_dir):
        return data_dir
    if data_dir and not os.path.isabs(data_dir):
        return os.path.abspath(os.path.join(_project_root(), data_dir))
    # use config first, else default to <project>/data
    if _CONF_DATA_DIR:
        return _CONF_DATA_DIR if os.path.isabs(_CONF_DATA_DIR) else os.path.abspath(
            os.path.join(_project_root(), _CONF_DATA_DIR)
        )
    return os.path.abspath(os.path.join(_project_root(), "data"))

def load_local_datasets(given_csv_path: str, synth_xlsx_path: str):
    """Kept for compatibility with older code."""
    old_df = pd.read_excel(given_csv_path)
    new_df = pd.read_excel(synth_xlsx_path)
    old_df = normalize_eligibility(old_df)
    new_df = normalize_eligibility(new_df)
    return old_df, new_df

def load_local_default(filename: str, data_dir: str | None = None) -> pd.DataFrame | None:
    """
    Loads a CSV/XLSX from the project's data directory (absolute + robust).
    Returns a DataFrame or None and logs a Streamlit warning if not found.
    """
    base_dir = _resolve_data_dir(data_dir)
    abs_path = filename if os.path.isabs(filename) else os.path.join(base_dir, filename)

    if not os.path.exists(abs_path):
        st.warning(f"⚠️ File not found: {abs_path}")
        # Helpful diagnostics
        try:
            entries = os.listdir(base_dir)
            st.caption(f"📂 data dir listing ({base_dir}): {entries}")
        except Exception:
            pass
        return None

    try:
        if abs_path.lower().endswith(".csv"):
            df = pd.read_excel(abs_path)
        else:
            df = pd.read_excel(abs_path)
        st.success(f"✅ Loaded: {os.path.basename(abs_path)} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df
    except Exception as e:
        st.error(f"❌ Could not load {abs_path}: {e}")
        return None

def load_training_corpus(
    files: tuple[str, ...] = ("given_data.xlsx", "synthetic_data_generated.xlsx"),
    data_dir: str | None = None
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Attempts to load each file from the (absolute) data dir and concatenates them.
    Returns (df_or_none, loaded_filenames).
    """
    loaded = []
    parts: list[pd.DataFrame] = []
    for fname in files:
        df = load_local_default(fname, data_dir=data_dir)
        if df is not None:
            parts.append(df)
            loaded.append(fname)

    if not parts:
        # Show strong diagnostics once
        base_dir = _resolve_data_dir(data_dir)
        st.error(
            "❌ No training files could be loaded. "
            f"Tried: {files} in {base_dir}"
        )
        try:
            st.caption(f"📂 data dir listing ({base_dir}): {os.listdir(base_dir)}")
        except Exception:
            pass
        return None, []

    df_all = pd.concat(parts, ignore_index=True)
    return df_all, loaded


# =====================================================
# 🧩 Data Normalization Utilities
# =====================================================
def ensure_list_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'Eligibility' column so that all entries are lists.
    Example:
      'Express' → ['Express']
      "['7(a)', '504']" → ['7(a)', '504']
    """
    if "Eligibility" not in df.columns:
        return df

    df = df.copy()
    def _normalize(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return ["Ineligible"]
        x = str(x)
        if x.startswith("[") and x.endswith("]"):
            try:
                parsed = ast.literal_eval(x)
                return parsed if isinstance(parsed, list) else [x]
            except Exception:
                return [x]
        return [x]
    df["Eligibility"] = df["Eligibility"].apply(_normalize)
    return df

def parse_state_from_location(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # find 'location' case-insensitively
    lc_map = {c.lower(): c for c in d.columns}
    if "location" not in lc_map:
        d["State"] = None
        return d
    loc_col = lc_map["location"]

    def _state(v):
        if pd.isna(v): return None
        s = str(v).strip()
        if "," in s:
            return s.split(",")[-1].strip().upper()
        return None

    d["State"] = d[loc_col].apply(_state)
    return d

import pandas as pd

def ensure_unique_applicant_id(df: pd.DataFrame, id_col: str = "Applicant ID") -> pd.DataFrame:
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = range(1, len(df) + 1)
        return df

    # If it exists, check duplicates
    if df[id_col].duplicated().any():
        # make a new unique column
        df[id_col] = (
            df[id_col]
            .astype(str)
            .groupby(df[id_col].astype(str))
            .cumcount()
            .radd(df[id_col].astype(str) + "_")
            .where(df[id_col].duplicated(), df[id_col].astype(str))
        )
        # now it's unique but is string; you can leave it or reindex
    return df

# -----------------------------
# import placed at bottom to avoid circulars
# -----------------------------
from .features import normalize_eligibility
