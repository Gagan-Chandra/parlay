# loan_app/src/viz.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def ks_similarity(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    stat, p = ks_2samp(a, b)
    return (1 - stat), p
