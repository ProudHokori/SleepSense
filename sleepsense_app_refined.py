"""
SleepSense – Environment Sleep Score Predictor
A Streamlit web app that trains models from the same pipeline as the notebook,
fetches live sensor readings, and explains predictions with SHAP.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pytz
import mysql.connector
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

from scipy.stats import gaussian_kde

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN CONFIG
# Edit this section to retheme the app. All colors, fonts, and opacity values
# are centralized here — no need to hunt through CSS or chart code.
# ─────────────────────────────────────────────────────────────────────────────
DESIGN_CONFIG = {
    # ── App backgrounds ──────────────────────────────────────────────────────
    "bg_app":        "#1E1E2B",   # main page background
    "bg_sidebar":    "#2B2B43",   # sidebar & score-card / panel background
    "bg_card_deep":  "#252538",   # score-card gradient end / deep panel
    "bg_plot_light": "#f8fafc",   # Plotly plot area (light-bg charts)

    # ── Text scale ───────────────────────────────────────────────────────────
    "text_primary":   "#E6E6E6",  # headings, strong content
    "text_secondary": "#C8C8DC",  # slider labels, table factor text
    "text_muted":     "#9090AA",  # chart tick labels, secondary text
    "text_faint":     "#7A7A9A",  # subtitles, label text
    "text_dim":       "#5A5A7A",  # nav labels, meta text
    "text_deepdim":   "#47476E",  # footer / footnote text
    "text_insight":   "#B8B8CC",  # insight / recommendation body text

    # ── Primary accent — warm gold ────────────────────────────────────────────
    "accent_gold":        "#CBAA7F",  # active tabs, primary button, GOOD score
    "accent_gold_border": "#b09060",  # positive SHAP bar border

    # ── Status / semantic colors ──────────────────────────────────────────────
    "accent_fair":        "#B7A789",  # FAIR score, Humidity snapshot line
    "accent_poor":        "#A17176",  # POOR score, HURTING indicator, neg SHAP
    "accent_poor_border": "#6a4448",  # negative SHAP bar border
    "accent_medium":      "#A797AF",  # Light snapshot line, journey step 2
    "accent_purple":      "#B094A9",  # Sound snapshot line, journey step 3

    # ── Sensor snapshot chart line colors ─────────────────────────────────────
    "snap_temp":  "#CBAA7F",  # Temperature line
    "snap_hum":   "#B88A95",  # Humidity line
    "snap_light": "#856B6A",  # Light line
    "snap_sound": "#A27997",  # Average Sound line
    "snap_pm25":  "#BFA79D",  # PM2.5 line

    # ── Data-synthesis source colors ──────────────────────────────────────────
    "src_real":   "#2d6a4f",  # Real (train) data
    "src_smoter": "#5a7fb5",  # SMOTER synthetic
    "src_knn":    "#e07b54",  # KNN synthetic

    # ── Chart neutral / grid (light-bg Plotly) ────────────────────────────────
    "grid_light":    "#e2e8f0",  # grid lines in light-bg charts
    "neutral_mid":   "#718096",  # muted neutral (No-Aug scenario)
    "neutral_dark":  "#4a5568",  # baseline annotation line
    "neutral_light": "#a0aec0",  # perfect-prediction diagonal line

    # ── Font ──────────────────────────────────────────────────────────────────
    "font_family": "Inter",

    # ── Alpha values ──────────────────────────────────────────────────────────
    "alpha_kde_real":  0.35,  # KDE fill — real training data
    "alpha_kde_synth": 0.20,  # KDE fill — SMOTER / KNN synthetic
    "alpha_plot_dark": 0.02,  # Plotly dark theme plot_bgcolor tint
}

_T = DESIGN_CONFIG  # short alias used throughout this file


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color + alpha into a CSS rgba() string."""
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


# Pre-computed rgba tokens — derived from DESIGN_CONFIG, used in CSS and HTML
_GOLD_07 = _rgba(_T["accent_gold"], 0.07)
_GOLD_08 = _rgba(_T["accent_gold"], 0.08)
_GOLD_12 = _rgba(_T["accent_gold"], 0.12)
_GOLD_18 = _rgba(_T["accent_gold"], 0.18)
_GOLD_22 = _rgba(_T["accent_gold"], 0.22)
_GOLD_28 = _rgba(_T["accent_gold"], 0.28)
_GOLD_40 = _rgba(_T["accent_gold"], 0.40)
_GOLD_45 = _rgba(_T["accent_gold"], 0.45)
_POOR_10 = _rgba(_T["accent_poor"], 0.10)
_POOR_14 = _rgba(_T["accent_poor"], 0.14)
_POOR_28 = _rgba(_T["accent_poor"], 0.28)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SleepSense Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={_T["font_family"]}:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: '{_T["font_family"]}', sans-serif !important; }}

.stApp {{ background-color: {_T["bg_app"]}; }}

[data-testid="stSidebar"] {{ background-color: {_T["bg_sidebar"]} !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important; }}

.block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}

h1, h2, h3, h4 {{ color: {_T["text_primary"]} !important; font-weight: 600;
                  letter-spacing: -0.02em; }}

/* Score card */
.zen-score-card {{
    background: linear-gradient(160deg, {_T["bg_sidebar"]} 0%, {_T["bg_card_deep"]} 100%);
    border: 1px solid {_GOLD_18};
    border-radius: 16px; padding: 32px 24px; text-align: center;
}}
.zen-score-label {{ color: {_T["text_faint"]}; font-size: 12px; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 8px; }}
.zen-score-number {{ font-size: 80px; font-weight: 700; line-height: 1;
    margin: 8px 0; letter-spacing: -4px; }}
.zen-score-grade {{ font-size: 12px; letter-spacing: 3px; text-transform: uppercase;
    font-weight: 500; margin-top: 6px; opacity: 0.85; }}
.zen-score-bar-bg {{ background: rgba(255,255,255,0.07); border-radius: 4px;
    height: 4px; margin-top: 16px; overflow: hidden; }}
.zen-score-bar-fill {{ height: 100%; border-radius: 4px; }}
.zen-score-synced {{ color: {_T["text_muted"]}; font-size: 12px; margin-top: 10px;
    letter-spacing: 0.3px; }}

/* Dividers */
hr {{ border-color: rgba(255,255,255,0.06) !important; }}

/* Buttons */
.stButton > button {{
    background: {_GOLD_12}; color: {_T["accent_gold"]};
    border: 1px solid {_GOLD_28}; border-radius: 8px;
    font-size: 13px; font-weight: 500; letter-spacing: 0.3px;
    transition: all 0.15s;
}}
.stButton > button:hover {{
    background: {_GOLD_22}; border-color: {_GOLD_45};
}}
button[kind="primaryFormSubmit"], .stButton > button[kind="primary"] {{
    background: {_T["accent_gold"]} !important; color: {_T["bg_sidebar"]} !important;
    border: none !important; font-weight: 600 !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{ color: {_T["text_dim"]} !important; font-size: 13px; }}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {_T["accent_gold"]} !important; border-bottom-color: {_T["accent_gold"]} !important;
}}

/* Expander */
details summary {{ color: {_T["text_muted"]} !important; font-size: 13px; }}

/* Alerts */
.stAlert {{ border-radius: 10px; }}

/* Dataframe */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}

/* Metrics */
div[data-testid="stMetric"] {{ background: {_T["bg_sidebar"]}; border-radius: 10px;
    padding: 12px; border: 1px solid rgba(255,255,255,0.06); }}
div[data-testid="stMetric"] label {{ color: {_T["text_faint"]} !important; font-size: 11px; }}

/* Sidebar radio nav */
[data-testid="stRadio"] label {{ font-size: 13px !important; color: {_T["text_muted"]} !important; }}
[data-testid="stRadio"] label:has(input:checked) {{ color: {_T["text_primary"]} !important; }}

/* Sliders */
[data-testid="stSlider"] label {{ color: {_T["text_secondary"]} !important; font-size: 13px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
KEY_FILE       = "sleepsense-491911-4c989b81c85d.json"
SENSOR_API     = "https://iot.cpe.ku.ac.th/red/b6410545576/api/sleepsense"
SHEET_CHECKIN  = "Sleep Sense Morning Check-in (Responses)"
SHEET_RESEARCH = "Sleep Sense Secondary Data"
TH_TZ          = pytz.timezone("Asia/Bangkok")
ALL_FACTORS    = ["temp_c", "hum_pct", "light_lux", "snd_avg", "snd_evt", "pm1", "pm25", "pm10"]

FACTOR_META = {
    "temp_c":    {"label": "Temperature",   "unit": "°C",     "min": 15.0, "max": 40.0, "step": 0.1,  "default": 25.0,  "ideal": "18–22 °C"},
    "hum_pct":   {"label": "Humidity",      "unit": "%",      "min": 10.0, "max": 95.0, "step": 1.0,  "default": 50.0,  "ideal": "40–60 %"},
    "light_lux": {"label": "Light",         "unit": "lux",    "min":  0.0, "max": 300.0,"step": 0.1,  "default":  5.0,  "ideal": "< 5 lux"},
    "snd_avg":   {"label": "Avg Sound",     "unit": "dB",     "min": 15.0, "max": 85.0, "step": 0.5,  "default": 35.0,  "ideal": "< 30 dB"},
    "snd_evt":   {"label": "Sound Events",  "unit": "events", "min":  0.0, "max": 30.0, "step": 1.0,  "default":  2.0,  "ideal": "0 events"},
    "pm1":       {"label": "PM1",           "unit": "µg/m³",  "min":  0.0, "max": 120.0,"step": 0.5,  "default": 15.0,  "ideal": "< 5 µg/m³"},
    "pm25":      {"label": "PM2.5",         "unit": "µg/m³",  "min":  0.0, "max": 160.0,"step": 0.5,  "default": 20.0,  "ideal": "< 12 µg/m³"},
    "pm10":      {"label": "PM10",          "unit": "µg/m³",  "min":  0.0, "max": 200.0,"step": 1.0,  "default": 25.0,  "ideal": "< 20 µg/m³"},
}

# Embedded research rubric (from Google Sheets "rubric" worksheet — static reference data)
RESEARCH_RUBRIC_ROWS = [
    {"Factor": "Tempurature (°C)",  "min": 18.00, "max": 20.00},
    {"Factor": "Humidity (%)",      "min": 30.00, "max": 50.00},
    {"Factor": "Light (Lux)",       "min":  0.00, "max":  1.00},
    {"Factor": "Noise (dB)",        "min": 20.00, "max": 40.00},
    {"Factor": "Sound Event (Time)","min":  0.00, "max":  0.00},
    {"Factor": "PM1 (µg/m³)",       "min":  0.00, "max":  5.00},
    {"Factor": "PM2.5 (µg/m³)",     "min":  0.00, "max": 20.00},
    {"Factor": "PM10 (µg/m³)",      "min":  0.00, "max": 30.00},
    {"Factor": "Tempurature (°C)",  "min": 15.00, "max": 19.00},
    {"Factor": "Tempurature (°C)",  "min": 16.00, "max": 18.00},
    {"Factor": "Humidity (%)",      "min": 30.00, "max": 60.00},
    {"Factor": "Tempurature (°C)",  "min": 15.60, "max": 19.00},
    {"Factor": "Tempurature (°C)",  "min": 17.00, "max": 28.00},
    {"Factor": "Humidity (%)",      "min": 40.00, "max": 60.00},
    {"Factor": "Tempurature (°C)",  "min": 15.60, "max": 20.00},
    {"Factor": "Tempurature (°C)",  "min": 15.60, "max": 18.30},
    {"Factor": "Light (Lux)",       "min":  0.00, "max":  5.00},
    {"Factor": "Tempurature (°C)",  "min": 15.60, "max": 19.40},
    {"Factor": "Humidity (%)",      "min": 30.00, "max": 50.00},
    {"Factor": "Humidity (%)",      "min": 40.00, "max": 60.00},
    {"Factor": "Humidity (%)",      "min": 30.00, "max": 60.00},
    {"Factor": "Tempurature (°C)",  "min": 15.60, "max": 18.30},
    {"Factor": "Tempurature (°C)",  "min": 15.60, "max": 20.00},
    {"Factor": "Tempurature (°C)",  "min": 18.33, "max": 19.44},
    {"Factor": "Noise (dB)",        "min":  0.00, "max": 30.00},
    {"Factor": "Noise (dB)",        "min":  0.00, "max": 45.00},
    {"Factor": "Noise (dB)",        "min":  0.00, "max": 20.00},
    {"Factor": "Light (Lux)",       "min":  0.00, "max":  1.00},
    {"Factor": "Humidity (%)",      "min": 30.00, "max": 50.00},
    {"Factor": "Humidity (%)",      "min": 40.00, "max": 60.00},
    {"Factor": "PM2.5 (µg/m³)",     "min":  0.00, "max": 12.00},
]

FACTOR_MAP = {
    "Tempurature (°C)":   "temp_c",
    "Humidity (%)":       "hum_pct",
    "Light (Lux)":        "light_lux",
    "Noise (dB)":         "snd_avg",
    "Sound Event (Time)": "snd_evt",
    "PM1 (µg/m³)":        "pm1",
    "PM2.5 (µg/m³)":      "pm25",
    "PM10 (µg/m³)":       "pm10",
}

MODEL_REGISTRY = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge  (α=1)":      lambda: Ridge(alpha=1.0),
    "Ridge  (α=10)":     lambda: Ridge(alpha=10.0),
    "Lasso  (α=0.5)":    lambda: Lasso(alpha=0.5, max_iter=5000),
    "Random Forest":     lambda: RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42),
}

# Maps MODEL_REGISTRY keys → sleep_score_history column names.
# "Ridge  (α=10)" is intentionally excluded — no matching DB column.
DB_SCORE_COLS = {
    "Linear Regression": "linear_regression_score",
    "Ridge  (α=1)":      "ridge1_score",
    "Lasso  (α=0.5)":    "lasso_score",
    "Random Forest":     "random_forest_score",
    "Gradient Boosting": "gradient_boosting_score",
}

SNAPSHOT_COLS = [
    ("temp_c",    "Temperature (°C)",  _T["snap_temp"]),
    ("hum_pct",   "Humidity (%)",      _T["snap_hum"]),
    ("light_lux", "Light (lux)",       _T["snap_light"]),
    ("snd_avg",   "Sound (dB)",        _T["snap_sound"]),
    ("pm25",      "PM2.5 (µg/m³)",     _T["snap_pm25"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING CLASSES
# ─────────────────────────────────────────────────────────────────────────────
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.05, upper_q=0.95):
        self.lower_q, self.upper_q = lower_q, upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower_q, axis=0)
        self.hi_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_prep():
    log_block = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p",  FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale",  RobustScaler()),
    ])
    comfort_block = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer()),
        ("scale",  RobustScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("log_skew", log_block,     ["light_lux", "snd_evt", "pm1", "pm25", "pm10"]),
            ("comfort",  comfort_block, ["temp_c", "hum_pct", "snd_avg"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA SYNTHESIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _impute(X_df: pd.DataFrame) -> np.ndarray:
    """Median-impute any NaN values — required before neighbor search."""
    return SimpleImputer(strategy="median").fit_transform(X_df.values.astype(float))


def synth_smoter(X_df: pd.DataFrame, y_s: pd.Series,
                 n_synth: int, k_neighbors: int = 5, seed: int = 42) -> tuple:
    """
    SMOTER (Torgo et al. 2013) — regression analog of SMOTE.
    Interpolates X and y with the SAME lambda so the label stays consistent.
    All points lie on line segments between real sessions (no extrapolation).
    """
    rng   = np.random.default_rng(seed)
    X_imp = _impute(X_df)
    y     = y_s.values.astype(float)
    n, k  = len(X_imp), min(k_neighbors, len(X_imp) - 1)

    X_sc     = RobustScaler().fit_transform(X_imp)
    _, nbrs  = NearestNeighbors(n_neighbors=k + 1).fit(X_sc).kneighbors(X_sc)

    i_idx = rng.integers(0, n, size=n_synth)
    j_col = rng.integers(1, k + 1, size=n_synth)
    j_idx = nbrs[i_idx, j_col]
    lam   = rng.uniform(0, 1, size=n_synth)

    X_syn = X_imp[i_idx] + lam[:, None] * (X_imp[j_idx] - X_imp[i_idx])
    y_syn = y[i_idx]     + lam           * (y[j_idx]     - y[i_idx])

    return pd.DataFrame(X_syn, columns=X_df.columns), pd.Series(y_syn, name=y_s.name)


def synth_knn(X_df: pd.DataFrame, y_s: pd.Series,
              n_synth: int, k_neighbors: int = 5, seed: int = 42) -> tuple:
    """
    KNN Regression Synthesis.
    X' = Dirichlet-weighted convex combination of seed + k neighbours.
    y' = distance-weighted KNN regression on real data (decoupled from X geometry).
    """
    rng   = np.random.default_rng(seed)
    X_imp = _impute(X_df)
    y     = y_s.values.astype(float)
    n, k  = len(X_imp), min(k_neighbors, len(X_imp) - 1)

    scaler   = RobustScaler().fit(X_imp)
    X_sc     = scaler.transform(X_imp)
    _, nbrs  = NearestNeighbors(n_neighbors=k + 1).fit(X_sc).kneighbors(X_sc)
    knn_reg  = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(X_sc, y)

    i_idx = rng.integers(0, n, size=n_synth)
    X_syn = np.zeros((n_synth, X_imp.shape[1]))
    for t in range(n_synth):
        pts     = np.vstack([X_imp[i_idx[t]], X_imp[nbrs[i_idx[t], 1:k + 1]]])
        weights = rng.dirichlet(np.ones(k + 1))
        X_syn[t] = weights @ pts

    y_syn = np.clip(knn_reg.predict(scaler.transform(X_syn)), y.min(), y.max())
    return pd.DataFrame(X_syn, columns=X_df.columns), pd.Series(y_syn, name=y_s.name)


def _augment(X_real, y_real, fn, n_synth):
    X_syn, y_syn = fn(X_real, y_real, n_synth)
    return (
        pd.concat([X_real, X_syn], ignore_index=True),
        pd.concat([y_real, y_syn], ignore_index=True),
    )


SYNTH_SCENARIOS = ["No Augmentation", "SMOTER ×3", "SMOTER ×5", "KNN Synth ×3", "KNN Synth ×5"]

SCENARIO_PALETTE = {
    "No Augmentation": _T["neutral_mid"],
    "SMOTER ×3":       _T["src_smoter"],
    "SMOTER ×5":       "#2b6cb0",
    "KNN Synth ×3":    _T["src_knn"],
    "KNN Synth ×5":    "#c05621",
}

MODEL_PALETTE = {
    "Linear Regression": _T["src_smoter"],
    "Ridge  (α=1)":      _T["src_knn"],
    "Ridge  (α=10)":     "#e0b554",
    "Lasso  (α=0.5)":    "#9b72cf",
    "Random Forest":     "#50a878",
    "Gradient Boosting": "#d05c87",
}


@st.cache_data(show_spinner=False)
def run_synthesis_experiment(_df_session: pd.DataFrame):
    """
    Run all 5 scenarios × 6 models on a held-out real test set.
    Returns results dict and per-scenario synthetic sample DataFrames for viz.
    """
    X = _df_session[ALL_FACTORS]
    y = _df_session["Sleep_Score"]
    groups = _df_session["Session ID"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_raw = X.iloc[train_idx].reset_index(drop=True)
    y_train_raw = y.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X.iloc[test_idx].reset_index(drop=True)
    y_test_raw  = y.iloc[test_idx].reset_index(drop=True)

    n_base = len(X_train_raw)

    def make_scenario(name):
        if name == "No Augmentation":
            return X_train_raw.copy(), y_train_raw.copy()
        method = synth_smoter if name.startswith("SMOTER") else synth_knn
        factor = 3 if "×3" in name else 5
        return _augment(X_train_raw, y_train_raw, method, n_base * factor)

    scenarios_data = {s: make_scenario(s) for s in SYNTH_SCENARIOS}

    # Store sample synthetic rows for distribution viz (×3 scenarios only)
    synth_samples = {
        "Real (train)": X_train_raw,
        "SMOTER":  synth_smoter(X_train_raw, y_train_raw, n_base * 3)[0],
        "KNN Synth": synth_knn(X_train_raw, y_train_raw, n_base * 3)[0],
    }

    results = {}
    fitted_pipes_lab = {}
    predictions_lab  = {}

    for scen_name, (X_aug, y_aug) in scenarios_data.items():
        _prep = build_prep()
        X_tr_t = _prep.fit_transform(X_aug)
        X_te_t = _prep.transform(X_test_raw)

        scen_r = {}
        for model_name, factory in MODEL_REGISTRY.items():
            m = factory()
            m.fit(X_tr_t, y_aug)
            y_pred = m.predict(X_te_t)
            fitted_pipes_lab[(scen_name, model_name)] = (_prep, m)
            predictions_lab[(scen_name, model_name)]  = y_pred
            scen_r[model_name] = {
                "MAE":  mean_absolute_error(y_test_raw, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test_raw, y_pred))),
                "R²":   r2_score(y_test_raw, y_pred),
            }
        results[scen_name] = scen_r

    return results, synth_samples, y_test_raw, predictions_lab, n_base


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_sensor(start=None, end=None):
    try:
        if start and end:
            r = requests.get(f"{SENSOR_API}/range",
                             params={"start": start, "end": end}, timeout=12)
        else:
            r = requests.get(SENSOR_API, timeout=12)
        if r.status_code == 200:
            df = pd.DataFrame(r.json().get("data", []))
            if not df.empty:
                df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_sensor():
    return _fetch_sensor()


def save_prediction_to_db(sensor_data: dict, all_scores: dict) -> None:
    """Insert one row of sensor readings + all model scores into sleep_score_history."""
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        connect_timeout=5,
    )
    sql = """
        INSERT INTO sleep_score_history
            (timestamp, temp_c, hum_pct, light_lux, snd_avg, snd_peak, snd_var, snd_evt,
             pm1, pm25, pm10,
             linear_regression_score, ridge1_score, lasso_score,
             random_forest_score, gradient_boosting_score)
        VALUES
            (%(timestamp)s, %(temp_c)s, %(hum_pct)s, %(light_lux)s,
             %(snd_avg)s, %(snd_peak)s, %(snd_var)s, %(snd_evt)s,
             %(pm1)s, %(pm25)s, %(pm10)s,
             %(linear_regression_score)s, %(ridge1_score)s, %(lasso_score)s,
             %(random_forest_score)s, %(gradient_boosting_score)s)
    """
    def _to_py(v):
        """Convert numpy scalar → native Python float/int so MySQL connector accepts it."""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    row = {
        "timestamp": sensor_data.get("recorded_at"),
        "temp_c":    _to_py(sensor_data.get("temp_c")),
        "hum_pct":   _to_py(sensor_data.get("hum_pct")),
        "light_lux": _to_py(sensor_data.get("light_lux")),
        "snd_avg":   _to_py(sensor_data.get("snd_avg")),
        "snd_peak":  _to_py(sensor_data.get("snd_peak")),
        "snd_var":   _to_py(sensor_data.get("snd_var")),
        "snd_evt":   _to_py(sensor_data.get("snd_evt")),
        "pm1":       _to_py(sensor_data.get("pm1")),
        "pm25":      _to_py(sensor_data.get("pm25")),
        "pm10":      _to_py(sensor_data.get("pm10")),
    }
    for model_name, col in DB_SCORE_COLS.items():
        row[col] = _to_py(all_scores.get(model_name))
    cursor = conn.cursor()
    cursor.execute(sql, row)
    conn.commit()
    cursor.close()
    conn.close()


def _load_checkin():
    try:
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        creds  = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE, scope)
        client = gspread.authorize(creds)
        sheet  = client.open(SHEET_CHECKIN).worksheet("Form Responses 1")
        return pd.DataFrame(sheet.get_all_records())
    except Exception:
        return pd.DataFrame()


def _expand_research_rubric():
    """Build training rows from the embedded research rubric (ideal ranges → score 100)."""
    target_cols = ["Session ID", "temp_c", "hum_pct", "light_lux",
                   "snd_avg", "snd_evt", "pm1", "pm25", "pm10", "Sleep_Score"]
    rows = []
    for i, entry in enumerate(RESEARCH_RUBRIC_ROWS):
        col = FACTOR_MAP.get(entry["Factor"])
        if not col:
            continue
        vmin, vmax = entry["min"], entry["max"]
        for val in np.arange(int(vmin), int(vmax) + 1):
            row = {c: np.nan for c in target_cols}
            row["Session ID"]  = f"R-{i+1:02d}"
            row["Sleep_Score"] = 100.0
            row[col]           = float(val)
            rows.append(row)
    return pd.DataFrame(rows, columns=target_cols)


def _build_sensor_sessions(df_checkin):
    """Match check-in sleep windows to sensor readings (same logic as notebook)."""
    df = df_checkin.copy()
    df.columns = [c.strip() for c in df.columns]
    try:
        df["Start Time"] = pd.to_datetime(df["Sleep Start Date"] + " " + df["Sleep Start Time"])
        df["End Time"]   = pd.to_datetime(df["Wake-Up Date"]     + " " + df["Wake-Up Time"])
    except Exception:
        return pd.DataFrame()

    parts = []
    for idx, session in df.iterrows():
        t1 = session["Start Time"].strftime("%Y-%m-%d %H:%M:%S")
        t2 = session["End Time"].strftime("%Y-%m-%d %H:%M:%S")
        df_raw = _fetch_sensor(start=t1, end=t2)
        if df_raw.empty:
            continue
        df_raw = df_raw.sort_values("recorded_at")
        df_raw["Session ID"]  = f"S-{idx+1:02d}"
        df_raw["Sleep_Score"] = session.get("Sleep Score", np.nan)
        n = len(df_raw)
        if n > 5:
            df_raw = df_raw.iloc[np.linspace(0, n - 1, 5).astype(int)]
        keep = [c for c in ["Session ID", "temp_c", "hum_pct", "light_lux",
                             "snd_avg", "snd_evt", "pm1", "pm25", "pm10", "Sleep_Score"]
                if c in df_raw.columns]
        parts.append(df_raw[keep])

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN (cached for the session lifetime)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on your sleep data…")
def load_and_train():
    # Research data (always available)
    df_research = _expand_research_rubric()

    # Real check-in sessions (optional — gracefully skipped if Sheets fails)
    df_checkin = _load_checkin()
    df_sensor_sessions = pd.DataFrame()
    if not df_checkin.empty:
        df_sensor_sessions = _build_sensor_sessions(df_checkin)

    parts = [p for p in [df_sensor_sessions, df_research] if not p.empty]
    df_combined = pd.concat(parts, ignore_index=True)

    numeric_cols = ALL_FACTORS + ["Sleep_Score"]
    df_combined[numeric_cols] = df_combined[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Session-level aggregation
    agg = {c: "median" for c in ALL_FACTORS}
    agg["Sleep_Score"] = "median"
    df_session = (
        df_combined.groupby("Session ID", as_index=False).agg(agg)
        .dropna(subset=["Sleep_Score"])
    )

    if len(df_session) < 3:
        return None, None, None, None, None

    # Fit preprocessing on ALL session data
    prep = build_prep()
    X = df_session[ALL_FACTORS]
    y = df_session["Sleep_Score"]
    prep.fit(X, y)
    feat_names = list(prep.get_feature_names_out())
    X_t = prep.transform(X)

    # Train every model
    trained = {}
    for name, factory in MODEL_REGISTRY.items():
        est = factory()
        est.fit(X_t, y)
        # Build a full pipe (cloned prep) for single-row prediction
        pipe = Pipeline([("prep", clone(prep)), ("model", clone(est))])
        pipe.fit(X, y)
        trained[name] = {"estimator": est, "pipe": pipe}

    # Build SHAP explainers
    explainers = {}
    for name, bundle in trained.items():
        est_bg = bundle["estimator"]
        if hasattr(est_bg, "estimators_"):                          # tree ensemble
            explainers[name] = shap.TreeExplainer(est_bg)
        elif hasattr(est_bg, "coef_"):                              # linear
            explainers[name] = shap.LinearExplainer(est_bg, X_t)
        else:
            bg = shap.kmeans(X_t, min(10, len(X_t)))
            explainers[name] = shap.KernelExplainer(est_bg.predict, bg)

    return trained, explainers, feat_names, df_session, X_t


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION + SHAP
# ─────────────────────────────────────────────────────────────────────────────
def predict_and_explain(pipe, explainer, input_values, feat_names):
    input_df  = pd.DataFrame([input_values], columns=ALL_FACTORS)
    raw       = pipe.predict(input_df)[0]
    predicted = float(np.clip(raw, 0, 100))

    X_in   = pipe.named_steps["prep"].transform(input_df)
    sv_raw = explainer.shap_values(X_in)
    sv     = np.array(sv_raw[0] if isinstance(sv_raw, list) else sv_raw).flatten()

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(np.array(ev).flatten()[0])
    else:
        ev = float(ev)

    return predicted, sv, ev


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def make_snapshot_chart(df_sensor: pd.DataFrame) -> go.Figure:
    df = df_sensor.copy()
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    df = df.sort_values("recorded_at").tail(40)
    df["time_th"] = df["recorded_at"].dt.tz_convert(TH_TZ)

    available = [(c, lbl, clr) for c, lbl, clr in SNAPSHOT_COLS if c in df.columns]
    n = len(available)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[lbl for _, lbl, _ in available],
        shared_yaxes=False,
    )
    for col_i, (col, _, color) in enumerate(available, start=1):
        vals = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=df["time_th"], y=vals,
                mode="lines",
                line=dict(color=color, width=1.5),
                name=col,
                showlegend=False,
                hovertemplate="%{y:.1f}<br>%{x|%H:%M}<extra></extra>",
            ),
            row=1, col=col_i,
        )
        if not vals.isna().all():
            last_val = vals.dropna().iloc[-1]
            fig.add_annotation(
                x=df["time_th"].iloc[-1], y=last_val,
                text=f"<b>{last_val:.1f}</b>",
                showarrow=False, xanchor="left", xshift=8,
                font=dict(color=color, size=11),
                row=1, col=col_i,
            )

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=f"rgba(255,255,255,{_T['alpha_plot_dark']})",
        font=dict(size=11, color=_T["text_muted"]),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     zeroline=False, tickfont=dict(color=_T["text_muted"], size=10))
    for ann in fig.layout.annotations:
        ann.font.color = _T["text_muted"]
        ann.font.size  = 11
    return fig


def make_shap_chart(sv, feat_names, predicted, base_value, model_name) -> go.Figure:
    labels = [FACTOR_META.get(f, {}).get("label", f) for f in feat_names]
    order  = np.argsort(sv)
    sv_ord = sv[order]
    lb_ord = [labels[i] for i in order]
    fn_ord = [feat_names[i] for i in order]

    colors  = [_T["accent_poor"] if v < 0 else _T["accent_gold"] for v in sv_ord]
    borders = [_T["accent_poor_border"] if v < 0 else _T["accent_gold_border"] for v in sv_ord]

    text_labels = [
        ("hurting  " if v < 0 else "helping  ") + f"{v:+.1f}"
        for v in sv_ord
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=lb_ord,
        x=sv_ord,
        orientation="h",
        marker=dict(color=colors, line=dict(color=borders, width=1)),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11, color=_T["text_faint"]),
        hovertemplate="<b>%{y}</b><br>Impact: %{x:+.2f} pts<extra></extra>",
        cliponaxis=False,
    ))

    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.15)", width=1))

    x_range = max(abs(sv_ord.min()), abs(sv_ord.max()))
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Factor Impact on Sleep Score</b><br>"
                f"<span style='font-size:12px;color:{_T['text_faint']};'>"
                f"Base: {base_value:.0f} pts → Predicted: <b>{predicted:.0f}</b> / 100"
                f"</span>"
            ),
            font=dict(size=14, color=_T["text_primary"]), x=0,
        ),
        xaxis=dict(
            title=dict(text="Score impact (SHAP, points)",
                       font=dict(color=_T["text_faint"], size=11)),
            range=[-(x_range * 1.7), x_range * 1.7],
            zeroline=False, showgrid=False,
            tickfont=dict(size=11, color=_T["text_muted"]),
        ),
        yaxis=dict(tickfont=dict(size=12, color=_T["text_secondary"]), automargin=True),
        height=max(300, len(feat_names) * 52),
        margin=dict(l=0, r=140, t=72, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS LAB CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def make_kde_chart(synth_samples: dict, feature: str) -> go.Figure:
    """Overlaid KDE curves: Real training data vs SMOTER vs KNN synthetic."""
    meta  = FACTOR_META.get(feature, {})
    label = meta.get("label", feature)
    unit  = meta.get("unit", "")

    colors = {
        "Real (train)": (_T["src_real"],   _T["alpha_kde_real"]),
        "SMOTER":       (_T["src_smoter"], _T["alpha_kde_synth"]),
        "KNN Synth":    (_T["src_knn"],    _T["alpha_kde_synth"]),
    }
    dash_styles = {
        "Real (train)": "solid",
        "SMOTER":       "dash",
        "KNN Synth":    "dot",
    }

    fig = go.Figure()
    for source_name, X_df in synth_samples.items():
        col = feature if feature in X_df.columns else None
        if col is None:
            continue
        vals = pd.to_numeric(X_df[col], errors="coerce").dropna().values
        if len(vals) < 3:
            continue
        color, fill_alpha = colors.get(source_name, ("#718096", 0.1))
        try:
            kde    = gaussian_kde(vals, bw_method="scott")
            x_grid = np.linspace(vals.min(), vals.max(), 300)
            y_grid = kde(x_grid)
        except Exception:
            continue

        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid,
            mode="lines",
            name=source_name,
            line=dict(color=color, width=2.5, dash=dash_styles.get(source_name, "solid")),
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},{fill_alpha})",
            hovertemplate=f"{source_name}<br>{label}: %{{x:.2f}} {unit}<extra></extra>",
        ))
        # Annotate mean
        fig.add_vline(
            x=float(vals.mean()), line=dict(color=color, width=1.2, dash="longdash"),
            annotation_text=f"μ={vals.mean():.1f}",
            annotation_position="top",
            annotation_font=dict(color=color, size=10),
        )

    ideal = meta.get("ideal", "")
    fig.update_layout(
        title=dict(
            text=f"<b>{label}</b>  ({unit})"
                 + (f"  <span style='color:{_T['neutral_mid']};font-size:12px;'>Ideal: {ideal}</span>" if ideal else ""),
            font=dict(size=14), x=0,
        ),
        xaxis=dict(title=f"{label} ({unit})", showgrid=True, gridcolor=_T["grid_light"]),
        yaxis=dict(title="Density", showgrid=True, gridcolor=_T["grid_light"], zeroline=False),
        height=320,
        margin=dict(l=0, r=0, t=56, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_T["bg_plot_light"],
        legend=dict(orientation="h", y=-0.25, x=0),
        hovermode="x unified",
    )
    return fig


def make_metrics_heatmap(results: dict, metric: str) -> go.Figure:
    """Interactive heatmap: models (rows) × scenarios (cols), cell = metric value."""
    model_names = list(MODEL_REGISTRY.keys())
    scen_names  = SYNTH_SCENARIOS
    lower_better = metric != "R²"

    z = [[results[s][m][metric] for s in scen_names] for m in model_names]
    z_text = [[f"{v:.3f}" for v in row] for row in z]

    colorscale = "RdYlGn_r" if lower_better else "RdYlGn"

    # Highlight best cell per model
    annotations = []
    for i, row in enumerate(z):
        best_j = int(np.argmin(row) if lower_better else np.argmax(row))
        annotations.append(dict(
            x=best_j, y=i,
            text=f"<b>★ {z[i][best_j]:.3f}</b>",
            showarrow=False,
            font=dict(size=11, color="white"),
            xref="x", yref="y",
        ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=scen_names,
        y=model_names,
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=colorscale,
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> + %{x}<br>" + metric + ": <b>%{z:.3f}</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{metric}  {'↓ lower = better' if lower_better else '↑ higher = better'}</b>"
                 f"  <span style='font-size:12px;color:{_T['neutral_mid']};'>— all 30 (model × scenario) combinations</span>",
            font=dict(size=13), x=0,
        ),
        height=340,
        margin=dict(l=0, r=0, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-25, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def make_delta_mae_chart(results: dict) -> go.Figure:
    """Line chart: ΔMAE per model vs augmentation scenario (baseline = No Augmentation)."""
    model_names = list(MODEL_REGISTRY.keys())
    scen_order  = [s for s in SYNTH_SCENARIOS if s != "No Augmentation"]

    fig = go.Figure()
    for model_name in model_names:
        base_mae = results["No Augmentation"][model_name]["MAE"]
        deltas   = [results[s][model_name]["MAE"] - base_mae for s in scen_order]
        color    = MODEL_PALETTE.get(model_name, "#718096")
        fig.add_trace(go.Scatter(
            x=scen_order, y=deltas,
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color, symbol="circle"),
            hovertemplate=(
                f"<b>{model_name}</b><br>Scenario: %{{x}}<br>"
                "ΔMAE: <b>%{y:+.3f}</b><extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line=dict(color=_T["neutral_dark"], width=1.5, dash="dash"),
                  annotation_text="Baseline (no augmentation)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color=_T["neutral_dark"]))
    fig.update_layout(
        title=dict(
            text=f"<b>ΔMAE vs No-Augmentation Baseline</b>"
                 f"  <span style='font-size:12px;color:{_T['neutral_mid']};'>points below 0 = synthesis helped</span>",
            font=dict(size=13), x=0,
        ),
        xaxis=dict(title="Augmentation Scenario", tickfont=dict(size=11)),
        yaxis=dict(title="ΔMAE (pts)", showgrid=True, gridcolor=_T["grid_light"], zeroline=False),
        height=360,
        margin=dict(l=0, r=0, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_T["bg_plot_light"],
        legend=dict(orientation="h", y=-0.3, x=0),
        hovermode="x unified",
    )
    return fig


def make_synth_scatter_chart(synth_samples: dict,
                              x_feat: str, y_feat: str,
                              color_feat: str | None = None) -> go.Figure:
    """
    2-D scatter of real training data vs SMOTER vs KNN synthetic points.
    Optionally colour points by a third feature (bubble colour scale).
    """
    SOURCE_STYLE = {
        "Real (train)": {"color": _T["src_real"],   "symbol": "circle",  "size": 11, "opacity": 0.85},
        "SMOTER":       {"color": _T["src_smoter"], "symbol": "diamond", "size":  9, "opacity": 0.70},
        "KNN Synth":    {"color": _T["src_knn"],    "symbol": "cross",   "size":  9, "opacity": 0.70},
    }
    x_meta = FACTOR_META.get(x_feat, {})
    y_meta = FACTOR_META.get(y_feat, {})

    fig = go.Figure()
    for src_name, X_src in synth_samples.items():
        if x_feat not in X_src.columns or y_feat not in X_src.columns:
            continue
        xs = pd.to_numeric(X_src[x_feat], errors="coerce")
        ys = pd.to_numeric(X_src[y_feat], errors="coerce")
        mask = xs.notna() & ys.notna()
        xs, ys = xs[mask], ys[mask]

        style = SOURCE_STYLE.get(src_name, {"color": "#718096", "symbol": "circle",
                                             "size": 8, "opacity": 0.6})

        if color_feat and color_feat in X_src.columns:
            cs = pd.to_numeric(X_src[color_feat], errors="coerce")[mask]
            c_meta = FACTOR_META.get(color_feat, {})
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name=src_name,
                marker=dict(
                    size=style["size"],
                    symbol=style["symbol"],
                    color=cs,
                    colorscale="Viridis",
                    showscale=(src_name == "Real (train)"),
                    colorbar=dict(
                        title=c_meta.get("label", color_feat),
                        thickness=10, len=0.6,
                    ),
                    opacity=style["opacity"],
                    line=dict(color="white", width=0.5),
                ),
                hovertemplate=(
                    f"<b>{src_name}</b><br>"
                    f"{x_meta.get('label', x_feat)}: %{{x:.2f}} {x_meta.get('unit','')}<br>"
                    f"{y_meta.get('label', y_feat)}: %{{y:.2f}} {y_meta.get('unit','')}<br>"
                    f"{c_meta.get('label', color_feat)}: %{{marker.color:.2f}}"
                    "<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name=src_name,
                marker=dict(
                    size=style["size"],
                    symbol=style["symbol"],
                    color=style["color"],
                    opacity=style["opacity"],
                    line=dict(color="white", width=0.5),
                ),
                hovertemplate=(
                    f"<b>{src_name}</b><br>"
                    f"{x_meta.get('label', x_feat)}: %{{x:.2f}} {x_meta.get('unit','')}<br>"
                    f"{y_meta.get('label', y_feat)}: %{{y:.2f}} {y_meta.get('unit','')}<br>"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        xaxis=dict(
            title=f"{x_meta.get('label', x_feat)} ({x_meta.get('unit','')})",
            showgrid=True, gridcolor=_T["grid_light"],
        ),
        yaxis=dict(
            title=f"{y_meta.get('label', y_feat)} ({y_meta.get('unit','')})",
            showgrid=True, gridcolor=_T["grid_light"],
        ),
        height=420,
        margin=dict(l=0, r=0, t=44, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_T["bg_plot_light"],
        legend=dict(
            orientation="h", y=-0.18, x=0,
            font=dict(size=12),
            itemsizing="constant",
        ),
        hovermode="closest",
        title=dict(
            text=(
                f"<b>{x_meta.get('label', x_feat)} vs {y_meta.get('label', y_feat)}</b>"
                + (f"  <span style='font-size:11px;color:{_T['neutral_mid']};'>coloured by "
                   f"{FACTOR_META.get(color_feat,{}).get('label', color_feat)}</span>"
                   if color_feat else "")
            ),
            font=dict(size=13), x=0,
        ),
    )
    return fig


def make_scatter_chart(y_test: pd.Series, y_pred: np.ndarray,
                       model_name: str, scenario: str) -> go.Figure:
    """Actual vs Predicted scatter, coloured by absolute error."""
    errs   = np.abs(y_test.values - y_pred)
    diag   = [min(y_test.min(), y_pred.min()) - 3,
               max(y_test.max(), y_pred.max()) + 3]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(diag), y=list(diag),
        mode="lines",
        line=dict(color=_T["neutral_light"], width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=y_test.values.tolist(),
        y=y_pred.tolist(),
        mode="markers",
        marker=dict(
            size=11,
            color=errs.tolist(),
            colorscale="YlOrRd",
            cmin=0, cmax=float(errs.max()) + 1,
            colorbar=dict(title="|error|", thickness=12, len=0.7),
            line=dict(color="white", width=0.6),
        ),
        text=[f"Error: {e:.2f}" for e in errs],
        hovertemplate="Actual: <b>%{x:.1f}</b><br>Predicted: <b>%{y:.1f}</b><br>%{text}<extra></extra>",
        showlegend=False,
    ))
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)
    fig.update_layout(
        title=dict(
            text=f"<b>{model_name} + {scenario}</b>"
                 f"  <span style='font-size:12px;color:{_T['neutral_mid']};'>"
                 f"MAE {mae:.2f} · RMSE {rmse:.2f} · R² {r2:.3f}</span>",
            font=dict(size=13), x=0,
        ),
        xaxis=dict(title="Actual Sleep Score", showgrid=True, gridcolor=_T["grid_light"]),
        yaxis=dict(title="Predicted Sleep Score", showgrid=True, gridcolor=_T["grid_light"]),
        height=380,
        margin=dict(l=0, r=0, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_T["bg_plot_light"],
    )
    return fig


def score_card_html(score: float, synced_str: str = "") -> str:
    if score >= 80:
        color, grade = _T["accent_gold"], "GOOD"
    elif score >= 60:
        color, grade = _T["accent_fair"], "FAIR"
    else:
        color, grade = _T["accent_poor"], "POOR"
    pct = int(score)
    synced_html = (
        f'<div class="zen-score-synced">Last synced {synced_str}</div>'
        if synced_str else ""
    )
    return (
        f'<div class="zen-score-card">'
        f'<div class="zen-score-label">Predicted Sleep Score</div>'
        f'<div class="zen-score-number" style="color:{color};">{score:.0f}</div>'
        f'<div class="zen-score-grade" style="color:{color};">{grade}</div>'
        f'<div class="zen-score-bar-bg">'
        f'<div class="zen-score-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
        f'<div style="color:{_T["text_faint"]};font-size:11px;margin-top:8px;">out of 100</div>'
        f'{synced_html}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _dark_fig(fig: go.Figure) -> None:
    """Apply Zen dark theme to any Plotly figure in-place."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=f"rgba(255,255,255,{_T['alpha_plot_dark']})",
        font=dict(color=_T["text_muted"]),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                     tickfont=dict(color=_T["text_muted"]),
                     title_font=dict(color=_T["text_faint"]))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                     tickfont=dict(color=_T["text_muted"]),
                     title_font=dict(color=_T["text_faint"]))
    for ann in fig.layout.annotations:
        if ann.font and not ann.font.color:
            ann.font.color = _T["text_muted"]


def _score_recommendation(score: float) -> str:
    if score >= 85:
        return ("Your environment is well-optimized for sleep. "
                "Conditions are ideal — maintain this setup.")
    elif score >= 70:
        return ("Conditions are generally favorable. Minor adjustments "
                "may further improve your sleep quality.")
    elif score >= 55:
        return ("Several environmental factors are working against restful sleep. "
                "See the impact breakdown below.")
    else:
        return ("Your environment significantly disrupts sleep quality. "
                "Priority improvements are highlighted below.")


def _env_table_html(input_values: dict, feat_names, sv) -> str:
    rows_html = ""
    for i, f in enumerate(ALL_FACTORS):
        m        = FACTOR_META[f]
        val      = input_values[f]
        shap_idx = list(feat_names).index(f) if f in feat_names else None
        impact   = sv[shap_idx] if shap_idx is not None else 0.0
        if impact < -0.5:
            indicator = f'<span style="color:{_T["accent_poor"]};font-size:10px;font-weight:600;">▼ HURTING</span>'
        elif impact > 0.5:
            indicator = f'<span style="color:{_T["accent_gold"]};font-size:10px;font-weight:600;">▲ HELPING</span>'
        else:
            indicator = f'<span style="color:{_T["text_dim"]};font-size:10px;">— NEUTRAL</span>'
        bg = "rgba(87,73,90,0.14)" if i % 2 == 0 else "rgba(87,73,90,0.04)"
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:9px 14px;color:{_T["text_secondary"]};font-size:13px;font-weight:500;">'
            f'{m["label"]}</td>'
            f'<td style="padding:9px 14px;color:{_T["text_primary"]};font-size:13px;font-weight:600;'
            f'text-align:right;">{val:.1f} {m["unit"]}</td>'
            f'<td style="padding:9px 14px;color:{_T["text_faint"]};font-size:11px;text-align:center;">'
            f'{m["ideal"]}</td>'
            f'<td style="padding:9px 14px;text-align:right;">{indicator}</td>'
            f'</tr>'
        )
    return (
        f'<table style="width:100%;border-collapse:collapse;background:{_rgba(_T["bg_sidebar"],0.45)};'
        'border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.06);">'
        '<thead><tr style="background:rgba(255,255,255,0.04);">'
        f'<th style="padding:10px 15px;text-align:left;font-size:10px;letter-spacing:1.5px;'
        f'color:{_T["text_muted"]};text-transform:uppercase;font-weight:600;">Factor</th>'
        f'<th style="padding:10px 15px;text-align:right;font-size:10px;letter-spacing:1.5px;'
        f'color:{_T["text_muted"]};text-transform:uppercase;font-weight:600;">Current</th>'
        f'<th style="padding:10px 15px;text-align:center;font-size:10px;letter-spacing:1.5px;'
        f'color:{_T["text_muted"]};text-transform:uppercase;font-weight:600;">Ideal Range</th>'
        f'<th style="padding:10px 15px;text-align:right;font-size:10px;letter-spacing:1.5px;'
        f'color:{_T["text_muted"]};text-transform:uppercase;font-weight:600;">Status</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
    )


def _section_label(text: str) -> None:
    st.markdown(
        f'<div style="font-size:14px;letter-spacing:2px;text-transform:uppercase;'
        f'color:{_T["text_secondary"]};font-weight:600;margin-bottom:20px;padding-bottom:8px;'
        f'border-bottom:1px solid rgba(255,255,255,0.06);">{text}</div>',
        unsafe_allow_html=True,
    )


def _render_recommendations(hurts, helps, neutral, input_values):
    rcol1, rcol2 = st.columns(2, gap="large")
    with rcol1:
        if hurts:
            st.markdown(
                f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                f'color:{_T["accent_poor"]};font-weight:600;margin-bottom:10px;">Hurting Sleep</div>',
                unsafe_allow_html=True,
            )
            for f, v in hurts:
                m   = FACTOR_META.get(f, {})
                val = input_values.get(f, 0.0)
                st.markdown(
                    f'<div style="background:{_POOR_10};'
                    f'border:1px solid {_POOR_28};'
                    f'border-radius:10px;padding:14px 16px;margin-bottom:8px;">'
                    f'<div style="font-size:13px;font-weight:600;color:{_T["accent_fair"]};margin-bottom:4px;">'
                    f'{m.get("label", f)}</div>'
                    f'<div style="font-size:12px;color:{_T["text_faint"]};">'
                    f'Currently <strong style="color:{_T["text_primary"]};">{val:.1f} {m.get("unit","")}</strong>'
                    f' — pulling score down by '
                    f'<strong style="color:{_T["accent_poor"]};">{abs(v):.1f} pts</strong>'
                    f'&nbsp;·&nbsp;Ideal: {m.get("ideal","–")}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div style="padding:16px;border-radius:10px;'
                f'background:{_GOLD_07};'
                f'border:1px solid {_GOLD_18};'
                f'font-size:13px;color:{_T["accent_gold"]};">'
                'No strongly negative factors detected.</div>',
                unsafe_allow_html=True,
            )
    with rcol2:
        if helps:
            st.markdown(
                f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
                f'color:{_T["accent_gold"]};font-weight:600;margin-bottom:10px;">Helping Sleep</div>',
                unsafe_allow_html=True,
            )
            for f, v in helps:
                m   = FACTOR_META.get(f, {})
                val = input_values.get(f, 0.0)
                st.markdown(
                    f'<div style="background:{_GOLD_08};'
                    f'border:1px solid {_GOLD_22};'
                    f'border-radius:10px;padding:14px 16px;margin-bottom:8px;">'
                    f'<div style="font-size:13px;font-weight:600;color:{_T["accent_gold"]};margin-bottom:4px;">'
                    f'{m.get("label", f)}</div>'
                    f'<div style="font-size:12px;color:{_T["text_faint"]};">'
                    f'Currently <strong style="color:{_T["text_primary"]};">{val:.1f} {m.get("unit","")}</strong>'
                    f' — boosting score by '
                    f'<strong style="color:{_T["accent_gold"]};">+{v:.1f} pts</strong>'
                    f'&nbsp;·&nbsp;Ideal: {m.get("ideal","–")}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div style="padding:16px;border-radius:10px;'
                'background:rgba(255,255,255,0.03);'
                'border:1px solid rgba(255,255,255,0.07);'
                f'font-size:13px;color:{_T["text_faint"]};">'
                'No strongly positive factors detected.</div>',
                unsafe_allow_html=True,
            )
    if neutral:
        with st.expander("Neutral / low-impact factors"):
            for f, v in neutral:
                m   = FACTOR_META.get(f, {})
                val = input_values.get(f, 0.0)
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);'
                    f'border:1px solid rgba(255,255,255,0.06);'
                    f'border-radius:8px;padding:10px 14px;margin-bottom:6px;'
                    f'font-size:12px;color:{_T["text_faint"]};">'
                    f'<strong style="color:{_T["text_secondary"]};">{m.get("label", f)}</strong>'
                    f' &nbsp;·&nbsp; {val:.1f} {m.get("unit","")} &nbsp;·&nbsp; '
                    f'impact {v:+.2f} pts &nbsp;·&nbsp; ideal: {m.get("ideal","–")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub_html = (
        f'<div style="font-size:13px;color:{_T["text_faint"]};margin-top:8px;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:24px;">'
        f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        f'color:{_T["text_faint"]};margin-bottom:6px;">{eyebrow}</div>'
        f'<div style="font-size:26px;font-weight:700;color:{_T["text_primary"]};letter-spacing:-0.5px;">'
        f'{title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _page_dashboard(trained, explainers, feat_names, df_session, selected_model):
    _page_header("Live Monitoring", "Dashboard")

    df_live      = fetch_live_sensor()
    synced_str   = ""
    input_values = {}

    if not df_live.empty:
        latest_row = df_live.sort_values("recorded_at").iloc[-1]
        ts = pd.to_datetime(latest_row["recorded_at"], utc=True).astimezone(TH_TZ)
        synced_str   = ts.strftime("%#d %B %Y  %H:%M:%S") + " (TH)"
        input_values = {
            f: float(pd.to_numeric(latest_row.get(f, 0), errors="coerce") or 0.0)
            for f in ALL_FACTORS
        }
    else:
        st.warning("Live sensor data unavailable — showing default values.")
        input_values = {f: FACTOR_META[f]["default"] for f in ALL_FACTORS}

    pipe      = trained[selected_model]["pipe"]
    explainer = explainers[selected_model]
    predicted, sv, base_val = predict_and_explain(pipe, explainer, input_values, feat_names)

    # ── Auto-save: predict all models and record if timestamp is new ──────────
    if not df_live.empty:
        latest_ts_key = str(latest_row["recorded_at"])
        if st.session_state.get("last_saved_ts") != latest_ts_key:
            try:
                all_scores = {}
                for model_name in DB_SCORE_COLS:
                    score_val, _, _ = predict_and_explain(
                        trained[model_name]["pipe"],
                        explainers[model_name],
                        input_values,
                        feat_names,
                    )
                    all_scores[model_name] = round(score_val, 2)
                sensor_row = {
                    "recorded_at": pd.to_datetime(
                        latest_row["recorded_at"], utc=True
                    ).astimezone(TH_TZ).replace(tzinfo=None),
                    **{col: latest_row.get(col) for col in [
                        "temp_c", "hum_pct", "light_lux", "snd_avg",
                        "snd_peak", "snd_var", "snd_evt", "pm1", "pm25", "pm10",
                    ]},
                }
                save_prediction_to_db(sensor_row, all_scores)
                st.session_state["last_saved_ts"] = latest_ts_key
                st.toast("Saved to database", icon="✅")
            except Exception as _db_err:
                st.toast(f"DB error: {_db_err}", icon="⚠️")

    # Hero row
    col_score, col_shap = st.columns([1, 2.2], gap="large")
    with col_score:
        st.markdown(score_card_html(predicted, synced_str), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
            f'padding:14px 16px;border:1px solid rgba(255,255,255,0.06);">'
            f'<div style="font-size:10px;letter-spacing:1.5px;color:{_T["text_dim"]};'
            f'text-transform:uppercase;margin-bottom:6px;">Insight</div>'
            f'<div style="font-size:13px;color:{_T["text_insight"]};line-height:1.6;">'
            f'{_score_recommendation(predicted)}</div></div>',
            unsafe_allow_html=True,
        )
    with col_shap:
        st.plotly_chart(
            make_shap_chart(sv, feat_names, predicted, base_val, selected_model),
            use_container_width=True,
            config={"displayModeBar": False},
            key="shap_dashboard",
        )

    st.divider()

    _section_label("Current Environment")
    st.markdown(_env_table_html(input_values, feat_names, sv), unsafe_allow_html=True)

    st.divider()

    _section_label("Sensor Trends — Last 40 Readings")
    if not df_live.empty:
        st.plotly_chart(
            make_snapshot_chart(df_live),
            use_container_width=True,
            config={"displayModeBar": False},
            key="snap_dashboard",
        )
    else:
        st.info("No trend data available.")

    st.divider()

    _section_label("Recommendations")
    shap_pairs = list(zip(feat_names, sv))
    hurts   = sorted([(f, v) for f, v in shap_pairs if v < -0.5], key=lambda x: x[1])
    helps   = sorted([(f, v) for f, v in shap_pairs if v >  0.5], key=lambda x: x[1], reverse=True)
    neutral = [(f, v) for f, v in shap_pairs if -0.5 <= v <= 0.5]
    _render_recommendations(hurts, helps, neutral, input_values)

    st.markdown(
        f'<div style="font-size:11px;color:{_T["text_deepdim"]};margin-top:24px;">'
        f'Model: {selected_model} &nbsp;·&nbsp; '
        f'Sessions: {len(df_session)} &nbsp;·&nbsp; '
        f'SHAP base: {base_val:.1f} pts</div>',
        unsafe_allow_html=True,
    )


def _page_simulation(trained, explainers, feat_names, df_session, selected_model):
    _page_header("Experiment", "Simulation Lab",
                 "Adjust each environmental factor to explore predicted sleep outcomes.")

    sim_col_sliders, sim_col_result = st.columns([1, 1.3], gap="large")

    with sim_col_sliders:
        _section_label("Environment Controls")
        custom_vals = {}
        for f in ALL_FACTORS:
            m = FACTOR_META[f]
            custom_vals[f] = st.slider(
                f"{m['label']}  ({m['unit']})",
                min_value=m["min"], max_value=m["max"],
                value=m["default"], step=m["step"],
                help=f"Research ideal: {m['ideal']}",
                key=f"sim_{f}",
            )

    pipe      = trained[selected_model]["pipe"]
    explainer = explainers[selected_model]
    predicted, sv, base_val = predict_and_explain(pipe, explainer, custom_vals, feat_names)

    with sim_col_result:
        _section_label("Predicted Outcome")
        st.markdown(score_card_html(predicted), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
            f'padding:14px 16px;border:1px solid rgba(255,255,255,0.06);">'
            f'<div style="font-size:10px;letter-spacing:1.5px;color:{_T["text_dim"]};'
            f'text-transform:uppercase;margin-bottom:6px;">Analysis</div>'
            f'<div style="font-size:13px;color:{_T["text_insight"]};line-height:1.6;">'
            f'{_score_recommendation(predicted)}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    _section_label("Factor Impact Breakdown")
    st.plotly_chart(
        make_shap_chart(sv, feat_names, predicted, base_val, selected_model),
        use_container_width=True,
        config={"displayModeBar": False},
        key="shap_simulation",
    )

    st.divider()
    _section_label("Environment Summary")
    st.markdown(_env_table_html(custom_vals, feat_names, sv), unsafe_allow_html=True)

    st.divider()
    _section_label("Recommendations")
    shap_pairs = list(zip(feat_names, sv))
    hurts   = sorted([(f, v) for f, v in shap_pairs if v < -0.5], key=lambda x: x[1])
    helps   = sorted([(f, v) for f, v in shap_pairs if v >  0.5], key=lambda x: x[1], reverse=True)
    neutral = [(f, v) for f, v in shap_pairs if -0.5 <= v <= 0.5]
    _render_recommendations(hurts, helps, neutral, custom_vals)


def _page_synthesis_lab(df_session):
    _page_header("Research", "Data Synthesis Lab",
                 "With ~37 real sessions, augmentation is essential. "
                 "Compare SMOTER and KNN synthesis across all 30 model-scenario pipelines.")

    col_run, col_info = st.columns([1, 2], gap="large")
    with col_run:
        run_btn = st.button("Run Experiment", type="primary", use_container_width=True,
                            help="6 models × 5 scenarios = 30 pipelines on GroupShuffleSplit.")
    with col_info:
        st.markdown(
            f'<div style="background:{_GOLD_07};'
            f'border-left:3px solid {_GOLD_40};'
            f'border-radius:8px;padding:12px 16px;font-size:13px;color:{_T["text_muted"]};">'
            f'<strong style="color:{_T["accent_gold"]};">SMOTER</strong> (Torgo 2013): interpolates X and y '
            'between k-nearest pairs with λ ~ U(0,1). &nbsp;|&nbsp; '
            f'<strong style="color:{_T["accent_gold"]};">KNN Synthesis</strong>: Dirichlet convex combination '
            'of k neighbours; y via distance-weighted KNN regression.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if "synth_results" not in st.session_state or run_btn:
        with st.spinner("Running 30 pipelines (5 scenarios × 6 models)…"):
            out = run_synthesis_experiment(df_session)
            st.session_state["synth_results"] = out
        st.success("Experiment complete.")

    results, synth_samples, y_test, predictions_lab, n_base = \
        st.session_state["synth_results"]

    feat_options = {FACTOR_META[f]["label"]: f for f in ALL_FACTORS}
    feat_labels  = list(feat_options.keys())

    # 1 — Distribution Explorer
    _section_label("1 — Synthesis Distribution Explorer")
    dist_tab_kde, dist_tab_scatter = st.tabs(
        ["KDE — 1-D Distribution", "Scatter — 2-D Feature Space"]
    )

    with dist_tab_kde:
        dc1, dc2 = st.columns([1, 3])
        with dc1:
            chosen_label   = st.selectbox("Feature", feat_labels, key="kde_feat_select")
            chosen_feature = feat_options[chosen_label]
            st.markdown("---")
            for src, X_src in synth_samples.items():
                if chosen_feature in X_src.columns:
                    vals  = pd.to_numeric(X_src[chosen_feature], errors="coerce").dropna()
                    color = {
                        "Real (train)": _T["snap_temp"],
                        "SMOTER":       _T["accent_fair"],
                        "KNN Synth":    _T["accent_medium"],
                    }.get(src, _T["text_faint"])
                    st.markdown(
                        f'<div style="border-left:3px solid {color};padding:4px 10px;'
                        f'margin:4px 0;font-size:12px;">'
                        f'<strong style="color:{color};">{src}</strong>'
                        f'<span style="color:{_T["text_faint"]};"> — n={len(vals)} · '
                        f'μ={vals.mean():.1f} · σ={vals.std():.1f}</span></div>',
                        unsafe_allow_html=True,
                    )
        with dc2:
            kf = make_kde_chart(synth_samples, chosen_feature)
            _dark_fig(kf)
            st.plotly_chart(kf, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"kde_selected_{chosen_feature}")

        with st.expander("Show KDE for all 8 features"):
            gc = st.columns(2)
            for idx, f in enumerate(ALL_FACTORS):
                with gc[idx % 2]:
                    kf2 = make_kde_chart(synth_samples, f)
                    _dark_fig(kf2)
                    st.plotly_chart(kf2, use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"kde_grid_{f}")

    with dist_tab_scatter:
        sc_c1, sc_c2, sc_c3 = st.columns(3)
        with sc_c1:
            x_label = st.selectbox("X axis", feat_labels, index=0, key="scatter_x")
        with sc_c2:
            y_label = st.selectbox("Y axis", feat_labels, index=1, key="scatter_y")
        with sc_c3:
            color_label = st.selectbox("Colour by", ["None"] + feat_labels,
                                       index=0, key="scatter_color")
        x_feat     = feat_options[x_label]
        y_feat     = feat_options[y_label]
        color_feat = feat_options.get(color_label) if color_label != "None" else None
        sf = make_synth_scatter_chart(synth_samples, x_feat, y_feat, color_feat)
        _dark_fig(sf)
        st.plotly_chart(sf, use_container_width=True, config={"displayModeBar": True},
                        key=f"synth_scatter_{x_feat}_{y_feat}_{color_feat}")

    # 2 — Metrics Heatmap
    st.divider()
    _section_label("2 — Error Metrics — All 30 Combinations")
    metric_tabs = st.tabs(["MAE", "RMSE", "R²"])
    for mt, metric in zip(metric_tabs, ["MAE", "RMSE", "R²"]):
        with mt:
            mf = make_metrics_heatmap(results, metric)
            _dark_fig(mf)
            st.plotly_chart(mf, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"heatmap_{metric}")

    # 3 — ΔMAE
    st.divider()
    _section_label("3 — ΔMAE vs No-Augmentation Baseline")
    dmf = make_delta_mae_chart(results)
    _dark_fig(dmf)
    st.plotly_chart(dmf, use_container_width=True,
                    config={"displayModeBar": False}, key="delta_mae")

    # 4 — Best per model
    st.divider()
    _section_label("4 — Best Configuration per Model")
    summary_rows = []
    for model_name in MODEL_REGISTRY:
        base_mae  = results["No Augmentation"][model_name]["MAE"]
        best_scen = min(results, key=lambda s: results[s][model_name]["MAE"])
        best_mae  = results[best_scen][model_name]["MAE"]
        delta     = best_mae - base_mae
        summary_rows.append({
            "Model":         model_name,
            "Baseline MAE":  round(base_mae,  3),
            "Best Scenario": best_scen,
            "Best MAE":      round(best_mae,  3),
            "Δ MAE":         round(delta,     3),
            "Improved?":     "Yes" if delta < 0 else "No",
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("Best MAE")
    st.dataframe(
        df_summary.style
        .map(
            lambda v: f"color:{_T['accent_gold']};font-weight:600"
            if v == "Yes" else (f"color:{_T['accent_poor']}" if v == "No" else ""),
            subset=["Improved?"],
        )
        .format({"Baseline MAE": "{:.3f}", "Best MAE": "{:.3f}", "Δ MAE": "{:+.3f}"}),
        hide_index=True,
        use_container_width=True,
    )

    # 5 — Actual vs Predicted
    st.divider()
    _section_label("5 — Actual vs Predicted")
    sc1, sc2 = st.columns(2)
    with sc1:
        scatter_model = st.selectbox("Model", list(MODEL_REGISTRY.keys()),
                                     index=5, key="scatter_model")
    with sc2:
        scatter_scen = st.selectbox("Scenario", SYNTH_SCENARIOS,
                                    index=4, key="scatter_scen")
    y_pred_sel = predictions_lab.get((scatter_scen, scatter_model))
    if y_pred_sel is not None:
        avp = make_scatter_chart(y_test, y_pred_sel, scatter_model, scatter_scen)
        _dark_fig(avp)
        st.plotly_chart(avp, use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"scatter_{scatter_model}_{scatter_scen}")
    else:
        st.warning("No results for this combination. Re-run the experiment.")

    st.markdown(
        f'<div style="font-size:11px;color:{_T["text_deepdim"]};margin-top:16px;">'
        f'GroupShuffleSplit(test_size=0.35) · ~{n_base} training sessions · '
        f'{len(y_test)} test sessions · '
        f'{len(MODEL_REGISTRY) * len(SYNTH_SCENARIOS)} pipelines</div>',
        unsafe_allow_html=True,
    )


def _page_about():
    _page_header("Background", "About the Project")

    # Vision
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{_rgba(_T["bg_sidebar"],0.9)},'
        f'{_rgba(_T["bg_card_deep"],0.9)});border:1px solid {_GOLD_18};'
        'border-radius:14px;padding:28px 32px;margin-bottom:24px;">'
        f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        f'color:{_T["accent_gold"]};font-weight:600;margin-bottom:12px;">Project Vision</div>'
        f'<div style="font-size:17px;font-weight:600;color:{_T["text_primary"]};margin:0 0 12px;">'
        'Can we quantify the bedroom environment\'s impact on sleep?</div>'
        f'<div style="color:{_T["text_muted"]};font-size:13px;line-height:1.9;">'
        'SleepSense began with a simple question: sleep quality is often treated as subjective, '
        'but the physical environment — temperature, humidity, light, sound, air quality — is '
        'measurable. This project builds a machine learning pipeline that connects IoT sensor '
        'readings to morning sleep quality self-reports, creating an objective, data-driven '
        'sleep environment predictor.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Technical Stack
    _section_label("Technical Stack")
    tc1, tc2, tc3 = st.columns(3, gap="medium")
    _card = (
        f"background:{_rgba(_T['bg_sidebar'],0.6)};border:1px solid rgba(255,255,255,0.07);"
        "border-radius:12px;padding:22px;"
    )
    with tc1:
        st.markdown(
            f'<div style="{_card}">'
            f'<div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:{_T["accent_poor"]};font-weight:600;margin-bottom:10px;">Hardware / IoT</div>'
            f'<div style="font-size:13px;color:{_T["text_secondary"]};font-weight:600;margin-bottom:8px;">'
            'Sensor Array</div>'
            f'<div style="font-size:12px;color:{_T["text_faint"]};line-height:1.8;">'
            'ESP32 node &nbsp;·&nbsp; DHT22 (Temp/Humidity)<br>'
            'BH1750 (Light) &nbsp;·&nbsp; MAX4466 (Sound)<br>'
            'PMS5003 (PM1, PM2.5, PM10)<br>'
            'REST API push every 60 seconds.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with tc2:
        st.markdown(
            f'<div style="{_card}">'
            f'<div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:{_T["accent_purple"]};font-weight:600;margin-bottom:10px;">Software</div>'
            f'<div style="font-size:13px;color:{_T["text_secondary"]};font-weight:600;margin-bottom:8px;">'
            'Data Pipeline</div>'
            f'<div style="font-size:12px;color:{_T["text_faint"]};line-height:1.8;">'
            'Python &nbsp;·&nbsp; Pandas &nbsp;·&nbsp; scikit-learn<br>'
            'Streamlit &nbsp;·&nbsp; Google Sheets API<br>'
            'RobustScaler + log-transform preprocessing<br>'
            'GroupShuffleSplit session-aware evaluation.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with tc3:
        st.markdown(
            f'<div style="{_card}">'
            f'<div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:{_T["accent_gold"]};font-weight:600;margin-bottom:10px;">AI / ML</div>'
            f'<div style="font-size:13px;color:{_T["text_secondary"]};font-weight:600;margin-bottom:8px;">'
            'Models &amp; Explainability</div>'
            f'<div style="font-size:12px;color:{_T["text_faint"]};line-height:1.8;">'
            'Random Forest &nbsp;·&nbsp; Gradient Boosting<br>'
            'Ridge &nbsp;·&nbsp; Lasso &nbsp;·&nbsp; Linear Regression<br>'
            'SHAP (TreeExplainer + LinearExplainer)<br>'
            'SMOTER + KNN data synthesis.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Development Journey
    _section_label("Development Journey")
    journey = [
        (_T["accent_poor"],   "1", "Sensor Calibration",
         "Deployed the IoT sensor array in the sleep environment. Validated readings against "
         "reference instruments and tuned sampling intervals for stable overnight data collection."),
        (_T["accent_medium"], "2", "Sleep Diary Integration",
         "Designed a Google Forms check-in workflow for morning sleep quality self-reports. "
         "Built the ETL pipeline aligning sensor windows with precise sleep/wake timestamps."),
        (_T["accent_purple"], "3", "Data Synthesis Research",
         "With only ~37 usable sessions, standard train/test splits were unstable. Implemented "
         "SMOTER (Torgo 2013) and KNN Synthesis to generate realistic augmented samples without "
         "extrapolating beyond observed distributions."),
        (_T["accent_fair"],   "4", "Model Evaluation",
         "Benchmarked 6 regression models across 5 augmentation scenarios (30 pipelines total) "
         "on a held-out real test set using GroupShuffleSplit to prevent session data leakage."),
        (_T["accent_gold"],   "5", "Explainability Layer",
         "Integrated SHAP to provide per-prediction factor attribution — turning the model from "
         "a black box into an actionable sleep environment advisor with clear recommendations."),
    ]
    for color, num, title, desc in journey:
        st.markdown(
            f'<div style="display:flex;gap:16px;margin-bottom:14px;">'
            f'<div style="flex-shrink:0;width:30px;height:30px;border-radius:50%;'
            f'background:{color};display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;font-weight:700;color:rgba(50,50,76,0.95);margin-top:2px;">{num}</div>'
            f'<div style="background:{_rgba(_T["bg_sidebar"],0.5)};border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:10px;padding:14px 18px;flex:1;">'
            f'<div style="font-size:13px;font-weight:600;color:{_T["text_primary"]};margin-bottom:4px;">'
            f'{title}</div>'
            f'<div style="font-size:12px;color:{_T["text_faint"]};line-height:1.7;">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Learnings
    _section_label("Key Learnings")
    learnings = [
        ("Data quality over quantity",
         "Even with synthesis, a small real dataset (n≈37 sessions) demands careful "
         "cross-validation. Session-level GroupShuffleSplit was critical to prevent leakage."),
        ("Synthesis is not magic",
         "SMOTER and KNN augmentation improved some models but not all. Gains were "
         "model-specific — tree ensembles benefited more than linear models."),
        ("Explainability earns trust",
         "SHAP transformed the app from a score machine into a diagnostic tool. "
         "Users engage far more when they understand why a score was assigned."),
        ("IoT integration complexity",
         "Sensor placement, calibration drift, and API reliability added significant "
         "engineering overhead that doesn't appear in notebook-only experiments."),
    ]
    lc1, lc2 = st.columns(2, gap="medium")
    for i, (title, desc) in enumerate(learnings):
        col = lc1 if i % 2 == 0 else lc2
        with col:
            st.markdown(
                f'<div style="background:{_rgba(_T["bg_sidebar"],0.5)};'
                f'border:1px solid rgba(255,255,255,0.06);'
                f'border-radius:10px;padding:18px 20px;margin-bottom:12px;">'
                f'<div style="font-size:12px;font-weight:600;color:{_T["accent_gold"]};margin-bottom:6px;">'
                f'{title}</div>'
                f'<div style="font-size:12px;color:{_T["text_faint"]};line-height:1.7;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;font-size:11px;color:{_T["text_deepdim"]};padding:16px;">'
        'SleepSense &nbsp;·&nbsp; Built with Streamlit, scikit-learn, SHAP, and Plotly'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    trained, explainers, feat_names, df_session, X_t_all = load_and_train()
    if trained is None:
        st.error("Not enough training data. Check API connectivity or credentials.")
        st.stop()

    with st.sidebar:
        st.markdown(
            f'<div style="padding:16px 4px 24px;">'
            '<div style="display:flex;align-items:center;gap:10px;">'
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none">'
            '<path d="M12 3C7.5 3 4 7 4 12s3.5 9 8 9c.8 0 1.5-.1 2.2-.3'
            'C13.4 20 13 19 13 18c0-3.3 2.7-6 6-6h.3c.2-.6.3-1.3.3-2'
            f'C19.6 5.8 16.1 3 12 3z" fill="{_T["accent_gold"]}"/>'
            f'<circle cx="19" cy="18" r="4" fill="{_T["accent_poor"]}" opacity="0.8"/>'
            '</svg>'
            '<div>'
            f'<div style="font-size:16px;font-weight:700;color:{_T["text_primary"]};letter-spacing:-0.2px;">'
            'SleepSense</div>'
            f'<div style="font-size:10px;color:{_T["text_dim"]};letter-spacing:1.5px;'
            'text-transform:uppercase;">Environment Predictor</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;'
            f'color:{_T["text_dim"]};font-weight:600;margin-bottom:6px;padding:0 4px;">Navigate</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "page",
            ["Dashboard", "Simulation Lab", "Data Synthesis Lab", "About"],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown(
            f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;'
            f'color:{_T["text_dim"]};font-weight:600;margin-bottom:6px;padding:0 4px;">Model</div>',
            unsafe_allow_html=True,
        )
        selected_model = st.selectbox(
            "model",
            list(MODEL_REGISTRY.keys()),
            index=4,
            label_visibility="collapsed",
            help="All models share the same preprocessing pipeline.",
        )

        st.divider()

        if page == "Dashboard":
            if st.button("Refresh Live Data", use_container_width=True):
                fetch_live_sensor.clear()
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:11px;color:{_T["text_dim"]};line-height:2;">'
            f'Sessions &nbsp;<span style="color:{_T["text_muted"]};font-weight:500;">'
            f'{len(df_session)}</span><br>'
            f'Features &nbsp;<span style="color:{_T["text_muted"]};font-weight:500;">'
            f'{len(feat_names)}</span></div>',
            unsafe_allow_html=True,
        )

    if page == "Dashboard":
        _page_dashboard(trained, explainers, feat_names, df_session, selected_model)
    elif page == "Simulation Lab":
        _page_simulation(trained, explainers, feat_names, df_session, selected_model)
    elif page == "Data Synthesis Lab":
        _page_synthesis_lab(df_session)
    else:
        _page_about()


if __name__ == "__main__":
    main()
