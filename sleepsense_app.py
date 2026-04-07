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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SleepSense Predictor",
    page_icon="🛌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px; padding: 28px 24px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .score-label { color: #a0aec0; font-size: 13px; letter-spacing: 1px;
                   text-transform: uppercase; margin-bottom: 4px; }
    .score-number { font-size: 72px; font-weight: 800; line-height: 1; margin: 4px 0; }
    .score-tag { font-size: 16px; font-weight: 600; letter-spacing: 0.5px; }
    .rec-hurt  { background: #fff5f5; border-left: 4px solid #e53e3e;
                 border-radius: 8px; padding: 12px 16px; margin: 6px 0; }
    .rec-help  { background: #f0fff4; border-left: 4px solid #38a169;
                 border-radius: 8px; padding: 12px 16px; margin: 6px 0; }
    .rec-neutral { background: #f7fafc; border-left: 4px solid #a0aec0;
                   border-radius: 8px; padding: 10px 16px; margin: 4px 0; }
    .factor-row { display: flex; justify-content: space-between;
                  align-items: center; font-size: 14px; }
    .ideal-badge { font-size: 11px; color: #718096; background: #edf2f7;
                   border-radius: 4px; padding: 2px 6px; }
    h1 { font-size: 1.9rem !important; }
    .stAlert { border-radius: 8px; }
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

SNAPSHOT_COLS = [
    ("temp_c",    "Temperature (°C)",  "#ef5350"),
    ("hum_pct",   "Humidity (%)",      "#42a5f5"),
    ("light_lux", "Light (lux)",       "#ffa726"),
    ("snd_avg",   "Sound (dB)",        "#ab47bc"),
    ("pm25",      "PM2.5 (µg/m³)",     "#26a69a"),
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
    "No Augmentation": "#718096",
    "SMOTER ×3":       "#5a7fb5",
    "SMOTER ×5":       "#2b6cb0",
    "KNN Synth ×3":    "#e07b54",
    "KNN Synth ×5":    "#c05621",
}

MODEL_PALETTE = {
    "Linear Regression": "#5a7fb5",
    "Ridge  (α=1)":      "#e07b54",
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
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                fill="tozeroy",
                fillcolor=(
                    color.replace(")", ", 0.08)").replace("rgb", "rgba")
                    if color.startswith("rgb")
                    else hex_to_rgba(color, 0.08)
                ),
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
        plot_bgcolor="#f8fafc",
        font=dict(size=11),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
    return fig


def make_shap_chart(sv, feat_names, predicted, base_value, model_name) -> go.Figure:
    labels = [FACTOR_META.get(f, {}).get("label", f) for f in feat_names]
    order  = np.argsort(sv)          # most-negative first (top)
    sv_ord = sv[order]
    lb_ord = [labels[i] for i in order]
    fn_ord = [feat_names[i] for i in order]

    colors  = ["#e53e3e" if v < 0 else "#38a169" for v in sv_ord]
    borders = ["#c53030" if v < 0 else "#276749" for v in sv_ord]

    text_labels = []
    for val, fn in zip(sv_ord, fn_ord):
        meta = FACTOR_META.get(fn, {})
        sign = "↓ hurting" if val < 0 else "↑ helping"
        text_labels.append(f"{sign}  {val:+.1f} pts")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=lb_ord,
        x=sv_ord,
        orientation="h",
        marker=dict(color=colors, line=dict(color=borders, width=1)),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11, color="#4a5568"),
        hovertemplate="<b>%{y}</b><br>Impact: %{x:+.2f} pts<extra></extra>",
        cliponaxis=False,
    ))

    fig.add_vline(x=0, line=dict(color="#4a5568", width=1.5))

    x_range = max(abs(sv_ord.min()), abs(sv_ord.max()))
    fig.update_layout(
        title=dict(
            text=f"<b>Environmental Impact on Sleep Score</b><br>"
                 f"<span style='font-size:12px;color:#718096;'>"
                 f"Base: {base_value:.0f} pts → Predicted: <b>{predicted:.0f}</b> / 100"
                 f"</span>",
            font=dict(size=14), x=0,
        ),
        xaxis=dict(
            title="Score impact (SHAP value, points)",
            range=[-(x_range * 1.7), x_range * 1.7],
            zeroline=False, showgrid=True, gridcolor="#e2e8f0",
            tickfont=dict(size=11),
        ),
        yaxis=dict(tickfont=dict(size=12), automargin=True),
        height=max(300, len(feat_names) * 52),
        margin=dict(l=0, r=120, t=72, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
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
        "Real (train)": ("#2d6a4f", 0.35),
        "SMOTER":       ("#5a7fb5", 0.20),
        "KNN Synth":    ("#e07b54", 0.20),
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
                 + (f"  <span style='color:#718096;font-size:12px;'>Ideal: {ideal}</span>" if ideal else ""),
            font=dict(size=14), x=0,
        ),
        xaxis=dict(title=f"{label} ({unit})", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(title="Density", showgrid=True, gridcolor="#e2e8f0", zeroline=False),
        height=320,
        margin=dict(l=0, r=0, t=56, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
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
                 f"  <span style='font-size:12px;color:#718096;'>— all 30 (model × scenario) combinations</span>",
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
    fig.add_hline(y=0, line=dict(color="#4a5568", width=1.5, dash="dash"),
                  annotation_text="Baseline (no augmentation)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#4a5568"))
    fig.update_layout(
        title=dict(
            text="<b>ΔMAE vs No-Augmentation Baseline</b>"
                 "  <span style='font-size:12px;color:#718096;'>points below 0 = synthesis helped</span>",
            font=dict(size=13), x=0,
        ),
        xaxis=dict(title="Augmentation Scenario", tickfont=dict(size=11)),
        yaxis=dict(title="ΔMAE (pts)", showgrid=True, gridcolor="#e2e8f0", zeroline=False),
        height=360,
        margin=dict(l=0, r=0, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
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
        "Real (train)": {"color": "#2d6a4f", "symbol": "circle",      "size": 11, "opacity": 0.85},
        "SMOTER":       {"color": "#5a7fb5", "symbol": "diamond",      "size":  9, "opacity": 0.70},
        "KNN Synth":    {"color": "#e07b54", "symbol": "cross",        "size":  9, "opacity": 0.70},
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
            showgrid=True, gridcolor="#e2e8f0",
        ),
        yaxis=dict(
            title=f"{y_meta.get('label', y_feat)} ({y_meta.get('unit','')})",
            showgrid=True, gridcolor="#e2e8f0",
        ),
        height=420,
        margin=dict(l=0, r=0, t=44, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        legend=dict(
            orientation="h", y=-0.18, x=0,
            font=dict(size=12),
            itemsizing="constant",
        ),
        hovermode="closest",
        title=dict(
            text=(
                f"<b>{x_meta.get('label', x_feat)} vs {y_meta.get('label', y_feat)}</b>"
                + (f"  <span style='font-size:11px;color:#718096;'>coloured by "
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
        line=dict(color="#a0aec0", width=1.5, dash="dash"),
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
                 f"  <span style='font-size:12px;color:#718096;'>"
                 f"MAE {mae:.2f} · RMSE {rmse:.2f} · R² {r2:.3f}</span>",
            font=dict(size=13), x=0,
        ),
        xaxis=dict(title="Actual Sleep Score", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(title="Predicted Sleep Score", showgrid=True, gridcolor="#e2e8f0"),
        height=380,
        margin=dict(l=0, r=0, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
    )
    return fig


def score_card_html(score: float) -> str:
    if score >= 80:
        color, tag = "#38a169", "Good"
    elif score >= 60:
        color, tag = "#d69e2e", "Fair"
    else:
        color, tag = "#e53e3e", "Poor"
    pct = int(score)
    return f"""
    <div class="score-card">
        <div class="score-label">Predicted Sleep Score</div>
        <div class="score-number" style="color:{color};">{score:.0f}</div>
        <div class="score-tag" style="color:{color};">{tag}</div>
        <div style="margin-top:14px;background:#2d3748;border-radius:8px;height:10px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:8px;
                        transition:width 0.6s ease;"></div>
        </div>
        <div style="color:#a0aec0;font-size:11px;margin-top:6px;">out of 100</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🛌 SleepSense — Environment Sleep Score Predictor")
    st.markdown(
        "Understand how your bedroom environment affects sleep quality "
        "and get actionable recommendations based on real sensor data.",
        unsafe_allow_html=False,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    trained, explainers, feat_names, df_session, X_t_all = load_and_train()

    if trained is None:
        st.error("Not enough training data. Check API connectivity or credentials.")
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        selected_model = st.selectbox(
            "Model",
            list(MODEL_REGISTRY.keys()),
            index=4,
            help="All models share the same preprocessing pipeline.",
        )

        st.divider()

        input_mode = st.radio(
            "Input source",
            ["🔴 Live sensor (latest reading)", "✏️ Manual entry"],
        )

        custom_vals: dict = {}
        if "Manual" in input_mode:
            st.subheader("Custom environment values")
            for f in ALL_FACTORS:
                m = FACTOR_META[f]
                custom_vals[f] = st.slider(
                    f"{m['label']} ({m['unit']})",
                    min_value=m["min"], max_value=m["max"],
                    value=m["default"], step=m["step"],
                    help=f"Research ideal: {m['ideal']}",
                )

        st.divider()
        st.caption(f"Training sessions: **{len(df_session)}**")
        st.caption(f"Features: {', '.join(feat_names)}")

        if st.button("🔄  Refresh live data", use_container_width=True):
            fetch_live_sensor.clear()
            st.rerun()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_pred, tab_lab = st.tabs([
        "🛌 Sleep Prediction",
        "🧬 Data Synthesis Lab",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — SLEEP PREDICTION (original functionality)
    # ══════════════════════════════════════════════════════════════════════
    with tab_pred:
        df_live = fetch_live_sensor()

        st.subheader("📡 Recent Sensor Readings")
        if not df_live.empty:
            snap_fig = make_snapshot_chart(df_live)
            st.plotly_chart(snap_fig, use_container_width=True,
                            config={"displayModeBar": False}, key="snap")
            df_live_sorted = df_live.sort_values("recorded_at")
            latest_row     = df_live_sorted.iloc[-1]
            ts = pd.to_datetime(latest_row["recorded_at"], utc=True).astimezone(TH_TZ)
            st.caption(f"Last sensor reading: **{ts.strftime('%#d %B %Y  %H:%M:%S')} (Thailand Time)**")
        else:
            st.warning("Could not fetch sensor data from API.")

        st.divider()

        if "Manual" in input_mode:
            input_values = custom_vals
            st.info("Using manually entered values for prediction.")
        else:
            if df_live.empty:
                st.error("Live sensor data unavailable. Switch to Manual entry.")
                st.stop()
            latest_row   = df_live.sort_values("recorded_at").iloc[-1]
            input_values = {f: float(pd.to_numeric(latest_row.get(f, 0), errors="coerce") or 0.0)
                            for f in ALL_FACTORS}

        pipe      = trained[selected_model]["pipe"]
        explainer = explainers[selected_model]
        predicted, sv, base_val = predict_and_explain(
            pipe, explainer, input_values, feat_names)

        col_score, col_shap = st.columns([1, 2.4])

        with col_score:
            st.markdown(score_card_html(predicted), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Current Environment**")
            env_rows = []
            for f in ALL_FACTORS:
                m        = FACTOR_META[f]
                val      = input_values[f]
                shap_idx = list(feat_names).index(f) if f in feat_names else None
                impact   = sv[shap_idx] if shap_idx is not None else 0.0
                arrow    = "🔴" if impact < -0.5 else ("🟢" if impact > 0.5 else "⚪")
                env_rows.append({"": arrow, "Factor": m["label"],
                                 "Value": f"{val:.1f}", "Unit": m["unit"],
                                 "Ideal": m["ideal"]})
            st.dataframe(pd.DataFrame(env_rows), hide_index=True,
                         use_container_width=True)

        with col_shap:
            shap_fig = make_shap_chart(sv, feat_names, predicted,
                                       base_val, selected_model)
            st.plotly_chart(shap_fig, use_container_width=True,
                            config={"displayModeBar": False}, key="shap_main")

        st.divider()
        st.subheader("💡 Recommendations")

        shap_pairs = list(zip(feat_names, sv))
        hurts   = sorted([(f, v) for f, v in shap_pairs if v < -0.5],
                          key=lambda x: x[1])
        helps   = sorted([(f, v) for f, v in shap_pairs if v >  0.5],
                          key=lambda x: x[1], reverse=True)
        neutral = [(f, v) for f, v in shap_pairs if -0.5 <= v <= 0.5]

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            if hurts:
                st.markdown("#### 🔴 Factors hurting your sleep")
                for f, v in hurts:
                    m   = FACTOR_META.get(f, {})
                    val = input_values.get(f, 0.0)
                    st.markdown(f"""
                    <div class="rec-hurt">
                      <div class="factor-row">
                        <span><b>{m.get('label', f)}</b></span>
                        <span class="ideal-badge">ideal: {m.get('ideal','–')}</span>
                      </div>
                      <div style="margin-top:6px;color:#718096;font-size:13px;">
                        Currently <b>{val:.1f} {m.get('unit','')}</b> —
                        pulling score down by <b style="color:#e53e3e;">{abs(v):.1f} pts</b>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No strongly negative factors detected.")

        with rcol2:
            if helps:
                st.markdown("#### 🟢 Factors helping your score")
                for f, v in helps:
                    m   = FACTOR_META.get(f, {})
                    val = input_values.get(f, 0.0)
                    st.markdown(f"""
                    <div class="rec-help">
                      <div class="factor-row">
                        <span><b>{m.get('label', f)}</b></span>
                        <span class="ideal-badge">ideal: {m.get('ideal','–')}</span>
                      </div>
                      <div style="margin-top:6px;color:#718096;font-size:13px;">
                        Currently <b>{val:.1f} {m.get('unit','')}</b> —
                        boosting score by <b style="color:#38a169;">+{v:.1f} pts</b>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No strongly positive factors at the moment.")

        if neutral:
            with st.expander("⚪ Neutral / low-impact factors"):
                for f, v in neutral:
                    m   = FACTOR_META.get(f, {})
                    val = input_values.get(f, 0.0)
                    st.markdown(f"""
                    <div class="rec-neutral">
                      <b>{m.get('label', f)}</b>: {val:.1f} {m.get('unit','')}
                      &nbsp;·&nbsp; impact <b>{v:+.2f} pts</b>
                      &nbsp;·&nbsp; ideal: {m.get('ideal','–')}
                    </div>""", unsafe_allow_html=True)

        st.divider()
        st.caption(
            f"Model: **{selected_model}** · "
            f"Training sessions: {len(df_session)} · "
            f"SHAP base: {base_val:.1f} pts"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — DATA SYNTHESIS LAB
    # ══════════════════════════════════════════════════════════════════════
    with tab_lab:
        st.markdown(
            "### 🧬 Data Synthesis Lab\n"
            "With only ~37 session rows, models generalise poorly without augmentation. "
            "This lab trains every model on each synthesis scenario and compares their "
            "error metrics on the **same real held-out test sessions**."
        )

        col_run, col_info = st.columns([1, 2])
        with col_run:
            run_btn = st.button(
                "▶  Run Experiment",
                type="primary",
                use_container_width=True,
                help="Trains 6 models × 5 scenarios = 30 pipelines on a GroupShuffleSplit "
                     "of the session table. Test set is always real sessions only.",
            )
        with col_info:
            st.markdown(
                "<div style='background:#ebf8ff;border-left:4px solid #3182ce;"
                "border-radius:6px;padding:10px 14px;font-size:13px;'>"
                "<b>SMOTER</b> (Torgo 2013): interpolates X and y between k-nearest session "
                "pairs with the same λ ~ U(0,1). &nbsp;|&nbsp; "
                "<b>KNN Synthesis</b>: generates X as a Dirichlet convex combination of k "
                "neighbours; assigns y via distance-weighted KNN regression."
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Auto-run on first load; re-run only when button is clicked
        if "synth_results" not in st.session_state or run_btn:
            with st.spinner("Running 30 pipelines (5 scenarios × 6 models)…"):
                out = run_synthesis_experiment(df_session)
                st.session_state["synth_results"] = out
            st.success("✅ Experiment complete!")

        results, synth_samples, y_test, predictions_lab, n_base = \
            st.session_state["synth_results"]

        # ── Section 1: Distribution Explorer ────────────────────────────
        st.subheader("1️⃣  Synthesis Distribution Explorer")
        st.markdown(
            "Compare how the synthetic data covers the real training distribution. "
            "A good synthesis method should **fill gaps** without creating impossible values."
        )

        feat_options  = {FACTOR_META[f]["label"]: f for f in ALL_FACTORS}
        feat_labels   = list(feat_options.keys())

        dist_tab_kde, dist_tab_scatter = st.tabs(["📈 KDE (1-D distribution)", "🔵 Scatter (2-D feature space)"])

        # ── KDE sub-tab ──────────────────────────────────────────────────
        with dist_tab_kde:
            dist_col1, dist_col2 = st.columns([1, 3])
            with dist_col1:
                chosen_label   = st.selectbox("Feature to inspect", feat_labels,
                                              key="kde_feat_select")
                chosen_feature = feat_options[chosen_label]
                st.markdown("---")
                for src, X_src in synth_samples.items():
                    if chosen_feature in X_src.columns:
                        vals = pd.to_numeric(X_src[chosen_feature], errors="coerce").dropna()
                        color = {"Real (train)": "#2d6a4f",
                                 "SMOTER": "#5a7fb5", "KNN Synth": "#e07b54"}.get(src, "#718096")
                        st.markdown(
                            f"<div style='border-left:3px solid {color};"
                            f"padding:4px 10px;margin:4px 0;font-size:12px;'>"
                            f"<b style='color:{color};'>{src}</b><br>"
                            f"n={len(vals)} &nbsp; μ={vals.mean():.1f} &nbsp; σ={vals.std():.1f}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            with dist_col2:
                kde_fig = make_kde_chart(synth_samples, chosen_feature)
                st.plotly_chart(kde_fig, use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"kde_selected_{chosen_feature}")

            with st.expander("📊 Show KDE for all 8 features"):
                grid_cols = st.columns(2)
                for idx, f in enumerate(ALL_FACTORS):
                    with grid_cols[idx % 2]:
                        st.plotly_chart(
                            make_kde_chart(synth_samples, f),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"kde_grid_{f}",
                        )

        # ── Scatter sub-tab ───────────────────────────────────────────────
        with dist_tab_scatter:
            sc_c1, sc_c2, sc_c3 = st.columns(3)
            with sc_c1:
                x_label = st.selectbox("X axis", feat_labels,
                                       index=0, key="scatter_x")
            with sc_c2:
                y_label = st.selectbox("Y axis", feat_labels,
                                       index=1, key="scatter_y")
            with sc_c3:
                color_options = ["None"] + feat_labels
                color_label   = st.selectbox("Colour points by (optional)", color_options,
                                             index=0, key="scatter_color")

            x_feat     = feat_options[x_label]
            y_feat     = feat_options[y_label]
            color_feat = feat_options.get(color_label) if color_label != "None" else None

            # Legend key for symbols
            st.markdown(
                "<div style='font-size:12px;color:#718096;margin-bottom:6px;'>"
                "⬤ <b style='color:#2d6a4f;'>Real (train)</b> &nbsp;◆ "
                "<b style='color:#5a7fb5;'>SMOTER synthetic</b> &nbsp;✕ "
                "<b style='color:#e07b54;'>KNN synthetic</b>"
                "</div>",
                unsafe_allow_html=True,
            )

            synth_key = f"synth_scatter_{x_feat}_{y_feat}_{color_feat}"
            st.plotly_chart(
                make_synth_scatter_chart(synth_samples, x_feat, y_feat, color_feat),
                use_container_width=True,
                config={"displayModeBar": True},
                key=synth_key,
            )

            st.caption(
                "Real training points use ● circles; SMOTER synthetic use ◆ diamonds; "
                "KNN synthetic use ✕ crosses. "
                "Synthetic points that cluster tightly around real points indicate "
                "conservative augmentation; wide spread may indicate extrapolation."
            )

        # ── Section 2: Metrics Heatmap ───────────────────────────────────
        st.divider()
        st.subheader("2️⃣  Error Metrics — All 30 Combinations")
        st.markdown(
            "Every cell is one trained model evaluated on the **real test sessions**. "
            "★ marks the best scenario per row."
        )

        metric_tabs = st.tabs(["MAE ↓", "RMSE ↓", "R² ↑"])
        for mt, metric in zip(metric_tabs, ["MAE", "RMSE", "R²"]):
            with mt:
                st.plotly_chart(
                    make_metrics_heatmap(results, metric),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"heatmap_{metric}",
                )

        # ── Section 3: ΔMAE Line Chart ───────────────────────────────────
        st.divider()
        st.subheader("3️⃣  ΔMAE vs No-Augmentation Baseline")
        st.markdown(
            "How much does each synthesis method move the MAE? "
            "Points **below zero** mean that scenario improved generalisation for that model."
        )
        st.plotly_chart(
            make_delta_mae_chart(results),
            use_container_width=True,
            config={"displayModeBar": False},
            key="delta_mae",
        )

        # ── Section 4: Best-per-model summary table ───────────────────────
        st.divider()
        st.subheader("4️⃣  Best Configuration per Model")
        summary_rows = []
        for model_name in MODEL_REGISTRY:
            base_mae  = results["No Augmentation"][model_name]["MAE"]
            best_scen = min(results, key=lambda s: results[s][model_name]["MAE"])
            best_mae  = results[best_scen][model_name]["MAE"]
            delta     = best_mae - base_mae
            summary_rows.append({
                "Model":          model_name,
                "Baseline MAE":   round(base_mae,  3),
                "Best Scenario":  best_scen,
                "Best MAE":       round(best_mae,  3),
                "Δ MAE":          round(delta,     3),
                "Improved?":      "✅ Yes" if delta < 0 else "❌ No",
            })
        df_summary = pd.DataFrame(summary_rows).sort_values("Best MAE")
        st.dataframe(
            df_summary.style
            .map(lambda v: "color:#38a169;font-weight:600"
                 if v == "✅ Yes" else ("color:#e53e3e" if v == "❌ No" else ""),
                 subset=["Improved?"])
            .format({"Baseline MAE": "{:.3f}", "Best MAE": "{:.3f}", "Δ MAE": "{:+.3f}"}),
            hide_index=True,
            use_container_width=True,
        )

        # ── Section 5: Interactive Actual vs Predicted ────────────────────
        st.divider()
        st.subheader("5️⃣  Actual vs Predicted — Pick Any Combination")
        sc1, sc2 = st.columns(2)
        with sc1:
            scatter_model = st.selectbox(
                "Model",
                list(MODEL_REGISTRY.keys()),
                index=5,
                key="scatter_model",
            )
        with sc2:
            scatter_scen = st.selectbox(
                "Augmentation scenario",
                SYNTH_SCENARIOS,
                index=4,
                key="scatter_scen",
            )

        y_pred_sel = predictions_lab.get((scatter_scen, scatter_model))
        if y_pred_sel is not None:
            st.plotly_chart(
                make_scatter_chart(y_test, y_pred_sel, scatter_model, scatter_scen),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"scatter_{scatter_model}_{scatter_scen}",
            )
        else:
            st.warning("No results found for this combination. Re-run the experiment.")

        st.divider()
        st.caption(
            f"Experiment: GroupShuffleSplit(test_size=0.35, random_state=42) · "
            f"Training sessions: ~{n_base} · Test sessions: {len(y_test)} · "
            f"Total fitted pipelines: {len(MODEL_REGISTRY) * len(SYNTH_SCENARIOS)}"
        )


if __name__ == "__main__":
    main()
