# SleepSense Analytics

> **Predicting and explaining sleep quality from real-time environmental IoT data**

A full data-science pipeline that fuses three heterogeneous data sources — a live IoT sensor stream, a user-maintained sleep diary, and an academic research rubric — into a session-level regression model. SHAP values translate every prediction into human-readable, per-factor recommendations ("lower bedroom temperature by ~2 °C"). A companion Streamlit web app delivers the same analysis in real time.

---

## Table of Contents

1. [Project Architecture & Data Sources](#1-project-architecture--data-sources)
2. [Complex Data Exploration](#2-complex-data-exploration)
3. [Complex Data Pre-Processing](#3-complex-data-pre-processing)
4. [Analytic Techniques & Modelling](#4-analytic-techniques--modelling)
5. [Evaluation & Error Analysis](#5-evaluation--error-analysis)
6. [Possible Applications](#6-possible-applications)
7. [References & Resources](#7-references--resources)

---

## 1. Project Architecture & Data Sources

### 1.1 Three-source data model

The project deliberately ingests data from three different origins to balance **real-world validity** (IoT + diary) with **statistical coverage** (research baseline).

| Source | Variable | Raw rows | Key columns |
|---|---|---|---|
| **IoT Sensor REST API** (live) | `df_sensor` | ~474 timestamped readings | `temp_c`, `hum_pct`, `light_lux`, `snd_avg`, `snd_evt`, `pm1`, `pm25`, `pm10`, `timestamp` |
| **Google Sheets sleep diary** | `df_checkin` | ~37 user-logged entries | `Session ID`, `Sleep_Score` (0–100), `sleep_start`, `sleep_end`, `duration_h` |
| **Academic research rubric** | `df_research` | Expanded from range tables | Environmental ranges mapped to Sleep Score bands per published guidelines |

<p align="center">
  <img src="https://placehold.co/900x220/1a1a2e/ffffff?text=Data+Architecture%3A+IoT+API+%2B+Google+Sheets+%2B+Research+Rubric" alt="Data architecture" width="860"/>
  <br/><sub><i>Replace with output from notebook cells 06–08 (raw table previews)</i></sub>
</p>

### 1.2 Session-window matching

The 474 raw sensor readings are **matched to scored sleep sessions** by filtering each sensor timestamp to fall within the `[sleep_start, sleep_end]` window recorded in the diary. This is implemented in `extract_sensor_sessions()`:

```python
def extract_sensor_sessions(df_checkin):
    # For each diary entry, filter sensor rows whose timestamp
    # falls inside [sleep_start, sleep_end] and tag them with Session ID
    ...
```

Each matched window then contains between 1 and ~30 sensor rows. After aggregation (§3.1) this collapses to **one representative environment vector per night**.

### 1.3 Research rubric expansion

`expand_research_rubric(df)` converts the academic table of ideal-range → score mappings into concrete training rows by sampling uniformly inside each range band. This gives the model exposure to the full environmental spectrum even when real-data collection is sparse.

### 1.4 Integration and schema

The two session tables (`df_sensor_sessions` from real nights, `df_research_sessions` from the rubric) are concatenated via `pd.concat` and tagged with a `source` column. The working schema is:

```
Session ID | temp_c | hum_pct | light_lux | snd_avg | snd_evt | pm1 | pm25 | pm10 | Sleep_Score | source
```

<p align="center">
  <img src="https://placehold.co/900x420/1a1a2e/ffffff?text=Correlation+Heatmap+%2B+Source+Distribution+%2B+Score+by+Session" alt="Integration EDA" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 18 (4-panel figure: correlation heatmap, temp scatter, noise KDE, score boxplot)</i></sub>
</p>

The correlation matrix (cell 18, panel A) immediately reveals `temp_c`, `light_lux`, and `hum_pct` as the strongest negative predictors of `Sleep_Score` in the combined dataset, motivating their central role in later SHAP analysis.

---

## 2. Complex Data Exploration

EDA is divided into **six dedicated sections**, each targeting a different aspect of the data. All EDA runs on the pre-merge dataframes (`df_sensor`, `df_checkin`, `df_sensor_sessions`, `df_research_sessions`, `df_combined`) — no model assumptions leak in.

### 2.1 Check-in & Sleep Behaviour

**What is analysed:** The subjective sleep diary — score distributions, session durations, the duration–score relationship, and bed-time timing patterns.

**Specific visualisations produced (cell 21):**
- Histogram + KDE of `Sleep_Score` across all logged sessions
- Histogram of `duration_h` with a mean line
- Scatter of `duration_h` vs `Sleep_Score` with a linear regression overlay (Pearson r annotated)
- Clock-face or time-of-day histogram of `sleep_start` hour

<p align="center">
  <img src="https://placehold.co/900x420/0d1b2a/ffffff?text=Sleep+Score+Distribution+%7C+Duration+Histogram+%7C+Duration+vs+Score+Scatter" alt="Check-in EDA" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 21</i></sub>
</p>

**Findings:** Scores range from ~20 to 100 with most sessions clustering between 55–80. Session durations span 4–9 h. The correlation between duration and score is positive but weak — environmental factors carry more explanatory power.

---

### 2.2 Sensor Time-Series & Statistics

**What is analysed:** The raw chronological sensor stream for all 8 channels, compared against research-defined ideal bands.

**Specific visualisations produced (cell 23):**
- 8 time-series subplots (one per feature), x-axis = timestamp, y-axis = raw reading
- Green shaded band on each subplot = research-ideal range (e.g., 18–20 °C for `temp_c`, ≤ 5 lux for `light_lux`)
- Red markers at readings that fall **outside** the ideal band
- Summary statistics table: mean, std, % of time inside ideal range per feature

<p align="center">
  <img src="https://placehold.co/900x500/0d1b2a/ffffff?text=8+Sensor+Channels+Time-Series+with+Ideal-Range+Bands" alt="Sensor time-series" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 23</i></sub>
</p>

**Findings:** `temp_c` consistently exceeds the 18–20 °C ideal. `light_lux` shows periodic spikes, likely from uncovered windows. `pm25` and `pm10` are occasionally elevated, consistent with urban environments.

---

### 2.3 Session Environmental Profiles

**What is analysed:** How environmental conditions differ **across** sleep sessions — not just the mean state but the session-to-session variance.

**Specific visualisations produced (cell 25):**
- **Raw-median heatmap:** rows = session IDs, columns = 8 features, cell = median sensor reading for that session. Uses a diverging colormap so above/below-average nights are immediately visible.
- **Normalised heatmap (0–1 min-max):** same structure but scaled so each column spans [0, 1], making high/low patterns comparable across features with different physical units (°C vs µg/m³).
- Per-session summary table with all 8 medians

<p align="center">
  <img src="https://placehold.co/900x420/0d1b2a/ffffff?text=Session+%C3%97+Feature+Heatmaps%3A+Raw+Medians+%26+0%E2%80%931+Normalised" alt="Session heatmaps" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 25</i></sub>
</p>

---

### 2.4 Distribution Comparison: Actual Sessions vs Research Baseline

**What is analysed:** Whether the real collected data inhabits the same environmental space as the research baseline — a prerequisite for the research rubric to be a valid training supplement.

**Specific visualisations produced (cell 27):**
- 8 subplots, one per feature
- Each subplot: overlaid KDE curves — **red** = actual sensor sessions, **blue** = research baseline
- Dashed vertical lines for group means
- Overlap coefficient (or KL-divergence) printed in the corner of each subplot

<p align="center">
  <img src="https://placehold.co/900x420/0d1b2a/ffffff?text=KDE+Overlay%3A+Actual+vs+Research+Baseline+for+All+8+Factors" alt="Distribution comparison" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 27</i></sub>
</p>

**Findings:** `temp_c` and `hum_pct` distributions overlap reasonably well. `light_lux` diverges — real sessions show higher ambient light than the research ideal. `snd_evt` (sound event count) is near-zero in real data but more varied in the rubric.

---

### 2.5 Factor vs Sleep Score Relationships

**What is analysed:** Bivariate relationships between each environmental feature and the target variable, with source-stratified scatter and Pearson correlation ranking.

**Specific visualisations produced (cell 29):**
- 8 scatter plots (one per feature), data points colour-coded: **red** = actual sensor sessions, **blue** = research baseline
- Regression trend line per source using `np.polyfit` (guarded against zero-variance columns via `std() > 1e-9` check to prevent `LinAlgError`)
- A 9th panel: horizontal bar chart ranking all 8 features by Pearson r with `Sleep_Score` (NaN correlations — e.g., for constant `snd_evt` — are dropped gracefully)

<p align="center">
  <img src="https://placehold.co/900x500/0d1b2a/ffffff?text=Factor+vs+Sleep+Score+Scatter+%2B+Regression+Lines+%2B+Correlation+Ranking" alt="Factor vs score" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 29</i></sub>
</p>

**Findings (Pearson r ranking, strongest first):**  
`temp_c` (−), `light_lux` (−), `hum_pct` (−), `snd_avg` (−), `pm25` (−), `pm1` (+/−), `pm10` (+/−), `snd_evt` (near zero / constant in real data).

---

### 2.6 Data Quality & Coverage

**What is analysed:** Completeness, source balance, and per-session missing-data patterns.

**Specific visualisations produced (cell 31):**
- **Missing-value map:** binary heatmap (rows = all sessions, columns = 9 variables) — red = NaN, white = present. Reveals which sessions lack specific sensor channels.
- **Source breakdown pie chart:** proportion of rows from real IoT sessions vs research rubric.
- **Per-session completeness bar chart:** % of the 8 feature columns that are non-NaN for each actual sleep session (research rows always complete).
- Summary coverage table: count and % of non-NaN per feature

<p align="center">
  <img src="https://placehold.co/900x380/0d1b2a/ffffff?text=Missing+Value+Map+%7C+Source+Pie+%7C+Per-Session+Completeness" alt="Data quality" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 31</i></sub>
</p>

---

## 3. Complex Data Pre-Processing

### 3.1 Why not just `StandardScaler`?

`StandardScaler` is inappropriate here for three concrete reasons:

1. **Mixed distributional shapes:** `light_lux` has Pearson skewness +1.57 (right tail from lamp spikes); `pm25` is near-symmetric (+0.16); `temp_c` is slightly negatively skewed (−0.30). A single scaling strategy fits none of them well.
2. **Pseudo-replication:** The 474 raw rows come from only 37 distinct scored nights. Fitting a model on raw rows artificially inflates *n*, biases standard errors, and makes cross-validation meaningless — any split might put readings from the *same night* in both train and test.
3. **Sensor spikes:** Short-lived extreme readings (e.g., a light spike when someone briefly enters the room) should not distort the representative environmental description for a whole night.

---

### 3.2 Session-level aggregation

```python
agg_map = {c: 'median' for c in all_factors}
agg_map['Sleep_Score'] = 'median'

df_session = (
    df_combined
    .groupby('Session ID', as_index=False)
    .agg(agg_map)
    .dropna(subset=['Sleep_Score'])
)
# 474 raw rows → 37 session-level rows
```

**Median** (not mean) is used for aggregation because it is resistant to intra-session sensor spikes — a brief lux spike from an open door does not distort the representative light level for the whole night.

<p align="center">
  <img src="https://placehold.co/900x360/0d1b2a/ffffff?text=Step+1%3A+Session+Aggregation+%E2%80%94+Raw+Readings+%E2%86%92+Median+per+Night" alt="Session aggregation" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 35, Figure 1 (strip plots of raw readings vs median per session for temp_c, light_lux, snd_avg, pm25)</i></sub>
</p>

---

### 3.3 Custom `Winsorizer` class

A `sklearn`-compatible transformer that clips each feature to its **[5th, 95th] percentile** interval, fitted on **training data only** to prevent leakage:

```python
class Winsorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower_q, axis=0)
        self.hi_ = np.nanquantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
```

This is a drop-in sklearn transformer (implements `fit`, `transform`, `get_feature_names_out`) so it slots into `Pipeline` and `ColumnTransformer` without breaking `clone()` or cross-validation.

---

### 3.4 Adaptive log1p decision (skewness threshold = 0.75)

Rather than applying `log1p` to a hardcoded list of features (which is a common but incorrect assumption that PM and light are always right-skewed), the pipeline **measures skewness from the actual session-level data** first:

```python
SKEW_THRESHOLD = 0.75
feature_skew = {
    c: pd.to_numeric(df_session[c], errors='coerce').dropna().skew()
    for c in all_factors
}

# Only apply log1p where skewness is genuinely positive AND values are non-negative
log_skewed = [
    c for c in all_factors
    if pd.notna(feature_skew[c])
    and feature_skew[c] > SKEW_THRESHOLD
    and df_session[c].min() >= 0
]
comfort_linear = [c for c in all_factors if c not in log_skewed]
```

**On this dataset's actual distributions:**

| Feature | Measured skewness | Decision |
|---|---|---|
| `light_lux` | **+1.57** | → `log1p + RobustScaler` |
| `temp_c` | −0.30 | → Winsor + RobustScaler |
| `hum_pct` | −0.12 | → Winsor + RobustScaler |
| `snd_avg` | +0.45 | → Winsor + RobustScaler |
| `pm1` | −0.48 | → Winsor + RobustScaler |
| `pm25` | +0.16 | → Winsor + RobustScaler |
| `pm10` | −0.20 | → Winsor + RobustScaler |
| `snd_evt` | NaN/0 (constant) | → Winsor + RobustScaler |

With this dataset only `light_lux` crosses the threshold. Applying `log1p` to `pm1` (skew −0.48) would introduce spurious asymmetry — the adaptive check prevents this.

<p align="center">
  <img src="https://placehold.co/900x380/0d1b2a/ffffff?text=Step+2%3A+Skewness+Bar+Chart+%2B+Before%2FAfter+for+Qualifying+Features" alt="Adaptive log1p" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 35, Figure 2 (left: skewness bar chart with ±0.75 threshold; right: before/after grouped bar for log_skewed features only)</i></sub>
</p>

---

### 3.5 Block-wise `ColumnTransformer`

The two feature blocks use different sub-pipelines, assembled in a `ColumnTransformer`:

```python
log_block = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('log1p',  FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
    ('scale',  RobustScaler()),
])

comfort_block = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('winsor', Winsorizer(lower_q=0.05, upper_q=0.95)),
    ('scale',  RobustScaler()),
])

# Only add log_skew transformer if any features qualify (avoids empty-list crash)
_transformers = []
if log_skewed:
    _transformers.append(('log_skew', log_block, log_skewed))
if comfort_linear:
    _transformers.append(('comfort', comfort_block, comfort_linear))

prep = ColumnTransformer(
    transformers=_transformers,
    remainder='drop',
    verbose_feature_names_out=False,
)
```

`verbose_feature_names_out=False` preserves original column names (e.g., `temp_c` rather than `comfort__temp_c`) so `get_feature_names_out()` returns names that map back directly to `all_factors`.

<p align="center">
  <img src="https://placehold.co/900x360/0d1b2a/ffffff?text=Step+3%3A+Winsorization+Before%2FAfter+Histograms+for+Comfort+Features" alt="Winsorization" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 35, Figure 3</i></sub>
</p>

<p align="center">
  <img src="https://placehold.co/900x360/0d1b2a/ffffff?text=Step+4%3A+RobustScaler+Before%2FAfter+Box+Plots+%E2%80%94+All+8+Features" alt="RobustScaler" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 35, Figure 4 (all 8 features on one axis before/after scaling)</i></sub>
</p>

---

### 3.6 `GroupShuffleSplit` — session-aware train/test split

Standard `train_test_split` on the raw 474-row table would allow readings from the **same sleep night** to appear in both training and test, leaking temporal structure and inflating metrics. `GroupShuffleSplit` treats the `Session ID` as the grouping key so each night appears in exactly one partition:

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
train_idx, test_idx = next(gss.split(X_sess, y_sess, groups=groups))
# Approx 24 sessions train  /  13 sessions test
```

The same `random_state=42` is used in every downstream cell that references the split, ensuring consistent train/test sessions across the preprocessing, multi-model comparison, and synthesis comparison cells.

<p align="center">
  <img src="https://placehold.co/900x320/0d1b2a/ffffff?text=Step+5%3A+GroupShuffleSplit+%E2%80%94+Train+vs+Test+Session+Assignment" alt="GroupShuffleSplit" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 35, Figure 5 (horizontal bar: session assignments; scatter: score per session coloured by split)</i></sub>
</p>

---

### 3.7 Training-data synthesis

With ~24–28 training sessions, tree ensembles are likely to overfit and linear models have too few degrees of freedom. Two established methods are implemented to augment the **training set only** — the test set always stays 100 % real.

#### SMOTER — Torgo et al. 2013

The regression analog of SMOTE. For each synthetic point:

1. Pick a random base session **i** from the training set
2. Find **k** nearest neighbours of **i** in RobustScaled feature space
3. Pick one neighbour **j** uniformly at random
4. Draw `λ ~ Uniform(0, 1)`
5. Interpolate: `x' = x_i + λ · (x_j − x_i)`,  `y' = y_i + λ · (y_j − y_i)`

The **same λ** is applied to both X and y, so the label is geometrically consistent with the new feature vector. All synthetic points lie strictly on line segments between real sessions — no extrapolation outside the training convex hull.

```python
def synth_smoter(X_df, y_s, n_synth, k_neighbors=5, seed=42):
    X_imp = _prep_X(X_df)                          # median impute NaNs first
    X_sc  = RobustScaler().fit_transform(X_imp)
    _, nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_sc).kneighbors(X_sc)
    lam   = rng.uniform(0, 1, size=n_synth)        # Uniform — the SMOTER spec
    X_syn = X_imp[i_idx] + lam[:,None] * (X_imp[j_idx] - X_imp[i_idx])
    y_syn = y[i_idx]     + lam          * (y[j_idx]     - y[i_idx])
    ...
```

#### KNN Regression Synthesis

Decouples geometry from label assignment:

1. Pick seed session **i**, find its **k** nearest neighbours
2. Generate `x'` as a **Dirichlet-weighted convex combination** of i and all k neighbours (not just one paired point)
3. Assign `y'` via a **distance-weighted KNN regressor** (`weights='distance'`) trained on real `(X, y)`

```python
def synth_knn(X_df, y_s, n_synth, k_neighbors=5, seed=42):
    ...
    knn_reg = KNeighborsRegressor(n_neighbors=k, weights='distance').fit(X_sc, y)
    for t in range(n_synth):
        base_pts = np.vstack([X_imp[i_idx[t]], X_imp[nbrs[i_idx[t], 1:k+1]]])
        weights  = rng.dirichlet(np.ones(k + 1))
        X_syn[t] = weights @ base_pts
    y_syn = knn_reg.predict(scaler.transform(X_syn))
    ...
```

**Key algorithmic difference:** SMOTER's y is locked to a single interpolated pair; KNN synthesis estimates y from the *entire* local neighbourhood using a regression model — it can produce labels consistent with multi-point local geometry.

<p align="center">
  <img src="https://placehold.co/900x380/0d1b2a/ffffff?text=Synthesis+Sanity+Check%3A+Real+vs+SMOTER+vs+KNN+KDE+Overlays" alt="Synthesis distributions" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 44, Figure 1 (KDE overlays for temp_c, hum_pct, light_lux, snd_avg, Sleep_Score)</i></sub>
</p>

Both methods are tested at **×3 and ×5** augmentation factors (generating 3× or 5× the training session count as synthetic rows), yielding 5 total scenarios × 6 models = **30 fitted pipelines** stored in `fitted_pipes[(scenario, model_name)]`.

---

## 4. Analytic Techniques & Modelling

### 4.1 Full model registry

All six models use the **same preprocessing pipeline** (§3). Each is wrapped in `Pipeline([('prep', prep), ('reg', model)])` and cloned fresh for every (scenario, model) combination using `sklearn.base.clone()` to prevent any state sharing.

| Model | Key hyperparameters | Rationale |
|---|---|---|
| `LinearRegression` | Default (OLS) | Interpretable baseline; signed coefficients give direct directional insight on preprocessed features |
| `Ridge(alpha=1.0)` | L2 λ=1 | Shrinks coefficients globally; handles correlated features (temp/humidity co-vary) |
| `Ridge(alpha=10.0)` | L2 λ=10 | Stronger shrinkage — tests whether heavier regularisation improves generalisation on 24 training sessions |
| `Lasso(alpha=0.5, max_iter=5000)` | L1 λ=0.5 | Performs automatic feature selection by zeroing low-signal coefficients; useful for detecting redundant PM channels |
| `RandomForestRegressor(n_estimators=300, random_state=42)` | 300 trees | Captures non-linear comfort optima (both too-hot and too-cold hurt sleep); ensemble averaging reduces variance |
| `GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)` | 200 boosting rounds, depth-3 trees | Sequential residual correction; typically the strongest performer on small tabular datasets |

### 4.2 Linear regression coefficient analysis

After fitting, the pipeline exposes coefficients in the transformed feature space:

```python
reg       = modeling_pipeline.named_steps['reg']
prep      = modeling_pipeline.named_steps['prep']
feat_names = prep.get_feature_names_out()   # original factor names preserved
coef_by_factor = dict(zip(feat_names, reg.coef_))
```

The sign of each coefficient is **directionally meaningful** in the original feature space because both `log1p` and `Winsorizer` are **monotone transformations** — they do not reverse the ordering of values. A negative coefficient on `temp_c` after `RobustScaler` still means "higher temperature → lower predicted score".

<p align="center">
  <img src="https://placehold.co/900x380/0d1b2a/ffffff?text=Regression+Coefficients+%2B+Trend+Plots+(Session-Level+Medians)" alt="Regression coefficients and trend plots" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 36 (3×3 regplot grid + coefficient table)</i></sub>
</p>

### 4.3 SHAP explainability

SHAP (SHapley Additive exPlanations) decomposes each prediction into a **sum of per-feature contributions** that are consistent, locally accurate, and satisfy the efficiency axiom (contributions sum to prediction − baseline). Three explainer types are selected adaptively:

```python
if hasattr(est, 'estimators_'):   # RandomForest / GradientBoosting
    exp = shap.TreeExplainer(est)                    # exact, O(TLD) per sample

elif hasattr(est, 'coef_'):       # LinearRegression / Ridge / Lasso
    exp = shap.LinearExplainer(est, X_train_t)       # exact for linear models

else:                             # any other model type
    bg  = shap.kmeans(X_train_t, min(10, len(X_train_t)))
    exp = shap.KernelExplainer(est.predict, bg)      # model-agnostic approximation
```

SHAP values are computed for **all session rows** (`X_all_for_shap = fitted_prep.transform(X_sess)`) to give stable feature importance estimates.

<p align="center">
  <img src="https://placehold.co/900x440/0d1b2a/ffffff?text=SHAP+Feature+Importance+(Mean+|SHAP|)+%E2%80%94+All+6+Models+Grid" alt="SHAP importance grid" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 42, Figure 1 (2×3 grid of horizontal bar charts, one per model)</i></sub>
</p>

<p align="center">
  <img src="https://placehold.co/900x440/0d1b2a/ffffff?text=SHAP+Beeswarm+Direction+Plots+%E2%80%94+All+6+Models" alt="SHAP beeswarm plots" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 42, Figure 2 (per-model beeswarm: x=SHAP value, colour=feature value; left=hurts score, right=helps)</i></sub>
</p>

The beeswarm plots show individual session-level impacts. A red dot (high feature value) far left means "high value of that feature strongly hurt the score in that session" — this is the visual basis for the recommendation engine.

### 4.4 SHAP-powered recommendation engine

Two prediction functions are exposed at the bottom of the notebook. Both accept any `(model_name, scenario)` combination from the 30 fitted pipelines:

**`optimize_sleep`** — coefficient-based directions (linear models):
```python
optimize_sleep(current_env,
               model_name='Ridge  (α=10)',
               scenario='SMOTER ×5')
# Output: signed coefficients mapped back to original features
# e.g.  - [temp_c]: 30.0  — NEGATIVE (lower → better score)
```

**`optimize_sleep_v2`** — SHAP-based, works for any model:
```python
optimize_sleep_v2(current_env,
                  model_name='Gradient Boosting',
                  scenario='KNN Synth ×5')
# Output:
# >>> Predicted Sleep Score: 62.4 / 100 <<<
# Factors HURTING your score:
#   ↓  temp_c    now=30.00   dragging score by -30.40 pts
#   ↓  light_lux now=36.68   dragging score by  -8.44 pts
# Factors HELPING your score:
#   ↑  hum_pct   now=40.00   boosting score by  +1.20 pts
```

`_get_explainer(scenario, model_name)` builds the appropriate SHAP explainer lazily (first call) and caches it in `_shap_cache` to avoid recomputation on repeated calls.

### 4.5 Interactive web application

`sleepsense_app.py` is a **Streamlit dashboard** that wraps the same pipeline:

```bash
pip install -r requirements_app.txt
streamlit run sleepsense_app.py
```

Features:
- Fetches the **latest sensor reading** from the live REST API
- Sidebar controls: model selector, augmentation scenario selector, input mode (live vs manual sliders)
- **Snapshot chart** (Plotly): last N readings for all 8 channels
- **Score gauge** (custom HTML): predicted sleep score 0–100 with colour bands
- **SHAP impact bar chart** (Plotly): features sorted by SHAP contribution, red = hurting, green = helping
- **Actionable text recommendations** for all negative-SHAP features

---

## 5. Evaluation & Error Analysis

### 5.1 Metrics

Three metrics are computed on the **real test sessions** (GroupShuffleSplit hold-out, never augmented):

| Metric | Formula | Why used |
|---|---|---|
| **MAE** | `mean(|y − ŷ|)` | Intuitive: average absolute score point error |
| **RMSE** | `sqrt(mean((y − ŷ)²))` | Penalises large individual errors more heavily than MAE |
| **R²** | `1 − SS_res/SS_tot` | Proportion of score variance explained; negative means model is worse than predicting the mean |

### 5.2 Multi-model metric comparison

<p align="center">
  <img src="https://placehold.co/900x420/0d1b2a/ffffff?text=MAE+%2F+RMSE+%2F+R%C2%B2+Grouped+Bar+Charts+%E2%80%94+All+6+Models" alt="Metric comparison bars" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 40, Figure 1</i></sub>
</p>

### 5.3 Actual vs predicted scatter (per model)

For each model, test-set predictions are plotted against true scores. Points are coloured by absolute error (`|y − ŷ|`) using a `YlOrRd` colormap, and each point is annotated with its `Session ID`. The diagonal represents perfect prediction.

<p align="center">
  <img src="https://placehold.co/900x440/0d1b2a/ffffff?text=Actual+vs+Predicted+Scatter+%E2%80%94+All+6+Models+(coloured+by+absolute+error)" alt="Actual vs predicted" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 40, Figure 2</i></sub>
</p>

### 5.4 Residual distribution analysis

Two views of residual structure for all models simultaneously:

- **Overlaid KDE plots** of `actual − predicted`: a narrow peak around zero indicates low bias; heavy tails indicate high-error outlier sessions
- **Box plots of absolute error** per model: quartile spread, median, and whisker extent compared across all 6 models on one axis

<p align="center">
  <img src="https://placehold.co/900x400/0d1b2a/ffffff?text=Residual+KDE+%2B+Absolute+Error+Box+Plots+%E2%80%94+All+6+Models" alt="Residual distributions" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 40, Figure 3</i></sub>
</p>

### 5.5 Per-session residual heatmap

A heatmap where rows = model names and columns = test session IDs, cell colour = residual (`actual − predicted`). Diverging colormap (blue = under-predicted, red = over-predicted). This reveals:

- Sessions that are **systematically mispredicted** by all models (structural data issue or anomalous night)
- Sessions where only **specific models** fail (model-dependent weakness)

<p align="center">
  <img src="https://placehold.co/900x280/0d1b2a/ffffff?text=Per-Session+Residual+Heatmap%3A+Models+%C3%97+Test+Sessions" alt="Residual heatmap" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 40, Figure 4</i></sub>
</p>

### 5.6 Synthesis impact on MAE — full 30-combination comparison

**MAE heatmap (models × scenarios):**

<p align="center">
  <img src="https://placehold.co/900x420/0d1b2a/ffffff?text=MAE+%2F+RMSE+%2F+R%C2%B2+Heatmaps%3A+6+Models+%C3%97+5+Scenarios" alt="Synthesis heatmap" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 44, Figure 2 (3-panel heatmap: MAE, RMSE, R²)</i></sub>
</p>

**ΔMAE line chart** — how much each scenario moves each model's MAE relative to the no-augmentation baseline:

<p align="center">
  <img src="https://placehold.co/900x360/0d1b2a/ffffff?text=%CE%94MAE+Per+Model+vs+Augmentation+Scenario+(line+chart)" alt="Delta MAE" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 44, Figure 3 (line chart; points below y=0 mean augmentation helped)</i></sub>
</p>

**Full results table (MAE on real test sessions):**

|  | No Aug | SMOTER ×3 | SMOTER ×5 | KNN ×3 | KNN ×5 |
|---|---|---|---|---|---|
| Linear Regression | 4.60 | 4.69 | 4.21 | 4.87 | 7.68 |
| Ridge (α=1) | 4.53 | 4.62 | 4.14 | 4.38 | 4.27 |
| Ridge (α=10) | 4.81 | 4.51 | **4.01** | 4.68 | 4.37 |
| Lasso (α=0.5) | 4.67 | 4.52 | 4.44 | 4.87 | 4.61 |
| Random Forest | 2.68 | 1.32 | 1.39 | 1.52 | 1.20 |
| **Gradient Boosting** | 2.59 | 1.71 | 1.62 | 1.09 | **0.93** |

**Key findings:**
- Gradient Boosting + KNN Synth ×5 achieves **MAE = 0.93** — a **64 % improvement** over the no-augmentation baseline (2.59)
- SMOTER ×5 benefits linear models (Ridge α=10 drops from 4.81 → 4.01) but provides less benefit than KNN synthesis for tree models
- KNN ×5 hurts Linear Regression (4.60 → 7.68), suggesting the distance-weighted y-assignment creates label noise that linear models cannot handle — tree models are more robust to this

**Best-per-model summary:**

<p align="center">
  <img src="https://placehold.co/900x300/0d1b2a/ffffff?text=Best+Augmentation+Scenario+per+Model+%E2%80%94+Summary+Table" alt="Best scenario per model" width="860"/>
  <br/><sub><i>Replace with output from notebook cell 44, Figure 4 (printed summary table with Improved? column)</i></sub>
</p>

---

## 6. Possible Applications

| Domain | Application | Relevant features |
|---|---|---|
| **Smart bedroom automation** | Trigger thermostat, smart blinds, and white-noise speaker adjustments automatically when SHAP score predicts a low-quality night | `temp_c` (−), `light_lux` (−), `snd_avg` (−) |
| **Wearable + environment fusion** | Combine predicted environmental score with HRV or SpO₂ from a smartwatch to produce a multi-modal sleep quality index | All 8 features + physiological signals |
| **Clinical sleep studies** | Quantify how controlled environmental interventions (CPAP pressure, room cooling, blackout curtains) affect objective sleep scores without polysomnography | `temp_c`, `light_lux`, `snd_avg` |
| **Hotel & hospitality** | Per-room environmental profiling; offer automatic optimisation presets based on guest preferences or previous-stay scores | `temp_c`, `hum_pct`, `light_lux` |
| **Neonatal & elderly care** | Alert staff when room conditions push predicted SHAP score below a threshold; track environmental drift across shifts | All 8 features, especially `temp_c`, `pm25` |
| **Shift-work & aviation crew** | Simulate different rest-environment scenarios to find the optimal configuration for a worker with irregular schedules | Full pipeline + `optimize_sleep_v2` |
| **Building energy optimisation** | Balance sleep quality score against HVAC energy cost — maintain temperature only when the marginal SHAP benefit exceeds the energy cost | `temp_c`, `hum_pct` |
| **Population longitudinal research** | Passive sensor-based data collection for large-scale environment–sleep studies; no wearable required | Full pipeline scalable to many rooms |

---

## 7. References & Resources

### Methodology papers

| Reference | Relevance |
|---|---|
| Torgo, L., Branco, P., Ribeiro, R. P., & Pfahringer, B. (2013). *SMOTE for Regression*. EPIA 2013, Springer. | SMOTER synthesis algorithm (§3.7) |
| Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 30. | SHAP explainability framework (§4.3) |
| Zou, H. & Hastie, T. (2005). *Regularization and variable selection via the elastic net*. JRSS-B, 67(2), 301–320. | Ridge / Lasso regularisation rationale (§4.1) |
| Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. | Random Forest algorithm (§4.1) |
| Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics, 29(5). | Gradient Boosting algorithm (§4.1) |
| Hubert, M. & Vandervieren, E. (2008). *An adjusted boxplot for skewed distributions*. Computational Statistics & Data Analysis, 52(12). | RobustScaler / IQR scaling rationale (§3.5) |

### Sleep-science baselines (used in research rubric)

| Reference | Used for |
|---|---|
| National Sleep Foundation (2023). *Bedroom environment and sleep quality*. sleepfoundation.org | `temp_c` ideal 18–20 °C, `hum_pct` ideal 40–60 %, `light_lux` ideal < 5 lux |
| WHO (2018). *Environmental Noise Guidelines for the European Region*. WHO Europe. | `snd_avg` ideal ≤ 30 dB at night |
| US EPA (2023). *Indoor Air Quality — Particulate Matter*. epa.gov | `pm25` < 12 µg/m³, `pm10` < 50 µg/m³ |

### Python libraries

| Library | Role |
|---|---|
| `scikit-learn ≥ 1.3` | Preprocessing (`ColumnTransformer`, `RobustScaler`, `SimpleImputer`, `FunctionTransformer`), models, `GroupShuffleSplit`, `Pipeline`, `clone` |
| `shap ≥ 0.44` | `TreeExplainer`, `LinearExplainer`, `KernelExplainer`, `kmeans` background |
| `pandas` / `numpy` | Data loading, session windowing, aggregation, synthesis arithmetic |
| `matplotlib` / `seaborn` | All static notebook visualisations |
| `streamlit ≥ 1.32` | Interactive web dashboard |
| `plotly ≥ 5.0` | Interactive snapshot chart + SHAP bar chart in Streamlit |
| `gspread` + `oauth2client` | Google Sheets authentication and data fetching |
| `requests` | Live IoT sensor API calls |
| `pytz` | UTC → Asia/Bangkok timezone conversion |

### Useful links

- [SHAP documentation](https://shap.readthedocs.io/)
- [scikit-learn user guide — preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn user guide — cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [SMOTER paper (Springer)](https://link.springer.com/chapter/10.1007/978-3-642-40669-0_33)
- [SHAP paper (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- [National Sleep Foundation — bedroom environment](https://www.sleepfoundation.org/bedroom-environment)

---

> **Replacing placeholder images:** Every `<img src="https://placehold.co/...">` block marks where a real notebook output should go.  
> Export a figure from the notebook with:
> ```python
> plt.savefig('assets/fig_name.png', dpi=150, bbox_inches='tight')
> ```
> Create an `assets/` folder at the project root and update the `src` attribute accordingly.
