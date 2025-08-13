# %%
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
from scipy.stats import shapiro
from itertools import product

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pandas_ta as ta
import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib


warnings.filterwarnings('ignore')

# %% [markdown]
# CACHING HELPERS
# - We cache only expensive, deterministic steps (data filters, features, matrices, models)
# - Cache keys include key parameters so caches invalidate when logic/inputs change

import os, json, hashlib
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _hash_key(obj) -> str:
    """Stable short hash for JSON-serializable objects."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=10).hexdigest()

def cache_path(name: str, key: str, ext: str = "pkl") -> Path:
    return CACHE_DIR / f"{name}-{key}.{ext}"

def save_pkl(obj, path: Path):
    joblib.dump(obj, path)

def load_pkl(path: Path):
    return joblib.load(path)

def save_parquet_df(df: pd.DataFrame, path: Path):
    df.to_parquet(path, engine="pyarrow", index=False)

def load_parquet_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

print(f"[cache] directory => {CACHE_DIR}")

# %% [markdown]
# DATA CLEANING
# 
# Data gathered from Kaggle: https://www.kaggle.com/datasets/jakewright/9000-tickers-of-stock-market-data-full-history
# - Covers over 9000 tickers from 1962 to 2024
# - 4.47 GB

# %%
#Using parquet because massive dataset
df = pd.read_parquet("../data/all_stock_data.parquet", engine="pyarrow")
df["Date"] = pd.to_datetime(df["Date"])
df

# %%
df['Volume'].median()

# %%
# Filter for recent rows only (CACHED)
cutoff_days = 45
min_avg_volume = 427_000
min_price = 5

cutoff = df["Date"].max() - timedelta(days=cutoff_days)

_top100_key = _hash_key({
    "cutoff": str(cutoff.date()),
    "min_avg_volume": min_avg_volume,
    "min_price": min_price,
})

summary_cache_path = cache_path("summary-recent", _top100_key, ext="parquet")
top100_cache_path = cache_path("df_top100", _top100_key, ext="parquet")

if top100_cache_path.exists():
    summary = load_parquet_df(summary_cache_path)
    df_top100 = load_parquet_df(top100_cache_path)
    print(f"[cache] loaded df_top100 => {top100_cache_path.name}")
else:
    df_recent = df[df["Date"] >= cutoff]
    df_recent = df_recent[(df_recent["Close"].notna()) & (df_recent["Volume"].notna())]

    summary = (
        df_recent.groupby("Ticker").agg(LastPrice=("Close", "last"), AvgVolume=("Volume", "mean"))
    )
    filtered = summary.query("AvgVolume > @min_avg_volume and LastPrice > @min_price")
    top_tickers = filtered.nlargest(100, "AvgVolume").index.tolist()
    df_top100 = df[df["Ticker"].isin(top_tickers)].reset_index(drop=True)

    save_parquet_df(summary.reset_index(), summary_cache_path)
    save_parquet_df(df_top100, top100_cache_path)
    print(f"[cache] saved df_top100 => {top100_cache_path.name}")

# expose for later cells
summary = summary.set_index("Ticker") if "Ticker" in summary.columns else summary

# %%
df_top100 = df_top100.reset_index(drop=True)
df_top100

# %% [markdown]
# Feature Engineering
# 
# Adding calculated indicators:
# - RSI: relative strength index (momentum)
# - SMA20: 20-day single moving average (trend)
# - MACD: moving average convergence divergence (momentum)
# - ATR: average true range (volitility)
# - Daily Return: percent change in close price
# - Bollinger Band Percent: position within upper/lower bands
# - Price to SMA: price relative to 20 day SMA
# - Close to open ratio (intraday momentum/sentiment)
# - Volume z: z-score normalized rolling 20-day volume
# 
# Then scaling all of the values using StandardScalar()

# %%

# helper: compute per‑ticker technical features used by the pipeline
# expects columns: Date, Open, High, Low, Close, Volume

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("Date").copy()

    # 1) Core indicators
    d["rsi"] = ta.rsi(d["Close"], length=14)
    d["sma20"] = ta.sma(d["Close"], length=20)

    macd_tbl = ta.macd(d["Close"])  # columns include MACD_12_26_9
    if macd_tbl is not None and "MACD_12_26_9" in macd_tbl.columns:
        d["macd"] = macd_tbl["MACD_12_26_9"]
    else:
        d["macd"] = np.nan

    d["atr"] = ta.atr(d["High"], d["Low"], d["Close"], length=14)
    d["daily_return"] = d["Close"].pct_change()

    # 2) Bollinger band percent (0..1 inside the bands)
    bb = ta.bbands(d["Close"], length=20, std=2.0)
    if bb is not None and "BBP_20_2.0" in bb.columns:
        d["bb_percent"] = bb["BBP_20_2.0"]
    elif bb is not None and {"BBL_20_2.0", "BBU_20_2.0"}.issubset(bb.columns):
        d["bb_percent"] = (d["Close"] - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"])
    else:
        d["bb_percent"] = np.nan

    # 3) Price ratios
    d["price_to_sma"] = d["Close"] / d["sma20"]
    d["co_ratio"] = d["Close"] / d["Open"]

    # 4) Volume z‑score using a 20‑day rolling window
    vol = d["Volume"].astype(float)
    vol_mean = vol.rolling(20, min_periods=5).mean()
    vol_std = vol.rolling(20, min_periods=5).std(ddof=0)
    d["volume_z"] = (vol - vol_mean) / vol_std

    return d

features_to_use = [
    "rsi", "sma20", "macd", "atr", "daily_return",
    "bb_percent", "price_to_sma", "co_ratio", "volume_z"
]

_feat_key = _hash_key({
    "features": features_to_use,
    "tickers": sorted(df_top100["Ticker"].unique().tolist())[:10],  # partial key sample for stability
})

features_cache_path = cache_path("df_features", _feat_key, ext="parquet")
scaler_cache_path   = cache_path("scaler", _feat_key, ext="pkl")
X_cache_path        = cache_path("X", _feat_key, ext="pkl")

if features_cache_path.exists() and scaler_cache_path.exists() and X_cache_path.exists():
    df_features = load_parquet_df(features_cache_path)
    scaler = load_pkl(scaler_cache_path)
    X = load_pkl(X_cache_path)
    print("[cache] loaded features, scaler, X")
else:
    df_features = (
        df_top100.groupby("Ticker", group_keys=False)
        .apply(add_indicators)
        .dropna(subset=features_to_use)
        .reset_index(drop=True)
    )
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features = df_features[df_features[features_to_use].abs().lt(1e10).all(axis=1)]
    df_features.dropna(subset=features_to_use, inplace=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_features[features_to_use])
    scaled_cols = [f"{col}_scaled" for col in features_to_use]
    df_features[scaled_cols] = features_scaled

    X = df_features[scaled_cols].values

    save_parquet_df(df_features, features_cache_path)
    save_pkl(scaler, scaler_cache_path)
    save_pkl(X, X_cache_path)
    print("[cache] saved features, scaler, X")

# If loaded from cache, ensure scaled_cols is defined
scaled_cols = [f"{col}_scaled" for col in features_to_use]

# %%
# Define parameter search space
n_obs, n_features = X.shape
n_components_range = range(1, 7)  # Number of regimes to test
covariance_types = ['full', 'diag']  # Covariance matrix types
tolerances = [1e-2, 1e-4, 1e-6]  # Convergence tolerances

results = []

for n_components, cov_type, tol in product(n_components_range, covariance_types, tolerances):
    try:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=cov_type,
            tol=tol,
            n_iter=1000,
            random_state=42
        )
        model.fit(X)
        logL = model.score(X)

        # Estimate number of parameters
        k = n_components * (n_components - 1)  # transition probs
        k += n_components - 1                 # initial probs
        k += n_components * n_features * 2    # means and variances

        bic = -2 * logL + k * np.log(n_obs)
        aic = -2 * logL + 2 * k

        results.append({
            'n_components': n_components,
            'cov_type': cov_type,
            'tol': tol,
            'log_likelihood': logL,
            'AIC': aic,
            'BIC': bic
        })
    except Exception as e:
        results.append({
            'n_components': n_components,
            'cov_type': cov_type,
            'tol': tol,
            'log_likelihood': None,
            'AIC': np.inf,
            'BIC': np.inf,
            'error': str(e)
        })

# Convert to DataFrame and find best config
df_results = pd.DataFrame(results)
best_model = df_results.loc[df_results['BIC'].idxmin()]
print("Best Config (Based on BIC):")
print(best_model[["n_components", "cov_type", "tol", "BIC"]])

# Visualize BIC and AIC
plt.figure(figsize=(10, 5))
for cov in covariance_types:
    subset = df_results[df_results["cov_type"] == cov]
    plt.plot(subset["n_components"], subset["BIC"], label=f"BIC ({cov})", marker="o")
    plt.plot(subset["n_components"], subset["AIC"], label=f"AIC ({cov})", linestyle="--", marker="x")

plt.title("AIC & BIC vs Number of Regimes")
plt.xlabel("Number of Regimes")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
X

# %%
best_model

# %%
# Fit/Predict GaussianHMM (CACHED)
_best_key = _hash_key({
    "n_components": int(best_model['n_components']),
    "cov_type": str(best_model['cov_type']),
    "tol": float(best_model['tol'])
})

gauss_model_path = cache_path("gaussian_hmm", _best_key, ext="pkl")

if gauss_model_path.exists():
    gaussian_model = load_pkl(gauss_model_path)
    print(f"[cache] loaded GaussianHMM => {gauss_model_path.name}")
else:
    gaussian_model = GaussianHMM(
        n_components=int(best_model['n_components']),
        covariance_type=best_model['cov_type'],
        tol=float(best_model['tol']),
        n_iter=1000,
        random_state=42
    )
    gaussian_model.fit(X)
    save_pkl(gaussian_model, gauss_model_path)
    print(f"[cache] saved GaussianHMM => {gauss_model_path.name}")

# Predict the hidden regime for each row
gauss_states = gaussian_model.predict(X)

df_regime = df_features.copy()
df_regime["regime"] = gauss_states

# %% [markdown]
# Assumption Checks
# 
# 1. Stationarity
# 2. Normality
# 3. Independence

# %%
def hmm_assumption_diagnostics(df_regime, features, regime_col="regime", date_col="Date", plot=False):
    """
    Tests HMM assumptions (Stationarity, Normality, Independence) for each feature in each regime.

    Parameters:
        df_regime (pd.DataFrame): DataFrame with features and regime labels.
        features (list): List of feature column names.
        regime_col (str): Column name for regimes.
        date_col (str): Date column for plotting.
        plot (bool): Whether to show rolling mean/std, Q-Q, and ACF plots.
    
    Returns:
        pd.DataFrame: Summary of test results.
    """
    results = []

    for regime in sorted(df_regime[regime_col].unique()):
        regime_data = df_regime[df_regime[regime_col] == regime]
        
        for feature in features:
            series = regime_data[feature].dropna()
            if len(series) < 30:
                continue  # skip small samples

            # --- 1. Stationarity Test (ADF) ---
            try:
                adf_stat, adf_p, *_ = adfuller(series)
            except:
                adf_p = np.nan

            # --- 2. Normality Test (Shapiro-Wilk) ---
            try:
                shapiro_stat, shapiro_p = shapiro(series.sample(min(5000, len(series))))
            except:
                shapiro_p = np.nan

            # --- 3. Independence Test (Ljung-Box) ---
            try:
                lb_test = acorr_ljungbox(series, lags=[10], return_df=True)
                lb_p = lb_test['lb_pvalue'].iloc[0]
            except:
                lb_p = np.nan

            results.append({
                "Regime": regime,
                "Feature": feature,
                "ADF_p (Stationarity)": adf_p,
                "Shapiro_p (Normality)": shapiro_p,
                "LjungBox_p (Independence)": lb_p
            })

            # plotting
            if plot:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                fig.suptitle(f"{feature} - Regime {regime}", fontsize=12)

                # Rolling mean/std
                series.rolling(50).mean().plot(ax=axes[0], title="Rolling Mean")
                series.rolling(50).std().plot(ax=axes[1], title="Rolling Std")

                # Q-Q Plot
                sm.qqplot(series, line='s', ax=axes[2])
                axes[2].set_title("Q-Q Plot")
                plt.show()

                # ACF plot
                sm.graphics.tsa.plot_acf(series, lags=20)
                plt.title("Autocorrelation (ACF)")
                plt.show()

    results_df = pd.DataFrame(results)

    # Mark assumptions (Pass if p > 0.05)
    results_df["Stationary?"] = results_df["ADF_p (Stationarity)"] < 0.05
    results_df["Normal?"] = results_df["Shapiro_p (Normality)"] > 0.05
    results_df["Independent?"] = results_df["LjungBox_p (Independence)"] > 0.05

    return results_df

# %%
scaled_cols

# %%
diagnostic_results = hmm_assumption_diagnostics(
    df_regime,
    features= scaled_cols,
    regime_col="regime",
    date_col="Date",
    plot=True 
)

# %%
diagnostic_results

# %% [markdown]
# Main takeaways: stationarity passes, normality and independence fail
# - independence failing is expected because in a time series, each observation is partly predictable from yesterday's value
# - some characteristics of financial time series includes:
# 	- Autocorrelation: Returns or features (like RSI, MACD, volume) often show serial correlation, meaning today’s value is partly predictable from yesterday’s value.
# 	- Volatility clustering: High-volatility periods (like during market crashes) tend to cluster together, creating time dependence in variance.
# 	- Market microstructure effects: Prices react to order flow, news, and liquidity, creating dependencies between successive time points.
# 
# Next, I want to address the normality problem

# %%
def transform_features_for_normality(df, features):
    """
    Automatically detects skewness and applies transformations to improve normality.
    Prints before/after Shapiro-Wilk p-values for each feature.

    Rules:
      - Winsorize extreme outliers
      - Log transform positive skewed features (volume, ATR)
      - Logit transform bounded [0,1] or [0,100] features (RSI, BB%)
      - Power/Yeo-Johnson transform remaining non-normal features
    """
    df = df.copy()
    summary = []
    
    for feature in features:
        series = df[feature].dropna()

        if len(series) < 30:
            summary.append({"Feature": feature, "Before_p": np.nan, "After_p": np.nan, "Transform": "Skipped"})
            continue

        # --- 1) Initial normality test ---
        try:
            before_p = shapiro(series.sample(min(5000, len(series))))[1]
        except:
            before_p = np.nan

        transformed = series.copy()
        transform_used = "None"

        # --- 2) Winsorize extreme outliers ---
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        transformed = transformed.clip(lower, upper)

        # --- 3) Log transform for volume/ATR or strong positive skew ---
        if (feature.lower().startswith("volume") or "atr" in feature.lower()) and (transformed > 0).all():
            transformed = np.log1p(transformed)
            transform_used = "Log"

        # --- 4) Logit transform for bounded features (0-1 or 0-100) ---
        if transformed.min() >= 0 and transformed.max() <= 1.1:
            transformed = np.log((transformed + 1e-6) / (1 - transformed + 1e-6))
            transform_used = "Logit"
        elif transformed.min() >= 0 and transformed.max() <= 100:
            normed = transformed / 100
            transformed = np.log((normed + 1e-6) / (1 - normed + 1e-6))
            transform_used = "Logit"

        # --- 5) Power Transform if still non-normal ---
        after_first = shapiro(pd.Series(transformed).dropna().sample(min(5000, len(transformed))))[1]
        if after_first < 0.05:
            pt = PowerTransformer(method='yeo-johnson')
            transformed = pt.fit_transform(transformed.values.reshape(-1, 1)).flatten()
            transform_used = transform_used + " + Power" if transform_used != "None" else "Power"

        # Save transformed values back
        df[feature] = transformed

        # --- 6) After transformation test ---
        try:
            after_p = shapiro(df[feature].dropna().sample(min(5000, len(df[feature]))))[1]
        except:
            after_p = np.nan

        summary.append({
            "Feature": feature,
            "Before_p": before_p,
            "After_p": after_p,
            "Transform": transform_used
        })

    # Print summary
    print("\n=== Normality Transformation Summary ===")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    return df, summary_df

# %%
df_regime_normalized, normality_summary = transform_features_for_normality(df_regime, scaled_cols)

# %% [markdown]
# Main Takeaways: despite transformations to the the features, the p-values of the Shapiro-Wilk test are still significantly below 0.05.
# 
# This is more or less normal (haha) in time series because:
# - stock data often has fat tails, skewness, and volatility clustering
# - price returns often follow heavy-tailed distributions, not Gaussian
# 
# So I can only conclude that: 
# - HMM with Gaussian assumptions may not fully capture real data patterns unless a change in the emission distribution
# 
# So my options: 
# - alternative models:
#     - Student-t HMM
#         - deals with fat tails better with degrees of freedom parameter handling tail heaviness but much harder estimating the DoF and higher dimensions makes it harder to to estimate mean/covariance 
#     - Gaussian Mixture HMM
#         - can approximate any non-normal distribution if K is large enough but too big K leads to overfitting and approximates heavy tails
# - More feature engineering 
#     - might lose economic meaning of features
# 
# After some more research, I believe a gaussian mixture tuned similar to student-t may be the best choice for my current implementation

# %% [markdown]
# GAUSSIAN MIXTURE HMM

# %%
from hmmlearn.hmm import GMMHMM

def train_gmmhmm_model(X, n_states=6, n_mix=3, covariance_type='full', n_iter=1000, tol = 0.01, random_state=42):
    """
    Trains a GMMHMM model on the given feature matrix.

    Parameters:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        n_states (int): Number of hidden regimes
        n_mix (int): Number of Gaussian mixtures per regime
        covariance_type (str): Type of covariance ('full', 'diag', 'tied', 'spherical')
        n_iter (int): Maximum number of iterations
        random_state (int): Random seed

    Returns:
        model (GMMHMM): Trained GMMHMM model
        hidden_states (np.ndarray): Predicted regime sequence
    """
    model = GMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol = tol,
        verbose=True
    )
    model.fit(X)
    hidden_states = model.predict(X)
    return model, hidden_states

# %%
# Fit/Predict GMMHMM (CACHED)
_gmm_key = _hash_key({"n_states": 6, "n_mix": 3, "cov": "full", "tol": 0.01})

gmm_model_path = cache_path("gmm_hmm", _gmm_key, ext="pkl")
regimes_gmm_path = cache_path("gmm_states", _gmm_key, ext="pkl")

if gmm_model_path.exists() and regimes_gmm_path.exists():
    gmm_model = load_pkl(gmm_model_path)
    regimes_gmm = load_pkl(regimes_gmm_path)
    print(f"[cache] loaded GMMHMM & states => {gmm_model_path.name}")
else:
    gmm_model, regimes_gmm = train_gmmhmm_model(X, n_states=6, n_mix=3, covariance_type='full', n_iter=1000, tol=0.01)
    save_pkl(gmm_model, gmm_model_path)
    save_pkl(regimes_gmm, regimes_gmm_path)
    print(f"[cache] saved GMMHMM & states => {gmm_model_path.name}")

# attach regimes
if "regime_gmm" not in df_regime.columns:
    df_regime["regime_gmm"] = regimes_gmm

# %% [markdown]
# GMMHMM model converges at the 794 iteration with tolerance set at 0.01. 

# %%
df_regime

# %%
def compare_hmm_models(df, price_col="Close", 
                       gauss_model=None, gmm_model=None, 
                       gauss_states=None, gmm_states=None):
    """
    Compare GaussianHMM and GMMHMM models on multiple metrics.

    Parameters:
        df (pd.DataFrame): DataFrame containing prices and returns
        price_col (str): Column name for price
        gauss_model (GaussianHMM): Trained GaussianHMM model
        gmm_model (GMMHMM): Trained GMMHMM model
        gauss_states (np.ndarray): Hidden states predicted by GaussianHMM
        gmm_states (np.ndarray): Hidden states predicted by GMMHMM

    Returns:
        metrics_df (pd.DataFrame): Comparison of metrics for both models
    """
    results = []

    # Compute log-likelihoods
    feat_cols = [f"{c}_scaled" for c in features_to_use if f"{c}_scaled" in df.columns]
    X_eval = df[feat_cols].values
    ll_gauss = gauss_model.score(X_eval)
    ll_gmm = gmm_model.score(X_eval)
    
    # AIC and BIC for GaussianHMM
    k_gauss = gauss_model.n_components * (gauss_model.n_features * 2)  # rough param count
    aic_gauss = 2*k_gauss - 2*ll_gauss
    bic_gauss = np.log(len(df)) * k_gauss - 2*ll_gauss
    
    # AIC and BIC for GMMHMM
    k_gmm = gmm_model.n_components * gmm_model.n_mix * (gmm_model.n_features * 2)
    aic_gmm = 2*k_gmm - 2*ll_gmm
    bic_gmm = np.log(len(df)) * k_gmm - 2*ll_gmm

    # Calculate returns
    df["return"] = df[price_col].pct_change().fillna(0)

    # Sharpe ratio & mean returns for each regime
    def regime_metrics(states, label):
        temp = df.copy()
        temp["regime"] = states
        metrics = []
        for state in np.unique(states):
            regime_returns = temp.loc[temp["regime"] == state, "return"]
            mean_ret = np.mean(regime_returns)
            std_ret = np.std(regime_returns)
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
            metrics.append({
                "Model": label,
                "Regime": state,
                "Mean_Return": mean_ret,
                "Sharpe_Ratio": sharpe
            })
        return pd.DataFrame(metrics)

    regime_gauss = regime_metrics(gauss_states, "GaussianHMM")
    regime_gmm = regime_metrics(gmm_states, "GMMHMM")

    # Combine results
    model_metrics = pd.DataFrame([
        {"Model": "GaussianHMM", "LogLikelihood": ll_gauss, "AIC": aic_gauss, "BIC": bic_gauss},
        {"Model": "GMMHMM", "LogLikelihood": ll_gmm, "AIC": aic_gmm, "BIC": bic_gmm},
    ])

    metrics_df = {
        "ModelMetrics": model_metrics,
        "RegimeMetrics": pd.concat([regime_gauss, regime_gmm])
    }

    return metrics_df

# %%
metrics = compare_hmm_models(
    df_regime,
    price_col="Close",
    gauss_model=gaussian_model,
    gmm_model=gmm_model,
    gauss_states=gauss_states,
    gmm_states=regimes_gmm
)

print("==== Model-Level Metrics ====")
print(metrics["ModelMetrics"])

print("\n==== Regime-Level Metrics ====")
print(metrics["RegimeMetrics"])

# %% [markdown]
# From this output:
# 
# Global Fit (Model-Level Metrics)
# - GaussianHMM has:
#     - Higher log-likelihood (less negative),
# 	- Lower AIC/BIC,  meaning it fits the entire dataset better overall with fewer penalties for complexity.
# - GMMHMM has:
# 	- Worse log-likelihood,
# 	- Higher AIC/BIC, suggesting the added mixture complexity isn’t improving global model fit enough to offset the penalty for extra parameters.
# 
# Regime-Level Fit (Regime-Level Metrics)
# 
# Here’s where it gets interesting:
# - GMMHMM absolutely dominates in some regimes:
# - Regime 1: Sharpe Ratio 3.95 vs Gaussian’s 0.17, a huge jump in quality.
# - Regime 4: Sharpe 2.37 vs Gaussian’s 1.78,  meaningful improvement.
# - In other regimes, it’s roughly similar or worse.
# 
# Interpretation
# - GaussianHMM is the better global fit (lower complexity cost, more stable overall likelihood).
# - GMMHMM is the better regime specialist — it nails certain states with extremely high returns and Sharpe ratios but at the cost of worse overall model metrics.
# - For financial regime detection, we often care less about perfect global fit and more about identifying high-performance regimes, making the GMMHMM results might actually be more actionable, especially if we only trade in its high-Sharpe regimes.

# %%
# ---------- 1) Regime stats (Sharpe, mean) ----------

def regime_stats(df, regimes, price_col="Close"):
    temp = df.copy()
    temp["regime"] = regimes
    temp["ret"] = temp[price_col].pct_change().fillna(0.0)

    rows = []
    for s in np.unique(regimes):
        r = temp.loc[temp["regime"] == s, "ret"]
        mean = r.mean()
        std  = r.std()
        sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0.0
        rows.append({"regime": int(s), "mean": mean, "std": std, "sharpe": sharpe, "count": len(r)})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# %%
# ---------- 2) Pick tradeable regimes ----------
def select_trade_regimes(stats_df, top_k=2, sharpe_min=None):
    ranked = stats_df.sort_values("sharpe", ascending=False)
    if sharpe_min is not None:
        ranked = ranked[ranked["sharpe"] >= sharpe_min]
    if top_k is not None:
        ranked = ranked.head(top_k)
    return ranked["regime"].tolist(), ranked

# %%
# ---------- 3) Backtest: long when regime in trade_regimes and confidence ≥ min_prob ----------
def backtest_regime_filter(df, price_col, regimes, model, X, trade_regimes,
                           min_prob=0.0, tc_bps=1.0):
    """
    min_prob: require max posterior prob ≥ min_prob (e.g., 0.6) to take a position
    tc_bps: round-trip transaction cost in basis points deducted on position flips
    """
    temp = df.copy()
    temp["ret"] = temp[price_col].pct_change().fillna(0.0)

    # Posterior probabilities (confidence)
    post = model.predict_proba(X)         # shape: (n_samples, n_states)
    max_prob = post.max(axis=1)

    # Position: 1 when regime in trade set AND confidence high, else 0 (flat)
    pos_raw = np.where((np.isin(regimes, trade_regimes)) & (max_prob >= min_prob), 1.0, 0.0)
    temp["position"] = pd.Series(pos_raw, index=temp.index)

    # One-bar delay to avoid look-ahead
    temp["position_shift"] = temp["position"].shift(1).fillna(0.0)

    # Transaction cost when position flips (enter/exit)
    flips = temp["position_shift"].diff().abs().fillna(0.0)
    tc = (tc_bps / 10000.0) * flips

    # Strategy return
    temp["strat_ret"] = temp["position_shift"] * temp["ret"] - tc

    # Cumulative curves
    temp["cum_strat"] = (1.0 + temp["strat_ret"]).cumprod()
    temp["cum_bh"]    = (1.0 + temp["ret"]).cumprod()

    # Quick stats
    def annualized_stats(r):
        mean = r.mean() * 252
        vol  = r.std() * np.sqrt(252)
        sharpe = mean / vol if vol > 0 else 0.0
        dd = (temp["cum_strat"].cummax() - temp["cum_strat"]).max()
        return mean, vol, sharpe, dd

    ann_mean, ann_vol, ann_sharpe, max_dd = annualized_stats(temp["strat_ret"])

    perf = {
        "trade_regimes": trade_regimes,
        "min_prob": min_prob,
        "tc_bps": tc_bps,
        "final_cum_strategy": float(temp["cum_strat"].iloc[-1]),
        "final_cum_buyhold": float(temp["cum_bh"].iloc[-1]),
        "ann_return": ann_mean,
        "ann_vol": ann_vol,
        "sharpe": ann_sharpe,
        "max_drawdown": max_dd
    }

    # Plot
    plt.figure(figsize=(11,5))
    plt.plot(temp.index, temp["cum_bh"], label="Buy & Hold", linestyle="--")
    plt.plot(temp.index, temp["cum_strat"], label="Regime Filter (long only)")
    plt.title("Cumulative Returns: Regime Filter vs Buy & Hold")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return temp, perf

# %%
# 4a) Compute regime stats for the GMMHMM predictions
gmm_stats = regime_stats(df_regime, regimes_gmm, price_col="Close")
print("GMMHMM regime stats (sorted by Sharpe):\n", gmm_stats, "\n")

# 4b) Choose tradeable regimes (example: top 2 by Sharpe, require Sharpe ≥ 0.5)
trade_regimes, chosen = select_trade_regimes(gmm_stats, top_k=2, sharpe_min=0.5)
print("Chosen regimes:", trade_regimes)
print(chosen, "\n")

# 4c) Backtest with optional confidence filter and transaction costs
bt_df, perf = backtest_regime_filter(
    df=df_regime,
    price_col="Close",
    regimes=regimes_gmm,
    model=gmm_model,
    X=X,                    # SAME features (rows aligned to df_regime.index)
    trade_regimes=trade_regimes,
    min_prob=0.60,          # e.g., only take trades when model is ≥60% confident
    tc_bps=2.0              # 2 bps per flip (enter/exit combined)
)

print("Performance summary:\n", pd.Series(perf))

# %% [markdown]
# From the results, a few things stand out:
# 
# 1. Regime stats
# - the GMMHMM found Regime 1 and Regime 4 as having the highest Sharpe ratios.
# - These were chosen for trading (trade_regimes = [1, 4]).
# - Regime 1 especially has an extremely high mean return and Sharpe ratio, which is suspiciously large and might be a result of outliers or very short periods with explosive returns (only 124 data points).
# 
# 2. Backtest results
# - final_cum_strategy = 0.0 means your regime-filtered strategy essentially went to zero, it completely wiped out at some point.
# - Meanwhile, final_cum_buyhold is huge (10,845x growth), suggesting your dataset has some extreme price appreciation (maybe a long historical index or stock run).
# - The annualized Sharpe for your strategy is negative (-0.96), so it’s underperforming random chance.
# - Max drawdown = 1.0 means the strategy lost 100% of equity at some point, again, full wipeout.
# 
# 3. Why the chart looks wrong (aside from bad scaling)
# - The Buy & Hold curve explodes upward because of your extreme mean values, it likely hit an exponential rally.
# - The Regime Filter line is flat at zero because once your equity goes to zero, cumulative returns can’t recover.
# - Your chosen regimes likely picked up very short-lived, volatile spikes, but in reality these didn’t persist long enough to profit.
# 
# 4. Potential reasons why this is happening
# - Overfitting to high returns: GMMHMM chose the regimes with the biggest spikes, not the most sustainable.
# - Mean/variance distortion: High mean return with huge std means unstable, even if Sharpe seems good.
# - Confidence filter (min_prob=0.6) + limited trades might mean you barely took any trades, so one or two losses killed the portfolio.
# - Possible lookahead bias if the regime classification is from the whole dataset without walk-forward training.
# 
# 5. Next Steps
# 	1.	Use walk-forward or train/test split, avoid fitting the model on all historical data.
# 	2.	Inspect regime returns in time-series form, see when regimes 1 and 4 appear and if they make sense visually.
# 	3.	Avoid trading only 1–2 regimes, start with a broader set (top 50% Sharpe) to ensure diversification.
# 	4.	Lower mean return outliers, winsorize or remove unrealistic returns before calculating Sharpe.
# 	5.	Run the same backtest on GaussianHMM, compare stability.
# 	6.	Reduce min_prob, maybe 0.4 to increase trade frequency and reduce catastrophic wipeouts.

# %%
# ----- helpers -----
def _posterior_max(model, X):
    proba = model.predict_proba(X)
    return proba.max(axis=1), proba

def _build_positions(regimes, trade_regimes, max_prob, min_prob):
    return np.where((np.isin(regimes, trade_regimes)) & (max_prob >= min_prob), 1.0, 0.0)

def _trades_from_position(idx, position_shift):
    flips = position_shift.diff().fillna(position_shift.iloc[0]).values
    pos = position_shift.values
    dates = idx

    trades = []
    open_i = None
    for i in range(len(pos)):
        if flips[i] > 0 and pos[i] == 1:     # flat -> long
            open_i = i
        elif flips[i] < 0 and pos[i] == 0:   # long -> flat
            if open_i is not None:
                trades.append((dates[open_i], dates[i], 1))
                open_i = None
    if open_i is not None:
        trades.append((dates[open_i], dates[-1], 1))
    return trades

def _per_trade_stats(df, price_col, trades, tc_bps):
    rows = []
    for (entry_dt, exit_dt, side) in trades:
        sl = df.loc[entry_dt:exit_dt]
        if len(sl) < 2:
            continue
        p0 = sl[price_col].iloc[0]
        p1 = sl[price_col].iloc[-1]
        r = (p1/p0 - 1.0) * side
        r -= 2 * (tc_bps / 10000.0)
        curve = (sl[price_col] / p0 - 1.0) * side
        rows.append({
            "entry": entry_dt, "exit": exit_dt, "bars": len(sl)-1,
            "side": "long", "ret": r,
            "MFE": curve.max(), "MAE": curve.min()
        })
    return pd.DataFrame(rows)

# helper to pick a single ticker while keeping arrays aligned
def slice_ticker(df, X, regimes, ticker_col, ticker):
    mask = (df[ticker_col] == ticker).values
    df_s = df.loc[mask].copy()
    X_s = X[mask] if isinstance(X, np.ndarray) else X.loc[mask].values
    r_s = regimes[mask] if isinstance(regimes, np.ndarray) else np.asarray(regimes)[mask]
    return df_s, X_s, r_s

# %%
# ----- main backtest + plotly figs -----
def backtest_regime_filter_plotly(
    df, price_col, regimes, model, X, trade_regimes,
    min_prob=0.60, tc_bps=2.0, regime_alpha=0.10,
    ticker: str | None = None,            # used in titles only
    allow_short: bool = False,            # enable shorting
    short_regimes: list[int] | None = None, # which regimes to short (if None, none)
    size_by_prob: bool = False,           # size position by max posterior prob
    marker_size: int = 12                  # make trade signals bigger
):
    temp = df.copy()
    temp["ret"] = temp[price_col].pct_change().fillna(0.0)

    # confidence
    max_prob, post = _posterior_max(model, X)

    # build position allowing long, optional short, optional sizing
    reg = np.asarray(regimes)
    long_mask  = np.isin(reg, trade_regimes) & (max_prob >= min_prob)
    short_mask = np.zeros_like(long_mask, dtype=bool)
    if allow_short and short_regimes:
        short_mask = np.isin(reg, np.array(short_regimes)) & (max_prob >= min_prob)

    pos_raw = np.zeros_like(max_prob, dtype=float)
    pos_raw[long_mask]  = 1.0
    pos_raw[short_mask] = -1.0

    # optionally size by posterior probability (confidence‑weighted)
    if size_by_prob:
        pos_raw = pos_raw * max_prob

    temp["position"] = pd.Series(pos_raw, index=temp.index)
    # one‑bar delay to execute next bar
    temp["position_shift"] = temp["position"].shift(1).fillna(0.0)

    # costs proportional to position change magnitude (works for sized positions)
    delta_pos = temp["position_shift"].diff().abs().fillna(temp["position_shift"].abs())
    tc = (tc_bps / 10000.0) * delta_pos

    # strategy return
    temp["strat_ret"] = temp["position_shift"] * temp["ret"] - tc
    temp["cum_strat"] = (1.0 + temp["strat_ret"]).cumprod()
    temp["cum_bh"]    = (1.0 + temp["ret"]).cumprod()

    # trades from executed position
    trades = _trades_from_position(temp.index, temp["position_shift"])
    trades_df = _per_trade_stats(temp, price_col, trades, tc_bps)

    # summary
    ann_ret = temp["strat_ret"].mean() * 252
    ann_vol = temp["strat_ret"].std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    max_dd  = float((temp["cum_strat"].cummax() - temp["cum_strat"]).max())
    exposure = float(temp["position_shift"].abs().mean())  # abs for sized/short

    perf = {
        "ticker": ticker,                            
        "trade_regimes": trade_regimes,
        "short_regimes": short_regimes if allow_short else [],
        "size_by_prob": size_by_prob,
        "min_prob": min_prob,
        "tc_bps": tc_bps,
        "final_cum_strategy": float(temp["cum_strat"].iloc[-1]),
        "final_cum_buyhold": float(temp["cum_bh"].iloc[-1]),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["ret"] > 0).mean()) if not trades_df.empty else np.nan,
        "avg_trade": float(trades_df["ret"].mean()) if not trades_df.empty else np.nan,
        "best_trade": float(trades_df["ret"].max()) if not trades_df.empty else np.nan,
        "worst_trade": float(trades_df["ret"].min()) if not trades_df.empty else np.nan,
        "exposure": exposure
    }

    # ---------- Plotly figure 1: Price with regime shading + trade markers ----------
    fig_price = go.Figure()

    ttl_ticker = f" — {ticker}" if ticker else ""
    fig_price.add_trace(go.Scatter(
        x=temp.index, y=temp[price_col],
        mode="lines", name="Close", line=dict(width=1.5, color="black")
    ))

    # regime shading (tradeable regimes only)
    r_series = pd.Series(reg, index=temp.index)
    for r in trade_regimes + (short_regimes if allow_short and short_regimes else []):
        mask = (r_series == r).values
        in_block = False
        start = None
        for i, on in enumerate(mask):
            if on and not in_block:
                in_block = True
                start = temp.index[i]
            last = (i == len(mask)-1)
            if (not on or last) and in_block:
                end = temp.index[i] if not on else temp.index[-1]
                color = "rgba(33,150,243,0.18)" if r in trade_regimes else "rgba(244,67,54,0.18)"  # CHANGED: blue for long, red for short
                fig_price.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0, opacity=regime_alpha, layer="below")
                in_block = False

    # bigger markers + separate long/short markers
    if not trades_df.empty:
        long_entries = trades_df.loc[trades_df["side"]=="long", "entry"]
        long_exits   = trades_df.loc[trades_df["side"]=="long", "exit"]
        short_entries= trades_df.loc[trades_df["side"]=="short","entry"]
        short_exits  = trades_df.loc[trades_df["side"]=="short","exit"]

        if len(long_entries):
            fig_price.add_trace(go.Scatter(
                x=long_entries, y=temp.loc[long_entries, price_col],
                mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", size=marker_size, color="green"),
                hovertemplate="Long Entry %{x|%Y-%m-%d}<extra></extra>"
            ))
        if len(long_exits):
            fig_price.add_trace(go.Scatter(
                x=long_exits, y=temp.loc[long_exits, price_col],
                mode="markers", name="Long Exit",
                marker=dict(symbol="triangle-down", size=marker_size, color="darkgreen"),
                hovertemplate="Long Exit %{x|%Y-%m-%d}<extra></extra>"
            ))
        if len(short_entries):
            fig_price.add_trace(go.Scatter(
                x=short_entries, y=temp.loc[short_entries, price_col],
                mode="markers", name="Short Entry",
                marker=dict(symbol="x", size=marker_size+2),
                hovertemplate="Short Entry %{x|%Y-%m-%d}<extra></extra>"
            ))
        if len(short_exits):
            fig_price.add_trace(go.Scatter(
                x=short_exits, y=temp.loc[short_exits, price_col],
                mode="markers", name="Short Exit",
                marker=dict(symbol="x-thin-open", size=marker_size+2),
                hovertemplate="Short Exit %{x|%Y-%m-%d}<extra></extra>"
            ))

    fig_price.update_layout(
        title=f"Price with Tradeable Regimes and Trade Markers{ttl_ticker}",
        xaxis_title="Date", yaxis_title=price_col,
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    # ---------- Plotly figure 2: Equity curves ----------
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=temp.index, y=temp["cum_bh"], mode="lines",
        name="Buy & Hold", line=dict(dash="dash")
    ))
    fig_equity.add_trace(go.Scatter(
        x=temp.index, y=temp["cum_strat"], mode="lines",
        name="Regime Filter", line=dict(width=2)
    ))
    fig_equity.update_layout(
        title=f"Cumulative Returns: Regime Filter vs Buy & Hold{ttl_ticker}",
        xaxis_title="Date", yaxis_title="Cumulative Return",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return temp, trades_df, perf, fig_price, fig_equity

# %%
# choose a ticker from df_regime["Ticker"]
chosen_ticker = "AAPL"  # <- set this

# Slice data consistently
df_t, X_t, r_t = slice_ticker(df_regime, X, regimes_gmm, ticker_col="Ticker", ticker=chosen_ticker)

# Choose regimes to long/short (you probably already picked trade_regimes earlier)
# e.g., trade_regimes from your Sharpe ranking; optionally define short_regimes:
short_regs = []  # e.g., [2, 5] if you want to short those

bt_df, trades_df, perf, fig_price, fig_equity = backtest_regime_filter_plotly(
    df=df_t,
    price_col="Close",
    regimes=r_t,
    model=gmm_model,
    X=X_t,
    trade_regimes=trade_regimes,      # your top regimes to go long
    short_regimes=short_regs,         # optional shorting
    allow_short=bool(short_regs),     # enable if you passed some
    size_by_prob=True,                # confidence‑weighted sizing
    min_prob=0.55,                    # confidence gate (tune 0.4–0.7)
    tc_bps=3.0,                       # higher TC if you want
    regime_alpha=0.14,                # slightly stronger shading
    marker_size=14,                   # bigger markers
    ticker=chosen_ticker              # appears in titles + perf
)

print("Performance summary:\n", pd.Series(perf))
fig_price.show()
fig_equity.show()
trades_df.head()

# %%
# (cached) models & data already saved above; skip duplicate dumps
print("[cache] artifacts persisted under:", CACHE_DIR)


# %%
gmm_model

# %%
gaussian_model

# %%
"""
things to continue:

1. fix error above
2. implement ability to choose ticker
3.5 build out STEP 7
4. make the trade signals bigger. cant tell from the graph

	•	Shorting in bad regimes: let position be -1 in low‑Sharpe regimes.
	•	Position sizing: scale by posterior probability (e.g., position = max_prob within trade regimes).
	•	Stop trading if confidence low: you already have min_prob, tune it!
	•	Walk‑forward: fit on a rolling window and predict out‑of‑sample for tighter realism.

heres a catch up on current thought process:
- .venv(Python 3.11.5) is my environment and kernal
- STEP 1: loaded and clean historical data from parquet
- STEP 2: feature engineering. adding all the features: features_to_use = ["rsi", "sma20", "macd", "atr", "daily_return", "bb_percent", "price_to_sma", "co_ratio", "volume_z"]
- STEP 3: hyperparameter tune for best params
- STEP 4: build and validate the HMM model
- STEP 5: create comprehensive graph going over confidence level, regimes, and trade signals 
- STEP 6: figure out what each regime means. Regimes are hidden patterns, doesnt tell us what they are just that a pattern exists. All the regime 1 is buy, regime 5 is sell, other regimes are hold.
- STEP 7: backtest (NEED TO DO)
- STEP 8: download current data and figure out what regime we're currently in. 
"""


