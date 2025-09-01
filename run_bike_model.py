# Predicting hourly bike rentals (cnt) — Short report + runnable code
'''

## How to run

1. Make sure Python 3.8+ is installed.
2. Create and activate a virtualenv.
3. Install dependencies: `pip install -r requirements.txt` (requirements included below).
4. Run `python run_bike_model.py` 



## Files inside this single-source deliverable

* `run_bike_model.py` — main script containing ingestion, EDA, featurization, training, evaluation, and model export.
* `requirements.txt` — required python packages.


```python
#!/usr/bin/env python3
"""
run_bike_model.py

Single-file pipeline to reproduce EDA, model training and export for the UCI Bike Sharing hourly dataset.
"""

'''

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

# lightgbm is optional if not installed; pip install lightgbm
try:
    import lightgbm as lgb
except Exception:
    lgb = None


DATA_URL = 'C:/Users/HP/Downloads/bike+sharing+dataset/hour.csv' #please include the path
OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / 'plots').mkdir(exist_ok=True)


def download_and_load(url=DATA_URL):
    """Downloads hour.csv and loads into a DataFrame."""
    local = OUT_DIR / 'hour.csv'
    if not local.exists():
        print('Downloading dataset...')
        try:
            import requests
            r = requests.get(url)
            r.raise_for_status()
            local.write_bytes(r.content)
        except Exception:
            # fallback to pandas read_csv directly (requests may not be available)
            print('Fallback: reading directly with pandas')
            df = pd.read_csv(url)
            df.to_csv(local, index=False)
            return df
    return pd.read_csv(local)


def basic_eda(df):
    print('Rows, cols:', df.shape)
    print(df.describe(include='all'))

    # Convert date
    df['dteday'] = pd.to_datetime(df['dteday'])
    df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)

    # ✅ FIX: build datetime column properly
    df['datetime'] = pd.to_datetime(df['dteday'].dt.strftime("%Y-%m-%d") + ' ' + df['hr'].astype(str) + ':00')

    # plot hourly cnt over time (sample)
    fig, ax = plt.subplots(figsize=(10, 3))
    df.set_index('datetime')['cnt'].rolling(24).mean().plot(ax=ax)
    ax.set_title('24-hour rolling mean of cnt (sample)')
    plt.tight_layout(); plt.savefig(OUT_DIR / 'plots' / 'rolling_cnt.png')

    # hourly distribution
    fig, ax = plt.subplots(figsize=(8,4))
    df.groupby('hr')['cnt'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Average rentals by hour')
    plt.tight_layout(); plt.savefig(OUT_DIR / 'plots' / 'avg_by_hour.png')

    # weekday pattern
    fig, ax = plt.subplots(figsize=(8,4))
    df.groupby('weekday')['cnt'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Average rentals by weekday (0=Sun)')
    plt.tight_layout(); plt.savefig(OUT_DIR / 'plots' / 'avg_by_weekday.png')

    return df


def featurize(df):
    # prepare base features
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.set_index('datetime')

    # categorical features as category dtype
    cat_cols = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']
    for c in cat_cols:
        df[c] = df[c].astype('category')

    # cyclic encoding for hour and month and weekday
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'].astype(int) / 24)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'].astype(int) / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['mnth'].astype(int) / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['mnth'].astype(int) / 12)

    # lag features: previous 1h, 24h and rolling mean 24h
    df['cnt_lag_1'] = df['cnt'].shift(1)
    df['cnt_lag_24'] = df['cnt'].shift(24)
    df['cnt_roll_24_mean'] = df['cnt'].shift(1).rolling(24, min_periods=1).mean()

    # use temperature/humidity/windspeed/atemp as-is
    feature_cols = [
        'season','yr','mnth','hr','holiday','weekday','workingday','weathersit',
        'temp','atemp','hum','windspeed',
        'hr_sin','hr_cos','month_sin','month_cos',
        'cnt_lag_1','cnt_lag_24','cnt_roll_24_mean'
    ]

    df = df.dropna(subset=['cnt_lag_1'])
    return df, feature_cols


def train_model(df, feature_cols):
    # time-based split: train on all data before last 60 days, validate on last 60 days
    last_date = df.index.max()
    val_start = last_date - pd.Timedelta(days=60)
    train_df = df[df.index < val_start]
    val_df = df[df.index >= val_start]

    X_train = train_df[feature_cols]
    y_train = train_df['cnt']
    X_val = val_df[feature_cols]
    y_val = val_df['cnt']

    print('Train shape:', X_train.shape, 'Val shape:', X_val.shape)

    # LightGBM training
    if lgb is None:
        raise RuntimeError('lightgbm is not installed. Please pip install lightgbm')

    categorical_features = [c for c in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit'] if c in feature_cols]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features, free_raw_data=False)

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }

    model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    num_boost_round=500,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
    )

    # predict and evaluate
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_val, y_pred)
    print(f'Validation MAE: {mae:.3f} bikes')

    # save model
    joblib.dump({'model': model, 'feature_cols': feature_cols}, OUT_DIR / 'model.joblib')
    print('Saved model to', OUT_DIR / 'model.joblib')

    # feature importance plot
    fi = pd.DataFrame({'feature': model.feature_name(), 'importance': model.feature_importance()}).sort_values('importance', ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    fi.set_index('feature').head(20).plot.bar(ax=ax, legend=False)
    ax.set_title('Feature importance (gain)')
    plt.tight_layout(); plt.savefig(OUT_DIR / 'plots' / 'feature_importance.png')

    # residual plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(y_val, y_pred, alpha=0.3)
    ax.set_xlabel('Actual cnt'); ax.set_ylabel('Predicted cnt'); ax.set_title('Predicted vs Actual')
    plt.tight_layout(); plt.savefig(OUT_DIR / 'plots' / 'pred_vs_actual.png')

    return model, mae



def predict_next_24h(model, df, feature_cols):
    df = df.copy()
    df['hr'] = pd.to_numeric(df['hr'], errors='coerce')
    df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)

    last_time = df['dteday'].max()
    last_hour = df[df['dteday'] == last_time]['hr'].astype(int).max()

    preds = []
    current_date = pd.to_datetime(last_time)
    current_hour = last_hour

    # List of categorical features used during training
    categorical_features = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']

    for i in range(24):
        current_hour += 1
        if current_hour > 23:
            current_hour = 0
            current_date += pd.Timedelta(days=1)

        row = {}
        for col in feature_cols:
            if col == 'season':
                row[col] = df['season'].mode()[0]
            elif col == 'yr':
                row[col] = current_date.year - 2011
            elif col == 'mnth':
                row[col] = current_date.month
            elif col == 'hr':
                row[col] = current_hour
            elif col == 'weekday':
                row[col] = current_date.weekday()
            elif col == 'workingday':
                row[col] = 1 if current_date.weekday() < 5 else 0
            elif col == 'holiday':
                row[col] = 0
            elif col == 'weathersit':
                row[col] = 1
            elif col in ['temp', 'atemp', 'hum', 'windspeed']:
                row[col] = df[col].mean()
            elif col == 'hr_sin':
                row[col] = np.sin(2 * np.pi * current_hour / 24)
            elif col == 'hr_cos':
                row[col] = np.cos(2 * np.pi * current_hour / 24)
            elif col == 'month_sin':
                row[col] = np.sin(2 * np.pi * current_date.month / 12)
            elif col == 'month_cos':
                row[col] = np.cos(2 * np.pi * current_date.month / 12)
            elif col == 'cnt_lag_1':
                row[col] = df['cnt'].iloc[-1]
            elif col == 'cnt_lag_24':
                row[col] = df['cnt'].iloc[-24] if len(df) >= 24 else df['cnt'].iloc[0]
            elif col == 'cnt_roll_24_mean':
                row[col] = df['cnt'].iloc[-24:].mean() if len(df) >= 24 else df['cnt'].mean()
            else:
                # fallback to 0 for unknown feature
                row[col] = 0

        X = pd.DataFrame([row])[feature_cols]

        # Convert categorical features to category dtype
        for cat_col in categorical_features:
            if cat_col in X.columns:
                X[cat_col] = X[cat_col].astype('category')

        # Predict
        y_pred = model.predict(X)[0]
        preds.append((current_date, current_hour, y_pred))

        # Append predicted value to df for next lag calculation
        df = pd.concat([df, pd.DataFrame({'cnt': [y_pred]})], ignore_index=True)

    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(preds, columns=['dteday', 'hr', 'predicted_cnt'])
    print("\nNext 24 hours predictions:")
    print(pred_df)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(pred_df['hr'], pred_df['predicted_cnt'], marker='o', linestyle='-', color='b')
    plt.title("Predicted Bike Usage for Next 24 Hours")
    plt.xlabel("Hour of Day")
    plt.ylabel("Predicted Count")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(range(0, 24))
    plt.show()

    return pred_df







if __name__ == '__main__':
    df = download_and_load()
    df = basic_eda(df)
    df, feature_cols = featurize(df)
    model, mae = train_model(df, feature_cols)
    # demonstrate prediction
    predict_next_24h(model, df, feature_cols)








