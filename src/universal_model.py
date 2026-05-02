import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

TRAINING_STOCKS = [
    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
    'RELIANCE.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS',
    'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS'
]

def build_universal_dataset(stocks: list = TRAINING_STOCKS) -> pd.DataFrame:
    """
    Fetch and process all sotvks and combine it into ons dataset
    this is the training data for the universal model
    """

    from data_pipeline import fetch_multiple_stocks
    from features import build_features
    from targets import add_targets

    logger.info(f"Building universal dataset from {len(stocks)} stocks...")

    all_dfs = []
    stock_data = fetch_multiple_stocks(stocks)

    for ticker, df_raw in stock_data.items():
        try:
            logger.info(f"Processing {ticker}...")
            df = build_features(df_raw, ticker=ticker)
            df = add_targets(df)
            df['ticker'] = ticker

            all_dfs.append(df)
            logger.info(f"  {ticker}: {len(df)} rows after processing")

        except Exception as e:
            logger.warning(f".  {ticker}:; processing failed - {e}")
            import traceback
            traceback.print_exc()

    if not all_dfs:
        raise ValueError("No stocks succesfully processed")
    
    combined = pd.concat(all_dfs, axis = 0)

    combined = combined.sort_index()

    logger.info(f"Universal dataset: {len(combined)} total rows")
    logger.info(f"Stocks included: {combined['ticker'].nunique()}")
    logger.info(f"Date range: {combined.index[0].date()} → {combined.index[-1].date()}")

    return combined

def temporal_split_universal(df:pd.DataFrame,
                             test_size: float = 0.2):
    
    """
    Temporal split for multi_stock dataset,
    Split by Date not by row ensures no future leakage
    All stocks data after the cutoff date goes to test
    """

    all_dates = df.index.unique().sort_values()
    split_idx = int(len(all_dates) * (1 - test_size))
    cutoff = all_dates[split_idx]

    train = df[df.index < cutoff]
    test = df[df.index >= cutoff]

    logger.info(f"Universal split cutoff: {cutoff.date()}")
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    logger.info(f"Train stocks: {train['ticker'].nunique()} | "
                f"Test stocks: {test['ticker'].nunique()}")

    return train, test


def train_universal_model(df: pd.DataFrame) -> tuple:
    """
    Train the universal model on combined multi-stock data.
    Returns trained RF, XGBoost, scaler and feature columns.
    """
    from model import (get_feature_columns, train_random_forest,
                       train_xgboost)

    # Get feature columns — exclude ticker column
    feature_cols = get_feature_columns(df)

    # Remove ticker from features if it snuck in
    feature_cols = [c for c in feature_cols if c != 'ticker']

    train_df, test_df = temporal_split_universal(df)

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    y_train = train_df['target_direction_1d']
    y_test  = test_df['target_direction_1d']

    # Scale
    scaler  = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols, index=X_test.index
    )

    logger.info(f"Training universal model on {len(X_train)} rows, "
                f"{len(feature_cols)} features")

    rf_model  = train_random_forest(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)

    # Evaluate
    rf_prob  = rf_model.predict_proba(X_test_scaled)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2
    ens_pred = (ens_prob >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print("UNIVERSAL MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Test rows: {len(y_test)} across {test_df['ticker'].nunique()} stocks")
    print(f"\nEnsemble Accuracy:  {accuracy_score(y_test, ens_pred)*100:.2f}%")
    print(f"Ensemble Precision: {precision_score(y_test, ens_pred, zero_division=0)*100:.2f}%")
    print(f"Ensemble Recall:    {recall_score(y_test, ens_pred, zero_division=0)*100:.2f}%")
    print(f"Ensemble F1:        {f1_score(y_test, ens_pred, zero_division=0)*100:.2f}%")

    # Per-stock breakdown
    print(f"\n--- Per-Stock Accuracy ---")
    for ticker in test_df['ticker'].unique():
        mask     = test_df['ticker'] == ticker
        y_t      = y_test[mask]
        y_p      = ens_pred[mask.values]
        acc      = accuracy_score(y_t, y_p) * 100
        baseline = y_t.mean() * 100
        print(f"  {ticker:<20} {acc:.1f}%  (baseline: {baseline:.1f}%)")

    return rf_model, xgb_model, scaler, feature_cols, test_df


def save_universal_model(rf_model, xgb_model,
                          scaler, feature_cols,
                          path: str = "models"):
    """Save the universal model."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(rf_model,     f"{path}/universal_rf.pkl")
    joblib.dump(xgb_model,    f"{path}/universal_xgb.pkl")
    joblib.dump(scaler,       f"{path}/universal_scaler.pkl")
    joblib.dump(feature_cols, f"{path}/universal_features.pkl")
    logger.info(f"Universal model saved to {path}/")


def load_universal_model(path: str = "models",
                          use_selected: bool = True):
    """Load the universal model — selected features by default."""
    if use_selected and os.path.exists(f"{path}/universal_rf_selected.pkl"):
        suffix = "_selected"
    else:
        suffix = ""

    rf_model     = joblib.load(f"{path}/universal_rf{suffix}.pkl")
    xgb_model    = joblib.load(f"{path}/universal_xgb{suffix}.pkl")
    scaler       = joblib.load(f"{path}/universal_scaler{suffix}.pkl")
    feature_cols = joblib.load(f"{path}/universal_features{suffix}.pkl")

    label = "selected" if suffix else "full"
    logger.info(f"Universal model loaded ({label} features, {len(feature_cols)} features)")

    return rf_model, xgb_model, scaler, feature_cols


def predict_with_universal_model(ticker: str,
                                  rf_model, xgb_model,
                                  scaler, feature_cols,
                                  force_refresh: bool = False) -> dict:
    """
    Make prediction for any stock using the universal model.
    This replaces the per-stock training in main.py.
    """
    from data_pipeline import get_data
    from features import build_features
    from targets import add_targets
    from risk import add_risk_to_predictions
    from explainability import (calculate_shap_values,
                                generate_explanation,
                                get_top_drivers)

    logger.info(f"Predicting {ticker} with universal model...")

    df_raw      = get_data(ticker, force_refresh=force_refresh)
    df_features = build_features(df_raw, ticker=ticker)
    df          = add_targets(df_features)

    # Add any missing sector columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].tail(500)  # use recent data for prediction
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols, index=X.index
    )

    rf_prob  = rf_model.predict_proba(X_scaled)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_scaled)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2

    latest_prob = float(ens_prob[-1])
    latest_rf   = float(rf_prob[-1])
    latest_xgb  = float(xgb_prob[-1])

    predictions = add_risk_to_predictions(df, ens_prob)
    latest_risk = predictions.iloc[-1]

    rf_shap, _  = calculate_shap_values(rf_model, X_scaled, 'rf')
    explanation = generate_explanation(rf_shap, feature_cols,
                                       latest_rf, row_idx=-1)
    top_drivers = get_top_drivers(rf_shap, feature_cols,
                                   row_idx=-1, top_n=5)

    return {
        'ticker':           ticker,
        'date':             str(df.index[-1].date()),
        'close':            round(float(df['Close'].iloc[-1]), 2),
        'prediction':       'UP' if latest_prob >= 0.5 else 'DOWN',
        'probability_up':   round(latest_prob * 100, 1),
        'probability_down': round((1 - latest_prob) * 100, 1),
        'confidence':       round(abs(latest_prob - 0.5) * 200, 1),
        'rf_probability':   round(latest_rf * 100, 1),
        'xgb_probability':  round(latest_xgb * 100, 1),
        'risk_label':       latest_risk['risk_label'],
        'risk_score':       round(float(latest_risk['risk_score']), 3),
        'explanation':      explanation,
        'top_drivers':      top_drivers.to_dict('records'),
        'price_history':    df[['Close', 'log_return']].tail(252),
        'model_type':       'universal',
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Step 1 — Build dataset
    logger.info("Building universal dataset...")
    logger.info("This will take 5-10 minutes (fetching 15 stocks)...")
    combined_df = build_universal_dataset()

    # Save combined dataset
    os.makedirs("data", exist_ok=True)
    combined_df.to_csv("data/universal_dataset.csv")
    logger.info(f"Dataset saved: {combined_df.shape}")

    # Step 2 — Train
    rf_model, xgb_model, scaler, feature_cols, test_df = \
        train_universal_model(combined_df)

    # Step 3 — Save
    save_universal_model(rf_model, xgb_model, scaler, feature_cols)

    # Step 4 — Test on a stock NOT in training emphasis
    print(f"\n{'='*60}")
    print("TESTING UNIVERSAL MODEL ON INDIVIDUAL STOCKS")
    print(f"{'='*60}")

    for test_ticker in ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS']:
        result = predict_with_universal_model(
            test_ticker, rf_model, xgb_model, scaler, feature_cols
        )
        print(f"\n{test_ticker}:")
        print(f"  Prediction:  {result['prediction']} "
              f"({result['probability_up']}%)")
        print(f"  Risk:        {result['risk_label']}")
        print(f"  Explanation: {result['explanation']}")