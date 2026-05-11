import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import logging 
import joblib
from sklearn.preprocessing import RobustScaler

logging.basicConfig (
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_models_exist():
    """
    Check if trained models exist.
    If not, train them from scratch automatically.
    First run takes ~10 minutes. After that models are cached.
    """
    if not os.path.exists("models/universal_rf_selected.pkl"):
        logger.info("No trained models found. Training from scratch...")
        logger.info("This will take 10-15 minutes on first run.")
        logger.info("Models will be saved and reused after this.")

        # Run the full universal model training pipeline
        from universal_model import (build_universal_dataset,
                                      train_universal_model,
                                      save_universal_model)
        from feature_selection import (calculate_universal_shap_importance,
                                        select_features,
                                        evaluate_feature_subset)
        import joblib

        # Build dataset
        combined_df = build_universal_dataset()
        os.makedirs("data", exist_ok=True)
        combined_df.to_csv("data/universal_dataset.csv")

        # Train universal model
        rf_model, xgb_model, scaler, feature_cols, test_df = \
            train_universal_model(combined_df)

        # Feature selection
        from universal_model import temporal_split_universal
        train_df, test_df_fs = temporal_split_universal(combined_df)
        feature_cols_clean = [c for c in feature_cols if c != 'ticker']

        X_test_scaled = pd.DataFrame(
            scaler.transform(test_df_fs[feature_cols_clean]),
            columns=feature_cols_clean,
            index=test_df_fs.index
        )

        importance_df = calculate_universal_shap_importance(
            rf_model, X_test_scaled, feature_cols_clean
        )
        top_50 = importance_df.head(50)['feature'].tolist()

        # Retrain with top 50
        from sklearn.preprocessing import RobustScaler
        from model import train_random_forest, train_xgboost

        train_df2, _ = temporal_split_universal(combined_df)
        scaler2 = RobustScaler()
        X_train2 = pd.DataFrame(
            scaler2.fit_transform(train_df2[top_50]),
            columns=top_50, index=train_df2.index
        )
        y_train2 = train_df2['target_direction_1d']

        rf_final  = train_random_forest(X_train2, y_train2)
        xgb_final = train_xgboost(X_train2, y_train2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(rf_final,  "models/universal_rf_selected.pkl")
        joblib.dump(xgb_final, "models/universal_xgb_selected.pkl")
        joblib.dump(scaler2,   "models/universal_scaler_selected.pkl")
        joblib.dump(top_50,    "models/universal_features_selected.pkl")
        importance_df.to_csv("models/feature_importance.csv", index=False)

        logger.info("Models trained and saved. Ready to use.")
    else:
        logger.info("Models found. Skipping training.")

def analyze_stock(ticker: str,
                  force_refresh: bool = False,
                  use_universal: bool = True) -> dict:
    """
    Master function — full pipeline from ticker to prediction.
    
    use_universal=True  → use the pre-trained universal model (faster, generalises better)
    use_universal=False → train a stock-specific model from scratch
    """
    from data_pipeline import get_data
    from features import build_features
    from targets import add_targets
    from risk import add_risk_to_predictions
    from explainability import (calculate_shap_values,
                                generate_explanation,
                                get_top_drivers)

    logger.info(f"Starting analysis for {ticker}")

    if use_universal:
        # Use pre-trained universal model
        from universal_model import (load_universal_model,
                                     predict_with_universal_model)

        universal_path = "models/universal_rf.pkl"
        if not os.path.exists(universal_path):
            logger.warning("Universal model not found. Train it first:")
            logger.warning("  python src/universal_model.py")
            logger.info("Falling back to stock-specific model...")
            use_universal = False
        else:
            rf_model, xgb_model, scaler, feature_cols = load_universal_model()
            result = predict_with_universal_model(
                ticker, rf_model, xgb_model, scaler, feature_cols,
                force_refresh=force_refresh
            )
            result['model_type'] = 'universal'
            return result

    # Stock-specific model (fallback)
    from model import (get_feature_columns, temporal_split,
                       train_random_forest, train_xgboost)

    df_raw      = get_data(ticker, force_refresh=force_refresh)
    df_features = build_features(df_raw, ticker=ticker)
    df          = add_targets(df_features)

    train_df, test_df = temporal_split(df)
    feature_cols      = get_feature_columns(df)

    scaler  = RobustScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[feature_cols]),
        columns=feature_cols, index=train_df.index
    )
    X_test = pd.DataFrame(
        scaler.transform(test_df[feature_cols]),
        columns=feature_cols, index=test_df.index
    )
    y_train = train_df['target_direction_1d']

    model_path = f"models/{ticker.replace('.', '_')}"
    rf_model   = train_random_forest(X_train, y_train)
    xgb_model  = train_xgboost(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model,     f"{model_path}_rf.pkl")
    joblib.dump(xgb_model,    f"{model_path}_xgb.pkl")
    joblib.dump(scaler,       f"{model_path}_scaler.pkl")
    joblib.dump(feature_cols, f"{model_path}_features.pkl")

    rf_prob  = rf_model.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2

    latest_prob = float(ens_prob[-1])
    latest_rf   = float(rf_prob[-1])
    latest_xgb  = float(xgb_prob[-1])

    predictions = add_risk_to_predictions(df, ens_prob)
    latest_risk = predictions.iloc[-1]

    rf_shap, _  = calculate_shap_values(rf_model, X_test, 'rf')
    explanation = generate_explanation(rf_shap, feature_cols,
                                       latest_rf, row_idx=-1)
    top_drivers = get_top_drivers(rf_shap, feature_cols,
                                   row_idx=-1, top_n=5)

    return {
        'ticker':           ticker,
        'date':             str(test_df.index[-1].date()),
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
        'model_type':       'stock-specific',
    }


def print_result(result: dict):
    print(f"\n{'='*60}")
    print(f"STOCK ANALYSIS: {result['ticker']}")
    print(f"{'='*60}")
    print(f"Date:         {result['date']}")
    print(f"Close Price:  ₹{result['close']:,.2f}")
    print(f"Model Type:   {result.get('model_type', 'unknown')}")
    print(f"\n--- Prediction ---")
    print(f"Direction:    {result['prediction']}")
    print(f"Probability:  UP {result['probability_up']}% | "
          f"DOWN {result['probability_down']}%")
    print(f"Confidence:   {result['confidence']}%")
    print(f"\n--- Model Breakdown ---")
    print(f"Random Forest:  {result['rf_probability']}%")
    print(f"XGBoost:        {result['xgb_probability']}%")
    print(f"Ensemble:       {result['probability_up']}%")
    print(f"\n--- Risk ---")
    print(f"Risk Level:   {result['risk_label']}")
    print(f"Risk Score:   {result['risk_score']}")
    print(f"\n--- Explanation ---")
    print(result['explanation'])
    print(f"\n--- Top 5 Drivers ---")
    for driver in result['top_drivers']:
        arrow = '↑' if driver['shap_value'] > 0 else '↓'
        print(f"  {arrow} {driver['feature']:<30} "
              f"SHAP: {driver['shap_value']:>+.4f}")

    # Only print training info if available
    if 'train_period' in result:
        print(f"\n--- Training Info ---")
        print(f"Train period: {result['train_period']}")
        print(f"Test period:  {result['test_period']}")
        print(f"Features:     {result['n_features']}")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Stock Analysis System')
    parser.add_argument('ticker', nargs='?', default='TCS.NS',
                        help='Stock ticker e.g. TCS.NS, RELIANCE.NS')
    parser.add_argument('--refresh', action='store_true',
                        help='Force refresh data')
    parser.add_argument('--specific', action='store_true',
                        help='Use stock-specific model instead of universal')
    args = parser.parse_args()

    result = analyze_stock(
        ticker=args.ticker,
        force_refresh=args.refresh,
        use_universal=not args.specific
    )
    print_result(result)