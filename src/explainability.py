import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)


def calculate_shap_values(model, X: pd.DataFrame, model_type: str = 'rf'):
    """
    Calculate SHAP values for a trained model.
    
    For Random Forest we use TreeExplainer — optimised specifically
    for tree-based models, runs in O(TLD) time where T=trees,
    L=leaves, D=depth. Much faster than the generic explainer.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Run: pip install shap")

    logger.info(f"Calculating SHAP values for {model_type}...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values is a list of 2 arrays
    # Index 1 = SHAP values for class 1 (UP prediction)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1] 
    else:
        shap_vals = shap_values

    logger.info(f"SHAP values shape: {shap_vals.shape}")
    return shap_vals, explainer


def get_top_drivers(shap_vals: np.ndarray,
                    feature_cols: list,
                    row_idx: int = -1,
                    top_n: int = 5) -> pd.DataFrame:
    """
    Get the top N features driving a specific prediction.
    
    row_idx=-1 means the most recent prediction (latest day).
    Positive SHAP = pushed toward UP
    Negative SHAP = pushed toward DOWN
    """
    row_shap = shap_vals[row_idx]

    drivers = pd.DataFrame({
        'feature':    feature_cols,
        'shap_value': row_shap
    })

    drivers['abs_shap']   = drivers['shap_value'].abs()
    drivers['direction']  = drivers['shap_value'].apply(
        lambda x: 'UP ↑' if x > 0 else 'DOWN ↓'
    )

    drivers = drivers.sort_values('abs_shap', ascending=False)

    return drivers.head(top_n)


def generate_explanation(shap_vals: np.ndarray,
                         feature_cols: list,
                         prob_up: float,
                         row_idx: int = -1) -> str:
    """
    Generate a human-readable explanation of the prediction.
    This is what gets displayed in the Streamlit UI.
    """
    drivers = get_top_drivers(shap_vals, feature_cols, row_idx, top_n=3)

    direction  = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = abs(prob_up - 0.5) * 200

    # Build explanation sentence
    up_drivers   = drivers[drivers['shap_value'] > 0]['feature'].tolist()
    down_drivers = drivers[drivers['shap_value'] < 0]['feature'].tolist()

    # Make feature names human readable
    def humanise(feature: str) -> str:
        mapping = {
            'macd_histogram':      'MACD momentum',
            'rsi_14':              'RSI (14-day)',
            'return_lag_1':        'yesterday\'s return',
            'vol_regime':          'volatility regime',
            'ma_20_50_cross':      'MA trend crossover',
            'adx':                 'trend strength (ADX)',
            'close_to_ma_200':     '200-day MA distance',
            'nifty_50_return_1d':  'Nifty 50 momentum',
            'nifty_it_return_1d':  'Nifty IT momentum',
            'usd_inr_return_1d':   'USD/INR movement',
            'bb_pct_b_20':         'Bollinger Band position',
            'hist_vol_20':         '20-day volatility',
            'atr_14_pct':          'Average True Range',
            'obv_signal':          'volume trend (OBV)',
        }
        return mapping.get(feature, feature.replace('_', ' '))

    up_readable   = [humanise(f) for f in up_drivers]
    down_readable = [humanise(f) for f in down_drivers]

    explanation = f"Prediction: {direction} ({prob_up*100:.1f}% probability, {confidence:.0f}% confidence). "

    if up_readable:
        explanation += f"Bullish signals: {', '.join(up_readable)}. "
    if down_readable:
        explanation += f"Bearish signals: {', '.join(down_readable)}. "

    return explanation.strip()


def global_shap_summary(shap_vals: np.ndarray,
                         feature_cols: list,
                         top_n: int = 20) -> pd.DataFrame:
    """
    Global feature importance from SHAP — average absolute SHAP value
    across all predictions. More reliable than built-in feature importance
    because it accounts for feature interactions.
    """
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    summary = pd.DataFrame({
        'feature':         feature_cols,
        'mean_abs_shap':   mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    summary['importance_pct'] = (
        summary['mean_abs_shap'] / summary['mean_abs_shap'].sum() * 100
    )

    print(f"\n--- Global SHAP Feature Importance (Top {top_n}) ---")
    print(f"{'Rank':<5} {'Feature':<30} {'Mean |SHAP|':>12} {'Importance':>10}")
    print("-" * 60)
    for i, row in summary.head(top_n).iterrows():
        print(f"{i+1:<5} {row['feature']:<30} "
              f"{row['mean_abs_shap']:>12.4f} "
              f"{row['importance_pct']:>9.2f}%")

    return summary


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    from data_pipeline import get_data
    from features import build_features
    from targets import add_targets
    from model import (get_feature_columns, temporal_split,
                       train_random_forest, train_xgboost)
    from sklearn.preprocessing import RobustScaler

    df = get_data("TCS.NS")
    df = build_features(df)
    df = add_targets(df)

    train_df, test_df = temporal_split(df)
    feature_cols = get_feature_columns(df)

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
    y_test  = test_df['target_direction_1d']

    # Train Random Forest (SHAP works best with RF)
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Calculate SHAP values
    rf_shap,  rf_explainer  = calculate_shap_values(rf_model,  X_test, 'rf')
    xgb_shap, xgb_explainer = calculate_shap_values(xgb_model, X_test, 'xgb')

    # Global importance
    print("\n=== RANDOM FOREST ===")
    rf_summary = global_shap_summary(rf_shap, feature_cols)

    print("\n=== XGBOOST ===")
    xgb_summary = global_shap_summary(xgb_shap, feature_cols)

    # Latest prediction explanation
    rf_prob  = rf_model.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2

    print(f"\n=== LATEST PREDICTION EXPLANATION ===")
    print(f"Date: {test_df.index[-1].date()}")
    print(f"Close: ₹{test_df['Close'].iloc[-1]:.2f}")
    print(f"\nRandom Forest:")
    print(generate_explanation(rf_shap, feature_cols, rf_prob[-1], row_idx=-1))

    print(f"\nTop 5 drivers (RF):")
    print(get_top_drivers(rf_shap, feature_cols, row_idx=-1).to_string())

    print(f"\nXGBoost:")
    print(generate_explanation(xgb_shap, feature_cols, xgb_prob[-1], row_idx=-1))

    print(f"\nEnsemble probability: {ens_prob[-1]*100:.1f}%")
    print(f"Direction: {'UP' if ens_prob[-1] >= 0.5 else 'DOWN'}")