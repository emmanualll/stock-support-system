import pandas as pd
import numpy as np
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)

def calculate_volatility_risk(df: pd.DataFrame) -> pd.Series:
    """
    Volatality risk is checking if the current volatality elevated vs the recent history

    we compare a 10 day volatality to a 30 day volatality
    if short term vol is much higher than long term then risk is elevated
    """

    short_vol = df['log_return'].rolling(10).std()
    long_vol = df['log_return'].rolling(30).std()

    vol_ratio = short_vol / long_vol.replace(0, np.nan)

#nomralize to 0-1 range if 1 then normal if 2 then high rish if 0.5 then low eish
    vol_risk = (vol_ratio - 0.5) / 1.5
    vol_risk = vol_risk.clip(0, 1)
    return vol_risk.fillna(0.5)


def calculate_trend_risk(df: pd.DataFrame) -> pd.Series:
    """
    Trend risk is if therei is a clear trend or if the market is just cleary choppy

    ADX measures the trend strength, adx of less than 20 means noo trend, risk for directinal bets
    if more than 25 then it means more predictable and lower risk

    output of 0 means low risk and 1 means high risk 
    """

    adx = df['adx']

    trend_risk = 1 - (adx / 50).clip(0, 1)

    return trend_risk.fillna(0.5)

def calculate_confidence_risk(y_prob: np.ndarray) -> np.ndarray:
    '''
    Confidence risk is how uncertain the model is
    probability neare 0.5 means the model has no idea i.e., high risk
    probability near 0 or 1 means the model is confident

    output is 0 means low risk and 1 means high risk
    '''

    distance_from_uncertain = np.abs(y_prob - 0.5)

    confidence_risk = 1 - (distance_from_uncertain / 0.5)

    return confidence_risk


def composite_risk_score(df: pd.DataFrame,
                         y_prob: np.ndarray,
                         vol_weight: float = 0.4,
                         trend_weight: float = 0.3,
                         conf_weight: float = 0.3) -> pd.Series:
    """
    Combine all three risk components into one score

    Weights reflect relative importance:
    -Volatality (40%) - most directly impacts prediction reliability
    Trend strength 30 and model confidence 30
    """

    assert abs(vol_weight + trend_weight + conf_weight - 1.0) < 1e-6, \
        "Weights must sum to 1.0"
    
    test_df = df.iloc[-len(y_prob):]

    vol_risk  = calculate_volatility_risk(test_df).values
    trend_risk = calculate_trend_risk(test_df).values
    conf_risk  = calculate_confidence_risk(y_prob)

    composite = (vol_weight  * vol_risk +
                 trend_weight * trend_risk +
                 conf_weight  * conf_risk) 
    
    return pd.Series(composite, index=test_df.index, name = 'risk_score')

def risk_label(score: float) -> str:
    """Convert numeric risk score to human-readable label."""
    if score < 0.35:
        return "LOW"
    elif score < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"

def add_risk_to_predictions(df: pd.DataFrame,
                             y_prob: np.ndarray) -> pd.DataFrame:
    """
    Master function — takes the test dataframe and model probabilities,
    returns a clean prediction DataFrame with risk scores.
    This is what the Streamlit UI will call.
    """
    test_df = df.iloc[-len(y_prob):].copy()

    risk_scores = composite_risk_score(df, y_prob)

    results = pd.DataFrame({
        'date':          test_df.index,
        'close':         test_df['Close'].values,
        'prob_up':       y_prob,
        'direction':     ['UP' if p >= 0.5 else 'DOWN' for p in y_prob],
        'confidence':    [round(abs(p - 0.5) * 200, 1) for p in y_prob],
        'risk_score':    risk_scores.values,
        'risk_label':    [risk_label(s) for s in risk_scores.values],
        'actual_return': test_df['log_return'].values,
    })

    results = results.set_index('date')

    # Log distribution
    risk_dist = results['risk_label'].value_counts()
    logger.info(f"Risk distribution: {risk_dist.to_dict()}")

    return results


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

    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    rf_prob  = rf_model.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2

    # Generate risk-annotated predictions
    predictions = add_risk_to_predictions(df, ens_prob)

    print("\n--- Sample Predictions with Risk ---")
    print(predictions[['close', 'prob_up', 'direction',
                        'confidence', 'risk_score', 'risk_label']].head(15).to_string())

    print(f"\n--- Risk Distribution ---")
    print(predictions['risk_label'].value_counts())

    print(f"\n--- Accuracy by Risk Level ---")
    predictions['correct'] = (
        ((predictions['prob_up'] >= 0.5) & (predictions['actual_return'] > 0)) |
        ((predictions['prob_up'] < 0.5)  & (predictions['actual_return'] < 0))
    )
    for label in ['LOW', 'MEDIUM', 'HIGH']:
        subset = predictions[predictions['risk_label'] == label]
        if len(subset) > 0:
            acc = subset['correct'].mean() * 100
            print(f"{label:>8}: {acc:.1f}% accuracy ({len(subset)} days)")