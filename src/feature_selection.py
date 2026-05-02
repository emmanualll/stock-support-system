import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

def calculate_universal_shap_importance(rf_model,
                                        X_test: pd.DataFrame,
                                        feature_cols: list) -> pd.DataFrame:
    """
    Calculating the shap based feature importance instead of using built in thing from sklearn. More reliable
    """

    import shap

    logger.info("Calculating SHAP vaalues for feature seleection..")
    logger.info(f"Test set: {X_test.shape[0]} rows x {X_test.shape[1]} features")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_values = shap_values
    
    mean_abs_shap = np.abs(shap_vals).mean(axis = 0)

    importance_df = pd.DataFrame({
        'feature':       feature_cols,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending = False).reset_index(drop=True)

    importance_df['importance_pct'] = (
        importance_df['mean_abs_shap'] /
        importance_df['mean_abs_shap'].sum() * 100
    )

    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
    importance_df['rank'] = range(1, len(importance_df) + 1)

    return importance_df

def select_features(importance_df: pd.DataFrame,
                    method: str= 'threshold',
                    threshold_pct: float = 0.3,
                    top_n: int = 50) -> list:
    #mthod = treshold means drop features below treshold pct importance
    #top n keeps only the n features 
    #method cumulative keeps features that explain 90 percent of the importance
    if method == 'threshold':
        selected = importance_df[importance_df['importance_pct'] >= threshold_pct]['feature'].tolist()
    elif method == 'top_n':
        selected = importance_df.head(top_n)['feature'].tolist()
    elif method == 'cumulative':
        selected = importance_df[
            importance_df['cumulative_pct'] <= 90
        ]['feature'].tolist()

    else:
        raise ValueError(f"Unknown method: {method}")

    dropped = len(importance_df) - len(selected)
    logger.info(f"Feature selection ({method}): "
                f"keeping {len(selected)}, dropping {dropped}")

    return selected


def evaluate_feature_subset(combined_df: pd.DataFrame,
                             feature_cols: list,
                             label: str = "") -> dict:
    """
    Train and evaluate a model using only the specified features.
    Returns accuracy and F1 for comparison.
    """
    from universal_model import temporal_split_universal
    from model import train_random_forest, train_xgboost

    train_df, test_df = temporal_split_universal(combined_df)

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    y_train = train_df['target_direction_1d']
    y_test  = test_df['target_direction_1d']

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols, index=X_test.index
    )

    rf_model  = train_random_forest(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)

    rf_prob  = rf_model.predict_proba(X_test_scaled)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2
    ens_pred = (ens_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, ens_pred)
    f1  = f1_score(y_test, ens_pred, zero_division=0)

    logger.info(f"{label}: {len(feature_cols)} features | "
                f"Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")

    return {
        'label':        label,
        'n_features':   len(feature_cols),
        'accuracy':     acc,
        'f1':           f1,
        'rf_model':     rf_model,
        'xgb_model':    xgb_model,
        'scaler':       scaler,
        'feature_cols': feature_cols,
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load existing universal model and dataset
    logger.info("Loading universal model and dataset...")

    rf_model     = joblib.load("models/universal_rf.pkl")
    xgb_model    = joblib.load("models/universal_xgb.pkl")
    scaler       = joblib.load("models/universal_scaler.pkl")
    feature_cols = joblib.load("models/universal_features.pkl")

    # Load combined dataset
    logger.info("Loading combined dataset...")
    combined_df = pd.read_csv("data/universal_dataset.csv",
                               index_col=0, parse_dates=True)

    # Remove ticker column from features
    feature_cols = [c for c in feature_cols if c != 'ticker']

    # Step 1 — get test set for SHAP calculation
    from universal_model import temporal_split_universal
    train_df, test_df = temporal_split_universal(combined_df)

    X_test_scaled = pd.DataFrame(
        scaler.transform(test_df[feature_cols]),
        columns=feature_cols, index=test_df.index
    )

    # Step 2 — calculate SHAP importance
    importance_df = calculate_universal_shap_importance(
        rf_model, X_test_scaled, feature_cols
    )

    # Print full ranking
    print(f"\n{'='*65}")
    print("FEATURE IMPORTANCE RANKING (SHAP-based, Universal Model)")
    print(f"{'='*65}")
    print(f"{'Rank':<5} {'Feature':<35} {'Importance':>10} {'Cumulative':>12}")
    print("-" * 65)
    for _, row in importance_df.iterrows():
        marker = " ←" if row['cumulative_pct'] <= 80 else ""
        print(f"{int(row['rank']):<5} {row['feature']:<35} "
              f"{row['importance_pct']:>9.2f}%  "
              f"{row['cumulative_pct']:>9.2f}%{marker}")

    # Step 3 — compare different feature subsets
    print(f"\n{'='*60}")
    print("COMPARING FEATURE SUBSETS")
    print(f"{'='*60}")

    results = []

    # Baseline — all features
    logger.info("Evaluating baseline (all features)...")
    r_all = evaluate_feature_subset(
        combined_df, feature_cols, "All features"
    )
    results.append(r_all)

    # Top 50 features
    top_50 = select_features(importance_df, method='top_n', top_n=50)
    logger.info("Evaluating top 50 features...")
    r_50 = evaluate_feature_subset(combined_df, top_50, "Top 50")
    results.append(r_50)

    # Top 40 features
    top_40 = select_features(importance_df, method='top_n', top_n=40)
    logger.info("Evaluating top 40 features...")
    r_40 = evaluate_feature_subset(combined_df, top_40, "Top 40")
    results.append(r_40)

    # Top 30 features
    top_30 = select_features(importance_df, method='top_n', top_n=30)
    logger.info("Evaluating top 30 features...")
    r_30 = evaluate_feature_subset(combined_df, top_30, "Top 30")
    results.append(r_30)

    # 90% cumulative importance
    top_90pct = select_features(importance_df, method='cumulative')
    logger.info("Evaluating 90% cumulative importance features...")
    r_90 = evaluate_feature_subset(
        combined_df, top_90pct, "90% cumulative"
    )
    results.append(r_90)

    # Step 4 — summary table
    print(f"\n{'='*55}")
    print("FEATURE SELECTION RESULTS")
    print(f"{'='*55}")
    print(f"{'Subset':<20} {'Features':>8} {'Accuracy':>10} {'F1':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['label']:<20} {r['n_features']:>8} "
              f"{r['accuracy']*100:>9.2f}%  {r['f1']*100:>7.2f}%")

    # Step 5 — save best model
    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest subset: {best['label']} "
          f"({best['n_features']} features, "
          f"F1: {best['f1']*100:.2f}%)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best['rf_model'],     "models/universal_rf_selected.pkl")
    joblib.dump(best['xgb_model'],    "models/universal_xgb_selected.pkl")
    joblib.dump(best['scaler'],       "models/universal_scaler_selected.pkl")
    joblib.dump(best['feature_cols'], "models/universal_features_selected.pkl")

    # Also save importance ranking
    importance_df.to_csv("models/feature_importance.csv", index=False)

    logger.info(f"Best model saved with {best['n_features']} features")
    logger.info("Feature importance saved to models/feature_importance.csv")
    