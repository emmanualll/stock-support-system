import pandas as pd
import numpy as np
import logging
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import joblib 
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, make_scorer, accuracy_score,
                             precision_score, recall_score, classification_report)

import xgboost as xgb

logger = logging.getLogger(__name__)

def tune_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       n_iter: int = 40,
                       n_splits: int = 5) -> tuple:
    logger.info("Tuning Random Forest...")
    logger.info(f"Trying {n_iter} combinations x {n_splits} folds "
                f" = {n_iter*n_splits} fits")
    
    param_grid = {
        'n_estimators': [200, 300, 500, 750],
        'max_depth': [4, 5, 6, 7, 8],
        'min_samples_leaf': [10, 15, 20, 30, 50],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'class_weight': ['balanced', None],
    }

    tscv = TimeSeriesSplit(n_splits = n_splits)
    scorer = make_scorer(f1_score, zero_division = 0)

    base_model = RandomForestClassifier(random_state = 42, n_jobs = -1)


    search = RandomizedSearchCV(
        estimator = base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    logger.info(f"Best RF params: {search.best_params_}")
    logger.info(f"Best RF CV F1:  {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_

def tune_xgboost(X_train: pd.DataFrame,
                 y_train: pd.Series,
                 n_iter: int = 40,
                 n_splits: int = 5) -> tuple:
    """
    Tune XGBoost using RandomizedSearchCV with TimeSeriesSplit.
    """
    logger.info("Tuning XGBoost...")
    logger.info(f"Trying {n_iter} combinations × {n_splits} folds "
                f"= {n_iter*n_splits} fits")

    param_grid = {
        'n_estimators':     [200, 300, 500],
        'max_depth':        [2, 3, 4, 5],
        'learning_rate':    [0.005, 0.01, 0.02, 0.05],
        'subsample':        [0.5, 0.6, 0.7, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'min_child_weight': [10, 20, 30, 50],
        'gamma':            [1, 3, 5, 10],
        'reg_alpha':        [0.1, 0.5, 1.0, 2.0],
        'reg_lambda':       [1.0, 3.0, 5.0, 10.0],
    }

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(f1_score, zero_division=0)

    base_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    logger.info(f"Best XGB params: {search.best_params_}")
    logger.info(f"Best XGB CV F1:  {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_tuned_models(rf_model, xgb_model,
                           X_test, y_test) -> tuple:
    """Compare tuned model performance."""
    rf_pred  = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    rf_prob  = rf_model.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2
    ens_pred = (ens_prob >= 0.5).astype(int)

    results = {}
    for name, pred in [('RF', rf_pred),
                        ('XGB', xgb_pred),
                        ('Ensemble', ens_pred)]:
        results[name] = {
            'accuracy':  accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall':    recall_score(y_test, pred, zero_division=0),
            'f1':        f1_score(y_test, pred, zero_division=0),
        }

    print(f"\n{'='*60}")
    print("TUNED MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<12} {m['accuracy']*100:>9.2f}% "
              f"{m['precision']*100:>9.2f}% "
              f"{m['recall']*100:>7.2f}% "
              f"{m['f1']*100:>7.2f}%")

    return results, ens_prob


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    from universal_model import temporal_split_universal

    logger.info("Loading universal dataset...")
    combined_df = pd.read_csv(
        "data/universal_dataset.csv",
        index_col=0, parse_dates=True
    )

    # Use top 50 selected features
    feature_cols = joblib.load("models/universal_features_selected.pkl")
    feature_cols = [c for c in feature_cols if c != 'ticker']

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

    logger.info(f"Training data: {X_train_scaled.shape}")
    logger.info(f"Test data:     {X_test_scaled.shape}")

    # Tune both models
    rf_tuned, rf_params, rf_score     = tune_random_forest(
        X_train_scaled, y_train, n_iter=40, n_splits=5
    )
    xgb_tuned, xgb_params, xgb_score = tune_xgboost(
        X_train_scaled, y_train, n_iter=40, n_splits=5
    )

    print(f"\n--- Best Parameters ---")
    print(f"Random Forest: {rf_params}")
    print(f"XGBoost:       {xgb_params}")
    print(f"\nCV F1 Scores:")
    print(f"Random Forest: {rf_score:.4f}")
    print(f"XGBoost:       {xgb_score:.4f}")

    # Evaluate on test set
    results, ens_prob = evaluate_tuned_models(
        rf_tuned, xgb_tuned, X_test_scaled, y_test
    )

    # Per stock breakdown
    print(f"\n--- Per-Stock Accuracy (Tuned Universal Model) ---")
    for ticker in test_df['ticker'].unique():
        mask     = test_df['ticker'] == ticker
        y_t      = y_test[mask]
        rf_prob  = rf_tuned.predict_proba(X_test_scaled[mask.values])[:, 1]
        xgb_prob = xgb_tuned.predict_proba(X_test_scaled[mask.values])[:, 1]
        ens      = (rf_prob + xgb_prob) / 2
        pred     = (ens >= 0.5).astype(int)
        acc      = accuracy_score(y_t, pred) * 100
        baseline = y_t.mean() * 100
        print(f"  {ticker:<20} {acc:.1f}%  (baseline: {baseline:.1f}%)")

    # Save tuned universal models
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_tuned,     "models/universal_rf_tuned.pkl")
    joblib.dump(xgb_tuned,    "models/universal_xgb_tuned.pkl")
    joblib.dump(scaler,       "models/universal_scaler_tuned.pkl")
    joblib.dump(feature_cols, "models/universal_features_tuned.pkl")
    logger.info("Tuned universal models saved to models/")

    print(f"\nTuned models saved.")
    print(f"Update main.py to load universal_rf_tuned.pkl for best performance.")