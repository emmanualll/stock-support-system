import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

logger = logging.getLogger(__name__)

TARGET_COLS = [
    'target_direction_1d', 'target_direction_3d', 'target_direction_5d',
    'target_direction_10d', 'target_return_1d', 'target_big_move'
]

RAW_PRICE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_200',  # raw MAs — non-stationary
    'ema_9', 'ema_21', 'ema_55',                   # raw EMAs — non-stationary
    'obv', 'pv_trend',                             # cumulative — non-stationary
    'volume_ma_10', 'volume_ma_20',                # raw volume MAs
    'obv_ma_20'                                    # raw OBV MA
]
def get_feature_columns(df: pd.DataFrame) -> list:
    #return only the cols that should be used as model featues this ecplicitly excludes targets and rwprices
    exclude = set(TARGET_COLS + RAW_PRICE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]
    logger.info(f"Using {len(feature_cols)} features")
    return feature_cols

def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    #this splits data into past and future and avoids randomness since we dont want our model to be trained in future data. 
    #o.2 indicates that 20% of the most recent data is being used to train
    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    logger.info(f"Train: {len(train)} rows | {train.index[0].date()} to {train.index[-1].date()}")
    logger.info(f"Test: {len(test)} rows | {test.index[0].date()} to {test.index[-1].date()}")
    return train, test

def prepare_data(df: pd.DataFrame, target_col: str = 'target_direction_1d'):
    #Preparing the X and Y features for training, we onlny call fit on the training data.

    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    scaler = RobustScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns = feature_cols,
        index = X.index
    )

    logger.info(f"Prepared {X_scaled.shape[0]} rows x {X_scaled.shape[1]} features")
    logger.info(f"Target: {target_col} | Class balance: {y.mean()*100:.1f}% positive")


    return X_scaled, y, scaler, feature_cols

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    #this is us training a random forest classifier, 500 treesm each giving us random subset of data and features

    logger.info("Training Random Forest:...")
    logger.info(f"Input: {X_train.shape[0]} rows and {X_train.shape[1]} features")

    model = RandomForestClassifier (
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    #Training accuracy is how well does it fit the data it leanrned from?
    #This should not be 100%, if it is then the model has memorised everything

    train_acc = model.score(X_train, y_train)
    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info("Random Forest training complete")

    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    #XGBoost is the sequential boosting omdel, in layman's terms it corrects the errors of the previous trees
    logger.info("Training XGBoost....")
    logger.info(f"Input: {X_train.shape[0]} rows x {X_train.shape[1]} features")

    model = xgb.XGBClassifier(
    n_estimators=300,       # reduce from 500
    max_depth=3,            # reduce from 4 — shallower trees
    learning_rate=0.01,     # reduce from 0.05 — smaller steps
    subsample=0.6,          # reduce from 0.8 — see less data per tree
    colsample_bytree=0.6,   # reduce from 0.8 — see fewer features per tree
    min_child_weight=30,    # increase from 20 — stricter leaf requirements
    gamma=5,                # increase from 1 — harder to make a split
    reg_alpha=1.0,          # increase from 0.1 — stronger L1 regularization
    reg_lambda=5.0,         # increase from 1.0 — stronger L2 regularization
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )

    train_acc = model.score(X_train, y_train)
    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info("XGBoost training complete")

    return model

def threshold_analysis(metrics: dict, y_test: pd.Series):
    """
    Instead of always using 0.5 as cutoff,
    find the threshold that maximises precision.
    
    In trading, making fewer high-confidence trades
    beats making many uncertain ones.
    """
    y_prob = metrics['y_prob']
    
    print(f"\n--- Threshold Analysis: {metrics['model_name']} ---")
    print(f"{'Threshold':>10} {'Coverage':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 55)

    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
        # Only predict UP when confidence exceeds threshold
        mask = y_prob >= threshold
        if mask.sum() < 10:
            break

        y_pred_filtered = (y_prob >= threshold).astype(int)
        coverage  = mask.mean() * 100
        precision = precision_score(y_test, y_pred_filtered, zero_division=0) * 100
        recall    = recall_score(y_test, y_pred_filtered, zero_division=0) * 100
        f1        = f1_score(y_test, y_pred_filtered, zero_division=0) * 100

        print(f"{threshold:>10.2f} {coverage:>9.1f}% {precision:>9.1f}% {recall:>9.1f}% {f1:>7.1f}%")

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    #Here we are evaluating the trained models, we are comparing both random forest and XGBoost side by side
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'model_name': model_name
    }

    # Log everything
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def get_feature_importance(model, feature_cols: list, top_n: int = 20) -> pd.DataFrame:
    """
    Extract and rank feature importances from a trained model.
    
    For Random Forest: importance = how much each feature reduces
    impurity (uncertainty) across all splits in all trees.
    
    For XGBoost: importance = how much each feature improves
    the loss function across all splits.
    
    Higher = more useful to the model.
    """
    importance_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    importance_df['importance_pct'] = (
        importance_df['importance'] / importance_df['importance'].sum() * 100
    )
    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()

    print(f"\n--- Top {top_n} Features: {type(model).__name__} ---")
    print(f"{'Rank':<5} {'Feature':<25} {'Importance':>10} {'Cumulative':>12}")
    print("-" * 55)
    for i, row in importance_df.head(top_n).iterrows():
        print(f"{i+1:<5} {row['feature']:<25} {row['importance_pct']:>9.2f}%  {row['cumulative_pct']:>9.2f}%")

    # How many features cover 80% of importance?
    n_80 = (importance_df['cumulative_pct'] <= 80).sum()
    print(f"\n→ Top {n_80} features explain 80% of model decisions")
    print(f"→ Bottom {len(feature_cols) - n_80} features contribute only 20%")

    return importance_df

def ensemble_predict(rf_model, xgb_model, X_test: pd.DataFrame, y_test: pd.Series, rf_weight: float = 0.5) -> dict:
    # Soft voting: Taking average of both the models probabilities

    # rf_weight = 0.5 means equal weight to both models
    # We can tune this later if one model proves to be more reliable

    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    ensemble_prob = (rf_weight * rf_prob) + ((1 - rf_weight) * xgb_prob)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    metrics = {
        'accuracy':   accuracy_score(y_test, ensemble_pred),
        'precision':  precision_score(y_test, ensemble_pred, zero_division=0),
        'recall':     recall_score(y_test, ensemble_pred, zero_division=0),
        'f1':         f1_score(y_test, ensemble_pred, zero_division=0),
        'y_pred':     ensemble_pred,
        'y_prob':     ensemble_prob,
        'model_name': 'Ensemble (RF+XGB)'
    }

    print(f"\n{'='*50}")
    print(f"MODEL: Ensemble (RF 50% + XGB 50%)")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))

    return metrics

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    from data_pipeline import get_data
    from features import build_features
    from targets import add_targets

    df = get_data("TCS.NS", force_refresh = True)
    df = build_features(df)
    df = add_targets(df)

    # Split first
    train_df, test_df = temporal_split(df)

    # Prepare — CRITICAL: fit scaler on train only, apply to both
    feature_cols = get_feature_columns(df)

    scaler = RobustScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[feature_cols]),
        columns=feature_cols,
        index=train_df.index
    )
    X_test = pd.DataFrame(
        scaler.transform(test_df[feature_cols]),
        columns=feature_cols,
        index=test_df.index
    )

    y_train = train_df['target_direction_1d']
    y_test  = test_df['target_direction_1d']

    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    rf_metrics  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Compare
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"{'Metric':<12} {'Random Forest':>15} {'XGBoost':>10}")
    print(f"{'-'*40}")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        rf_val  = rf_metrics[metric] * 100
        xgb_val = xgb_metrics[metric] * 100
        print(f"{metric:<12} {rf_val:>14.2f}%  {xgb_val:>9.2f}%")
    
    threshold_analysis(rf_metrics,  y_test)
    threshold_analysis(xgb_metrics, y_test)

    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    rf_importance  = get_feature_importance(rf_model,  feature_cols)
    xgb_importance = get_feature_importance(xgb_model, feature_cols)

    # Features both models agree on
    top_rf  = set(rf_importance.head(15)['feature'])
    top_xgb = set(xgb_importance.head(15)['feature'])
    overlap = top_rf & top_xgb
    print(f"\n→ Features in BOTH models' top 15: {overlap}")

    ensemble_metrics = ensemble_predict(rf_model, xgb_model, X_test, y_test)
    threshold_analysis(ensemble_metrics, y_test)

    # Final three-way comparison
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"{'Metric':<12} {'RF':>8} {'XGB':>8} {'Ensemble':>10}")
    print(f"{'-'*42}")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        rf_val  = rf_metrics[metric] * 100
        xgb_val = xgb_metrics[metric] * 100
        ens_val = ensemble_metrics[metric] * 100
        print(f"{metric:<12} {rf_val:>7.2f}%  {xgb_val:>7.2f}%  {ens_val:>9.2f}%")