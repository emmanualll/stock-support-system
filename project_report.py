import sys
import os

# --------------------------------------------------
# Add src folder to Python path
# --------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

print("\n" + "=" * 80)
print("STOCK DECISION SUPPORT SYSTEM - FULL PROJECT REPORT")
print("=" * 80)

# ==================================================
# UNIVERSAL MODEL
# ==================================================

try:
    print("\n\n[1] UNIVERSAL MODEL EVALUATION")

    from universal_model import (
        build_universal_dataset,
        train_universal_model
    )

    combined_df = build_universal_dataset()

    rf_model, xgb_model, scaler, feature_cols, test_df = \
        train_universal_model(combined_df)

    print(f"\nUniversal Dataset Rows: {len(combined_df):,}")
    print(f"Test Rows: {len(test_df):,}")
    print(f"Features Used: {len(feature_cols)}")

except Exception as e:
    print(f"\nUniversal model failed:")
    print(e)

# ==================================================
# FEATURE INFORMATION
# ==================================================

try:
    print("\n\n[2] FEATURE INFORMATION")
    print(f"Total Features: {len(feature_cols)}")

except Exception as e:
    print(e)

# ==================================================
# WALK FORWARD VALIDATION
# ==================================================

try:
    print("\n\n[3] WALK FORWARD VALIDATION")

    from data_pipeline import get_data
    from features import build_features
    from targets import add_targets
    from walk_forward import (
        walk_forward_validation,
        print_walk_forward_summary
    )

    ticker = "TCS.NS"

    df = get_data(ticker)
    df = build_features(df, ticker=ticker)
    df = add_targets(df)

    wf_results = walk_forward_validation(df)

    print_walk_forward_summary(wf_results)

    print("\nAverage Walk Forward Metrics")
    print("-" * 40)

    print(
        wf_results[
            ["accuracy", "precision", "recall", "f1"]
        ].mean()
    )

except Exception as e:
    print(f"\nWalk-forward validation failed:")
    print(e)

# ==================================================
# BACKTEST
# ==================================================

try:
    print("\n\n[4] BACKTEST")

    from model import (
        get_feature_columns,
        temporal_split,
        train_random_forest,
        train_xgboost
    )

    from sklearn.preprocessing import RobustScaler

    from backtest import (
        run_backtest,
        calculate_metrics,
        print_backtest_summary
    )

    feature_cols_bt = get_feature_columns(df)

    train_df, test_df = temporal_split(df)

    scaler = RobustScaler()

    X_train = scaler.fit_transform(
        train_df[feature_cols_bt]
    )

    X_test = scaler.transform(
        test_df[feature_cols_bt]
    )

    y_train = train_df["target_direction_1d"]

    rf_model_bt = train_random_forest(X_train, y_train)
    xgb_model_bt = train_xgboost(X_train, y_train)

    rf_prob = rf_model_bt.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model_bt.predict_proba(X_test)[:, 1]

    ensemble_prob = (rf_prob + xgb_prob) / 2

    bt = run_backtest(
        df=df,
        y_prob=ensemble_prob
    )

    metrics = calculate_metrics(bt)

    print_backtest_summary(metrics)

except Exception as e:
    print(f"\nBacktest failed:")
    print(e)

# ==================================================
# SHAP EXPLANATION
# ==================================================

try:
    print("\n\n[5] SHAP EXPLANATION")

    from explainability import (
        calculate_shap_values,
        get_top_drivers,
        generate_explanation
    )

    import pandas as pd

    X_test_df = pd.DataFrame(
        X_test,
        columns=feature_cols_bt,
        index=test_df.index
    )

    shap_values, _ = calculate_shap_values(
        rf_model_bt,
        X_test_df,
        "rf"
    )

    explanation = generate_explanation(
        shap_values,
        feature_cols_bt,
        rf_prob[-1],
        row_idx=-1
    )

    print("\nGenerated Explanation:")
    print(explanation)

    print("\nTop Drivers:")

    drivers = get_top_drivers(
        shap_values,
        feature_cols_bt,
        row_idx=-1,
        top_n=10
    )

    print(drivers)

except Exception as e:
    print(f"\nSHAP analysis failed:")
    print(e)

# ==================================================
# UNIVERSAL MODEL PREDICTION
# ==================================================

try:
    print("\n\n[6] UNIVERSAL MODEL PREDICTION")

    from universal_model import (
        load_universal_model,
        predict_with_universal_model
    )

    rf_u, xgb_u, scaler_u, feature_cols_u = \
        load_universal_model()

    prediction = predict_with_universal_model(
        ticker="TCS.NS",
        rf_model=rf_u,
        xgb_model=xgb_u,
        scaler=scaler_u,
        feature_cols=feature_cols_u
    )

    print("\nPrediction Output:")
    print("-" * 40)

    for key, value in prediction.items():
        if key not in ["price_history"]:
            print(f"{key}: {value}")

except Exception as e:
    print(f"\nUniversal prediction failed:")
    print(e)

# ==================================================
# SAVED MODEL CHECK
# ==================================================

try:
    print("\n\n[7] SAVED MODELS")

    model_dir = os.path.join(ROOT_DIR, "models")

    if os.path.exists(model_dir):

        files = sorted(os.listdir(model_dir))

        for file in files:
            print(file)

    else:
        print("No models folder found")

except Exception as e:
    print(e)

# ==================================================
# FINISH
# ==================================================

print("\n")
print("=" * 80)
print("REPORT COMPLETE")
print("=" * 80)