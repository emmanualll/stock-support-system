import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import get_feature_columns, train_random_forest, train_xgboost

logger = logging.getLogger(__name__)


def walk_forward_validation(df: pd.DataFrame,
                            start_train_years: int = 3,
                            test_window_months: int = 12) -> pd.DataFrame:
    """
    Walk-forward validation — retrain every year on all available
    history, test on next unseen year. Repeat across all years.
    Most realistic evaluation method for financial ML.
    """
    feature_cols = get_feature_columns(df)
    results = []

    first_date       = df.index[0]
    first_test_start = first_date + pd.DateOffset(years=start_train_years)

    test_starts = pd.date_range(
        start=first_test_start,
        end=df.index[-1] - pd.DateOffset(months=test_window_months),
        freq=f'{test_window_months}ME'
    )

    logger.info(f"Walk-forward: {len(test_starts)} rounds, "
                f"{test_window_months}-month test windows")

    for i, test_start in enumerate(test_starts):
        test_end = test_start + pd.DateOffset(months=test_window_months)

        train_df = df[df.index < test_start]
        test_df  = df[(df.index >= test_start) & (df.index < test_end)]

        if len(train_df) < 200 or len(test_df) < 20:
            continue  # ← skip rounds with insufficient data

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
        ens_pred = (ens_prob >= 0.5).astype(int)

        round_result = {
            'round':       i + 1,
            'train_start': train_df.index[0].date(),
            'train_end':   train_df.index[-1].date(),
            'test_start':  test_df.index[0].date(),
            'test_end':    test_df.index[-1].date(),
            'train_rows':  len(train_df),
            'test_rows':   len(test_df),
            'accuracy':    accuracy_score(y_test, ens_pred),
            'precision':   precision_score(y_test, ens_pred, zero_division=0),
            'recall':      recall_score(y_test, ens_pred, zero_division=0),
            'f1':          f1_score(y_test, ens_pred, zero_division=0),
            'up_pct':      y_test.mean(),
        }

        logger.info(
            f"Round {i+1}: {round_result['test_start']} → {round_result['test_end']} | "
            f"Acc: {round_result['accuracy']*100:.1f}% | "
            f"F1: {round_result['f1']*100:.1f}%"
        )

        results.append(round_result)

    return pd.DataFrame(results)


def print_walk_forward_summary(results_df: pd.DataFrame):
    print(f"\n{'='*75}")
    print("WALK-FORWARD VALIDATION RESULTS")
    print(f"{'='*75}")
    print(f"{'Rnd':<5} {'Test Period':<25} {'Rows':<6} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 75)

    for _, row in results_df.iterrows():
        print(f"{int(row['round']):<5} "
              f"{str(row['test_start'])} → {str(row['test_end']):<12} "
              f"{int(row['test_rows']):<6} "
              f"{row['accuracy']*100:>6.1f}% "
              f"{row['precision']*100:>6.1f}% "
              f"{row['recall']*100:>6.1f}% "
              f"{row['f1']*100:>6.1f}%")

    print("-" * 75)
    print(f"{'MEAN':<5} {'':<25} {'':<6} "
          f"{results_df['accuracy'].mean()*100:>6.1f}% "
          f"{results_df['precision'].mean()*100:>6.1f}% "
          f"{results_df['recall'].mean()*100:>6.1f}% "
          f"{results_df['f1'].mean()*100:>6.1f}%")
    print(f"{'STD':<5} {'':<25} {'':<6} "
          f"{results_df['accuracy'].std()*100:>6.1f}% "
          f"{results_df['precision'].std()*100:>6.1f}% "
          f"{results_df['recall'].std()*100:>6.1f}% "
          f"{results_df['f1'].std()*100:>6.1f}%")
    print(f"{'BEST':<5} {'':<25} {'':<6} "
          f"{results_df['accuracy'].max()*100:>6.1f}%")
    print(f"{'WORST':<5} {'':<25} {'':<6} "
          f"{results_df['accuracy'].min()*100:>6.1f}%")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    from data_pipeline import get_data
    from features import build_features
    from targets import add_targets

    logger.info("Loading data...")
    df = get_data("TCS.NS")
    df = build_features(df)
    df = add_targets(df)

    logger.info("Starting walk-forward validation...")
    results = walk_forward_validation(df, start_train_years=3, test_window_months=12)

    print_walk_forward_summary(results)

    os.makedirs("models", exist_ok=True)
    results.to_csv("models/walk_forward_results.csv", index=False)
    logger.info("Results saved to models/walk_forward_results.csv")