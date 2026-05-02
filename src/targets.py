import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #TARGET 1: Next day direction -- 1 if tomorrow's close > today's close else 0
    df['target_direction_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    #TARGET 2: Multi-Horizon direction -- 1 if prize is higher in N days than today:
    for horizon in [3, 5, 10]:
        df[f'target_direction_{horizon}d'] = (
            df['Close'].shift(-horizon) > df['Close']
        ).astype(int)

    #TARGET 3: Next day continous log return: useful for regression models later
    df['target_return_1d'] = df['log_return'].shift(-1)

    #TARGET 4: Return magnitude: Is tommorrow a big move or a small one? This is useful for risk management
    next_return = df['log_return'].shift(-1).abs()
    df['target_big_move'] = (next_return > next_return.rolling(20).mean()).astype(int)

    #Drop the last N rows where the targets are null since we shifted forward the last rows have no future data
    df = df.dropna(subset=['target_direction_1d', 'target_return_1d'])

    #log class balance (kinda important lowkey) -- a market that goes up 60% of days means a model that always predicts up, gets 6-% percent accuracy without learning anything
    up_pct = df['target_direction_1d'].mean() * 100
    logger.info(f"Target balance: {up_pct: .1f}% up days, {100 - up_pct: .1f}% down days")

    logger.info(f"Targets added. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    import logging
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

    from data_pipeline import get_data
    from features import build_features

    df = get_data("TCS.NS")
    df = build_features(df)
    df = add_targets(df)

    print("\n--- Target columns ---")
    target_cols = [c for c in df.columns if c.startswith('target')]
    print(df[target_cols].head(10))

    print("\n--- Target distributions ---")
    for col in target_cols:
        if df[col].nunique() == 2:  # binary targets only
            print(f"{col}: {df[col].mean()*100:.1f}% positive")

    print(f"\nFinal shape: {df.shape}")