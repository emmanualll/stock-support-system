import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)


def run_backtest(df: pd.DataFrame,
                 y_prob: np.ndarray,
                 threshold_long: float = 0.55,
                 threshold_short: float = 0.45,
                 transaction_cost: float = 0.001,
                 stop_loss: float = 0.02,
                 allow_short: bool = True) -> pd.DataFrame:
    """
    Professional backtest with:
    - Long and short positions
    - Dynamic position sizing based on confidence
    - Stop loss
    - Transaction costs

    Position sizing:
        prob > 0.60          → 100% long
        0.55 < prob < 0.60   → 50% long
        0.50 < prob < 0.55   → 25% long
        0.45 < prob < 0.50   → 25% short
        0.40 < prob < 0.45   → 50% short
        prob < 0.40          → 100% short
    """
    test_df = df.iloc[-len(y_prob):].copy()

    bt = pd.DataFrame(index=test_df.index)
    bt['actual_return'] = test_df['log_return'].values
    bt['close']         = test_df['Close'].values
    bt['prob_up']       = y_prob

    # --- Dynamic position sizing ---
    def get_position(prob):
        if not allow_short:
            # Long only — scale position by confidence
            if prob >= 0.60:
                return 1.0
            elif prob >= 0.55:
                return 0.5
            elif prob >= threshold_long:
                return 0.25
            else:
                return 0.0  # stay in cash
        else:
            # Long + short
            if prob >= 0.60:
                return 1.0
            elif prob >= 0.55:
                return 0.5
            elif prob >= 0.50:
                return 0.25
            elif prob >= 0.45:
                return -0.25
            elif prob >= 0.40:
                return -0.5
            else:
                return -1.0
            
    bt['position'] = y_prob.apply(get_position) if hasattr(y_prob, 'apply') \
                    else np.array([get_position(p) for p in y_prob])

    # --- Stop loss ---
    # Track running position return, exit if loss exceeds threshold
    position_return = 0.0
    positions       = bt['position'].values.copy()

    for i in range(1, len(positions)):
        if positions[i-1] != 0:
            position_return += positions[i-1] * bt['actual_return'].iloc[i]
            if position_return < -stop_loss:
                # Stop loss triggered — exit position
                positions[i] = 0.0
                position_return = 0.0
                logger.debug(f"Stop loss triggered on {bt.index[i].date()}")
        else:
            position_return = 0.0

    bt['position'] = positions

    # --- Transaction costs ---
    # Pay cost whenever position changes
    position_change   = pd.Series(positions).diff().abs().fillna(0).values
    bt['trade_cost']  = position_change * transaction_cost

    # --- Strategy return ---
    bt['strategy_return'] = (bt['position'] * bt['actual_return']) - bt['trade_cost']

    # --- Cumulative returns ---
    bt['cumulative_market']   = (1 + bt['actual_return']).cumprod()
    bt['cumulative_strategy'] = (1 + bt['strategy_return']).cumprod()

    # --- Drawdown ---
    roll_max          = bt['cumulative_strategy'].cummax()
    bt['drawdown']    = (bt['cumulative_strategy'] - roll_max) / roll_max

    return bt


def calculate_metrics(bt: pd.DataFrame) -> dict:
    """Professional trading metrics."""
    strat = bt['strategy_return']
    mkt   = bt['actual_return']

    n_years      = len(bt) / 252
    strat_annual = (bt['cumulative_strategy'].iloc[-1] ** (1/n_years)) - 1
    mkt_annual   = (bt['cumulative_market'].iloc[-1]   ** (1/n_years)) - 1

    # Sharpe ratio
    sharpe = (strat.mean() / strat.std()) * np.sqrt(252) if strat.std() > 0 else 0

    # Sortino ratio — like Sharpe but only penalises downside volatility
    downside     = strat[strat < 0].std()
    sortino      = (strat.mean() / downside) * np.sqrt(252) if downside > 0 else 0

    # Max drawdown
    max_drawdown = bt['drawdown'].min()

    # Win rate — only on days we had a position
    active_days = bt[bt['position'] != 0]
    win_rate    = (active_days['strategy_return'] > 0).mean() \
                  if len(active_days) > 0 else 0

    # Profit factor — gross profit / gross loss
    profits  = strat[strat > 0].sum()
    losses   = strat[strat < 0].abs().sum()
    profit_factor = profits / losses if losses > 0 else np.inf

    # Long/short breakdown
    long_days  = (bt['position'] > 0).sum()
    short_days = (bt['position'] < 0).sum()
    flat_days  = (bt['position'] == 0).sum()

    return {
        'strategy_total_return':  (bt['cumulative_strategy'].iloc[-1] - 1) * 100,
        'market_total_return':    (bt['cumulative_market'].iloc[-1] - 1) * 100,
        'strategy_annual_return': strat_annual * 100,
        'market_annual_return':   mkt_annual * 100,
        'sharpe_ratio':           sharpe,
        'sortino_ratio':          sortino,
        'max_drawdown':           max_drawdown * 100,
        'win_rate':               win_rate * 100,
        'profit_factor':          profit_factor,
        'total_days':             len(bt),
        'long_days':              int(long_days),
        'short_days':             int(short_days),
        'flat_days':              int(flat_days),
        'n_trades':               int((bt['position'].diff().abs() > 0).sum()),
    }


def print_backtest_summary(metrics: dict, label: str = ""):
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS {label}")
    print(f"{'='*60}")
    print(f"Period:           {metrics['total_days']} trading days")
    print(f"Long days:        {metrics['long_days']}")
    print(f"Short days:       {metrics['short_days']}")
    print(f"Flat days:        {metrics['flat_days']}")
    print(f"Total trades:     {metrics['n_trades']}")
    print(f"\n--- Returns ---")
    print(f"Strategy total:   {metrics['strategy_total_return']:>8.2f}%")
    print(f"Market total:     {metrics['market_total_return']:>8.2f}%")
    print(f"Strategy annual:  {metrics['strategy_annual_return']:>8.2f}%")
    print(f"Market annual:    {metrics['market_annual_return']:>8.2f}%")
    print(f"Alpha:            {metrics['strategy_annual_return'] - metrics['market_annual_return']:>8.2f}%")
    print(f"\n--- Risk ---")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>8.3f}")
    print(f"Sortino Ratio:    {metrics['sortino_ratio']:>8.3f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>8.2f}%")
    print(f"Win Rate:         {metrics['win_rate']:>8.2f}%")
    print(f"Profit Factor:    {metrics['profit_factor']:>8.3f}")


def walk_forward_backtest(df: pd.DataFrame,
                          start_train_years: int = 3,
                          test_window_months: int = 12,
                          allow_short: bool = True) -> pd.DataFrame:
    """
    Run improved backtest across all walk-forward windows.
    This gives a fair picture across both bull and bear years.
    """
    from model import (get_feature_columns, train_random_forest,
                       train_xgboost)
    from sklearn.preprocessing import RobustScaler

    feature_cols = get_feature_columns(df)
    all_bt_results = []

    first_date       = df.index[0]
    first_test_start = first_date + pd.DateOffset(years=start_train_years)

    test_starts = pd.date_range(
        start=first_test_start,
        end=df.index[-1] - pd.DateOffset(months=test_window_months),
        freq=f'{test_window_months}ME'
    )

    logger.info(f"Walk-forward backtest: {len(test_starts)} rounds")

    for i, test_start in enumerate(test_starts):
        test_end = test_start + pd.DateOffset(months=test_window_months)

        train_df = df[df.index < test_start]
        test_df  = df[(df.index >= test_start) & (df.index < test_end)]

        if len(train_df) < 200 or len(test_df) < 20:
            continue

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

        rf_model  = train_random_forest(X_train, y_train)
        xgb_model = train_xgboost(X_train, y_train)

        rf_prob  = rf_model.predict_proba(X_test)[:, 1]
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        ens_prob = (rf_prob + xgb_prob) / 2

        bt      = run_backtest(test_df, ens_prob, allow_short=allow_short)
        metrics = calculate_metrics(bt)

        logger.info(
            f"Round {i+1}: {test_start.date()} → {test_end.date()} | "
            f"Strategy: {metrics['strategy_total_return']:.1f}% | "
            f"Market: {metrics['market_total_return']:.1f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}"
        )

        all_bt_results.append({
            'round':              i + 1,
            'test_start':         test_start.date(),
            'test_end':           test_end.date(),
            'strategy_return':    metrics['strategy_total_return'],
            'market_return':      metrics['market_total_return'],
            'alpha':              metrics['strategy_total_return'] - metrics['market_total_return'],
            'sharpe':             metrics['sharpe_ratio'],
            'sortino':            metrics['sortino_ratio'],
            'max_drawdown':       metrics['max_drawdown'],
            'win_rate':           metrics['win_rate'],
        })

    return pd.DataFrame(all_bt_results)


def print_wf_backtest_summary(results_df: pd.DataFrame):
    print(f"\n{'='*75}")
    print("WALK-FORWARD BACKTEST SUMMARY")
    print(f"{'='*75}")
    print(f"{'Rnd':<5} {'Period':<25} {'Strategy':>10} {'Market':>8} "
          f"{'Alpha':>8} {'Sharpe':>8}")
    print("-" * 75)

    for _, row in results_df.iterrows():
        print(f"{int(row['round']):<5} "
              f"{str(row['test_start'])} → {str(row['test_end']):<12} "
              f"{row['strategy_return']:>9.1f}% "
              f"{row['market_return']:>7.1f}% "
              f"{row['alpha']:>7.1f}% "
              f"{row['sharpe']:>7.2f}")

    print("-" * 75)
    print(f"{'MEAN':<5} {'':<25} "
          f"{results_df['strategy_return'].mean():>9.1f}% "
          f"{results_df['market_return'].mean():>7.1f}% "
          f"{results_df['alpha'].mean():>7.1f}% "
          f"{results_df['sharpe'].mean():>7.2f}")
    print(f"\nBest year:  {results_df['strategy_return'].max():.1f}%")
    print(f"Worst year: {results_df['strategy_return'].min():.1f}%")
    print(f"Positive alpha years: "
          f"{(results_df['alpha'] > 0).sum()}/{len(results_df)}")


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

    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    rf_prob  = rf_model.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2

    # Compare three strategies
    strategies = [
        ("Long Only (0.50)",  0.50, 0.50, False),
        ("Long Only (0.55)",  0.55, 0.50, False),
        ("Long Only (0.60)",  0.60, 0.50, False),
    ]

    results = []
    for label, thresh_long, thresh_short, allow_short in strategies:
        bt = run_backtest(
            test_df, ens_prob,
            threshold_long=thresh_long,
            threshold_short=thresh_short,
            allow_short=allow_short
        )
        m = calculate_metrics(bt)
        results.append((label, m))
        print_backtest_summary(m, f"({label})")

    # Summary table
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<25} {'Return':>8} {'vs Market':>10} "
          f"{'Sharpe':>8} {'Win%':>8} {'Days In':>8}")
    print("-" * 70)

    bt_market = run_backtest(test_df, ens_prob, threshold_long=0.0,
                             threshold_short=1.0, allow_short=False)
    mkt_return = (bt_market['cumulative_market'].iloc[-1] - 1) * 100

    for label, m in results:
        alpha = m['strategy_total_return'] - mkt_return
        days_in = m['long_days'] + m['short_days']
        print(f"{label:<25} {m['strategy_total_return']:>7.1f}% "
              f"{alpha:>+9.1f}% "
              f"{m['sharpe_ratio']:>8.2f} "
              f"{m['win_rate']:>7.1f}% "
              f"{days_in:>8}")

    print(f"{'Buy & Hold':<25} {mkt_return:>7.1f}%  {'':>9} {'':>8} {'':>8} {'444':>8}")

    # Walk-forward with long-only strict threshold
    print(f"\n\n{'='*60}")
    print("WALK-FORWARD BACKTEST (Long Only, threshold=0.55)")
    print(f"{'='*60}")
    print("Running 5 rounds...")

    wf_results = walk_forward_backtest(df, allow_short=False)
    print_wf_backtest_summary(wf_results)

    os.makedirs("models", exist_ok=True)
    wf_results.to_csv("models/wf_backtest_results.csv", index=False)