import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']
    for window in [5, 10, 20, 50, 200]:
        df[f'ma_{window}'] = close.rolling(window).mean()
        df[f'close_to_ma_{window}'] = (close - df[f'ma_{window}']) / df[f'ma_{window}']


    #exponential moving averages i.e., more weight on recent prices
    for window in [9, 21, 55]:
        df[f'ema_{window}'] = close.ewm(span=window, adjust = False).mean()
        df[f'close_to_ema_{window}'] = (close - df[f'ema_{window}']) / df[f'ema_{window}']

    df['ma_5_20_cross'] = df['ma_5'] - df['ma_20']
    df['ma_20_50_cross'] = df['ma_20'] - df['ma_50']
    df['ema_9_21_cross'] = df['ema_9'] - df['ema_21']

    logger.info("Trend indicators added")
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']

    #RSI is the realtive strength index it measures the speed and magnitude of the recent price changes

    for window in [9, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window -1, adjust = False).mean()
        avg_loss = loss.ewm(com=window -1, adjust = False).mean()
        rs = avg_gain / avg_loss
        df[f'rsi_{window}'] = 100 - (100/ (1 + rs))

 #Rate of change is the percentage over N days i.e., How much the price has moved in N days?
    for window in [5, 10, 20]:
        df[f'roc_{window}'] = close.pct_change(periods = window) * 100

    #Wiliams %R measures whre close is relative to the recent high low range
    for window in [14, 21]:
        highest_high = df['High'].rolling(window).max()
        lowest_low = df['Low'].rolling(window).min()
        df[f'williams_r_{window}'] = ((highest_high - close) / (highest_high - lowest_low)) * -100

    #MACD - Moving Average Convergaecnce and Divergence 
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']

    logger.info("Momentum indicators Added")
    return df

def add_volatality_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']

    for window in [20]:
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        df[f'bb_pct_b_{window}'] = (close - lower) / (upper - lower)
        df[f'bb_bandwidth_{window}'] = (upper - lower) / ma

    for window in [14, 21]:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - close.shift(1)).abs()
        low_close = (df['Low'] - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis = 1).max(axis=1)
        df[f'atr_{window}'] = true_range.ewm(span = window, adjust = False).mean()
        df[f'atr_{window}_pct'] = df[f'atr_{window}'] / close

    for window in [10, 20, 30]:
        df[f'hist_vol_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(252)

    logger.info("Volatility Indicators are added")
    return df

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']
    volume = df['Volume']

    for window in [10, 20]:
        df[f'volume_ma_{window}'] = volume.rolling(window).mean()
        df[f'volume_ratio_{window}'] = volume / df[f'volume_ma_{window}']

    #OBV - On Balance Volume that is if obv rises with price it is healthy
    #if it fails while price rise it means divergence so a potential reversal 
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_ma_20'] = obv.rolling(20).mean()

    df['obv_signal'] = (obv - df['obv_ma_20']) / df['obv_ma_20'].abs()

    df['pv_trend'] = (df['pct_return'] * volume).cumsum()

    logger.info("Volume Indicators Added")
    return df

def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    #price action features captures the shape of each candles and gaps between these sessions 
    # a long upper wich candle indicates sellers pushed the price back down i.e.e, bearish
    #long lower wich indicates the buyers pushed the price back up (bullish)
    #large body means conviction in direction

    df = df.copy()
    df['candle_body'] = (df['Close'] - df['Open']) / df['Open']
    #How far did the price go above the body
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['upper_wick_ratio'] = upper_wick / (df['High'] - df['Low'] + 1e-9)

    #How far did the wick go below the body
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['lower_wick_ratio'] = lower_wick / (df['High'] - df['Low'] + 1e-9)

    df['gap_up'] = ((df['Open'] > df['Close'].shift(1)) &
        (df['Low'] > df['Close'].shift(1))).astype(int)
    df['gap_down'] = ((df['Open'] < df['Close'].shift(1)) &
        (df['High'] < df['Close'].shift(1))).astype(int)
    
    logger.info("Prices action features added")
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # This is used to give the model memory
    # Our model will see one row at a time, without lag it will have no idea what happened yesterday or the day before
    #Lag ecplicitly hand it that historical context
    df = df.copy()

    for lag in [1, 2, 3, 4, 5, 10]:
        df[f'return_lag_{lag}'] = df['log_return'].shift(lag)

    for lag in [1, 2, 3]:
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
    
    for lag in [1, 2]:
        df[f'volume_ratio_lag_{lag}'] = df['volume_ratio_20'].shift(lag)

    logger.info("Lag features added")
    return df
def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    # Here we are adding some more market context because when training the model we found that it is not performing that well enough
    # 1. NIFTY IT index: because of the sector significance
    # 2. NiFTY50: Because it follows the broad market
    # 3. USD/INR : Currency impact on IT earnings
    # Without these the model is blind to all the conditions that happened, we know the market stuggled in 2024-2026,
    #  the IT sector correction was a macro even and it waas not visible in tcs price alnoe
    import yfinance as yf
    logger.info("Fetching market context data:..")

    start = df.index[0].strftime('%Y-%m-%d')
    end = df.index[-1].strftime('%Y-%m-%d')
    logger.info(f"Fetching context from {start} to {end}")

    context_tickers = {
        'nifty_it':  '^CNXIT',
        'nifty_50':  '^NSEI',
        'usd_inr':   'USDINR=X',
    }

    for name, ticker in context_tickers.items():
        try:
            raw = yf.download(ticker, start= start, end= end, auto_adjust = True, progress = False)

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            
            if raw.empty:
                logger.warning(f"No data for {ticker}, skipping..")
                continue

            close = raw['Close'].reindex(df.index).ffill().bfill()
            
            
            nan_count = close.isna().sum()
            logger.info(f"{ticker}: {len(close)} rows after reindex, {nan_count} NaN values")

            if close.isna().sum() > len(df) * 0.1:
                logger.warning(f"{ticker}: too many NaN after reindex ({close.isna().sum()}), skipping")
                continue

            ma_20 = close.rolling(20).mean()
            ma_50 = close.rolling(50).mean()

            df[f'{name}_return_1d']    = close.pct_change()
            df[f'{name}_return_5d']    = close.pct_change(5)
            df[f'{name}_close_to_ma20'] = (close - ma_20) / ma_20
            df[f'{name}_close_to_ma50'] = (close - ma_50) / ma_50
            df[f'{name}_trend']        = (ma_20 - ma_50) / ma_50

            logger.info(f"Added {name} features")

        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            import traceback
            traceback.print_exc()
    return df

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    window = 14
    high, low, close = df['High'], df['Low'], df['Close']

    plus_dm = high.diff()
    minus_dm = - low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=window, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['adx'] = dx.ewm(span=window, adjust=False).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    #is current volatitluty high or low vs the recent history
    vol_20 = df['log_return'].rolling(20).std()
    vol_60 = df['log_return'].rolling(60).std()
#Volatality > 1 = it is expanding or less than 1 then its contracting
    df['vol_regime'] = vol_20 / vol_60

    df['dist_to_52w_high'] = (df['Close'] - df['High'].rolling(252).max()) / df['High'].rolling(252).max()
    df['dist_to_52w_low'] = (df['Close'] - df['Low'].rolling(252).min()) / df['Low'].rolling(252).min()

    logger.info("Regime features added")
    return df

def add_stock_identity(df: pd.DataFrame,ticker: str) -> pd.DataFrame:
    """
    Add tock identity, this tails the model which type of stock it's looking at.
    Without this the model can't distiguish between an it and an banking stock
    """
    df = df.copy()

    sector_map = {
        'TCS.NS':        'it',
        'INFY.NS':       'it',
        'WIPRO.NS':      'it',
        'HCLTECH.NS':    'it',
        'TECHM.NS':      'it',
        'HDFCBANK.NS':   'banking',
        'ICICIBANK.NS':  'banking',
        'KOTAKBANK.NS':  'banking',
        'AXISBANK.NS':   'banking',
        'RELIANCE.NS':   'energy',
        'HINDUNILVR.NS': 'consumer',
        'BAJFINANCE.NS': 'finance',
        'MARUTI.NS':     'auto',
        'SUNPHARMA.NS':  'pharma',
        'TITAN.NS':      'consumer',
    }

    sector = sector_map.get(ticker, 'unknown')

    df['is_it'] = 1 if sector == 'it' else 0
    df['is_banking'] = 1 if sector == 'banking' else 0
    df['is_energy'] = 1 if sector == 'energy' else 0
    df['is_consumer'] = 1 if sector == 'consume' else 0
    df['is_finance'] = 1 if sector == 'finance' else 0
    df['is_auto'] = 1 if sector == 'auto' else 0
    df['is_pharma'] = 1 if sector == 'pharma' else 0

    logger.info(f"Added identity features for {ticker} (sector: {sector})")
    return df


def build_features(df: pd.DataFrame,
                   ticker: str = None) -> pd.DataFrame:

    logger.info("Starting feature engineering...")

    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatality_indicators(df)
    df = add_volume_indicators(df)
    df = add_price_action_features(df)
    df = add_lag_features(df)
    df = add_regime_features(df)
    df = add_market_context(df)    

    if ticker:
        df = add_stock_identity(df, ticker)

    #drop the rows with NaN, these appear at the start because of rolling 
    initial = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial - len(df)} rows with NaN (expected — rolling window warmup)")
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    logger.info(f"Total features: {len(df.columns)}")
    return df


if __name__ == "__main__":
    from data_pipeline import get_data

    df_raw = get_data("TCS.NS")
    df_features = build_features(df_raw)

    print("\n--- All features ---")
    for col in df_features.columns:
        print(col)

    print(f"\nTotal: {len(df_features.columns)} columns")
    print(f"Rows: {len(df_features)}")