import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    logger.info(f"Fetching data for {ticker}, period={period}")
    
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # Flatten MultiIndex columns (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename to standard names just to be safe
    df.columns = [col.strip().title() for col in df.columns]

    logger.info(f"Columns after fix: {df.columns.tolist()}")
    logger.info(f"Fetched {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    #To catch data errors
    invalid_h1 = df['High'] < df['Low']
    #High should always be greater than low
    if invalid_h1.sum() > 0:
        logger.warning(f"Dropping {invalid_h1.sum()} rows where High < Low")
        df = df[~invalid_h1]
    
    invalid_close = (df['Close'] < df['Low']) | (df['Close'] > df['High'])
    if invalid_close.sum() > 0:
        logger.warning(f"Dropping {invalid_close.sum()} rows where Close is outside High or Low")
        df = df[~invalid_close]

    #volume shluld be postitive
    invalid_vol = df['Volume'] <= 0
    if invalid_vol.sum() > 0:
        logger.warning(f"Dropping {invalid_vol.sum()} rows where the volume is negative")
        df = df[~invalid_vol]
    
    #if single day return is more than 50% it is likely a bad data so remove it
    daily_returns = df['Close'].pct_change()
    extreme = daily_returns.abs() > 0.5
    if extreme.sum() > 0:
        logger.warning(f"Found {extreme.sum()} rows where return >50%. Review these Manually pls")
        logger.warning(df[extreme][['Open', 'High', 'Low', 'Close', 'Volume']])
        #we are flagging but not auto remving to ensure if it could be real event

    rows_removed = initial_rows - len(df)
    logger.info(f"Removed {rows_removed} bad rows. Validation is complete. {len(df)} rows remain")

    return df

def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['pct_return'] = df['Close'].pct_change()
    df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['intraday_range'] = (df['High'] - df['Low']) / df['Close']

    return df

def save_data(df: pd.DataFrame, ticker:str, path:str = 'data/raw') -> str:
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{ticker.replace('.', '_')}.csv")
    df.to_csv(filepath)
    logger.info(f"Saved to {filepath}")
    return filepath

def load_data(ticker: str, path: str = 'data/raw') -> pd.DataFrame:
    filepath = os.path.join(path, f"{ticker.replace('.', '_')}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved data for {ticker}. Run fetch first.")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df

def get_data(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    filepath = f"data/raw/{ticker.replace('.', '_')}.csv"
    if not force_refresh and os.path.exists(filepath):
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_age < timedelta(hours=12):
            logger.info("Using cached data (less than 12 hours old)")
            return load_data(ticker)

    df = fetch_stock_data(ticker)
    df = validate_data(df)
    df = add_basic_returns(df)
    save_data(df, ticker)

    return df

def fetch_multiple_stocks(tickers: list,
                          period: str = "10y") -> dict:
    """
    Fetch data for multiple stocks.
    returns a dict of {ticker: dataframe}
    """
    stock_data = {}
    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker}....")
            df = get_data(ticker, force_refresh=False)
            if len(df) > 500:
                stock_data[ticker] = df
                logger.info(f" {ticker}: {len(df)} rows...")
            else:
                logger.warning(f". {ticker}: insufficient data ({len(df)} rows), skipping...")
        except Exception as e:
            logger.warning(f" {ticker}: failed - {e}")
    
    logger.info(f" Succesfully loaded {len(stock_data)}/{len(tickers)} stocks")
    return stock_data


if __name__ == "__main__":
    ticker = "TCS.NS"
    df = get_data(ticker, force_refresh=True)
    
    print("\n--- Shape ---")
    print(df.shape)
    
    print("\n--- Columns ---")
    print(df.columns.tolist())
    
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- Last 5 rows ---")
    print(df.tail())
    
    print("\n--- Data types ---")
    print(df.dtypes)
    
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    
    print("\n--- Basic stats ---")
    print(df[['Close', 'Volume', 'log_return', 'pct_return']].describe())