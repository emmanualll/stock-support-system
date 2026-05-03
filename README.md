<<<<<<< HEAD
**# AI-Powered Stock Decision Support System

A machine learning system that analyzes historical stock data to provide 
probabilistic insights, risk evaluation, and explainable signals for 
better investment decisions.

## What it does
- Fetches 10 years of historical data for any NSE stock
- Engineers 75+ technical, momentum, volatility and market context features
- Trains an ensemble of Random Forest + XGBoost on 15 stocks simultaneously
- Outputs probability of price direction, risk level, and SHAP explanations

## Results
- 52.3% average accuracy across 5-year walk-forward validation
- Beats buy-and-hold during bear markets with strict confidence filtering
- Works on any NSE stock ticker

## Tech Stack
Python, Pandas, NumPy, scikit-learn, XGBoost, SHAP, yfinance

## Setup

### 1. Clone the repo
git clone https://github.com/yourusername/stock-project.git
cd stock-project

### 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Train the universal model
python src/universal_model.py

### 5. Run analysis on any stock
python main.py TCS.NS
python main.py RELIANCE.NS
python main.py SUNPHARMA.NS

## Project Structure
src/
  data_pipeline.py     # Data fetching and validation
  features.py          # Feature engineering (75+ features)
  targets.py           # Target variable creation
  model.py             # Random Forest + XGBoost training
  universal_model.py   # Multi-stock universal model
  walk_forward.py      # Walk-forward validation
  backtest.py          # Backtesting engine
  risk.py              # Risk scoring
  explainability.py    # SHAP explainability
  feature_selection.py # SHAP-based feature selection
  tuning.py            # Hyperparameter tuning
main.py                # Single entry point

## Important Note
Models and data are not included in this repo (too large).
Run universal_model.py to train from scratch — takes ~10 minutes.**
=======
# AI-Powered Stock Decision Support System

A machine learning system that analyzes historical stock data to provide 
probabilistic insights, risk evaluation, and explainable signals for 
better investment decisions.

## What it does
- Fetches 10 years of historical data for any NSE stock
- Engineers 75+ technical, momentum, volatility and market context features
- Trains an ensemble of Random Forest + XGBoost on 15 stocks simultaneously
- Outputs probability of price direction, risk level, and SHAP explanations

## Results
- 52.3% average accuracy across 5-year walk-forward validation
- Beats buy-and-hold during bear markets with strict confidence filtering
- Works on any NSE stock ticker

## Tech Stack
Python, Pandas, NumPy, scikit-learn, XGBoost, SHAP, yfinance

## Setup

### 1. Clone the repo
git clone https://github.com/yourusername/stock-project.git
cd stock-project

### 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Train the universal model
python src/universal_model.py

### 5. Run analysis on any stock
python main.py TCS.NS
python main.py RELIANCE.NS
python main.py SUNPHARMA.NS

## Project Structure
src/
  data_pipeline.py     # Data fetching and validation
  features.py          # Feature engineering (75+ features)
  targets.py           # Target variable creation
  model.py             # Random Forest + XGBoost training
  universal_model.py   # Multi-stock universal model
  walk_forward.py      # Walk-forward validation
  backtest.py          # Backtesting engine
  risk.py              # Risk scoring
  explainability.py    # SHAP explainability
  feature_selection.py # SHAP-based feature selection
  tuning.py            # Hyperparameter tuning
main.py                # Single entry point

## Important Note
Models and data are not included in this repo (too large).
Run universal_model.py to train from scratch — takes ~10 minutes.
>>>>>>> cc9c107 (Added .gitignore to improve struct and README.md)
