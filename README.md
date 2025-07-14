# 📈 AlphaLine – A Linear Machine Learning Model for Alpha Generation

**LineAlpha** is a linear model-based framework for alpha generation in financial markets. It integrates statistical learning, factor modeling, and backtesting infrastructure to predict asset returns and develop systematic trading strategies.

---

## 🔍 Overview

The goal of LineAlpha is to leverage both market and fundamental features using a linear machine learning approach to generate consistent alpha. The project utilizes a range of engineered features, technical indicators, and macro factors such as the Fama-French 5-factor model.

---

## ⚙️ Feature Engineering

The dataset is built from historical OHLC (Open, High, Low, Close) stock data and enriched with a variety of predictive features:

- **Price-Derived Features**:
  - Stock returns
  - Momentum indicators
  - Volatility metrics
  - Simple Moving Averages (SMA)
  - Min-max price over a rolling window
  - Lagged versions of all the above features

- **Technical Indicators**:
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - Average True Range (ATR)
  - Moving Average Convergence Divergence (MACD)

- **Factor-Based Features**:
  - Fama-French risk factors including:
    - Size (SMB)
    - Value (HML)
    - Profitability (RMW)
    - Investment (CMA)
    - Momentum (WML)

---

## 🧪 Statistical Testing

To ensure the robustness and validity of the linear models, a range of statistical tests were conducted:

- **Assumption Testing**:
  - Normality of residuals
  - Homoskedasticity
  - Multicollinearity (via VIF)
  - Correlation matrix analysis

- **Model Evaluation Metrics**:
  - R-squared (R²)
  - AIC (Akaike Information Criterion)
  - BIC (Bayesian Information Criterion)
  - Omnibus test
  - Skewness and Kurtosis of residuals
  - Durbin-Watson statistic
  - T-tests for coefficient significance

---

## 🧪 Backtesting and Forward Testing

- **Framework**: Zipline (backtesting engine)
- **Evaluation**: PyFolio (performance analytics)

### Metrics used:
- **Return Analysis**:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Maximum Drawdown
  - Beta and Alpha
  - Volatility and Omega Ratio

- **Information Analysis**:
  - Information Ratio
  - Hit Ratio
  - Cumulative returns
  - Annualized returns

---

## 🧬 Regularization Models

To mitigate overfitting and enhance model generalization, the following regularized linear models were used and compared:

- **Lasso Regression** – L1 regularization
- **Ridge Regression** – L2 regularization
- **ElasticNet** – Combination of L1 and L2

Each model’s performance was compared based on backtest returns and statistical evaluation scores.

---

## 📊 Comparative Strategy Analysis

LineAlpha’s performance was benchmarked against well-known factor strategies:

- **Fama-French 3-Factor and 5-Factor Models**
- **SMB (Small Minus Big)**
- **HML (High Minus Low)**
- **RMW (Robust Minus Weak)**
- **CMA (Conservative Minus Aggressive)**
- **WML (Winner Minus Loser – Momentum)**

Each strategy was evaluated using Zipline for backtesting and forward-testing to determine relative outperformance.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`, `scikit-learn`, `statsmodels`
  - `ta` (technical analysis indicators)
  - `zipline-reloaded` (backtesting)
  - `pyfolio` (performance analysis)
  - `matplotlib`, `seaborn` (visualization)

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/yourusername/AlphaLine.git

