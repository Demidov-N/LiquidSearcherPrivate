# Data Sources

## WRDS Data Description

| Dataset | Source | Description |
|---------|--------|-------------|
| CRSP | Wharton | Daily security prices, returns, shares outstanding |
| Compustat | S&P | Quarterly fundamental data (book value, earnings, sales) |
| IBES | Refinitiv | Analyst forecasts, target prices, recommendations |
| TAQ | NYSE | Minute-level trade/quote data for liquidity metrics |

## Data Formats

### OHLCV (Price Data)
- **Open**: Opening price
- **High**: Daily high
- **Low**: Daily low
- **Close**: Closing price
- **Volume**: Trading volume
- **VWAP**: Volume-weighted average price

### Fundamentals (G1-G4)
- **G1**: Size and value factors (market cap, book-to-market)
- **G2**: Profitability factors (ROE, ROA, profit margins)
- **G3**: Investment factors (asset growth, capex)
- **G4**: Momentum factors (12-month, 6-month returns)

### Price Features (G5)
- **G5**: Temporal patterns from OHLCV time series
  - Price volatility (realized, implied)
  - Return distributions (skewness, kurtosis)
  - Volume dynamics (turnover, flow)

### Risk Factors (G6)
- **G6**: Idiosyncratic risk measures
  - Beta (market, industry)
  - Residual volatility
  - Correlation with market

## Expected Schemas

### OHLCV DataFrame
```
Column          | Type      | Description
----------------|-----------|-------------
date            | date      | Trading date
symbol          | str       | Stock ticker
open            | float     | Opening price
high            | float     | Daily high
low             | float     | Daily low
close           | float     | Closing price
volume          | int       | Trading volume
vwap            | float     | Volume-weighted avg price
```

### Fundamentals DataFrame
```
Column          | Type      | Description
----------------|-----------|-------------
symbol          | str       | Stock ticker
date            | date      | Reporting period
market_cap      | float     | Market capitalization
book_value      | float     | Book value per share
revenue         | float     | Total revenue
earnings        | float     | Net earnings
roe             | float     | Return on equity
roa             | float     | Return on assets
```

### Liquidity Metrics DataFrame
```
Column          | Type      | Description
----------------|-----------|-------------
symbol          | str       | Stock ticker
date            | date      | Calculation date
amihud          | float     | Amihud illiquidity ratio
darliq          | float     | DArLiQ (depth-adjusted liquidity)
bid_ask_spread  | float     | Effective spread
avg_volume      | float     | Average daily volume
turnover        | float     | Turnover ratio
```

## Data Requirements

- **Universe**: Russell 2000 + S&P 400 (~2,400 stocks)
- **Period**: 2010–2024
- **Fallback**: ML-estimated spreads if TAQ unavailable