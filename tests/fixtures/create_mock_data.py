"""Mock data generator for testing the training pipeline."""

import os
from datetime import datetime, timedelta

import numpy as np
import polars as pl


def generate_business_days(start_date: str, end_date: str) -> list:
    """Generate business days between start and end date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        # Monday=0, Friday=4 (skip Saturday=5, Sunday=6)
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)

    return dates


def create_random_walk_prices(n_days: int, start_price: float = 100.0) -> np.ndarray:
    """Generate prices using random walk."""
    # Random walk with small daily returns
    daily_returns = np.random.normal(0.0005, 0.015, n_days)
    prices = start_price * np.exp(np.cumsum(daily_returns))
    return prices


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from prices."""
    returns = np.diff(prices) / prices[:-1]
    # Add a leading zero for the first day
    return np.concatenate([[0.0], returns])


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from prices."""
    log_returns = np.diff(np.log(prices))
    # Add a leading zero for the first day
    return np.concatenate([[0.0], log_returns])


def compute_realized_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling realized volatility."""
    vol = np.zeros_like(returns)
    for i in range(len(returns)):
        start_idx = max(0, i - window + 1)
        vol[i] = np.std(returns[start_idx : i + 1]) * np.sqrt(252)
    return vol


def compute_momentum(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute momentum as price change over window."""
    momentum = np.zeros_like(prices)
    for i in range(len(prices)):
        start_idx = max(0, i - window)
        momentum[i] = (prices[i] / prices[start_idx] - 1) * 100
    return momentum


def compute_rsi(returns: np.ndarray, window: int = 14) -> np.ndarray:
    """Compute Relative Strength Index."""
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)

    rsi = np.zeros_like(returns)
    for i in range(len(returns)):
        start_idx = max(0, i - window + 1)
        avg_gain = np.mean(gains[start_idx : i + 1])
        avg_loss = np.mean(losses[start_idx : i + 1])

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(prices: np.ndarray) -> tuple:
    """Compute MACD and signal line."""
    # EMAs
    ema_12 = np.zeros_like(prices)
    ema_26 = np.zeros_like(prices)

    ema_12[0] = prices[0]
    ema_26[0] = prices[0]

    alpha_12 = 2 / (12 + 1)
    alpha_26 = 2 / (26 + 1)

    for i in range(1, len(prices)):
        ema_12[i] = alpha_12 * prices[i] + (1 - alpha_12) * ema_12[i - 1]
        ema_26[i] = alpha_26 * prices[i] + (1 - alpha_26) * ema_26[i - 1]

    macd = ema_12 - ema_26

    # Signal line (9-day EMA of MACD)
    signal = np.zeros_like(macd)
    signal[0] = macd[0]
    alpha_signal = 2 / (9 + 1)

    for i in range(1, len(macd)):
        signal[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal[i - 1]

    return macd, signal


def create_mock_features(
    symbol: str,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "data/processed/features",
) -> None:
    """Generate synthetic feature data for a symbol.

    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Directory to save parquet file
    """
    # Generate business days
    dates = generate_business_days(start_date, end_date)
    n_days = len(dates)

    if n_days == 0:
        raise ValueError(f"No business days between {start_date} and {end_date}")

    # Generate random walk prices
    start_price = np.random.uniform(50.0, 500.0)
    close_prices = create_random_walk_prices(n_days, start_price)

    # Generate OHLC from close
    daily_volatility = np.random.uniform(0.01, 0.03, n_days)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, daily_volatility)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, daily_volatility)))
    open_prices = low_prices + np.random.uniform(0, 1, n_days) * (high_prices - low_prices)

    # Ensure proper ordering: low <= open <= close <= high
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))

    # Generate volume
    base_volume = np.random.uniform(1e6, 50e6)
    volume = base_volume * (1 + np.random.normal(0, 0.3, n_days))
    volume = np.maximum(volume, 1e5)

    # Compute temporal features
    returns = compute_returns(close_prices)
    log_returns = compute_log_returns(close_prices)
    realized_vol_20d = compute_realized_vol(returns, 20)
    momentum_20d_temp = compute_momentum(close_prices, 20)
    momentum_60d_temp = compute_momentum(close_prices, 60)
    rsi_14d = compute_rsi(returns, 14)
    macd, signal = compute_macd(close_prices)

    # Generate categorical features (constant per symbol)
    # Use simple indices: gsector 0-10, ggroup 0-24
    gsector = np.random.randint(0, 11)
    ggroup = np.random.randint(0, 25)

    # Generate continuous tabular features
    # These vary slightly over time but are relatively stable
    market_beta_60d = np.random.uniform(0.5, 1.5) + np.random.normal(0, 0.05, n_days)
    market_beta_60d = np.clip(market_beta_60d, 0.3, 2.0)

    downside_beta_60d = np.random.uniform(0.3, 1.8) + np.random.normal(0, 0.05, n_days)
    downside_beta_60d = np.clip(downside_beta_60d, 0.1, 2.5)

    idiosyncratic_vol_60d = np.random.uniform(0.1, 0.5) + np.random.normal(0, 0.02, n_days)
    idiosyncratic_vol_60d = np.clip(idiosyncratic_vol_60d, 0.05, 0.8)

    momentum_20d_tab = compute_momentum(close_prices, 20)
    momentum_60d_tab = compute_momentum(close_prices, 60)
    momentum_120d_tab = compute_momentum(close_prices, 120)

    volatility_20d = compute_realized_vol(returns, 20)
    volatility_60d = compute_realized_vol(returns, 60)

    # Financial ratios (relatively stable)
    roe = np.random.uniform(0.05, 0.35) + np.random.normal(0, 0.01, n_days)
    roe = np.clip(roe, -0.2, 0.5)

    roa = np.random.uniform(0.02, 0.20) + np.random.normal(0, 0.005, n_days)
    roa = np.clip(roa, -0.1, 0.3)

    debt_to_equity = np.random.uniform(0.1, 2.0) + np.random.normal(0, 0.05, n_days)
    debt_to_equity = np.clip(debt_to_equity, 0.0, 4.0)

    price_to_book = np.random.uniform(0.8, 8.0) + np.random.normal(0, 0.2, n_days)
    price_to_book = np.clip(price_to_book, 0.1, 15.0)

    price_to_earnings = np.random.uniform(5.0, 50.0) + np.random.normal(0, 2.0, n_days)
    price_to_earnings = np.clip(price_to_earnings, 1.0, 100.0)

    market_cap = np.random.uniform(1e9, 2e12) * (1 + np.cumsum(np.random.normal(0, 0.0003, n_days)))

    dividend_yield = np.random.uniform(0.0, 0.05) + np.random.normal(0, 0.001, n_days)
    dividend_yield = np.clip(dividend_yield, 0.0, 0.1)

    # Create DataFrame
    df = pl.DataFrame(
        {
            "date": dates,
            # Temporal features (13)
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
            "returns": returns,
            "log_returns": log_returns,
            "realized_vol_20d": realized_vol_20d,
            "momentum_20d": momentum_20d_temp,
            "momentum_60d": momentum_60d_temp,
            "rsi_14d": rsi_14d,
            "macd": macd,
            "signal": signal,
            # Continuous tabular features (15)
            "market_beta_60d": market_beta_60d,
            "downside_beta_60d": downside_beta_60d,
            "idiosyncratic_vol_60d": idiosyncratic_vol_60d,
            "momentum_20d_tab": momentum_20d_tab,
            "momentum_60d_tab": momentum_60d_tab,
            "momentum_120d": momentum_120d_tab,
            "volatility_20d": volatility_20d,
            "volatility_60d": volatility_60d,
            "roe": roe,
            "roa": roa,
            "debt_to_equity": debt_to_equity,
            "price_to_book": price_to_book,
            "price_to_earnings": price_to_earnings,
            "market_cap": market_cap,
            "dividend_yield": dividend_yield,
            # Categorical features (2)
            "gsector": gsector,
            "ggroup": ggroup,
        }
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save as parquet
    output_path = os.path.join(output_dir, f"{symbol}_features.parquet")
    df.write_parquet(output_path)
    print(f"Created mock data: {output_path} ({n_days} rows)")


def main():
    """Generate mock data for multiple symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM"]

    print("Generating mock feature data...")
    for symbol in symbols:
        create_mock_features(symbol)

    print(f"\nCompleted! Generated data for {len(symbols)} symbols in data/processed/features/")


if __name__ == "__main__":
    main()
