#!/usr/bin/env python3
"""Generate 90-day z-score metrics for major assets and export CSV/PNG outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


TICKERS: Dict[str, str] = {
    "SP500": "^GSPC",
    "US10Y": "IEF",
    "GOLD": "GC=F",
    "BTC": "BTC-USD",
}

LOOKBACK_DAYS = 90
HIST_WINDOW = 252
HISTORY_YEARS = 5
HEATMAP_DAYS = 180

TZ_LOCAL = "Asia/Tokyo"


@dataclass
class Dataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    zscores: pd.DataFrame


def fetch_prices(tickers: Dict[str, str]) -> pd.DataFrame:
    """Fetch adjusted close prices for the provided tickers."""
    symbols = list(tickers.values())
    print(f"Fetching {len(symbols)} tickers for {HISTORY_YEARS}y history...")
    data = yf.download(
        tickers=symbols,
        period=f"{HISTORY_YEARS}y",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError("No data returned from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        if "Close" not in data.columns:
            raise RuntimeError("Close prices not found in downloaded data.")
        close = data[["Close"]]
        close.columns = symbols

    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])

    rename_map = {symbol: name for name, symbol in tickers.items()}
    close = close.rename(columns=rename_map)

    idx = pd.DatetimeIndex(close.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    close.index = idx.tz_convert(TZ_LOCAL)
    close.index.name = "date_jst"

    close = close.sort_index().ffill().dropna(how="all")
    print(f"Fetched price data shape: {close.shape}")
    return close


def compute_90d_returns(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compute 90-day percentage returns."""
    returns = prices.pct_change(lookback)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.ffill()
    print("Computed 90-day returns.")
    return returns


def zscore_from_history(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate rolling z-scores using the provided window."""
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    rolling_std = returns.rolling(window=window, min_periods=window).std(ddof=0)
    zscores = (returns - rolling_mean) / rolling_std
    zscores = zscores.replace([np.inf, -np.inf], np.nan)
    print("Calculated rolling z-scores.")
    return zscores


def plot_heatmap(zscores: pd.DataFrame, output_path: str, days: int = HEATMAP_DAYS) -> None:
    """Plot a heatmap of the last `days` of z-score data."""
    subset = zscores.tail(days).dropna(how="all")
    if subset.empty:
        print("No z-score data available for heatmap.")
        return

    values = subset.values
    fig, ax = plt.subplots(figsize=(8, max(4, values.shape[0] * 0.15)))
    im = ax.imshow(values, aspect="auto", interpolation="none", origin="upper")

    ax.set_xticks(np.arange(subset.shape[1]))
    ax.set_xticklabels(subset.columns)

    ytick_count = min(subset.shape[0], 12)
    if ytick_count > 1:
        step = max(1, subset.shape[0] // ytick_count)
        yticks = np.arange(0, subset.shape[0], step)
    else:
        yticks = np.array([0])
    ax.set_yticks(yticks)
    ax.set_yticklabels([subset.index[i].strftime("%Y-%m-%d") for i in yticks])

    latest_timestamp = subset.index[-1]
    title = f"90d Z-Scores Heatmap (latest: {latest_timestamp.strftime('%Y-%m-%d %H:%M %Z')})"
    ax.set_title(title)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Date (JST)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Z-Score")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def prepare_dataset() -> Dataset:
    prices = fetch_prices(TICKERS)
    returns = compute_90d_returns(prices, LOOKBACK_DAYS)
    zscores = zscore_from_history(returns, HIST_WINDOW)
    zscores = zscores.dropna(how="all")
    aligned_returns = returns.loc[zscores.index]
    return Dataset(prices=prices, returns=aligned_returns, zscores=zscores)


def save_outputs(dataset: Dataset) -> None:
    zscore_path = os.path.join("data", "zscore_90d.csv")
    latest_path = os.path.join("data", "zscore_90d_latest.csv")
    heatmap_path = os.path.join("plots", "heatmap.png")

    dataset.zscores.to_csv(zscore_path, float_format="%.6f")
    print(f"Saved z-score history to {zscore_path}")

    latest_date = dataset.zscores.index[-1]
    latest_z = dataset.zscores.loc[latest_date]
    latest_returns = dataset.returns.loc[latest_date]

    snapshot = pd.DataFrame({
        "z_90d": latest_z,
        "ret_90d": latest_returns,
    })
    snapshot = snapshot.dropna()
    snapshot = snapshot.sort_values("z_90d", ascending=False)
    snapshot.index.name = "asset"
    snapshot.to_csv(latest_path, float_format="%.6f")
    print(f"Saved latest snapshot to {latest_path}")

    plot_heatmap(dataset.zscores, heatmap_path, days=HEATMAP_DAYS)


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    dataset = prepare_dataset()
    if dataset.zscores.empty:
        raise RuntimeError("No z-score data available to save.")

    save_outputs(dataset)
    latest_timestamp = dataset.zscores.index[-1]
    print(f"Pipeline complete. Latest timestamp: {latest_timestamp}")


if __name__ == "__main__":
    main()
