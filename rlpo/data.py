"""Price data, splits and the feature tensor the agents see.

Convention: weights chosen at the close of day t-1 earn the simple return from close t-1 to
close t. The features for step t only use log returns up to and including day t-1, so there is
no lookahead anywhere in the pipeline.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC', 'VNQ', 'BIL']
CASH = 'BIL'  # 1-3 month t-bills, used as the cash asset and the risk-free rate
WINDOW = 60

SPLITS = {
    'train': ('2007-01-01', '2017-12-31'),
    'val': ('2018-01-01', '2019-12-31'),
    'test': ('2020-01-01', '2026-12-31'),
}

ROOT = Path(__file__).resolve().parent.parent


def download(path=ROOT / 'data' / 'prices.csv'):
    import yfinance as yf

    df = yf.download(ASSETS, start='2007-01-01', auto_adjust=True, progress=False)['Close'][ASSETS]
    df.dropna().to_csv(path)


def load_prices(path=ROOT / 'data' / 'prices.csv'):
    return pd.read_csv(path, index_col=0, parse_dates=True)[ASSETS]


class Market:
    """Simple returns R[t], log-return windows X[t] and split index ranges."""

    def __init__(self, prices=None, window=WINDOW):
        prices = load_prices() if prices is None else prices
        self.dates = prices.index
        self.assets = list(prices.columns)
        self.cash = self.assets.index(CASH)
        p = prices.to_numpy()
        self.R = np.zeros_like(p)
        self.R[1:] = p[1:] / p[:-1] - 1
        logr = np.zeros_like(p)
        logr[1:] = np.log(p[1:] / p[:-1])
        self.window = window
        # X[t] holds the `window` log returns of days t-window .. t-1, shape (T, N, window)
        self.X = np.zeros((len(p), p.shape[1], window), dtype=np.float32)
        for t in range(window + 1, len(p)):
            self.X[t] = logr[t - window:t].T
        self.first = window + 1

    def span(self, split):
        lo, hi = SPLITS[split]
        idx = np.where((self.dates >= lo) & (self.dates <= hi))[0]
        return max(idx[0], self.first), idx[-1] + 1
