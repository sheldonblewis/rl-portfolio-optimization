"""Classical allocation strategies the agents are compared against.

All estimation-based strategies refit on a trailing 252-day window and rebalance monthly
(every 21 trading days), long-only, fully invested across the 10 assets including cash.
"""
import numpy as np
from scipy.optimize import minimize

LOOKBACK = 252
EVERY = 21


def _shrunk_cov(r, delta=0.3):
    # annualised: daily covariances are ~1e-5, below SLSQP's default tolerance
    s = np.cov(r, rowvar=False) * 252
    return (1 - delta) * s + delta * np.diag(np.diag(s))


def _solve(obj, n, exclude=None):
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    bounds = [(0, 0) if i == exclude else (0, 1) for i in range(n)]
    res = minimize(obj, np.full(n, 1 / n), bounds=bounds, constraints=cons, method='SLSQP')
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(hist, cash):
    # over the risky assets only, otherwise the answer is trivially 100% t-bills
    S = _shrunk_cov(hist)
    return _solve(lambda w: w @ S @ w, hist.shape[1], exclude=cash)


def max_sharpe(hist, cash):
    # mean shrunk halfway to the cross-sectional average: raw sample means are mostly noise
    mu = hist.mean(0) * 252
    mu = 0.5 * mu + 0.5 * mu.mean()
    S = _shrunk_cov(hist)
    rf = mu[cash]
    return _solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ S @ w + 1e-12), hist.shape[1])


def half_kelly(hist, cash):
    # growth-optimal (kelly) fraction, halved: maximise w.mu - (1/2k) w'Sw with k = 0.5
    mu = hist.mean(0) * 252
    mu = 0.5 * mu + 0.5 * mu.mean()
    S = _shrunk_cov(hist)
    return _solve(lambda w: -(w @ mu - w @ S @ w), hist.shape[1])


def risk_parity(hist, cash):
    # naive risk parity: inverse volatility, cash excluded (its vol is ~0)
    vol = hist.std(0)
    inv = np.where(np.arange(len(vol)) == cash, 0.0, 1 / (vol + 1e-12))
    return inv / inv.sum()


def fixed(weights):
    return lambda hist, cash: weights


def schedule(market, lo, hi, rule):
    """Monthly target weights for steps lo..hi-1, None on non-rebalance days."""
    targets = []
    for k, t in enumerate(range(lo, hi)):
        if k % EVERY:
            targets.append(None)
            continue
        hist = market.R[max(1, t - LOOKBACK):t]
        targets.append(rule(hist, market.cash))
    return targets


def all_rules(market):
    n = len(market.assets)
    a = market.assets
    eq = np.full(n, 1 / n)
    spy = np.eye(n)[a.index('SPY')]
    sixty = 0.6 * np.eye(n)[a.index('SPY')] + 0.4 * np.eye(n)[a.index('IEF')]
    return {
        'equal weight (1/N)': fixed(eq),
        '60/40 spy/ief': fixed(sixty),
        'spy buy & hold': None,  # handled as a single initial allocation
        'min variance': min_variance,
        'max sharpe (mean-variance)': max_sharpe,
        'half kelly': half_kelly,
        'inverse vol': risk_parity,
    }, spy
