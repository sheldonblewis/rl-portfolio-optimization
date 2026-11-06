"""A cost-aware backtester and the metrics used everywhere in the report."""
import numpy as np

TRADING_DAYS = 252


def run(targets, R, cost):
    """Backtest a sequence of target weights.

    targets[k] is either an (N,) weight vector to rebalance to before step k, or None to let the
    portfolio drift. Costs are proportional to one-way turnover. Returns daily net simple returns,
    turnover and the weights actually held.
    """
    T, N = R.shape
    held = np.full(N, 1.0 / N)
    rets = np.zeros(T)
    turn = np.zeros(T)
    weights = np.zeros((T, N))
    first = True
    for k in range(T):
        tgt = targets[k]
        if tgt is not None:
            turn[k] = 0.0 if first else np.abs(tgt - held).sum()
            held = np.asarray(tgt, dtype=float)
            first = False
        weights[k] = held
        gross = held @ (1 + R[k])
        rets[k] = gross * (1 - cost * turn[k]) - 1
        held = held * (1 + R[k]) / gross
    return rets, turn, weights


def metrics(rets, rf, turnover=None):
    """Annualised summary statistics. Sharpe and Sortino are in excess of the cash return."""
    rets = np.asarray(rets)
    ex = rets - rf
    years = len(rets) / TRADING_DAYS
    wealth = np.cumprod(1 + rets)
    cagr = wealth[-1] ** (1 / years) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / (ex.std() + 1e-12) * np.sqrt(TRADING_DAYS)
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    sortino = ex.mean() / (downside + 1e-12) * np.sqrt(TRADING_DAYS)
    peak = np.maximum.accumulate(wealth)
    mdd = (wealth / peak - 1).min()
    out = {
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': mdd,
        'calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
    }
    if turnover is not None:
        out['turnover_ann'] = np.sum(turnover) / years
    return out


def drawdown(rets):
    wealth = np.cumprod(1 + np.asarray(rets))
    return wealth / np.maximum.accumulate(wealth) - 1
