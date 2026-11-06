"""Two ways of training the same allocation policy.

The policy follows the EIIE idea (Jiang, Xu & Liang, 2017): one small network scores every asset
independently from its own return window and its current weight, and a softmax across assets
turns the scores into long-only portfolio weights. Sharing the scorer across assets keeps the
parameter count tiny (~2.5k), which matters with only ~2.6k training days.

* DPO, "direct policy optimisation": the market step is differentiable in the weights, so the
  policy is trained by backpropagating the risk-adjusted log return of whole rollouts straight
  into the network. This is the deterministic policy gradient with a known transition model.
* PPO: the model-free route. The same scorer parameterises a Dirichlet over the simplex and is
  trained with clipped PPO and GAE, treating the market purely as a black-box reward.
"""
import copy

import numpy as np
import torch
from torch import nn

from .backtest import metrics

SCALE = 100.0  # daily log returns are ~1e-2, so feed them in percent


def features(x):
    """Compact per-asset signals from a (..., W) window of daily log returns.

    Momentum at 5/20/60 days, volatility at 20/60 days, and the 20-day momentum relative to the
    cross-section. Raw 60-day windows let the network memorise the training period (train Sharpe
    ~9 vs validation ~0, see results/v2), so the policy only sees these six numbers per asset.
    """
    mom = torch.stack([x[..., -h:].sum(-1) for h in (5, 20, 60)], -1) * SCALE / 5
    vol = torch.stack([x[..., -h:].std(-1) for h in (20, 60)], -1) * SCALE
    rel = (mom[..., 1:2] - mom[..., 1:2].mean(-2, keepdim=True))
    return torch.clamp(torch.cat([mom, vol, rel], -1), -10, 10)


class Scorer(nn.Module):
    def __init__(self, window, hidden=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(6, hidden), nn.GELU())
        self.head = nn.Linear(hidden + 1, 1)

    def forward(self, x, prev):
        # x: (B, N, W) log-return windows, prev: (B, N) weights currently held
        h = self.enc(features(x))
        return self.head(torch.cat([h, prev.unsqueeze(-1)], -1)).squeeze(-1)


def step(w, prev, R, cost):
    """One differentiable market step. Returns net log return and the drifted weights."""
    turnover = (w - prev).abs().sum(-1)
    gross = (w * (1 + R)).sum(-1)
    net = gross * (1 - cost * turnover)
    drifted = w * (1 + R) / gross.unsqueeze(-1)
    return torch.log(net), drifted


@torch.no_grad()
def rollout(policy, market, lo, hi, cost, dirichlet=False):
    """Run a trained policy through days lo..hi-1 with its true held weights."""
    n = len(market.assets)
    prev = torch.full((1, n), 1.0 / n)
    targets = []
    for t in range(lo, hi):
        x = torch.from_numpy(market.X[t]).unsqueeze(0)
        z = policy(x, prev)
        w = _dirichlet_mean(z) if dirichlet else torch.softmax(z, -1)
        targets.append(w[0].numpy().astype(float))
        R = torch.from_numpy(market.R[t]).float().unsqueeze(0)
        _, prev = step(w, prev, R, cost)
    return targets


def _score(policy, market, lo, hi, cost, dirichlet=False):
    from .backtest import run

    targets = rollout(policy, market, lo, hi, cost, dirichlet)
    rets, turn, _ = run(targets, market.R[lo:hi], cost)
    return metrics(rets, market.R[lo:hi, market.cash], turn)['sharpe']


def train_dpo(market, cost=0.0005, risk=5.0, cost_mult=1.0, iters=1500, batch=64, horizon=64, lr=1e-3,
              seed=0, eval_every=50, log=None):
    """Maximise E[log return - cash] - risk * Var[log return] over random rollouts in train.

    `cost_mult` inflates transaction costs during training only. Untreated, the policy learns to
    churn the whole book daily to chase short-horizon signal that real costs then erase.

    Early-stops on validation Sharpe; returns the best policy and its learning curve.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(market.assets)
    pol = Scorer(market.window)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-2)
    lo, hi = market.span('train')
    vlo, vhi = market.span('val')
    X = torch.from_numpy(market.X)
    R = torch.from_numpy(market.R).float()
    best, best_state, curve = -np.inf, None, []
    for it in range(1, iters + 1):
        starts = rng.integers(lo, hi - horizon, batch)
        prev = torch.full((batch, n), 1.0 / n)
        logs, cash = [], []
        for k in range(horizon):
            t = starts + k
            w = torch.softmax(pol(X[t], prev), -1)
            lr_t, prev = step(w, prev, R[t], cost * cost_mult)
            logs.append(lr_t)
            cash.append(torch.log1p(R[t, market.cash]))
        ex = torch.stack(logs) - torch.stack(cash)  # (horizon, batch)
        loss = -(ex.mean() - risk * ex.var(0).mean()) * 1e3
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        if it % eval_every == 0:
            tr = _score(pol, market, lo, hi, cost)
            va = _score(pol, market, vlo, vhi, cost)
            curve.append((it, tr, va))
            if log:
                log(f'dpo seed={seed} risk={risk} it={it} train_sharpe={tr:.2f} val_sharpe={va:.2f}')
            if va > best:
                best, best_state = va, copy.deepcopy(pol.state_dict())
    pol.load_state_dict(best_state)
    return pol, curve


def _dirichlet_mean(z):
    a = nn.functional.softplus(z) + 1e-3
    return a / a.sum(-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, window, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n * 6, 64), nn.Tanh(), nn.Linear(64, 1))
        self.window = window

    def forward(self, x, prev):
        return self.net(features(x).flatten(-2)).squeeze(-1)


def train_ppo(market, cost=0.0005, iters=300, envs=32, horizon=128, epochs=4, clip=0.2,
              gamma=0.9, lam=0.95, lr=1e-3, conc=20.0, seed=0, eval_every=10, log=None):
    """Clipped PPO with a Dirichlet policy. `conc` scales the concentration so early exploration
    is not absurdly noisy (a Dirichlet with all alphas ~1 is uniform over the simplex)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(market.assets)
    pol = Scorer(market.window)
    critic = Critic(market.window, n)
    opt = torch.optim.Adam(list(pol.parameters()) + list(critic.parameters()), lr=lr)
    lo, hi = market.span('train')
    vlo, vhi = market.span('val')
    X = torch.from_numpy(market.X)
    R = torch.from_numpy(market.R).float()
    best, best_state, curve = -np.inf, None, []

    def dist(z):
        return torch.distributions.Dirichlet((nn.functional.softplus(z) + 1e-3) * conc)

    for it in range(1, iters + 1):
        starts = rng.integers(lo, hi - horizon, envs)
        prev = torch.full((envs, n), 1.0 / n)
        buf = {k: [] for k in ('x', 'prev', 'a', 'logp', 'r', 'v')}
        with torch.no_grad():
            for k in range(horizon):
                t = starts + k
                z = pol(X[t], prev)
                d = dist(z)
                a = d.sample().clamp_min(1e-6)
                a = a / a.sum(-1, keepdim=True)
                lr_t, nxt = step(a, prev, R[t], cost)
                r = (lr_t - torch.log1p(R[t, market.cash])) * 100
                for key, val in zip(buf, (X[t], prev, a, d.log_prob(a), r, critic(X[t], prev))):
                    buf[key].append(val)
                prev = nxt
        rew, val = torch.stack(buf['r']), torch.stack(buf['v'])
        adv = torch.zeros_like(rew)
        gae = torch.zeros(envs)
        for k in reversed(range(horizon)):
            nv = val[k + 1] if k + 1 < horizon else torch.zeros(envs)
            delta = rew[k] + gamma * nv - val[k]
            gae = delta + gamma * lam * gae
            adv[k] = gae
        ret = adv + val
        flat = {k: torch.stack(v).flatten(0, 1) for k, v in buf.items() if k not in ('r', 'v')}
        adv_f = adv.flatten()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        ret_f = ret.flatten()
        for _ in range(epochs):
            for idx in torch.randperm(len(adv_f)).split(512):
                d = dist(pol(flat['x'][idx], flat['prev'][idx]))
                ratio = torch.exp(d.log_prob(flat['a'][idx]) - flat['logp'][idx])
                pg = -torch.min(ratio * adv_f[idx], ratio.clamp(1 - clip, 1 + clip) * adv_f[idx]).mean()
                vf = (critic(flat['x'][idx], flat['prev'][idx]) - ret_f[idx]).pow(2).mean()
                loss = pg + 0.5 * vf - 1e-3 * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        if it % eval_every == 0:
            tr = _score(pol, market, lo, hi, cost, dirichlet=True)
            va = _score(pol, market, vlo, vhi, cost, dirichlet=True)
            curve.append((it, tr, va))
            if log:
                log(f'ppo seed={seed} it={it} train_sharpe={tr:.2f} val_sharpe={va:.2f}')
            if va > best:
                best, best_state = va, copy.deepcopy(pol.state_dict())
    pol.load_state_dict(best_state)
    return pol, curve
