import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rlpo.backtest import drawdown  # noqa: E402

plt.rcParams.update({
    'figure.dpi': 140, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'legend.frameon': False,
})
AGENT = '#1f3a5f'
PPO = '#7d93b8'
SHOWN = ['equal weight (1/N)', '60/40 spy/ief', 'spy buy & hold', 'max sharpe (mean-variance)', 'half kelly', 'inverse vol']
GREYS = ['#999', '#c9a46c', '#bbb', '#b88a9f', '#8fb39a', '#d0b0a0']


def make_all(m, everything, dpo, ppo, cost_table, summary):
    lo, hi = m.span('test')
    dates = m.dates[lo:hi]
    dpo_name = 'dpo agent (5-seed ensemble)'
    ppo_name = 'ppo agent (3-seed ensemble)'

    # 1. equity curves
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, c in zip(SHOWN, GREYS):
        ax.plot(dates, np.cumprod(1 + everything[name]['test'][0]), color=c, lw=1, label=name)
    ax.plot(dates, np.cumprod(1 + everything[ppo_name]['test'][0]), color=PPO, lw=1.4, label=ppo_name)
    ax.plot(dates, np.cumprod(1 + everything[dpo_name]['test'][0]), color=AGENT, lw=2, label=dpo_name)
    ax.set_yscale('log')
    ax.set_title('out-of-sample growth of $1, 2020 to 2026 (5 bps costs)')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig('figures/equity_test.png')
    plt.close(fig)

    # 2. agent allocation through the test period
    w = everything[dpo_name]['test'][2]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cmap = plt.get_cmap('tab10')
    ax.stackplot(dates, w.T, labels=m.assets, colors=[cmap(i) for i in range(len(m.assets))], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title('dpo agent holdings, out of sample')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/weights_test.png')
    plt.close(fig)

    # 3. drawdowns
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, c in [('spy buy & hold', '#bbb'), ('equal weight (1/N)', '#999'), (ppo_name, PPO), (dpo_name, AGENT)]:
        ax.plot(dates, drawdown(everything[name]['test'][0]), color=c, lw=1.4 if 'agent' in name else 1, label=name)
    ax.set_title('drawdown from peak')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/drawdown_test.png')
    plt.close(fig)

    # 4. learning curves
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, group, title in [(axes[0], dpo, 'dpo'), (axes[1], ppo, 'ppo')]:
        for r in group:
            c = np.array(r['curve'])
            ax.plot(c[:, 0], c[:, 1], color='#bbb', lw=0.8)
            ax.plot(c[:, 0], c[:, 2], color=AGENT if title == 'dpo' else PPO, lw=1)
        ax.set_title(f'{title}: train (grey) vs validation sharpe')
        ax.set_xlabel('iteration')
    fig.tight_layout()
    fig.savefig('figures/learning_curves.png')
    plt.close(fig)

    # 5. transaction cost sensitivity
    c = np.array([[x['cost_bps'], x['agent_sharpe'], x['agent_turnover'], x['equal_weight_sharpe']] for x in cost_table])
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(c[:, 0], c[:, 1], 'o-', color=AGENT, label='agent trained at that cost')
    ax.plot(c[:, 0], c[:, 3], 's--', color='#999', label='equal weight')
    ax.set_xlabel('transaction cost (bps)')
    ax.set_ylabel('test sharpe')
    ax2 = ax.twinx()
    ax2.bar(c[:, 0], c[:, 2], width=2.5, color=AGENT, alpha=0.15)
    ax2.set_ylabel('annual turnover (x)')
    ax2.grid(False)
    ax.legend(fontsize=7, loc='lower left')
    ax.set_title('costs teach the agent to trade less')
    fig.tight_layout()
    fig.savefig('figures/cost_sensitivity.png')
    plt.close(fig)

    # 6. sharpe bar chart with per-seed spread
    t = summary['table']
    names = sorted(t, key=lambda k: t[k]['test']['sharpe'])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cols = [AGENT if k == dpo_name else PPO if k == ppo_name else '#bbb' for k in names]
    ax.barh(names, [t[k]['test']['sharpe'] for k in names], color=cols)
    for key, name in [('dpo', dpo_name), ('ppo', ppo_name)]:
        s = [x['sharpe'] for x in summary['per_seed_test'][key]]
        y = names.index(name)
        ax.scatter(s, [y] * len(s), color='k', s=8, zorder=3)
    ax.set_xlabel('out-of-sample sharpe (dots: individual seeds)')
    fig.tight_layout()
    fig.savefig('figures/sharpe_test.png')
    plt.close(fig)
