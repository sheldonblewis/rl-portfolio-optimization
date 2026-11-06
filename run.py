"""Reproduce every number and figure in the README.

    python run.py            # ~10 min on a laptop CPU, results/ and figures/ are rewritten
"""
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch  # noqa: E402

from rlpo import agents, baselines  # noqa: E402
from rlpo.backtest import drawdown, metrics, run  # noqa: E402
from rlpo.data import Market  # noqa: E402

COST = 0.0005  # 5 bps per unit of one-way turnover
RISKS = [1.0, 5.0]
MULTS = [1.0, 4.0, 10.0]
SEEDS = [0, 1, 2, 3, 4]
PPO_SEEDS = [0, 1, 2]
COSTS = [0.0, 0.0005, 0.002, 0.005]
OUT = Path('results')
FIG = Path('figures')


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'train.log', 'a') as f:
        f.write(msg + '\n')


def job(spec):
    torch.set_num_threads(1)
    m = Market()
    kind, kw = spec
    if kind == 'dpo':
        pol, curve = agents.train_dpo(m, log=log, **kw)
        dirichlet = False
    else:
        pol, curve = agents.train_ppo(m, log=log, **kw)
        dirichlet = True
    out = {'kind': kind, **kw, 'curve': curve}
    for split in ('val', 'test'):
        lo, hi = m.span(split)
        out[split] = agents.rollout(pol, m, lo, hi, kw.get('cost', COST), dirichlet)
    return out


def evaluate(m, split, targets, cost=COST):
    lo, hi = m.span(split)
    rets, turn, w = run(targets, m.R[lo:hi], cost)
    return rets, metrics(rets, m.R[lo:hi, m.cash], turn), w


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    (OUT / 'train.log').write_text('')
    m = Market()

    # 1. classical baselines
    rules, spy = baselines.all_rules(m)
    base = {}
    for split in ('val', 'test'):
        lo, hi = m.span(split)
        for name, rule in rules.items():
            if rule is None:
                targets = [spy] + [None] * (hi - lo - 1)
            else:
                targets = baselines.schedule(m, lo, hi, rule)
            base.setdefault(name, {})[split] = evaluate(m, split, targets)

    # 2. sweep risk aversion x training cost multiplier for DPO (seed 0), chosen on validation only
    with Pool(min(6, os.cpu_count())) as pool:
        grid = [{'risk': r, 'cost_mult': c, 'seed': 0, 'cost': COST} for r in RISKS for c in MULTS]
        sweep = pool.map(job, [('dpo', g) for g in grid])
        val_sharpe = {(s['risk'], s['cost_mult']): evaluate(m, 'val', s['val'])[1]['sharpe'] for s in sweep}
        risk, mult = max(val_sharpe, key=val_sharpe.get)
        log(f'chosen (risk, cost_mult) on validation: {(risk, mult)} ({val_sharpe})')

        # 3. seeds for the chosen DPO config, PPO seeds, and the transaction cost sweep
        best = {'risk': risk, 'cost_mult': mult}
        specs = [('dpo', {**best, 'seed': s, 'cost': COST}) for s in SEEDS if s != 0]
        specs += [('ppo', {'seed': s, 'cost': COST}) for s in PPO_SEEDS]
        specs += [('dpo', {**best, 'seed': 0, 'cost': c}) for c in COSTS if c != COST]
        runs = pool.map(job, specs)

    dpo = [s for s in sweep if (s['risk'], s['cost_mult']) == (risk, mult)]
    dpo += [r for r in runs if r['kind'] == 'dpo' and r['cost'] == COST]
    ppo = [r for r in runs if r['kind'] == 'ppo']
    cost_runs = [s for s in dpo if s['seed'] == 0] + [r for r in runs if r['kind'] == 'dpo' and r['cost'] != COST]

    # ensemble = average of the seeds' target weights each day
    def ensemble(group, split):
        return list(np.mean([g[split] for g in group], 0))

    agents_out = {
        'dpo agent (5-seed ensemble)': {s: evaluate(m, s, ensemble(dpo, s)) for s in ('val', 'test')},
        'ppo agent (3-seed ensemble)': {s: evaluate(m, s, ensemble(ppo, s)) for s in ('val', 'test')},
    }
    per_seed = {
        'dpo': [evaluate(m, 'test', d['test'])[1] for d in dpo],
        'ppo': [evaluate(m, 'test', p['test'])[1] for p in ppo],
    }
    cost_table = []
    for r in sorted(cost_runs, key=lambda r: r['cost']):
        met = evaluate(m, 'test', r['test'], r['cost'])[1]
        eq = evaluate(m, 'test', baselines.schedule(m, *m.span('test'), rules['equal weight (1/N)']), r['cost'])[1]
        cost_table.append({'cost_bps': r['cost'] * 1e4, 'agent_sharpe': met['sharpe'],
                           'agent_turnover': met['turnover_ann'], 'equal_weight_sharpe': eq['sharpe']})

    everything = {**agents_out, **base}
    table = {name: {s: v[s][1] for s in ('val', 'test')} for name, v in everything.items()}
    summary = {
        'cost_bps': COST * 1e4,
        'chosen_risk_aversion': risk,
        'chosen_cost_mult': mult,
        'sweep_val_sharpe': {f'risk={k[0]},cost_mult={k[1]}': v for k, v in val_sharpe.items()},
        'sweep_test_turnover': {f"risk={s['risk']},cost_mult={s['cost_mult']}": evaluate(m, 'test', s['test'])[1]['turnover_ann'] for s in sweep},
        'table': table,
        'per_seed_test': per_seed,
        'cost_sweep': cost_table,
        'curves': {f"{r['kind']}-seed{r['seed']}": r['curve'] for r in dpo + ppo},
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    write_table(table, per_seed, cost_table)

    from plots import make_all
    make_all(m, everything, dpo, ppo, cost_table, summary)
    log('done')


def write_table(table, per_seed, cost_table):
    cols = ['cagr', 'vol', 'sharpe', 'sortino', 'max_dd', 'turnover_ann']
    lines = ['| strategy | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for name, v in sorted(table.items(), key=lambda kv: -kv[1]['test']['sharpe']):
        t = v['test']
        cells = [f"{t['cagr']:.1%}", f"{t['vol']:.1%}", f"{t['sharpe']:.2f}", f"{t['sortino']:.2f}",
                 f"{t['max_dd']:.1%}", f"{t['turnover_ann']:.1f}x"]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    for k, v in per_seed.items():
        s = np.array([x['sharpe'] for x in v])
        lines.append(f'{k} per-seed test sharpe: mean {s.mean():.2f}, std {s.std():.2f}, range {s.min():.2f} to {s.max():.2f}')
    lines.append('')
    lines.append('| cost (bps) | agent sharpe | agent turnover | 1/N sharpe |')
    lines.append('|---|---|---|---|')
    for c in cost_table:
        lines.append(f"| {c['cost_bps']:.0f} | {c['agent_sharpe']:.2f} | {c['agent_turnover']:.1f}x | {c['equal_weight_sharpe']:.2f} |")
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
