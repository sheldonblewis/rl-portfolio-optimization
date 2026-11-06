| strategy | cagr | vol | sharpe | sortino | max_dd | turnover_ann |
|---|---|---|---|---|---|---|
| half kelly | 19.9% | 20.7% | 0.85 | 1.18 | -28.8% | 5.7x |
| max sharpe (mean-variance) | 13.9% | 14.2% | 0.80 | 1.10 | -26.6% | 4.9x |
| spy buy & hold | 14.9% | 20.0% | 0.66 | 0.93 | -33.7% | 0.0x |
| equal weight (1/N) | 8.8% | 11.2% | 0.57 | 0.78 | -19.6% | 0.3x |
| ppo agent (3-seed ensemble) | 8.8% | 11.4% | 0.56 | 0.78 | -19.4% | 8.4x |
| dpo agent (5-seed ensemble) | 8.8% | 11.4% | 0.56 | 0.78 | -20.0% | 6.2x |
| 60/40 spy/ief | 9.1% | 12.0% | 0.56 | 0.79 | -21.3% | 0.2x |
| inverse vol | 6.9% | 10.3% | 0.44 | 0.60 | -21.6% | 0.4x |
| min variance | 2.6% | 6.6% | 0.02 | 0.02 | -18.8% | 0.8x |

dpo per-seed test sharpe: mean 0.55, std 0.02, range 0.53 to 0.59
ppo per-seed test sharpe: mean 0.54, std 0.02, range 0.52 to 0.55

| cost (bps) | agent sharpe | agent turnover | 1/N sharpe |
|---|---|---|---|
| 0 | 0.84 | 263.4x | 0.57 |
| 5 | 0.54 | 6.1x | 0.57 |
| 20 | 0.56 | 0.7x | 0.56 |
| 50 | 0.57 | 0.2x | 0.55 |
