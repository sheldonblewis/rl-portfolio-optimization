| strategy | cagr | vol | sharpe | sortino | max_dd | turnover_ann |
|---|---|---|---|---|---|---|
| max sharpe (mean-variance) | 12.5% | 12.7% | 0.79 | 1.08 | -22.4% | 4.3x |
| spy buy & hold | 14.9% | 20.0% | 0.66 | 0.93 | -33.7% | 0.0x |
| equal weight (1/N) | 8.8% | 11.2% | 0.57 | 0.78 | -19.6% | 0.3x |
| min variance | 8.8% | 11.2% | 0.57 | 0.78 | -19.6% | 0.3x |
| ppo agent (3-seed ensemble) | 8.8% | 11.4% | 0.56 | 0.78 | -19.4% | 8.4x |
| 60/40 spy/ief | 9.1% | 12.0% | 0.56 | 0.79 | -21.3% | 0.2x |
| inverse vol | 6.9% | 10.3% | 0.44 | 0.60 | -21.6% | 0.4x |
| half kelly | 6.5% | 11.7% | 0.36 | 0.48 | -24.0% | 2.7x |
| dpo agent (5-seed ensemble) | 6.3% | 14.3% | 0.31 | 0.44 | -30.4% | 177.6x |

dpo per-seed test sharpe: mean 0.19, std 0.12, range 0.01 to 0.34
ppo per-seed test sharpe: mean 0.54, std 0.02, range 0.52 to 0.55

| cost (bps) | agent sharpe | agent turnover | 1/N sharpe |
|---|---|---|---|
| 0 | 0.84 | 263.4x | 0.57 |
| 5 | 0.27 | 173.8x | 0.57 |
| 20 | -0.13 | 49.9x | 0.56 |
| 50 | 0.30 | 6.1x | 0.55 |
