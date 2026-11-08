# P10-D 字面交易覆盖层（Dow Trading Overlay）

- 指数配对：沪深300 + 中证500
- 样本：2005-01-04 ~ 2026-09-24 (5279 交易日)
- 成本：单边 5bp，信号次日生效

**这一层不是道氏理论**，是把市场状态映射成仓位的规则，必须显式隔离，否则无法回答"是道氏有效，还是仓位规则有效"。

## 覆盖层绩效

| strategy | total_return | cagr | ann_vol | sharpe | sortino | max_drawdown | calmar | exposure | time_wait_pct | n_entries | turnover_per_year | in_market_ann_ret | in_market_hit | fee_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CASH_MODEL | 1.3529 | 0.0417 | 0.0831 | 0.534 | 0.218 | -0.2652 | 0.157 | 0.117 | 0.883 | 16 | 1.53 | 0.4683 | 0.595 | 5 |
| HOLD_MODEL | 8.1892 | 0.1117 | 0.2349 | 0.569 | 0.691 | -0.5596 | 0.2 | 0.899 | 0.101 | 14 | 1.29 | 0.161 | 0.533 | 5 |
| PAIR_MODEL(A/B) | 10.0169 | 0.1214 | 0.2833 | 0.547 | 0.679 | -0.7242 | 0.168 | 0.117 | 0.883 | 16 | 1.55 | 0.4683 | 0.595 | 5 |

## 三种映射

- `CASH_MODEL`：RISK_ON → 持有指数 A；WAIT / RISK_OFF → 现金。
- `HOLD_MODEL`：只有 RISK_OFF 才离场，WAIT 仍持有（适合长期持有者）。
- `PAIR_MODEL(A/B)`：RISK_ON → 持有 A；否则持有 B（防御资产）。

