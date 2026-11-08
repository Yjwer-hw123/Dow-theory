# EP-02 全池复权覆盖率报告

- 生成时间：2026-11-08 10:52:50
- 股票池：5555
- 复权因子覆盖：5546（99.84%，门槛 ≥99%）
- 可回测（因子+面板构建成功）：5483（98.70%；差额为历史不足或数据异常的次新股，非复权因子失败）
- 失败：72；原因：{'prepare_fail': 63, 'no_factor': 9}
- 不可计算 bar 占比（股票×交易日）：0.3274%（门槛 ≤1%）

## 失败股票
| code | reason | bars |
|---|---|---|
| 000003 | prepare_fail | 2419 |
| 000015 | prepare_fail | 1972 |
| 000047 | prepare_fail | 1811 |
| 000508 | prepare_fail | 940 |
| 000556 | prepare_fail | 1692 |
| 000588 | prepare_fail | 1264 |
| 000653 | prepare_fail | 1277 |
| 000658 | prepare_fail | 1262 |
| 000675 | prepare_fail | 1270 |
| 000689 | prepare_fail | 1258 |
| 001232 | prepare_fail | 63 |
| 001237 | prepare_fail | 114 |
| 001248 | prepare_fail | 86 |
| 001257 | prepare_fail | 148 |
| 001312 | prepare_fail | 134 |
| 001365 | prepare_fail | 118 |
| 001393 | prepare_fail | 119 |
| 001399 | prepare_fail | 90 |
| 002660 | no_factor | 3381 |
| 002661 | no_factor | 3486 |
| 002662 | no_factor | 3378 |
| 002663 | no_factor | 3326 |
| 002664 | no_factor | 3344 |
| 002665 | no_factor | 2696 |
| 002666 | no_factor | 3540 |
| 301513 | prepare_fail | 136 |
| 301531 | prepare_fail | 123 |
| 301583 | prepare_fail | 80 |
| 301599 | prepare_fail | 127 |
| 301655 | prepare_fail | 51 |
| 301666 | prepare_fail | 134 |
| 301669 | prepare_fail | 102 |
| 301677 | prepare_fail | 66 |
| 301680 | prepare_fail | 165 |
| 301682 | prepare_fail | 152 |
| 301683 | prepare_fail | 147 |
| 301686 | prepare_fail | 28 |
| 301688 | prepare_fail | 42 |
| 301689 | prepare_fail | 36 |
| 301696 | prepare_fail | 141 |
| 301697 | prepare_fail | 43 |
| 301699 | prepare_fail | 37 |
| 301707 | prepare_fail | 60 |
| 301717 | prepare_fail | 58 |
| 600625 | prepare_fail | 1839 |
| 600709 | prepare_fail | 1474 |
| 600813 | prepare_fail | 2008 |
| 601008 | no_factor | 4727 |
| 601091 | prepare_fail | 31 |
| 601123 | prepare_fail | 43 |
| 603293 | prepare_fail | 137 |
| 603407 | prepare_fail | 123 |
| 603435 | prepare_fail | 114 |
| 603448 | prepare_fail | 39 |
| 603459 | prepare_fail | 143 |
| 603468 | prepare_fail | 61 |
| 688635 | prepare_fail | 111 |
| 688781 | prepare_fail | 152 |
| 688797 | prepare_fail | 92 |
| 688801 | prepare_fail | 35 |
| 688806 | prepare_fail | 73 |
| 688808 | prepare_fail | 131 |
| 688811 | prepare_fail | 141 |
| 688813 | prepare_fail | 148 |
| 688820 | prepare_fail | 134 |
| 688825 | prepare_fail | 69 |
| 688826 | prepare_fail | 53 |
| 688828 | prepare_fail | 58 |
| 688835 | prepare_fail | 48 |
| 688836 | prepare_fail | 52 |
| 688837 | prepare_fail | 32 |
| 689009 | no_factor | 1461 |

说明：qfq = raw / f（新浪累计因子），raw 为新浪不复权全量重抓；
`amount` 保持原值；复权失败股票区间标 `not_computable`，不进入 L1 分子分母。