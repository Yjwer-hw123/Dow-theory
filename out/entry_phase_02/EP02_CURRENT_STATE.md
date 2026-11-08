# EP-02 当前状态（唯一入口）

> 进入本目录任何人 / Agent：**先读这一页，再读其它文件。**
> 本目录里大量报告是阶段产物，文件名不带时效标记，按文件名或按「结论」类关键词检索会读到已被取代的版本。
> 最后更新：2026-11-09

## 0. 本目录属于哪条线

- **属于**：ENTRY-PHASE 入场策略（B 类信号）——唯一在推进的研究主线。
- **不属于**：2026-09-25 冻结的**道氏 Regime 状态机**。已于 2026-11-09 **物理拆出**到 `d:\python_prog\dow-regime`（`core/`、`config/`、`adapters/`、`research/`、`report/`、`data/` 副本、`tests/run_tests.py`、`run_regime.py` 等）。本仓库不再含该层代码。
- 拆出前已验证：`entry_phase*.py` 不导入 `core` / `config` / `adapters`；拆分后入口线导入自检通过。
- 本目录内所有文件均为 EP-02 产物。「Regime 层打不过 MA200」等结论**不属于**本目录，也不构成本目录任何阶段的前置条件。

## 1. 当前有效冻结基准：B0（60 bar，不止损）

| 项目 | 数值 |
|---|---|
| 信号池 | 11,419 笔（`out/entry_phase_03/p3_features.csv`） |
| 出场 | 第 60 根 bar 收盘卖出，不设止损/止盈 |
| 组合 | 100 万、3 仓、每笔 1/3 NAV、同日先到先得、回款次日可用 |
| 期末 NAV | **5,719,552.238656651** |
| CAGR / MDD | 8.38% / 75.35% |
| 成交 | 201 笔（成交率 1.76%） |
| 单笔净期望 / 胜率 / PF | +8.21% / 54.78% / 2.385 |
| 集中度 | 2006 年入场的 9 笔贡献净收益 96.54%；剔除后净利 163,077 元 |

复现锚点：任何脚本跑 B0 都必须得到上述 NAV 与 201 笔，相对偏差 ≤ 1e-9，否则视为环境或口径错误。

## 2. 重点候选（未采用、未冻结）

**固定 15 日上限、不止损**（EP-02g 口径）：

- 期末 22,335,139.62 元、CAGR 15.42%、MDD 49.48%、672 笔、单笔均值 1.84%、胜率 53.6%。
- 对数财富 45.1% 来自 2005–2007；22 年中 12 年跑赢沪指；牛市期间年化 41.6% vs 沪指 67.3%。
- 结论标签：期限 `STABLE_PLATEAU`（15–60 递减平台，10 日断崖）、周期 `CONDITIONAL`（震荡市为主要超额来源）。
- 定位：**研究候选，不得表述为「已验证」或「可采用」**。

## 3. 已否决 / 已失效

| 方向 | 结论 | 依据 |
|---|---|---|
| 固定 10 日 | 179 万、CAGR 2.73%、MDD 81.63%、941 笔；累计超额 -44.9% | EP-02g |
| 5 条固定百分比止损 | 全部 `REJECT_ALL` | EP-02d |
| ATR×2 / ATR×1.5 / ATR×3 | 60 日下曾优于 B0；**15 日上限下 ATR×2 终值仅 500 万，较固定 15 日损失约 77%** | EP-02f 演示 / EP-02g |
| σ×2 | 15 日下 2,484 万，但同样属参数扫描产物，不再继续 | EP-02g |
| 20 仓 / 单笔 5% NAV | 未提高单笔质量、未降低回撤，只压缩资金占用 | ATTR_* / L2 |

## 4. 已被取代的结论与对应新文件

| 旧结论（文件） | 状态 | 取代者 |
|---|---|---|
| 协议 v0.1 / v0.2 / v0.3 及两份 diff（已删除） | 被冻结版取代 | `ENTRY_PHASE_02_PROTOCOL_v0.3_FROZEN.md` |
| `BACKTEST_REPORT_B_L1.md`（L1 逐笔 +6.27%） | 单笔口径，仍成立但不代表组合 | `P2G_HORIZON_CYCLE_REPORT.md` |
| `L2_REPORT.md` / `ATTR_CONCLUSION_ONEPAGE.md`（20 仓） | 机制已被 3 仓主线取代 | EP-02e / EP-02f |
| `P2C_HOLDING_ANATOMY_REPORT.md` | 持有结构诊断，结论已被止损测试覆盖 | `P2C_STOP_BACKTEST_REPORT.md` → `EP02D_STOP_DECISION.md` |
| 「持约 60 个交易日」 | 不再唯一 | 见第 1、2 节 |
| 「容量不足、方向应停止」 | 仅指 3 仓 33 万名义的实盘容量；组合级归因显示容量收益在 10bp 假设内真实 | `P2F_ATTRIBUTION_REPORT.md` |
| 「收益集中在 2006 年」 | 仅 B0 口径 | 15 日口径的集中度见第 2 节 |

## 5. 下一步待冻结（尚未执行）

- 文件：`EP02H_TREND_RANGE_PROTOCOL.md` v1.4
- SHA256：`B79DE0B84D9F777CBAF173B23C505ABD8283290F670C22A78DD2017EEFBD1CBD`
- 状态：**未冻结、未执行、无 `EP02H_*_FREEZE.json`**
- 内容：趋势结构 vs 震荡环境下 B 类信号表现差异；固定 H=15；三个预注册判定 T0–T6；不改规则、不建开关、不扫参数。
- 待用户确认后，才写 FREEZE 记录并实现 `scripts/entry_phase_02h_trend_range.py`。

## 6. 冻结文件清单（不得修改）

| 文件 | SHA256 前缀 |
|---|---|
| `EP02C_STOP_PREREGISTER.md` | `0FF76B97…` |
| `EP02E_INVESTABILITY_PROTOCOL.md` | `61A87D33…` |
| `EP02F_ATTR_PROTOCOL.md` | `7F2BC933…` |
| `EP02G_HORIZON_CYCLE_PROTOCOL.md` | `FA8C04BB…` |
| `EP02H_TREND_RANGE_PROTOCOL.md` | `B79DE0B8…`（尚未冻结） |

配套 `*_FREEZE.json` 与 `EP02D_STOP_DECISION_FREEZE.json` 同为冻结记录，不得改写。

## 7. 读取顺序（按阶段）

1. 基准与口径：`ENTRY_PHASE_02_PROTOCOL_v0.3_FROZEN.md`、`EXECUTION_ASSUMPTIONS.md`
2. 止损裁决：`EP02D_STOP_DECISION.md`
3. 可投资性：`P2E_INVESTABILITY_REPORT.md`
4. 归因：`P2F_ATTRIBUTION_REPORT.md`
5. 期限 × 周期：`P2G_HORIZON_CYCLE_REPORT.md`
6. 当前待办：`EP02H_TREND_RANGE_PROTOCOL.md`

## 8. 裁决状态一览

| 方向 | 当前决定 |
|---|---|
| B 类信号 | 保留 |
| B0 / 60 日 | 保留为冻结基准 |
| 固定 15 日 | 重点候选，待样本外验证 |
| ATR / σ 组合 | 停止参数扫描 |
| 牛熊识别 | 纳入交互研究，不建开关 |
| 实盘 | 不开放（10bp 滑点不含冲击成本；2,233 万时单仓约 744 万） |

本页只描述状态，不构成采用建议，也不构成对 EP-02 策略线的终止裁决。

## 9. 已清理的中间产物（可再生，不影响结论）

2026-11-09 删除 L1 / L2 / p2c / p2d / p2f 的逐笔与日度中间表（约 35 MB）。各阶段**报告、summary.json、冻结记录**全部保留。

需要重新生成时：

| 缺失文件 | 重跑脚本 |
|---|---|
| `trades_b_l1.csv`、`annual_b_l1.csv`、`summary_b_l1.json` | `python scripts/entry_phase_02_l1.py` |
| `l2_*` | `python scripts/entry_phase_02_l2.py` |
| `attr_*` | `python scripts/entry_phase_02_attribution.py` |
| `p2c_*` | `python scripts/entry_phase_02c_anatomy.py` |
| `p2d_trades.csv`、`p2d_main_nav_S*.csv` | `python scripts/entry_phase_02d_stoploss.py --workers 8` |
| `p2f_attr_trades_*.csv`、`p2f_attr_new_trades.csv`、`p2f_attr_early_exit.csv` | `python scripts/entry_phase_02f_attribution.py` |

注：`p2f_attr_*_p0/p1/p2` 等逐笔台账已删，但 `p2f_attr_decomp.csv`、`p2f_attr_summary.json`、`p2f_attr_yearly.csv` 与 `P2F_ATTRIBUTION_REPORT.md` 保留，桥接结论可复核。
