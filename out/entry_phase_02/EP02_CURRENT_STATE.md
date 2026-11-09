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
- 结论标签：期限 `STABLE_PLATEAU`（15–60 递减平台，10 日断崖）、周期 `CONDITIONAL`。
- **周期一栏已被 EP-02h 修正**：02g 的「震荡市是主要超额来源」是**时间暴露**陈述（震荡月多、信号自然多），不是**单笔优势**陈述。02h 显示单笔层面趋势环境显著优于震荡环境（见第 5 节）。
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

## 5. EP-02h：趋势 vs 震荡（已冻结、已执行，2026-11-09）

- 协议 v1.4，SHA256 `B79DE0B84D9F777CBAF173B23C505ABD8283290F670C22A78DD2017EEFBD1CBD`；冻结记录 `EP02H_TREND_RANGE_PROTOCOL_FREEZE.json`。
- 脚本 `scripts/entry_phase_02h_trend_range.py`；报告 `P2H_TREND_RANGE_REPORT.md`。
- 复现：H15 组合 NAV 22,335,139.62、偏差 0.00e+00、672 笔 → 通过冻结复现闸门。
- 样本：信号 11,419 笔；`range` 10,242 笔 / 232 月，`trend` 1,177 笔 / 42 月（`up` 1,141、`down` 36）；`neutral` 0。

| 环境 | 笔数 | 单笔均值 | 中位数 | 胜率 | 盈亏比 | \|MAE\| |
|---|---|---|---|---|---|---|
| range | 10,242 | +1.79% | +0.91% | 53.90% | 1.291 | 6.12% |
| trend | 1,177 | +5.68% | +2.94% | 61.26% | 1.931 | 5.96% |

- Δ = range − trend = **-3.89%**，月度聚类 90% CI **[-7.33%, -0.84%]**，有效 10,000/10,000。
- 四个定义同向为负：主 -3.89%、S-a -2.38%、S-b -5.61%、S-c -2.36%（T1 通过）。
- 闸门：T0 PASS、T1 PASS、T2 FAIL、T3 FAIL、T4 FAIL、T5 PASS、T6 FAIL（Δ_T6 +0.36%，CI 下界 -0.30%）。
- 脚本按字面输出标签：**`YEAR_CONCENTRATION (T4:b)`**（`range` 前 3 年净利占比 65.85% > 60%，最大年份 2025）。
- **实质结论**：单笔层面趋势环境优于震荡环境（CI 上界亦 < 0，非「无差异」）；02g 的「震荡市贡献 84.5% 对数财富」成立但属时间暴露，不构成单笔优势。
- **协议缺陷（未改判，待裁决）**：T4 的 c/d 项要求「删除最大年份后 Δ 仍 > 0」，在 Δ < 0 时 `TREND_EDGE_REAL` 结构性不可达；T4 b 项检验的是 `range` 利润的年份集中度，用于否证「range 优势」，不能用于否证「trend 优势」。因此 `YEAR_CONCENTRATION` 不是对本次结果的正确描述。是否出 v1.5（仅做对称化标签逻辑、不重算数据、不重跑）由用户裁决；**冻结后不自行改判**。
- 补充事实：删除 `range` 最大净利年份 2025 后 Δ = -5.91%，CI [-10.19%, -1.68%]，反方向仍稳健。

## 5b. 已冻结待执行（无）

当前无未执行的冻结协议。EP-02h 已执行完毕；后续任何新诊断须先立协议、再冻结、再执行。

## 6. 冻结文件清单（不得修改）

| 文件 | SHA256 前缀 |
|---|---|
| `EP02C_STOP_PREREGISTER.md` | `0FF76B97…` |
| `EP02E_INVESTABILITY_PROTOCOL.md` | `61A87D33…` |
| `EP02F_ATTR_PROTOCOL.md` | `7F2BC933…` |
| `EP02G_HORIZON_CYCLE_PROTOCOL.md` | `FA8C04BB…` |
| `EP02H_TREND_RANGE_PROTOCOL.md` | `B79DE0B8…`（2026-11-09 冻结） |

配套 `*_FREEZE.json` 与 `EP02D_STOP_DECISION_FREEZE.json` 同为冻结记录，不得改写。

## 7. 读取顺序（按阶段）

1. 基准与口径：`ENTRY_PHASE_02_PROTOCOL_v0.3_FROZEN.md`、`EXECUTION_ASSUMPTIONS.md`
2. 止损裁决：`EP02D_STOP_DECISION.md`
3. 可投资性：`P2E_INVESTABILITY_REPORT.md`
4. 归因：`P2F_ATTRIBUTION_REPORT.md`
5. 期限 × 周期：`P2G_HORIZON_CYCLE_REPORT.md`
6. 趋势 vs 震荡诊断：`P2H_TREND_RANGE_REPORT.md`
7. 待办：无（EP-02h 已执行；新阶段须先立协议）

## 8. 裁决状态一览

| 方向 | 当前决定 |
|---|---|
| B 类信号 | 保留 |
| B0 / 60 日 | 保留为冻结基准 |
| 固定 15 日 | 重点候选，待样本外验证 |
| ATR / σ 组合 | 停止参数扫描 |
| 牛熊识别 | 纳入交互研究，不建开关 |
| 趋势 vs 震荡（EP-02h） | 诊断完成：趋势环境单笔占优；不建开关、不改规则、不筛样本 |
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
| `p2h_*.csv`（11 个产物中的 CSV，已被 `.gitignore` 的 `out/**/*.csv` 排除） | `python scripts/entry_phase_02h_trend_range.py --workers 8` |

注：`p2f_attr_*_p0/p1/p2` 等逐笔台账已删，但 `p2f_attr_decomp.csv`、`p2f_attr_summary.json`、`p2f_attr_yearly.csv` 与 `P2F_ATTRIBUTION_REPORT.md` 保留，桥接结论可复核。
