# dow-theory —— EP-02 入场策略（B 类信号）

本仓库**只保留一条研究线**：ENTRY-PHASE 入场策略。核心问题是「上升趋势中的缩量回调买入（B 类信号）到底值不值得做，以及怎么持有」。

> **唯一入口**：[out/entry_phase_02/EP02_CURRENT_STATE.md](out/entry_phase_02/EP02_CURRENT_STATE.md)
> 进入本仓库（尤其是 Agent）请先读那一页。`out/` 下大量文件是阶段产物，按文件名检索很容易读到已被取代的版本。

---

## 1. 边界：这里没有什么

- **不含道氏 Regime 状态机**。`core/`、`config/`、`adapters/`、`research/`、`report/`、`run_regime.py`、`run_p10*.py`、`run_p11*.py`、`run_p12_survivorship.py` 已于 2026-11-09 **物理拆出**至 [`d:\python_prog\dow-regime`](../dow-regime)。
  - 该层 2026-09-25 已冻结，结论是「择时有信息但打不过 MA200（CAGR 4.17% vs 11.12%）」，与本仓库的任何结论**无关**，也不构成本仓库任何阶段的前置条件。
- 本仓库的 `entry_phase*.py` **不导入** `core` / `config` / `adapters`；拆分后已做过导入自检。
- 两套东西混放导致的「读到旧版本」问题，由上述唯一入口文档 + 物理拆分解决。

## 2. 当前状态（快照，可能落后于入口文档）

| 项目 | 状态 |
|---|---|
| B 类信号 | 保留为研究主线 |
| 基准 B0 | 60 bar 收盘卖出、3 仓、每笔 1/3 NAV；NAV `5,719,552.238656651`、MDD 75.35%、201 笔、单笔净期望 +8.21%、胜率 54.78%、PF 2.385。**冻结基准，任何重跑必须复现该 NAV 与 201 笔（偏差 ≤ 1e-9）** |
| 固定 15 日上限 | **重点候选，未采用**：期末 22,335,139.62、CAGR 15.42%、MDD 49.48%、672 笔、单笔 1.84%、胜率 53.6% |
| 止损 / ATR / σ | 已停止参数扫描（5 条固定百分比止损全部 `REJECT_ALL`） |
| 牛熊识别 | 只作为交互研究的分层变量，不建开关 |
| 实盘 | 不开放（10bp 滑点不含市场冲击成本；2,233 万时单仓约 744 万） |
| 下一步 | `EP02H_TREND_RANGE_PROTOCOL.md` v1.4，**未冻结、未执行**，待确认 |

阶段报告按顺序读：`ENTRY_PHASE_02_PROTOCOL_v0.3_FROZEN.md` → `EP02D_STOP_DECISION.md` → `P2E_INVESTABILITY_REPORT.md` → `P2F_ATTRIBUTION_REPORT.md` → `P2G_HORIZON_CYCLE_REPORT.md`。

## 3. 目录结构

```
dow-theory/
├── README.md                  # 本文件
├── scripts/                   # 全部可执行脚本（见第 4 节）
├── data/                      # stock_loader.py（cache/stocks 读取）、universe/*.json
├── cache/                     # 本地行情缓存，见第 5 节（不进版本控制）
├── docs/                      # 道氏原书与笔记（历史资料，入场线不依赖）
├── notes/                     # 结论纪要、立项规划
├── out/
│   ├── entry_phase_02/        # EP-02 产物 + 唯一入口 EP02_CURRENT_STATE.md
│   ├── entry_phase_03/        # 信号池 p3_features.csv（11,419 笔）
│   └── step1/                 # STEP1 多周期趋势过滤（NO-GO，已冻结）
└── tests/                     # test_pit_universe.py
```

## 4. 常用命令

解释器：`C:/Users/16079/AppData/Local/Programs/Python/Python311/python.exe`（pandas / numpy 已装）。

```powershell
# 信号池特征（B 类信号来源）
python scripts/entry_phase_03_features.py

# 逐笔口径（L1）
python scripts/entry_phase_02_l1.py

# 止损裁决（EP-02d，主组合回测）
python scripts/entry_phase_02d_stoploss.py --workers 8

# 期限 × 周期（EP-02g，固定 15 日候选的来源）
python scripts/entry_phase_02g_horizon_cycle.py --workers 8

# 可投资性 / 归因
python scripts/entry_phase_02e_investability.py
python scripts/entry_phase_02f_attribution.py
```

其它：`build_qfq.py`（合成 qfq）、`fetch_*.py`（抓取）、`audit_lineage.py`（数据血缘）、`step1_trend_validation.py`、`uptrend_detector.py`。

## 5. 数据与缓存（约 1.6 GB，均被脚本引用）

| 目录 | 大小 | 用途 |
|---|---|---|
| `cache/qfq` | 938 MB | 前复权价，p2c/p2d/p2e/p2f/p2g/p3 直接读 |
| `cache/raw` | 658 MB | raw 价，成本计算与 `qfq = raw/f` 合成；同时是价格覆盖审计的数据源 |
| `cache/factors` | 3 MB | 复权因子 |

**股票池名单已与缓存解耦**：5,555 只沪深 A 的代码固化在 `data/universe/stock_codes.json`，所有脚本通过 `data.pit_universe.load_stock_codes()` 读取。原 `cache/stocks`（655 MB，与 `cache/raw` 代码集合完全相同、仅用于提供名单）已于 2026-11-09 清空释放；`data/stock_loader.py` 只在真正重新下载个股时按需重建该目录。

另有根目录 6 个指数 parquet（`000001_daily.parquet` 等）。**不要随手删**，删除后需重新抓取才能复现冻结基准。

## 6. 口径约定（改动即破坏可比性）

- 收益：`net = qfq 空间收益 − raw 空间成本`，成本统一用 `entry_phase_02_l1.eval_trade(..., "base")`（0.030%/边 + 10bp 滑点、佣金最低 5 元、过户费与印花税分段）。
- 执行：收盘确认信号 → **次日开盘**成交；开盘跌停顺延；涨跌幅限制按板块 10%/20%/30%。
- 组合：100 万、3 仓、每笔 1/3 NAV、同日 `(signal_date, code)` 先到先得、回款次日可用。
- 冻结协议 + SHA256：运行前强制校验，不符即终止。已冻结：`EP02C` `0FF76B97…`、`EP02E` `61A87D33…`、`EP02F` `7F2BC933…`、`EP02G` `FA8C04BB…`。

## 7. 历史结论（属于本仓库，已裁决）

- **STEP1 多周期趋势过滤（月 MA20 + 周 MA60）：NO-GO（2026-11-06 冻结）**。四个假设全部 REJECTED；B 只保留 A_m 约 38% 的样本却没换来更好收益。否定对象仅限「月线 + 周线趋势过滤能提高未来收益」，**不等于**「日线回调买入无效」。
- **EP-02d 止损**：5 条固定百分比止损全部 `REJECT_ALL`。
- **EP-02e 可投资性**：3 仓容量下的收益在 10bp 假设内真实，但实盘容量极小。
- **EP-02f 归因**：`CAPACITY_REAL_SIGNAL` —— 组合级超额主要来自周转，信号级单笔质量反而下降。
- **EP-02g 期限 × 周期**：期限 `STABLE_PLATEAU`（15–60 递减平台，10 日断崖）；周期 `CONDITIONAL`（震荡市是主要超额来源）。

## 8. 相关仓库

- 道氏 Regime 状态机（已冻结，独立项目）：[`d:\python_prog\dow-regime`](../dow-regime)。

本 README 只描述状态与口径，不构成采用建议，也不构成对 EP-02 策略线的终止裁决。
