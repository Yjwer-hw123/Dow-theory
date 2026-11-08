# dow-theory 项目长期记忆

**项目**：`D:/python_prog/dow-theory` —— Quantitative Dow Theory Engine（A 股版道氏理论量化引擎）
**版本**：Dow-Literal-V0.1 —— **已冻结**（2026-09-25）
**定位**：整个 A 股交易系统的 **Market Regime Engine 候选**，不是交易系统本身。
真正的收益在 Alpha 层（P12 起），本层只负责"大周期环境"。

---

## 1. 冻结边界（不可违反）

- **数值参数冻结**：`config/dow_params.py::DOW_LITERAL_V01`，每个值必须能在原书中找到出处。
- **交易规则冻结**。冻结期内只接受两类改动：
  ① 让状态机**形式上更自洽**（语义拆分 / 不变量 / 防前瞻 / 结构定义一致化）；
  ② 修复逻辑错误。
- **禁止为改善收益 / Sharpe / 年化而改任何参数或语义。**
  判准：*这个改动能让"市场照原样重来一次时策略更赚钱"吗？* 会 → 拒绝。
- **信号 t 日收盘产生、t+1 生效。** 回测口径统一冻结在单独的执行层。
- **无前瞻必须可回归验证**：`tests/run_tests.py::test_engine_prefix_invariance`
  （5 配对 × 61 点，当前 305/305 一致）。
- **比较策略必须同时看**"持仓期年化"和"同平均暴露对照组"，
  否则持仓占比差异会让结论无效（常数缩放不改变 Sharpe，这正是对照组的意义）。

## 2. 已确认的结论（负面结果同样是产出）

- 道氏结构**可以**机械化、跨指数稳定提取（4~5 组配对双指数确认率 90%~95%）。
- 状态**有**信息含量：RISK_ON 后 fwd60 +8.64% vs WAIT +2.94%（pair0 全样本）。
- **但打不过 MA200**：Dow 引擎 CAGR 4.17% / Sharpe 0.534 vs MA200 11.12% / 0.682。
- **组合层（P11-C）**：6 组映射中 5 组 Sharpe 优于同暴露对照（0.379 → 0.400~0.589），
  说明"择时有信息、不是仓位变低的假象"；**但只有 1/6 组超过 MA200 参照，换配对 0/6。**
- **信息含量高度集中在 2005-2010**：ON−WAIT 从 +9.6pp 衰减到 2016-2020 的 −1.7pp；
  OFF−WAIT 从 −21.3pp 衰减到 −0.9pp（避风价值基本消失）。
- **12% 低持仓不是实现 bug**：85.8% 空仓来自 TRANSITION + TREND_DOWN（理论核心设计）。
- **消融 A→F 单调递减** → 闸门越多不代表越好，纯 HH/HL 反而最好。
- **TRANSITION 子状态收益跨配对完全翻转**（T4：−1.15% ↔ +11.56%）
  → **不得把任何子状态写成 RISK_ON。**
- `integrity == "INTACT"` 在信号判定中是**冗余**的（UP 趋势下 DAMAGED ⟺ INVALIDATED，INV-02）。

### STEP 1：月 MA20 + 周 MA60 多周期趋势过滤（2026-11-06，NO-GO · 已冻结）

- **裁定**：NO-GO。不进入 STEP 2，不再优化 MA 参数，未授权不得重启该假设。
- **范围**：PIT 沪深 A 股 5,555 只，2005-01-01 至 2026-09-30，每月最后一个已完成交易日
  为观察时点，未来 5/20/60/120 个交易日，次一交易日开盘入场。
  分组：A_all（全池）/ A_m（月线可评估）/ B（月 MA20 上行 + 收盘在 MA20 上方）/
  A_w、B_w / C（B + 周 MA60 上行 + 收盘在 MA60 上方）。
- **四个假设全部 REJECTED**：
  ① B 四个周期均值/中位数/正收益比例全面低于 A_m；
  ② C 相对 B_w 无显著增量（t 全部不显著）；
  ③ 熊市 B 120 日 −7.34% vs A_m +2.75%，趋势过滤穿越熊市失败；
  ④ B 相对 A_m 为正的年份仅少数（120 日 8/22），跨年份不稳定。
- **B 只保留 A_m 约 38% 的观察样本**，却没有换来更好收益 → 既没提高质量，又压缩机会。
- **月度配对差 B−A_m**：−0.141% / −0.430% / −0.611% / −1.190%（5/20/60/120 日），
  t −1.77 / −2.31 / −1.90 / −1.98。
- **收益更集中**：120 日正收益前 10% 股票贡献 B 41.10% / C 47.60% vs A_m 31.11%。
- **没有证明**：所有趋势交易无效 / 强势股无持续性 / 板块趋势无价值 / 个股相对强度无价值 /
  日线回调买入无效。否定对象仅限“月 MA20 上行 + 周 MA60 上行能提高 A 股未来收益”。
- **产物**：`out/step1/STEP1_FINAL_REPORT.md` + 8 个结果文件，已由 `out/step1/FREEZE.sha256`
  冻结（8 个 SHA256），冻结后不得重算或换参数重跑。台账见 `notes/2026-09-25_结论纪要.md` 第四节。
- **下一阶段原则**：任何交易想法先验证能否产生**超越简单市场基准**的收益，再决定是否开发；
  不自动开启新项目。

## 3. 数据源经验（踩过的坑）

- 指数：新浪 `money.finance.sina.com.cn`（主，稳定）→ 腾讯 → 东财（本机易被断连）。
- 个股：**mootdx 已完全不可用**（任何参数组合都返 0 行）；网易 `quotes.money.163.com` 502；
  东财 `push2his` 单次 0.9s 但连续几十次后 `RemoteDisconnected`（只作兜底）。
  最终主源 = **新浪 `CN_MarketData.getKLineData` + `datalen=6000`**
  （上限不是 1023，那只是默认值；实测回溯 2001-08；6 线程抓 280 只约 5 分钟，0 失败）。
- **个股符号映射与指数不同**（静默错误）：个股 `000001` = 平安银行 = 深市 `sz`；
  指数 `000001` = 上证指数 = 沪市。别复用同一个映射函数。
- **`close == 0` 不是价格**：新浪用 0 标记停牌，不处理会产生单日 −1650% 的假收益，
  再顺着权重漂移 `w*(1+r)/denom` 把权益曲线打穿。
- **两腿时间窗必须相同**：个股数据可回溯到 1991 年，必须裁剪到与指数一致。

## 4. 环境与运行

- 解释器：`C:/Users/16079/AppData/Local/Programs/Python/Python311/python.exe`
  （pandas / numpy / mootdx 已装）。
- 入口：`tests/run_tests.py`（15 用例）、`scripts/run_regime.py N`、`run_p10.py`、
  `run_p10_1.py`、`run_p11ab.py`、`run_p11c.py --pair N`、`audit_lookahead.py --all`。
- 缓存已就绪，**不必重抓**：`cache/*.parquet`（6 指数）、`cache/stocks/*.parquet`（HS300 非ST 280 只）。

## 5. 原书资料（已迁入项目内，自包含）

**本项目不依赖任何 sibling 工作空间。** 2026-09-25 已把资料复制进项目（md5 逐文件校验一致）：

- `docs/道氏理论+珍藏版+(高清).pdf`（8.2 MB，183 页扫描版原书）
- `docs/道氏理论_精读笔记.md`（19 章分章要点 + 定理速查表 + 35 年次级反应统计）
- `docs/dow_full.txt`（183 页 OCR 合并全文，9.2 万字，`===== PAGE N =====` 分页标记；
  N 与书中页码差约 8 页）
- `data/universe/*.json`（股票池 hs300 / zz500 / wide，原在 stock-analysis-skill）

`data/stock_loader.py::UNIVERSE_DIR` 已改为项目内相对路径。
唯一外部依赖是行情数据源（新浪 / 东财）。

> 原副本仍留在 `D:/python_prog/stock-analysis-skill/`（未删）。两边若再修改会产生漂移，
> **以本项目 `docs/` 下的为准**。

## 6. 下一步（P12 Alpha Discovery）

真正决定收益的是 **Alpha 层**：本次最简横截面动量在 A 股**跑输等权**
（7.19% vs 16.98%，Sharpe 0.379 vs 0.721）→ 说明不值得再往 Regime 上加模块。

P12 流程约定：
`Hypothesis → Feature → IC → Quantile Return → Decay → Regime Conditional → Cost → Walk Forward → Blind`
（PA 只是候选 Alpha 之一，若 IC≈0 / 样本外失效，就删掉，不预设结论。）

## 7. P12 主线规划（2026-09-25 定稿，存档于 notes/P12_alpha_discovery_plan.md）

- 主线：`Dow Regime（V0.1 冻结）` 与 `Alpha Discovery` 两条独立线路，最后才在 Portfolio 层合流。
- **P12-0 第一优先**：修 Universe 数据地基（point-in-time 成分/ST/停牌/上市退市/复权/可交易性），
  消除 P11-C 的生存者偏差嫌疑；在此之前一切结果只能标 `Survivor-conditioned research`。
- 因子首批：Momentum（含剔除近5/20日）、Relative Strength（vs Market/Sector）、Volume-Price、
  MoneyFlow（含标准化）、Fundamental rank、PA-as-factor。产出物 = 因子总表，不是策略收益。
- 纪律：Alpha 阶段不看 Dow 结果；PA 无增量即淘汰；第一阶段不碰 PA/Dow/LLM。
