# P12 = A 股 Alpha Discovery（项目主线规划）

> 存档时间：2026-09-25 23:05
> 来源：用户 2026-09-25 会话中确认的项目主线规划（P11 收尾后制定）
> 前置背景：P11 已完成，"继续优化 Dow 自己"这条路基本关掉；Dow V0.1 冻结

这次 P11 做完以后，项目的主线已经非常清楚了，而且**不要再往 Dow 里加东西**。

现在正式切换成：

# P12 = A 股 Alpha Discovery

规则钉死：

> **Alpha 先脱离 Dow 独立发现；Dow V0.1 冻结，暂时不参与因子筛选。**
>
> 先找"什么东西在 A 股本身能赚钱"，再问"Dow 有没有能力改善它"。

这样才能把两个问题分开。

---

# 项目两条独立线路

```text
                 A股量化系统
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Market Regime                Alpha
       Dow                    P12 Discovery
          │                       │
          │              ┌────────┼────────┐
          │              │        │        │
          │            动量      量价     相对强弱
          │              │        │        │
          │              │     资金流    基本面
          │              │        │        │
          │              └────────┼────────┘
          │                       │
          └──────────────┬────────┘
                         ▼
                    Portfolio
```

现在还**不要**把 PA 塞进去。PA 也和动量一样，只是 `Alpha Candidate`，不是默认答案。

---

# P12 开始前必须先修正：Survivorship Bias（生存者偏差）

P11-C 的：

```text
HS300 非ST 280只
```

如果这个 280 是**当前时点的沪深300成分股向历史回看**，那么 P12 不能直接沿用。

因为这会产生：

> **Survivorship Bias（生存者偏差）**

今天还活着、还在沪深300里的公司，被拿去回测 2010 年；而那些后来退市、跌出指数、被淘汰的股票没有进入历史样本。那么：

```text
历史 Alpha
```

会天然被抬高。

尤其现在已经看到：

```text
等权    16.98%
简单动量 7.19%
```

这个差距很大。第一件事不是研究"为什么动量这么差"，而是先确认：

> **16.98% 的等权基准到底是不是 point-in-time universe。**

否则 P12 很可能从数据层开始就污染。

---

# P12-0 先做"研究地基"

不是因子，而是：

```text
Universe
Membership
ST status
停牌
上市日期
退市日期
复权方式
价格可交易性
行业归属
数据发布日期
```

尤其是：**每一天到底哪些股票"当时真的可以进入研究池"？**

这是 P12 最重要的第一条。

如果暂时没有历史成分股，就不要把"当前 280 只股票 × 2005-2026"叫做严格的历史 Alpha 回测。可以研究，但必须明确标记为：

```text
Survivor-conditioned research
```

不能拿它作为最终样本外证据。

---

# P12-1 单因子实验室

每天：

```text
T 日
│
├── 计算股票特征
├── 按特征排序
├── 分成 5/10 组
└── 看 T+1 / T+5 / T+20 / T+60
```

得到：

```text
Factor → Rank → Forward Return → IC → Decay
```

## 第一批候选因子（全部平等测试，不预设哪个好）

### Momentum
```text
mom20 / mom60 / mom120 / mom250
```
必须做剔除最近 5/20 日的变体，看是不是短期反转污染了长期动量。

### Relative Strength（值得重点测试）
```text
Stock Return − Market Return
Stock Return − Sector Return
```
A 股很多时候真正重要的是：**这只股票是不是正在脱离它所在的市场/行业。**

### Volume / Price
不是传统 PA 的"放量突破=买"，而是纯因子：
```text
volume_change / turnover_change / price_efficiency
volume_surprise / up_volume_ratio / down_volume_ratio
```

### Money Flow
现有数据层已能拿主力/超大单/大单/中单/小单：
```text
moneyflow_5 / moneyflow_20 / moneyflow_60
moneyflow / turnover   （标准化形式）
```

### Fundamental
横截面 rank，不是"ROE 高→买"：
```text
ROE rank / Gross margin rank / Revenue growth rank
Debt rank / Earnings quality
```

### Technical Structure（这里才轮到 PA）
不是 `Hammer → BUY`，而是测试 `Hammer = factor`，再看：
```text
Hammer + market position
Hammer + volume
Hammer + relative strength
```
看有没有稳定增量。**如果没有：直接淘汰。** 这也是想从 PA 项目得到的答案。

---

# P12-2 先看 IC

每天 `IC_t`，长期看：

```text
mean(IC) / std(IC) / ICIR
monthly IC / yearly IC
bull IC / bear IC / transition IC
```

最重要的一点：

> **不要因为平均 IC 很好看就保留。**

例如 `IC = 0.04` 但 `2010-2015 +0.08 / 2016-2020 -0.01 / 2021-2026 +0.02`——这种不算稳定 Alpha。与 Dow P11-A/B 同一套思想。

---

# P12-3 分组收益

```text
        Q1      Q2      Q3      Q4      Q5
Factor ↓
Return
```

真正想看到的是**单调性**（Q1<Q2<Q3<Q4<Q5，或反向），而不是复杂关系。

---

# P12-4 衰减

```text
Signal → 5D → 20D → 60D → 120D
```

可能发现 `20D 最强 / 60D 还在 / 120D 消失`——由此才知道这个 Alpha 适合短线、中线还是长线，而不是拍脑袋。

---

# P12-5 加入成本

```text
Gross Alpha → 换手 → 滑点 → 手续费
→ 涨跌停不可成交 → T+1 → Net Alpha
```

毛收益 +18% 但换手 20 倍、成本 −15% 的因子是垃圾 Alpha。

---

# P12-6 最后才研究 Dow × Alpha

终局实验：

```text
Alpha A
Alpha A + MA200
Alpha A + Dow
```

比较：

```text
CAGR / MaxDD / Sharpe / Calmar / Tail loss / Exposure / Turnover
```

回答：**Dow 到底有没有提供 MA200 之外的增量信息。**

---

# 研究纪律

## Alpha Discovery 阶段不能知道 Dow 的结果

P12 先做 Alpha 单独研究。不要问"Dow Bull 时哪个因子最好"然后直接调——容易把两个层级相互拟合。

正确顺序：

```text
P12 Alpha → 冻结 Alpha → Dow → 组合
```

## P12 第一批结果不是"策略收益"，而是一张因子总表

| Factor    | Mean IC | ICIR | Q5-Q1 | Decay | Turnover | Net Return | Stable Years |
| --------- | ------: | ---: | ----: | ----: | -------: | ---------: | -----------: |
| MOM20     |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| MOM60     |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| MOM120    |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| RS20      |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| RS60      |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| Volume    |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| MoneyFlow |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| ROE       |       ? |    ? |     ? |     ? |        ? |          ? |            ? |
| PA        |       ? |    ? |     ? |     ? |        ? |          ? |            ? |

这张表出来之后，才会第一次真正知道"A股里哪些东西有 alpha"。

---

# 一句话钉死主线

现在不是"我要把道氏理论做成一个交易策略"，也不是"我要做一个 Dow + PA 策略"，而是：

> **先建立可靠的 A 股市场状态模型，再从 A 股市场本身寻找经过样本外验证的 Alpha，最后测试 Regime 对 Alpha 的增量价值。**

```text
                Market
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   Dow Regime           Alpha Discovery
   "什么时候做"          "做什么赚钱"
        │                   │
        └─────────┬─────────┘
                  ▼
             Portfolio
                  ▼
             Walk Forward
                  ▼
             Blind Test
```

P12 第一阶段**完全不碰 PA、不碰 Dow、不碰 LLM**，先用最朴素的横截面因子实验把 A 股的"赚钱来源"摸出来。

**先把"什么东西真的有 Alpha"搞清楚，再谈怎么把 Dow 放进去。**
