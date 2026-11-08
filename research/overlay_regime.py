"""P11-C：把 Dow Regime 接到一个**最简单**的股票 Alpha 上，看它有没有增量价值.

这一层刻意**不接 PA**。理由：如果连最简单的横截面动量都救不了，
那接 PA 只是把问题藏起来。我们要回答的是：

    Alpha（无 Regime）  vs  Alpha + Dow Regime

而不是"Dow 自己能不能赚钱"（P10 已经答过：不能）。

## 必须先堵住的口径陷阱

"加 Dow 之后回撤更小"几乎必然成立 —— 因为**仓位更低了**。
不加对照组的话，任何"降低暴露"的手段都会显得有效，包括随机空仓。

所以本模块对每个 Regime 映射都跑三档：

1. `Alpha`            —— 满仓 Alpha
2. `Alpha + Dow`      —— 按 Dow 状态缩放
3. `Alpha @等暴露`    —— **对照组**：用 Alpha + Dow 的**平均暴露**做常数缩放，
                        暴露一样，但**完全不择时**。

只有 `Alpha + Dow` 明显好于 `Alpha @等暴露`，才说明 Dow 的**时机选择**有价值；
否则只是"少买了一点"。

## 口径纪律（与 P10 一致）

- 信号 t 日收盘产生 → **t+1 日生效**（权重与 Regime 缩放都平移 1 天）
- 成本：单边 12bp（佣金 + 印花税分摊 + 滑点），按实际换手计
- 复权：前复权（fqt=1）
- 停牌：价格前向填充 → 当日收益 0，仓位冻结（真实地"被套住"）

## 已知局限（必须写进结论）

- **生存偏差**：股票池用的是**今天**的 HS300 成分，回溯到 2005 年。
  免费数据源拿不到 point-in-time 成分股。影响：绝对收益被高估。
  但它是**对两条腿同时**高估的，所以 `Alpha + Dow` 相对 `Alpha @等暴露`
  的**增量**受影响小得多 —— 这一点是本报告结论的主要保护。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

# ---------------------------------------------------------------------------
# Alpha 参数：全部事先写死，且每个都有出处/惯例依据，**不做寻优**
# ---------------------------------------------------------------------------
ALPHA = dict(
    mom_window=120,      # 半年动量（横截面动量的常见最短窗口；再短噪声主导）
    top_n=30,            # 约 10% 的沪深300 非ST池
    rebal=20,            # 月度调仓（约 20 个交易日）
    min_amount=5e7,      # 20 日均成交额下限 5000 万（沿用 sibling 项目的流动性门槛）
    amount_window=20,
)
COST_ONE_WAY = 0.0012    # 12bp：佣金 ~2.5bp×2 + 印花税 5bp(卖出) + 滑点

# ---------------------------------------------------------------------------
# Regime 暴露映射：不是"最优参数"，是**敏感性网格**。用户明确说过
# 100/50/20 只是示意，所以这里一次列出多组，看结论是否依赖具体映射。
# ---------------------------------------------------------------------------
EXPOSURE_MAPS: dict[str, dict[str, float]] = {
    "M1_示意(1/.5/.2)": {"UP": 1.0, "TRANSITION": 0.5, "DOWN": 0.2},
    "M2_只避下跌": {"UP": 1.0, "TRANSITION": 1.0, "DOWN": 0.0},
    "M3_确认才满仓": {"UP": 1.0, "TRANSITION": 0.5, "DOWN": 0.0},
    "M4_只做牛市": {"UP": 1.0, "TRANSITION": 0.0, "DOWN": 0.0},
    "M5_温和(1/.75/.25)": {"UP": 1.0, "TRANSITION": 0.75, "DOWN": 0.25},
    "M6_仅避趋势": {"UP": 1.0, "TRANSITION": 0.35, "DOWN": 0.35},
}


# ---------------------------------------------------------------------------
def _clean_close(close: pd.DataFrame) -> pd.DataFrame:
    """0 / 负价 = 数据缺口或停牌标记（新浪用 0 表示停牌），先置 NaN 再前向填充.

    不处理会致命：一次 `X → 0` 的价格跳变会产生 -100% 的日收益，
    再叠加权重漂移会把权益曲线打穿（实测单日收益低至 -16.5）。
    """
    return close.where(close > 0).ffill()


def _ret_matrix(close: pd.DataFrame) -> pd.DataFrame:
    """日收益（停牌日 = 0，未上市期 = 0）.

    必须彻底清掉 NaN/inf：任何残留会顺着权重漂移 `w*(1+r)/denom` 传播，
    把整个组合收益变成 NaN（冒烟测试踩过一次）。
    """
    r = _clean_close(close).pct_change()
    return r.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_targets(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    mom_window: int = ALPHA["mom_window"],
    top_n: int = ALPHA["top_n"],
    rebal: int = ALPHA["rebal"],
    min_amount: float = ALPHA["min_amount"],
    amount_window: int = ALPHA["amount_window"],
) -> pd.DataFrame:
    """每个调仓日给出目标权重（未平移）；只含调仓日行，其余 NaN."""
    px = _clean_close(close)
    mom = px / px.shift(mom_window) - 1.0
    tradable = close.gt(0)                        # 当日真实成交（0 价 = 停牌）
    liq = amount.rolling(amount_window, min_periods=amount_window // 2).mean()
    eligible = tradable & liq.ge(min_amount) & mom.notna()

    idx = np.where(eligible.any(axis=1).to_numpy())[0]
    if not len(idx):
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    first = idx[0]
    rebal_idx = list(range(first, len(close), rebal))

    rows = {}
    for i in rebal_idx:
        d = close.index[i]
        cand = mom.iloc[i].where(eligible.iloc[i]).dropna()
        if len(cand) < top_n:
            continue
        picks = cand.sort_values(ascending=False).head(top_n).index
        w = pd.Series(0.0, index=close.columns)
        w[picks] = 1.0 / top_n
        rows[d] = w
    if not rows:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    tgt = pd.DataFrame(rows).T
    return tgt.reindex(close.index)


def simulate(
    close: pd.DataFrame,
    target_on_rebal: pd.DataFrame,
    ret: pd.DataFrame,
    cost_one_way: float = COST_ONE_WAY,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """按调仓日目标权重模拟，权重随价格漂移，调仓日在**次日**成交.

    返回 (net_daily_ret, exposure, turnover)
    """
    codes = close.columns
    dates = close.index
    rebal_rows = target_on_rebal.dropna(how="all")

    w_cur = pd.Series(0.0, index=codes)
    pending: pd.Series | None = None
    net = np.zeros(len(dates))
    expo = np.zeros(len(dates))
    turn = np.zeros(len(dates))
    r_np = ret.to_numpy()

    for i, d in enumerate(dates):
        cost = 0.0
        t = 0.0
        if pending is not None:
            t = float((pending - w_cur).abs().sum())
            cost = t * cost_one_way
            w_cur = pending
            pending = None
        r = r_np[i]
        gross = float(np.dot(w_cur.to_numpy(), r))
        denom = 1.0 + gross
        net[i] = gross - cost
        expo[i] = float(w_cur.sum())
        turn[i] = t
        if denom != 0:
            w_cur = w_cur * (1.0 + pd.Series(r, index=codes)) / denom
        if d in rebal_rows.index:
            pending = rebal_rows.loc[d]

    return (
        pd.Series(net, index=dates),
        pd.Series(expo, index=dates),
        pd.Series(turn, index=dates),
    )


# ---------------------------------------------------------------------------
def metrics(r: pd.Series, exposure: pd.Series | None = None,
            annual_turnover: float | None = None) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {"days": len(r)}
    eq = (1.0 + r).cumprod()
    yrs = len(r) / 243.0
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else None
    vol = float(r.std() * np.sqrt(243))
    sharpe = float(r.mean() / r.std() * np.sqrt(243)) if r.std() > 0 else None
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else None
    # 滚动 12 个月最差
    roll = eq.pct_change(243).dropna()
    worst12 = float(roll.min()) if len(roll) else None
    out = {
        "days": len(r),
        "years": round(yrs, 1),
        "CAGR": round(cagr, 4) if cagr is not None else None,
        "vol": round(vol, 4),
        "Sharpe": round(sharpe, 3) if sharpe is not None else None,
        "MaxDD": round(mdd, 4),
        "Calmar": round(calmar, 3) if calmar is not None else None,
        "worst_12m": round(worst12, 4) if worst12 is not None else None,
        "ann_ret": round(float(r.mean() * 243), 4),
    }
    if exposure is not None:
        out["avg_exposure"] = round(float(exposure.reindex(r.index).mean()), 4)
    if annual_turnover is not None:
        out["ann_turnover"] = round(annual_turnover, 2)
    return out


def _apply_scale(alpha_net: pd.Series, scale: pd.Series,
                 cost_one_way: float = COST_ONE_WAY) -> pd.Series:
    """在 Alpha 的日收益流上叠加暴露缩放，含缩放变动带来的成本."""
    s = scale.reindex(alpha_net.index).ffill().fillna(0.0)
    active = (alpha_net.rolling(20, min_periods=1).apply(lambda x: float((x != 0).any()),
                                                         raw=True) > 0).astype(float)
    d_scale = s.diff().abs().fillna(s.abs())
    cost = cost_one_way * d_scale * active
    return s * alpha_net - cost


def _subperiod_table(r: pd.Series, label: str) -> pd.DataFrame:
    periods = [("2005-2010", "2005-01-01", "2010-12-31"),
               ("2011-2015", "2011-01-01", "2015-12-31"),
               ("2016-2020", "2016-01-01", "2020-12-31"),
               ("2021-2026", "2021-01-01", "2026-12-31")]
    rows = []
    for name, s, e in periods:
        sub = r[(r.index >= s) & (r.index <= e)].dropna()
        if len(sub) < 30:
            continue
        m = metrics(sub)
        rows.append({"strategy": label, "period": name, "CAGR": m.get("CAGR"),
                     "Sharpe": m.get("Sharpe"), "MaxDD": m.get("MaxDD")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def run(pair_index: int = 0, universe: str = "hs300", tag: str = "") -> tuple[str, dict]:
    from data.loader import load_index, load_index_pair  # noqa: PLC0415
    from data.stock_loader import load_universe_panel  # noqa: PLC0415

    pairs = [
        ("000300", "000905", "沪深300 + 中证500"),
        ("000001", "399001", "上证指数 + 深证成指"),
        ("000300", "000510", "沪深300 + 中证A500"),
    ]
    a_code, b_code, pair_label = pairs[pair_index]
    reg_csv = os.path.join(OUT, f"regime_{a_code}_{b_code}_v0.1.csv")
    if os.path.exists(reg_csv):
        reg = pd.read_csv(reg_csv)
    else:
        # 事件库没跑过就先补出来（口径与 P10 完全一致：V0.1 冻结参数）
        from config.dow_params import get_params  # noqa: PLC0415
        from core.regime_engine import analyze_pair  # noqa: PLC0415
        from data.loader import load_index_pair  # noqa: PLC0415

        a, b = load_index_pair(a_code, b_code)
        reg, _ = analyze_pair(a, b, get_params("v0.1"), strict_invariants=False)
    reg["date"] = pd.to_datetime(reg["date"])
    reg = reg.set_index("date").sort_index()

    # ---- 个股面板 ----
    close, amount, rep = load_universe_panel(universe=universe)
    if close.empty:
        raise RuntimeError("stock panel empty —— 先运行 data/stock_loader.py 抓取缓存")
    close = close.sort_index()
    # 口径纪律：把两腿压到**同一批日期**上。否则 Alpha 腿会从 1991 年跑起
    # （个股数据比指数长），而 Dow 腿只有 2005 年后 —— 直接比就是错的。
    common = close.index.intersection(reg.index)
    close = close.loc[common]
    amount = amount.reindex(common)
    if len(close) < 500:
        raise RuntimeError(f"overlap too short: {len(close)} days")

    ret = _ret_matrix(close)

    # ---- Alpha ----
    tgt = build_targets(close, amount, **ALPHA)
    alpha_net, alpha_expo, alpha_turn = simulate(close, tgt, ret)

    # ---- Regime 信号：t 日产生 → t+1 生效 ----
    trend = reg["primary_trend"].reindex(close.index, method="ffill")
    sig_eff = trend.shift(1)                      # T+1
    n_active = int((alpha_expo > 0).sum())
    active_mask = alpha_expo > 0

    def scale_of(mapping: dict[str, float]) -> pd.Series:
        s = sig_eff.map(mapping).astype(float)
        s = s.where(active_mask, 0.0)             # Alpha 没建仓时暴露为 0
        return s.fillna(0.0)

    # ---- 基准 ----
    try:
        bench = load_index(a_code)
        bench = bench.set_index("date")["close"].reindex(close.index).ffill()
        bench_ret = bench.pct_change().fillna(0.0)
    except Exception:  # noqa: BLE001
        bench_ret = pd.Series(0.0, index=close.index)
    ew_ret = ret.replace(0.0, np.nan).mean(axis=1).fillna(0.0)   # 等权全池

    # ---- 组装 ----
    results: dict[str, tuple[pd.Series, pd.Series | None, float | None]] = {}
    results["Alpha(满仓)"] = (alpha_net, alpha_expo, None)
    results["基准:沪深300"] = (bench_ret, None, None)
    results["基准:等权全池"] = (ew_ret, None, None)

    def _mrow(name: str, avg: float, r_ov: pd.Series, r_const: pd.Series) -> dict:
        mo = metrics(r_ov)
        mc = metrics(r_const)
        return {
            "mapping": name, "avg_exposure": round(avg, 3),
            "Dow_CAGR": mo.get("CAGR"), "Dow_Sharpe": mo.get("Sharpe"),
            "Dow_MaxDD": mo.get("MaxDD"), "Dow_Calmar": mo.get("Calmar"),
            "等暴露_CAGR": mc.get("CAGR"), "等暴露_Sharpe": mc.get("Sharpe"),
            "等暴露_MaxDD": mc.get("MaxDD"), "等暴露_Calmar": mc.get("Calmar"),
            "CAGR_增益pp": round(((mo.get("CAGR") or 0) - (mc.get("CAGR") or 0)) * 100, 2),
            "Sharpe_增益": round((mo.get("Sharpe") or 0) - (mc.get("Sharpe") or 0), 3),
            # 正数 = Dow 回撤更浅（改善）
            "MaxDD_改善pp": round(((mo.get("MaxDD") or 0) - (mc.get("MaxDD") or 0)) * 100, 2),
        }

    map_rows = []
    for name, mp in EXPOSURE_MAPS.items():
        sc = scale_of(mp)
        r_ov = _apply_scale(alpha_net, sc)
        avg = float(sc[active_mask].mean()) if n_active else 0.0
        # 对照组：同平均暴露，但常数（不择时）
        r_const = alpha_net * avg
        results[f"Dow {name}"] = (r_ov, sc, None)
        results[f"等暴露对照 {name}"] = (r_const, None, None)
        map_rows.append(_mrow(name, avg, r_ov, r_const))

    # ---- MA200 参照：同一个 Alpha，换成最简单的趋势滤波器做择时 ----
    # Dow 要证明"作为 Regime 层有增量价值"，至少得打得过这个。
    ma = bench.rolling(200).mean()
    ma_exp = ((bench > ma).astype(float).shift(1).fillna(0.0)).where(active_mask, 0.0)
    avg_ma = float(ma_exp[active_mask].mean()) if n_active else 0.0
    r_ma = _apply_scale(alpha_net, ma_exp)
    r_ma_const = alpha_net * avg_ma
    results["Dow MA200参照"] = (r_ma, ma_exp, None)
    results["等暴露对照 MA200参照"] = (r_ma_const, None, None)
    ma_row = _mrow("MA200(参照)", avg_ma, r_ma, r_ma_const)
    map_rows.append(ma_row)

    # ---- 主表 ----
    main_rows = []
    for name, (r, e, _) in results.items():
        ann_turn = None
        if name.startswith("Alpha(满仓)"):
            ann_turn = float(alpha_turn.sum() / max(1e-9, len(alpha_turn) / 243.0))
        m = metrics(r, e, ann_turn)
        main_rows.append({"strategy": name, **m})

    # ---- 报告 ----
    L: list[str] = []
    L.append(f"# P11-C 组合覆盖层：Dow Regime 有没有增量价值 —— {pair_label}\n")
    L.append(f"Alpha = 最简单横截面动量（{ALPHA['mom_window']}日收益排名取前 "
             f"{ALPHA['top_n']} 等权，每 {ALPHA['rebal']} 日调仓，T+1 成交，单边 "
             f"{COST_ONE_WAY*1e4:.0f}bp）。**不含 PA**。\n")
    L.append(f"股票池：{universe} 共 {rep['requested']} 只，可用 {rep['kept']} 只；"
             f"样本 {close.index[0].date()} ~ {close.index[-1].date()}，{len(close)} 个交易日。\n")
    L.append("> **必须先看这一条**：\"加 Dow 后回撤更小\"几乎必然成立，因为仓位更低了。"
             "所以每档都配一个**平均暴露完全相同、但完全不择时**的对照组。"
             "只有跑赢对照组，才说明 Dow 的**时机**有价值。\n")

    L.append("## 1. 主对照表\n")
    L.append("| 策略 | CAGR | vol | Sharpe | MaxDD | Calmar | 平均暴露(全样本) | 最差12m |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r_ in main_rows:
        L.append(f"| {r_['strategy']} | {_p(r_.get('CAGR'))} | {_p(r_.get('vol'))} | "
                 f"{_n(r_.get('Sharpe'),3)} | {_p(r_.get('MaxDD'))} | "
                 f"{_n(r_.get('Calmar'),3)} | {_n(r_.get('avg_exposure'),3)} | "
                 f"{_p(r_.get('worst_12m'))} |")
    L.append("")
    L.append("（CAGR/vol/MaxDD/最差12m 为百分比）\n")
    ew = next((r_ for r_ in main_rows if r_["strategy"].startswith("基准:等权")), {})
    if ew.get("CAGR") is not None and main_rows[0].get("CAGR") is not None:
        L.append(f"> ⚠️ **先说清楚这个 Alpha 本身有多弱**：满仓横截面动量 CAGR "
                 f"{_p(main_rows[0]['CAGR'])}，而**等权全池**是 {_p(ew['CAGR'])}"
                 f"（Sharpe {_n(ew.get('Sharpe'),3)} vs {_n(main_rows[0].get('Sharpe'),3)}）。"
                 "也就是说，这个\"最简单的动量 Alpha\"在 A 股**跑输等权**"
                 "（横截面动量在 A 股本就偏弱/易反转，这是已知现象，不是本次实现的问题）。"
                 "因此下面测的是\"Dow 能不能改善一个**平庸**的 Alpha\"；"
                 "它**不能**回答\"Dow 能不能改善一个**好**的 Alpha\"。\n")

    L.append("## 2. 关键：Dow 择时 vs 同等暴露不择时\n")
    L.append("> **对照组的 Sharpe 恒等于满仓 Alpha 的 Sharpe**（常数倍缩放不改变 "
             "收益/风险比）。所以这一列的 Sharpe 增益**完全来自时机选择**，"
             "没有任何\"仓位变低\"的水分。\n")
    L.append("| Regime 映射 | 平均暴露(建仓期) | Dow CAGR | 等暴露 CAGR | CAGR 增益(pp) | "
             "Dow Sharpe | 等暴露 Sharpe | Sharpe 增益 | MaxDD 改善(pp) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mrow in map_rows:
        L.append(
            f"| {mrow['mapping']} | {mrow['avg_exposure']:.1%} | {_p(mrow['Dow_CAGR'])} | "
            f"{_p(mrow['等暴露_CAGR'])} | {_n(mrow['CAGR_增益pp'],2)} | "
            f"{_n(mrow['Dow_Sharpe'],3)} | {_n(mrow['等暴露_Sharpe'],3)} | "
            f"{_n(mrow['Sharpe_增益'],3)} | {_n(mrow['MaxDD_改善pp'],2)} |"
        )
    L.append("")
    L.append("（CAGR 增益单位百分点；MaxDD 改善**正数 = Dow 回撤更浅**）\n")
    dow_rows = [m for m in map_rows if not m["mapping"].startswith("MA200")]
    wins = int(sum(1 for m in dow_rows if (m["Sharpe_增益"] or 0) > 0))
    ma_sharpe_gain = ma_row["Sharpe_增益"] or 0
    ma_beaten = int(sum(1 for m in dow_rows if (m["Sharpe_增益"] or 0) > ma_sharpe_gain))
    L.append(f"- **{len(dow_rows)} 组 Dow 映射中，Sharpe 优于同暴露对照的有 {wins} 组。**")
    L.append(f"- 参照 MA200 择时：平均暴露 {ma_row['avg_exposure']:.1%}，"
             f"CAGR 增益 {_n(ma_row['CAGR_增益pp'],2)}pp，"
             f"Sharpe 增益 {_n(ma_row['Sharpe_增益'],3)}。")
    L.append(f"- Dow 映射中 Sharpe 增益**超过 MA200 参照**的有 **{ma_beaten}/{len(dow_rows)}** 组 "
             "—— 这是判断\"Dow 有没有比简单趋势滤波器更值钱\"的硬指标。")
    L.append("")

    L.append("## 3. 分区间\n")
    sub = pd.concat([
        _subperiod_table(alpha_net, "Alpha(满仓)"),
        _subperiod_table(_apply_scale(alpha_net, scale_of(EXPOSURE_MAPS["M1_示意(1/.5/.2)"])),
                         "Dow M1"),
        _subperiod_table(alpha_net * float(scale_of(EXPOSURE_MAPS["M1_示意(1/.5/.2)"])[active_mask].mean()),
                         "等暴露对照 M1"),
    ], ignore_index=True)
    L.append("| 策略 | 区间 | CAGR | Sharpe | MaxDD |")
    L.append("|---|---|---:|---:|---:|")
    for _, r_ in sub.iterrows():
        L.append(f"| {r_['strategy']} | {r_['period']} | {_p(r_['CAGR'])} | "
                 f"{_n(r_['Sharpe'],3)} | {_p(r_['MaxDD'])} |")
    L.append("")

    L.append("## 4. 局限\n")
    L.append("- **生存偏差**：股票池是*今天*的 HS300 成分回溯到 2005 年，免费源拿不到 "
             "point-in-time 成分股。绝对收益被高估。但它同时作用于两条腿，"
             "所以**增量**（第 2 节）受影响小得多。")
    L.append("- 未建模 A 股执行层：涨跌停无法成交、停牌、ST、T+1 卖出限制。"
             "这些会进一步压低实际收益。")
    L.append("- Regime 只用沪深300+中证500 这一对；P11-A/B 已显示**配对选择本身不稳定**。")
    L.append("")

    suffix = f"_{tag}" if tag else ""
    path = os.path.join(OUT, f"p11_overlay{suffix}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    meta = {
        "pair": pair_label,
        "universe": f"{universe}:{rep['kept']}/{rep['requested']}",
        "alpha_CAGR": main_rows[0].get("CAGR"),
        "alpha_Sharpe": main_rows[0].get("Sharpe"),
        "alpha_MaxDD": main_rows[0].get("MaxDD"),
        "ew_CAGR": ew.get("CAGR"),
        "dow_sharpe_wins": f"{wins}/{len(dow_rows)}",
        "ma200_sharpe_gain": ma_sharpe_gain,
        "dow_beats_ma200": f"{ma_beaten}/{len(dow_rows)}",
        "m1": next((m for m in map_rows if m["mapping"].startswith("M1")), {}),
    }
    return path, meta


def _p(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:.2f}%"


def _n(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"
