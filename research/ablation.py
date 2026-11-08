"""P10-B 消融实验：把道氏拆成可开关的闸门，逐个加回去，看贡献.

**为什么必须做**
只看"完整 Dow 的年化"无法回答一个更重要的问题：

    到底是哪个组件在贡献？还是全都靠 buy&hold 的 beta？

所以从最松的规则开始，一层层加闸门：

    A  HH/HL        trend==UP 且结构完好（integrity==INTACT）
    B  + dual       再加双指数确认
    C  + secondary  再加次级反应闸门（OVERDUE / INVALIDATED 时退出）
    D  + line       再加线态向下突破的否决
    E  + volume     再加成交量过热（EXHAUSTION）的否决
    F  full Dow     = E（当前最严）

**注意一个关键事实**：当前 V0.1 引擎实际发出的 `dow_signal` 恰好等于 **C**。
D / E 是本实验提出的、V0.1 里尚未启用的扩展闸门 —— 所以本表同时也是
"V0.1 之外还能不能再加点东西"的证据。

判断标准不是只有 CAGR。一个模块即使不提高收益，也可能：
    - 降低最大回撤（避免错误的市场状态）
    - 降低换手 / 提高时间等待占比（少踩坑）

所以输出同时保留 maxDD / 换手 / 待机时间。
"""

from __future__ import annotations

import pandas as pd

from .perf import backtest, metrics


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)

# 消融阶梯：键=变体名，值=启用的模块集合（顺序即"加回去"的顺序）
#
# structure_source 说明：
#   "index_a"  只用指数 A 自己的浪构建 HH/HL —— 真正的"只有 HH/HL"（A 变体）
#   "dual"     用双指数确认过的浪构建结构      —— B 及以后
LADDER: list[tuple[str, frozenset, str]] = [
    ("A_HHHL", frozenset({"hh_hl"}), "index_a"),
    ("B_dual", frozenset({"hh_hl", "dual"}), "dual"),
    ("C_secondary", frozenset({"hh_hl", "dual", "secondary"}), "dual"),
    ("D_line", frozenset({"hh_hl", "dual", "secondary", "line"}), "dual"),
    ("E_volume", frozenset({"hh_hl", "dual", "secondary", "line", "volume"}), "dual"),
    ("F_full", frozenset({"hh_hl", "dual", "secondary", "line", "volume"}), "dual"),
]


def gate_positions(df: pd.DataFrame, modules: frozenset) -> pd.Series:
    """按启用的模块，把每日状态转成目标仓位（次日生效在 backtest 里做）.

    闸门是**与**关系：任一条不过关，当日就不持有。
    """
    d = _sort(df)

    trend = d["primary_trend"]
    ok = pd.Series(True, index=d.index)

    # A: HH/HL —— 只做多上升趋势，且结构必须完好
    ok &= (trend == "UP") & (d["trend_integrity"] == "INTACT")

    # B: 双指数确认
    if "dual" in modules:
        ok &= d["dual_confirmation"] == "CONFIRMED"

    # C: 次级反应闸门（超期 / 已失效都退出）
    if "secondary" in modules:
        ok &= ~d["secondary_reaction"].isin(["OVERDUE", "INVALIDATED"])

    # D: 线态向下突破直接否决
    if "line" in modules:
        ok &= d["line_breakout"] != "DOWN_CONFIRMED"

    # E: 成交量过热否决（第 15 章：牛市以过度活跃收场）
    if "volume" in modules:
        ok &= d["volume_state"] != "EXHAUSTION"

    return ok.astype(float)


def run_ablation(engine_dual: pd.DataFrame,
                 engine_index_a: pd.DataFrame | None = None,
                 fee_bps: float = 5.0) -> pd.DataFrame:
    """跑消融阶梯.

    Parameters
    ----------
    engine_dual : `structure_source="dual"` 的引擎输出（V0.1 默认）
    engine_index_a : `structure_source="index_a"` 的引擎输出，供 A 变体使用。
        若为 None，A 变体退化为用 dual 引擎 —— 此时**它不是真正的"只有 HH/HL"**，
        报告里会明确标注（这是防止误读的关键）。
    """
    frames = {"dual": _sort(engine_dual)}
    frames["index_a"] = _sort(engine_index_a) if engine_index_a is not None else frames["dual"]
    a_is_true_single = engine_index_a is not None

    rows: list[dict] = []
    for name, mods, src in LADDER:
        d = frames[src]
        close = d["close_a"].astype(float)
        dates = d["date"]
        # 与覆盖层 / 基准统一口径：t 日收盘产生的信号，t+1 日才生效。
        # 不平移等于用当天收盘信号去吃当天收益 —— 那是前瞻偏差。
        pos = gate_positions(d, mods).shift(1).fillna(0.0)
        bt = backtest(close, pos, dates=dates, fee_bps=fee_bps)
        row = metrics(bt, name, fee_bps=fee_bps)
        row["structure_source"] = src if a_is_true_single else (
            "dual(A 退化)" if name == "A_HHHL" else src)
        rows.append(row)

    out = pd.DataFrame(rows)
    # 边际贡献：相对上一级的增量
    out["d_cagr"] = out["cagr"].diff().round(4)
    out["d_maxdd"] = out["max_drawdown"].diff().round(4)
    return out


def verify_engine_matches_ladder(engine_df: pd.DataFrame) -> dict:
    """自检：引擎自己发出的 dow_signal 应当**恰好等于阶梯中的某一个变体**.

    这是消融表可信的前提。不要硬编码"引擎 = C" —— 引擎里还有一条
    "线态向下突破否决 RISK_ON" 的规则，所以它其实等于 **D**。
    这里逐个比对，把答案算出来而不是猜出来。
    """
    d = _sort(engine_df)
    engine_on = (d["dow_signal"] == "RISK_ON").astype(float)
    mismatches = {}
    for name, mods, _src in LADDER:
        pos = gate_positions(d, mods)
        mismatches[name] = int((pos != engine_on).sum())

    best = min(mismatches, key=lambda k: mismatches[k])
    match = best if mismatches[best] == 0 else None
    return {
        "engine_risk_on_days": int(engine_on.sum()),
        "engine_risk_off_days": int((d["dow_signal"] == "RISK_OFF").sum()),
        "mismatch_by_variant": mismatches,
        "engine_equals": match,
    }
