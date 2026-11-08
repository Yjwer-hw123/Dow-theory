"""次级反应模块（第 10 章）.

书中核心范围：次级反应通常回调前期主要趋势行程的 **1/3 ~ 2/3**。
第 10 章的历史统计还给了分布真相：

    回撤 10~25%:  7.1%
    回撤 25~40%: 25.4%
    回撤 40~55%: 18.8%
    回撤 55~70%: 26.7%
    回撤 70~85%:  8.5%
    回撤 >85%  : 14.0%      ← 注意：超过 85% 的次级反应也并不罕见

所以这里 **绝不** 写 `if retracement == 57.6%: secondary_reaction_end = True`。
57.6% 只是均值，我们要输出的是"它现在处在什么状态"。

正确用法是把次级反应做成一个状态机：

    BULL
      ↓ 出现 >=3% 反向浪
    SECONDARY_ACTIVE
      ↓ 回撤进入 33~66%
      ↓ 前低(HL)未被破坏
      ↓ 下跌缩量 / 结构上出现再次上升
    SECONDARY_COMPLETED → PRIMARY_TREND_RESUMED
"""

# ----------------------------------------------------------------------------
# V0.1.1 最重要的一条语义修正
# ----------------------------------------------------------------------------
# OVERDUE（时间超期）与 INVALIDATED（结构失效）是两种完全不同的信息，
# 必须拆成两个状态，不能都塞进 OVERDUE：
#
#     ACTIVE       retracement <= 上限、未超期、结构完好
#     OVERDUE      时间超过 secondary_max_days，但结构仍然完好
#     INVALIDATED  已经吃掉了本段主要运动行程的起点（ratio > 1）
#                  它已经不再是"次级反应"，而是主要趋势可能已经改变
#
# 为什么必须拆：旧实现把"跌破前低"也报成 OVERDUE，于是会出现
#
#     secondary_reaction = OVERDUE
#     retracement_ratio  = 108.89%
#     duration           = 261
#
# 这种组合。看报告的人会误以为"只是回调拖得久了点"，而实际上市场早已跌穿了
# 本段行程的起点，这在道氏语言里意味着 HH/HL 结构破坏 —— 是质变不是量变。
#
# 注意 ratio > 1 与 hl_intact 失效在数学上是同一个条件（H > L 时
# C < L ⟺ (H-C)/(H-L) > 1），所以这里统一用"结构"表达，不再用两套说法。
# ----------------------------------------------------------------------------

from __future__ import annotations

from datetime import date as Date
from typing import Optional, Sequence

import pandas as pd

from .types import SecondaryState


def compute_retracement(
    prev_low: float, prev_high: float, corrective_low: float
) -> float:
    """次级反应回撤比例：(H - C) / (H - L)."""
    span = prev_high - prev_low
    if span <= 0:
        return 0.0
    return (prev_high - corrective_low) / span


def evaluate_secondary(
    trend: str,
    prev_low: Optional[float],
    prev_high: Optional[float],
    corrective_low: float,
    corrective_start: Optional[Date],
    today: Date,
    params,
    hl_intact: bool = True,
    resumed: bool = False,
) -> dict:
    """次级反应状态判定.

    Parameters
    ----------
    trend           : "UP" / "DOWN" / "TRANSITION" / "UNCONFIRMED"（主趋势）
    prev_low/high   : 上一主要运动行程的起点低点与终点高点（牛市情形）
                      熊市情形请传入 prev_high=起点高点、prev_low=终点低点
    corrective_low  : 当前反向运动到达的最低价（熊市中为反弹最高价）
    corrective_start: 反向运动开始日
    today           : 当前日期
    hl_intact       : 前低/前高是否仍未被破坏
    resumed         : 是否已经重新突破反向运动的高点（趋势恢复）
    """
    out = {
        "state": "NONE",
        "retracement": 0.0,
        "duration": 0,
        "in_band": False,
        "note": "",
    }
    if trend not in ("UP", "DOWN"):
        out["note"] = "no established trend"
        return out
    if prev_low is None or prev_high is None or corrective_start is None:
        out["note"] = "insufficient structure"
        return out

    hi = max(prev_high, prev_low)
    lo = min(prev_high, prev_low)
    span = hi - lo
    duration = (today - corrective_start).days
    if span <= 0:
        out["note"] = "degenerate leg (zero span)"
        return out

    # 方向敏感的回撤定义：
    #   UP   (H - 回调低点) / (H - L)
    #   DOWN (反弹高点 - L) / (H - L)
    # 熊市下 `corrective_low` 这个参数实际传入的是反弹最高价。
    ratio = ((hi - corrective_low) / span) if trend == "UP" else ((corrective_low - lo) / span)

    # 诊断值：允许 > 1（表示已经吃掉本段行程起点），只在极端异常时截断防溢出
    out["retracement"] = round(max(0.0, min(ratio, 3.0)), 4)
    out["duration"] = duration

    # ---- 第一优先：结构是否已经被吃掉（质变） ----
    beyond_origin = (not hl_intact) or ratio > 1.0
    if beyond_origin:
        out["state"] = "INVALIDATED"
        out["note"] = (
            "reaction consumed prior trend-leg origin -> "
            "no longer a secondary reaction (structure damaged)"
        )
        return out

    if resumed:
        out["state"] = "COMPLETED"
        out["note"] = "reaction extreme reclaimed by counter-swing"
        return out

    if ratio <= 0.05:
        out["state"] = "NONE"
        out["note"] = "no meaningful counter-move yet"
        return out

    in_band = params.retrace_min <= ratio <= params.retrace_max
    out["in_band"] = in_band

    # ---- 第二优先：时间超期（量变，结构仍完好） ----
    if duration > params.secondary_max_days:
        out["state"] = "OVERDUE"
        out["note"] = f"correction longer than {params.secondary_max_days} days but leg origin intact"
    elif in_band:
        out["state"] = "ACTIVE"
        out["note"] = "inside classic 1/3~2/3 secondary band"
    elif ratio < params.retrace_min:
        out["state"] = "ACTIVE"
        out["note"] = "shallow correction, still within normal development"
    else:
        out["state"] = "ACTIVE"
        out["note"] = "deep retracement beyond 2/3 -> watch for trend change"
    return out


def secondary_state_for_date(
    trend_states: dict[Date, str],
    structure_by_date: dict[Date, dict],
    dates: Sequence[Date],
    highs: Sequence[float],
    lows: Sequence[float],
    params,
) -> pd.DataFrame:
    """逐日次级反应状态（供事件数据库调用）.

    对每个交易日：
      1. 取当日已成立的主趋势；
      2. 取该趋势的"上一主要运动行程"(L→H 或 H→L)；
      3. 取自该行程结束以来的反向极值；
      4. 计算回撤比例、持续天数、再创新高是否发生。
    """
    rows = []
    for i, d in enumerate(dates):
        trend = trend_states.get(d, "UNCONFIRMED")
        struct = structure_by_date.get(d, {})
        rows.append(
            {
                "date": d,
                "trend": trend,
                **_one_day(d, trend, struct, dates[: i + 1], highs[: i + 1], lows[: i + 1], params),
            }
        )
    return pd.DataFrame(rows)


def _one_day(d, trend, struct, dates, highs, lows, params) -> dict:
    if trend not in ("UP", "DOWN"):
        return {"secondary_reaction": "NONE", "retracement_ratio": 0.0,
                "secondary_duration": 0, "secondary_note": "no established trend"}

    # 上一主要运动行程：牛市取最近一个已确认低点→其后高点；熊市反之
    if trend == "UP":
        start_extreme, end_extreme = struct.get("L2"), struct.get("H2")
    else:
        start_extreme, end_extreme = struct.get("H2"), struct.get("L2")
    if start_extreme is None or end_extreme is None:
        return {"secondary_reaction": "NONE", "retracement_ratio": 0.0,
                "secondary_duration": 0, "secondary_note": "insufficient structure"}

    # 行程终点之后，反向极值
    span_lo, span_hi = min(start_extreme, end_extreme), max(start_extreme, end_extreme)
    try:
        end_idx = max(
            i for i in range(len(dates))
            if (lows[i] <= span_lo + 1e-9 or highs[i] >= span_hi - 1e-9)
        )
    except ValueError:
        return {"secondary_reaction": "NONE", "retracement_ratio": 0.0,
                "secondary_duration": 0, "secondary_note": "extreme not found"}

    tail_lows = lows[end_idx:]
    tail_highs = highs[end_idx:]
    if not tail_lows:
        return {"secondary_reaction": "NONE", "retracement_ratio": 0.0,
                "secondary_duration": 0, "secondary_note": "at extreme"}

    corrective_extreme = min(tail_lows) if trend == "UP" else max(tail_highs)
    resumed = (max(tail_highs) > span_hi) if trend == "UP" else (min(tail_lows) < span_lo)

    # 结构是否仍完整：牛市看上一级低点 L1 是否守住
    if trend == "UP":
        referee = struct.get("L1")
        hl_intact = referee is None or min(tail_lows) >= referee
    else:
        referee = struct.get("H1")
        hl_intact = referee is None or max(tail_highs) <= referee

    res = evaluate_secondary(
        trend=trend,
        prev_low=start_extreme,
        prev_high=end_extreme,
        corrective_low=corrective_extreme,
        corrective_start=dates[end_idx],
        today=d,
        params=params,
        hl_intact=hl_intact,
        resumed=resumed,
    )
    return {
        "secondary_reaction": res["state"],
        "retracement_ratio": res["retracement"],
        "secondary_duration": res["duration"],
        "secondary_note": res["note"],
    }
