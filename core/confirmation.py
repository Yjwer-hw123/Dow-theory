"""双指数确认引擎 —— 道氏理论的灵魂（第 12 章）.

书中反复强调：
    "两个平均价格指数必须相互确认……如果只是一个平均价格指数做出相应运动
     而没有得到另一个指数的确认，所得结论一般总是错误的。"
    "相应地列出必要条件：并不一定要求两个指数在同一天出现新高/新低。"

两种确认模式
------------
1. `until_next_reversal`（V0 默认，无额外参数）
   本指数的浪，必须在自己 **下一个反向浪开始之前**，
   被另一个指数的同向浪确认。这是最贴近原书"很重要延迟也可能成立、
   但不能无限期等待"的描述，且不引入可调参数。

2. `lag_days`
   另一指数的同向浪结束日必须落在本浪结束日 ±N 个交易日窗口内。
   这是 V1 阶段用于 A 股敏感性研究的候选，不是一个"更好"的选择。
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from .types import ConfirmedSwing, Swing


def _to_dt(d) -> datetime:
    return datetime(d.year, d.month, d.day)


def date_max():
    return datetime(9999, 12, 31).date()


def trading_days_between(d1, d2, calendar: Sequence[datetime] | None = None) -> int:
    """粗略交易日间隔：有日历时按位置差，无日历时按自然日折算（*5/7）."""
    if calendar:
        try:
            return abs(list(calendar).index(d1) - list(calendar).index(d2))
        except ValueError:
            pass
    return abs((_to_dt(d2) - _to_dt(d1)).days)


def confirm_primary(
    primary: list[Swing],
    other: list[Swing],
    mode: str = "until_next_reversal",
    lag_days: int | None = None,
    calendar: Sequence[datetime] | None = None,
) -> list[ConfirmedSwing]:
    """对一个指数的浪列表做确认.

    Parameters
    ----------
    primary : 主指数的浪（升序）
    other   : 用于确认的另一指数浪（升序）
    mode    : "until_next_reversal" | "lag_days"
    """
    # provisional（开放浪）不能参与历史确认判定：
    # 它的"终点"就是数据末尾，会随新数据不断变化，拿它当窗口上界必然引入未来函数。
    closed_primary = [s for s in primary if not s.provisional]
    other_closed = [o for o in other if not o.provisional]

    out: list[ConfirmedSwing] = []
    for i, s in enumerate(closed_primary):
        nxt = closed_primary[i + 1] if i + 1 < len(closed_primary) else None
        # 窗口的**价格**上界：B 的同向浪必须落在 A 这段运动之内（第 12 章语义）。
        # 之所以允许跨过下一个反向浪，是因为第 12 章明确写：
        # "并不一定要求两个指数在同一天出现新高/新低"，现实中的确认常常滞后。
        window_end = nxt.end_date if nxt is not None else None
        # 窗口**关闭**的可知时点：只有等 A 自己的下一条浪被识别出来，
        # 市场才知道"s 的确认窗口已经关闭"这件事。
        window_close_known = nxt.known_date if nxt is not None else None
        if mode == "lag_days" and lag_days is not None:
            from datetime import timedelta

            lag_cut = (_to_dt(s.end_date) + timedelta(days=int(lag_days * 1.6))).date()
            window_end = min(window_end or date_max(), lag_cut)

        # 找出所有"市场已经看得到"的匹配浪，取**最早被识别**的那条作为确认者。
        # 不能只取第一条：other_closed 是按极值日排的，而极值日早的浪未必
        # 最早被识别出来。取 earliest-known 才符合"市场最早何时知道确认成立"。
        hits = []
        for o in other_closed:
            if o.direction != s.direction:
                continue
            if o.end_date < s.start_date:
                continue
            if window_end is not None and o.end_date > window_end:
                continue
            # 可知性约束：o 必须在窗口关闭之前就被市场识别出来，
            # 否则事后匹配会用到"窗口都关了才浮现的浪"。
            if window_close_known is not None and o.known_date > window_close_known:
                continue
            hits.append(o)

        hit = min(hits, key=lambda o: o.known_date) if hits else None
        lag = trading_days_between(s.end_date, hit.end_date, calendar) if hit else None

        # ---- 防前瞻偏差：可用日期一律按 known_date 计算 ----
        #   已确认 → max(本浪识别日, 确认浪识别日)：市场最早在确认浪被识别时，
        #            才知道"这条浪被确认了"。
        #   未确认 → 确认窗口关闭的可知时点；若窗口至今未关闭（后面再没有更多浪），
        #            则 = date_max()，即"永远还不知道它会不会被确认"，不得使用。
        if hit is not None:
            available = max(s.known_date, hit.known_date)
        elif window_close_known is not None:
            available = max(s.known_date, window_close_known)
        else:
            available = date_max()

        out.append(ConfirmedSwing(swing=s, confirmed_by=hit, lag_days=lag,
                                  available_date=available))
    return out


def confirmation_summary(confirmed: list[ConfirmedSwing]) -> dict:
    if not confirmed:
        return {"n": 0, "confirmed_rate": 0.0}
    n = len(confirmed)
    k = sum(1 for c in confirmed if c.confirmed)
    lags = [c.lag_days for c in confirmed if c.lag_days is not None]
    return {
        "n": n,
        "confirmed": k,
        "confirmed_rate": round(k / n, 4),
        "mean_lag_days": round(sum(lags) / len(lags), 1) if lags else None,
        "max_lag_days": max(lags) if lags else None,
    }
