"""浪识别器 —— 整个项目最核心的模块（第 13 章）.

书中原话：
    "便于讨论，我们把在一天或多于一天的时间里，如果平均价格指数的反向运动
     超过了平均价格指数的 3%，就把这样的运动看成是一个上升浪或下跌浪。"

要点
----
1. **完全确定性的状态机**，绝不依赖人工画线。
2. 只使用**日收盘价**（原书研究的就是平均价格指数的日收盘值）。
3. 输出交替的 UP / DOWN 浪；最后一个未完成的浪标记为 `provisional=True`，
   在计算历史状态时应当被丢弃以避免未来函数。
"""

from __future__ import annotations

from datetime import date as Date
from typing import Sequence, Union

import pandas as pd

from .types import Swing


def detect_swings(
    dates: Sequence[Union[pd.Timestamp, Date]],
    closes: Sequence[float],
    pct: float = 0.03,
    index: str = "A",
) -> list[Swing]:
    """ZigZag 风格但其精神来自第 13 章的最小确定性浪识别.

    Parameters
    ----------
    dates, closes : 日度序列（升序），closes 为平均价格指数收盘值
    pct           : 反向运动阈值，默认 3%

    Returns
    -------
    list[Swing]   : 方向交替的浪列表，最后一个可能是 provisional。
    """
    n = len(closes)
    if n < 2 or pct <= 0:
        return []

    dates = list(pd.to_datetime(pd.Series(list(dates))).tolist())
    closes = [float(c) for c in closes]

    swings: list[Swing] = []

    def add(direction: str, i0: int, i1: int, provisional: bool = False,
            detected_i: int | None = None) -> None:
        """记录一条浪.

        `detected_i` 是"这条浪被市场识别出来"的索引 —— 即价格从 `i1`（极值点）
        反向走够 `pct` 的那一天。它通常比 `i1` 晚好几周，是 `Swing.known_date`
        的唯一来源，也是防第二层前瞻偏差的关键：
        **极值日不等于可知日**。
        """
        if i0 == i1:
            return
        swings.append(
            Swing(
                direction=direction,  # type: ignore[arg-type]
                start_date=dates[i0].date(),
                start_price=closes[i0],
                end_date=dates[i1].date(),
                end_price=closes[i1],
                index=index,
                provisional=provisional,
                detected_date=(dates[detected_i].date() if detected_i is not None else None),
            )
        )

    # ---- 初始方向：谁先突破阈值谁定 ----
    start_i = 0
    hi_i = 0
    lo_i = 0
    trend: str = ""  # "" 未定方向, "UP"/"DOWN"
    ext_i = 0
    first_up_i: int | None = None
    first_down_i: int | None = None

    for i in range(1, n):
        c = closes[i]
        if trend != "DOWN":
            if c > closes[hi_i]:
                hi_i = i
        if trend != "UP":
            if c < closes[lo_i]:
                lo_i = i

        if trend == "":
            if first_up_i is None and c >= closes[lo_i] * (1 + pct):
                first_up_i = i
            if first_down_i is None and c <= closes[hi_i] * (1 - pct):
                first_down_i = i
            # 谁先触发谁先定向；同日同时触发时以更早形成的极值为准
            if first_up_i is not None and (first_down_i is None or first_up_i <= first_down_i):
                trend = "UP"
                add("DOWN", 0, lo_i, detected_i=i)
                start_i, ext_i = lo_i, i
                hi_i = i
            elif first_down_i is not None:
                trend = "DOWN"
                add("UP", 0, hi_i, detected_i=i)
                start_i, ext_i = hi_i, i
                lo_i = i
            continue

        if trend == "UP":
            if c > closes[ext_i]:
                ext_i = i
            elif c <= closes[ext_i] * (1 - pct):
                add("UP", start_i, ext_i, detected_i=i)
                trend = "DOWN"
                start_i, ext_i = ext_i, i
                lo_i = i
        else:  # DOWN
            if c < closes[ext_i]:
                ext_i = i
            elif c >= closes[ext_i] * (1 + pct):
                add("DOWN", start_i, ext_i, detected_i=i)
                trend = "UP"
                start_i, ext_i = ext_i, i
                hi_i = i

    # 收尾：开放浪
    if start_i != n - 1:
        add(trend, start_i, n - 1, provisional=True)

    return swings


def detect_swings_df(df: pd.DataFrame, pct: float = 0.03, index: str = "A") -> list[Swing]:
    """DataFrame 便捷入口（需要 date / close 列）."""
    if df.empty:
        return []
    return detect_swings(df["date"].tolist(), df["close"].tolist(), pct=pct, index=index)


def swings_completed_asof(swings: list[Swing], when: Date) -> list[Swing]:
    """取截至 `when`（含）为止已经完成的浪 —— 丢弃 provisional 与未来浪，
    这是所有历史回算必须使用的函数（防未来函数）."""
    out = []
    for s in swings:
        if s.provisional:
            continue
        if s.end_date <= when:
            out.append(s)
    return out


def last_extreme_price(swings: list[Swing], direction: str) -> tuple[float, Date] | None:
    """最近的某个方向的极值：UP 浪取终点(高点)，DOWN 浪取终点(低点)."""
    hits = [s for s in swings if s.direction == direction]
    if not hits:
        return None
    s = hits[-1]
    return s.end_price, s.end_date
