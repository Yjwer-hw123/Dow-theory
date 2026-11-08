"""线态窄幅区间盘整（第 14 章）.

书中定义：
    "一个『线态窄幅区间盘整』的形态，一般持续 2~3 周，或者更长的时间。
     在此期间内，两个平均价格指数的价格变化处于大约 5% 的窄幅区间范围内。
     这样的运动表示市场在收集或者派发。
     如果两个平均价格指数同时向上突破……就表明市场是处在收集阶段……"

纪律：
    * 必须 **两个指数同时** 满足窄幅条件；
    * 必须 **两个指数同时** 突破才算有效方向；
    * 单指数突破 → 一律不确认（第 12 / 14 章反复强调）。

V0.1.1 结构调整（不是调参数，是修逻辑不可能）
-------------------------------------------
旧实现把**突破日自己**算进了区间：

    seg = [j0, i]        ← 包含第 i 天
    hi  = max(highs[seg])
    突破条件 := close[i] > hi

因为 hi 已经包含了第 i 天的最高价，`close[i] > hi` 在任何情况下都不可能成立 ——
也就是说 **`ACCUMULATION / DISTRIBUTION / *_CONFIRMED` 三个状态在数学上永远不会触发**。
这就是"21 年 1092 个成线日、0 次双指数同时突破"的真正原因，不是市场没给机会。

正确语义应该完全贴合第 14 章原文：

    盘整区间 := 过去 N 根 K 线（不含当日）
    突破     := 当日收盘价 越过 该区间

所以这里改为：区间取 `[i-N, i-1]`，第 i 根只用于判定突破方向。
**任何数值参数都没动**（5% / 10 根都原封不动）。
"""

from __future__ import annotations

from datetime import date as Date
from typing import Optional, Sequence

import pandas as pd


def range_width(highs: Sequence[float], lows: Sequence[float]) -> tuple[float, float, float]:
    """返回 (上沿, 下沿, 宽度比例)."""
    hi = max(highs)
    lo = min(lows)
    width = (hi - lo) / lo if lo > 0 else float("nan")
    return hi, lo, width


def detect_line_state(
    dates: Sequence[Date],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    i: int,
    params,
) -> dict:
    """在第 i 个交易日判断"线"的状态.

    Returns
    -------
    dict: {line_active, line_width, line_duration, line_high, line_low, line_state}
    """
    n_bars = params.line_min_bars
    # 区间 = 过去 n_bars 根（不含当日）；当日只用来判定是否突破
    if i < n_bars:
        return {"line_active": False, "line_width": None, "line_duration": 0,
                "line_high": None, "line_low": None, "line_state": "NONE",
                "line_breakout_dir": "NONE"}

    j0, j1 = i - n_bars, i          # 切片 [j0, j1) = n_bars 根
    seg_h, seg_l = highs[j0:j1], lows[j0:j1]
    hi, lo, width = range_width(seg_h, seg_l)
    active = width <= params.line_width_pct

    state = "FORMING" if active else "NONE"
    direction = "NONE"
    if active:
        last_c = closes[i]
        if last_c > hi:
            direction = "UP"
            state = "ACCUMULATION"
        elif last_c < lo:
            direction = "DOWN"
            state = "DISTRIBUTION"

    return {
        "line_active": active,
        "line_width": round(width, 5) if width == width else None,
        "line_duration": n_bars,
        "line_high": round(hi, 4),
        "line_low": round(lo, 4),
        "line_state": state,
        "line_breakout_dir": direction,
    }


def _dates(df: pd.DataFrame) -> list[Date]:
    """兼容 datetime 列与 datetime.date 列."""
    s = df["date"]
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.date.tolist()
    return pd.to_datetime(s).dt.date.tolist()


def combine_line(a: dict, b: dict) -> dict:
    """把两个单指数的线态结果合成双指数结果（纯函数，不碰 DataFrame）.

    与 `detect_line_pair` 的输出完全一致，抽出来是为了让引擎能在主循环里
    **复用预提取好的列表** —— 旧实现每行都重建整个 tolist()，是 O(n²)，
    21 年数据要跑 26 秒，Prefix Invariance 审计根本跑不动。
    """
    both_active = bool(a["line_active"] and b["line_active"])
    # 单指数突破方向始终作为诊断输出 —— 用于回答"0 次同时突破到底是市场没给，
    # 还是同时性要求过严"（见 scripts/audit_line.py）
    dir_a, dir_b = a.get("line_breakout_dir", "NONE"), b.get("line_breakout_dir", "NONE")
    diag = {"line_breakout_a": dir_a, "line_breakout_b": dir_b}
    wa, wb, dur = a["line_width"], b["line_width"], a["line_duration"]

    if not both_active:
        return {**diag, "line_state": "NONE", "line_width_a": wa,
                "line_width_b": wb, "line_active": False,
                "line_breakout": "NONE", "line_duration": dur}

    if dir_a == "UP" and dir_b == "UP":
        return {**diag, "line_state": "ACCUMULATION", "line_breakout": "UP_CONFIRMED",
                "line_width_a": wa, "line_width_b": wb,
                "line_active": True, "line_duration": dur}
    if dir_a == "DOWN" and dir_b == "DOWN":
        return {**diag, "line_state": "DISTRIBUTION", "line_breakout": "DOWN_CONFIRMED",
                "line_width_a": wa, "line_width_b": wb,
                "line_active": True, "line_duration": dur}
    return {**diag, "line_state": "FORMING", "line_breakout": "NO_CONFIRMATION",
            "line_width_a": wa, "line_width_b": wb,
            "line_active": True, "line_duration": dur}


def detect_line_pair(
    df_a: pd.DataFrame, df_b: pd.DataFrame, i: int, params
) -> dict:
    """两个指数同时满足才算成线 / 同时突破才算方向确认.

    便捷入口（内部做 tolist()，**不要在逐日主循环里调用** —— 用 `combine_line`
    配合预提取的列表，见 `core/regime_engine.py`）。
    """
    a = detect_line_state(
        None, df_a["high"].tolist(),
        df_a["low"].tolist(), df_a["close"].tolist(), i, params,
    )
    b = detect_line_state(
        None, df_b["high"].tolist(),
        df_b["low"].tolist(), df_b["close"].tolist(), i, params,
    )
    return combine_line(a, b)


def line_series(df_a: pd.DataFrame, df_b: pd.DataFrame, params) -> pd.DataFrame:
    rows = []
    for i in range(len(df_a)):
        rows.append({"date": df_a["date"].iloc[i].date(), **detect_line_pair(df_a, df_b, i, params)})
    return pd.DataFrame(rows)
