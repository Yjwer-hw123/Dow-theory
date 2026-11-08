"""量价关系（第 15 章）.

书中核心不是"量大=看涨"，而是 **价格结果与成交量努力是否匹配**：

    "当市场处于超买状态时，就会在上升中表现出呆滞乏力，而在下跌中表现出活跃的动力；
     反之，当市场处于超卖状态时，就会在下跌中表现呆滞乏力，而在上升中表现出活跃的动力。
     牛市会以市场中相对较低的成交量为开始，而以市场的过度活跃而告终。"

因此输出的是 **关系状态**，而不是一个 volume_score。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

VOLUME_STATES = ("HEALTHY", "WEAK", "EXHAUSTION", "INSUFFICIENT_DATA")


def up_down_volume_arr(opens: Sequence[float], closes: Sequence[float],
                       volumes: Sequence[float], i: int, window: int = 20) -> dict:
    """`up_down_volume` 的等语义数组版（供逐日主循环使用）.

    旧版每行都做一次 `df.iloc[j0:i+1]` 的 DataFrame 切片 + 布尔索引，
    21 年数据累计是主要耗时来源之一。这里改为对预提取的列表切片，
    数值语义与 `up_down_volume` 完全一致（mean 同样走 numpy）。
    """
    j0 = max(0, i - window + 1)
    o = np.asarray(opens[j0: i + 1], dtype=float)
    c = np.asarray(closes[j0: i + 1], dtype=float)
    v = np.asarray(volumes[j0: i + 1], dtype=float)
    n = len(v)
    if n < max(5, window // 3):
        return {"up_vol": None, "down_vol": None, "ratio": None, "vol_trend": None}

    up = v[c > o]
    down = v[c < o]
    up_mean = float(up.mean()) if len(up) else 0.0
    down_mean = float(down.mean()) if len(down) else 0.0
    ratio = (up_mean / down_mean) if down_mean > 0 else None

    half = max(3, n // 2)
    recent = float(v[-half:].mean())
    prior = float(v[: n - half].mean()) if n > half else recent
    vol_trend = (recent / prior - 1) if prior > 0 else 0.0

    return {
        "up_vol": round(up_mean, 3),
        "down_vol": round(down_mean, 3),
        "ratio": round(ratio, 3) if ratio else None,
        "vol_trend": round(vol_trend, 4),
    }


def up_down_volume(df: pd.DataFrame, i: int, window: int = 20) -> dict:
    """上升日均量 vs 下跌日均量 + 与前期总量的对比."""
    j0 = max(0, i - window + 1)
    seg = df.iloc[j0: i + 1]
    if len(seg) < max(5, window // 3):
        return {"up_vol": None, "down_vol": None, "ratio": None, "vol_trend": None}

    up = seg[seg["close"] > seg["open"]]["volume"]
    down = seg[seg["close"] < seg["open"]]["volume"]
    up_mean = float(up.mean()) if len(up) else 0.0
    down_mean = float(down.mean()) if len(down) else 0.0
    ratio = (up_mean / down_mean) if down_mean > 0 else None

    # 成交量趋势：最近 window/2 均量 / 前 window/2 均量
    half = max(3, len(seg) // 2)
    recent = float(seg["volume"].iloc[-half:].mean())
    prior = float(seg["volume"].iloc[: len(seg) - half].mean()) if len(seg) > half else recent
    vol_trend = (recent / prior - 1) if prior > 0 else 0.0

    return {
        "up_vol": round(up_mean, 3),
        "down_vol": round(down_mean, 3),
        "ratio": round(ratio, 3) if ratio else None,
        "vol_trend": round(vol_trend, 4),
    }


def classify_volume(trend: str, vol: dict, price_change_20d: float | None = None) -> str:
    """按第 15 章判据分类.

    HEALTHY   : 与趋势方向一致的努力（牛市上升增量 / 回调缩量）
    WEAK      : 努力与结果不匹配（牛市上涨缩量、上涨乏力下跌活跃）
    EXHAUSTION: 顶部/底部的过度活跃，或回调末端缩量止跌（分类器里的两种极端之一）
    """
    if vol.get("ratio") is None:
        return "INSUFFICIENT_DATA"
    ratio = vol["ratio"]
    vt = vol.get("vol_trend") or 0.0

    if trend == "UP":
        if ratio >= 1.15 and vt >= -0.05:
            return "HEALTHY"
        if ratio < 0.85 and price_change_20d is not None and price_change_20d < 0:
            return "EXHAUSTION"   # 上涨乏力、下跌活跃 = 超买
        if ratio < 1.0:
            return "WEAK"
        return "HEALTHY"
    if trend == "DOWN":
        if ratio <= 0.85:
            return "HEALTHY"      # 熊市下跌带量 = 趋势有效
        if ratio > 1.15 and vt > 0.1:
            return "EXHAUSTION"   # 下跌缩量后被踊跃买盘收复 = 超卖/可能反转
        return "WEAK"
    # TRANSITION / UNCONFIRMED
    return "HEALTHY" if 0.85 <= ratio <= 1.15 else "WEAK"


def volume_series(df: pd.DataFrame, trend_states: dict, params) -> pd.DataFrame:
    rows = []
    dates = df["date"].dt.date.tolist()
    for i, d in enumerate(dates):
        vol = up_down_volume(df, i, window=params.volume_window)
        p20 = None
        if i >= 20:
            p20 = float(df["close"].iloc[i] / df["close"].iloc[i - 20] - 1)
        state = classify_volume(trend_states.get(d, "UNCONFIRMED"), vol, p20)
        rows.append({"date": d, "volume_state": state, "vol_up_down_ratio": vol["ratio"],
                     "vol_trend": vol["vol_trend"]})
    return pd.DataFrame(rows)
