"""A 股适配层（V1 研究用，不是 V0）.

职责：
    1. 提供候选指数配对并对齐交易日；
    2. 跑多组参数的**横向对比**，回答那个关键问题：
       "如果换一组相近的市场指数 / 换一个参数，道氏信号还成立吗？"

纪律：这里的结果只能用于 **样本外 / Walk-Forward** 检验，
     不得反过来修改 V0.1 冻结参数。
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from config.dow_params import ASHARE_CANDIDATES, DowParams
from core.regime_engine import analyze_pair
from data.loader import load_index_pair
from data.universe import PAIRS, IndexMeta, get_pair, INDICES


def run_pair(
    pair_idx: int | str = 0,
    params: DowParams | None = None,
    refresh: bool = False,
    with_volume: bool = True,
):
    name, code_a, code_b = get_pair(pair_idx)
    df_a, df_b = load_index_pair(code_a, code_b, refresh=refresh)
    if df_a.empty or df_b.empty:
        raise RuntimeError(f"无法获取指数数据: {name}")
    df, meta = analyze_pair(df_a, df_b, params or ASHARE_CANDIDATES.get("ashare_swing025"), with_volume=with_volume)
    meta["pair"] = name
    meta["code_a"] = code_a
    meta["code_b"] = code_b
    return df, meta


def robustness_matrix(
    pairs: Iterable[int | str] = (0, 1, 2, 3),
    param_versions: Iterable[str] = ("v0.1",),
    refresh: bool = False,
) -> pd.DataFrame:
    """跑「指数配对 × 参数版本」的鲁棒性矩阵."""
    from config.dow_params import get_params

    rows = []
    for p in pairs:
        name, code_a, code_b = get_pair(p)
        df_a, df_b = load_index_pair(code_a, code_b, refresh=refresh)
        if df_a.empty or df_b.empty:
            rows.append({"pair": name, "error": "no data"})
            continue
        for ver in param_versions:
            params = get_params(ver)
            df, meta = analyze_pair(df_a, df_b, params)
            row = {
                "pair": name,
                "version": params.name,
                "swings_a": meta["swings_a"],
                "swings_b": meta["swings_b"],
                "confirmed_rate": meta["confirmation"]["confirmed_rate"],
                "mean_lag_days": meta["confirmation"]["mean_lag_days"],
                "line_days": meta["line_days"],
            }
            for k, v in meta["trend_distribution"].items():
                row[f"trend_{k}"] = v
            for k, v in meta["signal_distribution"].items():
                row[f"signal_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)
