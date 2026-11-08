"""道氏事件数据库（P9）.

README 里的原则：
    **如果这个事件数据库都做不出来，就不要谈交易回测。**

每一行 = 一个交易日的完整机器可读市场状态。列结构直接对应
README 里定义的 market_state schema，不含任何"涨/跌预测"。
"""

from __future__ import annotations

import os

import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def build_event_store(df_a: pd.DataFrame, df_b: pd.DataFrame, params, with_volume: bool = True):
    from core.regime_engine import analyze_pair

    return analyze_pair(df_a, df_b, params, with_volume=with_volume)


def save_event_store(df: pd.DataFrame, tag: str = "regime") -> str:
    path = os.path.join(OUT_DIR, f"{tag}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_event_store(tag: str = "regime") -> pd.DataFrame:
    path = os.path.join(OUT_DIR, f"{tag}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df


def trend_changes(df: pd.DataFrame) -> pd.DataFrame:
    """状态切换事件表（原趋势 → 新趋势）."""
    rows = []
    prev = None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for _, r in df.iterrows():
        st = r["primary_trend"]
        if st == "UNCONFIRMED":
            continue
        if prev is None or prev["primary_trend"] != st:
            if prev is not None:
                rows.append(
                    {
                        "date": r["date"],
                        "from": prev["primary_trend"],
                        "to": st,
                        "days_in_prev": (r["date"] - prev["date"]).days,
                        "close_a": r["close_a"],
                        "dow_signal": r["dow_signal"],
                    }
                )
            prev = r
        else:
            prev = r
    return pd.DataFrame(rows)


def forward_returns(df_event: pd.DataFrame, horizons=(20, 60, 120)) -> pd.DataFrame:
    """状态的信息含量验证：各 state 之后的未来收益分布（不是回测，只是条件统计）."""
    df = df_event.copy()
    rows = []
    for h in horizons:
        df[f"fwd_{h}"] = df["close_a"].shift(-h) / df["close_a"] - 1
    for state, g in df.groupby("primary_trend"):
        row = {"state": state, "n": len(g), "share": round(len(g) / len(df), 3)}
        for h in horizons:
            col = f"fwd_{h}"
            row[f"fwd_{h}_mean"] = round(float(g[col].mean()), 4)
            row[f"fwd_{h}_win"] = round(float((g[col] > 0).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)
