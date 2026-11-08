"""P10-C 简单基准：Dow 必须打败"一个简单的长期趋势滤波器"才算有增量信息.

这一点经常被忽略。如果 Dow 的年化和 MA200 差不多，那么整个三层结构、
双指数确认、次级反应统计就没有证明自己的价值 —— 它们只是把 MA200 重新
发明了一遍（而且还更复杂）。

参与的基准（全部只用指数的**收盘价**，不看成交额、不看第二指数）：

    BUY_HOLD      永远满仓
    MA200         收盘 > 200 日均线
    MA50_200      50 日线上穿 200 日线（金叉/死叉）
    MOM120        收盘 > 120 交易日前收盘（12 个月动量）

所有基准都遵守同一条执行口径（见 perf.py）：信号收盘产生、次日生效。
"""

from __future__ import annotations

import pandas as pd

from .perf import backtest, metrics, positions_from_signals


def _price_frame(engine_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    d = engine_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    return d["close_a"].astype(float), d["date"], d


def ma_signal(close: pd.Series, window: int) -> pd.Series:
    """收盘在均线之上 → RISK_ON，否则 WAIT（次日生效由 backtest 统一处理）."""
    ma = close.rolling(window, min_periods=window).mean()
    sig = pd.Series("WAIT", index=close.index)
    sig[close > ma] = "RISK_ON"
    return sig


def ma_cross_signal(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    sig = pd.Series("WAIT", index=close.index)
    sig[f > s] = "RISK_ON"
    return sig


def momentum_signal(close: pd.Series, lookback: int = 120) -> pd.Series:
    sig = pd.Series("WAIT", index=close.index)
    sig[close > close.shift(lookback)] = "RISK_ON"
    return sig


def build_baselines(engine_df: pd.DataFrame, fee_bps: float = 5.0) -> pd.DataFrame:
    """返回每个基准的绩效行（含 buy&hold）."""
    close, dates, d = _price_frame(engine_df)
    rows: list[dict] = []

    # BUY & HOLD
    pos = pd.Series(1.0, index=close.index)
    bt = backtest(close, pos, dates=dates, fee_bps=fee_bps)
    rows.append(metrics(bt, "BUY_HOLD", fee_bps=fee_bps))

    variants = {
        "MA200": ma_signal(close, 200),
        "MA50_200": ma_cross_signal(close, 50, 200),
        "MOM120": momentum_signal(close, 120),
    }
    for name, sig in variants.items():
        pos = positions_from_signals(sig, on="RISK_ON")
        bt = backtest(close, pos, dates=dates, fee_bps=fee_bps)
        rows.append(metrics(bt, name, fee_bps=fee_bps))

    return pd.DataFrame(rows)


def baseline_positions(engine_df: pd.DataFrame) -> dict[str, pd.Series]:
    """给别的模块复用：返回各基准的次日生效仓位."""
    close, _, _ = _price_frame(engine_df)
    out = {"BUY_HOLD": pd.Series(1.0, index=close.index)}
    out["MA200"] = positions_from_signals(ma_signal(close, 200), on="RISK_ON")
    out["MA50_200"] = positions_from_signals(ma_cross_signal(close, 50, 200), on="RISK_ON")
    out["MOM120"] = positions_from_signals(momentum_signal(close, 120), on="RISK_ON")
    return out
