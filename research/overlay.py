"""P10-D 字面交易覆盖层（Dow Trading Overlay）.

**这一层不是道氏理论。** 这是把引擎输出的市场状态映射成仓位的规则，
是研究者自己选的。必须显式隔离，否则以后无法回答：

    是道氏有效，还是仓位规则有效？

提供三种覆盖层（都不加杠杆、不做空，遵守 perf.py 的执行口径）：

    CASH_MODEL     RISK_ON → 100% 指数；WAIT / RISK_OFF → 现金
    HOLD_MODEL     只有 RISK_OFF 才离场（WAIT 仍然持有）—— 适合长期持有者
    PAIR_MODEL     RISK_ON → 持有指数 A；RISK_OFF → 持有指数 B（B 作为防御资产）

第三种是给后面"Dow Regime 作为市场环境闸门"做准备：道氏只负责决定
"现在该不该承担 beta 风险"，而不是"买哪只票"。
"""

from __future__ import annotations

import pandas as pd

from .perf import backtest, metrics


def _frame(engine_df: pd.DataFrame) -> pd.DataFrame:
    d = engine_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def cash_model(d: pd.DataFrame, fee_bps: float = 5.0) -> pd.DataFrame:
    close, dates = d["close_a"].astype(float), d["date"]
    pos = (d["dow_signal"] == "RISK_ON").astype(float).shift(1).fillna(0.0)
    return backtest(close, pos, dates=dates, fee_bps=fee_bps)


def hold_model(d: pd.DataFrame, fee_bps: float = 5.0) -> pd.DataFrame:
    """WAIT 仍持有，只有 RISK_OFF 离场."""
    close, dates = d["close_a"].astype(float), d["date"]
    pos = (d["dow_signal"] != "RISK_OFF").astype(float).shift(1).fillna(0.0)
    return backtest(close, pos, dates=dates, fee_bps=fee_bps)


def pair_model(d: pd.DataFrame, fee_bps: float = 5.0) -> pd.DataFrame:
    """RISK_ON → 持有指数 A；否则持有指数 B（防御）.

    这是一个"道氏决定 beta 开关、第二指数做防御"的最小化版本，
    用来验证道氏状态是否能带来正的资产切换收益。
    """
    d = d.copy()
    a = d["close_a"].astype(float).reset_index(drop=True)
    b = d["close_b"].astype(float).reset_index(drop=True)

    want_a = (d["dow_signal"] == "RISK_ON").astype(float)
    want_a = want_a.shift(1).fillna(0.0).reset_index(drop=True)
    w_a = want_a
    w_b = 1.0 - want_a

    ret_a = a.pct_change().fillna(0.0)
    ret_b = b.pct_change().fillna(0.0)
    ret = w_a * ret_a + w_b * ret_b

    turnover = (w_a.diff().abs().fillna(w_a.abs()) + w_b.diff().abs().fillna(w_b.abs())) / 2.0
    cost = turnover * (fee_bps / 1e4)
    net = ret - cost
    equity = (1 + net).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    out = pd.DataFrame({
        "date": d["date"].reset_index(drop=True),
        "close": a,
        "position": w_a,
        "ret": ret,
        "turnover": turnover,
        "cost": cost,
        "equity": equity,
        "drawdown": dd,
    })
    return out


def run_overlays(engine_df: pd.DataFrame, fee_bps: float = 5.0) -> pd.DataFrame:
    d = _frame(engine_df)
    rows = [
        metrics(cash_model(d, fee_bps), "CASH_MODEL", fee_bps=fee_bps),
        metrics(hold_model(d, fee_bps), "HOLD_MODEL", fee_bps=fee_bps),
        # A/B 切换：ret 已是加权组合收益，不能再乘 position
        metrics(pair_model(d, fee_bps), "PAIR_MODEL(A/B)", fee_bps=fee_bps, weighted=True),
    ]
    return pd.DataFrame(rows)
