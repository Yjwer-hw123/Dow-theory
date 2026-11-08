"""回测绩效层（P10 共用）.

这里冻结的是**交易执行口径**，不是道氏参数 —— 两者必须分开：

    Dow Literal Engine  ≠  Dow Trading Overlay

口径（写死在注释里，避免以后偷偷改）：
    1. 信号在 t 日收盘产生，t+1 日起生效（position_t = signal_{t-1}）。
       任何忽略这一点的回测都会凭空多出收益。
    2. 仓位只有 0 / 1（指数覆盖层不做杠杆、不做做空）。
    3. 成本：按换手量线性计费，默认单边 5bp（`fee_bps`）。
    4. 收益用收盘价对数/简单收益均可取简单：r_t = C_t / C_{t-1} - 1。
    5. 所有策略共用同一段样本、同一个 Target(指数 A)。

不在这一层做的事：选股、行业、资金流、止损 —— 那是 V2 交易系统的范围。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_FEE_BPS = 5.0        # 单边成本（bp）
TRADING_DAYS = 252


def positions_from_signals(signal: pd.Series,
                           on: str = "RISK_ON",
                           off: tuple = ("RISK_OFF", "WAIT")) -> pd.Series:
    """信号 → 次日生效的目标仓位."""
    pos = signal.eq(on).astype(float)
    return pos.shift(1).fillna(0.0)


def backtest(prices: pd.Series,
             target: pd.Series,
             dates: Optional[pd.Series] = None,
             fee_bps: float = DEFAULT_FEE_BPS) -> pd.DataFrame:
    """按 target 仓位跑一遍，返回含净值与回撤的日序列."""
    px = pd.Series(prices).astype(float).reset_index(drop=True)
    pos = pd.Series(target).astype(float).reset_index(drop=True).fillna(0.0)
    pos = pos.clip(0.0, 1.0)
    ret = px.pct_change().fillna(0.0)

    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (fee_bps / 1e4)
    gross = pos * ret
    net = gross - cost
    equity = (1 + net).cumprod()

    peak = equity.cummax()
    dd = equity / peak - 1.0
    out = pd.DataFrame({
        "date": (dates.reset_index(drop=True) if dates is not None else px.index),
        "close": px,
        "position": pos,
        "ret": ret,
        "turnover": turnover,
        "cost": cost,
        "equity": equity,
        "drawdown": dd,
    })
    return out


def metrics(bt: pd.DataFrame, label: str = "", fee_bps: float = DEFAULT_FEE_BPS,
            weighted: bool = False) -> dict:
    """从回测日序列算绩效.

    weighted : 该序列的 `ret` 是否**已经是加权后的组合收益**。
        普通单资产策略为 False（`ret` 是资产收益，需乘 position）；
        像 A/B 切换这种组合为 True（`ret` 已含权重，不能再乘）。
    """
    eq = bt["equity"]
    years = len(eq) / TRADING_DAYS
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else 0.0
    ret_series = (bt["ret"] - bt["cost"]) if weighted else (bt["ret"] * bt["position"] - bt["cost"])
    vol = float(ret_series.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(ret_series.mean() / ret_series.std() * np.sqrt(TRADING_DAYS)) if ret_series.std() > 0 else 0.0
    max_dd = float(bt["drawdown"].min())
    downside = ret_series[ret_series < 0]
    sortino = float(ret_series.mean() / downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) and downside.std() > 0 else 0.0

    exposure = float(bt["position"].mean())
    waiting = float((bt["position"] <= 0).mean())
    n_trades = int(((bt["position"].diff().fillna(0) != 0) & (bt["position"] == 1)).sum())
    turnover_per_year = float(bt["turnover"].sum() / years) if years > 0 else 0.0

    # ---- 口径无关的"信号质量"指标 ----
    # 不同策略的持仓比例可以差很多（Dow 12% vs MA200 52%），直接比年化不公平。
    # 下面两个指标把"持有多少天"的影响剥掉，只看**持仓期间**的收益质量：
    held = bt["ret"][bt["position"] > 0]
    in_market_ann = float((1 + held.mean()) ** TRADING_DAYS - 1) if len(held) else 0.0
    held_hit = float((held > 0).mean()) if len(held) else 0.0

    return {
        "strategy": label,
        "total_return": round(total, 4),
        "cagr": round(cagr, 4),
        "ann_vol": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(cagr / abs(max_dd), 3) if max_dd < 0 else None,
        "exposure": round(exposure, 3),
        "time_wait_pct": round(waiting, 3),
        "n_entries": n_trades,
        "turnover_per_year": round(turnover_per_year, 2),
        "in_market_ann_ret": round(in_market_ann, 4),
        "in_market_hit": round(held_hit, 3),
        "fee_bps": fee_bps,
    }


def compare_table(rows: list[dict], bh_max_dd: Optional[float] = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if bh_max_dd is not None and "max_drawdown" in df:
        df["dd_avoidance"] = (df["max_drawdown"] - bh_max_dd).round(4)
    return df
