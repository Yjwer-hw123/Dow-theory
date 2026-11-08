"""P10-A 事件研究：这个状态**之后**市场发生了什么.

注意这**不是回测**。回测会被仓位规则和入场时点污染；事件研究只看一件事：

    某个状态 / 事件出现之后，未来 20/60/120 个交易日的收益分布是否发生了稳定变化？

包含三块：
    1. 状态的条件收益表（primary_trend / integrity / secondary / signal）
    2. 事件研究（CONFIRMED_UP / CONFIRMED_DOWN / INVALIDATED / line breakout）
       含 MFE / MAE：最大有利偏移 / 最大不利偏移
    3. TRANSITION 子状态拆解（T1~T6）—— TRANSITION 占 57%，必须拆开看

MFE/MAE 的意义：即使某个状态的期望收益一般，只要 MFE 远大于 |MAE|，
它对"能不能拿得住"仍然有信息。
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

HORIZONS = (20, 60, 120)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    for h in HORIZONS:
        d[f"fwd_{h}"] = d["close_a"].shift(-h) / d["close_a"] - 1.0
    return d


def _agg(g: pd.DataFrame, label: str) -> dict:
    row = {"state": label, "n": len(g)}
    for h in HORIZONS:
        col = g[f"fwd_{h}"].dropna()
        row[f"fwd{h}_mean"] = round(float(col.mean()), 4) if len(col) else None
        row[f"fwd{h}_med"] = round(float(col.median()), 4) if len(col) else None
        row[f"fwd{h}_win"] = round(float((col > 0).mean()), 3) if len(col) else None
        row[f"fwd{h}_std"] = round(float(col.std()), 4) if len(col) else None
    return row


def state_table(df: pd.DataFrame, by: str = "primary_trend") -> pd.DataFrame:
    d = _prepare(df)
    rows = [_agg(g, str(k)) for k, g in d.groupby(by)]
    out = pd.DataFrame(rows).sort_values("fwd60_mean", ascending=False)
    total = len(d)
    out["share"] = (out["n"] / total).round(3)
    return out


def transition_substates(df: pd.DataFrame) -> pd.DataFrame:
    """把 TRANSITION 这个大桶拆成子状态.

    注意一个容易踩的坑：TRANSITION 的**定义**就是 `(H2>H1) != (L2>L1)`（一侧抬升、
    另一侧下降）。所以任何形如"用 HH/LL 是否矛盾"来判断"是不是真结构冲突"的写法，
    都会对 100% 的 TRANSITION 成立，把一个 catch-all 当成主类别（本脚本第一版就
    踩了这个坑，T6 一度占了 90%）。

    正确做法是用**变化的幅度**来分层，并按"最能解释当前混乱的原因"排优先级：

        T5 成线          两个指数同时进入 5% 窄幅区间 —— 盘整本身就能解释结构混乱
        T4 内部浪噪声     高低点两侧变化都 < 3%，结构其实没动
        T1 刚失去原趋势    integrity=DAMAGED（原趋势的结构已经破了）
        T3 指数背离       双指数未同步
        T2 正在形成新趋势  只有**一侧**在明显延伸（另一侧只是微动）
        T6 真结构冲突     两侧都在明显变化且方向相反 —— 这才是真正的矛盾
    """
    d = _prepare(df)
    d["substate"] = label_transition_substates(d)
    t = d[d["primary_trend"] == "TRANSITION"]
    if t.empty:
        return pd.DataFrame()

    rows = [_agg(g, str(k)) for k, g in t.groupby("substate")]
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out["share_of_transition"] = (out["n"] / len(t)).round(3)
    return out


def label_transition_substates(d: pd.DataFrame) -> pd.Series:
    """给每一行打 TRANSITION 子状态标签（非 TRANSITION 行 → None）.

    抽成独立函数，是为了让归因模块（`research/attribution.py`）复用**同一套**
    标签定义 —— 两份实现各自漂移会让"T2 是否稳定"这类结论失去意义。

    要求 `d` 已含 H1/H2/L1/L2/dual_confirmation/trend_integrity/line_active/
    primary_trend 列。
    """
    dh = d["H2"] / d["H1"] - 1
    dl = d["L2"] / d["L1"] - 1
    band = 0.03                      # "有没有明显变化"的界：沿用第 13 章的 3%，不新增参数
    big_h, big_l = dh.abs() >= band, dl.abs() >= band

    # 从低优先级到高优先级依次覆盖 → 最终等价于"最高优先级的命中项胜出"
    label = pd.Series("T6_real_structure_conflict", index=d.index)
    label[big_h ^ big_l] = "T2_forming_new_trend"
    label[d["dual_confirmation"] == "DIVERGENCE"] = "T3_index_divergence"
    label[d["trend_integrity"] == "DAMAGED"] = "T1_just_lost_trend"
    label[~big_h & ~big_l] = "T4_micro_swing_noise"
    label[d["line_active"] == True] = "T5_line_forming"  # noqa: E712

    is_tr = d["primary_trend"] == "TRANSITION"
    out = pd.Series(index=d.index, dtype="object")
    out[is_tr] = label[is_tr]
    return out


def _mfe_mae(d: pd.DataFrame, idx: Sequence[int], h: int) -> tuple[float, float]:
    mfe, mae = [], []
    for i in idx:
        if i + h >= len(d):
            continue
        window = d["close_a"].iloc[i + 1: i + 1 + h]
        if window.empty:
            continue
        base = d["close_a"].iloc[i]
        mfe.append(float(window.max() / base - 1))
        mae.append(float(window.min() / base - 1))
    return (float(sum(mfe) / len(mfe)) if mfe else float("nan"),
            float(sum(mae) / len(mae)) if mae else float("nan"))


def event_study(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """每个事件首次出现日的前瞻收益 / MFE / MAE."""
    d = _prepare(df)
    d[f"fwd_{horizon}"] = d["close_a"].shift(-horizon) / d["close_a"] - 1.0

    u = (d["primary_trend"] == "UP") & (d["dual_confirmation"] == "CONFIRMED")
    dn = (d["primary_trend"] == "DOWN") & (d["dual_confirmation"] == "CONFIRMED")
    inv = d["secondary_reaction"] == "INVALIDATED"
    events: dict[str, list[int]] = {
        "CONFIRMED_UP_trend": _runs_starts(u),
        "CONFIRMED_DOWN_trend": _runs_starts(dn),
        "SECONDARY_INVALIDATED": _runs_starts(inv),
        "LINE_BREAKOUT_UP": list(d.index[d["line_breakout"] == "UP_CONFIRMED"]),
        "LINE_BREAKOUT_DOWN": list(d.index[d["line_breakout"] == "DOWN_CONFIRMED"]),
    }

    rows = []
    for name, idxs in events.items():
        if not idxs:
            continue
        fut = d[f"fwd_{horizon}"].iloc[idxs].dropna()
        mfe, mae = _mfe_mae(d, idxs, horizon)
        rows.append({
            "event": name,
            "n_events": len(idxs),
            f"fwd{horizon}_mean": round(float(fut.mean()), 4) if len(fut) else None,
            f"fwd{horizon}_median": round(float(fut.median()), 4) if len(fut) else None,
            f"fwd{horizon}_win": round(float((fut > 0).mean()), 3) if len(fut) else None,
            "MFE": round(mfe, 4) if mfe == mfe else None,
            "MAE": round(mae, 4) if mae == mae else None,
            "MFE_MAE_ratio": (round(mfe / abs(mae), 2) if mae and abs(mae) > 1e-9 else None),
        })
    return pd.DataFrame(rows)


def _runs_starts(mask: pd.Series) -> list[int]:
    """连续 True 区段的起点（避免把持续 300 天的状态算成 300 个事件）."""
    return [int(x) for x in _starts(mask)]


def _starts(mask: pd.Series) -> list:
    idx = mask.index[mask].tolist()
    if not idx:
        return []
    starts = [idx[0]]
    for prev, cur in zip(idx, idx[1:]):
        if cur != prev + 1:
            starts.append(cur)
    return starts


def false_reversal_rate(df: pd.DataFrame, horizon: int = 60) -> dict:
    """假反转率：进入 DOWN/RISK_OFF 之后，未来 h 日却是上涨的比例."""
    d = _prepare(df)
    col = f"fwd_{horizon}"
    off = d[d["dow_signal"] == "RISK_OFF"][col].dropna()
    on = d[d["dow_signal"] == "RISK_ON"][col].dropna()
    return {
        f"RISK_OFF_n": int(len(off)),
        f"RISK_OFF_false_{horizon}d_up": round(float((off > 0).mean()), 3) if len(off) else None,
        f"RISK_ON_n": int(len(on)),
        f"RISK_ON_wrong_{horizon}d_down": round(float((on < 0).mean()), 3) if len(on) else None,
    }
