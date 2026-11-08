"""P10.1 归因层：空仓原因、闸门解除敏感性、TRANSITION 子状态稳健性.

回答三个问题
------------
1. **P10.1-A 空仓归因** —— 那 87.6% 没持仓的日子，究竟是谁在拒绝？
   （TRANSITION / 未确认 / 双指数未确认 / 结构破坏 / 次级反应超期 / 线否决）
2. **P10.1-B 闸门解除敏感性** —— 假设只撤掉一道闸门，市场暴露会发生什么？
3. **P10.1-D TRANSITION 子状态稳健性** —— T2/T3 的正收益是稳定的，
   还是只靠某一年？

纪律（必须写在这里，防止以后被误读）
------------------------------------
**这一层只做诊断，不产出策略。**
`gate_variants` 里"撤掉某道闸门"的结果一律**不得**解释为"新的更好策略" ——
它只回答"哪一道闸门吃掉了最多的市场暴露"，用来给 P11 决定先研究哪个结构规则。
真正的参数/规则变更必须走"规则语义研究"，不能从这张表里挑一行抄。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .event_study import label_transition_substates

# 阻断原因的固定顺序（越靠上越根本：先看有没有趋势，再看结构，再看位置）
BLOCKER_ORDER = [
    "NO_TREND_UNCONFIRMED",     # 数据不足，连趋势都还没形成
    "TRANSITION",               # 结构混合，道氏拒绝二分
    "TREND_DOWN",               # 方向就是空头（RISK_OFF），不是"被阻断"
    "STRUCTURE_DAMAGED",        # 原趋势的结构已经破了
    "NO_DUAL_CONFIRMATION",     # 最近一条浪没被第二个指数确认
    "SECONDARY_INVALIDATED",    # 次级反应吃掉了本段行程起点
    "SECONDARY_OVERDUE",        # 次级反应超期
    "LINE_DOWN_OVERRIDE",       # 线态向下突破，单边否决 RISK_ON
    "OTHER",
]


def with_fwd(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """加一列前瞻收益（仅用于诊断统计，不参与任何信号生成）."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d[f"fwd_{horizon}"] = d["close_a"].shift(-horizon) / d["close_a"] - 1.0
    return d


def _n_episodes(idx) -> int:
    """连续区段数（把"连续 300 天同一状态"算成 1 次，而不是 300 次）."""
    idx = list(idx)
    if not idx:
        return 0
    n = 1
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            n += 1
    return n


def _mfe_mae(close: np.ndarray, idx, h: int) -> tuple[float, float]:
    """事件之后的平均最大有利偏移 / 最大不利偏移."""
    mfe, mae = [], []
    n = len(close)
    for i in idx:
        if i + h >= n:
            continue
        w = close[i + 1: i + 1 + h]
        if w.size == 0:
            continue
        mfe.append(float(w.max() / close[i] - 1))
        mae.append(float(w.min() / close[i] - 1))
    return (float(np.mean(mfe)) if mfe else float("nan"),
            float(np.mean(mae)) if mae else float("nan"))


def blocker_table(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """P10.1-A：每个"非 RISK_ON"交易日按**首个阻断原因**归类.

    第一个阻断原因是按原书闸门顺序（结构 → 趋势 → 确认 → 位置 → 线）判定的，
    由 `core/regime_engine.py` 在生成事件库时写入 `blocker` 列。
    """
    d = with_fwd(df, horizon)
    off = d[d["dow_signal"] != "RISK_ON"]
    if off.empty or "blocker" not in off.columns:
        return pd.DataFrame()

    col = f"fwd_{horizon}"
    close = d["close_a"].to_numpy(dtype=float)
    rows = []
    for k, g in off.groupby("blocker"):
        fut = g[col].dropna()
        mfe, mae = _mfe_mae(close, list(g.index), horizon)
        rows.append({
            "blocker": str(k),
            "days": int(len(g)),
            "share_of_no_position": round(len(g) / len(off), 4),
            "episodes": _n_episodes(g.index),
            "avg_len": round(len(g) / max(1, _n_episodes(g.index)), 1),
            f"fwd{horizon}_mean": round(float(fut.mean()), 4) if len(fut) else None,
            f"fwd{horizon}_win": round(float((fut > 0).mean()), 3) if len(fut) else None,
            "MFE": round(mfe, 4) if mfe == mfe else None,
            "MAE": round(mae, 4) if mae == mae else None,
            "MFE_MAE_ratio": (round(mfe / abs(mae), 2)
                              if mae == mae and abs(mae) > 1e-9 else None),
        })
    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(BLOCKER_ORDER)}
    out["_o"] = out["blocker"].map(order).fillna(99)
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def gate_variants(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """P10.1-B：从 Base 出发，每次只撤掉一道闸门（leave-one-gate-out）.

    **这不是消融实验的替代品，也不是策略候选。** P10 的 ablation 是"逐层
    叠加组件"，这里问的是相反的问题：给定当前这个已经过度选择性的引擎，
    哪一道闸门是市场暴露的主要瓶颈。
    """
    d = with_fwd(df, horizon)
    up = d["primary_trend"] == "UP"
    tr = d["primary_trend"] == "TRANSITION"
    conf = d["dual_confirmation"] == "CONFIRMED"
    intact = d["trend_integrity"] == "INTACT"
    blocked = d["secondary_reaction"].isin(["OVERDUE", "INVALIDATED"])
    line_down = d["line_breakout"] == "DOWN_CONFIRMED"

    base = up & conf & intact & ~blocked & ~line_down
    variants = {
        "Base = 引擎实际 RISK_ON": base,
        "撤掉次级反应闸门(OVERDUE/INVALIDATED)": up & conf & intact & ~line_down,
        "撤掉线态向下否决": up & conf & intact & ~blocked,
        "撤掉双指数确认要求": up & intact & ~blocked & ~line_down,
        "撤掉结构完好要求(integrity)": up & conf & ~blocked & ~line_down,
        "把 TRANSITION 也当作可入场": (up | tr) & conf & intact & ~blocked & ~line_down,
    }

    col = f"fwd_{horizon}"
    n = len(d)
    rows = []
    for name, mask in variants.items():
        fut = d.loc[mask, col].dropna()
        rows.append({
            "variant": name,
            "days": int(mask.sum()),
            "exposure": round(float(mask.mean()), 4),
            "vs_base_days": int(mask.sum() - base.sum()),
            f"fwd{horizon}_mean": round(float(fut.mean()), 4) if len(fut) else None,
            f"fwd{horizon}_win": round(float((fut > 0).mean()), 3) if len(fut) else None,
        })
    out = pd.DataFrame(rows)
    out.insert(0, "share_of_sample", (out["days"] / n).round(4))
    return out


def transition_stability(df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """P10.1-D：TRANSITION 子状态的正收益，是稳定的还是靠某一年？

    对每个子状态，除整体 fwd 收益外，再给出**分年度胜率的分布**：
      * `years_win_gt50`：胜率 > 50% 的年份数
      * `years_with_data`：有数据的年份数
      * `year_win_min/max`：年度胜率的极值

    如果某个子状态整体 fwd60 是正的，但 year_win_min 很低、years_win_gt50 很小，
    那它的"正收益"就是少数年份贡献的，不应写进任何规则。
    """
    d = with_fwd(df, horizon)
    if "H1" not in d.columns or "line_active" not in d.columns:
        return pd.DataFrame()
    d["substate"] = label_transition_substates(d)
    t = d[d["substate"].notna()].copy()
    if t.empty:
        return pd.DataFrame()
    t["year"] = t["date"].dt.year

    col = f"fwd_{horizon}"
    rows = []
    for k, g in t.groupby("substate"):
        fut = g[col].dropna()
        yearly = g.groupby("year")[col].apply(
            lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else np.nan
        )
        yearly = yearly.dropna()
        rows.append({
            "substate": str(k),
            "days": int(len(g)),
            "share_of_transition": round(len(g) / len(t), 4),
            f"fwd{horizon}_mean": round(float(fut.mean()), 4) if len(fut) else None,
            f"fwd{horizon}_win": round(float((fut > 0).mean()), 3) if len(fut) else None,
            "years_with_data": int(len(yearly)),
            "years_win_gt50": int((yearly > 0.5).sum()),
            "year_win_min": round(float(yearly.min()), 3) if len(yearly) else None,
            "year_win_max": round(float(yearly.max()), 3) if len(yearly) else None,
        })
    return pd.DataFrame(rows).sort_values(
        f"fwd{horizon}_mean", ascending=False).reset_index(drop=True)


def render_attribution_md(df: pd.DataFrame, title: str = "",
                          horizon: int = 60, extra: str = "") -> str:
    """把三块归因渲染成 Markdown."""
    lines: list[str] = []
    lines.append(f"# {title or 'P10.1 归因报告'}")
    lines.append("")
    lines.append(f"> 样本 {len(df)} 交易日；前瞻窗口 {horizon} 个交易日。")
    lines.append("> **本报告只做诊断，不产出策略。**"
                 "「撤掉闸门 → 暴露变多」的结果都不是新策略候选。")
    lines.append("")

    lines.append("## A. 空仓归因：谁在拒绝？")
    lines.append("")
    bt = blocker_table(df, horizon)
    if bt.empty:
        lines.append("（无数据）")
    else:
        lines.append("| 阻断原因 | 天数 | 占空仓 | 段数 | 平均段长 | fwd 均值 | fwd 胜率 | MFE | MAE | MFE/MAE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in bt.iterrows():
            lines.append(
                f"| {r['blocker']} | {r['days']} | {r['share_of_no_position']:.1%} | "
                f"{r['episodes']} | {r['avg_len']} | "
                f"{r[f'fwd{horizon}_mean']} | {r[f'fwd{horizon}_win']} | "
                f"{r['MFE']} | {r['MAE']} | {r['MFE_MAE_ratio']} |"
            )
    lines.append("")

    lines.append("## B. 闸门解除敏感性（leave-one-gate-out）")
    lines.append("")
    gv = gate_variants(df, horizon)
    lines.append("| 变体 | 天数 | 占样本 | 相对 Base | fwd 均值 | fwd 胜率 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in gv.iterrows():
        lines.append(
            f"| {r['variant']} | {r['days']} | {r['share_of_sample']:.1%} | "
            f"{r['vs_base_days']:+d} | {r[f'fwd{horizon}_mean']} | "
            f"{r[f'fwd{horizon}_win']} |"
        )
    lines.append("")

    lines.append("## D. TRANSITION 子状态稳健性（分年度）")
    lines.append("")
    ts = transition_stability(df, horizon)
    if ts.empty:
        lines.append("（无数据）")
    else:
        lines.append("| 子状态 | 天数 | 占 TRANSITION | fwd 均值 | fwd 胜率 | "
                     "胜率>50%年份 | 有数据年份 | 年度胜率区间 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for _, r in ts.iterrows():
            lines.append(
                f"| {r['substate']} | {r['days']} | {r['share_of_transition']:.1%} | "
                f"{r[f'fwd{horizon}_mean']} | {r[f'fwd{horizon}_win']} | "
                f"{r['years_win_gt50']} | {r['years_with_data']} | "
                f"{r['year_win_min']} ~ {r['year_win_max']} |"
            )
    lines.append("")
    if extra:
        lines.append(extra)
        lines.append("")
    return "\n".join(lines)
