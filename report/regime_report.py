"""Markdown 报告生成.

输出的不是"明天买什么"，而是：
    当前市场状态 / 最近状态切换 / 次级反应位置 / 线 / 量价 / 是否构成 risk gate。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from research.event_store import forward_returns, trend_changes


def render_report(df: pd.DataFrame, meta: dict, out_path: str | None = None) -> str:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last = df.iloc[-1]
    lines = [
        "# Dow Theory Regime Report",
        "",
        f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 参数版本: **{meta.get('params')}**",
        f"- 指数配对: {meta.get('pair', 'n/a')} ({meta.get('code_a', '')}/{meta.get('code_b', '')})",
        f"- 样本: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()} ({len(df)} 交易日)",
        "",
        "## 当前状态",
        "",
        "| 字段 | 值 |",
        "|---|---|",
        f"| primary_trend | **{last['primary_trend']}** |",
        f"| last_confirmed_swing | {last['last_confirmed_swing']} |",
        f"| dual_confirmation | {last['dual_confirmation']} |",
        f"| secondary_reaction | {last['secondary_reaction']} |",
        f"| retracement_ratio | {last['retracement_ratio']:.2%} |",
        f"| secondary_duration | {int(last['secondary_duration'])} 天 |",
        f"| line_state | {last['line_state']} |",
        f"| volume_state | {last['volume_state']} |",
        f"| dow_signal | **{last['dow_signal']}** |",
        f"| notes | {last['notes']} |",
        "",
        "## 结构值（已确认浪的高低点）",
        "",
        f"H1={last['H1']}  H2={last['H2']}  L1={last['L1']}  L2={last['L2']}",
        "",
        "## 引擎统计",
        "",
        f"- 浪数量: A={meta['swings_a']}  B={meta['swings_b']}",
        f"- 确认率: {meta['confirmation']['confirmed_rate']:.1%}, "
        f"平均确认延迟 {meta['confirmation']['mean_lag_days']} 天",
        f"- 成线天数: {meta['line_days']}，双指数同时突破天数: {meta['line_breakout_days']}",
        f"- 趋势分布: {meta['trend_distribution']}",
        f"- 信号分布: {meta['signal_distribution']}",
        "",
    ]

    chg = trend_changes(df)
    if not chg.empty:
        lines += ["## 最近趋势状态切换", "", "| 日期 | 从 | 到 | 前状态持续(自然日) | Dow 信号 |", "|---|---|---|---|---|"]
        for _, r in chg.tail(10).iterrows():
            lines.append(
                f"| {r['date'].date()} | {r['from']} | **{r['to']}** | {r['days_in_prev']} | {r['dow_signal']} |"
            )
        lines.append("")

    fr = forward_returns(df)
    if not fr.empty:
        lines += ["## 状态信息含量（条件统计，非回测）", ""]
        lines.append("| state | n | 占比 | fwd20均值 | fwd20胜率 | fwd60均值 | fwd60胜率 | fwd120均值 | fwd120胜率 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in fr.iterrows():
            lines.append(
                f"| {r['state']} | {r['n']} | {r['share']:.1%} | {r['fwd_20_mean']:.2%} | {r['fwd_20_win']:.1%} "
                f"| {r['fwd_60_mean']:.2%} | {r['fwd_60_win']:.1%} | {r['fwd_120_mean']:.2%} | {r['fwd_120_win']:.1%} |"
            )
        lines.append("")

    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
