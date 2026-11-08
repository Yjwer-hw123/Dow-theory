"""V0.1 —— 原书复刻适配器（禁止调优）.

这一层存在的意义只有一个：让"我有没有正确复刻雷亚"这个问题可以被回答。

用法：
    from adapters.dow_literal import run_literal
    df, meta = run_literal(df_300, df_500)
"""

from __future__ import annotations

import pandas as pd

from config.dow_params import DOW_LITERAL_V01
from core.regime_engine import analyze_pair, latest_state

LITERAL = DOW_LITERAL_V01


def run_literal(df_a: pd.DataFrame, df_b: pd.DataFrame, with_volume: bool = True):
    """用冻结参数跑一遍."""
    assert LITERAL.is_literal, "参数已被修改，不再是原书复刻版！"
    return analyze_pair(df_a, df_b, LITERAL, with_volume=with_volume)


def freeze_report() -> str:
    """输出冻结参数列表，便于写进研究报告."""
    p = LITERAL
    lines = [
        f"# 冻结参数: {p.name}",
        "",
        "| 参数 | 值 | 来源 |",
        "|---|---|---|",
        f"| swing_pct | {p.swing_pct:.1%} | 第13章 3% 反向运动 |",
        f"| confirmation_mode | {p.confirmation_mode} | 第12章 不要求同日确认 |",
        f"| retrace band | {p.retrace_min:.0%} ~ {p.retrace_max:.0%} | 第2/10章 1/3~2/3 |",
        f"| secondary_max_days | {p.secondary_max_days} | 第10章 通常 3周~数月 |",
        f"| line_width_pct | {p.line_width_pct:.0%} | 第14章 约 5% 窄幅区间 |",
        f"| line_min_bars | {p.line_min_bars} | 第14章 2~3 周 |",
        f"| volume_window | {p.volume_window} | 第15章 量价关系比较窗口 |",
    ]
    return "\n".join(lines)


def latest(df_a: pd.DataFrame, df_b: pd.DataFrame, with_volume: bool = True):
    df, meta = run_literal(df_a, df_b, with_volume=with_volume)
    return latest_state(df, LITERAL.name), meta
