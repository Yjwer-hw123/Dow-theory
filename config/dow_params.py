"""道氏理论量化引擎 —— 参数配置（冻结层）.

重要纪律
--------
V0.1 (`DOW_LITERAL_V01`) 是 **雷亚《道氏理论》原书复刻版**，所有参数必须
来自书中明确的量化锚点，不允许任何为了回测效果而做的调整。
修改它之前请先阅读 README 的 "三代演进" 章节。

书中锚点出处（刘志刚译《道氏理论（珍藏版）》，地震出版社）：
- 3%    → 第 13 章：反向运动超过平均价格指数的 3%，视为一个上升浪/下跌浪
- 33~66%→ 第 2 章、第 10 章：次级反应通常回调前期主要趋势行程的 1/3~2/3
- 2~3 周 → 第 2 章、第 14 章：线态窄幅区间盘整通常持续 2~3 周或更长
- 约 5%  → 第 14 章：线态窄幅区间盘整期间两个指数在约 5% 的窄幅区间内波动
- 双指数 → 第 12 章：两个平均价格指数必须互相确认（不要求同一天）
- HH/HL → 第 13 章：高点抬升且回调不破前低 = 牛市；LH/LL = 熊市
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class DowParams:
    """道氏理论引擎参数（不可变）."""

    name: str = "unnamed"

    # ---- 第 13 章：浪（Swing）定义 ----
    swing_pct: float = 0.03            # 反向运动超过 3% 视为一个新浪

    # ---- 第 12 章：双指数确认 ----
    confirmation_mode: str = "until_next_reversal"
    #   until_next_reversal : 另一指数在本指数下一个反向浪出现之前给出同向浪（V0 默认，无参数）
    #   lag_days            : 另一指数同向浪结束日落在 N 个交易日窗口内
    confirmation_lag_days: Optional[int] = None

    # ---- 第 10 章：次级反应 ----
    retrace_min: float = 0.33          # 次级反应经验下限
    retrace_max: float = 0.66          # 次级反应经验上限
    secondary_max_days: int = 90       # 超过此窗口仍未恢复，视为趋势可能改变（书中：通常 3 周~数月）

    # ---- 第 14 章：线态窄幅区间盘整 ----
    line_width_pct: float = 0.05       # 两个指数在约 5% 的窄幅区间内
    line_min_bars: int = 10            # 至少 2 周（10 个交易日）
    line_max_width_ratio: int = 2      # 允许放宽倍数上限（用于诊断，不改变是否成线）

    # ---- 第 13 章：结构比较的同级别过滤（结构规则，非数值参数） ----
    # True = 丢掉两端都被前一浪价格区间包含的"内含微浪"，只比较真正突破/破坏级别的浪
    filter_internal: bool = True

    # ---- 第 15 章：成交量 ----
    volume_window: int = 20            # 比较上升/下跌日均量的窗口（交易日）

    @property
    def is_literal(self) -> bool:
        """是否为完全原书参数（V0 冻结版）."""
        literal = DOW_LITERAL_V01
        return all(
            abs(getattr(self, f) - getattr(literal, f)) < 1e-9
            for f in ("swing_pct", "retrace_min", "retrace_max", "line_width_pct")
        ) and self.line_min_bars == literal.line_min_bars


# --------------------------------------------------------------------------
# V0.1 —— 原书复刻，禁止调优
# --------------------------------------------------------------------------
DOW_LITERAL_V01 = DowParams(
    name="Dow-Literal-V0.1",
    swing_pct=0.03,
    confirmation_mode="until_next_reversal",
    retrace_min=0.33,
    retrace_max=0.66,
    secondary_max_days=90,
    line_width_pct=0.05,
    line_min_bars=10,
    volume_window=20,
)


# --------------------------------------------------------------------------
# V1 候选：A 股适配（尚未验证，仅用于走 前推/样本外 敏感性研究）
# 极重要：这些不是"更好"，只是"待检验"。
# --------------------------------------------------------------------------
ASHARE_CANDIDATES: dict[str, DowParams] = {
    "ashare_swing025": replace(DOW_LITERAL_V01, name="A-share swing 2.5%", swing_pct=0.025),
    "ashare_swing035": replace(DOW_LITERAL_V01, name="A-share swing 3.5%", swing_pct=0.035),
    "ashare_swing040": replace(DOW_LITERAL_V01, name="A-share swing 4.0%", swing_pct=0.040),
    "ashare_line04": replace(DOW_LITERAL_V01, name="A-share line 4%", line_width_pct=0.04),
    "ashare_line06": replace(DOW_LITERAL_V01, name="A-share line 6%", line_width_pct=0.06),
    "ashare_lag10": replace(
        DOW_LITERAL_V01,
        name="A-share confirmation lag 10D",
        confirmation_mode="lag_days",
        confirmation_lag_days=10,
    ),
}


def get_params(version: str) -> DowParams:
    """按名称取参数集.

    version: "V0.1" / "literal" 或 ASHARE_CANDIDATES 中的键.
    """
    key = version.strip().lower()
    if key in ("v0.1", "v01", "literal", "dow-literal-v0.1"):
        return DOW_LITERAL_V01
    if version in ASHARE_CANDIDATES:
        return ASHARE_CANDIDATES[version]
    raise KeyError(f"unknown param version: {version} (known: v0.1, {list(ASHARE_CANDIDATES)})")
