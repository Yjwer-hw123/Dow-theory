"""核心类型定义.

所有模块只用这里的轻量 dataclass，避免 numpy/pandas 泄漏到接口里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Literal, Optional

Direction = Literal["UP", "DOWN"]


@dataclass(frozen=True)
class Swing:
    """一个道氏浪（第 13 章定义：反向运动 >= 3%）."""

    direction: Direction
    start_date: Date
    start_price: float
    end_date: Date
    end_price: float
    index: str = "A"
    provisional: bool = False   # True = 正在进行中的开放浪

    # ---- 防前瞻偏差（第二层）：极值日 ≠ 可知日 ----
    # `end_date` 是**极值点**当天，但这条浪要到价格从极值反向走够阈值
    # （第 13 章的 3%）之后才会被识别出来，两者可能相差数周。
    # 因此任何"这条浪何时可以进入结构"的判断都必须用 known_date，
    # 用 end_date 会让引擎在极值日当天就用上几周后才知道的信息。
    detected_date: Optional[Date] = None

    @property
    def known_date(self) -> Date:
        """市场最早知道这条浪存在的日期（识别日；缺失时退化为极值日兜底）."""
        return self.detected_date or self.end_date

    @property
    def move_pct(self) -> float:
        base = self.start_price
        if base <= 0:
            return 0.0
        return (self.end_price - self.start_price) / base

    @property
    def move_abs(self) -> float:
        return self.end_price - self.start_price

    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days

    def as_dict(self, bars: Optional[int] = None) -> dict:
        d = {
            "direction": self.direction,
            "start_date": self.start_date,
            "start_price": self.start_price,
            "end_date": self.end_date,
            "end_price": self.end_price,
            "move_pct": round(self.move_pct, 4),
            "index": self.index,
            "provisional": self.provisional,
        }
        if bars is not None:
            d["bars"] = bars
        return d


@dataclass
class ConfirmedSwing:
    """经另一个指数确认后的浪（第 12 章）."""

    swing: Swing
    confirmed_by: Optional[Swing] = None   # 确认它的另一指数浪
    lag_days: Optional[int] = None         # 主浪结束日与确认浪结束日的交易日间隔

    # ---- 防前瞻偏差的关键字段 ----
    # 第 12 章允许"另一指数稍后才确认"，但**研究者必须只在市场已经知道确认的那天**
    # 才能使用这条信息。因此：
    #     available_date = max(本浪的 known_date, 确认浪的 known_date)
    # 若最终没有确认，则 = 确认窗口关闭的**可知时点**；窗口尚未关闭时 = date_max()
    # （即"永远还不知道它会不会被确认"，因此不得使用）。
    #
    # 全部用 known_date 而非 end_date —— 这是 P10.1 修掉的第二层前瞻偏差：
    # 第一层是"用了另一指数几周后才给出的确认"，第二层是"把极值日当成了可知日"。
    available_date: Optional[Date] = None

    @property
    def confirmed(self) -> bool:
        return self.confirmed_by is not None

    @property
    def usable_date(self) -> Date:
        """可以进入结构的最早日期（缺失时退化为本浪的 known_date，仅用于兜底）."""
        return self.available_date or self.swing.known_date

    @property
    def direction(self) -> Direction:
        return self.swing.direction

    @property
    def high(self) -> float:
        return max(self.swing.start_price, self.swing.end_price)

    @property
    def low(self) -> float:
        return min(self.swing.start_price, self.swing.end_price)


TrendState = Literal["UP", "DOWN", "TRANSITION", "UNCONFIRMED"]
LineState = Literal["NONE", "FORMING", "ACCUMULATION", "DISTRIBUTION"]

# V0.1.1：OVERDUE 与 INVALIDATED 必须分开
#   OVERDUE     = 时间超期，但结构还没坏（还能算次级反应）
#   INVALIDATED = 已经吃掉了本段主要运动行程的起点 —— 它已经不是"次级反应"了
SecondaryState = Literal["NONE", "ACTIVE", "COMPLETED", "OVERDUE", "INVALIDATED"]
TrendIntegrity = Literal["INTACT", "DAMAGED", "UNKNOWN"]
ReversalCandidate = Literal["NONE", "BULL", "BEAR"]
VolumeState = Literal["HEALTHY", "WEAK", "EXHAUSTION", "INSUFFICIENT_DATA"]
DowSignal = Literal["RISK_ON", "RISK_OFF", "WAIT"]


@dataclass
class DowState:
    """每日输出的机器可读市场状态."""

    date: Optional[Date] = None
    params_name: str = ""

    primary_trend: TrendState = "UNCONFIRMED"
    last_confirmed_swing: Optional[str] = None

    # V0.1.1：把"历史上最后确认的主要趋势"与"这个趋势是否还完好"分开表达
    trend_integrity: TrendIntegrity = "UNKNOWN"
    reversal_candidate: ReversalCandidate = "NONE"

    secondary_reaction: SecondaryState = "NONE"
    retracement_ratio: float = 0.0          # 诊断值，允许 >1（表示已跌破本段行程起点）
    secondary_duration: int = 0

    # lineage（用于审计，回答"这条 reaction 到底挂在哪个 leg 上"）
    primary_leg_id: Optional[int] = None
    secondary_id: Optional[str] = None

    dual_confirmation: Literal["CONFIRMED", "DIVERGENCE", "NONE"] = "NONE"
    line_state: LineState = "NONE"

    volume_state: VolumeState = "INSUFFICIENT_DATA"
    dow_signal: DowSignal = "WAIT"

    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "date": self.date,
            "market_state": self.primary_trend,
            "trend_integrity": self.trend_integrity,
            "reversal_candidate": self.reversal_candidate,
            "last_confirmed_swing": self.last_confirmed_swing,
            "secondary_reaction": self.secondary_reaction,
            "primary_leg_id": self.primary_leg_id,
            "secondary_id": self.secondary_id,
            "retracement_ratio": round(self.retracement_ratio, 4),
            "secondary_duration": self.secondary_duration,
            "dual_confirmation": self.dual_confirmation,
            "line_state": self.line_state,
            "volume_state": self.volume_state,
            "dow_signal": self.dow_signal,
            "notes": "|".join(self.notes),
        }
