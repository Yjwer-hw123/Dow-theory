"""趋势状态机 —— HH/HL 与 LH/LL（第 13 章）.

书中定义：
    "如果上升浪能连续不断地突破前期的高点，而随后的下跌能在前期低点之上结束，
     这样的市场就是牛市趋势。反之，若上升浪不能超过前期的高点，而随后下跌的
     低点却比前期的低点低，这样的市场就是熊市趋势。"

设计要点
--------
* **只允许三种状态：UP / DOWN / TRANSITION**（外加数据不足时的 UNCONFIRMED）。
  道氏理论并 *没有* 要求市场每天必须是多或空 —— 强行二分是绝大多数技术指标的毛病。
* 只使用 **已确认**（另一个指数确认过）的浪，避免单指数假象。
* 趋势改变必须走完：
      原趋势 → 失败的延续 → 结构破坏 → 确认 → 状态切换
  因此这里不存在"跌 5% 转熊"这种东西。
"""

from __future__ import annotations

from datetime import date as Date
from typing import Optional

from .types import ConfirmedSwing, ReversalCandidate, Swing, TrendIntegrity, TrendState


def filter_internal_swings(swings: list[Swing]) -> list[Swing]:
    """过滤"同级别内含微浪"（结构规则，不改任何数值参数）.

    为什么必须有这一步
    ------------------
    第 13 章给了 3% 的浪阈值，但现实里任何阈值都会产生这样的东西：

        UP 4713 → 5059        （主要运动的一段）
        DOWN 5059 → 4529      （次级反应）
        UP 4529 → 4739        ← 完全落在上一段 DOWN 的价格区间内
        DOWN 4739 → 4543
        UP 4543 → 4741        ← 同样被包含

    这些"内含浪"既没创新高也没创新低，如果拿它们去比较高低点，会出现
    "H2=4741 比 H1=4739 高了 0.04% 所以还是牛市"这种荒谬结论 ——
    而实际上市场早已经跌破了 6 月低点（4529）。

    汉密尔顿在图上永远是 **相同级别的浪互相比较**。所以这里把两端都落在
    前一浪价格区间内的浪丢掉，只保留真正的"突破/破坏"级别的运动。
    """
    kept: list[Swing] = []
    for s in sorted(swings, key=lambda x: x.end_date):
        if not kept:
            kept.append(s)
            continue
        prev = kept[-1]
        contained = (max(s.start_price, s.end_price) <= max(prev.start_price, prev.end_price)
                     and min(s.start_price, s.end_price) >= min(prev.start_price, prev.end_price))
        if contained:
            continue
        kept.append(s)
    return kept


def build_structure(confirmed_up: list[Swing], confirmed_down: list[Swing]) -> dict:
    """从已确认的上升浪/下跌浪中取出最近的两个高点与两个低点."""
    struct = {"H2": None, "H1": None, "L2": None, "L1": None}
    highs = [s.end_price for s in confirmed_up]
    lows = [s.end_price for s in confirmed_down]
    if len(highs) >= 2:
        struct["H2"], struct["H1"] = highs[-1], highs[-2]
    elif len(highs) == 1:
        struct["H2"] = highs[-1]
    if len(lows) >= 2:
        struct["L2"], struct["L1"] = lows[-1], lows[-2]
    elif len(lows) == 1:
        struct["L2"] = lows[-1]
    return struct


def classify(struct: dict) -> TrendState:
    """HH+HL → UP；LH+LL → DOWN；混合 → TRANSITION；数据不足 → UNCONFIRMED."""
    h1, h2 = struct.get("H1"), struct.get("H2")
    l1, l2 = struct.get("L1"), struct.get("L2")
    if h1 is None or h2 is None or l1 is None or l2 is None:
        return "UNCONFIRMED"
    higher_high = h2 > h1
    higher_low = l2 > l1
    if higher_high and higher_low:
        return "UP"
    if not higher_high and not higher_low:
        return "DOWN"
    return "TRANSITION"


def assess_integrity(
    struct: dict,
    trend: TrendState,
    corrective_low: Optional[float] = None,
    corrective_high: Optional[float] = None,
    ref_low: Optional[float] = None,
    ref_high: Optional[float] = None,
) -> TrendIntegrity:
    """趋势完好度：**当前**价格是否还守住了本段趋势的高低点结构.

    参照点可以显式指定（`ref_low` / `ref_high`），默认取 struct 的最新已确认
    低点 / 高点。引擎里传入的是"本段行程起点与最新结构水平中更严格的一侧"，
    以保证 trend_integrity 与 secondary_reaction 永远自洽（见 INV-01）。

    与 secondary_reaction 用的是同一个参照点，但语义不同：

        UP  : 只要价格没有跌破最近一个已确认低点 L2，结构就算 INTACT
        DOWN: 只要价格没有涨破最近一个已确认高点 H2，结构就算 INTACT
        TRANSITION / 数据不足 → UNKNOWN（拒绝强行二分）

    这样"历史上最后确认的主要趋势是 UP"与"这个 UP 现在还完好不存在"
    就是两个可以独立读取的字段。
    """
    if trend == "UP":
        ref = struct.get("L2") if ref_low is None else ref_low
        if ref is None or corrective_low is None:
            return "UNKNOWN"
        return "INTACT" if corrective_low >= ref else "DAMAGED"
    if trend == "DOWN":
        ref = struct.get("H2") if ref_high is None else ref_high
        if ref is None or corrective_high is None:
            return "UNKNOWN"
        return "INTACT" if corrective_high <= ref else "DAMAGED"
    return "UNKNOWN"


def reversal_candidate(trend: TrendState, integrity: TrendIntegrity) -> ReversalCandidate:
    """结构被破坏 => 候选反转方向（但**尚未确认**）.

    重要：候选 ≠ 确认。`primary_trend` 仍然保持原值，
    因为道氏的状态切换必须走完 失败延续 → 结构破坏 → 确认 这条链。
    """
    if integrity != "DAMAGED":
        return "NONE"
    if trend == "UP":
        return "BEAR"
    if trend == "DOWN":
        return "BULL"
    return "NONE"


def trend_asof(confirmed_swings: list[ConfirmedSwing], when: Date) -> tuple[TrendState, dict, Optional[str]]:
    """截至 `when` 的趋势判定（仅用已确认、且市场在 `when` 之前已经知道的浪）.

    注意判断条件是 `usable_date <= when`，不是 `end_date <= when` ——
    后者会把"极值日"当成"可知日"，让引擎用到几周后才浮现的浪。
    """
    ups = [c.swing for c in confirmed_swings
           if c.confirmed and c.swing.direction == "UP" and c.usable_date <= when]
    downs = [c.swing for c in confirmed_swings
             if c.confirmed and c.swing.direction == "DOWN" and c.usable_date <= when]
    struct = build_structure(ups, downs)
    state = classify(struct)
    last = None
    ordered = sorted(ups + downs, key=lambda s: s.end_date)
    if ordered:
        last = f"{ordered[-1].direction}@{ordered[-1].end_date}"
    return state, struct, last


def detect_changes(states: list[tuple[Date, TrendState]]) -> list[dict]:
    """列出状态切换事件（原趋势→新趋势），便于构建事件数据库."""
    changes = []
    prev: tuple[Date, TrendState] | None = None
    for d, st in states:
        if st == "UNCONFIRMED":
            continue
        if prev is None or prev[1] != st:
            if prev is not None:
                changes.append(
                    {"date": d, "from": prev[1], "to": st, "prev_change": prev[0]}
                )
            prev = (d, st)
        else:
            prev = (d, st)
    return changes
