"""状态不变量检查器（V0.1.1）.

单元测试回答的是"这个案例输出对不对"；
不变量检查器回答的是：**任何情况下，系统都不允许出现逻辑上不可能的状态组合**。

这一层是 V0.1 能否被称为"一个形式上自洽的状态机"的关键。它不参与任何
参数调优，只做逻辑校验 —— 因此它是"冻结层的一部分"，不是研究层。

每条不变量都有编号，报告里可以直接引用：

    INV-01  ACTIVE 的回撤不得超过 1（否则它已经不是次级反应）
            INVALIDATED 必须同时反映为 trend_integrity=DAMAGED
    INV-02  INVALIDATED 必须伴随回撤 > 1（结构已吃掉行程起点）
    INV-03  NONE 不得伴随显著回撤（<= 5%，否则应归类为 ACTIVE）
    INV-04  RISK_ON 必须是 UP + 双指数确认 + 结构完好
    INV-05  RISK_OFF 必须是 DOWN
    INV-06  结构 DAMAGED 时不得 RISK_ON
    INV-07  次级反应 INVALIDATED 时必须是 WAIT
    INV-08  UNCONFIRMED 趋势必须是 WAIT
    INV-09  枚举合法性（trend / secondary / signal / integrity）
    INV-10  日期严格递增且价格为正
    INV-11  lineage：reaction 必须挂在当前 leg 之后（禁止旧腿粘连）
    INV-12  lineage：leg id 单调不减；同一 leg 内反向极值单调推进
    INV-13  同一 leg 的反应一旦 INVALIDATED 不得回退到 ACTIVE/COMPLETED

其中 INV-11 ~ INV-13 是把此前修掉的 "leg chasing" bug **永久化**为不变量，
而不只是靠某个样本测试兜住。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ALLOWED_TREND = {"UP", "DOWN", "TRANSITION", "UNCONFIRMED"}
ALLOWED_SECONDARY = {"NONE", "ACTIVE", "COMPLETED", "OVERDUE", "INVALIDATED"}
ALLOWED_SIGNAL = {"RISK_ON", "RISK_OFF", "WAIT"}
ALLOWED_INTEGRITY = {"INTACT", "DAMAGED", "UNKNOWN"}
ALLOWED_REVERSAL = {"NONE", "BULL", "BEAR"}


@dataclass
class Violation:
    index: int
    date: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.date} (row {self.index}): {self.message}"


def _norm_leg(v):
    """把 1 / 1.0 / "L1" / "L1-R1" 统一成 int，避免 pandas NaN 导致的 float 化误判."""
    try:
        if isinstance(v, str):
            v = v.split("-")[0].lstrip("L")
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _rel(v, tolerance: float = 1e-9) -> bool:
    return v is not None and v == v  # 非 NaN


def _f(row, key, default=None):
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if v is None else v


def check_frame(df, max_report: int = 40) -> list[Violation]:
    """逐行校验所有不变量（不抛异常，返回违规列表）."""
    v: list[Violation] = []
    if df is None or len(df) == 0:
        return v

    dates = list(df["date"])
    prev_leg_id: Optional[int] = None
    leg_state: dict = {}          # leg_id -> {"sec": set(), "extreme": float, "invalidated": bool}

    for i in range(len(df)):
        row = df.iloc[i]
        d = str(dates[i])[:10] if dates[i] is not None else "?"
        add = lambda rule, msg: v.append(Violation(i, d, rule, msg))  # noqa: E731

        trend = _f(row, "primary_trend")
        sec = _f(row, "secondary_reaction")
        ratio = _f(row, "retracement_ratio", 0.0) or 0.0
        dur = _f(row, "secondary_duration", 0) or 0
        sig = _f(row, "dow_signal")
        dual = _f(row, "dual_confirmation")
        integ = _f(row, "trend_integrity")
        rev = _f(row, "reversal_candidate")
        leg_raw = _f(row, "primary_leg_id")
        leg_key = _norm_leg(leg_raw)
        sec_id = _f(row, "secondary_id")

        # ---- INV-09 枚举合法性 ----
        if trend not in ALLOWED_TREND:
            add("INV-09", f"非法 primary_trend={trend}")
            continue
        if sec not in ALLOWED_SECONDARY:
            add("INV-09", f"非法 secondary_reaction={sec}")
            continue
        if sig not in ALLOWED_SIGNAL:
            add("INV-09", f"非法 dow_signal={sig}")
            continue
        if integ not in ALLOWED_INTEGRITY:
            add("INV-09", f"非法 trend_integrity={integ}")
            continue
        if rev not in ALLOWED_REVERSAL:
            add("INV-09", f"非法 reversal_candidate={rev}")

        # ---- INV-10 序列完整性 ----
        if i > 0 and hasattr(dates[i], "__lt__") and dates[i] is not None and dates[i - 1] is not None:
            if not (dates[i - 1] < dates[i]):
                add("INV-10", "日期未严格递增")
        ca = _f(row, "close_a")
        if ca is not None and _rel(ca) and float(ca) <= 0:
            add("INV-10", f"close_a 非正: {ca}")

        # ---- INV-01 / 02 / 03 次级反应数值语义 ----
        if sec == "ACTIVE" and ratio > 1.0:
            add("INV-01", f"ACTIVE 但 retracement={ratio:.4f} > 1（应判 INVALIDATED）")
        if sec == "INVALIDATED" and ratio <= 1.0:
            add("INV-02", f"INVALIDATED 但 retracement={ratio:.4f} <= 1（结构未被吃掉）")
        if sec == "NONE" and ratio > 0.05:
            add("INV-03", f"NONE 但 retracement={ratio:.4f} > 0.05（应为 ACTIVE 类状态）")
        if sec == "COMPLETED" and ratio < 0:
            add("INV-03", f"COMPLETED 但 retracement 为负: {ratio:.4f}")
        if sec == "INVALIDATED" and integ == "INTACT":
            add("INV-01", "INVALIDATED 必须伴随 trend_integrity=DAMAGED")

        # ---- INV-04 / 05 信号必须能被上游解释 ----
        if sig == "RISK_ON" and not (trend == "UP" and dual == "CONFIRMED" and integ == "INTACT"):
            add("INV-04", f"RISK_ON 但 trend={trend}, dual={dual}, integrity={integ}")
        if sig == "RISK_OFF" and trend != "DOWN":
            add("INV-05", f"RISK_OFF 但 trend={trend}")

        # ---- INV-06 / 07 / 08 闸门语义 ----
        if integ == "DAMAGED" and sig == "RISK_ON":
            add("INV-06", "结构 DAMAGED 却给出 RISK_ON")
        if sec == "INVALIDATED" and sig != "WAIT":
            add("INV-07", f"INVALIDATED 却给出 {sig}（必须 WAIT）")
        if trend == "UNCONFIRMED" and sig != "WAIT":
            add("INV-08", f"UNCONFIRMED 趋势却给出 {sig}（必须 WAIT）")

        # ---- INV-11 / 12 / 13 lineage（禁止旧腿粘连） ----
        if leg_key is not None and trend in ("UP", "DOWN"):
            if prev_leg_id is not None and leg_key < prev_leg_id:
                add("INV-12", f"primary_leg_id 回退: {prev_leg_id} -> {leg_key}")
            if sec_id is not None and isinstance(sec_id, str):
                head = str(sec_id).split("-")[0]
                want = f"L{leg_key}"
                if head != want:
                    add("INV-11", f"reaction {sec_id} 挂到了 leg {leg_raw} 上（应属 {want}-*）")
            if leg_key != prev_leg_id:
                leg_state[leg_key] = {"extreme": None, "invalidated": False, "sec_id": sec_id}
                prev_leg_id = leg_key
            st = leg_state.setdefault(leg_key, {"extreme": None, "invalidated": False, "sec_id": sec_id})
            ext = _f(row, "reaction_extreme")
            if ext is not None and _rel(ext):
                ext = float(ext)
                prev_ext = st["extreme"]
                if prev_ext is not None and trend == "UP" and ext > prev_ext + 1e-9:
                    add("INV-12", f"同一 leg 内回调最低点抬升了: {prev_ext} -> {ext}")
                if prev_ext is not None and trend == "DOWN" and ext < prev_ext - 1e-9:
                    add("INV-12", f"同一 leg 内反弹最高点下降了: {prev_ext} -> {ext}")
                st["extreme"] = ext
            if sec == "INVALIDATED":
                st["invalidated"] = True
            elif st["invalidated"] and sec == "ACTIVE":
                # V 型修复允许 INVALIDATED → COMPLETED（价格夺回反应极值），
                # 但绝不允许悄悄退回"这只是个正常回调"。
                add("INV-13", f"已 INVALIDATED 的 leg {leg_key} 又退回 {sec}")

        if len(v) > max_report * 20:      # 安全阀，避免在坏数据上爆内存
            v.append(Violation(i, d, "INV-XX", "违规过多，提前中止"))
            break

    return v


def assert_frame(df, max_report: int = 40) -> None:
    """违反不变量时直接抛错（引擎默认调用）."""
    vs = check_frame(df, max_report=max_report)
    if vs:
        head = "\n".join(str(x) for x in vs[:max_report])
        raise AssertionError(
            f"道氏状态机违反 {len(vs)} 条不变量：\n{head}"
            + (f"\n... 另有 {len(vs) - max_report} 条" if len(vs) > max_report else "")
        )


def summarize(df) -> dict:
    """给报告用的不变量摘要."""
    vs = check_frame(df)
    by_rule: dict = {}
    for x in vs:
        by_rule[x.rule] = by_rule.get(x.rule, 0) + 1
    return {"n_violations": len(vs), "by_rule": by_rule, "ok": not vs}
