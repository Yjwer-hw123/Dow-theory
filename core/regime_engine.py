"""Dow Regime Engine —— 把所有模块合成每日市场状态（第 2 章 + 第 13 章）.

架构（严格按双指数并行的原书逻辑）：

        Index A Swing Engine ──┐
                               ├── Confirmation Engine ──> Confirmed Swing
        Index B Swing Engine ──┘                                │
                                        ┌───────────────────────┼──────────────┐
                                        ▼                       ▼              ▼
                                Trend Engine          Reaction Engine     Line Engine
                                        └───────────────────────┼──────────────┘
                                                                ▼
                                                        Volume Engine
                                                                ▼
                                                          Dow Regime

重要：**第一阶段不打分**。
输出的是状态与事件，不是一个 0~100 的分数 —— 因为一旦打分，就会迅速退化成
"看起来很科学、实际上参数高度自由"的技术指标。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .confirmation import confirm_primary, confirmation_summary
from .invariants import check_frame, summarize as inv_summary
from .line_detector import combine_line, detect_line_state
from .secondary_reaction import evaluate_secondary
from .swing_detector import detect_swings_df
from .trend_engine import (
    assess_integrity,
    build_structure,
    classify,
    filter_internal_swings,
    reversal_candidate,
)
from .types import DowState
from .volume_analysis import classify_volume, up_down_volume_arr


def _as_date(v):
    """统一转成 datetime.date（兼容 pandas Timestamp / datetime / date）."""
    if hasattr(v, "date") and not isinstance(v, __import__("datetime").date):
        return v.date()
    return v


class _AsConfirmed:
    """把单指数浪包装成"视为已确认"，专供消融实验的 A 变体使用.

    不做双指数确认时，结构应当完全由**指数 A 自己的浪**构建 —— 否则所谓
    "只有 HH/HL"其实已经偷偷包含了双指数确认（因为默认的 `completed` 列表
    本身就是双指数确认后的结果），消融表就会失真。
    """

    __slots__ = ("swing", "confirmed", "lag_days", "available_date", "usable_date")

    def __init__(self, s):
        self.swing = s
        self.confirmed = True
        self.lag_days = 0
        # 单指数结构：不做双指数确认，但仍必须等这条浪被**识别**出来
        # （known_date），而不是它在极值点就"可知"。
        self.available_date = s.known_date
        self.usable_date = s.known_date


def analyze_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    params,
    calendar=None,
    with_volume: bool = True,
    strict_invariants: bool = True,
    structure_source: str = "dual",
) -> tuple[pd.DataFrame, dict]:
    """对一对指数跑完整的道氏引擎.

    Parameters
    ----------
    strict_invariants : 违反 core/invariants.py 中的任何一条时直接抛 AssertionError。
        默认开启 —— V0.1 应当是"形式上自洽的状态机"，不允许带着逻辑矛盾往外跑。
    structure_source : 趋势结构（HH/HL）用哪种浪来构建。
        - `"dual"`（默认，V0.1）：只用**双指数确认过**的浪 → 结构本身已内含第 12 章确认。
        - `"index_a"`：只用指数 A 自己的浪，跳过确认 → 供消融实验 A 变体
          （"只有 HH/HL"）使用。

    Returns
    -------
    (daily DataFrame, meta dict)
    """
    if df_a.empty or df_b.empty or len(df_a) != len(df_b):
        raise ValueError("两个指数数据必须等长且非空（请先对齐交易日）")

    dates = df_a["date"].dt.date.tolist()
    highs_a = df_a["high"].astype(float).tolist()
    lows_a = df_a["low"].astype(float).tolist()
    closes_a = df_a["close"].astype(float).tolist()
    highs_b = df_b["high"].astype(float).tolist()
    lows_b = df_b["low"].astype(float).tolist()
    closes_b = df_b["close"].astype(float).tolist()
    # 量价模块复用：预先提取一次，避免主循环里反复做 DataFrame 切片（O(n²) 来源）
    opens_a = df_a["open"].astype(float).tolist()
    volumes_a = df_a["volume"].astype(float).tolist()

    # ---------- 1. 浪识别（两个指数各自独立） ----------
    swings_a = detect_swings_df(df_a, pct=params.swing_pct, index="A")
    swings_b = detect_swings_df(df_b, pct=params.swing_pct, index="B")

    # ---------- 2. 双指数确认 ----------
    confirmed_a = confirm_primary(
        swings_a, swings_b,
        mode=params.confirmation_mode,
        lag_days=params.confirmation_lag_days,
        calendar=calendar,
    )
    if structure_source == "index_a":
        # 消融用：不做双指数确认，结构完全由指数 A 自己的浪构建
        completed = [_AsConfirmed(s) for s in swings_a
                     if not getattr(s, "provisional", False)]
    else:
        completed = [c for c in confirmed_a if not c.swing.provisional]
    # 按"市场可知日期"排序 —— 这个列表的消费顺序决定了引擎有没有前瞻偏差
    completed.sort(key=lambda c: c.usable_date)

    # ---------- 3. 逐日推进 ----------
    rows: list[dict] = []
    ptr = 0
    prior: list = []       # 已确认的同级别浪（历史）
    pending: list = []     # 当日新完成、待合并的浪
    ups_new: list = []     # 截至当日的上升浪列表（升序）
    downs_new: list = []
    filter_internal = getattr(params, "filter_internal", True)

    # ---- lineage：每一段主要运动行程及其附属次级反应 ----
    leg_id = 0                    # 主要运动行程编号 L{id}
    leg_trend: Optional[str] = None
    leg_origin_date = None        # 本段行程起点（牛市=前低 / 熊市=前高）
    leg_origin_price = None
    leg_extreme_date = None       # 本段行程终点反向点，也是次级反应的起点
    leg_extreme_price = None
    leg_end_idx = None
    ups_at_leg = downs_at_leg = 0
    corr_low = None               # 牛市：回调最低价
    corr_high = None              # 熊市：反弹最高价
    corr_extreme_date = None
    invalidated_date = None       # 首次吃掉本段行程起点的日期
    leg_key: tuple = ()           # struct 参照对，用来判断是否已换 leg
    resumed = False
    last_swing_label: Optional[str] = None
    dual_state = "NONE"
    prev_trend: Optional[str] = None

    def _new_leg(trend: str, key: tuple, origin_date, origin_price,
                 extreme_date, extreme_price, i: int) -> None:
        """开启一段新的主要运动行程，并把旧的次级反应彻底作废.

        行程完全由 **struct（最新两个已确认高低点）** 定义：

            UP   : L2 → H2        DOWN : H2 → L2

        这样次级反应、trend_integrity、趋势分类三者共用同一个参照点，不会出现
        "指标 A 用一个参照、指标 B 用另一个参照"的自相矛盾（这是 V0.1.1 里
        被不变量检查器抓出来的第三个真 bug）。
        """
        nonlocal leg_id, leg_trend, leg_origin_date, leg_origin_price
        nonlocal leg_extreme_date, leg_extreme_price, leg_end_idx
        nonlocal ups_at_leg, downs_at_leg, corr_low, corr_high, corr_extreme_date
        nonlocal invalidated_date, leg_key
        leg_key = key
        leg_id += 1
        leg_trend = trend
        leg_origin_date, leg_origin_price = origin_date, origin_price
        leg_extreme_date, leg_extreme_price = extreme_date, extreme_price
        leg_end_idx = i
        ups_at_leg, downs_at_leg = len(ups_new), len(downs_new)
        corr_low = corr_high = None
        corr_extreme_date = None
        invalidated_date = None

    for i, d in enumerate(dates):
        # 加入当日"市场已经知道"的浪。
        # 注意用的是 usable_date（= max(本浪结束日, 确认浪结束日)），不是本浪结束日 ——
        # 用后者会引入前瞻偏差：引擎会在 t 日就用到指数 B 几个月后才给出的确认。
        while ptr < len(completed) and completed[ptr].usable_date <= d:
            c = completed[ptr]
            s = c.swing
            dual_state = "CONFIRMED" if c.confirmed else "DIVERGENCE"
            last_swing_label = f"{s.direction}@{s.end_date}"
            if c.confirmed:
                pending.append(s)
            ptr += 1

        if pending:
            all_raw = prior + pending
            merged = filter_internal_swings(sorted(all_raw, key=lambda s: s.end_date)) \
                if filter_internal else sorted(all_raw, key=lambda s: s.end_date)
            prior, pending = merged, []
            ups_new = [s for s in merged if s.direction == "UP"]
            downs_new = [s for s in merged if s.direction == "DOWN"]

        struct = build_structure(ups_new, downs_new)
        trend = classify(struct)

        # ---- 主要运动行程：与当前趋势同向的那一段已确认浪 ----
        # 直接用浪自身的 start_price → end_price，天然保证起点在时间上早于终点
        # （旧实现分别从 ups/downs 列表里取两个不同的浪，会出现"起点日期晚于
        #   终点日期"的倒挂 leg，这是 V0.1.1 抓到的第三个真 bug）。
        if trend == "UP" and ups_new:
            last = ups_new[-1]
            key = (trend, last.start_date, last.start_price, last.end_date, last.end_price)
            if key != leg_key:
                _new_leg("UP", key, last.start_date, last.start_price,
                         last.end_date, last.end_price, i)
        elif trend == "DOWN" and downs_new:
            last = downs_new[-1]
            key = (trend, last.start_date, last.start_price, last.end_date, last.end_price)
            if key != leg_key:
                _new_leg("DOWN", key, last.start_date, last.start_price,
                         last.end_date, last.end_price, i)

        # ---- 次级反应：跟踪反向极值 ----
        # 反向极值必须在**趋势状态切换时**清零，否则上一段趋势遗留的极值会污染
        # 新趋势的结构完好度判断（INV-04 靠这里兜住）。
        if trend != prev_trend:
            corr_low = corr_high = None
            corr_extreme_date = None
        prev_trend = trend

        resumed = False
        if trend == "UP":
            if corr_low is None or lows_a[i] < corr_low:
                corr_low, corr_extreme_date = lows_a[i], d
            resumed = leg_extreme_price is not None and highs_a[i] >= leg_extreme_price
        elif trend == "DOWN":
            if corr_high is None or highs_a[i] > corr_high:
                corr_high, corr_extreme_date = highs_a[i], d
            resumed = leg_extreme_price is not None and lows_a[i] <= leg_extreme_price

        # ---- 结构参照点：取本段行程起点与最新结构水平中"更严格"的一侧 ----
        # UP   ：ref = max(行程起点低点, 最新已确认低点 L2)
        # DOWN ：ref = min(行程起点高点, 最新已确认高点 H2)
        # 这样保证 "跌破参照" ⟹ 一定同时反映为 DAMAGED 与 INVALIDATED，
        # 两个字段永远不会互相打架（不变量 INV-01）。
        ref_low = ref_high = None
        if trend == "UP":
            cand = [p for p in (leg_origin_price, struct.get("L2")) if p is not None]
            ref_low = max(cand) if cand else None
        elif trend == "DOWN":
            cand = [p for p in (leg_origin_price, struct.get("H2")) if p is not None]
            ref_high = min(cand) if cand else None

        integrity = assess_integrity(struct, trend, corrective_low=corr_low,
                                     corrective_high=corr_high,
                                     ref_low=ref_low, ref_high=ref_high)
        rev_candidate = reversal_candidate(trend, integrity)

        # ---- 次级反应 ----
        sec = {"state": "NONE", "retracement": 0.0, "duration": 0, "note": ""}
        if trend in ("UP", "DOWN") and leg_extreme_price is not None and leg_origin_price is not None \
                and leg_trend == trend:
            if trend == "UP":
                corrective = corr_low
                hl_intact = corrective is None or ref_low is None or corrective >= ref_low
            else:
                corrective = corr_high
                hl_intact = corrective is None or ref_high is None or corrective <= ref_high
            if corrective is not None:
                # 回撤的分母端点必须用**与 trend_integrity 同一个参照**，
                # 否则会出现 ratio<1 却判 INVALIDATED 的自相矛盾（INV-02）。
                sec = evaluate_secondary(
                    trend=trend,
                    prev_low=ref_low if trend == "UP" else leg_extreme_price,
                    prev_high=leg_extreme_price if trend == "UP" else ref_high,
                    corrective_low=corrective,
                    corrective_start=leg_extreme_date if leg_extreme_date is not None else d,
                    today=d,
                    params=params,
                    hl_intact=hl_intact,
                    resumed=resumed,
                )
                if sec["state"] == "INVALIDATED" and invalidated_date is None:
                    invalidated_date = d

        # ---- 线 ----
        # 用预提取列表 + combine_line：与 detect_line_pair 输出完全一致，
        # 但不再在主循环里重建 tolist()（那是 O(n²)，21 年数据要 26 秒）。
        line = combine_line(
            detect_line_state(None, highs_a, lows_a, closes_a, i, params),
            detect_line_state(None, highs_b, lows_b, closes_b, i, params),
        ) if i >= params.line_min_bars else {
            "line_state": "NONE", "line_breakout": "NONE", "line_width_a": None,
            "line_width_b": None, "line_active": False, "line_duration": 0,
            "line_breakout_a": "NONE", "line_breakout_b": "NONE",
        }

        # ---- 量价 ----
        if with_volume and i >= max(5, params.volume_window // 3):
            vol = up_down_volume_arr(opens_a, closes_a, volumes_a, i,
                                     window=params.volume_window)
            p20 = float(closes_a[i] / closes_a[i - 20] - 1) if i >= 20 else None
            vstate = classify_volume(trend, vol, p20)
        else:
            vol, vstate = {"ratio": None, "vol_trend": None}, "INSUFFICIENT_DATA"

        # ---- 合成信号（V0 只有几条硬规则，没有打分） ----
        # 闸门顺序严格对应原书：结构完好 → 才谈趋势 → 才谈次级反应位置。
        notes = []
        blocked = sec["state"] in ("OVERDUE", "INVALIDATED")
        if trend == "UP" and dual_state == "CONFIRMED" and integrity == "INTACT" and not blocked:
            signal = "RISK_ON"
        elif trend == "DOWN" and dual_state == "CONFIRMED" and integrity == "INTACT":
            signal = "RISK_OFF"
        else:
            signal = "WAIT"
            if trend == "UP" and sec["state"] == "OVERDUE":
                notes.append("correction overdue (leg origin intact) -> withhold")
            if trend == "UP" and sec["state"] == "INVALIDATED":
                notes.append("secondary reaction invalidated: prior trend-leg origin consumed")
            if trend == "TRANSITION":
                notes.append("mixed HH/LL structure")
            if dual_state == "DIVERGENCE":
                notes.append("latest swing not confirmed by second index")
        if integrity == "DAMAGED":
            notes.append(f"trend integrity DAMAGED (reversal candidate={rev_candidate}, unconfirmed)")
        if line.get("line_breakout") == "UP_CONFIRMED":
            notes.append("line breakout up (accumulation acc. ch.14)")
        if line.get("line_breakout") == "DOWN_CONFIRMED":
            notes.append("line breakout down (distribution acc. ch.14)")
            if signal == "RISK_ON":
                signal = "WAIT"
                notes.append("downside line breakout overrides RISK_ON")

        # ---- 空仓归因（P10.1-A）：这天为什么不是 RISK_ON ----
        # 按原书闸门顺序取**第一个**阻断原因：有没有趋势 → 结构 → 确认 → 位置 → 线。
        # 越靠前的越根本，所以先判定它，归因表才能回答"谁在拒绝"。
        blocker = None
        if signal != "RISK_ON":
            if trend == "UNCONFIRMED":
                blocker = "NO_TREND_UNCONFIRMED"
            elif trend == "TRANSITION":
                blocker = "TRANSITION"
            elif trend == "DOWN":
                blocker = "TREND_DOWN"
            elif integrity == "DAMAGED":
                blocker = "STRUCTURE_DAMAGED"
            elif dual_state != "CONFIRMED":
                blocker = "NO_DUAL_CONFIRMATION"
            elif sec["state"] == "INVALIDATED":
                blocker = "SECONDARY_INVALIDATED"
            elif sec["state"] == "OVERDUE":
                blocker = "SECONDARY_OVERDUE"
            elif line.get("line_breakout") == "DOWN_CONFIRMED":
                blocker = "LINE_DOWN_OVERRIDE"
            else:
                blocker = "OTHER"

        rows.append(
            {
                "date": d,
                "close_a": closes_a[i],
                "close_b": closes_b[i],
                "primary_trend": trend,
                "trend_integrity": integrity,
                "reversal_candidate": rev_candidate,
                "last_confirmed_swing": last_swing_label,
                "dual_confirmation": dual_state,
                "secondary_reaction": sec["state"],
                "retracement_ratio": sec["retracement"],
                "secondary_duration": sec["duration"],
                "secondary_note": sec["note"],
                # ---- lineage（审计用：这条 reaction 到底挂在哪个 leg 上） ----
                "primary_leg_id": leg_id if leg_end_idx is not None else None,
                "leg_origin_date": leg_origin_date,
                "leg_origin_price": leg_origin_price,
                "leg_extreme_date": leg_extreme_date,
                "leg_extreme_price": leg_extreme_price,
                "structure_ref": ref_low if trend == "UP" else ref_high,
                "secondary_id": (f"L{leg_id}-R1" if leg_end_idx is not None
                                 and trend in ("UP", "DOWN") else None),
                "reaction_start": leg_extreme_date if leg_end_idx is not None else None,
                "reaction_extreme": (corr_low if trend == "UP" else corr_high)
                if leg_end_idx is not None and leg_trend == trend else None,
                "reaction_extreme_date": corr_extreme_date,
                "invalidated_date": invalidated_date,
                "line_state": line["line_state"],
                "line_active": line["line_active"],
                "line_width_a": line.get("line_width_a"),
                "line_width_b": line.get("line_width_b"),
                "line_breakout": line.get("line_breakout", "NONE"),
                "line_breakout_a": line.get("line_breakout_a", "NONE"),
                "line_breakout_b": line.get("line_breakout_b", "NONE"),
                "volume_state": vstate,
                "vol_up_down_ratio": vol.get("ratio"),
                "vol_trend": vol.get("vol_trend"),
                "dow_signal": signal,
                "blocker": blocker,
                "H1": struct.get("H1"),
                "H2": struct.get("H2"),
                "L1": struct.get("L1"),
                "L2": struct.get("L2"),
                "notes": "|".join(notes),
            }
        )

    df = pd.DataFrame(rows)

    # ---- 状态不变量：任何情况下都不允许出现逻辑上不可能的组合 ----
    inv = inv_summary(df)
    meta = {
        "params": params.name,
        "structure_source": structure_source,
        "n_rows": len(df),
        "swings_a": len(swings_a),
        "swings_b": len(swings_b),
        "confirmation": confirmation_summary(completed),
        "trend_distribution": df["primary_trend"].value_counts().to_dict(),
        "integrity_distribution": df["trend_integrity"].value_counts().to_dict(),
        "secondary_distribution": df["secondary_reaction"].value_counts().to_dict(),
        "signal_distribution": df["dow_signal"].value_counts().to_dict(),
        "blocker_distribution": (df["blocker"].value_counts().to_dict()
                                 if "blocker" in df.columns else {}),
        "line_days": int(df["line_active"].sum()),
        "line_breakout_days": int((df["line_breakout"].isin(["UP_CONFIRMED", "DOWN_CONFIRMED"])).sum()),
        "line_single_breakout_days": int(
            ((df["line_breakout_a"] != "NONE") | (df["line_breakout_b"] != "NONE")).sum()
        ),
        "legs": int(df["primary_leg_id"].nunique()),
        "invariants": inv,
    }
    if strict_invariants:
        check = check_frame(df)
        if check:
            head = "\n".join(str(x) for x in check[:20])
            raise AssertionError(
                f"道氏状态机违反 {len(check)} 条不变量：\n{head}"
                + (f"\n... 另有 {len(check) - 20} 条" if len(check) > 20 else "")
            )
    return df, meta


class _Dummy:
    """轻量结构占位：只为复用 trend_engine.build_structure."""

    def __init__(self, end_price: float, end_date):
        self.end_price = end_price
        self.end_date = end_date


def latest_state(df: pd.DataFrame, params_name: str = "") -> DowState:
    """把最后一行 DataFrame 转成 DowState 对象."""
    if df.empty:
        return DowState(params_name=params_name)
    r = df.iloc[-1]
    st = DowState(
        params_name=params_name,
        primary_trend=r["primary_trend"],
        trend_integrity=r.get("trend_integrity", "UNKNOWN"),
        reversal_candidate=r.get("reversal_candidate", "NONE"),
        last_confirmed_swing=r["last_confirmed_swing"],
        secondary_reaction=r["secondary_reaction"],
        retracement_ratio=float(r["retracement_ratio"]),
        secondary_duration=int(r["secondary_duration"]),
        primary_leg_id=(None if pd.isna(r.get("primary_leg_id")) else int(r["primary_leg_id"])),
        secondary_id=(None if pd.isna(r.get("secondary_id")) else r["secondary_id"]),
        dual_confirmation=r["dual_confirmation"],
        line_state=r["line_state"],
        volume_state=r["volume_state"],
        dow_signal=r["dow_signal"],
    )
    st.date = r["date"]
    if isinstance(r["notes"], str) and r["notes"]:
        st.notes = r["notes"].split("|")
    return st
