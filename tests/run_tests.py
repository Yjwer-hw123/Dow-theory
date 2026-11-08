"""合成数据单元测试 —— 全部用例都不依赖网络.

运行:
    python tests/run_tests.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.dow_params import DOW_LITERAL_V01 as P  # noqa: E402
from core.confirmation import confirm_primary  # noqa: E402
from core.line_detector import detect_line_pair, detect_line_state  # noqa: E402
from core.secondary_reaction import compute_retracement, evaluate_secondary  # noqa: E402
from core.swing_detector import detect_swings  # noqa: E402
from core.trend_engine import classify  # noqa: E402

START = pd.Timestamp("2020-01-01")


def dates_for(values: list[float]) -> list[pd.Timestamp]:
    return [START + timedelta(days=i) for i in range(len(values))]


def _case(name: str):
    print(f"  - {name}")


# ---------------------------------------------------------------- 浪识别
def test_swing_basic():
    # README 里那个例子：100 → 110 (+10%)，再回落到 104 (-5.45%)
    closes = [100, 102, 105, 108, 110, 107, 104]
    sw = detect_swings(dates_for(closes), closes, pct=0.03)
    _case("basic up then down")
    ups = [s for s in sw if s.direction == "UP"]
    downs = [s for s in sw if s.direction == "DOWN"]
    assert ups, "应当识别出上升浪"
    assert ups[0].start_price == 100 and ups[0].end_price == 110, ups[0]
    assert abs(ups[0].move_pct - 0.10) < 1e-9
    assert downs, "应当识别出下降浪"
    assert downs[-1].start_price == 110, downs[-1]
    assert downs[-1].end_price == 104
    assert downs[-1].provisional is True
    # 波浪必须交替出现
    seq = [s.direction for s in sw]
    for a, b in zip(seq, seq[1:]):
        assert a != b, "浪必须方向交替"


def test_swing_below_threshold():
    # 噪声级别波动不应该产生浪
    closes = [100, 101, 100.5, 101.2, 100.8]
    sw = [s for s in detect_swings(dates_for(closes), closes, pct=0.03) if not s.provisional]
    _case("noise below 3% produces no completed swing")
    assert sw == [], sw


def test_swing_alternation_long():
    closes = [100, 104, 100, 105, 98, 110, 104, 115, 110]
    sw = detect_swings(dates_for(closes), closes, pct=0.03)
    _case("long alternation")
    seq = [s.direction for s in sw]
    assert seq[0] in ("UP", "DOWN")
    for a, b in zip(seq, seq[1:]):
        assert a != b


# ---------------------------------------------------------------- 趋势状态机
def test_trend_classify():
    _case("HH+HL -> UP")
    assert classify({"H1": 100, "H2": 110, "L1": 90, "L2": 95}) == "UP"
    _case("LH+LL -> DOWN")
    assert classify({"H1": 110, "H2": 105, "L1": 95, "L2": 90}) == "DOWN"
    _case("mixed -> TRANSITION (拒绝强行二分)")
    assert classify({"H1": 100, "H2": 110, "L1": 95, "L2": 90}) == "TRANSITION"
    assert classify({"H1": 110, "H2": 105, "L1": 90, "L2": 95}) == "TRANSITION"
    _case("insufficient -> UNCONFIRMED")
    assert classify({"H1": None, "H2": 110, "L1": 95, "L2": 95}) == "UNCONFIRMED"


# ---------------------------------------------------------------- 双指数确认
def test_confirmation():
    d = dates_for([0] * 10)
    from core.types import Swing

    a = [
        Swing("UP", d[0].date(), 100, d[2].date(), 110, "A"),
        Swing("DOWN", d[2].date(), 110, d[5].date(), 104, "A"),
    ]
    b = [Swing("UP", d[1].date(), 50, d[4].date(), 56, "B")]
    conf = confirm_primary(a, b, mode="until_next_reversal")
    _case("primary UP swing confirmed by B within next reversal")
    assert conf[0].confirmed is True
    assert not conf[1].confirmed  # DOWN 浪没有同向确认

    b_diverge = [Swing("DOWN", d[1].date(), 50, d[4].date(), 48, "B")]
    conf2 = confirm_primary(a, b_diverge, mode="until_next_reversal")
    _case("diverging index -> no confirmation")
    assert conf2[0].confirmed is False


# ---------------------------------------------------------------- 次级反应
def test_retracement_math():
    _case("retracement = (H-C)/(H-L)")
    assert abs(compute_retracement(100, 150, 120) - 0.60) < 1e-9


def test_secondary_states():
    today = START.date()
    start = today
    r = evaluate_secondary("UP", prev_low=100, prev_high=150, corrective_low=120,
                           corrective_start=start, today=today, params=P, hl_intact=True)
    _case("60% retracement, HL intact -> ACTIVE in band")
    assert r["state"] == "ACTIVE" and r["in_band"] is True

    r = evaluate_secondary("UP", prev_low=100, prev_high=150, corrective_low=95,
                           corrective_start=start, today=today, params=P, hl_intact=False)
    _case("prior low broken -> INVALIDATED (not OVERDUE)")
    assert r["state"] == "INVALIDATED", r
    assert r["retracement"] > 1.0

    r = evaluate_secondary("UP", prev_low=100, prev_high=150, corrective_low=148,
                           corrective_start=start, today=today, params=P, hl_intact=True)
    _case("no meaningful counter-move -> NONE")
    assert r["state"] == "NONE"

    r = evaluate_secondary("UP", prev_low=100, prev_high=150, corrective_low=120,
                           corrective_start=start - timedelta(days=200), today=today,
                           params=P, hl_intact=True)
    _case("correction too long but structure intact -> OVERDUE")
    assert r["state"] == "OVERDUE"

    r = evaluate_secondary("UP", prev_low=100, prev_high=150, corrective_low=120,
                           corrective_start=start, today=today, params=P,
                           hl_intact=True, resumed=True)
    _case("reaction extreme reclaimed -> COMPLETED")
    assert r["state"] == "COMPLETED"

    _case("OVERDUE 必须伴随结构完好（INV-01）")
    for c in (115, 130, 145):
        rr = evaluate_secondary("UP", prev_low=100, prev_high=150, corrective_low=c,
                                corrective_start=start - timedelta(days=200),
                                today=today, params=P, hl_intact=True)
        assert rr["state"] == "OVERDUE" and rr["retracement"] <= 1.0, (c, rr)


def test_integrity_and_reversal():
    from core.trend_engine import assess_integrity, reversal_candidate

    struct = {"H1": 100, "H2": 120, "L1": 90, "L2": 100}
    _case("UP + 回调守住 L2 -> INTACT / NONE")
    assert assess_integrity(struct, "UP", corrective_low=105) == "INTACT"
    assert reversal_candidate("UP", "INTACT") == "NONE"
    _case("UP + 跌破 L2 -> DAMAGED / 候选 BEAR（但 trend 仍是 UP）")
    assert assess_integrity(struct, "UP", corrective_low=98) == "DAMAGED"
    assert reversal_candidate("UP", "DAMAGED") == "BEAR"
    _case("DOWN + 涨破 H2 -> DAMAGED / 候选 BULL")
    sd = {"H1": 120, "H2": 100, "L1": 100, "L2": 80}
    assert assess_integrity(sd, "DOWN", corrective_high=99) == "INTACT"
    assert assess_integrity(sd, "DOWN", corrective_high=103) == "DAMAGED"
    assert reversal_candidate("DOWN", "DAMAGED") == "BULL"
    _case("TRANSITION -> UNKNOWN，拒绝强行二分")
    assert assess_integrity(struct, "TRANSITION", corrective_low=50) == "UNKNOWN"
    assert reversal_candidate("TRANSITION", "UNKNOWN") == "NONE"


def test_invariants_on_synthetic_frame():
    """不变量检查器本身必须能在坏数据上报错."""
    from core.invariants import ALLOWED_SECONDARY, check_frame, summarize
    import pandas as pd

    good = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "close_a": 100.0, "primary_trend": "UP",
         "trend_integrity": "INTACT", "reversal_candidate": "NONE",
         "secondary_reaction": "ACTIVE", "retracement_ratio": 0.4, "secondary_duration": 10,
         "dual_confirmation": "CONFIRMED", "dow_signal": "RISK_ON",
         "primary_leg_id": 1, "secondary_id": "L1-R1", "reaction_extreme": 95.0},
        {"date": pd.Timestamp("2020-01-02"), "close_a": 101.0, "primary_trend": "UP",
         "trend_integrity": "DAMAGED", "reversal_candidate": "BEAR",
         "secondary_reaction": "INVALIDATED", "retracement_ratio": 1.2, "secondary_duration": 20,
         "dual_confirmation": "CONFIRMED", "dow_signal": "WAIT",
         "primary_leg_id": 1, "secondary_id": "L1-R1", "reaction_extreme": 88.0},
    ])
    _case("自洽事件帧 -> 0 违规")
    vs = check_frame(good)
    assert vs == [], [str(x) for x in vs]
    assert summarize(good)["ok"] is True

    _case("ACTIVE 但 ratio>1 -> INV-01")
    bad = good.copy()
    bad.loc[0, "retracement_ratio"] = 1.4
    assert any(v.rule == "INV-01" for v in check_frame(bad))

    _case("结构 DAMAGED 却 RISK_ON -> INV-06")
    bad2 = good.copy()
    bad2.loc[1, "dow_signal"] = "RISK_ON"
    assert any(v.rule == "INV-06" for v in check_frame(bad2))

    _case("reaction 旧腿粘连 -> INV-11")
    bad3 = good.copy()
    bad3.loc[1, "secondary_id"] = "L7-R1"
    assert any(v.rule == "INV-11" for v in check_frame(bad3))

    _case("同 leg 内回调极值抬升 -> INV-12")
    bad4 = good.copy()
    bad4.loc[1, "reaction_extreme"] = 99.0
    bad4.loc[1, "secondary_reaction"] = "ACTIVE"
    bad4.loc[1, "retracement_ratio"] = 0.2
    bad4.loc[1, "trend_integrity"] = "INTACT"
    assert any(v.rule == "INV-12" for v in check_frame(bad4))
    assert set(ALLOWED_SECONDARY) == {"NONE", "ACTIVE", "COMPLETED", "OVERDUE", "INVALIDATED"}


# ---------------------------------------------------------------- 线
def test_line_detection():
    n = 12
    dates = dates_for([0] * n)
    highs = [100 + i * 0.1 for i in range(n)]      # 窄幅
    lows = [99 + i * 0.1 for i in range(n)]
    closes = [99.5 + i * 0.1 for i in range(n)]
    r = detect_line_state([d.date() for d in dates], highs, lows, closes, n - 1, P)
    _case("~1% range over 12 bars -> line active")
    assert r["line_active"] is True
    assert r["line_width"] < P.line_width_pct

    highs2 = [100 + i * 1.5 for i in range(n)]     # 宽幅
    lows2 = [99 + i * 1.5 for i in range(n)]
    closes2 = [99.5 + i * 1.5 for i in range(n)]
    r2 = detect_line_state([d.date() for d in dates], highs2, lows2, closes2, n - 1, P)
    _case("wide range -> no line")
    assert r2["line_active"] is False


def test_line_pair_requires_both():
    n = 12
    dates = [d.date() for d in dates_for([0] * n)]
    flat_h = [100 + i * 0.05 for i in range(n)]
    flat_l = [99.5 + i * 0.05 for i in range(n)]
    flat_c = [99.7 + i * 0.05 for i in range(n)]
    df_a = pd.DataFrame({"date": dates, "high": flat_h, "low": flat_l, "close": flat_c})
    wide = [100 + i * 1.5 for i in range(n)]
    df_b = pd.DataFrame({"date": dates, "high": wide,
                         "low": [w - 1 for w in wide],
                         "close": [w - 0.5 for w in wide]})
    r = detect_line_pair(df_a, df_b, n - 1, P)
    _case("only one index narrow -> no line (第12/14章)")
    assert r["line_state"] == "NONE"

    r2 = detect_line_pair(df_a, df_a, n - 1, P)
    _case("both narrow -> line forms")
    assert r2["line_active"] is True


def test_confirmation_has_no_lookahead():
    """防前瞻偏差回归测试（P10 阶段发现的引擎级 bug）.

    第 12 章允许"另一指数稍后才确认"，但**引擎只能在市场已经知道确认的那天**
    才能使用这条浪。第一版实现按"本浪结束日"把浪放进结构，于是引擎在指数 A 的浪
    走完当天，就用上了指数 B 几周甚至几个月后才给出的确认 —— 实盘上这个状态根本
    不存在。后果很严重：它让 RISK_ON 精确地在顶部亮起（回测里次日照仓收益 +66%，
    平移一天后变成 -7.65%）。

    本用例直接构造浪对象（不依赖浪识别），断言：
      * 被晚确认的浪，usable_date = 确认浪结束日（不是自己的结束日）
      * 没被确认的浪，usable_date = 确认窗口关闭日
    """
    from core.types import Swing

    d = dates_for([0] * 12)
    a = [
        Swing("UP", d[0].date(), 100, d[2].date(), 110, "A"),     # A 在 t2 走完
        Swing("DOWN", d[2].date(), 110, d[9].date(), 95, "A"),    # 下一反向浪 t9 走完
    ]
    # B 的同向上升浪直到 t9 才结束 —— 仍在 A 的第一条浪的确认窗口内
    b = [Swing("UP", d[4].date(), 50, d[9].date(), 60, "B")]

    conf = confirm_primary(a, b, mode="until_next_reversal")
    _case("晚确认的浪：usable_date 必须等于确认浪结束日")
    assert conf[0].confirmed is True
    assert conf[0].swing.end_date == d[2].date()
    assert conf[0].usable_date == d[9].date(), conf[0].usable_date
    assert conf[0].usable_date > conf[0].swing.end_date

    _case("未确认的浪：usable_date 不得早于确认窗口关闭日")
    assert conf[1].confirmed is False
    # 第二条浪后面没有更多浪 -> 窗口无上界，此时才"知道"它不会被确认
    assert conf[1].usable_date >= conf[1].swing.end_date

    # 同日确认：usable_date 就是本浪结束日
    b_same = [Swing("UP", d[0].date(), 50, d[2].date(), 60, "B")]
    conf2 = confirm_primary(a, b_same, mode="until_next_reversal")
    _case("同日确认：usable_date 与本浪结束日一致")
    assert conf2[0].confirmed and conf2[0].usable_date == d[2].date()


def test_line_breakout_is_reachable():
    """V0.1.1 之前这条用例在数学上不可能通过：区间包含了突破日自己。"""
    n = 14
    dates = [d.date() for d in dates_for([0] * n)]
    flat_h = [100 + (i % 3) * 0.1 for i in range(n)]     # 前 n-1 根窄幅
    flat_l = [99.4 + (i % 3) * 0.1 for i in range(n)]
    flat_c = [99.7 + (i % 3) * 0.1 for i in range(n)]
    flat_c[-1] = 106.0                                    # 最后一根向上突破
    df_a = pd.DataFrame({"date": dates, "high": flat_h, "low": flat_l, "close": flat_c})
    r = detect_line_pair(df_a, df_a, n - 1, P)
    _case("盘整后向上突破 -> 双指数 UP_CONFIRMED")
    assert r["line_active"] is True, r
    assert r["line_breakout"] == "UP_CONFIRMED", r
    assert r["line_state"] == "ACCUMULATION"

    flat_c2 = list(flat_c[:-1]) + [92.0]
    df_b = pd.DataFrame({"date": dates, "high": flat_h, "low": flat_l, "close": flat_c2})
    r2 = detect_line_pair(df_a, df_b, n - 1, P)
    _case("一个向上一个向下 -> NO_CONFIRMATION（第14章：单边不算）")
    assert r2["line_breakout"] == "NO_CONFIRMATION", r2
    assert r2["line_breakout_a"] == "UP" and r2["line_breakout_b"] == "DOWN"

    # 单边突破必须仍然产出诊断信息，否则无法审计"0 次同时突破"
    wide = [100 + i * 1.5 for i in range(n)]
    df_c = pd.DataFrame({"date": dates, "high": wide, "low": [w - 1 for w in wide],
                         "close": [w - 0.5 for w in wide]})
    r3 = detect_line_pair(df_a, df_c, n - 1, P)
    _case("只有一个指数成线 -> line NONE 但保留诊断")
    assert r3["line_active"] is False and r3["line_state"] == "NONE"


def _synth_pair(n: int = 600, seed: int = 7):
    """确定性合成双指数（不联网）：振幅足够产生 3% 级别的浪与线态."""
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.arange(n)
    lab = pd.to_datetime(START) + pd.to_timedelta(t, unit="D")

    def mk(amp1, per1, amp2, per2, phase, noise, vbase):
        c = (1000.0 + amp1 * np.sin(t / per1 + phase) + amp2 * np.sin(t / per2)
             + np.cumsum(rng.normal(0, noise, n)))
        return pd.DataFrame({
            "date": lab,
            "open": c * 0.999,
            "high": c * 1.008,
            "low": c * 0.992,
            "close": c,
            "volume": vbase + rng.normal(0, vbase * 0.1, n),
            "amount": vbase,
        })

    return mk(300, 43, 120, 11, 0.0, 6.0, 1e6), mk(340, 47, 100, 13, 0.4, 7.0, 1e6)


def test_swing_known_date_is_later_than_extreme():
    """极值日 ≠ 可知日 —— P10.1 修掉的第二层前瞻偏差的来源.

    浪的 `end_date` 是**极值点**当天，但它要到价格从极值反向走够 3% 才会被
    识别出来。引擎必须按 `known_date` 消费；若按 `end_date`，就会在极值日当天
    用上几周之后才知道的信息。
    """
    closes = [100, 102, 105, 108, 110, 107, 104]
    ds = dates_for(closes)
    sw = detect_swings(ds, closes, pct=0.03)
    up = [s for s in sw if s.direction == "UP"][0]
    _case("极值在第 5 天；识别发生在跌破 110*0.97=106.7 的那天")
    assert up.end_date == ds[4].date(), up.end_date
    assert up.detected_date is not None
    assert up.detected_date > up.end_date, (up.end_date, up.detected_date)
    assert up.known_date == up.detected_date


def test_engine_prefix_invariance():
    """★ 核心回归测试：全引擎无前瞻（Prefix Invariance）.

    对同一个日期 t：
        用**全部**数据跑出的 t 日状态  ==  只用**前 t+1 个点**跑出的最后一天状态

    任何一个字段不一致，就说明引擎在 t 日消费了未来信息。

    这是最重要的一条测试 —— 它**不预设泄漏发生在哪个模块**：
    swing / confirmation / trend / secondary / line / volume / regime
    任何一层的未来函数都会被它抓住。改动上述任一模块后都必须跑通。
    （P10 修掉的是"确认日"型前瞻；P10.1 修掉的是"极值日当可知日"型前瞻。）
    """
    from core.regime_engine import analyze_pair
    from research.lookahead_audit import COMPARE_FIELDS, diff_rows

    a, b = _synth_pair(600)
    full, _ = analyze_pair(a, b, P, strict_invariants=False)

    bad = []
    for t in (180, 260, 340, 420, 500, 570):
        pre_a = a.iloc[: t + 1].reset_index(drop=True)
        pre_b = b.iloc[: t + 1].reset_index(drop=True)
        pre, _ = analyze_pair(pre_a, pre_b, P, strict_invariants=False)
        d = diff_rows(full.iloc[t], pre.iloc[-1], fields=COMPARE_FIELDS)
        if d:
            bad.append((str(full.iloc[t]["date"])[:10], d[0]))
    _case("prefix == full（不一致即存在第二层前瞻偏差）")
    assert not bad, f"Prefix Invariance 被破坏: {bad}"


TESTS = [
    test_swing_basic,
    test_swing_below_threshold,
    test_swing_alternation_long,
    test_trend_classify,
    test_confirmation,
    test_retracement_math,
    test_secondary_states,
    test_integrity_and_reversal,
    test_invariants_on_synthetic_frame,
    test_line_detection,
    test_line_pair_requires_both,
    test_line_breakout_is_reachable,
    test_confirmation_has_no_lookahead,
    test_swing_known_date_is_later_than_extreme,
    test_engine_prefix_invariance,
]


def main() -> int:
    print(f"道氏引擎单元测试（{len(TESTS)} 个用例）")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print("-" * 50)
    print(f"{len(TESTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
