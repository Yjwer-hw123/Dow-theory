# -*- coding: utf-8 -*-
"""ENTRY-PHASE-02 第一层 L1：B 类信号扣成本盈利性回测。

严格按 out/entry_phase_02/ENTRY_PHASE_02_PROTOCOL_v0.3_FROZEN.md（§4.6 写死）：
- 信号：复用 entry_phase.py 的 episode / 候选周 / B 类标签（不改定义）；
- 价格：cache/raw（新浪不复权）+ cache/factors 在内存合成 qfq = raw/f；
- 成本基数：回到原始价 raw_price = qfq × f，名义本金 N=50,000 元/笔；
- 入场：信号周后第一个有效 bar 开盘 + 滑点；
- 退出：入场后第 60 个有效 bar 收盘 − 滑点；
- 三情景：base 0.030%/边+10bp；cons 0.100%/边+20bp；stress 0.300%/边+30bp；
- 不模拟涨跌停、停牌顺延、资金占用、容量（属于 L2）。

用法：
    python scripts/entry_phase_02_l1.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.uptrend_detector as ud  # noqa: E402
from scripts.step1_trend_validation import FAC_DIR, RAW_DIR  # noqa: E402
from data.pit_universe import load_stock_codes  # noqa: E402

OUT_DIR = os.path.join(ROOT, "out", "entry_phase_02")
PROTOCOL = os.path.join(OUT_DIR, "ENTRY_PHASE_02_PROTOCOL_v0.3_FROZEN.md")
FREEZE_RECORD = os.path.join(OUT_DIR, "EP02_FREEZE_RECORD.json")

# ---- 冻结常量（§4.6，不得修改）----
NOTIONAL = 50000.0          # 每笔名义本金（元）
MIN_COMM = 5.0              # 最低佣金 5 元/边
MIN_TRANSFER = 1.0          # 面额口径最低过户费 1 元/边
SLIP = {"base": 0.0010, "cons": 0.0020, "stress": 0.0030}
COMM = {"base": 0.00030, "cons": 0.00100, "stress": 0.00300}
SCENARIOS = ("base", "cons", "stress")
HOLD_BARS = 60
YEARS_FULL = tuple(range(2005, 2026))
BOOT_SEED = 20261107
N_BOOT = 1000
COVERAGE_MIN = 0.99

# 印花税 6 段（HISTORICAL_FEE_RULES.csv）
STAMP_SEGS = (
    ("2005-01-01", "2005-01-23", 0.002, True),
    ("2005-01-24", "2007-05-29", 0.001, True),
    ("2007-05-30", "2008-04-23", 0.003, True),
    ("2008-04-24", "2008-09-18", 0.001, True),
    ("2008-09-19", "2023-08-27", 0.001, False),
    ("2023-08-28", "2030-12-31", 0.0005, False),
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_sh(code: str) -> bool:
    return code[0] in "65"


def stamp_rate(d, side: str) -> float:
    t = pd.Timestamp(d)
    for a, b, rate, both in STAMP_SEGS:
        if pd.Timestamp(a) <= t <= pd.Timestamp(b):
            return rate if (both or side == "sell") else 0.0
    return 0.0


def transfer_fee(code: str, d, side: str, amount: float,
                 shares: float, scenario: str) -> float:
    """沪/深过户费分段；按交易所路由，禁止相加（R1）。"""
    t = pd.Timestamp(d)
    if is_sh(code):
        if t < pd.Timestamp("2012-06-01"):
            rate, by_face, floor = 0.001, True, MIN_TRANSFER
        elif t < pd.Timestamp("2012-09-01"):
            rate, by_face, floor = 0.000375, True, MIN_TRANSFER
        elif t < pd.Timestamp("2015-08-01"):
            rate, by_face, floor = 0.0003, True, MIN_TRANSFER
        elif t < pd.Timestamp("2022-04-29"):
            rate, by_face, floor = 0.00002, False, 0.0
        else:
            rate, by_face, floor = 0.00001, False, 0.0
        if (scenario == "stress"
                and pd.Timestamp("2012-06-01") <= t < pd.Timestamp("2015-08-01")):
            rate *= 2.0
    else:
        if t < pd.Timestamp("2015-08-01"):
            return 0.0          # 深市 0.0255‰ 从经手费分成，已含在佣金内
        if t < pd.Timestamp("2022-04-29"):
            rate, by_face, floor = 0.00002, False, 0.0
        else:
            rate, by_face, floor = 0.00001, False, 0.0
    base = shares * 1.0 if by_face else amount
    return max(base * rate, floor)


def eval_trade(code: str, d_entry, d_exit, raw_entry: float,
               raw_exit: float, scenario: str) -> dict:
    shares = NOTIONAL / raw_entry
    p_buy = raw_entry * (1.0 + SLIP[scenario])
    buy_amt = shares * p_buy
    buy_comm = max(buy_amt * COMM[scenario], MIN_COMM)
    buy_tf = transfer_fee(code, d_entry, "buy", buy_amt, shares, scenario)
    buy_cash = buy_amt + buy_comm + buy_tf

    p_sell = raw_exit * (1.0 - SLIP[scenario])
    sell_amt = shares * p_sell
    sell_comm = max(sell_amt * COMM[scenario], MIN_COMM)
    sell_stamp = sell_amt * stamp_rate(d_exit, "sell")
    sell_tf = transfer_fee(code, d_exit, "sell", sell_amt, shares, scenario)
    sell_cash = sell_amt - sell_comm - sell_stamp - sell_tf

    gross_pnl = shares * (raw_exit - raw_entry)
    net_pnl = sell_cash - buy_cash
    return dict(
        gross_return=gross_pnl / NOTIONAL,
        net_return=net_pnl / NOTIONAL,
        comm=(buy_comm + sell_comm) / NOTIONAL,
        stamp=sell_stamp / NOTIONAL,
        transfer=(buy_tf + sell_tf) / NOTIONAL,
        slippage=gross_pnl / NOTIONAL - net_pnl / NOTIONAL
        - (buy_comm + sell_comm + sell_stamp + buy_tf + sell_tf) / NOTIONAL,
    )


def ns_of(x) -> int:
    return int(pd.Timestamp(x).value)


def load_facs(code: str) -> list[dict] | None:
    p = os.path.join(FAC_DIR, f"{code}.json")
    if not os.path.exists(p):
        return None
    try:
        obj = json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(obj, list) and len(obj) > 0:
        return obj
    return None


def factor_arrays(facs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    d = np.array([ns_of(x["d"]) for x in facs], dtype=np.int64)
    f = np.array([float(x["f"]) for x in facs], dtype=np.float64)
    o = np.argsort(d)
    return d[o], f[o]


def f_at(darr: np.ndarray, farr: np.ndarray, t: int) -> float:
    i = int(np.searchsorted(darr, int(t), side="right")) - 1
    if i < 0:
        i = 0
    return float(farr[i])


def process_stock(code: str) -> dict:
    """单只股票：构建 qfq 面板 → B 类 episode → 逐笔交易与成本。"""
    out = dict(code=code, ok=False, reason=None, trades=[], open_positions=[],
               n_episodes=0, n_sig=0, n_bars=0,
               excl=dict(no_category=0, untradable=0, no_r60=0, b0_lt5=0))
    rp = os.path.join(RAW_DIR, f"{code}.parquet")
    if not os.path.exists(rp):
        out["reason"] = "no_raw"
        return out
    try:
        raw = pd.read_parquet(rp)
    except Exception:  # noqa: BLE001
        out["reason"] = "raw_read_fail"
        return out
    if raw is None or raw.empty:
        out["reason"] = "raw_empty"
        return out
    facs = load_facs(code)
    if facs is None:
        out["reason"] = "no_factor"
        try:
            out["n_bars"] = int(len(pd.read_parquet(rp)))
        except Exception:  # noqa: BLE001
            out["n_bars"] = 0
        return out
    A = ud.prepare_stock(raw, facs)
    if A is None:
        out["reason"] = "prepare_fail"
        out["n_bars"] = int(len(raw))
        return out

    s = ep.build_stock(code, A)
    dates = A["dates"]
    c = A["c"]
    n = dates.size
    out["n_bars"] = int(n)
    darr, farr = factor_arrays(facs)

    for ep_i, a, b in ep.episode_groups(s):
        out["n_episodes"] += 1
        lab = s["LABEL_B"][a:b]
        obs = np.nonzero(lab)[0] + a
        if obs.size == 0:
            out["excl"]["no_category"] += 1
            continue
        out["n_sig"] += int(obs.size)
        i0 = int(obs[0])
        if not bool(s["TRADABLE"][i0]):
            out["excl"]["untradable"] += 1
            continue
        di = int(s["ENTRY_DI"][i0])
        xi = di + HOLD_BARS - 1
        if not np.isfinite(s["R60"][i0]) or xi >= n:
            out["excl"]["no_r60"] += 1
            qe = float(s["ENTRY_OPEN"][i0])
            ql = float(c[n - 1])
            out["open_positions"].append(dict(
                code=code,
                signal_date=str(pd.Timestamp(ep.from_ns(int(s["W_END"][i0])).date())),
                entry_date=str(pd.Timestamp(dates[di]).date()),
                last_date=str(pd.Timestamp(dates[n - 1]).date()),
                open_return=ql / qe - 1.0,
            ))
            continue
        fin = np.isfinite(s["R60"][a:b])
        b0 = np.nonzero((~lab) & fin)[0] + a
        if b0.size < ep.ENTRY_MIN_B0:
            out["excl"]["b0_lt5"] += 1
            continue

        qe = float(s["ENTRY_OPEN"][i0])
        qx = float(c[xi])
        de = dates[di]
        dx = dates[xi]
        fe = f_at(darr, farr, ns_of(de))
        fx = f_at(darr, farr, ns_of(dx))
        raw_e = qe * fe
        raw_x = qx * fx
        if not (np.isfinite(raw_e) and raw_e > 0
                and np.isfinite(raw_x) and raw_x > 0):
            out["excl"]["no_r60"] += 1
            continue

        rec = dict(
            code=code, ep=ep_i,
            signal_date=str(pd.Timestamp(ep.from_ns(int(s["W_END"][i0])).date())),
            entry_date=str(pd.Timestamp(de).date()),
            exit_date=str(pd.Timestamp(dx).date()),
            hold_bars=HOLD_BARS,
            qfq_entry=qe, qfq_exit=qx, f_entry=fe, f_exit=fx,
            raw_entry=raw_e, raw_exit=raw_x,
            age=int(s["AGE"][i0]), depth=float(s["DEPTH"][i0]),
            vol_ratio=float(s["VOL_RATIO"][i0]), b0_count=int(b0.size),
            year_exit=int(pd.Timestamp(dx).year),
            year_signal=int(ep.from_ns(int(s["W_END"][i0])).year),
            gross_return=raw_x / raw_e - 1.0,
        )
        for sc in SCENARIOS:
            t = eval_trade(code, de, dx, raw_e, raw_x, sc)
            rec[f"net_return_{sc}"] = t["net_return"]
            rec[f"cost_total_{sc}"] = t["gross_return"] - t["net_return"]
            if sc == "base":
                rec["fee_comm"] = t["comm"]
                rec["fee_stamp"] = t["stamp"]
                rec["fee_transfer"] = t["transfer"]
                rec["cost_slippage"] = t["slippage"]
        out["trades"].append(rec)

    out["ok"] = True
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def ci95_boot(r: np.ndarray, reps: int = N_BOOT, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = r.size
    if n == 0:
        return (np.nan, np.nan)
    m = np.array([r[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def metrics(r: np.ndarray, gross: np.ndarray) -> dict:
    n = int(r.size)
    if n == 0:
        return dict(n=0)
    pos = r[r > 0]
    neg = r[r <= 0]
    payoff = (float(pos.mean() / abs(neg.mean()))
              if pos.size and neg.size and neg.mean() != 0 else np.nan)
    k = max(1, int(math.ceil(n * 0.01)))
    ex = np.sort(r)[::-1][k:]
    lo, hi = ci95_boot(r)
    return dict(
        n=n,
        gross_expectancy=float(gross.mean()),
        gross_total=float(gross.sum()),
        gross_total_amount=float(gross.sum() * NOTIONAL),
        cost_total=float((gross - r).sum()),
        cost_total_amount=float((gross - r).sum() * NOTIONAL),
        net_expectancy=float(r.mean()),
        net_total=float(r.sum()),
        net_total_amount=float(r.sum() * NOTIONAL),
        net_median=float(np.median(r)),
        net_win_rate=float((r > 0).mean()),
        payoff=payoff,
        ci95=[lo, hi],
        ex_top1_expectancy=float(ex.mean()) if ex.size else np.nan,
        cost_drag_share=float((gross - r).sum() / gross.sum()) if gross.sum() > 0 else np.nan,
    )


def annual_table(trades: list[dict], key: str) -> list[dict]:
    rows = []
    for y in sorted({t["year_exit"] for t in trades} | set(YEARS_FULL) | {2026}):
        sel = [t for t in trades if t["year_exit"] == y]
        r = np.array([t[key] for t in sel], dtype=np.float64)
        g = np.array([t["gross_return"] for t in sel], dtype=np.float64)
        rows.append(dict(
            year=y, n=len(sel),
            gross_total=float(g.sum()) if g.size else 0.0,
            cost_total=float((g - r).sum()) if g.size else 0.0,
            net_total=float(r.sum()) if r.size else 0.0,
            net_amount=float(r.sum() * NOTIONAL) if r.size else 0.0,
            win_rate=float((r > 0).mean()) if r.size else np.nan,
            avg_net=float(r.mean()) if r.size else np.nan,
        ))
    return rows


def curve_metrics(trades: list[dict], key: str) -> dict:
    """按退出日（同日按代码）排序的等额净值曲线。"""
    ts = sorted(trades, key=lambda t: (t["exit_date"], t["code"]))
    r = np.array([t[key] for t in ts], dtype=np.float64)
    equity = np.r_[1.0, 1.0 + np.cumsum(r)]
    peak = np.maximum.accumulate(equity)
    mdd = float(np.max((peak - equity) / peak)) if equity.size else 0.0
    burn = 0
    worst = 0
    for x in r:
        if x <= 0:
            burn += 1
            worst = max(worst, burn)
        else:
            burn = 0
    comp = np.cumprod(1.0 + r)
    peak2 = np.maximum.accumulate(np.r_[1.0, comp])
    mdd_comp = float(np.max((peak2 - np.r_[1.0, comp]) / peak2))
    return dict(nav_end=float(equity[-1]), mdd_l1=mdd,
                max_consec_loss=int(worst), nav_end_compound=float(comp[-1]),
                mdd_compound=mdd_comp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ep.init_ctx()
    codes = load_stock_codes()
    if args.limit:
        codes = codes[:args.limit]
    log(f"[L1] 股票池 {len(codes)} 只（data/universe/stock_codes.json）")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_stock, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                results.append(dict(code=futs[fut], ok=False,
                                    reason=f"exception:{type(e).__name__}",
                                    trades=[], open_positions=[], n_bars=0,
                                    n_episodes=0, n_sig=0,
                                    excl=dict(no_category=0, untradable=0,
                                              no_r60=0, b0_lt5=0)))
            done += 1
            if done % 500 == 0 or done == len(codes):
                ntr = sum(len(r["trades"]) for r in results)
                log(f"  build {done}/{len(codes)} trades={ntr} {time.time()-t0:.0f}s")

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    trades = [t for r in ok for t in r["trades"]]
    opens = [p for r in ok for p in r["open_positions"]]
    trades.sort(key=lambda t: (t["exit_date"], t["code"]))

    total_bars = sum(r["n_bars"] for r in ok)
    bad_bars = sum(r["n_bars"] for r in bad)
    coverage = len(ok) / max(len(codes), 1)
    n_fac = sum(1 for c in codes if load_facs(c) is not None)
    factor_coverage = n_fac / max(len(codes), 1)
    bar_share_bad = bad_bars / max(total_bars + bad_bars, 1)
    reasons: dict[str, int] = {}
    for r in bad:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    log(f"[L1] 因子覆盖 {n_fac}/{len(codes)} ({factor_coverage:.2%})；"
        f"可回测股票 {len(ok)}/{len(codes)} ({coverage:.2%})；"
        f"不可计算 bar 占比 {bar_share_bad:.4%}；交易 {len(trades)} 笔；"
        f"未平仓 {len(opens)}；失败原因 {reasons}")

    # ---- 逐笔 CSV ----
    cols = list(trades[0].keys()) if trades else []
    pd.DataFrame(trades, columns=cols).to_csv(
        os.path.join(OUT_DIR, "trades_b_l1.csv"), index=False, encoding="utf-8")

    # ---- 指标 ----
    m = {sc: metrics(np.array([t[f"net_return_{sc}"] for t in trades], np.float64),
                     np.array([t["gross_return"] for t in trades], np.float64))
         for sc in SCENARIOS}
    curves = {sc: curve_metrics(trades, f"net_return_{sc}") for sc in SCENARIOS}
    annual = {sc: annual_table(trades, f"net_return_{sc}") for sc in SCENARIOS}
    pd.DataFrame(annual["base"]).to_csv(
        os.path.join(OUT_DIR, "annual_b_l1.csv"), index=False, encoding="utf-8")

    base = m["base"]
    years_pos = sum(1 for row in annual["base"]
                    if row["year"] in YEARS_FULL and row["net_total"] > 0)
    years_share = years_pos / len(YEARS_FULL)

    gates = {
        "L1-1_trades_ge_500": base["n"] >= 500,
        "L1-2_expectancy_pos_ci_lo_pos": bool(
            base["net_expectancy"] > 0 and base["ci95"][0] > 0),
        "L1-3_ex_top1_expectancy_pos": bool(base["ex_top1_expectancy"] > 0),
        "L1-4_year_pos_share_ge_60pct": bool(years_share >= 0.60),
        "L1-5_median_pos": bool(base["net_median"] > 0),
        "L1-6_mdd_le_30pct_and_streak_le_12": bool(
            curves["base"]["mdd_l1"] <= 0.30
            and curves["base"]["max_consec_loss"] <= 12),
    }
    core_pass = all(gates.values())
    data_ok = (factor_coverage >= COVERAGE_MIN
               and bar_share_bad <= 0.01 and base["n"] >= 500)

    if not data_ok:
        conclusion = "数据不足"
    elif core_pass:
        conclusion = "盈利"
    else:
        conclusion = "不盈利"

    summary = dict(
        meta=dict(
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            script="scripts/entry_phase_02_l1.py",
            protocol="ENTRY-PHASE-02 v0.3 FROZEN",
            protocol_sha256=json.load(open(FREEZE_RECORD, encoding="utf-8-sig"))["sha256"],
            notional=NOTIONAL,
            hold_bars=HOLD_BARS,
            boot_seed=BOOT_SEED,
            price_source="cache/raw + cache/factors → qfq=raw/f",
            cost_model="base 0.030%/side+10bp; cons 0.100%/side+20bp; stress 0.300%/side+30bp(+沪过户×2 2012-06~2015-07)",
        ),
        coverage=dict(universe=len(codes), computable=len(ok),
                      coverage_ratio=coverage,
                      factor_ok=n_fac, factor_coverage=factor_coverage,
                      failed=len(bad), failed_reasons=reasons,
                      not_computable_bar_share=bar_share_bad,
                      open_positions=len(opens),
                      excluded=dict(
                          no_category=sum(r["excl"]["no_category"] for r in ok),
                          untradable=sum(r["excl"]["untradable"] for r in ok),
                          no_r60=sum(r["excl"]["no_r60"] for r in ok),
                          b0_lt5=sum(r["excl"]["b0_lt5"] for r in ok),
                      )),
        primary=dict(scenario="base", metrics=base, curve=curves["base"],
                     annual=annual["base"],
                     years_positive=years_pos, years_denominator=len(YEARS_FULL),
                     years_positive_share=years_share),
        sensitivity=dict(
            cons=dict(metrics=m["cons"], curve=curves["cons"], annual=annual["cons"]),
            stress=dict(metrics=m["stress"], curve=curves["stress"],
                        annual=annual["stress"]),
        ),
        gates=gates,
        data_ok=data_ok,
        conclusion=conclusion,
    )
    with open(os.path.join(OUT_DIR, "summary_b_l1.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    log(f"[L1] 结论：{conclusion}；净期望 {base['net_expectancy']:.4%}；"
        f"CI95 [{base['ci95'][0]:.4%}, {base['ci95'][1]:.4%}]；笔数 {base['n']}")

    # ---- 覆盖率报告 ----
    write_coverage(summary, bad)
    # ---- 简明报告 ----
    write_report(summary, m, curves, annual)
    log("[L1] 已写出 trades_b_l1.csv / annual_b_l1.csv / summary_b_l1.json / "
        "BACKTEST_REPORT_B_L1.md / EP02_QFQ_COVERAGE.md")


def write_coverage(summary: dict, bad: list[dict]) -> None:
    cv = summary["coverage"]
    lines = ["# EP-02 全池复权覆盖率报告", "",
             f"- 生成时间：{summary['meta']['generated_at']}",
             f"- 股票池：{cv['universe']}",
             f"- 复权因子覆盖：{cv['factor_ok']}"
             f"（{cv['factor_coverage']:.2%}，门槛 ≥99%）",
             f"- 可回测（因子+面板构建成功）：{cv['computable']}"
             f"（{cv['coverage_ratio']:.2%}；差额为历史不足或数据异常的次新股，"
             f"非复权因子失败）",
             f"- 失败：{cv['failed']}；原因：{cv['failed_reasons']}",
             f"- 不可计算 bar 占比（股票×交易日）：{cv['not_computable_bar_share']:.4%}"
             f"（门槛 ≤1%）", "",
             "## 失败股票"]
    if bad:
        lines.append("| code | reason | bars |")
        lines.append("|---|---|---|")
        for r in sorted(bad, key=lambda x: x["code"]):
            lines.append(f"| {r['code']} | {r['reason']} | {r['n_bars']} |")
    else:
        lines.append("无。")
    lines += ["", "说明：qfq = raw / f（新浪累计因子），raw 为新浪不复权全量重抓；",
              "`amount` 保持原值；复权失败股票区间标 `not_computable`，不进入 L1 分子分母。"]
    with open(os.path.join(OUT_DIR, "EP02_QFQ_COVERAGE.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


def pct(x: float, nd: int = 2) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{nd}%}"


def write_report(summary: dict, m: dict, curves: dict, annual: dict) -> None:
    b = m["base"]
    cv = summary["coverage"]
    c = curves["base"]
    lines = [
        "# EP-02 第一层 L1：B 类信号扣成本盈利性回测", "",
        f"- 协议：`ENTRY_PHASE_02_PROTOCOL_v0.3_FROZEN.md`"
        f"（SHA256 `{summary['meta']['protocol_sha256']}`）",
        f"- 生成：{summary['meta']['generated_at']}",
        f"- 价格：{summary['meta']['price_source']}；名义本金 {NOTIONAL:,.0f} 元/笔",
        f"- 成本（基准）：{summary['meta']['cost_model'].split(';')[0].strip()}",
        f"- 数据覆盖：{cv['computable']}/{cv['universe']} = {cv['coverage_ratio']:.2%}；"
        f"不可计算 bar {cv['not_computable_bar_share']:.4%}；"
        f"期末未平仓 {cv['open_positions']} 笔（不计入已实现）",
        "",
        "## 结论", "",
        f"**{summary['conclusion']}**", "",
        "## 核心指标（基准情景）", "",
        "| 指标 | 结果 |",
        "|---|---|",
        f"| 总交易笔数 | {b['n']:,} |",
        f"| 总毛收益 | {pct(b['gross_total'])}（{b['gross_total_amount']:,.0f} 元） |",
        f"| 总交易成本 | {pct(b['cost_total'])}（{b['cost_total_amount']:,.0f} 元） |",
        f"| 总净收益 | {pct(b['net_total'])}（{b['net_total_amount']:,.0f} 元） |",
        f"| 单笔净期望 | {pct(b['net_expectancy'], 4)} |",
        f"| 单笔净收益中位数 | {pct(b['net_median'], 4)} |",
        f"| 胜率（净） | {pct(b['net_win_rate'])} |",
        f"| 盈亏比（净） | {b['payoff']:.3f} |",
        f"| 盈利年份 / 完整年份 | {summary['primary']['years_positive']} / "
        f"{summary['primary']['years_denominator']}（{pct(summary['primary']['years_positive_share'])}） |",
        f"| 最大回撤（等额 NAV） | {pct(c['mdd_l1'])} |",
        f"| 最长连续亏损 | {c['max_consec_loss']} 笔 |",
        f"| 95% CI（净期望） | [{pct(b['ci95'][0], 4)}, {pct(b['ci95'][1], 4)}] |",
        f"| 剔除前 1% 后净期望 | {pct(b['ex_top1_expectancy'], 4)} |",
        f"| 成本拖累（占毛收益） | {pct(b['cost_drag_share'])} |",
        "",
        "## 成本敏感性", "",
        "| 情景 | 佣金/滑点 | 净期望 | 净中位数 | 胜率 | 盈亏比 | 净总收益 | CI 下限 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = dict(base="0.030%/边 + 10bp", cons="0.100%/边 + 20bp",
                  stress="0.300%/边 + 30bp")
    for sc in SCENARIOS:
        x = m[sc]
        lines.append(
            f"| {sc} | {labels[sc]} | {pct(x['net_expectancy'], 4)} | "
            f"{pct(x['net_median'], 4)} | {pct(x['net_win_rate'])} | "
            f"{x['payoff']:.3f} | {pct(x['net_total'])} | {pct(x['ci95'][0], 4)} |")
    lines += ["", "## 年度净收益（基准情景）", "",
              "| 年份 | 笔数 | 毛收益 | 成本 | 净收益 | 净收益(元) | 胜率 |",
              "|---|---|---|---|---|---|---|"]
    for row in annual["base"]:
        tag = "（YTD）" if row["year"] == 2026 else ""
        lines.append(
            f"| {row['year']}{tag} | {row['n']} | {pct(row['gross_total'])} | "
            f"{pct(row['cost_total'])} | {pct(row['net_total'])} | "
            f"{row['net_amount']:,.0f} | {pct(row['win_rate'])} |")
    lines += ["", "## 验收门槛", "", "| 编号 | 结果 |", "|---|---|"]
    for k, v in summary["gates"].items():
        lines.append(f"| {k} | {'通过' if v else '未通过'} |")
    lines += [
        "",
        "## 口径声明",
        "",
        "- 本结果为 **L1 策略盈利性**：逐笔等额、无限资金，不模拟涨跌停、停牌顺延、资金占用与容量。",
        "- 信号与收益使用前复权价格；成本基数回到 `raw_price = qfq × f` 的原始成交金额。",
        "- 所有成交均为日线代理，非逐笔订单回放；L1 不判定可成交性（属于 L2）。",
        "- ST 历史状态未知不影响 L1（L1 不做涨跌停判定）；未以当前 ST 名单过滤历史样本。",
        f"- 最大回撤按协议 §4.6 的等额 NAV；单仓复利曲线 MDD 仅作参考："
        f"{pct(curves['base']['mdd_compound'])}。",
    ]
    with open(os.path.join(OUT_DIR, "BACKTEST_REPORT_B_L1.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
