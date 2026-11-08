# -*- coding: utf-8 -*-
"""ENTRY-PHASE-02 L2：B 类策略在真实交易约束与 100 万资金下的可交易性验证。

严格按冻结协议 v0.3 §5 实现，不改信号、不改交易规则：
- 初始资金 1,000,000 元；最大持仓 20 笔；单笔 = 买入时组合最新 NAV 的 5%；
- 无杠杆、不加仓、不调仓；同日多信号按 SHA256(seed|信号日|代码) 排序；
- 买入：信号后下一有效交易日开盘，开盘涨停/一字涨停不可买，顺延最多 10 个市场交易日；
- 卖出：第 61 个有效 bar 起开盘价，开盘跌停不可卖，不设强制期限；
- 涨跌停：P_ref 用除权参考价（不复权口径），0.01 元四舍五入；
- ST_UNKNOWN：主口径 5%，敏感性 10%（两套分别跑）；
- 容量：ADV_APPROX = 信号日前 20 个有效 bar 的 mean(volume×close)，CAP ∈ {0.25%, 0.5%, 1%}；
- 成本：沿用 L1 基准（0.030%/边 + 10bp），另跑保守/压力情景。

用法：
    python scripts/entry_phase_02_l2.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02_l1 as l1  # noqa: E402
import scripts.uptrend_detector as ud  # noqa: E402
from scripts.step1_trend_validation import RAW_DIR  # noqa: E402

OUT_DIR = os.path.join(ROOT, "out", "entry_phase_02")
TRADES_L1 = os.path.join(OUT_DIR, "trades_b_l1.csv")

SEED = 20261107
CAPITAL0 = 1_000_000.0
MAX_POS = 20
POS_PCT = 0.05
ENTRY_WINDOW = 10          # 市场交易日
SELL_BAR = 61              # 第 61 个有效 bar 开盘卖
WIN_DAYS = 400             # 单信号窗口（市场日）
CAPS = (0.0025, 0.005, 0.01)
ST_RATES = (0.05, 0.10)

CAL_NS: np.ndarray | None = None
CAL_POS: dict = {}
NCAL = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hash_key(sig_date: str, code: str) -> str:
    return hashlib.sha256(f"{SEED}|{sig_date}|{code}".encode()).hexdigest()


def round2(x):
    return np.floor(np.asarray(x, dtype=np.float64) * 100.0 + 0.5) / 100.0


def build_stock(code: str, tl: pd.DataFrame) -> list[dict]:
    """一只股票的所有 B 信号 → 逐信号紧凑窗口（市场日对齐）。"""
    raw = pd.read_parquet(os.path.join(RAW_DIR, f"{code}.parquet"))
    facs = l1.load_facs(code)
    A = ud.prepare_stock(raw, facs)
    if A is None:
        return []
    dns = A["dates"].astype("datetime64[ns]").astype(np.int64)
    # 原始量价按 A 的 bar 对齐
    r = raw.copy()
    r["date"] = pd.to_datetime(r["date"]).values.astype("datetime64[ns]")
    r = r.drop_duplicates("date").set_index("date")
    dates = pd.DatetimeIndex(A["dates"])
    vol = r["volume"].reindex(dates).to_numpy(float) if "volume" in r else np.ones(dns.size)
    amt = r["amount"].reindex(dates).to_numpy(float) if "amount" in r else vol * A["c"]
    ro_raw = r["open"].reindex(dates).to_numpy(float)
    rc_raw = r["close"].reindex(dates).to_numpy(float)
    ro_raw = np.where(np.isfinite(ro_raw), ro_raw, 0.0)
    rc_raw = np.where(np.isfinite(rc_raw), rc_raw, 0.0)
    if not np.isfinite(amt).all():
        amt = np.where(np.isfinite(amt), amt, vol * A["c"])
    fs = pd.Series([x["f"] for x in (facs or [])],
                   index=pd.to_datetime([x["d"] for x in (facs or [])])).sort_index()
    f_at = fs.reindex(dates, method="ffill").fillna(1.0).to_numpy(float)

    out = []
    for row in tl.itertuples():
        sd = pd.Timestamp(row.signal_date)
        si = int(np.searchsorted(dns, sd.value, side="right") - 1)
        if si < 0 or (si >= 0 and dns[si] != sd.value and sd.normalize() != pd.Timestamp(dns[si]).normalize()):
            continue
        adv_n = 0
        if si >= 20:
            adv = float(np.nanmean(vol[si - 20:si] * rc_raw[si - 20:si]))
            adv_n = 20
        elif si >= 15:
            adv = float(np.nanmean(vol[:si] * rc_raw[:si]))
            adv_n = si
        else:
            adv = np.nan
        g0 = CAL_POS.get(pd.Timestamp(row.entry_date))
        if g0 is None:
            continue
        g1 = min(g0 + WIN_DAYS - 1, NCAL - 1)
        day_ns = CAL_NS[g0:g1 + 1]
        idx = np.searchsorted(dns, day_ns, side="right") - 1
        has = idx >= 0
        has[has] &= dns[idx[has]] == day_ns[has]
        idx = np.maximum(idx, 0)
        prev = np.maximum(idx - 1, 0)
        fi, fp = f_at[idx], f_at[prev]
        ref = np.where(fi == fp, rc_raw[prev], rc_raw[prev] * fi / fp)
        ro, rc = ro_raw[idx], rc_raw[idx]
        data = dict(
            code=code, signal_date=str(pd.Timestamp(row.signal_date).date()),
            entry_date=str(pd.Timestamp(row.entry_date).date()),
            exit_l1=str(pd.Timestamp(row.exit_date).date()),
            g0=g0, adv=adv, adv_n=adv_n, si=si,
            has=has.astype(np.uint8),
            ro=ro.astype(np.float32), rc=rc.astype(np.float32),
            f=fi.astype(np.float32),
            buy5=(has & (ro >= round2(ref * 1.05) - 1e-9)).astype(np.uint8),
            buy10=(has & (ro >= round2(ref * 1.10) - 1e-9)).astype(np.uint8),
            sell5=(has & (ro <= round2(ref * 0.95) + 1e-9)).astype(np.uint8),
            sell10=(has & (ro <= round2(ref * 0.90) + 1e-9)).astype(np.uint8),
        )
        out.append(data)
    return out


def run(sigs: list[dict], cap: float, st_rate: float, cost: str) -> dict:
    slip = l1.SLIP[cost]
    comm = l1.COMM[cost]
    buy_blk = "buy5" if st_rate == 0.05 else "buy10"
    sell_blk = "sell5" if st_rate == 0.05 else "sell10"

    dates = pd.to_datetime(ep.CTX["cal"])
    cash = CAPITAL0
    pos: dict[str, dict] = {}
    pend: list[dict] = []
    by_day: dict[int, list[dict]] = {}
    for s in sigs:
        by_day.setdefault(s["g0"], []).append(s)

    nav_prev = CAPITAL0
    daily = np.full(NCAL, np.nan)
    npos = np.zeros(NCAL, dtype=np.int32)
    invested = np.zeros(NCAL)
    cash_rec = np.zeros(NCAL)
    trades: list[dict] = []
    reasons: dict[str, int] = {}

    def add_reason(k: str) -> None:
        reasons[k] = reasons.get(k, 0) + 1

    for g in range(NCAL):
        # ---------- 卖出 ----------
        for code in list(pos.keys()):
            p = pos[code]
            s = p["sig"]
            i = g - s["g0"]
            if i <= 0 or i >= s["has"].size:
                continue
            if not s["has"][i]:
                continue
            if s["f"][i] != p["f_last"]:
                p["shares"] *= float(p["f_last"]) / float(s["f"][i])
                p["f_last"] = s["f"][i]
            p["last_c"] = float(s["rc"][i])
            p["bars"] += 1
            if p["bars"] >= SELL_BAR and not s[sell_blk][i]:
                gross = p["shares"] * float(s["ro"][i])
                amt = gross * (1.0 - slip)
                fees = (max(amt * comm, l1.MIN_COMM)
                        + amt * l1.stamp_rate(dates[g], "sell")
                        + l1.transfer_fee(code, dates[g], "sell", amt,
                                          p["shares"], cost))
                cash += amt - fees
                p["exit_g"] = g
                p["exit_date"] = str(dates[g].date())
                p["exit_reason"] = "ok"
                p["exit_price"] = float(s["ro"][i])
                p["proceeds"] = amt - fees
                p["pnl"] = amt - fees - p["cost_in"]
                trades.append(p)
                del pos[code]

        # ---------- 买入 ----------
        nav_now = nav_prev
        cand = list(pend)
        pend = []
        for s in sorted(by_day.get(g, []), key=lambda x: x["hkey"]):
            cand.append(dict(sig=s, deadline=s["g0"] + ENTRY_WINDOW - 1, last="new"))
        cand.sort(key=lambda o: o["sig"]["hkey"])
        for o in cand:
            s = o["sig"]
            code = s["code"]
            if g > o["deadline"]:
                add_reason(o["last"])
                continue
            if code in pos:
                add_reason("duplicate_open")
                continue
            if len(pos) >= MAX_POS:
                o["last"] = "max_positions"
                pend.append(o)
                continue
            i = g - s["g0"]
            if i < 0 or i >= s["has"].size:
                o["last"] = "entry_timeout"
                pend.append(o)
                continue
            if not s["has"][i]:
                o["last"] = "suspension"
                pend.append(o)
                continue
            if s[buy_blk][i]:
                o["last"] = "limit_up"
                pend.append(o)
                continue
            target = POS_PCT * nav_now
            if not np.isfinite(s["adv"]):
                add_reason("adv_insufficient")
                continue
            cap_amt = s["adv"] * cap
            if cash < target:
                o["last"] = "capital_insufficient"
                pend.append(o)
                continue
            notional = min(target, cap_amt)
            shares = notional / float(s["ro"][i])
            amt = notional * (1.0 + slip)
            fees = (max(amt * comm, l1.MIN_COMM)
                    + l1.transfer_fee(code, dates[g], "buy", amt, shares, cost))
            cash -= amt + fees
            pos[code] = dict(
                sig=s, code=code, shares=shares, notional=notional,
                f_last=float(s["f"][i]), last_c=float(s["rc"][i]),
                entry_g=g, entry_date=str(dates[g].date()),
                entry_price=float(s["ro"][i]), bars=1,
                cost_in=amt + fees, partial=bool(notional < target - 1e-6),
            )

        # ---------- 盯市 ----------
        val = sum(p["shares"] * p["last_c"] for p in pos.values())
        nav = cash + val
        daily[g] = nav
        npos[g] = len(pos)
        invested[g] = val
        cash_rec[g] = cash
        nav_prev = nav

    # 期末未平仓
    open_end = []
    for p in pos.values():
        open_end.append(dict(code=p["code"], entry_date=p["entry_date"],
                             bars=p["bars"], value=p["shares"] * p["last_c"],
                             notional=p["notional"]))
    trades_df = pd.DataFrame([dict(
        code=p["code"], signal_date=p["sig"]["signal_date"],
        entry_date=p["entry_date"], exit_date=p.get("exit_date", ""),
        hold_bars=p["bars"], notional=p["notional"],
        entry_price=p["entry_price"], exit_price=p.get("exit_price", np.nan),
        pnl=p.get("pnl", np.nan), ret=(p.get("pnl", np.nan) / p["notional"]),
        exit_reason=p.get("exit_reason", "open_at_data_end"),
    ) for p in trades])

    nav = daily.copy()
    first = int(np.argmax(npos > 0)) if (npos > 0).any() else 0
    peak = np.maximum.accumulate(nav[first:])
    dd = (peak - nav[first:]) / peak
    i_dd = int(np.argmax(dd)) + first
    i_pk = int(np.argmax(nav[first:i_dd + 1])) + first
    yrs = (dates[-1] - dates[first]).days / 365.25
    cagr = (nav[-1] / CAPITAL0) ** (1.0 / yrs) - 1 if nav[-1] > 0 else -1

    return dict(
        cap=cap, st=st_rate, cost=cost,
        final_nav=float(nav[-1]),
        total_return=float(nav[-1] / CAPITAL0 - 1),
        cagr=float(cagr),
        mdd=float(dd[int(np.argmax(dd))]),
        mdd_peak_date=str(dates[i_pk].date()),
        mdd_trough_date=str(dates[i_dd].date()),
        n_signals=len(sigs), n_filled=len(trades),
        fill_rate=len(trades) / max(len(sigs), 1),
        reasons=reasons,
        open_end=open_end, open_end_value=float(sum(x["value"] for x in open_end)),
        trades=trades_df,
        daily=pd.DataFrame(dict(date=[str(d.date()) for d in dates],
                                nav=nav, cash=cash_rec, invested=invested,
                                n_open=npos)),
        avg_util=float(np.nanmean(np.where(nav[first:] > 0,
                                           invested[first:] / nav[first:], 0))),
        max_invested=float(np.nanmax(invested)),
        max_open=int(npos.max()),
        dates=dates,
    )


def annual_table(res: dict) -> pd.DataFrame:
    d = res["daily"].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date")
    ye = d["nav"].resample("YE").last()
    ys = d["nav"].resample("YE").first()
    out = pd.DataFrame(dict(nav_end=ye,
                            ret=ye / ys.values - 1))
    out.index = out.index.year
    out = out[out.index >= 2005]
    return out.reset_index().rename(columns={"date": "year"})


def crash_2015(res: dict) -> dict:
    d = res["daily"]
    dd = pd.to_datetime(d["date"])
    w = (dd >= "2015-06-01") & (dd <= "2015-10-31")
    seg = d[w].set_index("date")["nav"]
    if seg.empty:
        return {}
    pk = seg.idxmax()
    tr = seg.loc[pk:].idxmin()
    return dict(peak_date=pk, trough_date=tr,
                loss=float(seg[pk] - seg[tr]),
                loss_pct=float(seg[tr] / seg[pk] - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    global CAL_NS, CAL_POS, NCAL
    ep.init_ctx()
    NCAL = len(ep.CTX["cal"])
    CAL_NS = np.array([pd.Timestamp(x).value for x in ep.CTX["cal"]], dtype=np.int64)
    CAL_POS = ep.CTX["cal_pos"]

    df = pd.read_csv(TRADES_L1, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    log(f"[L2] 原始 B 信号 {len(df)} 笔，{df['code'].nunique()} 只股票")
    groups = {c: g for c, g in df.groupby("code")}

    t0 = time.time()
    sigs: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_stock, c, g): c for c, g in groups.items()}
        done = 0
        for fut in as_completed(futs):
            ss = fut.result()
            for s in ss:
                s["hkey"] = hash_key(s["signal_date"], s["code"])
            sigs.extend(ss)
            done += 1
            if done % 500 == 0 or done == len(groups):
                log(f"  signals {done}/{len(groups)} got={len(sigs)} {time.time()-t0:.0f}s")
    sigs.sort(key=lambda s: (s["g0"], s["hkey"]))
    log(f"[L2] 可用信号 {len(sigs)}")

    results = {}
    for cost in ("base", "cons", "stress"):
        for st in ST_RATES:
            for cap in CAPS:
                key = f"{cost}_st{int(st*100)}_cap{cap*100:g}"
                r = run(sigs, cap, st, cost)
                results[key] = r
                log(f"  {key}: NAV={r['final_nav']:,.0f} ret={r['total_return']:.2%} "
                    f"MDD={r['mdd']:.2%} filled={r['n_filled']}")

    main_key = "base_st5_cap0.5"
    main = results[main_key]
    main["trades"].to_csv(os.path.join(OUT_DIR, "l2_trades_b_main.csv"),
                          index=False, encoding="utf-8")
    main["daily"].to_csv(os.path.join(OUT_DIR, "l2_daily_nav.csv"), index=False,
                         encoding="utf-8")
    annual_table(main).to_csv(os.path.join(OUT_DIR, "l2_annual.csv"),
                              index=False, encoding="utf-8")

    cmp = []
    for k, r in results.items():
        cmp.append(dict(run=k, cost=r["cost"], st=r["st"], cap=r["cap"],
                        final_nav=r["final_nav"], total_return=r["total_return"],
                        cagr=r["cagr"], mdd=r["mdd"], n_filled=r["n_filled"],
                        fill_rate=r["fill_rate"], avg_util=r["avg_util"],
                        max_invested=r["max_invested"],
                        open_end=len(r["open_end"])))
    pd.DataFrame(cmp).to_csv(os.path.join(OUT_DIR, "l2_scenario_compare.csv"),
                             index=False, encoding="utf-8")

    tr = main["trades"]
    win = float((tr["ret"] > 0).mean()) if len(tr) else np.nan
    pos_r = tr.loc[tr["ret"] > 0, "ret"]
    neg_r = tr.loc[tr["ret"] <= 0, "ret"]
    payoff = float(pos_r.mean() / abs(neg_r.mean())) if len(pos_r) and len(neg_r) else np.nan
    exp = float(tr["ret"].mean()) if len(tr) else np.nan
    med = float(tr["ret"].median()) if len(tr) else np.nan

    s = dict(
        meta=dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  capital0=CAPITAL0, max_positions=MAX_POS,
                  position_pct=POS_PCT, seed=SEED,
                  entry_window=ENTRY_WINDOW, sell_bar=SELL_BAR,
                  main_run=main_key),
        main=dict(final_nav=main["final_nav"],
                  total_return=main["total_return"], cagr=main["cagr"],
                  mdd=main["mdd"], mdd_peak_date=main["mdd_peak_date"],
                  mdd_trough_date=main["mdd_trough_date"],
                  n_signals=main["n_signals"], n_filled=main["n_filled"],
                  fill_rate=main["fill_rate"], reasons=main["reasons"],
                  win_rate=win, payoff=payoff, expectancy=exp, median=med,
                  avg_util=main["avg_util"], max_invested=main["max_invested"],
                  max_open=main["max_open"],
                  open_end=len(main["open_end"]),
                  open_end_value=main["open_end_value"],
                  crash_2015=crash_2015(main)),
        runs=dict((k, dict(final_nav=r["final_nav"], total_return=r["total_return"],
                           cagr=r["cagr"], mdd=r["mdd"], n_filled=r["n_filled"],
                           fill_rate=r["fill_rate"], avg_util=r["avg_util"],
                           open_end=len(r["open_end"])))
                  for k, r in results.items()),
    )
    with open(os.path.join(OUT_DIR, "l2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2, default=float)
    write_report(s, results, main)
    log(f"[L2] 完成。main NAV={main['final_nav']:,.0f} 总收益={main['total_return']:.2%} "
        f"MDD={main['mdd']:.2%} 成交={main['n_filled']}")


def write_report(s: dict, results: dict, main: dict) -> None:
    m = s["main"]
    k15 = m["crash_2015"] or {}
    lines = [
        "# EP-02 L2 实际可交易性回测报告", "",
        f"- 生成：{s['meta']['generated_at']}",
        f"- 初始资金 {CAPITAL0:,.0f} 元；最大持仓 {MAX_POS}；单笔 5% NAV；"
        f"信号 {m['n_signals']:,} 笔",
        f"- 主口径：基准成本、ST_UNKNOWN 5%、容量 0.5%", "",
        "## 直接回答", "",
        "| 问题 | 结果 |",
        "|---|---|",
        f"| 1. 期末净资产 | **{m['final_nav']:,.0f} 元** |",
        f"| 2. 累计 / 年化 | **{m['total_return']:.2%} / {m['cagr']:.2%}** |",
        f"| 3. 最大回撤 | **{m['mdd']:.2%}**（{m['mdd_peak_date']} → {m['mdd_trough_date']}） |",
        f"| 4. 胜率 / 盈亏比 / 净期望 / 中位数 | {m['win_rate']:.2%} / {m['payoff']:.3f} / "
        f"{m['expectancy']:.4%} / {m['median']:.4%} |",
        f"| 5. 实际成交 / 原始信号 | **{m['n_filled']:,} / {m['n_signals']:,}"
        f"（{m['fill_rate']:.2%}）** |",
        f"| 6. 机会损失原因 | {m['reasons']} |",
        f"| 7. 平均资金利用率 / 峰值投入 | {m['avg_util']:.2%} / {m['max_invested']:,.0f} 元 |",
        f"| 8. 期末未平仓 | {m['open_end']} 笔，市值 {m['open_end_value']:,.0f} 元 |",
        f"| 9. 2015 股灾 | {k15.get('peak_date','')} → {k15.get('trough_date','')}，"
        f"损失 {k15.get('loss', np.nan):,.0f} 元（{k15.get('loss_pct', np.nan):.2%}） |",
        "",
        "## 情景对照", "",
        "| 成本 | ST | 容量 | 期末 NAV | 累计 | 年化 | MDD | 成交 | 成交率 | 利用率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results.values():
        lines.append(f"| {r['cost']} | {r['st']:.0%} | {r['cap']:.2%} | "
                     f"{r['final_nav']:,.0f} | {r['total_return']:.2%} | {r['cagr']:.2%} | "
                     f"{r['mdd']:.2%} | {r['n_filled']:,} | {r['fill_rate']:.2%} | "
                     f"{r['avg_util']:.2%} |")
    lines += [
        "",
        "## 口径声明", "",
        "- 信号集、买卖日期、成本模型与 L1 完全一致；未删除亏损交易，未调整信号或参数。",
        "- 涨跌停参考价用不复权前收（遇因子变化按 f 比值还原除权参考价），0.01 元四舍五入。",
        "- ST_UNKNOWN 主口径 5%、敏感性 10%，均为保守代理，不是历史 ST 状态还原。",
        "- 现金分红未计入（轻微低估总收益）；拆股/送转通过因子变化调整持股数。",
        "- 期末未平仓按最后可得收盘价估值，计入期末 NAV，单列为未实现。",
        "- ADV_APPROX = 信号日前 20 个有效 bar 的 mean(volume×close)，非交易所真实成交额。",
    ]
    with open(os.path.join(OUT_DIR, "L2_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
