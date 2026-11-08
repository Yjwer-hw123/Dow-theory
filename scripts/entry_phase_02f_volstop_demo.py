# -*- coding: utf-8 -*-
"""EP-02f（情景演示，不属于冻结协议）：波动率止损 + 2.5R 止盈，3 仓。

四个变体：
  V15  ATR20 × 1.5，止盈 2.5R
  V20  ATR20 × 2.0
  V30  ATR20 × 3.0
  VS2  20 日收盘价标准差 × 2
对照：B0 不止损、60 bar 收盘卖。

执行口径与 EP-02d 主口径一致：收盘确认触发，次日开盘卖出；开盘封跌停顺延；
保留 60 bar 上限；base 成本；3 仓、每笔 1/3 NAV、同日先到先得。
本脚本只为画图，不写入任何冻结记录，结果不得作为采用依据。

用法：python scripts/entry_phase_02f_volstop_demo.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02_l1 as l1  # noqa: E402
import scripts.entry_phase_02d_stoploss as d  # noqa: E402

OUT = os.path.join(ROOT, "out", "entry_phase_02")
QFQ = os.path.join(ROOT, "cache", "qfq")
HOLD = 60
RULES = (
    ("B0", None, None),
    ("V15", "atr", 1.5),
    ("V20", "atr", 2.0),
    ("V30", "atr", 3.0),
    ("VS2", "sd", 2.0),
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def simulate(args):
    code, g = args
    p = os.path.join(QFQ, f"{code}.parquet")
    if not os.path.exists(p):
        return []
    df = pd.read_parquet(p)
    dates = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)
    qo = df["open"].to_numpy(float)
    qh = df["high"].to_numpy(float)
    ql = df["low"].to_numpy(float)
    qc = df["close"].to_numpy(float)
    facs = l1.load_facs(code)
    if facs is None:
        f = np.ones(dates.size)
    else:
        da, fa = l1.factor_arrays(facs)
        f = fa[np.maximum(np.searchsorted(da, dates, side="right") - 1, 0)].astype(float)
    n = dates.size
    pos_map = {int(x): i for i, x in enumerate(dates)}
    ro = qo * f
    rc = qc * f
    rate = d.board_limit_rate(code, dates)
    blk = d.limitdown_block(ro, rc, f, rate)
    prev_c = np.concatenate([[np.nan], qc[:-1]])
    tr = np.maximum(qh - ql, np.maximum(np.abs(qh - prev_c), np.abs(ql - prev_c)))

    out = []
    for row in g.itertuples():
        di = pos_map.get(int(pd.Timestamp(row.entry_date).value))
        if di is None or di == 0 or di + HOLD - 1 >= n:
            continue
        qe = qo[di]
        if not np.isfinite(qe) or qe <= 0:
            continue
        de = pd.Timestamp(row.entry_date)
        raw_e = float(qe * f[di])
        atr = float(np.nanmean(tr[max(0, di - 20):di]))
        sd = float(np.nanstd(qc[max(0, di - 20):di]))
        if not np.isfinite(atr) or atr <= 0:
            continue
        if not np.isfinite(sd) or sd <= 0:
            sd = atr
        exits = {"B0": dict(off=HOLD - 1, day=int(dates[di + HOLD - 1]),
                            raw=float(rc[di + HOLD - 1]), reason=f"fixed_{HOLD}_bar",
                            delay=0, f=float(f[di + HOLD - 1]))}
        for rid, mode, x in RULES:
            if rid == "B0":
                continue
            dist = (atr if mode == "atr" else sd) * x
            stop, tgt = qe - dist, qe + 2.5 * dist
            t_hit, kind = -1, ""
            for t in range(HOLD):
                if qc[di + t] <= stop:
                    t_hit, kind = t, "stop"
                    break
                if qc[di + t] >= tgt:
                    t_hit, kind = t, "tp"
                    break
            if t_hit < 0:
                exits[rid] = exits["B0"]
                continue
            j = di + t_hit + 1
            while j < n and (ro[j] <= 0 or not np.isfinite(ro[j])
                             or (kind == "stop" and blk[j])):
                j += 1
            if j >= n:
                j = di + HOLD - 1
                kind, raw_x = "fallback", float(rc[j])
            else:
                raw_x = float(ro[j])
            exits[rid] = dict(off=j - di, day=int(dates[j]), raw=raw_x,
                              reason=kind, delay=max(0, j - (di + t_hit + 1)),
                              f=float(f[j]))
        out.append(dict(
            code=code, signal_date=str(pd.Timestamp(row.signal_date).date()),
            entry_date=str(de.date()), raw_entry=raw_e,
            day_ns=dates[di:di + 400], rc_raw=rc[di:di + 400], f=f[di:di + 400],
            exits=exits, atr=atr, sd=sd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hold", type=int, default=60, help="最长持股交易日（bar）")
    ap.add_argument("--tag", type=str, default=None, help="输出文件后缀，默认沿用 60 bar 命名")
    a = ap.parse_args()
    global HOLD
    HOLD = int(a.hold)
    suffix = "" if a.tag is None else f"_{a.tag}"
    t0 = time.time()
    ep.init_ctx()
    cal_ns = pd.to_datetime(ep.CTX["cal"]).values.astype("datetime64[ns]").astype(np.int64)
    cal_pos = {pd.Timestamp(k): v for k, v in ep.CTX["cal_pos"].items()}
    sig = d.load_signals()
    groups = list(sig.groupby("code", sort=False))
    wins = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, part in enumerate(ex.map(simulate, groups), 1):
            wins.extend(part)
            if i % 1000 == 0:
                log(f"{i}/{len(groups)} {time.time()-t0:.0f}s")
    log(f"信号 {len(wins)} 笔")

    res = {}
    for rid, _m, _x in RULES:
        r = d.run_main_portfolio(wins, cal_ns, cal_pos, rid, "base")
        res[rid] = r
        log(f"{rid}: NAV {r['final_nav']:,.0f}  MDD {r['mdd']:.2%}  成交 {len(r['trades'])}")
    chk = (abs(res["B0"]["final_nav"] - 5719552.238656651) / 5719552.238656651
           if HOLD == 60 else float("nan"))
    log(f"B0 校验偏差 {chk:.2e}" if HOLD == 60 else f"HOLD={HOLD}，不适用 60 bar 基线校验")

    # ---- 沪指对照（同区间）----
    ix = pd.read_parquet(os.path.join(ROOT, "cache", "000001_daily.parquet"))
    ix["date"] = pd.to_datetime(ix["date"])
    ixs = ix.drop_duplicates("date").set_index("date")["close"].sort_index()
    dates = res["B0"]["dates"]
    x0 = res["B0"]["first_g"]
    common_end = min(dates[-1], ixs.index[-1])
    xe = int(np.searchsorted(dates.values.astype("datetime64[ns]"),
                             np.datetime64(common_end, "ns"), side="right") - 1)
    idxv = ixs.reindex(dates[x0:xe + 1], method="ffill").to_numpy(float)
    yrs_c = (dates[xe] - dates[x0]).days / 365.25
    idx_ret = float(idxv[-1] / idxv[0] - 1)
    idx_cagr = float((1 + idx_ret) ** (1 / yrs_c) - 1)
    log(f"沪指 {dates[x0].date()}→{dates[xe].date()}：{idx_ret:.1%}，CAGR {idx_cagr:.2%}")

    def yearly_ret(vals):
        base = float(vals[0])
        d = pd.DataFrame(dict(d=dates[x0:xe + 1], v=vals))
        y = d.groupby(d["d"].dt.year)["v"].last()
        r = y / y.shift(1) - 1
        r.iloc[0] = y.iloc[0] / base - 1
        return r

    idx_y = yearly_ret(idxv)
    ex_rows = []

    rows = []
    for rid, _m, _x in RULES:
        r = res[rid]
        tr = r["trades"]
        navc = r["nav"][x0:xe + 1]
        sy = yearly_ret(navc)
        for yr in idx_y.index:
            ex_rows.append(dict(rule=rid, year=int(yr), strat_ret=float(sy.loc[yr]),
                                idx_ret=float(idx_y.loc[yr]),
                                excess=float(sy.loc[yr] - idx_y.loc[yr])))
        str_ret = float(navc[-1] / navc[0] - 1)
        cagr_s = float((1 + str_ret) ** (1 / yrs_c) - 1)
        rel = (navc / navc[0]) / (idxv / idxv[0])
        exc_mdd = float(np.max(1 - rel / np.maximum.accumulate(rel)))
        yrs = (r["dates"][-1] - r["dates"][r["first_g"]]).days / 365.25
        cg = (r["final_nav"] / 1_000_000.0) ** (1 / yrs) - 1
        rows.append(dict(
            rule=rid, final_nav=r["final_nav"], total_return=r["final_nav"] / 1e6 - 1,
            mdd=r["mdd"], cagr=cg, cagr_mdd=cg / r["mdd"], fills=len(tr),
            avg_ret=float(tr["ret"].mean()) if len(tr) else np.nan,
            win_rate=float((tr["ret"] > 0).mean()) if len(tr) else np.nan,
            avg_hold=float(tr["hold_bars"].mean()) if len(tr) else np.nan,
            stop_share=float((tr["reason"] == "stop").mean()) if len(tr) else np.nan,
            tp_share=float((tr["reason"] == "tp").mean()) if len(tr) else np.nan,
            fixed_share=float((tr["reason"] == f"fixed_{HOLD}_bar").mean()) if len(tr) else np.nan,
            excess_total=float((1 + str_ret) / (1 + idx_ret) - 1),
            excess_cagr=float(cagr_s - idx_cagr),
            excess_mdd=exc_mdd,
        ))
    pd.DataFrame(ex_rows).to_csv(os.path.join(OUT, f"p2f_excess_yearly{suffix}.csv"),
                                 index=False, encoding="utf-8")
    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(OUT, f"p2f_volstop_demo_summary{suffix}.csv"),
              index=False, encoding="utf-8")
    log(sm.to_string(index=False))

    # ---- HTML 图 ----
    dates = res["B0"]["dates"]
    xs = np.arange(x0, dates.size)
    colors = {"B0": "#9e9e9e", "V15": "#ef5350", "V20": "#ff9800",
              "V30": "#42a5f5", "VS2": "#66bb6a", "IDX": "#ffffff"}
    names = {"B0": ("B0 不止损" if HOLD == 60 else f"固定 {HOLD} 日（不止损）"),
             "V15": "ATR×1.5", "V20": "ATR×2",
             "V30": "ATR×3", "VS2": "σ×2", "IDX": "沪指(归一100万)"}
    W, H1, H2, H3, PAD = 1180, 420, 190, 170, 64
    idx_line = idxv / idxv[0] * 1e6
    ymin = 0
    ymax = max(max(float(np.nanmax(res[r]["nav"][x0:])) for r in res),
               float(idx_line.max())) * 1.05
    def X(i): return PAD + (i - xs[0]) / max(1, xs[-1] - xs[0]) * (W - PAD - 24)
    def Y(v): return 28 + (1 - (v - ymin) / (ymax - ymin)) * (H1 - 48)
    svg = [f'<svg viewBox="0 0 {W} {H1+H2+H3+96}" xmlns="http://www.w3.org/2000/svg" '
           f'style="background:#111;font-family:-apple-system,Segoe UI,sans-serif">']
    svg.append(f'<text x="{PAD}" y="20" fill="#eee" font-size="14">组合净值（100 万起，3 仓，base 成本）</text>')
    for k in range(5):
        v = ymax * k / 4
        y = Y(v)
        svg.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W-24}" y2="{y:.1f}" stroke="#333"/>')
        svg.append(f'<text x="{PAD-8}" y="{y+4:.1f}" fill="#888" font-size="11" text-anchor="end">{v/1e4:.0f}万</text>')
    for rid in [r for r, _m, _x in RULES]:
        nav = res[rid]["nav"][x0:]
        pts = " ".join(f"{X(x0+i):.1f},{Y(nav[i]):.1f}" for i in range(nav.size) if np.isfinite(nav[i]))
        dash = ' stroke-dasharray="6 4"' if rid == "B0" else ""
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[rid]}" '
                   f'stroke-width="{1.8 if rid!="B0" else 1.2}"{dash}/>')
    pts = " ".join(f"{X(x0+i):.1f},{Y(idx_line[i]):.1f}"
                   for i in range(idx_line.size) if np.isfinite(idx_line[i]))
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors["IDX"]}" '
               f'stroke-width="1.1" stroke-dasharray="3 3"/>')
    yrs = pd.to_datetime(dates)
    top = H1 + 22
    top3 = top + H2 + 34
    last_y, ticks = None, []
    for i in range(x0, dates.size):
        y = yrs[i].year
        if y % 3 == 0 and y != last_y:
            ticks.append(i)
            last_y = y
    for i in ticks:
        x = X(i)
        svg.append(f'<line x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{top3+H3-10}" stroke="#222"/>')
        svg.append(f'<text x="{x:.1f}" y="{top3+H3+22}" fill="#888" font-size="11" '
                   f'text-anchor="middle">{yrs[i].year}</text>')
    lx = PAD + 10
    for rid in [r for r, _m, _x in RULES] + ["IDX"]:
        svg.append(f'<rect x="{lx}" y="{H1-8}" width="12" height="3" fill="{colors[rid]}"/>')
        svg.append(f'<text x="{lx+17}" y="{H1-2}" fill="#ccc" font-size="11">{names[rid]}</text>')
        lx += 118
    # 回撤面板
    svg.append(f'<text x="{PAD}" y="{top}" fill="#eee" font-size="13">回撤（水下深度）</text>')
    dmax = max(float(np.nanmax(1 - res[r]["nav"][x0:] / np.maximum.accumulate(res[r]["nav"][x0:])))
               for r in res)
    def Y2(v): return top + 14 + v / max(dmax, 1e-6) * (H2 - 36)
    for k in range(3):
        v = dmax * k / 2
        y = Y2(v)
        svg.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W-24}" y2="{y:.1f}" stroke="#333"/>')
        svg.append(f'<text x="{PAD-8}" y="{y+4:.1f}" fill="#888" font-size="11" text-anchor="end">{v*100:.0f}%</text>')
    for rid in [r for r, _m, _x in RULES]:
        nav = res[rid]["nav"][x0:]
        peak = np.maximum.accumulate(nav)
        dd = 1 - nav / peak
        pts = " ".join(f"{X(x0+i):.1f},{Y2(dd[i]):.1f}" for i in range(dd.size) if np.isfinite(dd[i]))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[rid]}" stroke-width="1.3"/>')
    # 相对强弱面板（策略 / 沪指，起点=1）
    svg.append(f'<text x="{PAD}" y="{top3}" fill="#eee" font-size="13">'
               f'相对沪指强弱（策略净值 / 沪指，起点=1.0）</text>')
    rels = {}
    for rid in [r for r, _m, _x in RULES]:
        nav = res[rid]["nav"][x0:xe + 1]
        rels[rid] = (nav / nav[0]) / (idxv / idxv[0])
    rlo = min(float(np.nanmin(v)) for v in rels.values())
    rhi = max(float(np.nanmax(v)) for v in rels.values())
    span = max(rhi - rlo, 1e-6)
    rlo, rhi = min(rlo, 1.0) - span * 0.06, max(rhi, 1.0) + span * 0.06
    def Y3(v): return top3 + 14 + (1 - (v - rlo) / (rhi - rlo)) * (H3 - 36)
    for k in range(4):
        v = rlo + (rhi - rlo) * k / 3
        y = Y3(v)
        svg.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W-24}" y2="{y:.1f}" stroke="#333"/>')
        svg.append(f'<text x="{PAD-8}" y="{y+4:.1f}" fill="#888" font-size="11" '
                   f'text-anchor="end">{v:.2f}×</text>')
    svg.append(f'<line x1="{PAD}" y1="{Y3(1.0):.1f}" x2="{W-24}" y2="{Y3(1.0):.1f}" '
               f'stroke="#666" stroke-width="1" stroke-dasharray="2 3"/>')
    for rid in [r for r, _m, _x in RULES]:
        v = rels[rid]
        pts = " ".join(f"{X(x0+i):.1f},{Y3(v[i]):.1f}" for i in range(v.size) if np.isfinite(v[i]))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[rid]}" stroke-width="1.3"/>')
    svg.append("</svg>")
    svg = "\n".join(svg)

    tbl = "".join(
        f"<tr><td>{names[r.rule]}</td><td>{r.final_nav/1e4:,.0f} 万</td>"
        f"<td>{r.total_return*100:.1f}%</td><td>{r.mdd*100:.1f}%</td>"
        f"<td>{r.cagr*100:.2f}%</td><td>{r.cagr_mdd:.2f}</td><td>{r.fills}</td>"
        f"<td>{r.avg_hold:.0f}</td><td>{r.stop_share*100:.0f}%</td>"
        f"<td>{r.tp_share*100:.0f}%</td><td>{r.fixed_share*100:.0f}%</td>"
        f"<td>{r.excess_total*100:+.1f}%</td><td>{r.excess_cagr*100:+.2f}%</td>"
        f"<td>{r.excess_mdd*100:.1f}%</td></tr>"
        for r in sm.itertuples())
    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>波动率止损 + 2.5R 止盈（{HOLD} 日上限，情景演示）</title>
<body style="margin:0;background:#111;color:#ddd;font-family:-apple-system,Segoe UI,sans-serif">
<div style="max-width:1220px;margin:0 auto;padding:24px">
<h2 style="color:#fff;margin:0 0 6px">波动率止损 + 2.5R 止盈 · 3 仓 · 最长 {HOLD} 个交易日</h2>
<p style="color:#999;margin:0 0 16px;font-size:13px">
收盘确认、次日开盘成交；开盘跌停顺延；最长持股 {HOLD} 个交易日；base 成本；3 仓每笔 1/3 NAV。
这是冻结协议之外的情景演示，只用于看图形，不能作为采用依据；止盈按收盘确认，实际盘中可能更早触发。</p>
{svg}
<div style="margin-top:18px">
<p style="color:#bbb;font-size:12px;margin:0 0 8px">
沪指同期（{dates[x0].date()} → {dates[xe].date()}，{yrs_c:.1f} 年）：总收益 {idx_ret*100:.1f}%，CAGR {idx_cagr*100:.2f}%。
超额口径：累计超额 =（1+策略总收益）/（1+沪指总收益）−1；年化超额 = 策略 CAGR − 沪指 CAGR；
超额 MDD = 相对强弱曲线（策略/沪指）自高点回撤。</p>
<table style="border-collapse:collapse;font-size:13px;width:100%">
<thead><tr style="color:#fff;background:#222">
<th align="left">规则</th><th>期末</th><th>总收益</th><th>MDD</th><th>CAGR</th>
<th>CAGR/MDD</th><th>成交</th><th>持有bar</th><th>止损</th><th>止盈</th><th>{HOLD}bar</th>
<th>累计超额</th><th>年化超额</th><th>超额MDD</th>
</tr></thead><tbody>{tbl}</tbody></table>
<p style="color:#888;font-size:12px;margin-top:14px">
口径：ATR20 与 20 日收盘价标准差均取入场前一日的 20 个 bar；止损距离 = k × 波动率；
止盈距入场 = 2.5 × 止损距离（2.5R）。触发后次日开盘成交，若开盘封跌停则顺延。
{("B0 与 EP-02d 冻结基线一致（偏差 " + f"{chk:.1e}" + "）。") if HOLD == 60 else f"本情景最长持股 {HOLD} 个交易日，灰线不再等于冻结 B0，仅作同规则下的期限对照。"} 沪指仅按收盘价归一，未计分红、未计成本。</p>
</div></body></html>"""
    with open(os.path.join(OUT, f"P2F_VOLSTOP_DEMO{suffix}.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"完成 {time.time()-t0:.0f}s → P2F_VOLSTOP_DEMO{suffix}.html")


if __name__ == "__main__":
    main()
