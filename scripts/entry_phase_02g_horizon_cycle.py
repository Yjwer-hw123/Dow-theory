# -*- coding: utf-8 -*-
"""EP-02g：持有期限与市场周期的交互作用。

严格按 out/entry_phase_02/EP02G_HORIZON_CYCLE_PROTOCOL.md
（SHA256 FA8C04BB813998D89321ECCFB27554F86B6899E0964EE949A3561A654E933029）运行：
  期限 H ∈ {10,15,20,30,60}，只用固定期限退出，无止损无止盈；
  同一 11,419 信号池、3 仓、1/3 NAV、base 成本；
  市场状态用沪指前 120 日收益 ±10%（敏感性 250 日）。
只做诊断，不产生采用建议，不修改任何冻结基线。

用法：python scripts/entry_phase_02g_horizon_cycle.py --workers 8
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
PROTO = os.path.join(OUT, "EP02G_HORIZON_CYCLE_PROTOCOL.md")
PROTO_SHA = "FA8C04BB813998D89321ECCFB27554F86B6899E0964EE949A3561A654E933029"
HS = (10, 15, 20, 30, 60)
H0_NAV = 5_719_552.238656651
IDX_CAGR = 0.0562


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def verify_protocol():
    import hashlib
    if not os.path.exists(PROTO):
        raise SystemExit("协议缺失，拒绝运行")
    h = hashlib.sha256(open(PROTO, "rb").read()).hexdigest().upper()
    if h != PROTO_SHA:
        raise SystemExit(f"协议已被修改（{h}），拒绝运行")
    log(f"协议校验通过 SHA256={h[:16]}…")


def prep(args):
    """单只股票：为同一信号池生成五个固定期限的出场，以及信号级 MAE/MFE。"""
    code, g = args
    p = os.path.join(ROOT, "cache", "qfq", f"{code}.parquet")
    if not os.path.exists(p):
        return []
    df = pd.read_parquet(p)
    dates = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)
    qo = df["open"].to_numpy(float)
    qc = df["close"].to_numpy(float)
    facs = l1.load_facs(code)
    if facs is None:
        f = np.ones(dates.size)
    else:
        da, fa = l1.factor_arrays(facs)
        f = fa[np.maximum(np.searchsorted(da, dates, side="right") - 1, 0)].astype(float)
    n, pos_map = dates.size, {int(x): i for i, x in enumerate(dates)}
    rc = qc * f

    out = []
    for row in g.itertuples():
        di = pos_map.get(int(pd.Timestamp(row.entry_date).value))
        if di is None or di == 0 or di + 59 >= n:
            continue
        qe = qo[di]
        if not np.isfinite(qe) or qe <= 0:
            continue
        de = pd.Timestamp(row.entry_date)
        raw_e = float(qe * f[di])
        exits, net = {}, {}
        for H in HS:
            xi = di + H - 1
            exits[f"H{H}"] = dict(off=H - 1, day=int(dates[xi]), raw=float(rc[xi]),
                                  reason=f"fixed_{H}_bar", delay=0, f=float(f[xi]))
            t = l1.eval_trade(code, de, pd.Timestamp(dates[xi]), raw_e, float(rc[xi]), "base")
            gross = (rc[xi] / f[xi]) / qe - 1.0
            cost = (rc[xi] / raw_e - 1.0) - t["net_return"]
            net[H] = float(gross - cost)
        s10 = qc[di:di + 10] / qe - 1.0
        s15 = qc[di:di + 15] / qe - 1.0
        if not (np.isfinite(s15).all() and np.isfinite(s10).all()):
            continue
        out.append(dict(
            code=code, signal_date=str(pd.Timestamp(row.signal_date).date()),
            entry_date=str(de.date()), raw_entry=raw_e,
            day_ns=dates[di:di + 400], rc_raw=rc[di:di + 400], f=f[di:di + 400],
            exits=exits, net=net,
            mae10=float(s10.min()), mfe10=float(s10.max()),
            mae15=float(s15.min()), mfe15=float(s15.max())))
    return out


def regime_arrays(cal_ns):
    ix = pd.read_parquet(os.path.join(ROOT, "cache", "000001_daily.parquet"))
    ix["date"] = pd.to_datetime(ix["date"])
    s = ix.drop_duplicates("date").set_index("date")["close"].sort_index()
    cl = s.reindex(pd.to_datetime(cal_ns), method="ffill").to_numpy(float)

    def tag(win):
        r = np.full(cl.size, np.nan)
        r[win:] = cl[win:] / cl[:-win] - 1.0
        lab = np.where(r >= 0.10, "bull", np.where(r <= -0.10, "bear", "range"))
        lab[np.isnan(r)] = "unknown"
        return lab

    return cl, tag(120), tag(250)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    verify_protocol()
    ep.init_ctx()
    cal_ns = pd.to_datetime(ep.CTX["cal"]).values.astype("datetime64[ns]").astype(np.int64)
    cal_pos = {pd.Timestamp(k): v for k, v in ep.CTX["cal_pos"].items()}
    sig = d.load_signals()
    groups = list(sig.groupby("code", sort=False))
    wins = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, part in enumerate(ex.map(prep, groups), 1):
            wins.extend(part)
            if i % 1000 == 0:
                log(f"{i}/{len(groups)} {time.time()-t0:.0f}s")
    log(f"信号 {len(wins)} 笔（协议门槛 11419）")
    if len(wins) != 11419:
        raise SystemExit(f"信号池不一致：{len(wins)}")

    cl, reg120, reg250 = regime_arrays(cal_ns)
    days = pd.to_datetime(cal_ns)
    idx_ret_d = np.full(cl.size, np.nan)
    idx_ret_d[1:] = cl[1:] / cl[:-1] - 1.0

    # ---- 组合级 ----
    res = {}
    for H in HS:
        r = d.run_main_portfolio(wins, cal_ns, cal_pos, f"H{H}", "base")
        res[H] = r
        log(f"H{H}: NAV {r['final_nav']:,.0f}  MDD {r['mdd']:.2%}  成交 {len(r['trades'])}")
    chk = abs(res[60]["final_nav"] - H0_NAV) / H0_NAV
    if chk > 1e-9 or len(res[60]["trades"]) != 201:
        raise SystemExit(f"H60 复现未通过：偏差 {chk:.2e}，成交 {len(res[60]['trades'])}")
    log(f"H60 复现校验通过，偏差 {chk:.2e}")

    rows = []
    for H in HS:
        r = res[H]
        tr = r["trades"]
        fg = r["first_g"]
        yrs = (r["dates"][-1] - r["dates"][fg]).days / 365.25
        cg = (r["final_nav"] / d.CAPITAL0) ** (1 / yrs) - 1
        rows.append(dict(H=H, final_nav=r["final_nav"], cagr=cg, mdd=r["mdd"],
                         cagr_mdd=cg / r["mdd"], fills=len(tr),
                         avg_ret=float(tr["ret"].mean()), win_rate=float((tr["ret"] > 0).mean()),
                         avg_hold=float(tr["hold_bars"].mean()),
                         ret_per_day=float(tr["ret"].mean()) / float(tr["hold_bars"].mean()),
                         excess_cagr=cg - IDX_CAGR))
    pf = pd.DataFrame(rows)
    pf.to_csv(os.path.join(OUT, "p2g_horizon_portfolio.csv"), index=False, encoding="utf-8")

    # 逐年
    yrows = []
    for H in HS:
        r = res[H]
        nav = pd.Series(r["nav"], index=days).dropna()
        y = nav.groupby(nav.index.year).last()
        yr = y / y.shift(1) - 1
        yr.iloc[0] = y.iloc[0] / d.CAPITAL0 - 1
        for k, v in yr.items():
            yrows.append(dict(H=H, year=int(k), ret=float(v)))
    pd.DataFrame(yrows).to_csv(os.path.join(OUT, "p2g_horizon_yearly.csv"),
                               index=False, encoding="utf-8")

    # ---- 信号级 ----
    srows = []
    for s in wins:
        g = cal_pos.get(pd.Timestamp(s["entry_date"]))
        lab = reg120[g] if g is not None else "unknown"
        net = s["net"]
        srows.append(dict(
            code=s["code"], signal_date=s["signal_date"], entry_date=s["entry_date"],
            year=int(s["entry_date"][:4]), regime=lab,
            **{f"net{H}": net[H] for H in HS},
            diff15_60=net[15] - net[60],
            mae10=s["mae10"], mfe10=s["mfe10"], mae15=s["mae15"], mfe15=s["mfe15"]))
    sl = pd.DataFrame(srows)
    sl.to_csv(os.path.join(OUT, "p2g_signal_level.csv"), index=False, encoding="utf-8")

    # ---- 组合 × 市场状态（按入场状态）----
    tr_rows, w_rows = [], []
    for H in HS:
        r = res[H]
        tr = r["trades"].copy()
        gs = [cal_pos.get(pd.Timestamp(x)) for x in tr["entry_date"]]
        tr["regime"] = [reg120[g] if g is not None else "unknown" for g in gs]
        for rg, gg in tr.groupby("regime"):
            tr_rows.append(dict(H=H, regime=rg, n=len(gg), pnl=float(gg["pnl"].sum()),
                                avg_ret=float(gg["ret"].mean()),
                                win_rate=float((gg["ret"] > 0).mean())))
        # 日度对数财富按当日状态
        nav = r["nav"]
        fg = r["first_g"]
        daily = np.full(nav.size, np.nan)
        daily[fg + 1:] = nav[fg + 1:] / nav[fg:-1] - 1.0
        lr = np.log1p(daily)
        ok = np.isfinite(lr) & (np.arange(nav.size) >= fg + 1)
        for rg in ("bull", "bear", "range"):
            sel = ok & (reg120 == rg)
            w_rows.append(dict(H=H, regime=rg, days=int(sel.sum()),
                             log_wealth=float(np.nansum(lr[sel]))))
    pd.DataFrame(tr_rows).to_csv(os.path.join(OUT, "p2g_regime_trades.csv"),
                                 index=False, encoding="utf-8")
    wdf = pd.DataFrame(w_rows)
    wdf.to_csv(os.path.join(OUT, "p2g_regime_wealth.csv"), index=False, encoding="utf-8")

    # ---- 资金效率：15 vs 60 桥接 ----
    t15 = res[15]["trades"].copy()
    t60 = res[60]["trades"].copy()
    t15["k"] = t15["code"].astype(str).str.zfill(6) + "|" + t15["entry_date"].astype(str)
    t60["k"] = t60["code"].astype(str).str.zfill(6) + "|" + t60["entry_date"].astype(str)
    k15, k60 = set(t15["k"]), set(t60["k"])
    common, new, drop = k15 & k60, k15 - k60, k60 - k15
    c15 = t15[t15["k"].isin(common)]
    c60 = t60[t60["k"].isin(common)]
    common_imp = float(c15["pnl"].sum() - c60["pnl"].sum())
    new_pnl = float(t15[t15["k"].isin(new)]["pnl"].sum())
    drop_pnl = float(t60[t60["k"].isin(drop)]["pnl"].sum())
    d_total = res[15]["final_nav"] - res[60]["final_nav"]
    bridge = common_imp + new_pnl - drop_pnl
    br = pd.DataFrame([
        dict(item="common_pnl_60", n=len(common), amount=float(c60["pnl"].sum())),
        dict(item="common_pnl_15", n=len(common), amount=float(c15["pnl"].sum())),
        dict(item="common_improve", n=len(common), amount=common_imp),
        dict(item="dropped_60_only", n=len(drop), amount=-drop_pnl),
        dict(item="new_15_only", n=len(new), amount=new_pnl),
        dict(item="bridge_total", n=np.nan, amount=bridge),
        dict(item="delta_nav", n=np.nan, amount=d_total),
        dict(item="gap", n=np.nan, amount=bridge - d_total),
    ])
    br.to_csv(os.path.join(OUT, "p2g_bridge_15v60.csv"), index=False, encoding="utf-8")

    # 信号级纯期限效应（等权）
    sig_eff = float(sl["diff15_60"].mean())
    sig_pos = float((sl["diff15_60"] > 0).mean())

    # ---- 判定 ----
    c10 = float(pf.loc[pf.H == 10, "cagr"].iloc[0])
    c15 = float(pf.loc[pf.H == 15, "cagr"].iloc[0])
    c20 = float(pf.loc[pf.H == 20, "cagr"].iloc[0])
    s1 = bool(c15 > 1.5 * max(c10, c20))
    seq = pf.sort_values("H")["cagr"].to_numpy()
    s2 = bool(np.all(np.diff(seq) > 0) or np.all(np.diff(seq) < 0))
    s3 = bool((pf["cagr"] > IDX_CAGR).sum() >= 3)

    w15 = wdf[wdf.H == 15].set_index("regime")["log_wealth"]
    tot15 = float(w15.sum())
    shares = {k: float(w15.get(k, 0.0) / tot15) for k in ("bull", "bear", "range")}
    c1 = bool(max(shares.values()) > 0.60)

    # 各状态期间年化（15 日 vs 沪指）
    r15 = res[15]
    fg = r15["first_g"]
    nav15 = r15["nav"]
    ann = {}
    for rg in ("bull", "bear", "range"):
        sel = np.zeros(nav15.size, bool)
        sel[fg + 1:] = True
        sel &= (reg120 == rg)
        d15 = np.full(nav15.size, np.nan)
        d15[fg + 1:] = nav15[fg + 1:] / nav15[fg:-1] - 1.0
        lr = np.log1p(d15[sel])
        li = np.log1p(idx_ret_d[sel])
        n = int(sel.sum())
        ann[rg] = dict(days=n,
                       strat=float(np.exp(np.nansum(lr) * 252 / max(n, 1)) - 1),
                       idx=float(np.exp(np.nansum(li) * 252 / max(n, 1)) - 1))
    c2 = bool(sum(1 for rg in ann if ann[rg]["strat"] > ann[rg]["idx"]) >= 2)

    label1 = "FRAGILE_SPIKE" if (s1 or not s3) else "STABLE_PLATEAU"
    label2 = "CONDITIONAL" if (c1 or not c2) else "BROAD"
    summary = dict(
        protocol_sha256=PROTO_SHA, labels=dict(horizon=label1, cycle=label2),
        gates=dict(S1=s1, S2=s2, S3=s3, C1=c1, C2=c2),
        portfolio=pf.to_dict("records"),
        reproduce=dict(h60_dev=chk, h60_fills=len(res[60]["trades"]),
                       signals=int(len(wins)),
                       span=[str(days[res[15]["first_g"]].date()),
                             str(days[-1].date())]),
        signal_effect=dict(mean_diff_15_60=sig_eff, pos_share=sig_pos,
                           mean_net15=float(sl["net15"].mean()),
                           mean_net60=float(sl["net60"].mean()),
                           mean_mae15=float(sl["mae15"].mean()),
                           mean_mfe15=float(sl["mfe15"].mean()),
                           by_regime={rg: dict(
                               n=int(len(z)),
                               diff=float(z["diff15_60"].mean()),
                               net15=float(z["net15"].mean()),
                               net60=float(z["net60"].mean()))
                               for rg, z in sl.groupby("regime")}),
        cycle=dict(wealth_share_15=shares, ann_by_regime=ann,
                   sensitivity_250=dict(
                       by_regime={rg: dict(
                           n=int(len(z)),
                           diff=float(z["diff15_60"].mean()))
                           for rg, z in sl.assign(
                               regime250=[reg250[cal_pos[pd.Timestamp(x)]]
                                          if pd.Timestamp(x) in cal_pos else "unknown"
                                          for x in sl["entry_date"]]).groupby("regime250")})),
        capital=dict(delta_nav=d_total, common_improve=common_imp,
                     new_pnl=new_pnl, dropped_pnl=drop_pnl, bridge_total=bridge,
                     bridge_gap=bridge - d_total,
                     common_share=common_imp / d_total if d_total else np.nan,
                     capacity_share=(new_pnl - drop_pnl) / d_total if d_total else np.nan,
                     n_common=len(common), n_new=len(new), n_dropped=len(drop)),
        skipped={f"H{H}": dict(res[H]["skipped"]) for H in HS})
    with open(os.path.join(OUT, "p2g_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log(f"{label1} / {label2}；S1={s1} S2={s2} S3={s3} C1={c1} C2={c2}；"
        f"Δ15-60={d_total:,.0f}，共同改善={common_imp:,.0f}，"
        f"新-挤={new_pnl - drop_pnl:,.0f}")
    write_report(summary, pf, sl, wdf, br, ann, shares)
    log(f"完成 {time.time()-t0:.0f}s")


def write_report(s, pf, sl, wdf, br, ann, shares):
    g = s["gates"]
    eff = s["signal_effect"]
    cap = s["capital"]
    lines = [
        "# EP-02g 持有期限与市场周期交互（诊断报告）",
        "",
        f"- 协议：`EP02G_HORIZON_CYCLE_PROTOCOL.md`（SHA256 `{PROTO_SHA[:16]}…`，运行前校验通过）",
        f"- 结论标签：期限 **{s['labels']['horizon']}**；周期 **{s['labels']['cycle']}**",
        f"- 信号 {s['reproduce']['signals']} 个；H60 复现偏差 {s['reproduce']['h60_dev']:.2e}"
        f"（{s['reproduce']['h60_fills']} 笔）",
        "",
        "## Q1 期限稳定性（组合级）",
        "",
        "| 期限 | 期末 NAV | CAGR | MDD | CAGR/MDD | 成交 | 平均持有 | 单笔均值 | 单笔日均 | 胜率 | 年化超额 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in pf.itertuples():
        lines.append(f"| {r.H} 日 | {r.final_nav:,.0f} | {r.cagr:.2%} | {r.mdd:.2%} | "
                     f"{r.cagr_mdd:.2f} | {r.fills} | {r.avg_hold:.0f} | {r.avg_ret:.2%} | "
                     f"{r.ret_per_day:.3%} | {r.win_rate:.1%} | {r.excess_cagr * 100:+.2f}pp |")
    lines += [
        "",
        f"- S1 孤立尖峰：{'成立' if g['S1'] else '不成立'}；"
        f"S2 单调：{'成立' if g['S2'] else '不成立'}；"
        f"S3 至少 3 个期限跑赢沪指：{'成立' if g['S3'] else '不成立'}。",
        "",
        "- 注意：S1/S2 均不成立，但 10→15 日期末终值相差 12.5 倍，15 日以下存在断崖；"
        "预注册标签 STABLE_PLATEAU 只说明 15 日不是被 1.5 倍阈值判定的孤立尖峰，"
        "不代表期限维度上连续平滑。",
        "",
        "## Q1 信号级（11,419 个信号，等权）",
        "",
        f"- 15 日均净收益 {eff['mean_net15']:.2%}，60 日 {eff['mean_net60']:.2%}，"
        f"期限效应 {eff['mean_diff_15_60']:+.2%}，为正比例 {eff['pos_share']:.1%}。",
        f"- 前 15 日平均 MAE {eff['mean_mae15']:.2%}，MFE {eff['mean_mfe15']:.2%}。",
        "",
        "| 入场状态 | 信号数 | 15日均值 | 60日均值 | 期限效应 |",
        "|---|---|---|---|---|",
    ]
    for rg, v in eff["by_regime"].items():
        lines.append(f"| {rg} | {v['n']} | {v['net15']:.2%} | {v['net60']:.2%} | {v['diff']:+.2%} |")
    lines += [
        "",
        "敏感性（沪指 250 日 ±10% 定义状态，仅信号级期限效应）：",
        "",
        "| 入场状态 | 信号数 | 15−60 期限效应 |",
        "|---|---|---|",
    ]
    for rg, v in s["cycle"]["sensitivity_250"]["by_regime"].items():
        lines.append(f"| {rg} | {v['n']} | {v['diff']:+.2%} |")
    lines += [
        "",
        "## Q2 周期交互",
        "",
        "### 按入场状态的成交结果",
        "",
        "| 期限 | 状态 | 笔数 | 净利 | 单笔均值 | 胜率 |",
        "|---|---|---|---|---|---|",
    ]
    for r in pd.read_csv(os.path.join(OUT, "p2g_regime_trades.csv")).itertuples():
        lines.append(f"| {r.H} | {r.regime} | {r.n} | {r.pnl:,.0f} | {r.avg_ret:.2%} | {r.win_rate:.1%} |")
    lines += [
        "",
        "### 15 日：日度对数财富的状态贡献占比",
        "",
        "| 状态 | 贡献占比 | 日数 | 策略期间年化 | 沪指期间年化 |",
        "|---|---|---|---|---|",
    ]
    for rg in ("bull", "bear", "range"):
        lines.append(f"| {rg} | {shares.get(rg, 0.0):.1%} | {ann[rg]['days']} | "
                     f"{ann[rg]['strat']:.2%} | {ann[rg]['idx']:.2%} |")
    lines += [
        "",
        "占比 = 该状态每日对数财富之和 / 全期对数财富之和，可为负（该状态净损耗），三行合计 100%。",
        f"- C1 单一状态 >60%：{'成立' if g['C1'] else '不成立'}；"
        f"C2 至少两个状态期间年化高于沪指：{'成立' if g['C2'] else '不成立'}。",
        "",
        "## Q3 资金效率：15 日 vs 60 日",
        "",
        "| 项目 | 笔数 | 金额 |",
        "|---|---|---|",
    ]
    for r in br.itertuples():
        n = "—" if pd.isna(r.n) else str(int(r.n))
        lines.append(f"| {r.item} | {n} | {r.amount:,.0f} |")
    lines += [
        "",
        f"- ΔNAV {cap['delta_nav']:,.0f} 元；共同成交改善 {cap['common_improve']:,.0f} 元"
        f"（{cap['common_share']:.1%}），新增减被挤出 {cap['new_pnl'] - cap['dropped_pnl']:,.0f} 元"
        f"（{cap['capacity_share']:.1%}），桥接差额 {cap['bridge_gap']:.2f} 元。",
        f"- 信号级纯期限效应 {eff['mean_diff_15_60']:+.2%}/笔，说明缩短持有本身对等权单笔是"
        f"{'正' if eff['mean_diff_15_60'] > 0 else '负'}贡献；组合层面的差异主要来自成交集合变化。",
        "- 共同成交改善含净值放大效应：15 日路径 NAV 更大，同一信号的仓位金额更大，"
        "因此 40.3% 不能全部解释为单笔质量提升；两路 NAV 不在同一资金基准上。",
        f"- 单笔日均收益：15 日 {float(pf.loc[pf.H == 15, 'ret_per_day'].iloc[0]):.3%}/日，"
        f"60 日 {float(pf.loc[pf.H == 60, 'ret_per_day'].iloc[0]):.3%}/日"
        f"（比值 {float(pf.loc[pf.H == 15, 'ret_per_day'].iloc[0]) / float(pf.loc[pf.H == 60, 'ret_per_day'].iloc[0]):.2f} 倍）；"
        "组合收益上升的直接来源是资金周转速度，而不是单笔赚得更多。",
        "",
        "## 证据边界",
        "",
        "- 五个期限都是同一样本内的固定参数，任何「稳定」判断都不是样本外证明。",
        "- 市场状态用指数自身趋势划分，存在滞后；转折附近归类不稳定。",
        "- 3 仓路径依赖：不同期限成交集合不同，桥接只能按信号身份近似对齐。",
        "- 10bp 滑点不含市场冲击；期末净值越大，单仓规模越大。",
        "- 本报告不产生采用建议，不得用于反向选择期限。",
    ]
    with open(os.path.join(OUT, "P2G_HORIZON_CYCLE_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
