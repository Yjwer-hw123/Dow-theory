# -*- coding: utf-8 -*-
"""EP-02f 归因：ATR×2 多赚的钱从哪里来。

严格按 out/entry_phase_02/EP02F_ATTR_PROTOCOL.md（SHA256
7F2BC93357A933B82FF9A74F6C748058555121473091B7F758D19F37EBD6C3F8）运行：
  P0 = B0（60 bar 收盘卖，原始资金节律）
  P1 = ATR20×2.0 止损 + 2.5R 止盈，实际释放
  P2 = 同 P1 出场，但卖出所得冻结至该笔交易在 B0 口径下的 60 bar 回款日次日
Δtotal = Δexit + Δcap；P2 仅为诊断反事实，不可投资。

用法：python scripts/entry_phase_02f_attribution.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02_l1 as l1  # noqa: E402
import scripts.entry_phase_02d_stoploss as d  # noqa: E402
from scripts.entry_phase_02f_volstop_demo import simulate  # noqa: E402

OUT = os.path.join(ROOT, "out", "entry_phase_02")
PROTO = os.path.join(OUT, "EP02F_ATTR_PROTOCOL.md")
PROTO_SHA = "7F2BC93357A933B82FF9A74F6C748058555121473091B7F758D19F37EBD6C3F8"
P0_NAV = 5_719_552.238656651
P1_NAV = 7_629_540.673526994
RULE1, RULE2 = "B0", "V20"


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


def run_attr(win_rows, cal_ns, cal_pos, rule, lock_b0=False):
    """复刻 3 仓组合，记录逐信号竞争与完整台账；lock_b0 时卖出所得冻结至 B0 回款日。"""
    slip, comm = l1.SLIP["base"], l1.COMM["base"]
    by_entry = defaultdict(list)
    for s in win_rows:
        g = cal_pos.get(pd.Timestamp(s["entry_date"]))
        if g is None:
            continue
        s["g_entry"] = g
        by_entry[g].append(s)
    for g in by_entry:
        by_entry[g].sort(key=lambda x: (x["signal_date"], x["code"]))

    NCAL = cal_ns.size
    days = pd.to_datetime(cal_ns)
    cash_av, cash_pend = d.CAPITAL0, 0.0
    locked = []          # (release_g, amount)
    pos, pending_sell = {}, {}
    nav = np.full(NCAL, np.nan)
    npos = np.zeros(NCAL, int)
    nav_prev = d.CAPITAL0
    trades, attempts = [], []
    skipped = defaultdict(int)

    def calg(ns):
        return min(int(np.searchsorted(cal_ns, ns, side="left")), NCAL - 1)

    def fee_sell(code, dt, amt, sh):
        return (max(amt * comm, l1.MIN_COMM) + amt * l1.stamp_rate(dt, "sell")
                + l1.transfer_fee(code, dt, "sell", amt, sh, "base"))

    def fee_buy(code, dt, amt, sh):
        return max(amt * comm, l1.MIN_COMM) + l1.transfer_fee(code, dt, "buy", amt, sh, "base")

    for g in range(NCAL):
        if locked:
            keep = []
            for rg, amt in locked:
                if rg <= g:
                    cash_av += amt
                else:
                    keep.append((rg, amt))
            locked = keep
        day, dns = days[g], int(cal_ns[g])

        # 卖出
        for code in list(pending_sell.keys()):
            p = pos.get(code)
            if p is None:
                pending_sell.pop(code, None)
                continue
            ex = p["exit"]
            if ex["day"] > dns:
                continue
            if np.isfinite(ex.get("f", np.nan)) and ex["f"] != p["f_last"]:
                p["shares"] *= p["f_last"] / ex["f"]
                p["f_last"] = ex["f"]
            px = ex["raw"]
            amt = p["shares"] * px
            f = fee_sell(code, day, amt, p["shares"])
            proceeds = amt * (1.0 - slip) - f
            p["pnl"] = proceeds - p["cost_in"]
            trades.append(dict(
                code=code, signal_date=p["signal_date"], entry_date=p["entry_date"],
                exit_date=str(pd.Timestamp(ex["day"]).date()),
                entry_g=p["entry_g"], exit_g=g, entry_px=p["entry_px"],
                notional=p["notional"], shares=p["shares"], pnl=p["pnl"],
                ret=p["pnl"] / p["notional"], hold_bars=max(1, g - p["entry_g"]),
                exit_px=px, exit_reason=ex["reason"], net60=p["net60"]))
            if lock_b0 and p["release_g"] > g + 1:
                locked.append((p["release_g"], proceeds))
            else:
                cash_pend += proceeds
            del pos[code]
            pending_sell.pop(code, None)

        # 买入
        for s in by_entry.get(g, []):
            key = (s["code"], s["signal_date"])
            base = dict(code=s["code"], signal_date=s["signal_date"],
                        entry_date=s["entry_date"], entry_g=g, net60=s.get("net60", np.nan))
            if len(pos) >= d.MAIN_MAX_POS:
                skipped["max_positions"] += 1
                attempts.append(dict(base, status="skipped", reason="max_positions"))
                continue
            if s["code"] in pos:
                skipped["duplicate"] += 1
                attempts.append(dict(base, status="skipped", reason="duplicate"))
                continue
            ex = s["exits"][rule]
            target = d.MAIN_POS_PCT * nav_prev
            if cash_av < target:
                skipped["cash_insufficient"] += 1
                attempts.append(dict(base, status="skipped", reason="cash_insufficient"))
                continue
            notional = target
            px = s["raw_entry"]
            shares = notional / px
            amt = notional * (1.0 + slip)
            f = fee_buy(s["code"], day, amt, shares)
            cash_av -= amt + f
            exb = s["exits"]["B0"]
            pos[s["code"]] = dict(
                shares=shares, notional=notional, entry_px=px,
                signal_date=s["signal_date"], entry_date=s["entry_date"], entry_g=g,
                f_last=float(s["f"][0]), last_c=float(s["rc_raw"][0]), ptr=1, win=s,
                exit=ex, cost_in=amt + f, pnl=np.nan, net60=s.get("net60", np.nan),
                release_g=min(calg(exb["day"]) + 1, NCAL))
            pending_sell[s["code"]] = ex
            attempts.append(dict(base, status="filled", reason=""))

        # 盯市
        for p in pos.values():
            w = p["win"]
            dn, rcp, ff = w["day_ns"], w["rc_raw"], w["f"]
            i = p["ptr"]
            while i < dn.size and dn[i] <= dns:
                i += 1
            i -= 1
            if i >= 0 and dn[i] == dns and i < dn.size:
                if ff[i] != p["f_last"] and np.isfinite(ff[i]) and ff[i] != 0:
                    p["shares"] *= p["f_last"] / ff[i]
                    p["f_last"] = ff[i]
                if np.isfinite(rcp[i]):
                    p["last_c"] = float(rcp[i])
            p["ptr"] = max(p["ptr"], i + 1)

        val = sum(p["shares"] * p["last_c"] for p in pos.values())
        nav[g] = cash_av + cash_pend + sum(a for _, a in locked) + val
        npos[g] = len(pos)
        nav_prev = nav[g]
        cash_av += cash_pend
        cash_pend = 0.0

    first = int(np.argmax(npos > 0)) if (npos > 0).any() else 0
    nav_f = nav[first:]
    peak = np.maximum.accumulate(nav_f)
    dd = (peak - nav_f) / peak
    return dict(final_nav=float(nav[-1]), nav=nav, dates=days, npos=npos,
                first_g=first, mdd=float(dd.max()) if len(dd) else np.nan,
                trades=pd.DataFrame(trades), attempts=pd.DataFrame(attempts),
                skipped=dict(skipped))


def yearly_of(rows, col, year_col="entry_date"):
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["year"] = pd.to_datetime(df[year_col]).dt.year
    return df.groupby("year")[col].sum().to_dict()


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
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, part in enumerate(ex.map(simulate, groups), 1):
            wins.extend(part)
            if i % 1000 == 0:
                log(f"{i}/{len(groups)} {time.time()-t0:.0f}s")
    log(f"信号 {len(wins)} 笔")

    # 60 bar 续持的净收益（同一入场价、同一成本口径）
    for w in wins:
        fe = w["f"][0]
        if not np.isfinite(fe) or fe <= 0:
            w["net60"] = np.nan
            continue
        ex = w["exits"]["B0"]
        qe = w["raw_entry"] / fe
        gross = (ex["raw"] / ex["f"]) / qe - 1.0 if ex["f"] > 0 else np.nan
        t = l1.eval_trade(w["code"], pd.Timestamp(w["entry_date"]),
                          pd.Timestamp(ex["day"]), w["raw_entry"], ex["raw"], "base")
        cost = (ex["raw"] / w["raw_entry"] - 1.0) - t["net_return"]
        w["net60"] = float(gross - cost)

    p0 = run_attr(wins, cal_ns, cal_pos, RULE1)
    p1 = run_attr(wins, cal_ns, cal_pos, RULE2)
    p2 = run_attr(wins, cal_ns, cal_pos, RULE2, lock_b0=True)
    for nm, r in (("P0", p0), ("P1", p1), ("P2", p2)):
        log(f"{nm}: NAV {r['final_nav']:,.0f}  MDD {r['mdd']:.2%}  成交 {len(r['trades'])}")

    chk0 = abs(p0["final_nav"] - P0_NAV) / P0_NAV
    chk1 = abs(p1["final_nav"] - P1_NAV) / P1_NAV
    if chk0 > 1e-9 or chk1 > 1e-9 or len(p0["trades"]) != 201 or len(p1["trades"]) != 370:
        raise SystemExit(f"复现门槛未通过：P0 {chk0:.2e}/{len(p0['trades'])}，"
                         f"P1 {chk1:.2e}/{len(p1['trades'])}")
    log(f"复现校验通过：P0 {chk0:.2e} / P1 {chk1:.2e}")

    for nm, r in (("p0", p0), ("p1", p1), ("p2", p2)):
        r["trades"].to_csv(os.path.join(OUT, f"p2f_attr_trades_{nm}.csv"),
                           index=False, encoding="utf-8")

    t0s = p0["trades"].copy()
    t1s = p1["trades"].copy()
    t0s["key"] = t0s["code"] + "|" + t0s["signal_date"]
    t1s["key"] = t1s["code"] + "|" + t1s["signal_date"]
    k0, k1 = set(t0s["key"]), set(t1s["key"])
    new_keys, drop_keys, common_keys = k1 - k0, k0 - k1, k0 & k1

    # Q1 新增成交来源
    att = p0["attempts"]
    att["key"] = att["code"] + "|" + att["signal_date"]
    att_map = {r.key: r.status + ":" + r.reason for r in att.itertuples()}
    new = t1s[t1s["key"].isin(new_keys)].copy()
    new["p0_outcome"] = [att_map.get(k, "not_attempted") for k in new["key"]]
    new["p0_npos"] = [int(p0["npos"][g]) if g < p0["npos"].size else 0
                      for g in new["entry_g"]]
    new["year"] = pd.to_datetime(new["entry_date"]).dt.year
    new.to_csv(os.path.join(OUT, "p2f_attr_new_trades.csv"), index=False, encoding="utf-8")
    q1 = new["p0_outcome"].value_counts().to_dict()

    dropped = t0s[t0s["key"].isin(drop_keys)]
    dropped_pnl = float(dropped["pnl"].sum()) if len(dropped) else 0.0

    # Q2 提前退出损失
    a0 = t0s[t0s["key"].isin(common_keys)].set_index("key")
    a1 = t1s[t1s["key"].isin(common_keys)].set_index("key")
    eg0 = a0["exit_g"]
    a1r = a1.reindex(eg0.index)
    early = a1r[a1r["exit_g"] < eg0].copy()
    early["b0_exit_date"] = a0.loc[early.index, "exit_date"]
    early["cont_gain"] = early["net60"] - early["ret"]           # 每单位名义
    early["cont_amt"] = early["cont_gain"] * early["notional"]   # 元
    early["b0_hold_bars"] = a0.loc[early.index, "hold_bars"]
    early["year"] = pd.to_datetime(early["entry_date"]).dt.year
    early = early.reset_index().rename(columns={"index": "key"})
    early.to_csv(os.path.join(OUT, "p2f_attr_early_exit.csv"), index=False, encoding="utf-8")
    q2 = {k: float(early.loc[early["exit_reason"] == k, "cont_amt"].sum())
          for k in ("stop", "tp")}
    q2["n_stop"] = int((early["exit_reason"] == "stop").sum())
    q2["n_tp"] = int((early["exit_reason"] == "tp").sum())
    q2["n_lost"] = int((early["cont_gain"] > 0).sum())
    q2["n_saved"] = int((early["cont_gain"] < 0).sum())

    # Q3 新增成交价值
    new_pnl = float(new["pnl"].sum())
    new_cd = float((new["notional"] * new["hold_bars"]).sum())
    p0_cd = float((t0s["notional"] * t0s["hold_bars"]).sum())
    new_ratio = new_pnl / new_cd if new_cd > 0 else np.nan
    p0_ratio = float(t0s["pnl"].sum()) / p0_cd if p0_cd > 0 else np.nan
    losers = new[new["pnl"] < 0]
    new_pos = new[new["pnl"] > 0].sort_values("pnl", ascending=False)
    top2_share = float(new_pos["pnl"].head(2).sum() / new_pnl) if new_pnl > 0 else np.nan
    new_no06 = new[new["year"] != 2006]
    new_no06_pnl = float(new_no06["pnl"].sum())
    new_ex_top2 = float(new["pnl"].sum() - new_pos["pnl"].head(2).sum())
    new_y_pnl = new.groupby("year")["pnl"].sum()
    top_year = int(new_y_pnl.idxmax()) if len(new_y_pnl) else 0
    top_year_share = float(new_y_pnl.max() / new_pnl) if new_pnl > 0 else np.nan
    new_ex_topyear = float(new_pnl - new_y_pnl.max())

    # Q4 拆解
    d_total = p1["final_nav"] - p0["final_nav"]
    d_exit = p2["final_nav"] - p0["final_nav"]
    d_cap = p1["final_nav"] - p2["final_nav"]
    cap_share = d_cap / d_total if d_total > 0 else np.nan
    # 逐笔桥接：共同成交改进 + (新增 - 被挤出)，应与 Δtotal 恒等
    common0_pnl = float(a0["pnl"].sum())
    common1_pnl = float(a1["pnl"].sum())
    bridge_exit = common1_pnl - common0_pnl
    bridge_cap = new_pnl - dropped_pnl
    bridge_total = bridge_exit + bridge_cap
    bridge_gap = bridge_total - d_total
    exit_share = bridge_exit / bridge_total if bridge_total else np.nan
    cap_net_share = bridge_cap / bridge_total if bridge_total else np.nan
    pd.DataFrame([
        dict(object="P0_B0", final_nav=p0["final_nav"], mdd=p0["mdd"],
             delta_vs_p0=0.0, fills=len(t0s)),
        dict(object="P1_ATR2", final_nav=p1["final_nav"], mdd=p1["mdd"],
             delta_vs_p0=d_total, fills=len(t1s)),
        dict(object="P2_locked", final_nav=p2["final_nav"], mdd=p2["mdd"],
             delta_vs_p0=d_exit, fills=len(p2["trades"])),
        dict(object="exit_effect", final_nav=np.nan, mdd=np.nan,
             delta_vs_p0=d_exit, fills=np.nan),
        dict(object="capacity_effect", final_nav=np.nan, mdd=np.nan,
             delta_vs_p0=d_cap, fills=np.nan),
        dict(object="capacity_share", final_nav=np.nan, mdd=np.nan,
             delta_vs_p0=cap_share, fills=np.nan),
    ]).to_csv(os.path.join(OUT, "p2f_attr_decomp.csv"), index=False, encoding="utf-8")

    # 逐年
    years = sorted(set(new["year"]) | set(early["year"]))
    yrows = []
    for y in years:
        nn = new[new["year"] == y]
        ee = early[early["year"] == y]
        yrows.append(dict(year=int(y), n_new=len(nn), new_pnl=float(nn["pnl"].sum()),
                          n_early=len(ee), early_loss=float(ee["cont_amt"].sum())))
    pd.DataFrame(yrows).to_csv(os.path.join(OUT, "p2f_attr_yearly.csv"),
                               index=False, encoding="utf-8")

    g1 = bool(d_total > 0 and np.isfinite(cap_share) and cap_share >= 0.60)
    g2 = bool(new_pnl > 0 and np.isfinite(new_ratio) and new_ratio >= p0_ratio)
    g3 = bool(new_no06_pnl > 0 and new_ex_top2 > 0 and np.isfinite(cap_share)
              and cap_share >= 0.50)
    if g1 and g2 and g3:
        verdict = "CAPACITY_REAL_SIGNAL"
    elif d_total > 0 and g1:
        verdict = "MIXED"
    else:
        verdict = "PATH_LUCK_SUSPECT"
    summary = dict(
        protocol_sha256=PROTO_SHA, verdict=verdict,
        gates=dict(G1=g1, G2=g2, G3=g3, G4="reported_only"),
        reproduce=dict(p0_nav=p0["final_nav"], p1_nav=p1["final_nav"],
                       p2_nav=p2["final_nav"], p0_dev=chk0, p1_dev=chk1,
                       p0_fills=len(t0s), p1_fills=len(t1s), p2_fills=len(p2["trades"]),
                       p0_mdd=p0["mdd"], p1_mdd=p1["mdd"], p2_mdd=p2["mdd"]),
        q1=dict(new=len(new), dropped=len(dropped), common=len(common_keys),
                new_by_outcome=q1, dropped_pnl=dropped_pnl,
                new_by_year={int(k): int(v) for k, v in
                             new["year"].value_counts().sort_index().items()},
                new_when_p0_npos={int(k): int(v) for k, v in
                                  new["p0_npos"].value_counts().sort_index().items()}),
        q2=dict(early_n=len(early),
                by_reason={k: q2[k] for k in ("stop", "tp")},
                n_stop=q2["n_stop"], n_tp=q2["n_tp"],
                n_lost=q2["n_lost"], n_saved=q2["n_saved"],
                total_amt=float(early["cont_amt"].sum()),
                share_amt_of_p0=float(early["cont_amt"].sum() / (p0["final_nav"] - d.CAPITAL0))
                if p0["final_nav"] > d.CAPITAL0 else np.nan,
                by_year={int(k): float(v) for k, v in
                         early.groupby("year")["cont_amt"].sum().items()}),
        q3=dict(new_pnl=new_pnl, new_capital_days=new_cd, new_cap_day_ret=new_ratio,
                p0_cap_day_ret=p0_ratio, losers=int(len(losers)),
                loss_amt=float(losers["pnl"].sum()) if len(losers) else 0.0,
                top2_share=top2_share, new_no06_pnl=new_no06_pnl,
                new_ex_top2_pnl=new_ex_top2,
                top_year=top_year, top_year_share=top_year_share,
                new_ex_topyear_pnl=new_ex_topyear),
        q4=dict(d_total=d_total, d_exit=d_exit, d_cap=d_cap, cap_share=cap_share),
        q5=dict(common0_pnl=common0_pnl, common1_pnl=common1_pnl,
                bridge_exit=bridge_exit, dropped_pnl=dropped_pnl,
                bridge_cap=bridge_cap, bridge_total=bridge_total,
                bridge_gap=bridge_gap, exit_share=exit_share,
                cap_net_share=cap_net_share),
        skipped=dict(p0=dict(p0["skipped"]), p1=dict(p1["skipped"]), p2=dict(p2["skipped"])))
    with open(os.path.join(OUT, "p2f_attr_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log(f"{verdict} Δtotal={d_total:,.0f} Δexit={d_exit:,.0f} Δcap={d_cap:,.0f} "
        f"cap_share={cap_share:.1%} 新增净利={new_pnl:,.0f}")
    write_report(summary, early, new)
    log(f"完成 {time.time()-t0:.0f}s")


def write_report(s, early, new):
    q1, q2, q3, q4 = s["q1"], s["q2"], s["q3"], s["q4"]
    rep = s["reproduce"]
    yrows = pd.read_csv(os.path.join(OUT, "p2f_attr_yearly.csv"))
    lines = [
        "# EP-02f 归因报告：ATR×2 多赚的钱来自哪里",
        "",
        f"- 协议：`EP02F_ATTR_PROTOCOL.md`（SHA256 `{PROTO_SHA[:16]}…`，运行前校验通过）",
        f"- 结论分类：**{s['verdict']}**（诊断结论，不构成采用建议）",
        "",
        "## 0. 复现校验",
        "",
        "| 组合 | 期末 NAV | MDD | 成交 | 相对 B0 |",
        "|---|---|---|---|---|",
        f"| P0 B0 | {rep['p0_nav']:,.0f} | {rep['p0_mdd']:.2%} | {rep['p0_fills']} | — |",
        f"| P1 ATR×2 | {rep['p1_nav']:,.0f} | {rep['p1_mdd']:.2%} | {rep['p1_fills']} | +{q4['d_total']:,.0f} |",
        f"| P2 冻结回款 | {rep['p2_nav']:,.0f} | {rep['p2_mdd']:.2%} | {rep['p2_fills']} | {q4['d_exit']:+,.0f} |",
        "",
        f"P0 偏差 {rep['p0_dev']:.2e}，P1 偏差 {rep['p1_dev']:.2e}。",
        "",
        "## Q1 提前退出释放了什么",
        "",
        f"- 新增成交 {q1['new']} 笔，被挤出 {q1['dropped']} 笔，共同成交 {q1['common']} 笔。",
        f"- 新增成交在 P0 的竞争结果：{q1['new_by_outcome']}。",
        f"- 新增成交发生时 P0 持仓占用分布：{q1['new_when_p0_npos']}。",
        f"- 被挤出成交在 P0 的净利合计 {q1['dropped_pnl']:,.0f} 元。",
        "",
        "## Q2 提前退出损失了什么",
        "",
        f"- 共同成交中提前退出 {q2['early_n']} 笔；少赚 {q2['n_lost']} 笔，避损 {q2['n_saved']} 笔。",
        f"- 续持反事实合计：{q2['total_amt']:,.0f} 元"
        f"（占 B0 全程净利的 {q2['share_amt_of_p0']:.1%}）。",
        f"- 拆分：止损组 {q2['by_reason']['stop']:,.0f} 元（{q2['n_stop']} 笔），"
        f"止盈组 {q2['by_reason']['tp']:,.0f} 元（{q2['n_tp']} 笔）。",
        "- 正值表示提前卖出放弃的后续收益，负值表示提前卖出避免的后续亏损。",
        "",
        "## Q3 新成交是否值得",
        "",
        f"- 新增净利 {q3['new_pnl']:,.0f} 元；资金日收益率 {q3['new_cap_day_ret']:.2%}"
        f"，P0 全部成交 {q3['p0_cap_day_ret']:.2%}。",
        f"- 新增亏损 {q3['losers']} 笔，合计 {q3['loss_amt']:,.0f} 元；"
        f"盈利 top2 占新增净利 {q3['top2_share']:.1%}。",
        f"- 剔除 2006 年后新增净利 {q3['new_no06_pnl']:,.0f} 元；"
        f"剔除新增净利前 2 笔后 {q3['new_ex_top2_pnl']:,.0f} 元。",
        f"- 年度集中度（补充观察，不参与闸门）：{q3['top_year']} 年占新增净利 "
        f"{q3['top_year_share']:.1%}；剔除该年后 {q3['new_ex_topyear_pnl']:,.0f} 元。",
        "",
        "## Q4 容量收益是否真实",
        "",
        f"- Δtotal = {q4['d_total']:,.0f} 元；Δexit = {q4['d_exit']:,.0f} 元；"
        f"Δcap = {q4['d_cap']:,.0f} 元；容量占比 {q4['cap_share']:.1%}。",
        "",
        "| 闸门 | 结果 |",
        "|---|---|",
        f"| G1 容量占比 ≥60% | {'通过' if s['gates']['G1'] else '未通过'} |",
        f"| G2 新增资金效率 ≥ P0 | {'通过' if s['gates']['G2'] else '未通过'} |",
        f"| G3 剔除 2006 与前 2 笔后仍为正 | {'通过' if s['gates']['G3'] else '未通过'} |",
        "| G4 提前退出 stop/tp 拆分 | 仅报告 |",
        "",
        "## 逐笔桥接（与 Δtotal 恒等）",
        "",
        "| 项目 | 笔数 | 金额（元） |",
        "|---|---|---|",
        f"| B0 共同成交（被提前退出） | {q1['common']} | {s['q5']['common0_pnl']:,.0f} → "
        f"{s['q5']['common1_pnl']:,.0f} |",
        f"| 被挤出 B0 成交（不再持有） | {q1['dropped']} | −{q1['dropped_pnl']:,.0f} |",
        f"| 新增成交（新买入） | {q1['new']} | +{q3['new_pnl']:,.0f} |",
        f"| 合计 | — | +{s['q5']['bridge_total']:,.0f} |",
        "",
        f"桥接与 Δtotal 的差额 {s['q5']['bridge_gap']:,.2f} 元（应为 0）。",
        f"其中共同成交的出场改善占 {s['q5']['exit_share']:.1%}，"
        f"新增减被挤出的净替代占 {s['q5']['cap_net_share']:.1%}。",
        "",
        "## 逐年",
        "",
        "| 年份 | 新增笔数 | 新增净利 | 提前退出笔数 | 续持损益 |",
        "|---|---|---|---|---|",
    ]
    for r in yrows.itertuples():
        lines.append(f"| {r.year} | {r.n_new} | {r.new_pnl:,.0f} | {r.n_early} | "
                     f"{r.early_loss:,.0f} |")
    lines += [
        "",
        "## 证据边界",
        "",
        "- ATR×2 是同一样本内的最优变体，归因不能消除选择偏差。",
        "- P2 的受限现金金额取决于 P1 的实际卖价，Δexit / Δcap 是路径近似拆分。",
        "- 新增与被挤出不是同一批交易，笔数差额不等于净额。",
        "- 收盘确认、次日开盘成交：真实盘中效果可能不同。",
        "- 本报告不构成采用建议；不得据此反向调整止损/止盈参数。",
    ]
    with open(os.path.join(OUT, "P2F_ATTRIBUTION_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
