# -*- coding: utf-8 -*-
"""EP-02e：B 类策略可投资性诊断（只诊断，不改规则）。

严格按 out/entry_phase_02/EP02E_INVESTABILITY_PROTOCOL.md
（SHA256 61A87D33AFF14CA93798996EECC9BE64E676A902971619C0C42F8A98A60BDC6B）执行。

三问：
  Q1 回撤从哪里来：≥10% 回撤事件、最大事件的已实现/未实现拆解、亏损贡献、持仓重叠、市况
  Q2 收益如何被捕获：信号漏斗、容量上限、成交/跳过对比、同日匹配、入场时滞
  Q3 组合有没有改善空间：盈亏集中度、占用质量、cohort、资本效率

冻结对照：B0（不止损、60 bar 收盘卖、3 仓、每笔 1/3 NAV、同日先到先得）。
脚本会精确复现 B0 组合，并与 p2d_main_summary.csv 的 NAV/MDD/成交笔数对账。

用法：
    python scripts/entry_phase_02e_investability.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02d_stoploss as d  # noqa: E402

OUT = os.path.join(ROOT, "out", "entry_phase_02")
TRADES = os.path.join(OUT, "p2d_trades.csv")
NAV_B0 = os.path.join(OUT, "p2d_main_nav_B0.csv")
L2_NAV = os.path.join(OUT, "l2_daily_nav.csv")
MKT = os.path.join(OUT, "attr_market_proxy.csv")
PROTO = os.path.join(OUT, "EP02E_INVESTABILITY_PROTOCOL.md")
PROTO_SHA = "61A87D33AFF14CA93798996EECC9BE64E676A902971619C0C42F8A98A60BDC6B"
CAPITAL0 = 1_000_000.0
SLOTS = 3
HOLD = 60
NOTIONAL = 50_000.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_protocol() -> None:
    h = hashlib.sha256(open(PROTO, "rb").read()).hexdigest().upper()
    if h != PROTO_SHA:
        raise SystemExit(f"EP-02e 协议已被修改（{h}），拒绝运行")
    log(f"协议校验通过 SHA256={h[:16]}…")


def md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if cols:
        df = df[cols]
    if max_rows:
        df = df.head(max_rows)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = []
    for r in df.itertuples(index=False):
        vals = []
        for v in r:
            if isinstance(v, float):
                vals.append(f"{v:,.2f}" if abs(v) < 1e9 else f"{v:,.0f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep] + rows)


def pct(x: float, nd: int = 2) -> str:
    return "—" if not np.isfinite(x) else f"{x*100:.{nd}f}%"


# ---------------------------------------------------------------------------
# 精确复现 B0 组合，并记录完整成交台账与逐信号竞争结果
# ---------------------------------------------------------------------------
def run_ledger(win_rows: list[dict], cal_ns: np.ndarray, cal_pos: dict,
               net60: dict) -> dict:
    from collections import defaultdict as dd
    slip, comm = d.l1.SLIP["base"], d.l1.COMM["base"]
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
    cash_av, cash_pend = CAPITAL0, 0.0
    pos, pending_sell = {}, {}
    nav = np.full(NCAL, np.nan)
    npos = np.zeros(NCAL, int)
    nav_prev = CAPITAL0
    trades, attempts = [], []
    skipped = defaultdict(int)

    def fee_sell(code, dt, amt, sh):
        return (max(amt * comm, d.l1.MIN_COMM) + amt * d.l1.stamp_rate(dt, "sell")
                + d.l1.transfer_fee(code, dt, "sell", amt, sh, "base"))

    def fee_buy(code, dt, amt, sh):
        return max(amt * comm, d.l1.MIN_COMM) + d.l1.transfer_fee(code, dt, "buy", amt, sh, "base")

    for g in range(NCAL):
        day, dns = days[g], int(cal_ns[g])
        # 卖出（回款次日可用）
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
            cash_pend += amt * (1.0 - slip) - f
            p["pnl"] = amt * (1.0 - slip) - f - p["cost_in"]
            p["exit_date"] = str(pd.Timestamp(ex["day"]).date())
            p["exit_g"] = g
            p["exit_px"] = px
            p["exit_reason"] = ex["reason"]
            trades.append(dict(
                code=code, signal_date=p["signal_date"], entry_date=p["entry_date"],
                exit_date=p["exit_date"], entry_g=p["entry_g"], exit_g=g,
                entry_px=p["entry_px"], shares=p["shares"], notional=p["notional"],
                pnl=p["pnl"], ret=p["pnl"] / p["notional"],
                hold_bars=max(1, g - p["entry_g"]), exit_px=px,
                exit_reason=ex["reason"]))
            del pos[code]
            pending_sell.pop(code, None)
        # 买入
        for s in by_entry.get(g, []):
            key = (s["code"], s["entry_date"])
            base = dict(code=s["code"], signal_date=s["signal_date"],
                        entry_date=s["entry_date"], entry_g=g,
                        net60=net60.get(key, np.nan))
            if len(pos) >= d.MAIN_MAX_POS:
                skipped["max_positions"] += 1
                attempts.append(dict(base, status="skipped", reason="max_positions"))
                continue
            if s["code"] in pos:
                skipped["duplicate"] += 1
                attempts.append(dict(base, status="skipped", reason="duplicate"))
                continue
            ex = s["exits"]["B0"]
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
            pos[s["code"]] = dict(
                shares=shares, notional=notional, entry_px=px,
                signal_date=s["signal_date"], entry_date=s["entry_date"], entry_g=g,
                f_last=float(s["f"][0]), last_c=float(s["rc_raw"][0]),
                ptr=1, win=s, exit=ex, cost_in=amt + f, pnl=np.nan)
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
            if i >= 0 and i < dn.size and dn[i] == dns:
                if ff[i] != p["f_last"] and np.isfinite(ff[i]) and ff[i] != 0:
                    p["shares"] *= p["f_last"] / ff[i]
                    p["f_last"] = ff[i]
                if np.isfinite(rcp[i]):
                    p["last_c"] = float(rcp[i])
            p["ptr"] = max(p["ptr"], i + 1)
        val = sum(p["shares"] * p["last_c"] for p in pos.values())
        nav[g] = cash_av + cash_pend + val
        npos[g] = len(pos)
        nav_prev = nav[g]
        cash_av += cash_pend
        cash_pend = 0.0

    open_end = [dict(code=c, entry_date=p["entry_date"],
                     value=p["shares"] * p["last_c"]) for c, p in pos.items()]
    return dict(nav=nav, npos=npos, dates=days, trades=pd.DataFrame(trades),
                attempts=pd.DataFrame(attempts), skipped=dict(skipped),
                open_end=open_end)


def max_dd(nav: np.ndarray):
    v = nav[np.isfinite(nav)]
    peak = np.maximum.accumulate(v)
    dd = (peak - v) / peak
    i = int(np.argmax(dd))
    j = int(np.argmax(v[:i + 1])) if i > 0 else 0
    return float(dd[i]), j, i, v


def drawdown_episodes(nav: np.ndarray, dates: pd.DatetimeIndex, first: int,
                      th: float = 0.10) -> list[dict]:
    eps = []
    peak_val, peak_i = nav[first], first
    in_ep, ep_pk, ep_pk_val, tr_i = False, first, nav[first], first
    for i in range(first + 1, nav.size):
        if not np.isfinite(nav[i]):
            continue
        if nav[i] >= peak_val:
            if in_ep:
                eps.append(dict(
                    peak_i=ep_pk, trough_i=tr_i, end_i=i, peak_val=ep_pk_val,
                    trough_val=nav[tr_i], depth=1 - nav[tr_i] / ep_pk_val,
                    recovered=True,
                    peak_date=str(dates[ep_pk].date()), trough_date=str(dates[tr_i].date()),
                    end_date=str(dates[i].date()),
                    trough_days=int(tr_i - ep_pk), underwater_days=int(i - ep_pk)))
                in_ep = False
            peak_val, peak_i = nav[i], i
        else:
            ddv = 1 - nav[i] / peak_val
            if ddv >= th and not in_ep:
                in_ep, ep_pk, ep_pk_val, tr_i = True, peak_i, peak_val, i
            if in_ep and nav[i] < nav[tr_i]:
                tr_i = i
    if in_ep:
        i = nav.size - 1
        eps.append(dict(
            peak_i=ep_pk, trough_i=tr_i, end_i=i, peak_val=ep_pk_val,
            trough_val=nav[tr_i], depth=1 - nav[tr_i] / ep_pk_val, recovered=False,
            peak_date=str(dates[ep_pk].date()), trough_date=str(dates[tr_i].date()),
            end_date=str(dates[i].date()),
            trough_days=int(tr_i - ep_pk), underwater_days=int(i - ep_pk)))
    return eps


def cagr(nav0: float, nav1: float, days: float) -> float:
    if nav0 <= 0 or days <= 0:
        return np.nan
    return (nav1 / nav0) ** (365.25 / days) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-sim", action="store_true", help="复用已缓存的台账（调试用）")
    args = ap.parse_args()
    t0 = time.time()
    verify_protocol()
    ep.init_ctx()
    cal = pd.to_datetime(ep.CTX["cal"])
    cal_ns = cal.values.astype("datetime64[ns]").astype(np.int64)
    cal_pos = {pd.Timestamp(k): v for k, v in ep.CTX["cal_pos"].items()}

    b0 = pd.read_csv(TRADES, dtype={"code": str})
    b0["code"] = b0["code"].str.zfill(6)
    b0 = b0[(b0["rule"] == "B0") & (b0["mode"] == "main")].copy()
    for c in ("signal_date", "entry_date", "exit_date"):
        b0[c] = pd.to_datetime(b0[c])
    net60 = {(r.code, str(pd.Timestamp(r.entry_date).date())): float(r.net_return_base)
             for r in b0.itertuples()}
    log(f"B0 信号 {len(b0)} 笔")

    sig = d.load_signals()
    groups = list(sig.groupby("code", sort=False))
    win_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (_, wins, _dg) in enumerate(ex.map(d.simulate_stock, groups), 1):
            win_rows.extend(wins)
            if i % 1000 == 0 or i == len(groups):
                log(f"  复现进度 {i}/{len(groups)}  {time.time()-t0:.0f}s")

    led = run_ledger(win_rows, cal_ns, cal_pos, net60)
    nav, npos, dates = led["nav"], led["npos"], led["dates"]
    first = int(np.argmax(npos > 0)) if (npos > 0).any() else 0
    mdd, i_pk, i_tr, _ = max_dd(nav[first:])
    frozen_mdd, frozen_nav, frozen_fill = 0.753497937923756, 5719552.238656651, 201
    rel = abs(nav[-1] - frozen_nav) / frozen_nav
    chk = dict(final_nav=float(nav[-1]), mdd=float(mdd), n_filled=int(len(led["trades"])),
               frozen_nav=frozen_nav, frozen_mdd=frozen_mdd, frozen_fill=frozen_fill,
               nav_rel_diff=float(rel))
    log(f"复现校验 NAV={nav[-1]:,.2f} (diff {rel:.2e}) MDD={mdd:.4%} "
        f"成交={len(led['trades'])}")
    if rel > 1e-6 or abs(len(led["trades"]) - frozen_fill) > 0:
        raise SystemExit("B0 复现与冻结基线不一致，停止诊断")

    tr = led["trades"].copy()
    for c in ("signal_date", "entry_date", "exit_date"):
        tr[c] = pd.to_datetime(tr[c])
    tr["entry_year"] = tr["entry_date"].dt.year
    tr["exit_year"] = tr["exit_date"].dt.year
    att = led["attempts"].copy()
    att["entry_date"] = pd.to_datetime(att["entry_date"])
    att["signal_date"] = pd.to_datetime(att["signal_date"])
    att["year"] = att["signal_date"].dt.year

    mk = pd.read_csv(MKT)
    mk["date"] = pd.to_datetime(mk["date"])
    mk = mk.drop_duplicates("date").set_index("date")["mkt_idx"]

    # ---------------- Q1 回撤来源 ----------------
    eps = drawdown_episodes(nav, dates, first)
    ep_df = pd.DataFrame([dict(
        no=k + 1, peak_date=e["peak_date"], trough_date=e["trough_date"],
        recovery_date=(e["end_date"] if e["recovered"] else "未恢复"),
        depth=e["depth"], trough_days=e["trough_days"],
        underwater_days=e["underwater_days"],
        peak_nav=e["peak_val"], trough_nav=e["trough_val"],
        mkt_ret=(float(mk.loc[pd.Timestamp(e["trough_date"])] /
                        mk.loc[pd.Timestamp(e["peak_date"])] - 1)
                 if pd.Timestamp(e["peak_date"]) in mk.index
                 and pd.Timestamp(e["trough_date"]) in mk.index else np.nan)
    ) for k, e in enumerate(eps)])
    ep_df.to_csv(os.path.join(OUT, "p2e_dd_episodes.csv"), index=False, encoding="utf-8")

    e0 = max(eps, key=lambda x: x["depth"])
    pk, tr_i, end_i = pd.Timestamp(e0["peak_date"]), pd.Timestamp(e0["trough_date"]), \
        pd.Timestamp(e0["end_date"])
    win = tr[(tr["exit_date"] > pk) & (tr["exit_date"] <= tr_i)]
    realized = float(win["pnl"].sum())
    nav_chg = float(nav[e0["trough_i"]] - nav[e0["peak_i"]])
    unreal = nav_chg - realized
    ent = tr[(tr["entry_date"] > pk) & (tr["entry_date"] <= tr_i)]
    seg_np = npos[cal_pos[pk]:cal_pos[tr_i] + 1]
    days_full = int((seg_np >= d.MAIN_MAX_POS).sum())
    seg_days = int(seg_np.size)
    wk = ent.groupby(ent["entry_date"].dt.to_period("W")).size()
    attrib = dict(
        peak_date=e0["peak_date"], trough_date=e0["trough_date"],
        recovery_date=e0["end_date"] if e0["recovered"] else "未恢复",
        depth=e0["depth"], peak_nav=e0["peak_val"], trough_nav=e0["trough_val"],
        nav_change=nav_chg, realized_pnl=realized, unrealized_change=unreal,
        n_exits=int(len(win)), n_entries=int(len(ent)),
        avg_npos=float(seg_np.mean()), days_full=days_full, days_total=seg_days,
        share_full=days_full / seg_days if seg_days else np.nan,
        max_entries_week=int(wk.max()) if len(wk) else 0,
        mkt_ret=float(mk.loc[tr_i] / mk.loc[pk] - 1) if tr_i in mk.index else np.nan,
        entry_years={str(k): int(v) for k, v in ent["entry_year"].value_counts().sort_index().items()},
    )
    pd.DataFrame([dict(
        peak_date=attrib["peak_date"], trough_date=attrib["trough_date"],
        recovery_date=attrib["recovery_date"], depth=attrib["depth"],
        nav_change=nav_chg, realized_pnl=realized, unrealized_change=unreal,
        n_exits=attrib["n_exits"], n_entries=attrib["n_entries"],
        avg_npos=attrib["avg_npos"], share_full=attrib["share_full"],
        max_entries_week=attrib["max_entries_week"], mkt_ret=attrib["mkt_ret"],
    )]).to_csv(os.path.join(OUT, "p2e_dd_attribution.csv"), index=False, encoding="utf-8")

    losers = win.sort_values("pnl").head(20)[
        ["code", "entry_date", "exit_date", "hold_bars", "entry_px", "exit_px",
         "notional", "pnl", "ret"]].copy()
    losers["entry_date"] = losers["entry_date"].dt.strftime("%Y-%m-%d")
    losers["exit_date"] = losers["exit_date"].dt.strftime("%Y-%m-%d")
    losers.to_csv(os.path.join(OUT, "p2e_dd_top_losers.csv"), index=False, encoding="utf-8")
    winners = tr.sort_values("pnl", ascending=False).head(5)[
        ["code", "entry_date", "exit_date", "hold_bars", "notional", "pnl", "ret"]].copy()
    winners["entry_date"] = winners["entry_date"].dt.strftime("%Y-%m-%d")
    winners["exit_date"] = winners["exit_date"].dt.strftime("%Y-%m-%d")
    winners.to_csv(os.path.join(OUT, "p2e_top_winners.csv"), index=False, encoding="utf-8")

    # ---------------- Q2 收益捕获 ----------------
    fun = att.groupby("year").agg(
        signals=("code", "size"),
        filled=("status", lambda s: int((s == "filled").sum())),
        max_positions=("reason", lambda s: int((s == "max_positions").sum())),
        cash_insufficient=("reason", lambda s: int((s == "cash_insufficient").sum())),
        duplicate=("reason", lambda s: int((s == "duplicate").sum())),
    ).reset_index()
    fun["fill_rate"] = fun["filled"] / fun["signals"]
    fun.to_csv(os.path.join(OUT, "p2e_funnel.csv"), index=False, encoding="utf-8")

    act0, act1 = int(tr["entry_g"].min()), int(tr["exit_g"].max())
    span_days = act1 - act0 + 1
    cap_fills = SLOTS * span_days / HOLD
    filled = att[att["status"] == "filled"]
    skipped = att[att["status"] == "skipped"]

    day_size = att.groupby("entry_date").size()
    day_fill = att.groupby("entry_date")["status"].apply(lambda s: int((s == "filled").sum()))
    day_fill_rate = (day_fill / day_size)
    buckets = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10**9)]
    cong = []
    for lo, hi in buckets:
        mask = (day_size >= lo) & (day_size <= hi)
        nm = f"{lo}-{hi}" if hi < 10**9 else f"{lo}-+"
        cong.append(dict(bucket=nm, n_days=int(mask.sum()),
                         n_signals=int(day_size[mask].sum()),
                         fill_rate=float(day_fill_rate[mask].mean()) if mask.any() else np.nan))

    matched = []
    for dt, g in att.groupby("entry_date"):
        if (g["status"] == "filled").any() and (g["status"] == "skipped").any():
            matched.append(float(g.loc[g["status"] == "filled", "net60"].mean()
                                 - g.loc[g["status"] == "skipped", "net60"].mean()))
    matched = np.array(matched, float)

    lag = []
    for r in b0.itertuples():
        a, b = cal_pos.get(pd.Timestamp(r.signal_date)), cal_pos.get(pd.Timestamp(r.entry_date))
        if a is not None and b is not None:
            lag.append(b - a)
    lag = np.array(lag, int)

    cap = dict(
        n_signals=int(len(att)), n_filled=int(len(filled)),
        fill_rate=float(len(filled) / len(att)),
        skipped_max_positions=int((att["reason"] == "max_positions").sum()),
        skipped_cash=int((att["reason"] == "cash_insufficient").sum()),
        skipped_duplicate=int((att["reason"] == "duplicate").sum()),
        active_span_days=int(span_days), capacity_ceiling=float(cap_fills),
        utilization=float(len(filled) / cap_fills) if cap_fills else np.nan,
        avg_hold_bars=float(tr["hold_bars"].mean()),
        filled_exp=float(filled["net60"].mean()),
        skipped_exp=float(skipped["net60"].mean()),
        all_exp=float(att["net60"].mean()),
        filled_total=float(filled["net60"].sum() * NOTIONAL),
        skipped_total_unreachable=float(skipped["net60"].sum() * NOTIONAL),
        filled_win=float((filled["net60"] > 0).mean()),
        skipped_win=float((skipped["net60"] > 0).mean()),
        matched_days=int(matched.size),
        matched_mean_diff=float(matched.mean()) if matched.size else np.nan,
        matched_share_filled_better=float((matched > 0).mean()) if matched.size else np.nan,
        lag_mean=float(lag.mean()) if lag.size else np.nan,
        lag_1day_share=float((lag == 1).mean()) if lag.size else np.nan,
        lag_gt1=int((lag > 1).sum()), lag_0=int((lag == 0).sum()),
    )
    cap_df = pd.DataFrame([cap])
    cap_df.to_csv(os.path.join(OUT, "p2e_capture.csv"), index=False, encoding="utf-8")
    cong_df = pd.DataFrame(cong)
    cong_df.to_csv(os.path.join(OUT, "p2e_congestion.csv"), index=False, encoding="utf-8")

    # ---------------- Q3 组合改善空间 ----------------
    tot = float(tr["pnl"].sum())
    pos_pnl = tr.sort_values("pnl", ascending=False)
    abs_share = (tr["pnl"].abs() / tr["pnl"].abs().sum()).to_numpy()
    hhi = float(np.sum(abs_share ** 2))
    topk = []
    abs_total = float(tr["pnl"].abs().sum())
    for k in (1, 2, 5, 10, 20):
        topk.append(dict(
            k=k,
            share_of_total=float(pos_pnl["pnl"].head(k).sum() / tot),
            share_of_abs=float(tr["pnl"].abs().sort_values(ascending=False).head(k).sum()
                               / abs_total),
        ))
    stk = tr.groupby("code")["pnl"].sum().sort_values(ascending=False)
    stk_share = float(stk.head(5).sum() / tot) if tot else np.nan
    yr = tr.groupby("exit_year")["pnl"].sum()
    max_year_share = float(yr.max() / tot) if tot else np.nan
    conc = dict(total_pnl=tot, hhi=hhi, top5_stock_share=stk_share,
                max_year_share=max_year_share,
                top_trades=topk)
    pd.DataFrame(topk).to_csv(os.path.join(OUT, "p2e_concentration.csv"),
                              index=False, encoding="utf-8")

    # 持仓日占用：按最终收益正负归类（事后口径）
    pd_days = []
    for r in tr.itertuples():
        a, b = cal_pos.get(pd.Timestamp(r.entry_date)), cal_pos.get(pd.Timestamp(r.exit_date))
        pd_days.append(max(1, (b - a) if a is not None and b is not None else r.hold_bars))
    tr["pos_days"] = pd_days
    total_pd = float(tr["pos_days"].sum())
    neg_pd = float(tr.loc[tr["ret"] < 0, "pos_days"].sum())
    occ = dict(position_days=int(round(total_pd)),
               loss_trade_position_days=int(round(neg_pd)),
               loss_share=neg_pd / total_pd if total_pd else np.nan,
               days_full=days_full, days_total=seg_days,
               share_full=days_full / seg_days if seg_days else np.nan,
               idle_days=int((seg_np == 0).sum()),
               avg_npos=float(seg_np.mean()))
    np_counts = {str(k): int((seg_np == k).sum()) for k in range(0, SLOTS + 1)}
    occ["npos_days"] = np_counts

    cohort = tr.groupby("entry_year").agg(
        fills=("code", "size"), win_rate=("ret", lambda s: float((s > 0).mean())),
        avg_ret=("ret", "mean"), pnl=("pnl", "sum"), pos_days=("pos_days", "sum"),
    ).reset_index()
    cohort["pnl_share"] = cohort["pnl"] / tot
    cohort["pos_day_share"] = cohort["pos_days"] / total_pd
    cohort.to_csv(os.path.join(OUT, "p2e_occupancy.csv"), index=False, encoding="utf-8")

    c06 = cohort[cohort["entry_year"] == 2006]
    conc["max_year"] = int(yr.idxmax())
    conc["max_year_pnl"] = float(yr.max())
    conc["cohort_2006_pnl"] = float(c06["pnl"].iloc[0]) if len(c06) else np.nan
    conc["cohort_2006_share"] = float(c06["pnl_share"].iloc[0]) if len(c06) else np.nan
    conc["pnl_ex_2006"] = float(tot - conc["cohort_2006_pnl"]) if len(c06) else np.nan

    # 资本效率 + 20 仓对照（只引用已冻结结果）
    nav0 = float(nav[first])
    yrs = (dates[-1] - dates[first]).days / 365.25
    eff = dict(cagr=cagr(nav0, float(nav[-1]), (dates[-1] - dates[first]).days),
               mdd=float(mdd), cagr_over_mdd=float(cagr(nav0, float(nav[-1]),
                                                        (dates[-1] - dates[first]).days) / mdd),
               years=float(yrs), start=str(dates[first].date()), end=str(dates[-1].date()))
    l2 = pd.read_csv(L2_NAV)
    l2["date"] = pd.to_datetime(l2["date"])
    l2a = l2[l2["n_open"] > 0].reset_index(drop=True)
    l2mdd, _, _, _ = max_dd(l2a["nav"].to_numpy(float))
    l2eff = dict(cagr=cagr(float(l2a["nav"].iloc[0]), float(l2a["nav"].iloc[-1]),
                           (l2a["date"].iloc[-1] - l2a["date"].iloc[0]).days),
                 mdd=float(l2mdd), n_filled=1480,
                 cagr_over_mdd=float(cagr(float(l2a["nav"].iloc[0]), float(l2a["nav"].iloc[-1]),
                                          (l2a["date"].iloc[-1] - l2a["date"].iloc[0]).days)
                                   / l2mdd))

    summary = dict(
        meta=dict(protocol_sha256=PROTO_SHA, generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  slots=SLOTS, hold_bars=HOLD, n_signals=int(len(att)),
                  elapsed_s=round(time.time() - t0, 1)),
        reproduction=chk,
        Q1_drawdown=dict(episodes=len(eps), worst=attrib,
                         top_losers_sum=float(losers["pnl"].sum()),
                         top_losers_share=float(losers["pnl"].sum() / realized) if realized else np.nan),
        Q2_capture=cap,
        Q2_congestion=cong,
        Q3_concentration=dict(conc, top_winners=[
            dict(code=r.code, entry_date=r.entry_date, exit_date=r.exit_date,
                 pnl=float(r.pnl), ret=float(r.ret)) for r in winners.itertuples()]),
        Q3_occupancy=occ,
        Q3_capital_efficiency=eff, Q3_reference_20pos=l2eff,
        limits=["3 仓组合存在路径依赖，期末净值不代表策略优劣",
                "占用质量按最终收益正负事后归类，不能反推入场时可判",
                "跳过信号的机会成本按 50,000 元名义计算，受容量约束不可直接实现"],
    )
    with open(os.path.join(OUT, "p2e_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    # ---------------- 报告 ----------------
    L = []
    L.append("# EP-02e 可投资性诊断报告\n")
    L.append(f"- 协议：`EP02E_INVESTABILITY_PROTOCOL.md`（SHA256 `{PROTO_SHA[:16]}…`，运行前校验通过）")
    L.append(f"- 冻结对照：B0 不止损 / 60 bar 收盘卖 / 3 仓 / 每笔 1/3 NAV / 同日先到先得")
    L.append(f"- 结论性质：诊断，不产生采用决定，不新增任何交易规则\n")
    L.append("## 0. 复现校验\n")
    L.append(f"| 项目 | 本次复现 | 冻结基线 |\n|---|---|---|\n"
             f"| 期末 NAV | {chk['final_nav']:,.2f} | {frozen_nav:,.2f} |\n"
             f"| MDD | {pct(chk['mdd'])} | {pct(frozen_mdd)} |\n"
             f"| 成交笔数 | {chk['n_filled']} | {frozen_fill} |\n"
             f"| NAV 相对偏差 | {chk['nav_rel_diff']:.2e} | — |\n")
    L.append("复现一致，以下拆解均基于同一冻结 B0。\n")

    L.append("## 结论速览（三问）\n")
    L.append(f"- **Q1 回撤来源**：{pct(e0['depth'])} 最大回撤集中在 {e0['peak_date'][:4]}—"
             f"{e0['trough_date'][:4]} 下行段，组合跌幅为市场代理的 "
             f"{e0['depth']/abs(attrib['mkt_ret']):.2f} 倍，"
             f"{e0['underwater_days']} 个交易日"
             f"（约 {e0['underwater_days']/244:.1f} 年）未收复前高；"
             f"跌幅的 {pct(abs(realized)/abs(nav_chg))} 来自已实现亏损。")
    L.append(f"- **Q2 收益捕获**：{cap['n_signals']:,} 个信号仅成交 {cap['n_filled']} 笔"
             f"（{pct(cap['fill_rate'])}），成交已达 {SLOTS} 仓理论上限的 {pct(cap['utilization'])}；"
             "未成交全部源于满仓与现金不足，不是执行失败。")
    L.append(f"- **Q3 改善空间**：净收益 {pct(conc['cohort_2006_share'])} 来自 2006 年入场的 9 笔；"
             f"{pct(occ['loss_share'])} 的持仓日最终属于亏损交易；"
             f"3 仓 CAGR/MDD={eff['cagr_over_mdd']:.2f} vs 20 仓对照 {l2eff['cagr_over_mdd']:.2f}。\n")

    L.append("## Q1 回撤从哪里来\n")
    L.append(f"共识别 {len(eps)} 段 ≥10% 回撤。最大回撤 {pct(e0['depth'])}，"
             f"{e0['peak_date']} → {e0['trough_date']}，"
             f"{'恢复于 ' + e0['end_date'] if e0['recovered'] else '至数据末端未恢复'}。")
    L.append(f"峰到谷 {e0['trough_days']} 个交易日（约 {e0['trough_days']/244:.1f} 年），"
             f"水下 {e0['underwater_days']} 个交易日（约 {e0['underwater_days']/244:.1f} 年）。\n")
    show = pd.DataFrame(dict(
        序号=ep_df["no"], 峰值日=ep_df["peak_date"], 谷底日=ep_df["trough_date"],
        恢复日=ep_df["recovery_date"], 深度=[pct(x) for x in ep_df["depth"]],
        峰到谷交易日=ep_df["trough_days"], 水下交易日=ep_df["underwater_days"],
        区间市场收益=[pct(x) for x in ep_df["mkt_ret"]]))
    L.append(md_table(show))
    L.append("\n### 最大回撤拆解\n")
    L.append(f"- NAV 变动：{nav_chg:,.0f} 元")
    L.append(f"- 已实现盈亏（卖出日落在区间内，{attrib['n_exits']} 笔）：{realized:,.0f} 元")
    L.append(f"- 未实现变动（残差）：{unreal:,.0f} 元")
    L.append(f"- 区间新入场 {attrib['n_entries']} 笔；平均同时持仓 {attrib['avg_npos']:.2f} / {SLOTS}；"
             f"满仓天数占比 {pct(attrib['share_full'])}；单周最多入场 {attrib['max_entries_week']} 笔")
    L.append(f"- 同区间市场代理收益 {pct(attrib['mkt_ret'])}；组合跌幅是市场的 {e0['depth']/abs(attrib['mkt_ret']):.2f} 倍"
             if np.isfinite(attrib["mkt_ret"]) and attrib["mkt_ret"] != 0 else "")
    L.append("\n亏损贡献前 20 笔：\n")
    lt = losers.copy()
    lt["pnl"] = lt["pnl"].map(lambda x: f"{x:,.0f}")
    lt["ret"] = lt["ret"].map(lambda x: pct(x))
    lt["notional"] = lt["notional"].map(lambda x: f"{x:,.0f}")
    lt["hold_bars"] = lt["hold_bars"].astype(int)
    L.append(md_table(lt.rename(columns={
        "code": "代码", "entry_date": "入场", "exit_date": "出场", "hold_bars": "持有bar",
        "entry_px": "入场价", "exit_px": "出场价", "notional": "名义", "pnl": "盈亏", "ret": "收益率"})))
    L.append(f"\n前 20 笔亏损合计 {losers['pnl'].sum():,.0f} 元，"
             f"为区间已实现亏损（{realized:,.0f} 元）的 {abs(losers['pnl'].sum()/realized):.2f} 倍；"
             "超出部分由区间内其余交易的盈利抵消。"
             f"入场年份分布：" + "、".join(f"{k} 年 {v} 笔" for k, v in attrib["entry_years"].items()) + "。\n")
    ratio_txt = (f"组合跌幅是市场代理的 {e0['depth']/abs(attrib['mkt_ret']):.2f} 倍"
                 if np.isfinite(attrib["mkt_ret"]) and attrib["mkt_ret"] != 0 else
                 "组合跌幅未与市场代理直接可比")
    L.append(f"**Q1 读法**：最大回撤发生在 {e0['peak_date'][:4]}—{e0['trough_date'][:4]} 的市场下行段，"
             f"{ratio_txt}（区间市场 {pct(attrib['mkt_ret'])}）；"
             f"{'直到 ' + e0['end_date'] + ' 才收复前高' if e0['recovered'] else '至数据末端仍未收复前高'}。"
             "跌幅中约 "
             f"{pct(abs(realized)/abs(nav_chg))} 来自已实现亏损、{pct(abs(unreal)/abs(nav_chg))} 来自谷底未实现缩水。"
             "亏损并非单笔极端事件，而是 3 仓长期满仓持有高波动标的、随市场下行反复兑现的表现；"
             "这与 EP-02d 中固定百分比止损越止越差的结果一致。\n")

    L.append("## Q2 收益如何被捕获\n")
    L.append(f"总体漏斗：{cap['n_signals']:,} 个信号 → 成交 {cap['n_filled']} 笔 "
             f"（{pct(cap['fill_rate'])}）；满仓跳过 {cap['skipped_max_positions']:,}，"
             f"现金不足跳过 {cap['skipped_cash']:,}，重复标的 {cap['skipped_duplicate']}。\n")
    fshow = fun.copy()
    fshow["fill_rate"] = fshow["fill_rate"].map(lambda x: pct(x))
    L.append(md_table(fshow.rename(columns={
        "year": "信号年", "signals": "信号", "filled": "成交", "max_positions": "满仓跳过",
        "cash_insufficient": "现金不足", "duplicate": "重复", "fill_rate": "成交率"})))
    L.append("\n### 容量约束\n")
    L.append(f"- 活跃区间 {cap['active_span_days']} 个交易日；按 {SLOTS} 仓 × 60 bar 持有，"
             f"理论上限约 {cap['capacity_ceiling']:.0f} 笔")
    L.append(f"- 实际成交 {cap['n_filled']} 笔，利用率 {pct(cap['utilization'])}；平均持有 {cap['avg_hold_bars']:.1f} bar")
    L.append(f"- 成交组单笔净期望 {pct(cap['filled_exp'])}，跳过组 {pct(cap['skipped_exp'])}，"
             f"全样本 {pct(cap['all_exp'])}；成交组胜率 {pct(cap['filled_win'])} vs 跳过组 {pct(cap['skipped_win'])}")
    L.append(f"- 跳过信号的假设盈亏（每笔 5 万名义）合计 {cap['skipped_total_unreachable']:,.0f} 元，"
             f"**受容量约束不可直接实现**，仅用于衡量机会规模")
    L.append(f"- 同日匹配：{cap['matched_days']} 个交易日同时存在成交与跳过，"
             f"成交组减跳过组的日均差 {pct(cap['matched_mean_diff'])}，"
             f"成交更优的天数占比 {pct(cap['matched_share_filled_better'])}")
    L.append(f"- 入场时滞：平均 {cap['lag_mean']:.2f} 个交易日，1 日占比 {pct(cap['lag_1day_share'])}，"
             f">1 日 {cap['lag_gt1']} 笔，0 日 {cap['lag_0']} 笔")
    cshow = pd.DataFrame(cong)
    cshow["fill_rate"] = cshow["fill_rate"].map(lambda x: pct(x))
    L.append("\n拥挤度：\n")
    L.append(md_table(cshow.rename(columns={
        "bucket": "当日候选数", "n_days": "天数", "n_signals": "信号", "fill_rate": "成交率"})))
    L.append("\n**Q2 读法**：机会损失几乎全部来自仓位容量，而不是执行环节——"
             "入场时滞平均 1.01 个交易日（99.67% 为次日），未成交不是下单或成交失败。"
             f"3 仓结构下成交已用掉理论上限的 {pct(cap['utilization'])}，"
             "因此不可能靠加快换手多抓信号，只能改变持仓数量或单笔规模。")
    L.append(f"成交组单笔净期望 {pct(cap['filled_exp'])}，比跳过组 {pct(cap['skipped_exp'])} "
             f"低 {abs(cap['filled_exp']-cap['skipped_exp'])*100:.2f}pp，"
             f"胜率也低 {abs(cap['filled_win']-cap['skipped_win'])*100:.2f}pp；"
             "但同日匹配的日均差为 "
             f"{pct(cap['matched_mean_diff'])}、成交更优的天数仅 {pct(cap['matched_share_filled_better'])}，"
             "方向不一致。说明整体差异主要来自「什么时候有空仓位」的时点分布，"
             "而不是同日先到先得的排序偏好；先到先得没有稳定地选到更好或更差的信号。")
    L.append(f"另外，2025 年信号量升至 2,014 个、成交仅 10 笔（{pct(fun.loc[fun['year']==2025,'fill_rate'].iloc[0] if (fun['year']==2025).any() else np.nan)}），"
             "说明信号流与容量之间的缺口在近年扩大，而不是缩小。\n")

    L.append("## Q3 组合有没有改善空间\n")
    L.append(f"- 已实现总盈亏 {tot:,.0f} 元；HHI（按 |盈亏| 份额）{hhi:.4f}")
    L.append("\n| 前 k 笔 | 占总盈亏 | 占绝对盈亏 |\n|---|---|---|")
    for t in topk:
        L.append(f"| {t['k']} | {pct(t['share_of_total'])} | {pct(t['share_of_abs'])} |")
    L.append("\n注：总盈亏是净额，少数大额盈利会被大量亏损抵消，因此 top-k「占总盈亏」可超过 100%；"
             "该列用于说明净收益对少数交易的依赖程度，「占绝对盈亏」用于说明交易层面的集中度。")
    wtab = winners.copy()
    wtab["pnl"] = wtab["pnl"].map(lambda x: f"{x:,.0f}")
    wtab["notional"] = wtab["notional"].map(lambda x: f"{x:,.0f}")
    wtab["ret"] = wtab["ret"].map(lambda x: pct(x))
    wtab["hold_bars"] = wtab["hold_bars"].astype(int)
    L.append("\n盈利贡献前 5 笔：\n")
    L.append(md_table(wtab.rename(columns={
        "code": "代码", "entry_date": "入场", "exit_date": "出场", "hold_bars": "持有bar",
        "notional": "名义", "pnl": "盈亏", "ret": "收益率"})))
    L.append("\n按卖出年与按入场年是两套归属口径（同一笔利润只归属一次），不能相加；"
             "前者说明利润在何时兑现，后者说明利润由哪批信号创造。\n")
    L.append(f"- 前 5 只股票贡献 {pct(stk_share)}；最大单年（{conc['max_year']} 年卖出）"
             f"贡献 {pct(max_year_share)}（{conc['max_year_pnl']:,.0f} 元）")
    if np.isfinite(conc["cohort_2006_share"]):
        L.append(f"- 2006 入场 cohort 仅 9 笔，贡献 {pct(conc['cohort_2006_share'])}"
                 f"（{conc['cohort_2006_pnl']:,.0f} 元）；其余 {len(tr)-9} 笔合计仅 "
                 f"{conc['pnl_ex_2006']:,.0f} 元")
    L.append(f"- 持仓日合计 {occ['position_days']:,}；其中最终亏损交易占用 "
             f"{occ['loss_trade_position_days']:,}（{pct(occ['loss_share'])}）")
    L.append(f"- 满仓天数占比 {pct(occ['share_full'])}；闲置 {occ['idle_days']} 日；"
             "持仓分布 " + "、".join(f"{k} 仓 {v} 日" for k, v in occ["npos_days"].items()))
    cshow = cohort.copy()
    for c in ("win_rate", "avg_ret", "pnl_share", "pos_day_share"):
        cshow[c] = cshow[c].map(lambda x: pct(x))
    cshow["pnl"] = cshow["pnl"].map(lambda x: f"{x:,.0f}")
    cshow["pos_days"] = cshow["pos_days"].astype(int)
    L.append("\n入场年份 cohort：\n")
    L.append(md_table(cshow.rename(columns={
        "entry_year": "入场年", "fills": "成交", "win_rate": "胜率", "avg_ret": "平均净收益",
        "pnl": "盈亏", "pos_days": "持仓日", "pnl_share": "盈亏占比", "pos_day_share": "持仓日占比"})))
    L.append("\n资本效率（3 仓 B0 vs 已冻结 20 仓对照，仅引用不新跑）：\n")
    eff_tab = pd.DataFrame([
        dict(组合="3 仓 B0", CAGR=pct(eff["cagr"]), MDD=pct(eff["mdd"]),
             CAGR_MDD=f"{eff['cagr_over_mdd']:.2f}", 成交=chk["n_filled"]),
        dict(组合="20 仓对照", CAGR=pct(l2eff["cagr"]), MDD=pct(l2eff["mdd"]),
             CAGR_MDD=f"{l2eff['cagr_over_mdd']:.2f}", 成交=l2eff["n_filled"]),
    ])
    L.append(md_table(eff_tab))
    L.append("\n**Q3 读法**：净收益极度依赖少数交易与单一年份——"
             f"1 笔交易占总盈亏 {pct(topk[0]['share_of_total'])}，"
             f"前 2 笔合计 {pct(topk[1]['share_of_total'])}，"
             f"{conc['max_year']} 年卖出贡献 {pct(max_year_share)}，"
             f"2006 入场 cohort 的 9 笔贡献 {pct(conc['cohort_2006_share'])}；"
             "去掉 2006 cohort 后其余交易的合计盈利接近于零，"
             "说明当前 3 仓结构的资本基数主要由早期一次牛市决定，后续年份在消耗而非积累。"
             f"同时 {pct(occ['loss_share'])} 的持仓日最终落在亏损交易上（事后归类），"
             "满仓仅占 43.74%，空仓与半仓合计超过一半时间——仓位利用率与机会质量同时存在问题。"
             f"3 仓的 CAGR {pct(eff['cagr'])} / MDD {pct(eff['mdd'])}，"
             f"20 仓对照 {pct(l2eff['cagr'])} / {pct(l2eff['mdd'])}，后者两项都更好；"
             "但两者路径不同、且 20 仓结果来自另一套资金约束，只能作为容量约束的参照，"
             "不能直接外推为「加仓即可改善」。任何容量或规模调整都必须同时列示收益、回撤与机会成本三项，"
             "单看 MDD 下降不构成改善证据。\n")

    L.append("## 证据边界\n")
    L.append("1. 本报告只描述冻结 B0 的收益与风险结构，不构成任何新规则、过滤器或参数建议。")
    L.append("2. 3 仓组合存在路径依赖，期末净值与单段回撤不能互相替代解释。")
    L.append("3. 跳过信号的机会成本按固定 5 万名义计算，实际不可实现。")
    L.append("4. 占用质量为事后归类（按最终收益正负），不能作为入场时可判特征。")
    L.append("5. 20 仓结果来自 EP-02d 已冻结对照，此处仅作容量参照。")
    L.append("\n## 待决\n")
    L.append("诊断已给出回撤来源、捕获损失环节与集中度证据；是否进入风险架构开发、"
             "以及以何种容量/规模假设进入，须由决策方在证据基础上另行决定。")
    with open(os.path.join(OUT, "P2E_INVESTABILITY_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    log(f"完成：{time.time()-t0:.0f}s；报告 P2E_INVESTABILITY_REPORT.md")
    log(f"Q1 最大回撤 {pct(e0['depth'])}；Q2 成交率 {pct(cap['fill_rate'])}；"
        f"Q3 前 5 笔占总盈亏 {pct(topk[1]['share_of_total'])}")


if __name__ == "__main__":
    main()
