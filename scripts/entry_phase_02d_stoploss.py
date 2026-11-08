# -*- coding: utf-8 -*-
"""EP-02d：B 类交易止损规则回测。

严格按 out/entry_phase_02/EP02C_STOP_PREREGISTER.md（已冻结，SHA256
0FF76B97E86AC8266208156C71B4B7D48389CA6F72CE73F3A8519ED7540D1993）运行：
  - 5 条候选规则 S1..S5 + 基线 B0（不止损，60 bar 收盘卖出）；
  - 主口径：收盘确认触发，次日开盘卖出；停牌/开盘封跌停顺延至首个可成交日开盘；
  - 辅助口径：盘中触价按阈值价成交，仅作上限对照，不得用于结论；
  - 成本复用 entry_phase_02_l1.eval_trade（base/cons/stress）；
  - 组合：主组合 100 万、3 仓、每笔 1/3 NAV、先到先得、满仓跳过、回款次日可用；
    对照组合复用 L2 冻结的 20 仓 / 单笔 5% NAV 机制，仅替换出场逻辑。

产物（out/entry_phase_02/）：
  p2d_trades.csv            逐笔出场（B0+S1..S5，主/辅口径）
  p2d_rule_summary.csv      交易级总表
  p2d_segment.csv           分段（train/validate/test）净期望
  p2d_yearly.csv            逐年净收益对比
  p2d_main_nav_<rule>.csv   主组合日度 NAV
  p2d_main_summary.csv      主组合 MDD / 收益 / 闲置
  p2d_control_summary.csv   20 仓对照组合
  p2d_summary.json          结构化结果 + G1..G5 判定
  P2C_STOP_BACKTEST_REPORT.md

用法：
    python scripts/entry_phase_02d_stoploss.py --workers 8
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02_l1 as l1  # noqa: E402

OUT_DIR = os.path.join(ROOT, "out", "entry_phase_02")
SIG_CSV = os.path.join(ROOT, "out", "entry_phase_03", "p3_features.csv")
QFQ_DIR = os.path.join(ROOT, "cache", "qfq")
PREREG_MD = os.path.join(OUT_DIR, "EP02C_STOP_PREREGISTER.md")
PREREG_SHA = "0FF76B97E86AC8266208156C71B4B7D48389CA6F72CE73F3A8519ED7540D1993"

HOLD = 60
NOTIONAL = 50000.0
SEGS = ("train", "validate", "test")
TRAIN_END, VALID_END = 2016, 2021
COMP_YEARS = tuple(range(2005, 2025))   # 预注册的“20 个可比年份”（2025/2026 仅列示不参与 G4）

# (id, start_bar, threshold_pct)  —— 冻结规则，不得增删改
RULES = (
    ("B0", None, None),
    ("S1", 10, 6),
    ("S2", 10, 8),
    ("S3", 20, 6),
    ("S4", 20, 8),
    ("S5", 20, 10),
)
STOP_IDS = ("S1", "S2", "S3", "S4", "S5")
COSTS = ("base", "cons", "stress")

CAPITAL0 = 1_000_000.0
MAIN_MAX_POS, MAIN_POS_PCT = 3, 1.0 / 3.0
CTRL_MAX_POS, CTRL_POS_PCT = 20, 0.05   # L2 冻结参数


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_prereg() -> dict:
    import hashlib
    if not os.path.exists(PREREG_MD):
        raise SystemExit("预注册文件缺失，拒绝运行")
    h = hashlib.sha256(open(PREREG_MD, "rb").read()).hexdigest().upper()
    if h != PREREG_SHA:
        raise SystemExit(f"预注册文件已被修改（{h}），拒绝运行")
    rec = json.load(open(os.path.join(OUT_DIR, "EP02C_STOP_PREREGISTER_FREEZE.json"), encoding="utf-8"))
    log(f"预注册校验通过 SHA256={h[:16]}…；规则={[r['id'] for r in rec['rules']]}")
    return rec


def round2(x):
    return np.floor(np.asarray(x, dtype=np.float64) * 100.0 + 0.5) / 100.0


def board_limit_rate(code: str, dates_ns: np.ndarray) -> np.ndarray:
    """按板块与日期的涨跌幅比例（主板10%、创业板20%（2020-08-24前10%）、科创20%、北交所30%）。"""
    c = str(code).zfill(6)
    t = dates_ns.astype("datetime64[ns]").astype("datetime64[D]")
    if c[:3] in ("688", "689"):
        return np.full(t.size, 0.20)
    if c[:3] in ("300", "301"):
        cut = np.datetime64("2020-08-24", "D")
        return np.where(t < cut, 0.10, 0.20)
    if c[0] in ("4", "8"):
        cut = np.datetime64("2021-11-15", "D")
        return np.where(t < cut, 0.10, 0.30)
    return np.full(t.size, 0.10)


def load_signals() -> pd.DataFrame:
    df = pd.read_csv(SIG_CSV, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["year"] = df["signal_date"].dt.year
    return df


def load_code_arrays(code: str):
    p = os.path.join(QFQ_DIR, f"{code}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    if d is None or d.empty:
        return None
    dates = pd.to_datetime(d["date"]).values.astype("datetime64[ns]").astype(np.int64)
    o = pd.to_numeric(d["open"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(d["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(d["low"], errors="coerce").to_numpy(float)
    cl = pd.to_numeric(d["close"], errors="coerce").to_numpy(float)
    facs = l1.load_facs(code)
    if facs is None:
        f = np.ones(dates.size)
    else:
        darr, farr = l1.factor_arrays(facs)
        i = np.searchsorted(darr, dates, side="right") - 1
        i = np.maximum(i, 0)
        f = farr[i].astype(float)
    return dict(dates=dates, o=o, h=hi, l=lo, c=cl, f=f)


def limitdown_block(ro: np.ndarray, rc: np.ndarray, f: np.ndarray,
                    rate: np.ndarray) -> np.ndarray:
    """开盘即封跌停 → 当日开盘不可卖。P_ref 用除权参考价（raw 口径），0.01 元四舍五入。"""
    n = ro.size
    blk = np.zeros(n, dtype=bool)
    if n < 2:
        return blk
    prev_c = rc[:-1].copy()
    same = f[1:] == f[:-1]
    ref = np.where(same, prev_c, prev_c * f[1:] / np.where(f[:-1] == 0, np.nan, f[:-1]))
    lim = round2(ref * (1.0 - rate[1:]))
    blk[1:] = np.isfinite(lim) & np.isfinite(ro[1:]) & (ro[1:] <= lim + 1e-9)
    return blk


def _exit_stop(cr: np.ndarray, lo_r: np.ndarray, start: int, x_pct: float):
    """收盘确认触发：返回 (t, None)；盘中辅助口径另算。"""
    th = -x_pct / 100.0
    seg = cr[start:]
    hit = np.nonzero(np.isfinite(seg) & (seg <= th))[0]
    return int(hit[0]) + start if hit.size else -1


def simulate_stock(args):
    """单只股票：逐信号计算 B0 与 5 条止损规则的出场（主口径 + 辅助口径）。"""
    code, g = args
    A = load_code_arrays(code)
    if A is None:
        return [], [], {"code": code, "reason": "no_qfq"}
    dates, qo, qh, ql, qc, f = A["dates"], A["o"], A["h"], A["l"], A["c"], A["f"]
    n = dates.size
    pos_map = {int(d): i for i, d in enumerate(dates)}
    ro = qo * f
    rc = qc * f
    rate = board_limit_rate(code, dates)
    blk = limitdown_block(ro, rc, f, rate)
    rate5 = np.full(n, 0.05)
    blk5 = limitdown_block(ro, rc, f, rate5)

    trade_rows, win_rows = [], []
    bad = 0
    for row in g.itertuples():
        ed = int(pd.Timestamp(row.entry_date).value)
        di = pos_map.get(ed)
        if di is None or di + HOLD - 1 >= n:
            bad += 1
            continue
        qe = qo[di]
        if not np.isfinite(qe) or qe <= 0:
            bad += 1
            continue
        raw_e = float(qe * f[di])
        de = pd.Timestamp(row.entry_date)
        cr = qc[di:di + HOLD] / qe - 1.0
        lr = ql[di:di + HOLD] / qe - 1.0
        if not np.isfinite(cr).all():
            bad += 1
            continue

        # B0：60 bar 收盘卖出
        xi = di + HOLD - 1
        raw_b0 = float(rc[xi])
        base = dict(rule="B0", mode="main", triggered=0, trigger_bar=-1,
                    exit_bar=HOLD - 1, delay_days=0, exit_reason="fixed_60_bar",
                    hold_bars=HOLD, exit_date=str(pd.Timestamp(dates[xi]).date()),
                    raw_exit=raw_b0, entry_g=None)
        b0 = dict(base)
        gq0 = qc[xi] / qe - 1.0
        for sc in COSTS:
            t = l1.eval_trade(code, de, dates[xi], raw_e, raw_b0, sc)
            cost = (raw_b0 / raw_e - 1.0) - t["net_return"]
            b0[f"cost_{sc}"] = cost
            b0[f"net_return_{sc}"] = gq0 - cost
        b0["gross_return"] = gq0

        exits = {"B0": dict(off=xi - di, day=int(dates[xi]), raw=raw_b0,
                            reason="fixed_60_bar", delay=0, f=float(f[xi]))}
        rows = [b0]

        for rid, S, x in RULES:
            if rid == "B0":
                continue
            # ---- 辅助口径（盘中触价，独立计算，仅对照）----
            addr = -x / 100.0
            hit = np.nonzero(np.isfinite(lr[S:]) & (lr[S:] <= addr))[0]
            aux = None
            if hit.size:
                ta = int(hit[0]) + S
                ja = di + ta
                raw_aux = float(qe * (1.0 - x / 100.0) * f[ja])
                aux = dict(base)
                aux.update(rule=rid, mode="aux", triggered=1, trigger_bar=ta,
                           exit_bar=ta, delay_days=0, exit_reason="intraday_at_threshold",
                           hold_bars=ta + 1, exit_date=str(pd.Timestamp(dates[ja]).date()),
                           raw_exit=raw_aux)
                for sc in COSTS:
                    tt = l1.eval_trade(code, de, dates[ja], raw_e, raw_aux, sc)
                    cost = (raw_aux / raw_e - 1.0) - tt["net_return"]
                    aux[f"net_return_{sc}"] = -x / 100.0 - cost
                aux["gross_return"] = -x / 100.0

            t = _exit_stop(cr, lr, S, x)
            if t < 0:
                rec = dict(base)
                rec["rule"] = rid
                for sc in COSTS:
                    rec[f"net_return_{sc}"] = b0[f"net_return_{sc}"]
                rec["gross_return"] = b0["gross_return"]
                rows.append(rec)
                exits[rid] = exits["B0"]
                if aux is not None:
                    rows.append(aux)
                continue
            j = di + t + 1
            while j < n and blk[j]:
                j += 1
            if j < n:
                reason = "stop_next_open" if j == di + t + 1 else "stop_delayed"
                raw_x = float(ro[j])
                off, day = j - di, int(dates[j])
            else:
                j = min(n - 1, di + HOLD - 1)
                reason = "fallback_close"
                raw_x = float(rc[j])
                off, day = j - di, int(dates[j])
            rec = dict(base)
            rec.update(rule=rid, triggered=1, trigger_bar=t, exit_bar=off,
                       delay_days=max(0, (j - (di + t + 1))), exit_reason=reason,
                       hold_bars=off + 1, exit_date=str(pd.Timestamp(dates[j]).date()),
                       raw_exit=raw_x)
            gq = (raw_x / f[j]) / qe - 1.0 if np.isfinite(f[j]) and f[j] > 0 else np.nan
            # ST=5% 敏感性：仅统计顺延天数差异，不参与结论
            j5 = di + t + 1
            while j5 < n and blk5[j5]:
                j5 += 1
            rec["delay_days_st5"] = (min(j5, n - 1) - (di + t + 1)) if j5 < n else -1
            for sc in COSTS:
                tt = l1.eval_trade(code, de, dates[j], raw_e, raw_x, sc)
                cost = (raw_x / raw_e - 1.0) - tt["net_return"]
                rec[f"cost_{sc}"] = cost
                rec[f"net_return_{sc}"] = gq - cost
            rec["gross_return"] = gq
            # 诊断：止损后到第 60 bar 是否继续下跌（≥3%）；原路径是否最终净盈利
            c_t = float(cr[t])
            c_end = float(cr[HOLD - 1])
            rec["post_stop_ret"] = (1.0 + c_end) / (1.0 + c_t) - 1.0 if 1.0 + c_t > 0 else np.nan
            rec["post_stop_decline"] = int(rec["post_stop_ret"] <= -0.03)
            rec["orig_net_base"] = b0["net_return_base"]
            rec["orig_net_pos"] = int(b0["net_return_base"] > 0)
            rows.append(rec)
            exits[rid] = dict(off=off, day=day, raw=raw_x, reason=reason,
                              delay=max(0, j - (di + t + 1)), f=float(f[j]))
            if aux is not None:
                rows.append(aux)

        for r in rows:
            r["code"] = code
            r["signal_date"] = str(pd.Timestamp(row.signal_date).date())
            r["entry_date"] = str(de.date())
            r["seg"] = row.seg
            r["year"] = int(row.year)
            r["signal_date_ts"] = row.signal_date
        trade_rows.extend(rows)

        K = min(HOLD + 1, n - di)
        win_rows.append(dict(
            code=code, signal_date=str(pd.Timestamp(row.signal_date).date()),
            entry_date=str(de.date()), seg=row.seg,
            entry_ns=int(dates[di]), raw_entry=raw_e,
            day_ns=dates[di:di + K], rc_raw=(rc[di:di + K].copy()),
            f=(f[di:di + K].copy()), exits=exits,
        ))
    return trade_rows, win_rows, {"code": code, "bad": bad}


# ---------------------------------------------------------------------------
# 交易级指标
# ---------------------------------------------------------------------------
def rule_stats(tr: pd.DataFrame, rid: str, mode: str, sc: str) -> dict:
    d = tr[(tr["rule"] == rid) & (tr["mode"] == mode)]
    n = len(d)
    net = d[f"net_return_{sc}"].to_numpy(float)
    win = net > 0
    pos_r, neg_r = net[win], net[~win]
    payoff = float(pos_r.mean() / abs(neg_r.mean())) if pos_r.size and neg_r.size else np.nan
    pf = float(pos_r.sum() / abs(neg_r.sum())) if pos_r.size and neg_r.size and neg_r.sum() != 0 else np.nan
    st = d[d["triggered"] == 1]
    n_base = n
    if mode == "aux":
        n_base = len(tr[(tr["rule"] == rid) & (tr["mode"] == "main")])
    out = dict(rule=rid, mode=mode, cost=sc, n=n,
               n_trigger=int(len(st)),
               trigger_rate=float(len(st) / n_base) if n_base else np.nan,
               expectancy=float(net.mean()) if n else np.nan,
               total_net=float(net.sum() * NOTIONAL),
               win_rate=float(win.mean()) if n else np.nan,
               payoff=payoff, profit_factor=pf,
               avg_hold_bars=float(d["hold_bars"].mean()) if n else np.nan,
               avg_net_win=float(pos_r.mean()) if pos_r.size else np.nan,
               avg_net_loss=float(neg_r.mean()) if neg_r.size else np.nan)
    if len(st) and "post_stop_decline" in st.columns:
        out.update(
            post_stop_decline_share=float(st["post_stop_decline"].mean()),
            rebound_profit_share=float(st["orig_net_pos"].mean()),
            misfire_cost=float(st.loc[st["orig_net_base"] > 0, "orig_net_base"].sum() * NOTIONAL),
            avoided_loss=float(-st.loc[st["orig_net_base"] < 0, "orig_net_base"].sum() * NOTIONAL),
            net_improvement=float(st["orig_net_base"].sum() * NOTIONAL),
            orig_net_sum=float(st["orig_net_base"].sum() * NOTIONAL),
            delayed_share=float((st["delay_days"] > 0).mean()),
            max_delay_days=int(st["delay_days"].max()),
            st5_delay_diff=int((st["delay_days_st5"] != st["delay_days"]).sum()),
        )
    else:
        for k in ("post_stop_decline_share", "rebound_profit_share", "misfire_cost",
                  "avoided_loss", "net_improvement", "orig_net_sum", "delayed_share",
                  "max_delay_days", "st5_delay_diff"):
            out[k] = np.nan if k != "max_delay_days" and k != "st5_delay_diff" else 0
    return out


def segment_table(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, _, _ in RULES:
        for seg in SEGS:
            d = tr[(tr["rule"] == rid) & (tr["mode"] == "main") & (tr["seg"] == seg)]
            net = d["net_return_base"].to_numpy(float)
            rows.append(dict(rule=rid, seg=seg, n=len(d),
                             expectancy=float(net.mean()) if len(d) else np.nan,
                             win_rate=float((net > 0).mean()) if len(d) else np.nan,
                             total_net=float(net.sum() * NOTIONAL)))
    return pd.DataFrame(rows)


def yearly_table(tr: pd.DataFrame) -> pd.DataFrame:
    d = tr[tr["mode"] == "main"]
    p = d.pivot_table(index="year", columns="rule", values="net_return_base",
                      aggfunc="sum") * NOTIONAL
    return p.reset_index()


# ---------------------------------------------------------------------------
# 主组合：100 万 / 3 仓 / 1-3 NAV
# ---------------------------------------------------------------------------
def run_main_portfolio(sigs: list[dict], cal_ns: np.ndarray, cal_pos: dict,
                       rule: str, cost: str) -> dict:
    from collections import defaultdict
    slip, comm = l1.SLIP[cost], l1.COMM[cost]
    by_entry = defaultdict(list)
    for s in sigs:
        g = cal_pos.get(pd.Timestamp(s["entry_date"]))
        if g is None:
            continue
        s["g_entry"] = g
        by_entry[g].append(s)
    for g in by_entry:
        by_entry[g].sort(key=lambda x: (x["signal_date"], x["code"]))

    NCAL = cal_ns.size
    days = pd.to_datetime(cal_ns)
    cash_av = CAPITAL0
    cash_pend = 0.0
    pos: dict[str, dict] = {}
    nav = np.full(NCAL, np.nan)
    nav_prev = CAPITAL0
    npos = np.zeros(NCAL, int)
    trades, skipped = [], defaultdict(int)
    pending_sell: dict[str, dict] = {}

    def fee_sell(code, d, amt, shares):
        return (max(amt * comm, l1.MIN_COMM) + amt * l1.stamp_rate(d, "sell")
                + l1.transfer_fee(code, d, "sell", amt, shares, cost))

    def fee_buy(code, d, amt, shares):
        return max(amt * comm, l1.MIN_COMM) + l1.transfer_fee(code, d, "buy", amt, shares, cost)

    for g in range(NCAL):
        day = days[g]
        dns = int(cal_ns[g])
        # 1) 卖出
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
            trades.append(dict(code=code, entry_date=p["entry_date"],
                               exit_date=p["exit_date"], hold_bars=p["exit_bar"],
                               ret=p["pnl"] / p["notional"], pnl=p["pnl"],
                               reason=ex["reason"], exit_price=px))
            del pos[code]
            pending_sell.pop(code, None)
        # 2) 买入
        cands = by_entry.get(g, [])
        for s in cands:
            if len(pos) >= MAIN_MAX_POS:
                skipped["max_positions"] += 1
                continue
            if s["code"] in pos:
                skipped["duplicate"] += 1
                continue
            ex = s["exits"][rule]
            target = MAIN_POS_PCT * nav_prev
            if cash_av < target:
                skipped["cash_insufficient"] += 1
                continue
            notional = target
            px = s["raw_entry"]
            shares = notional / px
            amt = notional * (1.0 + slip)
            f = fee_buy(s["code"], day, amt, shares)
            cash_av -= amt + f
            pos[s["code"]] = dict(
                shares=shares, notional=notional, entry_date=str(pd.Timestamp(s["entry_date"]).date()),
                entry_bar=0, f_last=float(s["f"][0]), last_c=float(s["rc_raw"][0]),
                ptr=1, win=s, exit=ex, exit_bar=ex["off"], cost_in=amt + f, pnl=np.nan)
            pending_sell[s["code"]] = ex
        # 3) 盯市
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
        nav[g] = cash_av + cash_pend + val
        npos[g] = len(pos)
        nav_prev = nav[g]
        cash_av += cash_pend
        cash_pend = 0.0

    open_end = [dict(code=c, entry_date=p["entry_date"], value=p["shares"] * p["last_c"])
                for c, p in pos.items()]
    first = int(np.argmax(npos > 0)) if (npos > 0).any() else 0
    nav_f = nav[first:]
    peak = np.maximum.accumulate(nav_f)
    dd = (peak - nav_f) / peak
    i_dd = int(np.argmax(dd)) + first
    i_pk = int(np.argmax(nav[first:i_dd + 1])) + first
    idle = int((npos[first:] == 0).sum())
    return dict(rule=rule, cost=cost, nav=nav, dates=days, npos=npos,
                final_nav=float(nav[-1]), total_return=float(nav[-1] / CAPITAL0 - 1),
                mdd=float(dd.max()) if len(dd) else np.nan,
                mdd_peak=str(days[i_pk].date()), mdd_trough=str(days[i_dd].date()),
                trades=pd.DataFrame(trades), skipped=dict(skipped),
                open_end=open_end, idle_days=idle, first_g=first)


# ---------------------------------------------------------------------------
# 对照组合：复用 L2 冻结机制（20 仓 / 5% NAV / 容量 0.5% / ST_UNKNOWN 5%）
# ---------------------------------------------------------------------------
def run_control(sigs: list[dict], rule: str, cost: str = "base") -> dict:
    import scripts.entry_phase_02_l2 as l2
    slip, comm = l1.SLIP[cost], l1.COMM[cost]
    dates = pd.to_datetime(ep.CTX["cal"])
    NCAL = l2.NCAL
    cash = CAPITAL0
    pos: dict[str, dict] = {}
    pend: list[dict] = []
    by_day: dict[int, list[dict]] = {}
    for s in sigs:
        by_day.setdefault(s["g0"], []).append(s)

    nav_prev = CAPITAL0
    daily = np.full(NCAL, np.nan)
    npos = np.zeros(NCAL, dtype=np.int32)
    trades: list[dict] = []
    reasons: dict[str, int] = {}

    def add_reason(k):
        reasons[k] = reasons.get(k, 0) + 1

    for g in range(NCAL):
        # ---- 卖出 ----
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
                p["f_last"] = float(s["f"][i])
            p["last_c"] = float(s["rc"][i])
            p["bars"] += 1
            t = p["bars"] - 1

            if rule != "B0" and not p["pending"]:
                _, S, x = next(r for r in RULES if r[0] == rule)
                if t >= S:
                    qe = p["entry_price"] / p["f_entry"]
                    qt = float(s["rc"][i]) / float(s["f"][i])
                    if np.isfinite(qt) and qt / qe - 1.0 <= -x / 100.0:
                        p["pending"] = True
                        p["trigger_bar"] = t
                        p["trigger_g"] = g

            do_sell = False
            if rule == "B0":
                do_sell = (p["bars"] >= l2.SELL_BAR) and not s["sell5"][i]
            else:
                # 触发后次日开盘卖；未触发者仍按 60 bar 固定持有（第 61 bar 开盘）出场
                do_sell = (((p["pending"] and g > p["trigger_g"])
                            or p["bars"] >= l2.SELL_BAR)
                           and not s["sell5"][i])
            if do_sell:
                gross = p["shares"] * float(s["ro"][i])
                amt = gross * (1.0 - slip)
                fees = (max(amt * comm, l1.MIN_COMM)
                        + amt * l1.stamp_rate(dates[g], "sell")
                        + l1.transfer_fee(code, dates[g], "sell", amt, p["shares"], cost))
                cash += amt - fees
                trades.append(dict(code=code, signal_date=s["signal_date"],
                                   entry_date=p["entry_date"],
                                   exit_date=str(dates[g].date()),
                                   hold_bars=p["bars"], notional=p["notional"],
                                   ret=(amt - fees - p["cost_in"]) / p["notional"],
                                   pnl=amt - fees - p["cost_in"],
                                   exit_reason=("fixed_61_bar" if rule == "B0"
                                                else p.get("exit_reason", "stop")),
                                   triggered=int(p["pending"])))
                del pos[code]

        # ---- 买入（与 L2 完全一致）----
        nav_now = nav_prev
        cand = list(pend)
        pend = []
        for s in sorted(by_day.get(g, []), key=lambda x: x["hkey"]):
            cand.append(dict(sig=s, deadline=s["g0"] + l2.ENTRY_WINDOW - 1, last="new"))
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
            if len(pos) >= l2.MAX_POS:
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
            if s["buy5"][i]:
                o["last"] = "limit_up"
                pend.append(o)
                continue
            target = l2.POS_PCT * nav_now
            if not np.isfinite(s["adv"]):
                add_reason("adv_insufficient")
                continue
            cap_amt = s["adv"] * 0.005
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
            pos[code] = dict(sig=s, code=code, shares=shares, notional=notional,
                             f_last=float(s["f"][i]), f_entry=float(s["f"][i]),
                             last_c=float(s["rc"][i]), entry_g=g,
                             entry_date=str(dates[g].date()),
                             entry_price=float(s["ro"][i]), bars=1,
                             cost_in=amt + fees, pending=False, trigger_g=-1,
                             trigger_bar=-1)

        val = sum(p["shares"] * p["last_c"] for p in pos.values())
        nav = cash + val
        daily[g] = nav
        npos[g] = len(pos)
        nav_prev = nav

    open_end = len(pos)
    first = int(np.argmax(npos > 0)) if (npos > 0).any() else 0
    nav_f = daily[first:]
    peak = np.maximum.accumulate(nav_f)
    dd = (peak - nav_f) / peak
    i_dd = int(np.argmax(dd)) + first
    i_pk = int(np.argmax(daily[first:i_dd + 1])) + first
    return dict(rule=rule, cost=cost, nav=daily, dates=dates, npos=npos,
                final_nav=float(daily[-1]), total_return=float(daily[-1] / CAPITAL0 - 1),
                mdd=float(dd.max()), mdd_peak=str(dates[i_pk].date()),
                mdd_trough=str(dates[i_dd].date()), trades=pd.DataFrame(trades),
                n_filled=len(trades), reasons=reasons, open_end=open_end)


def seg_mdd(nav: np.ndarray, dates: pd.DatetimeIndex, start: str, end: str) -> dict:
    w = (dates >= start) & (dates <= end) & np.isfinite(nav)
    if w.sum() < 2:
        return dict(mdd=np.nan, peak="", trough="")
    x = nav[w]
    d = dates[w]
    peak = np.maximum.accumulate(x)
    dd = (peak - x) / peak
    i = int(np.argmax(dd))
    j = int(np.argmax(x[:i + 1]))
    return dict(mdd=float(dd[i]), peak=str(d[j].date()), trough=str(d[i].date()))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-control", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    verify_prereg()
    ep.init_ctx()
    cal = pd.to_datetime(ep.CTX["cal"])
    cal_ns = cal.values.astype("datetime64[ns]").astype(np.int64)
    cal_pos = {pd.Timestamp(k): v for k, v in ep.CTX["cal_pos"].items()}
    log(f"日历 {len(cal)} 日")

    sig = load_signals()
    log(f"信号 {len(sig)} 笔 / {sig['code'].nunique()} 只")

    trade_rows, win_rows, diag = [], [], []
    groups = list(sig.groupby("code", sort=False))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (trs, wins, dg) in enumerate(ex.map(simulate_stock, groups), 1):
            trade_rows.extend(trs)
            win_rows.extend(wins)
            diag.append(dg)
            if i % 500 == 0 or i == len(groups):
                log(f"  进度 {i}/{len(groups)}  {time.time()-t0:.0f}s")

    tr = pd.DataFrame(trade_rows)
    tr = tr.drop(columns=["signal_date_ts"])
    tr.to_csv(os.path.join(OUT_DIR, "p2d_trades.csv"), index=False, encoding="utf-8")
    n_bad = sum(d.get("bad", 0) for d in diag if "bad" in d)
    n_nofile = sum(1 for d in diag if d.get("reason") == "no_qfq")
    log(f"逐笔 {len(tr)} 行；无法模拟 {n_bad} 笔 / 缺文件 {n_nofile} 只")

    # 自检：B0 base 净收益应等于 p3_features 的 net60
    key = sig.set_index(["code", "signal_date"])["net60"]
    b0 = tr[(tr["rule"] == "B0") & (tr["mode"] == "main")].copy()
    idx = pd.MultiIndex.from_arrays([b0["code"], pd.to_datetime(b0["signal_date"])])
    diff = np.abs(b0["net_return_base"].to_numpy(float) - key.reindex(idx).to_numpy(float))
    log(f"自检 B0 vs net60: max|diff|={np.nanmax(diff):.2e} nan={int(np.isnan(diff).sum())}")

    # ---- 交易级总表 ----
    rows = []
    for rid, _, _ in RULES:
        for sc in COSTS:
            rows.append(rule_stats(tr, rid, "main", sc))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT_DIR, "p2d_rule_summary.csv"), index=False, encoding="utf-8")
    base_tab = summ[summ["cost"] == "base"].copy()

    aux_rows = [rule_stats(tr, rid, "aux", "base") for rid in STOP_IDS]
    aux_tab = pd.DataFrame(aux_rows)
    aux_tab.to_csv(os.path.join(OUT_DIR, "p2d_aux_summary.csv"), index=False, encoding="utf-8")

    seg_tab = segment_table(tr)
    seg_tab.to_csv(os.path.join(OUT_DIR, "p2d_segment.csv"), index=False, encoding="utf-8")
    year_tab = yearly_table(tr)
    year_tab.to_csv(os.path.join(OUT_DIR, "p2d_yearly.csv"), index=False, encoding="utf-8")

    # ---- 主组合 ----
    mains = {}
    for rid, _, _ in RULES:
        for sc in COSTS:
            r = run_main_portfolio(win_rows, cal_ns, cal_pos, rid, sc)
            mains[(rid, sc)] = r
            if sc == "base":
                pd.DataFrame(dict(date=r["dates"], nav=r["nav"], n_open=r["npos"])).to_csv(
                    os.path.join(OUT_DIR, f"p2d_main_nav_{rid}.csv"), index=False, encoding="utf-8")
            log(f"  主组合 {rid}/{sc}: NAV={r['final_nav']:,.0f} ret={r['total_return']:.2%} "
                f"MDD={r['mdd']:.2%} 成交={len(r['trades'])} 闲置={r['idle_days']}")

    mrows = []
    for rid, _, _ in RULES:
        for sc in COSTS:
            r = mains[(rid, sc)]["nav"]
            mrows.append(dict(rule=rid, cost=sc,
                              final_nav=mains[(rid, sc)]["final_nav"],
                              total_return=mains[(rid, sc)]["total_return"],
                              mdd=mains[(rid, sc)]["mdd"],
                              mdd_peak=mains[(rid, sc)]["mdd_peak"],
                              mdd_trough=mains[(rid, sc)]["mdd_trough"],
                              n_filled=len(mains[(rid, sc)]["trades"]),
                              idle_days=mains[(rid, sc)]["idle_days"],
                              skipped=json.dumps(mains[(rid, sc)]["skipped"], ensure_ascii=False),
                              open_end=len(mains[(rid, sc)]["open_end"])))
    main_df = pd.DataFrame(mrows)
    main_df.to_csv(os.path.join(OUT_DIR, "p2d_main_summary.csv"), index=False, encoding="utf-8")

    seg_mdd_rows = []
    for rid, _, _ in RULES:
        r = mains[(rid, "base")]
        for nm, a, b in (("train", "2005-01-01", "2016-12-31"),
                         ("validate", "2017-01-01", "2021-12-31"),
                         ("test", "2022-01-01", "2026-12-31")):
            s = seg_mdd(r["nav"], r["dates"], a, b)
            seg_mdd_rows.append(dict(rule=rid, seg=nm, mdd=s["mdd"],
                                     peak=s["peak"], trough=s["trough"]))
    seg_mdd_df = pd.DataFrame(seg_mdd_rows)
    seg_mdd_df.to_csv(os.path.join(OUT_DIR, "p2d_main_seg_mdd.csv"), index=False, encoding="utf-8")

    # ---- 对照组合（L2 机制）----
    ctrl_csv = os.path.join(OUT_DIR, "p2d_control_summary.csv")
    ctrl_df = pd.DataFrame()
    if args.skip_control and os.path.exists(ctrl_csv):
        ctrl_df = pd.read_csv(ctrl_csv)
        log("对照组合：复用已有 p2d_control_summary.csv")
    elif not args.skip_control:
        import scripts.entry_phase_02_l2 as l2
        l2.NCAL = len(cal)
        l2.CAL_NS = cal_ns
        l2.CAL_POS = ep.CTX["cal_pos"]
        tl = sig[["code", "signal_date", "entry_date"]].copy()
        tl["exit_date"] = tl["entry_date"]
        csigs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(l2.build_stock, c, g) for c, g in tl.groupby("code")]
            for f in futs:
                for s in f.result():
                    s["hkey"] = l2.hash_key(s["signal_date"], s["code"])
                    csigs.append(s)
        csigs.sort(key=lambda s: (s["g0"], s["hkey"]))
        log(f"对照组合可用信号 {len(csigs)}")
        crows = []
        for rid, _, _ in RULES:
            r = run_control(csigs, rid)
            crows.append(dict(rule=rid, final_nav=r["final_nav"], total_return=r["total_return"],
                              mdd=r["mdd"], mdd_peak=r["mdd_peak"], mdd_trough=r["mdd_trough"],
                              n_filled=r["n_filled"], open_end=r["open_end"],
                              reasons=json.dumps(r["reasons"], ensure_ascii=False)))
            log(f"  对照 {rid}: NAV={r['final_nav']:,.0f} MDD={r['mdd']:.2%} 成交={r['n_filled']}")
        ctrl_df = pd.DataFrame(crows)
        ctrl_df.to_csv(os.path.join(OUT_DIR, "p2d_control_summary.csv"), index=False, encoding="utf-8")

    # ---- 门槛判定 ----
    gates = eval_gates(base_tab, seg_tab, year_tab, mains, ctrl_df)
    passed_rules = [r for r in STOP_IDS if gates["rule_" + r]["passed"]]
    adopted = (max(passed_rules,
                   key=lambda r: base_tab.loc[base_tab["rule"] == r, "expectancy"].iloc[0])
               if passed_rules else None)
    verdict = "ADOPT" if adopted else "REJECT_ALL"

    s = dict(
        meta=dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  prereg_sha256=PREREG_SHA, n_signals=int(len(sig)),
                  n_trades=int(len(tr[(tr["rule"] == "B0") & (tr["mode"] == "main")])),
                  n_bad=int(n_bad), n_no_qfq=int(n_nofile),
                  rules=[dict(id=r, start_bar=S, threshold_pct=x) for r, S, x in RULES],
                  elapsed_s=round(time.time() - t0, 1)),
        trade=base_tab.to_dict(orient="records"),
        trade_all_costs=summ.to_dict(orient="records"),
        aux=aux_tab.to_dict(orient="records"),
        segment=seg_tab.to_dict(orient="records"),
        yearly=year_tab.to_dict(orient="records"),
        main_portfolio=main_df.to_dict(orient="records"),
        main_seg_mdd=seg_mdd_df.to_dict(orient="records"),
        control=ctrl_df.to_dict(orient="records") if len(ctrl_df) else [],
        gates=gates, verdict=verdict, adopted=adopted,
    )
    with open(os.path.join(OUT_DIR, "p2d_summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2, default=float)

    write_report(s, base_tab, aux_tab, seg_tab, year_tab, main_df, seg_mdd_df, ctrl_df)
    log(f"完成：verdict={verdict} adopted={adopted} 用时 {time.time()-t0:.0f}s")


def eval_gates(base_tab, seg_tab, year_tab, mains, ctrl_df) -> dict:
    b0 = base_tab[base_tab["rule"] == "B0"].iloc[0]
    g = {}
    for rid in STOP_IDS:
        r = base_tab[base_tab["rule"] == rid].iloc[0]
        sg = seg_tab[seg_tab["rule"] == rid].set_index("seg")["expectancy"]
        b0seg = seg_tab[seg_tab["rule"] == "B0"].set_index("seg")["expectancy"]
        yb = year_tab.set_index("year")["B0"]
        yr = year_tab.set_index("year")[rid]
        yrs = [y for y in COMP_YEARS if y in yr.index and y in yb.index]
        bad_years = int((yr.reindex(yrs) < yb.reindex(yrs)).sum())
        m0 = mains[("B0", "base")]["mdd"]
        m1 = mains[(rid, "base")]["mdd"]
        g1 = bool(r["expectancy"] >= b0["expectancy"] and r["total_net"] >= b0["total_net"])
        g2 = bool((m0 - m1) >= 0.05 and ((m0 - m1) / m0) >= 0.15)
        g3 = bool(sg.get("validate", np.nan) >= b0seg.get("validate", np.nan)
                  and sg.get("test", np.nan) >= b0seg.get("test", np.nan))
        g4 = bool(bad_years <= 9)
        g5 = bool(r["net_improvement"] > 0)
        g["rule_" + rid] = dict(rule=rid, G1=g1, G2=g2, G3=g3, G4=g4, G5=g5,
                                bad_years=bad_years, n_years=int(len(yrs)),
                                mdd_base=float(m0), mdd_rule=float(m1),
                                mdd_drop_pp=float((m0 - m1) * 100),
                                mdd_drop_rel=float((m0 - m1) / m0),
                                expectancy_base=float(b0["expectancy"]),
                                expectancy_rule=float(r["expectancy"]),
                                passed=bool(g1 and g2 and g3 and g4 and g5))
    g["G1"] = dict(id="G1", name="不砍盈利空间",
                   pass_all=any(g["rule_" + r]["G1"] for r in STOP_IDS))
    g["G2"] = dict(id="G2", name="回撤实质下降",
                   pass_all=any(g["rule_" + r]["G2"] for r in STOP_IDS))
    g["G3"] = dict(id="G3", name="样本外成立",
                   pass_all=any(g["rule_" + r]["G3"] for r in STOP_IDS))
    g["G4"] = dict(id="G4", name="稳定性",
                   pass_all=any(g["rule_" + r]["G4"] for r in STOP_IDS))
    g["G5"] = dict(id="G5", name="误杀可控",
                   pass_all=any(g["rule_" + r]["G5"] for r in STOP_IDS))
    return g


def write_report(s, base_tab, aux_tab, seg_tab, year_tab, main_df, seg_mdd_df, ctrl_df):
    m = s["meta"]
    L = []
    L.append("# EP-02d：B 类交易止损规则回测报告")
    L.append("")
    L.append(f"- 生成时间：{m['generated_at']}　用时 {m['elapsed_s']} 秒")
    L.append(f"- 预注册协议 SHA256：`{m['prereg_sha256']}`（运行前已校验，未改动）")
    L.append(f"- 样本：{m['n_trades']:,} 笔 B 类交易（信号 {m['n_signals']:,}）；"
             f"无法模拟 {m['n_bad']} 笔，缺价格文件 {m['n_no_qfq']} 只")
    L.append("- 规则：S1(10/−6) S2(10/−8, 主假设) S3(20/−6) S4(20/−8) S5(20/−10)，B0 不止损")
    L.append("")
    L.append("## 0. 结论")
    L.append("")
    if s["adopted"]:
        L.append(f"**采用 {s['adopted']}**（全部 G1—G5 门槛通过，按净期望改善幅度排序）。")
    else:
        L.append("**5 条规则全部未通过预注册门槛 → 不采用止损，本方向结项；"
                 "不新增阈值、不改起点、不重试。**")
    L.append("")
    L.append("按规则逐条判定（`pass` = G1—G5 全部满足）：")
    L.append("")
    L.append("| 规则 | G1 不砍盈利 | G2 回撤≥5pp/15% | G3 样本外 | G4 坏年份≤9 | G5 误杀净改善 | 结论 |")
    L.append("|---|---|---|---|---|---|---|")
    for rid in STOP_IDS:
        g = s["gates"]["rule_" + rid]
        L.append(f"| {rid} | {'通过' if g['G1'] else '不通过'} | {'通过' if g['G2'] else '不通过'} "
                 f"| {'通过' if g['G3'] else '不通过'} | {'通过' if g['G4'] else '不通过'}"
                 f"（{g['bad_years']}/{g['n_years']}） | {'通过' if g['G5'] else '不通过'} "
                 f"| **{'采用' if g['passed'] else '不采用'}** |")
    L.append("")
    L.append("G2 计算明细（主组合，base 成本）：")
    L.append("")
    L.append("| 规则 | B0 MDD | 本规则 MDD | 绝对下降(pp) | 相对下降 |")
    L.append("|---|---|---|---|---|")
    for rid in STOP_IDS:
        g = s["gates"]["rule_" + rid]
        L.append(f"| {rid} | {g['mdd_base']:.2%} | {g['mdd_rule']:.2%} | "
                 f"{g['mdd_drop_pp']:+.2f} | {g['mdd_drop_rel']:+.2%} |")
    L.append("")
    L.append("## 1. 交易级总表（主口径：收盘确认、次日开盘成交）")
    L.append("")
    L.append("| 规则 | 笔数 | 触发率 | 净期望 | 总净收益(元) | 胜率 | 盈亏比 | PF | 平均持有bar |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in base_tab.iterrows():
        L.append(f"| {r['rule']} | {int(r['n']):,} | {r['trigger_rate']:.2%} | "
                 f"{r['expectancy']:+.4%} | {r['total_net']:,.0f} | {r['win_rate']:.2%} | "
                 f"{r['payoff']:.3f} | {r['profit_factor']:.3f} | {r['avg_hold_bars']:.1f} |")
    L.append("")
    L.append("成本敏感性（净期望 / total_net）：")
    L.append("")
    L.append("| 规则 | base | cons | stress |")
    L.append("|---|---|---|---|")
    for rid, _, _ in RULES:
        row = []
        for sc in COSTS:
            d = s["trade_all_costs"]
            x = next(r for r in d if r["rule"] == rid and r["cost"] == sc)
            row.append(f"{x['expectancy']:+.4%} / {x['total_net']:,.0f}")
        L.append(f"| {rid} | " + " | ".join(row) + " |")
    L.append("")
    L.append("## 2. 分段净期望（train 2005—2016 / validate 2017—2021 / test 2022—2026）")
    L.append("")
    piv = seg_tab.pivot_table(index="rule", columns="seg", values="expectancy")
    L.append("| 规则 | train | validate | test |")
    L.append("|---|---|---|---|")
    for rid, _, _ in RULES:
        if rid in piv.index:
            v = piv.loc[rid]
            L.append(f"| {rid} | {v.get('train', np.nan):+.4%} | {v.get('validate', np.nan):+.4%} "
                     f"| {v.get('test', np.nan):+.4%} |")
    L.append("")
    L.append("## 3. 逐年净收益（元，交易级，未隐藏任何年份）")
    L.append("")
    cols = ["year"] + [rid for rid, _, _ in RULES]
    L.append("| " + " | ".join(["年份"] + [rid for rid, _, _ in RULES]) + " |")
    L.append("|" + "---|" * (len(cols)) )
    for _, r in year_tab.sort_values("year").iterrows():
        L.append("| " + " | ".join([str(int(r["year"]))] +
                                   [f"{r[c]:,.0f}" if np.isfinite(r[c]) else "-" for c in cols[1:]]) + " |")
    L.append("")
    L.append("## 4. 误杀统计（G5）")
    L.append("")
    L.append("| 规则 | 被止损 | 止损后继续下跌≥3% | 反弹并最终盈利 | 误杀代价(元) | 避免损失(元) | 净改善(元) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in base_tab[base_tab["rule"].isin(STOP_IDS)].iterrows():
        L.append(f"| {r['rule']} | {int(r['n_trigger']):,} | {r['post_stop_decline_share']:.2%} | "
                 f"{r['rebound_profit_share']:.2%} | {r['misfire_cost']:,.0f} | "
                 f"{r['avoided_loss']:,.0f} | {r['net_improvement']:+,.0f} |")
    L.append("")
    L.append("口径：误杀代价 = 被止损交易中原本净盈利者的原 60 bar 净收益之和；"
             "避免损失 = 原本净亏损者的亏损绝对值之和；净改善 = 避免损失 − 误杀代价。")
    L.append("")
    L.append("## 5. 辅助口径（盘中触价，仅作上限对照，不得用于结论）")
    L.append("")
    L.append("| 规则 | 触发率 | 净期望 | 总净收益(元) | 胜率 |")
    L.append("|---|---|---|---|---|")
    for _, r in aux_tab.iterrows():
        L.append(f"| {r['rule']} | {r['trigger_rate']:.2%} | {r['expectancy']:+.4%} | "
                 f"{r['total_net']:,.0f} | {r['win_rate']:.2%} |")
    L.append("")
    L.append("**辅助口径按协议要求把成交价钉在阈值价上，因此每笔止损的毛收益恒为 −x，"
             "它机械地锁死了止损单的亏损上限，并不构成“乐观上限”；主结论只看第 1 节。**")
    L.append("")
    L.append("## 6. 主组合（100 万 / 3 仓 / 每笔 1/3 NAV / 先到先得 / 回款次日可用）")
    L.append("")
    L.append("| 规则 | 成本 | 期末 NAV | 累计 | MDD | MDD 区间 | 成交 | 闲置天数 | 期末未平仓 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in main_df.iterrows():
        L.append(f"| {r['rule']} | {r['cost']} | {r['final_nav']:,.0f} | {r['total_return']:.2%} | "
                 f"{r['mdd']:.2%} | {r['mdd_peak']} → {r['mdd_trough']} | {int(r['n_filled']):,} | "
                 f"{int(r['idle_days']):,} | {int(r['open_end'])} |")
    L.append("")
    L.append("分段 MDD（主组合 base）：")
    L.append("")
    L.append("| 规则 | train | validate | test |")
    L.append("|---|---|---|---|")
    sp = seg_mdd_df.pivot_table(index="rule", columns="seg", values="mdd")
    for rid, _, _ in RULES:
        if rid in sp.index:
            v = sp.loc[rid]
            L.append(f"| {rid} | {v.get('train', np.nan):.2%} | {v.get('validate', np.nan):.2%} "
                     f"| {v.get('test', np.nan):.2%} |")
    L.append("")
    if len(ctrl_df):
        L.append("## 7. 对照组合（L2 冻结机制：20 仓 / 5% NAV / 容量 0.5% / ST_UNKNOWN 5%）")
        L.append("")
        L.append("| 规则 | 期末 NAV | 累计 | MDD | MDD 区间 | 成交 | 期末未平仓 |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in ctrl_df.iterrows():
            L.append(f"| {r['rule']} | {r['final_nav']:,.0f} | {r['total_return']:.2%} | "
                     f"{r['mdd']:.2%} | {r['mdd_peak']} → {r['mdd_trough']} | "
                     f"{int(r['n_filled']):,} | {int(r['open_end'])} |")
        L.append("")
        L.append("B0 行即 EP-02 L2 冻结结果（历史 55.60% MDD），用于确认本脚本对 L2 机制的复刻无误；"
                 "止损列的差异只来自出场逻辑替换。")
        L.append("")
    L.append("## 8. 执行统计（主口径）")
    L.append("")
    L.append("| 规则 | 顺延成交占比 | 最大顺延天数 | ST=5% 敏感性下顺延不同的笔数 | 触发率 |")
    L.append("|---|---|---|---|---|")
    for _, r in base_tab[base_tab["rule"].isin(STOP_IDS)].iterrows():
        L.append(f"| {r['rule']} | {r['delayed_share']:.2%} | {int(r['max_delay_days'])} | "
                 f"{int(r['st5_delay_diff'])} | {r['trigger_rate']:.2%} |")
    L.append("")
    L.append("## 9. 口径与局限")
    L.append("")
    L.append("1. 入场价 = 信号次日开盘（qfq 换算 raw），与 L1/L2 一致；B0 = 60 bar 收盘卖出。")
    L.append("2. 主口径触发用**收盘价**，卖出用**次日开盘**；开盘即封跌停视为不可成交，顺延至首个可成交日开盘；"
             "若数据在顺延中结束，按最后可得收盘成交并计入 `fallback_close`。")
    L.append("3. 涨跌停按板块与日期（主板 10%、创业板 20%（2020-08-24 前 10%）、科创板 20%、北交所 30%）。"
             "`HISTORICAL_SECURITY_STATUS.csv` 无 ST 历史状态记录，主口径未叠加 ST=5%；"
             "第 8 节给出 ST=5% 敏感性下的顺延差异（不参与结论）。")
    L.append("4. 现金分红未计入（轻微低估收益）；拆股/送转通过因子变化调整持股数。")
    L.append("5. 主组合在卖出回款次日才可用于买入；现金不足 1/3 NAV 时跳过该信号并记录，不降规模买入。")
    L.append("6. 期末未平仓按最后收盘估值，不计入已实现收益。")
    L.append("7. 本报告未做任何阈值/起点/观察期的增补，也不按年份或板块选择性应用止损。")
    L.append("8. 主组合只有 3 个仓位、且回款次日才可用，现金约束会在信号入口处产生路径依赖："
             "成本档之间的极小差异可能改变当天能否建仓，进而选中完全不同的少数交易。"
             "因此 **G2 只用 base 档判定**，cons/stress 两档的主组合净值仅作参考，不作结论依据。")
    L.append("")
    with open(os.path.join(OUT_DIR, "P2C_STOP_BACKTEST_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
