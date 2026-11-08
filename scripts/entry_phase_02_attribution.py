# -*- coding: utf-8 -*-
"""EP-02 归因：55.60% 最大回撤来源 + 20 仓筛选偏差。

不修改策略、不做参数搜索。输入全部来自已完成的 L1/L2 结果与 cache/qfq。
"""
from __future__ import annotations

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

OUT = os.path.join(ROOT, "out", "entry_phase_02")
QFQ = os.path.join(ROOT, "cache", "qfq")
SEED = 20261107


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def worker(code: str, rows: pd.DataFrame, cal_ns: np.ndarray, ncal: int):
    p = os.path.join(QFQ, f"{code}.parquet")
    if not os.path.exists(p):
        return code, None, {}
    df = pd.read_parquet(p)
    d = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)
    c = df["close"].to_numpy(float)
    pos = np.searchsorted(cal_ns, d)
    ok = (pos < ncal)
    ok[ok] &= cal_ns[pos[ok]] == d[ok]
    ret = np.full(ncal, np.nan, np.float32)
    if c.size > 1:
        r = c[1:] / c[:-1] - 1.0
        ret[pos[1:][ok[1:]]] = r[ok[1:]]
    out = {}
    for row in rows.itertuples():
        di = int(np.searchsorted(d, np.int64(pd.Timestamp(row.entry_date).value)))
        if di >= c.size or pd.Timestamp(d[di]).normalize() != pd.Timestamp(row.entry_date).normalize():
            continue
        if di + 19 < c.size and np.isfinite(row.qfq_entry) and row.qfq_entry > 0:
            out[pd.Timestamp(row.signal_date).strftime("%Y-%m-%d")] = float(
                c[di + 19] / row.qfq_entry - 1.0)
    return code, ret, out


def metrics(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return dict(n=0)
    pos = x[x > 0]
    neg = x[x <= 0]
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()),
                win=float(len(pos) / len(x)),
                payoff=float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else np.nan)


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    d = np.empty(n)
    for i in range(n):
        d[i] = rng.choice(a, a.size, True).mean() - rng.choice(b, b.size, True).mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def board(code: str) -> str:
    if code.startswith(("600", "601", "603", "605")):
        return "沪主板"
    if code.startswith(("000", "001", "002", "003")):
        return "深主板/中小"
    if code.startswith("300"):
        return "创业板"
    if code.startswith("688"):
        return "科创板"
    return "其他"


def main() -> None:
    ep.init_ctx()
    cal = pd.DatetimeIndex(ep.CTX["cal"])
    cal_ns = cal.values.astype("datetime64[ns]").astype(np.int64)
    ncal = len(cal)

    l1 = pd.read_csv(os.path.join(OUT, "trades_b_l1.csv"), dtype={"code": str})
    l1["code"] = norm_code(l1["code"])
    for col in ("signal_date", "entry_date", "exit_date"):
        l1[col] = pd.to_datetime(l1[col]).dt.normalize()
    l2 = pd.read_csv(os.path.join(OUT, "l2_trades_b_main.csv"), dtype={"code": str})
    l2["code"] = norm_code(l2["code"])
    for col in ("signal_date", "entry_date", "exit_date"):
        l2[col] = pd.to_datetime(l2[col]).dt.normalize()

    key = list(zip(l1["code"], l1["signal_date"]))
    fk = set(zip(l2["code"], l2["signal_date"]))
    l1["filled"] = [k in fk for k in key]
    l1["r60q"] = l1["qfq_exit"] / l1["qfq_entry"] - 1.0

    # ---- 并行：20 日收益 + 等权市场代理 ----
    t0 = time.time()
    r20 = {}
    msum = np.zeros(ncal)
    mcnt = np.zeros(ncal)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(worker, c, g, cal_ns, ncal): c for c, g in l1.groupby("code")}
        done = 0
        for fut in as_completed(futs):
            code, ret, o = fut.result()
            for k, v in o.items():
                r20[(code, k)] = v
            if ret is not None:
                msum += np.nan_to_num(ret, nan=0.0)
                mcnt += np.isfinite(ret)
            done += 1
            if done % 500 == 0 or done == len(futs):
                log(f"  qfq {done}/{len(futs)} {time.time()-t0:.0f}s")
    l1["r20q"] = [r20.get((c, s), np.nan) for c, s in
                  zip(l1["code"], l1["signal_date"].dt.strftime("%Y-%m-%d"))]

    mret = np.where(mcnt > 0, msum / np.maximum(mcnt, 1), np.nan)
    midx = np.cumprod(1.0 + np.nan_to_num(mret, nan=0.0))
    mi = pd.Series(midx, index=cal)
    mkt60 = mi / mi.shift(60) - 1.0
    l1["mkt60"] = [mkt60.get(s, np.nan) for s in l1["signal_date"]]
    pd.DataFrame(dict(date=cal, mkt_ret=mret, mkt_idx=midx)).to_csv(
        os.path.join(OUT, "attr_market_proxy.csv"), index=False)

    # ---- 成交 vs 未成交 ----
    rows = []
    for grp, name in ((l1[l1["filled"]], "成交(20仓内)"), (l1[~l1["filled"]], "未成交(被排除)")):
        for h in ("r60q", "r20q"):
            m = metrics(grp[h])
            rows.append(dict(group=name, horizon=h, **m))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "attr_selection_metrics.csv"),
                              index=False, encoding="utf-8")

    a = l1.loc[l1["filled"], "r60q"].to_numpy(float)
    b = l1.loc[~l1["filled"], "r60q"].to_numpy(float)
    lo, hi = boot_diff(a, b)
    diff = dict(mean_diff=float(np.nanmean(a) - np.nanmean(b)),
                ci_lo=lo, ci_hi=hi,
                mean_diff_20=float(np.nanmean(l1.loc[l1["filled"], "r20q"]) -
                                   np.nanmean(l1.loc[~l1["filled"], "r20q"])))
    by_year = []
    for y, g in l1.groupby(l1["signal_date"].dt.year):
        if y < 2005:
            continue
        by_year.append(dict(year=int(y),
                            filled_n=int(g["filled"].sum()),
                            filled_mean=float(g.loc[g["filled"], "r60q"].mean()),
                            unfilled_n=int((~g["filled"]).sum()),
                            unfilled_mean=float(g.loc[~g["filled"], "r60q"].mean())))
    pd.DataFrame(by_year).to_csv(os.path.join(OUT, "attr_selection_by_year.csv"),
                                 index=False, encoding="utf-8")

    # 同一入场日的对照（消除日期效应）
    day_cmp = []
    for d0, g in l1.groupby("entry_date"):
        if g["filled"].any() and (~g["filled"]).any():
            day_cmp.append(float(g.loc[g["filled"], "r60q"].mean() -
                                 g.loc[~g["filled"], "r60q"].mean()))
    day_cmp = np.array(day_cmp)
    matched = dict(n_days=int(day_cmp.size),
                   mean_within_day_diff=float(np.nanmean(day_cmp)) if day_cmp.size else np.nan,
                   share_filled_better=float(np.nanmean(day_cmp > 0)) if day_cmp.size else np.nan)

    # 拥挤度
    cong = l1.groupby("entry_date")["filled"].transform("size")
    l1["cong"] = cong
    buckets = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 999)]
    cong_rows = []
    for lo_, hi_ in buckets:
        g = l1[(l1["cong"] >= lo_) & (l1["cong"] <= hi_)]
        if len(g) == 0:
            continue
        cong_rows.append(dict(bucket=f"{lo_}-{hi_ if hi_ < 999 else '+'}",
                              n=int(len(g)), fill_rate=float(g["filled"].mean()),
                              filled_mean=float(g.loc[g["filled"], "r60q"].mean()),
                              unfilled_mean=float(g.loc[~g["filled"], "r60q"].mean())))
    pd.DataFrame(cong_rows).to_csv(os.path.join(OUT, "attr_selection_by_congestion.csv"),
                                   index=False, encoding="utf-8")

    # 市场阶段
    reg_rows = []
    for lo_, hi_, nm in ((-1.0, -0.10, "市场60日<-10%"), (-0.10, 0.0, "-10%~0"),
                         (0.0, 0.10, "0~+10%"), (0.10, 999.0, "+10%以上")):
        g = l1[(l1["mkt60"] > lo_) & (l1["mkt60"] <= hi_)]
        reg_rows.append(dict(regime=nm, n=int(len(g)),
                             filled_mean=float(g.loc[g["filled"], "r60q"].mean()),
                             unfilled_mean=float(g.loc[~g["filled"], "r60q"].mean())))
    pd.DataFrame(reg_rows).to_csv(os.path.join(OUT, "attr_selection_by_regime.csv"),
                                  index=False, encoding="utf-8")

    # ---- 组合回撤归因 ----
    daily = pd.read_csv(os.path.join(OUT, "l2_daily_nav.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")
    nav = daily["nav"]
    m = pd.DataFrame(dict(nav=nav.resample("ME").last()))
    m.index = m.index.to_period("M")
    m["ret"] = m["nav"].pct_change()
    m["n_open"] = daily["n_open"].resample("ME").mean().set_axis(m.index)
    m["util"] = (daily["invested"] / daily["nav"]).resample("ME").mean().set_axis(m.index)
    m["cash"] = daily["cash"].resample("ME").last().set_axis(m.index)
    l2["pexit"] = l2["exit_date"].dt.to_period("M")
    l2["pentry"] = l2["entry_date"].dt.to_period("M")
    m["pnl_real"] = l2.groupby("pexit")["pnl"].sum().reindex(m.index).fillna(0.0)
    m["entries"] = l2.groupby("pentry").size().reindex(m.index).fillna(0).astype(int)
    m["exits"] = l2.groupby("pexit").size().reindex(m.index).fillna(0).astype(int)
    m["hold_avg"] = l2.groupby("pexit")["hold_bars"].mean().reindex(m.index)
    m["mkt_ret"] = pd.Series(mret, index=cal).resample("ME").apply(
        lambda x: float(np.nanprod(1 + x) - 1) if len(x) else np.nan).set_axis(m.index)
    m["mkt_ret"] = m["mkt_ret"].fillna(0.0)
    m["dd"] = m["nav"] / m["nav"].cummax() - 1.0
    m["nav_delta"] = m["nav"].diff()
    m["pnl_unreal"] = m["nav_delta"] - m["pnl_real"]
    m.to_csv(os.path.join(OUT, "attr_monthly.csv"), encoding="utf-8")

    w0, w1 = pd.Timestamp("2008-01-31"), pd.Timestamp("2012-12-31")
    w = m.loc[pd.Period("2008-01", "M"):pd.Period("2012-12", "M")].copy()
    seg = dict(
        window="2008-01-15 ~ 2012-12-03",
        nav_start=float(nav.loc[:w0].iloc[-1]),
        nav_low=float(nav.loc[w0:w1].min()),
        nav_end=float(nav.loc[w0:w1].iloc[-1]),
        dd=float((nav.loc[w0:w1].min() / nav.loc[:w0].cummax().iloc[-1]) - 1),
        pnl_real=float(w["pnl_real"].sum()),
        pnl_unreal=float(w["pnl_unreal"].sum()),
        n_entries=int(w["entries"].sum()),
        n_exits=int(w["exits"].sum()),
        util_avg=float(w["util"].mean()),
        util_max=float(w["util"].max()),
        n_open_avg=float(w["n_open"].mean()),
        hold_avg=float(np.nanmean(w["hold_avg"])) if w["hold_avg"].notna().any() else np.nan,
        mkt_sum=float(np.nansum(w["mkt_ret"])),
        down_months=int((w["ret"] < 0).sum()), up_months=int((w["ret"] > 0).sum()),
    )
    r = m["ret"]
    corr = float(r.corr(m["mkt_ret"]))
    seg["corr_mkt"] = corr
    seg["ret_in_mkt_down"] = float(r[m["mkt_ret"] < 0].mean())
    seg["ret_in_mkt_up"] = float(r[m["mkt_ret"] > 0].mean())

    tw = l2[(l2["exit_date"] >= w0) & (l2["exit_date"] <= w1)]
    stock = tw.groupby("code")["pnl"].agg(["sum", "count"]).sort_values("sum")
    stock["board"] = [board(c) for c in stock.index]
    stock.to_csv(os.path.join(OUT, "attr_stock_pnl_2008_2012.csv"), encoding="utf-8")
    bd = stock.groupby("board")["sum"].agg(["sum", "count"]).sort_values("sum")
    bd.to_csv(os.path.join(OUT, "attr_board_pnl_2008_2012.csv"), encoding="utf-8")

    cohort = l2.groupby("pentry")["pnl"].sum()
    cohort.to_csv(os.path.join(OUT, "attr_cohort_pnl.csv"), encoding="utf-8")

    pnl_by_exit_year = l2.groupby(l2["exit_date"].dt.year)["pnl"].sum()
    pnl_by_exit_year.to_csv(os.path.join(OUT, "attr_pnl_by_exit_year.csv"), encoding="utf-8")

    allst = l2.groupby("code")["pnl"].agg(["sum", "count"]).sort_values("sum")
    total_pnl = float(l2["pnl"].sum())
    profit = dict(
        total_realized_pnl=total_pnl,
        gross_profit=float(l2.loc[l2["pnl"] > 0, "pnl"].sum()),
        gross_loss=float(l2.loc[l2["pnl"] <= 0, "pnl"].sum()),
        top1_stock=allst.tail(1)["sum"].iloc[0] if len(allst) else np.nan,
        top1_share=float(allst.tail(1)["sum"].iloc[0] / total_pnl) if len(allst) and total_pnl else np.nan,
        top10_share=float(allst.tail(10)["sum"].sum() / total_pnl) if len(allst) and total_pnl else np.nan,
        n_stocks=int(len(allst)),
        n_trades=int(len(l2)),
        by_entry_year={str(k): float(v) for k, v in l2.groupby("pentry")["pnl"].sum().items()},
        by_exit_year={str(k): float(v) for k, v in pnl_by_exit_year.items()},
    )

    # 2015
    w15 = m.loc[pd.Period("2015-06", "M"):pd.Period("2015-10", "M")]
    seg15 = dict(window="2015-06-12 ~ 2015-09-15",
                 peak=float(nav.loc[:"2015-06-30"].max()),
                 low=float(nav.loc["2015-06-01":"2015-10-31"].min()),
                 loss=float(nav.loc[:"2015-06-30"].max() -
                            nav.loc["2015-06-12":"2015-09-30"].min()),
                 pnl_real=float(w15["pnl_real"].sum()),
                 n_entries=int(w15["entries"].sum()), n_exits=int(w15["exits"].sum()),
                 util=float(w15["util"].mean()))
    t15 = l2[(l2["exit_date"] >= "2015-06-01") & (l2["exit_date"] <= "2015-10-31")]
    s15 = t15.groupby("code")["pnl"].agg(["sum", "count"]).sort_values("sum")
    s15.head(10).to_csv(os.path.join(OUT, "attr_stock_pnl_2015.csv"), encoding="utf-8")

    summary = dict(selection=dict(
        filled=int(l1["filled"].sum()), unfilled=int((~l1["filled"]).sum()),
        diff=diff, matched=matched,
        congestion=cong_rows, regime=reg_rows,
    ), drawdown=seg, crash_2015=seg15, profit=profit,
        worst_months=[dict(month=str(k), ret=float(v), mkt=float(m.loc[k, "mkt_ret"]),
                           n_open=float(m.loc[k, "n_open"]), util=float(m.loc[k, "util"]))
                      for k, v in m["ret"].nsmallest(8).items()],
        top_losers_2008_2012=stock.head(10).reset_index().to_dict("records"),
        boards_2008_2012=bd.reset_index().to_dict("records"),
        cohort=dict((str(k), float(v)) for k, v in cohort.items()))
    with open(os.path.join(OUT, "attr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    log("归因完成")
    print(json.dumps(dict(selection=diff, matched=matched, drawdown=seg,
                          crash2015=seg15), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
