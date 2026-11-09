"""TREND-POSTMORTEM-01：月 MA20 + 周 MA60 失败归因（一次性诊断脚本）。

边界
----
- 不改写、不重算、不覆盖 out/step1 下任何冻结结果。
- 不落 qfq/行情/缓存/快照；复权因子仅内存取得；本地 raw 只读。
- 只在 out/postmortem/ 写一份聚合诊断 JSON（不含逐条行情），供报告引用。
- 股票池、区间、观察时点、收益口径与 STEP 1 完全一致。

输出
----
- out/postmortem/postmortem_diagnostics.json
"""
from __future__ import annotations

import argparse
import glob
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

from scripts.step1_trend_validation import (  # noqa: E402
    BAD_RAW, END, FAC_DIR, HORIZONS, OUT_DIR,
    apply_qfq, build_market, clean_raw, is_finite, load_factor,
    log, merge_fresh, read_local_raw,
)
from data.pit_universe import load_stock_codes  # noqa: E402
from data.stock_loader import fetch_sina  # noqa: E402

OUT = os.path.join(ROOT, "out", "postmortem")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
def regime_map() -> dict[int, str]:
    p = os.path.join(OUT_DIR, "regime_annual.csv")
    df = pd.read_csv(p, encoding="utf-8-sig")
    m: dict[int, str] = {}
    for _, row in df.iterrows():
        for y in str(row["year_list"]).split("|"):
            try:
                m[int(y)] = str(row["regime"])
            except ValueError:
                pass
    m[2026] = "range"
    return m


def align_to_cal(series: pd.Series, cal: list[pd.Timestamp]) -> np.ndarray:
    """按市场日历对齐，ffill 停牌缺口，上市前/退市后置 NaN。"""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    a = s.reindex(cal).ffill()
    first, last = s.index.min(), s.index.max()
    a[(a.index < first) | (a.index > last)] = np.nan
    return a.to_numpy(dtype=float)


def roll(a: np.ndarray, win: int, how: str, minp: int) -> np.ndarray:
    s = pd.Series(a)
    r = s.rolling(win, min_periods=minp)
    if how == "std":
        out = r.std()
    elif how == "min":
        out = r.min()
    elif how == "max":
        out = r.max()
    elif how == "mean_sq_neg":
        out = s.where(s < 0).pow(2).rolling(win, min_periods=minp).mean().pow(0.5)
    else:
        raise ValueError(how)
    return out.to_numpy(dtype=float)


def process_stock(code, raw, facs, ctx) -> list[tuple]:
    d = clean_raw(raw)
    if d.empty or facs is None:
        return []
    d = apply_qfq(d, facs)
    d = clean_raw(d)
    if len(d) < 30:
        return []

    cal, cal_pos = ctx["cal"], ctx["cal_pos"]
    m_end, m_pos, mpe = ctx["m_end"], ctx["m_pos"], ctx["month_period_to_end"]
    w_end, wmap = ctx["w_end"], ctx["week_end_by_iso"]
    obs = ctx["obs"]
    bm_open, bm_close, bm_cal = ctx["bm_open"], ctx["bm_close"], ctx["bm_close_cal"]

    s = d.set_index("date").sort_index()
    close_d, open_d = s["close"].astype(float), s["open"].astype(float)
    high_d, low_d = s["high"].astype(float), s["low"].astype(float)

    # ---- 月线 ----
    mm = pd.DataFrame({"c": close_d.values, "last": close_d.index.values},
                      index=close_d.index)
    mm["p"] = mm.index.to_period("M")
    g = mm.groupby("p")
    mclose = g["c"].last()
    mlast = pd.to_datetime(g["last"].last())
    tied = pd.Index([mpe.get(p, pd.NaT) for p in mclose.index])
    ok = ~tied.isna()
    mclose.index = tied
    mlast.index = tied
    mclose = mclose[ok].sort_index()
    mlast = mlast[ok].sort_index()
    # 不完整月：该月最后一条 bar 不在市场月末
    partial = (mlast.to_numpy() != mclose.index.to_numpy()).astype(float)
    mclose = mclose.reindex(m_end)
    partial = pd.Series(partial, index=mlast.index).reindex(m_end)
    n_partial_cum = np.nancumsum(np.nan_to_num(partial.to_numpy()))
    ma20 = mclose.rolling(20, min_periods=20).mean()
    ma20_lag3 = ma20.shift(3)
    b_cond = ((mclose > ma20) & (ma20 > ma20_lag3)).to_numpy(dtype=bool)
    m_ready = (ma20.notna() & ma20_lag3.notna()).to_numpy(dtype=bool)
    mclose_a = mclose.to_numpy(dtype=float)
    ma20_a = ma20.to_numpy(dtype=float)

    trend_age = np.zeros(len(m_end), dtype=np.int32)
    age = 0
    for i in range(len(m_end)):
        age = age + 1 if (m_ready[i] and b_cond[i]) else 0
        trend_age[i] = age

    # ---- 周线 ----
    iso = close_d.index.isocalendar()
    wdf = pd.DataFrame({"c": close_d.values, "iy": iso["year"].values,
                        "iw": iso["week"].values})
    wclose = wdf.groupby(["iy", "iw"])["c"].last()
    wclose.index = pd.Index([wmap.get((int(y), int(w),), pd.NaT) for y, w in wclose.index])
    wclose = wclose[~wclose.index.isna()].sort_index().reindex(w_end)
    ma60 = wclose.rolling(60, min_periods=60).mean()
    ma60_lag4 = ma60.shift(4)
    c_cond = ((wclose > ma60) & (ma60 > ma60_lag4)).to_numpy(dtype=bool)
    w_ready = (ma60.notna() & ma60_lag4.notna()).to_numpy(dtype=bool)

    # ---- 日线对齐与滚动统计 ----
    c_al = align_to_cal(close_d, cal)
    o_al = align_to_cal(open_d, cal)
    h_al = align_to_cal(high_d, cal)
    l_al = align_to_cal(low_d, cal)
    ret = pd.Series(c_al).pct_change().to_numpy(dtype=float)
    vol119 = roll(ret, 119, "std", 20)
    dsd119 = roll(ret, 119, "mean_sq_neg", 10)
    minret119 = roll(ret, 119, "min", 20)
    rmin119 = roll(c_al, 119, "min", 1)
    lmin119 = roll(l_al, 119, "min", 1)
    hi252 = roll(h_al, 252, "max", 60)
    lo252 = roll(l_al, 252, "min", 60)

    has_bar = set(pd.to_datetime(close_d.index).tolist())
    code_int = int(code)
    rows: list[tuple] = []
    n_cal = len(cal)

    for t in obs:
        t = pd.Timestamp(t)
        if t not in has_bar:
            continue
        mi = m_pos.get(t)
        if mi is None:
            continue
        m_rdy = bool(m_ready[mi])
        b_on = bool(b_cond[mi]) if m_rdy else False
        wi = int(np.searchsorted(w_end, t, side="right") - 1)
        w_rdy = bool(w_ready[wi]) if wi >= 0 else False
        c_on = bool(c_cond[wi]) if (wi >= 0 and w_rdy) else False
        b_w = bool(b_on and w_rdy)
        c_flag = bool(b_w and c_on)

        ci = cal_pos.get(t)
        if ci is None or ci + 1 >= n_cal:
            continue
        entry = cal[ci + 1]
        o = open_d.get(entry, np.nan)
        if not is_finite(o) or float(o) <= 0:
            continue
        o = float(o)

        vals: dict[int, tuple[float, float]] = {}
        for h in HORIZONS:
            xi = ci + h - 1
            if xi >= n_cal:
                vals[h] = (np.nan, np.nan)
                continue
            exit_d = cal[xi]
            c = close_d.get(exit_d, np.nan)
            if not is_finite(c):
                seg = close_d.loc[entry:exit_d].dropna()
                if seg.empty:
                    vals[h] = (np.nan, np.nan)
                    continue
                c = float(seg.iloc[-1])
            bo, bc = bm_open.get(entry, np.nan), bm_close.get(exit_d, np.nan)
            br = (float(bc) / float(bo) - 1.0) if (is_finite(bo) and is_finite(bc)
                                                    and float(bo) > 0) else np.nan
            vals[h] = (float(c) / o - 1.0, br)

        def sino(k):
            j = ci - k
            if j < 0:
                return np.nan, np.nan
            return c_al[ci] / c_al[j] - 1.0, bm_cal[ci] / bm_cal[j] - 1.0

        p20, bp20 = sino(20)
        p60, bp60 = sino(60)
        p120, bp120 = sino(120)

        def mret(k):
            j = mi - k
            if j < 0 or not is_finite(mclose_a[j]) or not is_finite(mclose_a[mi]):
                return np.nan
            return mclose_a[mi] / mclose_a[j] - 1.0

        m1, m3, m6, m12 = mret(1), mret(3), mret(6), mret(12)
        dev = (mclose_a[mi] / ma20_a[mi] - 1.0) if is_finite(ma20_a[mi]) else np.nan
        dhi = (c_al[ci] / hi252[ci] - 1.0) if is_finite(hi252[ci]) else np.nan
        dlo = (c_al[ci] / lo252[ci] - 1.0) if is_finite(lo252[ci]) else np.nan
        np20 = int(n_partial_cum[mi] - (n_partial_cum[mi - 19] if mi >= 19 else 0))
        tage = int(trend_age[mi])

        e = ci + 119
        if e < n_cal:
            vol, dsd, mdd, mae = vol119[e], dsd119[e], np.nan, np.nan
            if is_finite(rmin119[e]):
                mdd = rmin119[e] / o - 1.0
            if is_finite(lmin119[e]):
                mae = lmin119[e] / o - 1.0
            tmin = minret119[e]
        else:
            vol = dsd = mdd = mae = tmin = np.nan

        r5, b5 = vals[5]
        r20, b20 = vals[20]
        r60, b60 = vals[60]
        r120, b120 = vals[120]

        rows.append((
            code_int, int(t.strftime("%Y%m%d")), int(t.year),
            int(m_rdy), int(b_on), int(b_w), int(c_flag),
            int(bool(m_rdy and wi >= 0 and w_rdy)),
            r5, r20, r60, r120, b5, b20, b60, b120,
            m1, m3, m6, m12, dev, tage, dhi, dlo, np20,
            p20, p60, p120, bp20, bp60, bp120,
            vol, dsd, mdd, mae, tmin,
        ))
    return rows


COLS = [
    "code", "date_i", "year", "a_m", "b", "b_w", "c", "a_w",
    "r5", "r20", "r60", "r120", "b5", "b20", "b60", "b120",
    "m1", "m3", "m6", "m12", "dev_ma20", "trend_age", "dist_hi252", "dist_lo252",
    "n_partial20",
    "p20", "p60", "p120", "bp20", "bp60", "bp120",
    "vol120", "dsd120", "mdd120", "mae120", "minret120",
]


# ---------------------------------------------------------------------------
def stats(r) -> dict | None:
    r = pd.Series(r).dropna()
    if r.empty:
        return None
    return {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "positive_rate": float((r > 0).mean()),
        "p10": float(r.quantile(0.10)),
        "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)),
        "p90": float(r.quantile(0.90)),
    }


def nw_t(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    T = len(x)
    mu = float(x.mean())
    e = x - mu
    lrv = float((e * e).sum() / T)
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        gl = float((e[l:] * e[:-l]).sum() / T)
        lrv += 2.0 * w * gl
    se = math.sqrt(max(lrv, 0.0) / T)
    return mu / se if se > 0 else float("nan")


def block_ci(x: np.ndarray, block: int, reps: int = 2000, seed: int = 11) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    T = len(x)
    nb = T // block
    if nb < 3:
        return float("nan"), float("nan")
    arr = x[: nb * block].reshape(nb, block)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, nb, nb)
        out[i] = arr[idx].mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--factor-workers", type=int, default=20)
    ap.add_argument("--raw-workers", type=int, default=12)
    ap.add_argument("--stock-workers", type=int, default=8)
    args = ap.parse_args()
    t_all = time.time()

    codes = load_stock_codes()
    if args.limit:
        codes = codes[: args.limit]
    log(f"股票池 {len(codes)} 只")

    cal, cal_pos, month_period_to_end, bm_open, bm_close, week_end_by_iso, _, m_end, w_end, obs = build_market()
    bm_cal = pd.Series(bm_close).reindex(cal).to_numpy(dtype=float)
    ctx = {
        "cal": cal, "cal_pos": cal_pos, "m_end": m_end,
        "m_pos": {d: i for i, d in enumerate(m_end)},
        "month_period_to_end": month_period_to_end,
        "w_end": w_end, "week_end_by_iso": week_end_by_iso,
        "obs": obs, "bm_open": bm_open, "bm_close": bm_close, "bm_close_cal": bm_cal,
    }
    log(f"日历 {len(cal)} 日；观察 {len(obs)} 月")

    log("加载复权因子（内存，不落盘）...")
    facs: dict[str, list[dict] | None] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.factor_workers) as ex:
        futs = {ex.submit(load_factor, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                facs[c] = fut.result()
            except Exception:  # noqa: BLE001
                facs[c] = None
            done += 1
            if done % 500 == 0 or done == len(codes):
                log(f"  因子 {done}/{len(codes)} fail={sum(1 for v in facs.values() if v is None)} {time.time()-t0:.0f}s")

    log("读取本地 raw，陈旧/坏文件内存刷新 ...")
    raw: dict[str, pd.DataFrame] = {}
    stale: list[str] = []
    for c in codes:
        df = read_local_raw(c)
        raw[c] = df
        if df is None or df.empty:
            stale.append(c)
            continue
        last = pd.to_datetime(df["date"]).max()
        if pd.Timestamp("2026-09-20") <= last <= pd.Timestamp("2026-09-29") or c in BAD_RAW:
            stale.append(c)
    stale = sorted(set(stale))
    log(f"需刷新 {len(stale)} 只")
    if stale:
        with ThreadPoolExecutor(max_workers=args.raw_workers) as ex:
            futs = {ex.submit(fetch_sina, c): c for c in stale}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    f = fut.result()
                except Exception:  # noqa: BLE001
                    f = pd.DataFrame()
                raw[c] = merge_fresh(raw.get(c, pd.DataFrame()), f)

    log(f"逐股票计算（并发 {args.stock_workers}）...")
    rows: list[tuple] = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.stock_workers) as ex:
        futs = {ex.submit(process_stock, c, raw[c], facs.get(c), ctx): c for c in codes}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                rows.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                log(f"  [warn] {c}: {type(e).__name__} {e}")
            done += 1
            if done % 500 == 0 or done == len(codes):
                log(f"  股票 {done}/{len(codes)} rows={len(rows)} {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows, columns=COLS)
    df["date"] = pd.to_datetime(df["date_i"].astype(str), format="%Y%m%d")
    df["a_all"] = True
    for g in ["a_m", "b", "b_w", "c", "a_w"]:
        df[g] = df[g].astype(bool)
    reg = regime_map()
    df["regime"] = df["year"].map(reg).fillna("unknown")
    log(f"总行 {len(df):,} / 股票 {df['code'].nunique()} / 月 {df['date'].nunique()}")

    diag: dict = {
        "meta": {
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "scripts/postmortem_trend.py",
            "rows": int(len(df)),
            "stocks": int(df["code"].nunique()),
            "months": int(df["date"].nunique()),
            "period": [str(cal[0].date()), str(cal[-1].date())],
            "obs_period": [str(pd.Timestamp(obs[0]).date()), str(pd.Timestamp(obs[-1]).date())],
        }
    }

    # ---- 0. 复现校验 ----
    frozen = json.load(open(os.path.join(OUT_DIR, "step1_summary.json"), encoding="utf-8"))
    rep = {}
    for g in ["a_m", "b", "b_w", "c"]:
        for h in HORIZONS:
            sub = df[df[g] & df[f"r{h}"].notna()]
            if sub.empty:
                continue
            fz = frozen["group_horizon"][g][str(h)]
            rep[f"{g}_{h}"] = {
                "n_now": int(len(sub), ),
                "n_frozen": fz["n"],
                "mean_now": float(sub[f"r{h}"].mean()),
                "mean_frozen": fz["mean"],
                "monthly_eq_now": float(sub.groupby("date")[f"r{h}"].mean().mean()),
                "monthly_eq_frozen": fz["monthly_equal_weight_mean"],
            }
    diag["reproduce"] = rep

    # ---- 0b. 口径可信度：不完整月 K ----
    b = df[df["b"] & df["r60"].notna()]
    diag["credibility"] = {
        "b_obs": int(len(b)),
        "n_partial20_dist": {str(k): int(v) for k, v in b["n_partial20"].value_counts().sort_index().items()},
        "share_any_partial": float((b["n_partial20"] > 0).mean()),
        "share_ge3_partial": float((b["n_partial20"] >= 3).mean()),
        "month_end_bar_share": float((b["n_partial20"] == 0).mean()),
        "same_obs_for_all_groups": True,
    }

    # ---- Q1. 信号前涨幅 / 趋势年龄 ----
    q1: dict = {"b_prior": {}, "a_m_prior": {}, "bucket_m12": [], "quintile_m12": [],
                "bucket_age": [], "spearman": {}, "dist": {}}
    for nm, col in [("m1", "m1"), ("m3", "m3"), ("m6", "m6"), ("m12", "m12"),
                    ("dev_ma20", "dev_ma20"), ("dist_hi252", "dist_hi252"),
                    ("dist_lo252", "dist_lo252"), ("trend_age", "trend_age")]:
        q1["b_prior"][nm] = stats(df.loc[df["b"], col])
        q1["a_m_prior"][nm] = stats(df.loc[df["a_m"], col])

    bt = df[df["b"]].copy()
    am = df[df["a_m"]].copy()

    m12_edges = [-np.inf, 0, 0.2, 0.5, 1.0, np.inf]
    m12_labels = ["<0%", "0-20%", "20-50%", "50-100%", ">100%"]
    for src, tag in [(bt, "b"), (am, "a_m")]:
        src = src.copy()
        src["bucket"] = pd.cut(src["m12"], m12_edges, labels=m12_labels)
        for lab in m12_labels:
            sub = src[src["bucket"] == lab]
            if sub.empty:
                continue
            q1["bucket_m12"].append({
                "group": tag, "bucket": lab, "n": int(len(sub)),
                "share": float(len(sub) / len(src)),
                "r20": stats(sub["r20"]), "r60": stats(sub["r60"]), "r120": stats(sub["r120"]),
            })

    q1["share_m12_gt50"] = float((bt["m12"] > 0.5).mean())
    q1["share_m12_gt100"] = float((bt["m12"] > 1.0).mean())

    # 五等分（仅 B）
    bq = bt[bt["m12"].notna()].copy()
    if len(bq):
        bq["q"] = pd.qcut(bq["m12"], 5, labels=False, duplicates="drop")
        for q, sub in bq.groupby("q"):
            q1["quintile_m12"].append({
                "q": int(q), "n": int(len(sub)),
                "m12_low": float(sub["m12"].min()), "m12_high": float(sub["m12"].max()),
                "m12_mean": float(sub["m12"].mean()),
                "r60": stats(sub["r60"]), "r120": stats(sub["r120"]),
            })

    age_edges = [0, 2, 6, 12, 24, 10**9]
    age_labels = ["1-2", "3-6", "7-12", "13-24", "25+"]
    bt["age_b"] = pd.cut(bt["trend_age"], age_edges, labels=age_labels)
    for lab in age_labels:
        sub = bt[bt["age_b"] == lab]
        if sub.empty:
            continue
        q1["bucket_age"].append({
            "bucket": lab, "n": int(len(sub)), "share": float(len(sub) / len(bt)),
            "r20": stats(sub["r20"]), "r60": stats(sub["r60"]), "r120": stats(sub["r120"]),
        })

    for x, y in [("m12", "r20"), ("m12", "r60"), ("m12", "r120"),
                 ("m6", "r60"), ("m6", "r120"), ("m3", "r20"),
                 ("trend_age", "r60"), ("trend_age", "r120")]:
        sub = bt[[x, y]].dropna()
        q1["spearman"][f"{x}~{y}"] = float(sub.corr(method="spearman").iloc[0, 1]) if len(sub) > 10 else None

    q1["dist"] = {
        "m12_gt20_share": float((bt["m12"] > 0.2).mean()),
        "m12_gt50_share": float((bt["m12"] > 0.5).mean()),
        "near_high_share": float((bt["dist_hi252"] > -0.10).mean()),
        "dev_ma20_median": float(bt["dev_ma20"].median()),
        "trend_age_median": float(bt["trend_age"].median()),
        "trend_age_mean": float(bt["trend_age"].mean()),
    }
    diag["q1"] = q1

    # ---- Q2. 绝对趋势 vs 相对强度 ----
    bt["prior60ex"] = bt["p60"] - bt["bp60"]
    bt["prior120ex"] = bt["p120"] - bt["bp120"]
    am["prior60ex"] = am["p60"] - am["bp60"]
    am["prior120ex"] = am["p120"] - am["bp120"]
    q2: dict = {}
    for col in ["prior60ex", "prior120ex"]:
        q2[col] = {
            "b_under_share": float((bt[col] < 0).mean()),
            "b_mean": float(bt[col].mean()),
            "a_m_under_share": float((am[col] < 0).mean()),
            "a_m_mean": float(am[col].mean()),
            "n": int(bt[col].notna().sum()),
        }
        q2[f"future_by_{col}"] = []
        for lab, sub in bt.assign(sign=np.where(bt[col] < 0, "under", "over")).groupby("sign"):
            q2[f"future_by_{col}"].append({
                "sign": lab, "n": int(len(sub)),
                "r20": stats(sub["r20"]), "r60": stats(sub["r60"]), "r120": stats(sub["r120"]),
            })
    # 前 60 日超额五分位
    bq2 = bt[bt["prior60ex"].notna()].copy()
    if len(bq2):
        bq2["q"] = pd.qcut(bq2["prior60ex"], 5, labels=False, duplicates="drop")
        q2["quintile_prior60ex"] = []
        for q, sub in bq2.groupby("q"):
            q2["quintile_prior60ex"].append({
                "q": int(q), "n": int(len(sub)),
                "prior60ex_mean": float(sub["prior60ex"].mean()),
                "r60": stats(sub["r60"]), "r120": stats(sub["r120"]),
            })
    q2["sector_mapping"] = "unavailable (no historical industry mapping; not built by design)"
    diag["q2"] = q2

    # ---- Q3. 市场环境 ----
    q3 = []
    for rg in ["bull", "bear", "range", "unknown"]:
        for g in ["a_m", "b", "c"]:
            for h in HORIZONS:
                sub = df[df[g] & df[f"r{h}"].notna() & (df["regime"] == rg)]
                if sub.empty:
                    continue
                q3.append({
                    "regime": rg, "group": g, "horizon": h, "n": int(len(sub)),
                    "mean": float(sub[f"r{h}"].mean()),
                    "median": float(sub[f"r{h}"].median()),
                    "positive_rate": float((sub[f"r{h}"] > 0).mean()),
                })
    diag["q3"] = q3

    # 环境 x 波动/回撤（Q5 用）
    q5 = []
    for rg in ["bull", "bear", "range", "unknown", "ALL"]:
        for g in ["a_m", "b", "c"]:
            sub = df[df[g] & df["r120"].notna() & df["mdd120"].notna()]
            if rg != "ALL":
                sub = sub[sub["regime"] == rg]
            if sub.empty:
                continue
            q5.append({
                "regime": rg, "group": g, "n": int(len(sub)),
                "vol120": float(sub["vol120"].mean()),
                "dsd120": float(sub["dsd120"].mean()),
                "mdd120_mean": float(sub["mdd120"].mean()),
                "mdd120_median": float(sub["mdd120"].median()),
                "mae120_mean": float(sub["mae120"].mean()),
                "minret120_p5": float(sub["minret120"].quantile(0.05)),
                "share_mdd_lt_-20": float((sub["mdd120"] < -0.20).mean()),
                "share_mdd_lt_-30": float((sub["mdd120"] < -0.30).mean()),
                "r120_mean": float(sub["r120"].mean()),
            })
    diag["q5_risk"] = q5

    # ---- Q4. 集中度 / 月份权重 ----
    q4: dict = {}
    b60 = df[df["b"] & df["r60"].notna()]
    a60 = df[df["a_m"] & df["r60"].notna()]
    nb = b60.groupby("date")["r60"].agg(["size", "mean"])
    na = a60.groupby("date")["r60"].agg(["size", "mean"])
    nb.columns = ["n_b", "m_b"]
    na.columns = ["n_a", "m_a"]
    j = nb.join(na, how="inner")
    j["w_b"] = j["n_b"] / j["n_b"].sum()
    j["w_a"] = j["n_a"] / j["n_a"].sum()
    j["contrib"] = j["w_b"] * j["m_b"] - j["w_a"] * j["m_a"]
    j["b_share"] = j["n_b"] / j["n_a"]
    # 基准未来 60 日收益（月末次日开盘到第 60 日收盘）
    fwd = []
    for t in j.index:
        ci = cal_pos.get(pd.Timestamp(t), None)
        if ci is None or ci + 59 >= len(cal):
            fwd.append(np.nan)
            continue
        bo = bm_open.get(cal[ci + 1], np.nan)
        bc = bm_close.get(cal[ci + 59], np.nan)
        fwd.append(float(bc) / float(bo) - 1 if is_finite(bo) and is_finite(bc) else np.nan)
    j["bm_fwd60"] = fwd
    q4["pooled_b"] = float((j["w_b"] * j["m_b"]).sum())
    q4["pooled_a"] = float((j["w_a"] * j["m_a"]).sum())
    q4["pooled_diff"] = q4["pooled_b"] - q4["pooled_a"]
    q4["equal_weight_diff"] = float((j["m_b"] - j["m_a"]).mean())
    q4["contrib_sum"] = float(j["contrib"].sum())
    q4["top_pos_months"] = [
        {"month": str(d.date()), "n_b": int(r.n_b), "b_share": float(r.b_share),
         "m_b": float(r.m_b), "m_a": float(r.m_a), "contrib": float(r.contrib)}
        for d, r in j.sort_values("contrib", ascending=False).head(5).iterrows()
    ]
    q4["top_neg_months"] = [
        {"month": str(d.date()), "n_b": int(r.n_b), "b_share": float(r.b_share),
         "m_b": float(r.m_b), "m_a": float(r.m_a), "contrib": float(r.contrib)}
        for d, r in j.sort_values("contrib", ascending=True).head(5).iterrows()
    ]
    q4["n_months"] = int(len(j))
    q4["corr_b_share_vs_fwd60"] = float(j[["b_share", "bm_fwd60"]].dropna().corr().iloc[0, 1])
    q4["corr_b_share_vs_m_b"] = float(j[["b_share", "m_b"]].dropna().corr().iloc[0, 1])
    q4["b_share_mean"] = float(j["b_share"].mean())
    q4["b_share_max"] = float(j["b_share"].max())
    q4["b_share_min"] = float(j["b_share"].min())

    by = b60.groupby("code")["r60"].sum()
    pos = by[by > 0]
    neg = by[by < 0]
    ks = [1, 5, 10]
    q4["stock_share_positive"] = {f"top{k}%": float(pos.sort_values(ascending=False).head(max(1, int(round(len(by) * k / 100)))).sum() / pos.sum()) for k in ks}
    q4["stock_share_negative"] = {f"bottom{k}%": float(neg.sort_values().head(max(1, int(round(len(by) * k / 100)))).sum() / neg.sum()) for k in ks}
    top10 = by.sort_values(ascending=False).head(10)
    q4["top10_stocks"] = [{"code": int(c), "sum_r60": float(v)} for c, v in top10.items()]

    # 剔除前 10 个贡献股票后的月度等权差
    keep = df[~df["code"].isin(top10.index)]
    kb = keep[keep["b"] & keep["r60"].notna()].groupby("date")["r60"].mean()
    ka = keep[keep["a_m"] & keep["r60"].notna()].groupby("date")["r60"].mean()
    dd = (kb - ka).dropna()
    q4["ew_diff_ex_top10"] = float(dd.mean())
    q4["ew_n_ex_top10"] = int(len(dd))
    diag["q4"] = q4

    # ---- Q6. t 值与统计口径 ----
    q6: dict = {"paired": [], "bootstrap": []}
    for g1, g2 in [("b", "a_m"), ("c", "b_w")]:
        for h in HORIZONS:
            s1 = df[df[g1] & df[f"r{h}"].notna()].groupby("date")[f"r{h}"].mean()
            s2 = df[df[g2] & df[f"r{h}"].notna()].groupby("date")[f"r{h}"].mean()
            diff = (s1 - s2).dropna()
            if len(diff) < 3:
                continue
            x = diff.to_numpy(dtype=float)
            lag = max(1, int(round(h / 20)))  # 20 交易日 ≈ 1 月
            q6["paired"].append({
                "a": g1, "b": g2, "horizon": h, "months": int(len(x)),
                "mean_diff": float(x.mean()),
                "t_raw": float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))),
                "t_nw_lag1": nw_t(x, 1), "t_nw_lag3": nw_t(x, 3), "t_nw_lag6": nw_t(x, 6),
                "t_nw_overlap": nw_t(x, lag),
            })
            for blk in (6, 12):
                lo, hi = block_ci(x, blk, 2000)
                q6["bootstrap"].append({
                    "a": g1, "b": g2, "horizon": h, "block_months": blk,
                    "mean_diff": float(x.mean()), "ci_low": lo, "ci_high": hi,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                })
    diag["q6"] = q6

    # ---- Q7. 周线增量 ----
    q7: dict = {}
    nb_all = int((df["b"] & df["r60"].notna()).sum())
    nbw = int((df["b_w"] & df["r60"].notna()).sum())
    nc = int((df["c"] & df["r60"].notna()).sum())
    q7["overlap"] = {
        "b_w_over_b": nbw / nb_all if nb_all else None,
        "c_over_b_w": nc / nbw if nbw else None,
        "c_over_b": nc / nb_all if nb_all else None,
    }
    dropped = df[df["b_w"] & ~df["c"] & df["r60"].notna()]
    q7["dropped_vs_c"] = {
        "dropped_n": int(len(dropped)),
        "dropped_r20": stats(dropped["r20"]), "dropped_r60": stats(dropped["r60"]),
        "dropped_r120": stats(dropped["r120"]),
        "c_r60": stats(df.loc[df["c"], "r60"]), "c_r120": stats(df.loc[df["c"], "r120"]),
        "dropped_m12": float(dropped["m12"].mean()), "c_m12": float(df.loc[df["c"], "m12"].mean()),
        "dropped_age": float(dropped["trend_age"].mean()), "c_age": float(df.loc[df["c"], "trend_age"].mean()),
    }
    # 周线删掉的股票未来是否更差：月度配对
    for h in [20, 60, 120]:
        sc = df[df["c"] & df[f"r{h}"].notna()].groupby("date")[f"r{h}"].mean()
        sd = df[df["b_w"] & ~df["c"] & df[f"r{h}"].notna()].groupby("date")[f"r{h}"].mean()
        diff = (sc - sd).dropna()
        q7[f"c_minus_dropped_{h}"] = {
            "months": int(len(diff)), "mean": float(diff.mean()),
            "t_raw": float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))),
        }
    diag["q7"] = q7

    # 冻结对照（避免与冻结数字出入时误读）
    diag["frozen_compare"] = {
        "b_minus_a_m_t": {str(x["horizon"]): x["t_stat"] for x in frozen["paired"] if x["group_a"] == "b" and x["group_b"] == "a_m"},
        "c_minus_b_w_t": {str(x["horizon"]): x["t_stat"] for x in frozen["paired"] if x["group_a"] == "c" and x["group_b"] == "b_w"},
    }

    p = os.path.join(OUT, "postmortem_diagnostics.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=1, default=float)
    log(f"WROTE {p} elapsed={time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
