# -*- coding: utf-8 -*-
"""UPTREND-DETECTOR-01：A 股上涨趋势识别与阶段延续性研究。

只做识别与阶段统计，不做买卖策略、不做参数搜索、不建设数据基础设施。
- 复用 STEP 1 的复权 / 清洗 / 月周聚合 / 市场日历口径；
- 月线方向 + 周线结构（2 左 2 右波段，后两根确认）；
- 首次识别日 = 条件第一次同时满足的完整周末收盘；
- 阶段终止 = 完整周收盘跌破最近一个已确认周线波段低点；
- MFE/MAE 以信号后下一交易日开盘为起点，跟踪到结构破坏；
- 同期对照组 = 同一信号日、可评估但未满足条件的 PIT 股票。

产出只写 out/uptrend/ 下的聚合统计。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.stock_loader import fetch_sina  # noqa: E402
from data.pit_universe import load_stock_codes  # noqa: E402
from scripts.step1_trend_validation import (  # noqa: E402
    BAD_RAW, END, FAC_DIR, apply_qfq, build_market, clean_raw,
    load_factor, log, merge_fresh, read_local_raw,
)

OUT_DIR = os.path.join(ROOT, "out", "uptrend")
FAC_CACHE: dict[str, list[dict] | None] = {}
CTX: dict = {}

MFE_EDGES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
MAE_EDGES = [-0.10, -0.20, -0.30]
STRENGTH_WIN = 26          # 周线回归窗口（与月线 6 个月中枢同尺度）
STRENGTH_MIN_PTS = 20      # 窗口内至少 20 个有效收盘（容忍停牌）
STRENGTH_T = 2.0           # 斜率 t 值门槛（常规 5% 显著性）
ER_WIN = 26                # 效率比窗口
FIXED_WINDOWS = (20, 60, 120)
FP_UP = (0.10, 0.20, 0.30)  # 阶段内首次触及的上行阈值
FP_DN = (0.10, 0.20)        # 阶段内首次触及的下行阈值
EVENT_FLOATS = ("mfe", "mae", "dur_days", "dur_weeks", "days_first_high",
                "days_max_high", "ma20_up_months", "slope_t", "er", "mdd",
                "r20", "r60", "r120", "d_up10", "d_dn10", "d_up20", "d_up30",
                "d_dn20", "lag_days", "ret_end", "runup_26")
MA_KEYS = ["m_ma5_above", "m_ma5_up", "m_ma10_above", "m_ma10_up",
           "m_ma20_above", "m_ma20_up", "w_ma20_above", "w_ma20_up",
           "w_ma60_above", "w_ma60_up"]


# ---------------------------------------------------------------------------
def dt64(x) -> np.datetime64:
    return np.datetime64(pd.Timestamp(x).to_datetime64())


def pivots(a: np.ndarray, kind: str, left: int = 2, right: int = 2):
    """固定 2 左 2 右局部极值。

    低点：严格低于左两根，且不高于右两根（等号归后一根，确定性）。
    高点：严格高于左两根，且不低于右两根。
    返回 (确认索引 = 极值索引 + right, 极值)。
    """
    n = a.size
    if n < left + right + 1:
        return np.empty(0, np.int64), np.empty(0, float)
    ok = np.ones(n - left - right, dtype=bool)
    for d in range(-left, right + 1):
        ok &= np.isfinite(a[left + d: n - right + d])
    c = a[left: n - right]
    if kind == "low":
        ok &= c < a[left - 1: n - right - 1]
        ok &= c < a[left - 2: n - right - 2]
        ok &= c <= a[left + 1: n - right + 1]
        ok &= c <= a[left + 2: n - right + 2]
    else:
        ok &= c > a[left - 1: n - right - 1]
        ok &= c > a[left - 2: n - right - 2]
        ok &= c >= a[left + 1: n - right + 1]
        ok &= c >= a[left + 2: n - right + 2]
    idx = np.nonzero(ok)[0]
    return idx + right, c[idx]


def running_two(conf_idx: np.ndarray, vals: np.ndarray, n: int):
    """截至每个 bar 已确认的最近两个极值（s1 最近，s2 前一个）。"""
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    src = np.full(n, -1, dtype=np.int64)
    if conf_idx.size:
        pos = np.searchsorted(conf_idx, np.arange(n), side="right") - 1
        m1 = pos >= 0
        s1[m1] = vals[pos[m1]]
        src[m1] = conf_idx[pos[m1]]
        m2 = pos >= 1
        s2[m2] = vals[pos[m2] - 1]
    return s1, s2, src


def rolling_ols_t(logc: np.ndarray, n: int, min_pts: int = 20):
    """周线对数收盘对时间回归：返回 (斜率 t 值, 斜率)，逐周、只用窗口内已完成数据。"""
    L = logc.size
    t_out = np.full(L, np.nan)
    slope_out = np.full(L, np.nan)
    if L < n:
        return t_out, slope_out
    m = np.isfinite(logc)
    x = np.arange(L, dtype=float)

    def cs(a: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(a)))

    s0 = cs(m.astype(float))
    sx = cs(np.where(m, x, 0.0))
    sxx = cs(np.where(m, x * x, 0.0))
    sy = cs(np.where(m, logc, 0.0))
    sxy = cs(np.where(m, x * logc, 0.0))
    syy = cs(np.where(m, logc * logc, 0.0))

    k = np.arange(n - 1, L)
    hi = k + 1
    lo = k - n + 1
    N = s0[hi] - s0[lo]
    SX = sx[hi] - sx[lo]
    SXX = sxx[hi] - sxx[lo]
    SY = sy[hi] - sy[lo]
    SXY = sxy[hi] - sxy[lo]
    SYY = syy[hi] - syy[lo]
    den = N * SXX - SX * SX
    ok = (N >= min_pts) & (N > 2) & (den > 0)
    b = np.full(k.size, np.nan)
    a = np.full(k.size, np.nan)
    b[ok] = (N[ok] * SXY[ok] - SX[ok] * SY[ok]) / den[ok]
    a[ok] = (SY[ok] - b[ok] * SX[ok]) / N[ok]
    sse = SYY - a * SY - b * SXY
    s2 = np.maximum(sse, 0.0) / (N - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_b = np.sqrt(s2 / (den / N))
        tb = b / se_b
    good = ok & np.isfinite(tb) & (se_b > 0)
    t_out[k[good]] = tb[good]
    slope_out[k[good]] = b[good]
    return t_out, slope_out


def efficiency_ratio(c: np.ndarray, n: int) -> np.ndarray:
    """Kaufman 效率比：|c[k]-c[k-n+1]| / Σ|Δc|，窗口 n 个周收盘。"""
    L = c.size
    out = np.full(L, np.nan)
    if L < n:
        return out
    d = np.abs(np.diff(c))
    cd = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(d), d, 0.0))))
    cn = np.concatenate(([0.0], np.cumsum(np.isfinite(d).astype(float))))
    k = np.arange(n - 1, L)
    lo = k - n + 1
    net = np.abs(c[k] - c[lo])
    path = cd[k] - cd[lo]
    cnt = cn[k] - cn[lo]
    ok = (cnt == n - 1) & np.isfinite(net) & (path > 0)
    out[k[ok]] = net[ok] / path[ok]
    return out


# ---------------------------------------------------------------------------
def prepare_stock(raw: pd.DataFrame, facs: list[dict] | None) -> dict | None:
    """日线 → 周/月对齐数组 + 波段 + 条件，全部为当时可确认信息。"""
    if facs is None:
        return None
    d = clean_raw(raw)
    if d.empty:
        return None
    d = clean_raw(apply_qfq(d, facs))
    d = d[d["date"] >= CTX["cal0"]]
    if len(d) < 150:
        return None

    dates = d["date"].to_numpy()
    o = d["open"].to_numpy(dtype=float)
    hi = d["high"].to_numpy(dtype=float)
    lo = d["low"].to_numpy(dtype=float)
    cl = d["close"].to_numpy(dtype=float)
    amt = d["amount"].to_numpy(dtype=float) if "amount" in d.columns else np.full(len(d), np.nan)

    w_end = CTX["w_end"]
    nw = len(w_end)
    w_end_np = CTX["w_end_np"]

    # ---- 周线聚合（自然 ISO 周，最后一个交易日收盘）----
    iso = pd.DatetimeIndex(dates).isocalendar()
    wkey = iso["year"].to_numpy().astype(np.int64) * 100 + iso["week"].to_numpy().astype(np.int64)
    wdf = pd.DataFrame({"key": wkey, "date": dates, "h": hi, "l": lo, "c": cl, "amt": amt})
    g = wdf.groupby("key", sort=True)
    keys = g.size().index.to_numpy()
    w_last = g["date"].last().to_numpy()
    w_c = g["c"].last().to_numpy()
    w_h = g["h"].max().to_numpy()
    w_l = g["l"].min().to_numpy()
    w_amt = g["amt"].sum(min_count=1).to_numpy()
    w_n = g["c"].size().to_numpy()

    wpos = np.array([CTX["wkey_to_pos"].get(int(k), -1) for k in keys], dtype=np.int64)
    keepw = wpos >= 0
    week_orphan = int((~keepw).sum())

    wclose = np.full(nw, np.nan)
    whigh = np.full(nw, np.nan)
    wlow = np.full(nw, np.nan)
    wamount = np.full(nw, np.nan)
    wdays = np.full(nw, np.nan)
    wlast = np.full(nw, np.datetime64("NaT", "ns"))
    p = wpos[keepw]
    wclose[p] = w_c[keepw]
    whigh[p] = w_h[keepw]
    wlow[p] = w_l[keepw]
    wamount[p] = w_amt[keepw]
    wdays[p] = w_n[keepw]
    wlast[p] = w_last[keepw]

    has_bar = (wlast == w_end_np)

    # ---- 月线聚合 ----
    per = pd.DatetimeIndex(dates).to_period("M")
    mdf = pd.DataFrame({"p": per, "date": dates, "c": cl, "h": hi, "l": lo})
    gm = mdf.groupby("p", sort=True)
    mperiod = gm.size().index
    m_c = gm["c"].last().to_numpy()
    m_h = gm["h"].max().to_numpy()
    m_l = gm["l"].min().to_numpy()
    mpos = np.array([
        CTX["mpos"].get(CTX["month_period_to_end"].get(pp), -1) for pp in mperiod
    ], dtype=np.int64)
    keepm = mpos >= 0
    nm = len(CTX["m_end"])
    mclose = np.full(nm, np.nan)
    mhigh = np.full(nm, np.nan)
    mlow = np.full(nm, np.nan)
    mclose[mpos[keepm]] = m_c[keepm]
    mhigh[mpos[keepm]] = m_h[keepm]
    mlow[mpos[keepm]] = m_l[keepm]

    # ---- 月线方向 ----
    ms = pd.Series(mclose)
    ma20 = ms.rolling(20, min_periods=20).mean().to_numpy()
    ma20_lag3 = pd.Series(ma20).shift(3).to_numpy()
    m6 = ms.rolling(6, min_periods=6).mean().to_numpy()
    m6_lag6 = pd.Series(m6).shift(6).to_numpy()
    ml1, ml2, ml_src = running_two(*pivots(mlow, "low"), nm)

    up_m = np.zeros(nm, dtype=bool)
    v = np.isfinite(ma20) & np.isfinite(ma20_lag3)
    up_m[v] = ma20[v] > ma20_lag3[v]

    run = np.zeros(nm, dtype=np.int32)
    r = 0
    for i in range(nm):
        r = r + 1 if up_m[i] else 0
        run[i] = r

    m_ready = np.isfinite(m6) & np.isfinite(m6_lag6) & np.isfinite(ml1) & np.isfinite(ml2)
    m_cond = m_ready & (ml1 > ml2) & (m6 > m6_lag6)

    # ---- 周线结构 ----
    ws = pd.Series(wclose)
    ma60 = ws.rolling(60, min_periods=60).mean().to_numpy()
    ma60_lag4 = pd.Series(ma60).shift(4).to_numpy()
    wh1, wh2, wh_src = running_two(*pivots(whigh, "high"), nw)
    wl1, wl2, wl_src = running_two(*pivots(wlow, "low"), nw)
    hp_idx, hp_val = pivots(whigh, "high")

    w_ready = np.isfinite(wh1) & np.isfinite(wh2) & np.isfinite(wl1) & np.isfinite(wl2)
    # 识别时结构必须仍然完好（周收盘不低于最近已确认波段低点），否则“趋势”已在终点
    w_cond = (np.isfinite(wclose) & (wh1 > wh2) & (wl1 > wl2)
              & np.isfinite(wl1) & (wclose >= wl1))

    # 月线条件映射到每周（使用已完成的最近一个月）
    mw = np.searchsorted(CTX["m_end_np"], w_end_np, side="right") - 1
    turn_from = np.arange(nm) - run + 1
    m_turn = np.full(nm, np.datetime64("NaT", "ns"))
    has_run = run > 0
    m_turn[has_run] = CTX["m_end_np"][turn_from[has_run]]

    m_ok_w = np.zeros(nw, dtype=bool)
    m_rdy_w = np.zeros(nw, dtype=bool)
    m_up_w = np.zeros(nw, dtype=np.int32)
    m_turn_w = np.full(nw, np.datetime64("NaT", "ns"))
    vw = mw >= 0
    m_ok_w[vw] = m_cond[mw[vw]]
    m_rdy_w[vw] = m_ready[mw[vw]]
    m_up_w[vw] = run[mw[vw]]
    m_turn_w[vw] = m_turn[mw[vw]]

    # ---- 均线状态：只作独立维度记录，不进入识别条件 ----
    ma5 = ms.rolling(5, min_periods=5).mean().to_numpy()
    ma10 = ms.rolling(10, min_periods=10).mean().to_numpy()
    wma20 = ws.rolling(20, min_periods=20).mean().to_numpy()

    def to_week(c):
        o = np.zeros(nw, dtype=bool)
        o[vw] = c[mw[vw]]
        return o

    ma_week = {
        "m_ma5_above": to_week(np.isfinite(ma5) & (mclose > ma5)),
        "m_ma5_up": to_week(np.isfinite(ma5) & (ma5 > pd.Series(ma5).shift(1).to_numpy())),
        "m_ma10_above": to_week(np.isfinite(ma10) & (mclose > ma10)),
        "m_ma10_up": to_week(np.isfinite(ma10) & (ma10 > pd.Series(ma10).shift(1).to_numpy())),
        "m_ma20_above": to_week(np.isfinite(ma20) & (mclose > ma20)),
        "m_ma20_up": to_week(up_m),
        "w_ma20_above": np.isfinite(wma20) & (wclose > wma20),
        "w_ma20_up": np.isfinite(wma20) & (wma20 > pd.Series(wma20).shift(1).to_numpy()),
        "w_ma60_above": np.isfinite(ma60) & (wclose > ma60),
        "w_ma60_up": np.isfinite(ma60) & (ma60 > pd.Series(ma60).shift(1).to_numpy()),
    }

    ready_struct = m_rdy_w & w_ready & has_bar & (w_end_np >= dt64(CTX["start"]))
    cond_struct_week = m_ok_w & w_cond & has_bar & (w_end_np >= dt64(CTX["start"]))

    # ---- 强度（周线 26 周对数收盘回归 t）与路径效率（Kaufman ER）----
    logc = np.full(nw, np.nan)
    pos_ok = np.isfinite(wclose) & (wclose > 0)
    logc[pos_ok] = np.log(wclose[pos_ok])
    slope_t, slope = rolling_ols_t(logc, STRENGTH_WIN, STRENGTH_MIN_PTS)
    er = efficiency_ratio(wclose, ER_WIN)
    strong_ok = np.isfinite(slope_t) & (slope_t >= STRENGTH_T)

    ready_week = ready_struct & np.isfinite(slope)
    cond_week = cond_struct_week & strong_ok

    # ---- 结构破坏与下一破坏周 ----
    brk = np.isfinite(wclose) & np.isfinite(wl1) & (wclose < wl1)
    nb = np.full(nw, -1, dtype=np.int64)
    nxt = -1
    for k in range(nw - 1, -1, -1):
        nb[k] = nxt
        if brk[k]:
            nxt = k

    return dict(
        dates=dates, o=o, h=hi, l=lo, c=cl,
        nw=nw, w_end_ts=w_end, wclose=wclose, whigh=whigh, whigh1=wh1,
        wlow1=wl1, wamount=wamount, wdays=wdays, has_bar=has_bar,
        ready_week=ready_week, cond_week=cond_week,
        ready_struct=ready_struct, cond_struct_week=cond_struct_week,
        slope_t=slope_t, slope=slope, er=er,
        nb=nb, hp_idx=hp_idx, hp_val=hp_val, m_up_w=m_up_w, m_turn_w=m_turn_w,
        last_date=dates[-1], week_orphan=week_orphan, ma_week=ma_week,
    )


def volume_label(A: dict, k: int) -> dict:
    """最近 12 个完整周：上涨周 vs 下跌周的平均日成交额。"""
    out = dict(vol_ok=None, vol_up_n=0, vol_dn_n=0, vol_up_amt=np.nan, vol_dn_amt=np.nan)
    if k - 12 < 0:
        return out
    cs = A["wclose"][k - 12:k + 1]
    am = A["wamount"][k - 11:k + 1]
    dy = A["wdays"][k - 11:k + 1]
    if not (np.all(np.isfinite(cs)) and np.all(np.isfinite(am)) and np.all(np.isfinite(dy))):
        return out
    if np.any(dy <= 0) or np.any(am <= 0):
        return out
    up = cs[1:] > cs[:-1]
    dn = cs[1:] < cs[:-1]
    out["vol_up_n"] = int(up.sum())
    out["vol_dn_n"] = int(dn.sum())
    if out["vol_up_n"] < 4 or out["vol_dn_n"] < 4:
        return out
    out["vol_up_amt"] = float(am[up].sum() / dy[up].sum())
    out["vol_dn_amt"] = float(am[dn].sum() / dy[dn].sum())
    out["vol_ok"] = bool(out["vol_up_amt"] > out["vol_dn_amt"])
    return out


def compute_event(k: int, A: dict) -> dict:
    """给定周末索引，按同一结构规则测量阶段（信号组/对照组共用）。"""
    t = A["w_end_ts"][k]
    dates = A["dates"]
    res = dict(
        sig_k=int(k), sig_date=t, tradable=False, end_type="UNTRADABLE",
        entry_date=None, entry_open=np.nan, end_date=None,
        mfe=np.nan, mae=np.nan, dur_days=np.nan, dur_weeks=np.nan,
        level=float(A["whigh1"][k]) if np.isfinite(A["whigh1"][k]) else np.nan,
        break_prior_high=None, new_pivot_high=None,
        days_first_high=np.nan, days_max_high=np.nan,
        ma20_up_months=int(A["m_up_w"][k]),
        ma20_turn_date=A["m_turn_w"][k],
        slope_t=float(A["slope_t"][k]) if np.isfinite(A["slope_t"][k]) else np.nan,
        er=float(A["er"][k]) if np.isfinite(A["er"][k]) else np.nan,
        mdd=np.nan, r20=np.nan, r60=np.nan, r120=np.nan,
        d_up10=np.nan, d_dn10=np.nan, d_up20=np.nan, d_up30=np.nan, d_dn20=np.nan,
        lag_days=0.0, ret_end=np.nan, runup_26=np.nan,
    )
    for nm in MA_KEYS:
        res[nm] = bool(A["ma_week"][nm][k])

    # 识别时“已经涨了多少”：信号前 26 个周收盘的累计涨幅（PIT）
    if k >= STRENGTH_WIN:
        c0 = A["wclose"][k - STRENGTH_WIN]
        c1 = A["wclose"][k]
        if np.isfinite(c0) and np.isfinite(c1) and c0 > 0:
            res["runup_26"] = float(c1) / float(c0) - 1.0

    di = int(np.searchsorted(dates, dt64(t), side="right"))
    if di >= dates.size:
        return res
    ci = CTX["cal_pos"].get(t)
    if ci is None:
        return res
    limit = CTX["cal"][min(ci + 10, len(CTX["cal"]) - 1)]
    if dates[di] > dt64(limit):
        return res  # 长期停牌，下一交易日不可执行

    entry_open = float(A["o"][di])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return res

    e = int(A["nb"][k])
    if e >= 0:
        end_date = A["w_end_ts"][e]
        dj = int(np.searchsorted(dates, dt64(end_date), side="right")) - 1
        end_type = "STRUCTURE_BREAK"
        upto = e
    else:
        dj = dates.size - 1
        end_date = pd.Timestamp(dates[-1])
        end_type = "RIGHT_CENSORED" if dates[-1] >= dt64(CTX["censor_cut"]) else "DATA_TRUNCATED"
        upto = A["nw"] - 1
    dj = max(dj, di)

    sh = A["h"][di:dj + 1]
    sl = A["l"][di:dj + 1]
    if sh.size == 0:
        return res
    imax = int(np.argmax(sh))
    peak = np.maximum.accumulate(np.maximum(sh, entry_open))
    res.update(
        tradable=True, end_type=end_type, entry_date=pd.Timestamp(dates[di]),
        entry_open=entry_open, end_date=end_date,
        mfe=float(sh[imax]) / entry_open - 1.0,
        mae=float(np.min(sl)) / entry_open - 1.0,
        mdd=-float(np.max(1.0 - sl / peak)),
        ret_end=float(A["c"][dj]) / entry_open - 1.0,
        dur_days=float((pd.Timestamp(end_date) - t).days),
        dur_weeks=float(e - k) if e >= 0 else np.nan,
        days_max_high=float((pd.Timestamp(dates[di + imax]) - t).days),
    )

    # 固定持有窗口（与结构终点无关，只看信号后第 N 个交易日收盘）
    for n in FIXED_WINDOWS:
        j = di + n - 1
        if j < dates.size:
            res[f"r{n}"] = float(A["c"][j]) / entry_open - 1.0

    # 阶段内首次触及：先 +x% 还是先 -y%
    day = np.timedelta64(1, "D")
    for thr in FP_UP:
        hit = np.nonzero(sh >= entry_open * (1.0 + thr))[0]
        if hit.size:
            res[f"d_up{int(round(thr * 100))}"] = float((dates[di + hit[0]] - dates[di]) / day)
    for thr in FP_DN:
        hit = np.nonzero(sl <= entry_open * (1.0 - thr))[0]
        if hit.size:
            res[f"d_dn{int(round(thr * 100))}"] = float((dates[di + hit[0]] - dates[di]) / day)

    level = res["level"]
    if np.isfinite(level):
        seg = A["whigh"][k + 1: upto + 1]
        fin = np.isfinite(seg)
        if fin.any():
            res["break_prior_high"] = bool(seg[fin].max() > level)
            rel = np.nonzero(fin & (seg > level))[0]
            if rel.size:
                res["days_first_high"] = float((A["w_end_ts"][k + 1 + int(rel[0])] - t).days)
        j0 = int(np.searchsorted(A["hp_idx"], k, side="right"))
        j1 = int(np.searchsorted(A["hp_idx"], upto, side="right"))
        if j1 > j0:
            res["new_pivot_high"] = bool((A["hp_val"][j0:j1] > level).any())
        else:
            res["new_pivot_high"] = False
    return res


# ---------------------------------------------------------------------------
def attach_volume(m: dict, A: dict, k: int) -> None:
    if m["tradable"]:
        m.update(volume_label(A, k))
    else:
        m.update(vol_ok=None, vol_up_n=0, vol_dn_n=0, vol_up_amt=np.nan, vol_dn_amt=np.nan)


def analyze_signal(code: str, A: dict | None):
    """一轮趋势只取一次信号。

    结构事件（S0）= 结构条件首次满足的周，用于衡量强度门槛的增量；
    主信号（A）= 该结构区间内强度条件首次满足的周（不得早于结构日）。
    均在同一结构破坏前有效，破坏后该区间内的候选全部作废。
    """
    if A is None:
        return [], [], [], 0
    ev_struct: list[dict] = []
    ev_main: list[dict] = []
    episodes: list[tuple] = []
    rejected = 0
    block_until = -1
    for k in np.nonzero(A["cond_struct_week"])[0]:
        k = int(k)
        if k <= block_until:
            continue
        e = int(A["nb"][k])
        block = e if e >= 0 else A["nw"]

        m0 = compute_event(k, A)
        m0.update(code=code, year=int(pd.Timestamp(m0["sig_date"]).year), variant="S0")
        attach_volume(m0, A, k)
        ev_struct.append(m0)

        cand = np.nonzero(A["cond_week"][k:min(block, A["nw"])])[0]
        if cand.size:
            j = k + int(cand[0])
            m1 = compute_event(j, A)
            m1.update(code=code, year=int(pd.Timestamp(m1["sig_date"]).year), variant="A",
                      lag_days=float((pd.Timestamp(m1["sig_date"])
                                      - pd.Timestamp(A["w_end_ts"][k])).days))
            attach_volume(m1, A, j)
            ev_main.append(m1)
        else:
            rejected += 1

        episodes.append((k, block))
        block_until = block
    return ev_struct, ev_main, episodes, rejected


def analyze_control(code: str, A: dict | None, sig_k: list[int], eps: list[tuple]):
    """同一信号日、可评估、未满足条件、不在已识别趋势中的 PIT 股票。"""
    if A is None or not sig_k:
        return None
    out = []
    cond = A["cond_week"]
    ready = A["ready_week"]
    has = A["has_bar"]
    for idx, k in enumerate(sig_k):
        if not has[k] or not ready[k] or cond[k]:
            continue
        if eps and any(s <= k <= e for s, e in eps):
            continue
        m = compute_event(k, A)
        if not m["tradable"]:
            continue
        flags = 0
        if m["end_type"] == "STRUCTURE_BREAK":
            flags |= 1
        if m["break_prior_high"]:
            flags |= 2
        if m["new_pivot_high"]:
            flags |= 4
        if m["end_type"] == "RIGHT_CENSORED":
            flags |= 8
        elif m["end_type"] == "DATA_TRUNCATED":
            flags |= 16
        out.append([
            idx, m["mfe"], m["mae"], m["dur_days"], flags,
            m["slope_t"], m["er"], m["mdd"],
            m["r20"], m["r60"], m["r120"],
            m["d_up10"], m["d_dn10"], m["d_up20"], m["d_up30"], m["d_dn20"],
            m["ret_end"], m["runup_26"],
        ])
    if not out:
        return None
    a = np.asarray(out, dtype=np.float32)
    return dict(
        date_idx=a[:, 0].astype(np.int32),
        mfe=a[:, 1], mae=a[:, 2], dur=a[:, 3],
        flags=a[:, 4].astype(np.int16),
        slope_t=a[:, 5], er=a[:, 6], mdd=a[:, 7],
        r20=a[:, 8], r60=a[:, 9], r120=a[:, 10],
        d_up10=a[:, 11], d_dn10=a[:, 12], d_up20=a[:, 13],
        d_up30=a[:, 14], d_dn20=a[:, 15],
        ret_end=a[:, 16],
        runup_26=a[:, 17],
        lag_days=np.zeros(len(a), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
def block_stats(vals, prefix: str) -> dict:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_n": 0}
    return {
        f"{prefix}_n": int(v.size),
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_q25": float(np.percentile(v, 25)),
        f"{prefix}_q75": float(np.percentile(v, 75)),
        f"{prefix}_q90": float(np.percentile(v, 90)),
        f"{prefix}_min": float(v.min()),
        f"{prefix}_max": float(v.max()),
    }


def tail_stats(vals, edges, prefix: str, below: bool = False) -> dict:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    if below:
        return {f"{prefix}_{abs(e):g}": float((v < e).mean()) for e in edges}
    return {f"{prefix}_{abs(e):g}": float((v > e).mean()) for e in edges}


def rate(num: int, den: int) -> dict:
    return {"n": int(den), "k": int(num), "rate": (float(num) / den) if den else None}


def paired(a_by_date: dict, c_by_date: dict, name: str, reps: int = 1000) -> dict:
    ds = sorted(set(a_by_date) & set(c_by_date))
    if len(ds) < 3:
        return {"name": name, "dates": len(ds)}
    a = np.array([a_by_date[d] for d in ds], dtype=float)
    c = np.array([c_by_date[d] for d in ds], dtype=float)
    d = a - c
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else None
    rng = np.random.default_rng(20261106)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(reps)])
    dm = np.median(d)
    bs_m = np.array([np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(reps)])
    return {
        "name": name, "dates": int(d.size),
        "a_mean": float(a.mean()), "c_mean": float(c.mean()),
        "diff_mean": float(d.mean()), "diff_median": float(dm),
        "t": t, "mean_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "median_ci95": [float(np.percentile(bs_m, 2.5)), float(np.percentile(bs_m, 97.5))],
        "a_positive": float((a > 0).mean()), "c_positive": float((c > 0).mean()),
        "diff_positive_rate": float((d > 0).mean()),
    }


def pack(events: list[dict]) -> dict:
    """事件列表 → 列数组，供统一统计。"""
    G: dict = {"n": len(events)}
    if not events:
        return G
    G["stock_key"] = np.array([e["code"] for e in events], dtype=object)
    G["date_idx"] = np.array([e["date_idx"] for e in events], dtype=np.int64)
    G["year"] = np.array([e["year"] for e in events], dtype=np.int32)
    G["regime"] = np.array([e["regime"] for e in events], dtype=object)
    G["end_type"] = np.array([e["end_type"] for e in events], dtype=object)
    G["vol_ok"] = np.array([e["vol_ok"] for e in events], dtype=object)
    G["bph"] = np.array([e["break_prior_high"] is True for e in events], dtype=bool)
    G["nph"] = np.array([e["new_pivot_high"] is True for e in events], dtype=bool)
    G["bph_na"] = np.array([e["break_prior_high"] is None for e in events], dtype=bool)
    G["complete"] = np.array([t == "STRUCTURE_BREAK" for t in G["end_type"]], dtype=bool)
    G["censored"] = np.array([t == "RIGHT_CENSORED" for t in G["end_type"]], dtype=bool)
    G["truncated"] = np.array([t == "DATA_TRUNCATED" for t in G["end_type"]], dtype=bool)
    G["untradable"] = np.array([t == "UNTRADABLE" for t in G["end_type"]], dtype=bool)
    for k in EVENT_FLOATS:
        G[k] = np.array([e[k] for e in events], dtype=float)
    for k in MA_KEYS:
        G[k] = np.array([e[k] for e in events], dtype=bool)
    return G


def pack_control(ctl: dict | None, stock_key: np.ndarray | None = None) -> dict:
    if ctl is None:
        return {"n": 0}
    f = np.asarray(ctl["flags"], dtype=np.int16)
    n = f.size
    et = np.full(n, "UNTRADABLE", dtype=object)
    et[(f & 16) > 0] = "DATA_TRUNCATED"
    et[(f & 8) > 0] = "RIGHT_CENSORED"
    et[(f & 1) > 0] = "STRUCTURE_BREAK"
    nan = np.full(n, np.nan)
    G = dict(
        n=n,
        stock_key=stock_key if stock_key is not None else np.zeros(n, dtype=np.int32),
        date_idx=ctl["date_idx"].astype(np.int64),
        end_type=et,
        bph=(f & 2) > 0, nph=(f & 4) > 0, bph_na=np.zeros(n, dtype=bool),
        complete=et == "STRUCTURE_BREAK",
        censored=et == "RIGHT_CENSORED",
        truncated=et == "DATA_TRUNCATED",
        untradable=et == "UNTRADABLE",
        mfe=ctl["mfe"].astype(float), mae=ctl["mae"].astype(float),
        dur_days=ctl["dur"].astype(float), dur_weeks=nan,
        days_first_high=nan, days_max_high=nan, ma20_up_months=nan,
        slope_t=ctl["slope_t"].astype(float), er=ctl["er"].astype(float),
        mdd=ctl["mdd"].astype(float),
        r20=ctl["r20"].astype(float), r60=ctl["r60"].astype(float),
        r120=ctl["r120"].astype(float),
        d_up10=ctl["d_up10"].astype(float), d_dn10=ctl["d_dn10"].astype(float),
        d_up20=ctl["d_up20"].astype(float), d_up30=ctl["d_up30"].astype(float),
        d_dn20=ctl["d_dn20"].astype(float),
        ret_end=ctl["ret_end"].astype(float),
        runup_26=ctl["runup_26"].astype(float),
        lag_days=ctl["lag_days"].astype(float),
    )
    return G


def passage_block(G: dict, mask: np.ndarray) -> dict:
    o: dict = {}
    for tag in ("up10", "up20", "up30", "dn10", "dn20"):
        v = G[f"d_{tag}"][mask]
        fin = np.isfinite(v)
        o[f"{tag}_n"] = int(fin.sum())
        o[f"{tag}_rate"] = float(fin.mean()) if v.size else None
        o[f"{tag}_days_median"] = float(np.median(v[fin])) if fin.any() else None
    for a, b in (("up10", "dn10"), ("up20", "dn10"), ("up10", "dn20"), ("up30", "dn20")):
        da = G[f"d_{a}"][mask]
        db = G[f"d_{b}"][mask]
        fa = np.isfinite(da)
        fb = np.isfinite(db)
        both = fa | fb
        nd = int(np.sum(both))
        a_first = (fa & ~fb) | (fa & fb & (da < db))
        b_first = (fb & ~fa) | (fa & fb & (db < da))
        tie = fa & fb & (da == db)
        o[f"race_{a}_{b}"] = {
            "n": nd,
            "a_first": float(a_first.sum() / nd) if nd else None,
            "b_first": float(b_first.sum() / nd) if nd else None,
            "tie": float(tie.sum() / nd) if nd else None,
            "neither": int((~both).sum()),
        }
    return o


def outcome_block(G: dict, comp: np.ndarray) -> dict:
    out: dict = {}
    for n in FIXED_WINDOWS:
        v = G[f"r{n}"][comp]
        v = v[np.isfinite(v)]
        out[f"r{n}_n"] = int(v.size)
        out[f"r{n}_mean"] = float(v.mean()) if v.size else None
        out[f"r{n}_median"] = float(np.median(v)) if v.size else None
        out[f"r{n}_win"] = float((v > 0).mean()) if v.size else None
        out[f"r{n}_gt10"] = float((v > 0.10).mean()) if v.size else None
    v = G["ret_end"][comp]
    v = v[np.isfinite(v)]
    out["ret_end_n"] = int(v.size)
    out["ret_end_mean"] = float(v.mean()) if v.size else None
    out["ret_end_median"] = float(np.median(v)) if v.size else None
    out["ret_end_win"] = float((v > 0).mean()) if v.size else None
    out["ret_end_gt10"] = float((v > 0.10).mean()) if v.size else None
    out.update(passage_block(G, comp))
    return out


def cond_buckets(G: dict, ev: np.ndarray, key: str, nq: int = 5) -> dict:
    """按信号时点可观测变量分位，看后续结果是否随分位变化。"""
    v = G[key][ev]
    fin = np.isfinite(v)
    if fin.sum() < 2 * nq:
        return {}
    qs = np.nanquantile(v[fin], np.linspace(0.0, 1.0, nq + 1))
    out: dict = {"edges": [float(x) for x in qs]}
    for i in range(nq):
        lo, hi = qs[i], qs[i + 1]
        m = ev & np.isfinite(G[key])
        m = m & (G[key] >= lo)
        m = m & ((G[key] <= hi) if i == nq - 1 else (G[key] < hi))
        c = m & G["complete"]
        out[f"q{i + 1}"] = {
            "n": int(m.sum()), "n_complete": int(c.sum()),
            "mfe_median": float(np.median(G["mfe"][c])) if c.sum() else None,
            "ret_end_median": float(np.median(G["ret_end"][c])) if c.sum() else None,
            "dur_median": float(np.median(G["dur_days"][c])) if c.sum() else None,
            "new_high_rate": float((c & G["bph"]).sum() / c.sum()) if c.sum() else None,
            "gt20": float((G["mfe"][c] > 0.20).mean()) if c.sum() else None,
        }
    return out


def group_block(G: dict, mask: np.ndarray | None = None) -> dict:
    n = G["n"]
    if n == 0:
        return {"n_events": 0}
    if mask is None:
        mask = np.ones(n, dtype=bool)
    untradable = G["untradable"]
    ev = mask & ~untradable
    comp = ev & G["complete"]
    d = {
        "n_events": int(ev.sum()),
        "n_stocks": int(len(set(G["stock_key"][ev]))),
        "n_dates": int(len(set(G["date_idx"][ev]))),
        "n_complete": int(comp.sum()),
        "n_censored": int((ev & G["censored"]).sum()),
        "n_truncated": int((ev & G["truncated"]).sum()),
        "n_untradable": int((mask & untradable).sum()),
        "continuation_prior_high": rate(int((comp & G["bph"]).sum()), int(comp.sum())),
        "continuation_new_pivot": rate(int((comp & G["nph"]).sum()), int(comp.sum())),
        "continuation_new_pivot_na": int((comp & G["bph_na"]).sum()),
        "duration_complete": block_stats(G["dur_days"][comp], "dur_days"),
        "duration_weeks": block_stats(G["dur_weeks"][comp], "dur_weeks"),
        "days_first_high": block_stats(G["days_first_high"][comp], "days"),
        "days_max_high": block_stats(G["days_max_high"][comp], "days"),
        "ma20_up_months": block_stats(G["ma20_up_months"][ev], "months"),
        "strength": block_stats(G["slope_t"][ev], "slope_t"),
        "strength_lag_days": block_stats(G["lag_days"][ev], "lag_days"),
        "efficiency_ratio": block_stats(G["er"][ev], "er"),
        "mdd": block_stats(G["mdd"][ev], "mdd"),
        "runup_26": block_stats(G["runup_26"][ev], "runup_26"),
        "by_er": cond_buckets(G, ev, "er"),
        "by_strength": cond_buckets(G, ev, "slope_t"),
        "by_runup": cond_buckets(G, ev, "runup_26"),
        "censored_share": float((ev & G["censored"]).sum() / ev.sum()) if ev.sum() else None,
        "outcomes": outcome_block(G, comp),
    }
    if comp.sum() and np.isfinite(G["dur_weeks"][comp]).any():
        d["w_le2"] = float((G["dur_weeks"][comp] <= 2).mean())
        d["w_le4"] = float((G["dur_weeks"][comp] <= 4).mean())
        d["w_le8"] = float((G["dur_weeks"][comp] <= 8).mean())
    elif comp.sum():
        d["w_le2"] = float((G["dur_days"][comp] <= 14).mean())
        d["w_le4"] = float((G["dur_days"][comp] <= 28).mean())
        d["w_le8"] = float((G["dur_days"][comp] <= 56).mean())
    else:
        d["w_le2"] = d["w_le4"] = d["w_le8"] = None
    if "vol_ok" in G:
        vol_unknown = np.array([v is None for v in G["vol_ok"]], dtype=bool)
        d["n_vol_unknown"] = int((ev & vol_unknown).sum())
    for tag, mm in (("complete", comp), ("censored", ev & G["censored"])):
        d[f"mfe_{tag}"] = block_stats(G["mfe"][mm], "mfe")
        d[f"mfe_{tag}_buckets"] = tail_stats(G["mfe"][mm], MFE_EDGES, "gt")
        d[f"mae_{tag}"] = block_stats(G["mae"][mm], "mae")
        d[f"mae_{tag}_tail"] = tail_stats(G["mae"][mm], MAE_EDGES, "lt", below=True)
    return d


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj).date())
    return obj


# ---------------------------------------------------------------------------
def build_ctx() -> dict:
    cal, cal_pos, month_period_to_end, bm_open, bm_close, week_end_by_iso, _, m_end, w_end, obs = build_market()
    w_end = [pd.Timestamp(x) for x in w_end]
    m_end = [pd.Timestamp(x) for x in m_end]
    wpos = {t: i for i, t in enumerate(w_end)}
    ctx = dict(
        cal=cal, cal_pos=cal_pos, m_end=m_end, w_end=w_end,
        w_end_np=np.array([dt64(t) for t in w_end]),
        m_end_np=np.array([dt64(t) for t in m_end]),
        wkey_to_pos={(int(y) * 100 + int(w)): wpos[pd.Timestamp(dd)]
                     for (y, w), dd in week_end_by_iso.items()},
        month_period_to_end=month_period_to_end,
        mpos={d: i for i, d in enumerate(m_end)},
        cal0=pd.Timestamp(cal[0]),
        start=pd.Timestamp("2005-01-01"),
        censor_cut=pd.Timestamp("2026-08-01"),
        bm_close=bm_close,
    )
    return ctx


def regime_by_year(bm_close: pd.Series) -> dict[int, str]:
    y_end = bm_close.groupby(bm_close.index.year).last()
    ret = y_end / y_end.shift(1) - 1.0
    out = {}
    for y, v in ret.items():
        if not np.isfinite(v):
            out[int(y)] = "unknown"
        elif v > 0.10:
            out[int(y)] = "bull"
        elif v < -0.10:
            out[int(y)] = "bear"
        else:
            out[int(y)] = "range"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--factor-workers", type=int, default=20)
    ap.add_argument("--raw-workers", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-control", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.time()
    global CTX
    codes = load_stock_codes()
    if args.limit:
        codes = codes[: args.limit]
    log(f"股票池 {len(codes)} 只")

    CTX = build_ctx()
    log(f"日历 {len(CTX['cal'])} 日；周末端 {len(CTX['w_end'])}；月端 {len(CTX['m_end'])}")
    year_reg = regime_by_year(CTX["bm_close"])

    log(f"复权因子 {len(codes)} 只（内存，不落盘，并发 {args.factor_workers}）...")
    t0 = time.time()
    facs: dict[str, list[dict] | None] = {}
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
                log(f"  因子 {done}/{len(codes)} fail={sum(1 for v in facs.values() if v is None)} "
                    f"{time.time()-t0:.0f}s")
    log(f"因子有效 {sum(1 for v in facs.values() if v is not None)}/{len(codes)}")

    log("读取本地 raw，陈旧/坏文件内存刷新 ...")
    raw_local: dict[str, pd.DataFrame] = {}
    stale: list[str] = []
    for c in codes:
        df = read_local_raw(c)
        raw_local[c] = df
        if df is None or df.empty:
            stale.append(c)
            continue
        last = pd.to_datetime(df["date"]).max()
        if pd.Timestamp("2026-09-20") <= last <= pd.Timestamp("2026-09-29"):
            stale.append(c)
        elif c in BAD_RAW:
            stale.append(c)
    stale = sorted(set(stale))
    log(f"需内存刷新 {len(stale)} 只")
    fresh: dict[str, pd.DataFrame] = {}
    if stale:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.raw_workers) as ex:
            futs = {ex.submit(fetch_sina, c): c for c in stale}
            done = 0
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    fresh[c] = fut.result()
                except Exception:  # noqa: BLE001
                    fresh[c] = pd.DataFrame()
                done += 1
                if done % 100 == 0 or done == len(stale):
                    log(f"  raw {done}/{len(stale)} empty={sum(1 for v in fresh.values() if v.empty)} "
                        f"{time.time()-t0:.0f}s")
    for c in stale:
        raw_local[c] = merge_fresh(raw_local.get(c, pd.DataFrame()), fresh.get(c, pd.DataFrame()))

    # ---- Pass 1：首次识别事件 + 趋势区间 ----
    log(f"Pass 1 逐股票识别（并发 {args.workers}）...")
    sig: list[dict] = []
    eps_by_code: dict[str, list[tuple]] = {}
    preparable = 0
    orphans = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(prepare_stock, raw_local.get(c, pd.DataFrame()), facs.get(c)): c
                for c in codes}
        A_map: dict[str, dict | None] = {}
        done = 0
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                A = fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"  [warn] {c} prepare 失败 {type(e).__name__} {e}")
                A = None
            A_map[c] = A
            if A is not None:
                preparable += 1
                orphans += A["week_orphan"]
            done += 1
            if done % 500 == 0 or done == len(codes):
                log(f"  prepare {done}/{len(codes)} 可评估={preparable} {time.time()-t0:.0f}s")

    log("信号汇总 ...")
    raw_local.clear()
    fresh.clear()
    log(f"释放 raw 内存，保留 prepared 数组 {len(A_map)} 只供 Pass 2")
    sig_struct: list[dict] = []
    sig: list[dict] = []
    rejected = 0
    for c in codes:
        e0, ea, ep, rj = analyze_signal(c, A_map.get(c))
        sig_struct.extend(e0)
        sig.extend(ea)
        eps_by_code[c] = ep
        rejected += rj
    sig_dates = sorted({pd.Timestamp(e["sig_date"]) for e in sig})
    didx = {d: i for i, d in enumerate(sig_dates)}
    s0_dates = sorted({pd.Timestamp(e["sig_date"]) for e in sig_struct})
    didx0 = {d: i for i, d in enumerate(s0_dates)}
    for e in sig:
        e["date_idx"] = didx[pd.Timestamp(e["sig_date"])]
        e["regime"] = year_reg.get(e["year"], "unknown")
    for e in sig_struct:
        e["date_idx"] = didx0[pd.Timestamp(e["sig_date"])]
        e["regime"] = year_reg.get(e["year"], "unknown")
    log(f"主信号 {len(sig)} 个 / 股票 {len({e['code'] for e in sig})} 只 / 日期 {len(sig_dates)} 个；"
        f"结构信号 {len(sig_struct)} 个；被强度门槛剔除的结构区间 {rejected} 个")

    # ---- Pass 2：同期对照 ----
    ctl = None
    if not args.skip_control and sig_dates:
        sig_k = [CTX["w_end"].index(d) for d in sig_dates if d in CTX["w_end"]]
        log(f"Pass 2 对照组（信号日 {len(sig_k)} 个，并发 {args.workers}）...")
        parts: list[dict] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(analyze_control, c, A_map.get(c), sig_k, eps_by_code.get(c, [])): c
                    for c in codes}
            done = 0
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001
                    log(f"  [warn] {c} control 失败 {type(e).__name__} {e}")
                    r = None
                if r is not None:
                    r["code_key"] = np.full(len(r["date_idx"]), c, dtype=object)
                    parts.append(r)
                done += 1
                if done % 500 == 0 or done == len(codes):
                    log(f"  对照 {done}/{len(codes)} 事件={sum(len(x['date_idx']) for x in parts)} "
                        f"{time.time()-t0:.0f}s")
        if parts:
            keys = ("date_idx", "mfe", "mae", "dur", "flags", "slope_t", "er", "mdd",
                    "r20", "r60", "r120", "d_up10", "d_dn10", "d_up20", "d_up30",
                    "d_dn20", "ret_end", "runup_26", "lag_days")
            ctl = {k: np.concatenate([p[k] for p in parts]) for k in keys}
            ctl["code_key"] = np.concatenate([p["code_key"] for p in parts])
            log(f"对照事件 {len(ctl['date_idx'])}")

    # ---- 统计 ----
    log("统计 ...")
    G0 = pack(sig_struct)   # S0：仅结构（用于衡量强度门槛的增量）
    GA = pack(sig)          # A：结构 + 强度
    volm = (np.array([v is True for v in GA["vol_ok"]], dtype=bool)
            if GA["n"] else np.zeros(0, dtype=bool))

    n0 = int((~G0["untradable"]).sum()) if G0["n"] else 0
    na = int((~GA["untradable"]).sum()) if GA["n"] else 0

    summary: dict = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "scripts/uptrend_detector.py",
            "period": ["2005-01-01", str(END.date())],
            "codes": len(codes),
            "preparable": preparable,
            "week_orphan_bars": int(orphans),
            "entry": "信号后下一交易日开盘（个股自身次一 bar，且须在信号后 10 个市场交易日内）",
            "signal_rule": "月线：最近两个已确认波段低点抬高 + 最近6个月收盘均值高于此前6个月；"
                           "周线：最近两个已确认波段高点抬高 + 最近两个已确认波段低点抬高；"
                           "强度：周线 26 周对数收盘回归斜率 t>=2。均线不参与识别，仅作为事后维度。",
            "strength_rule": "26 周窗口内至少 20 个有效收盘；t = slope / se(slope)",
            "end_rule": "完整周收盘 < 最近已确认周线波段低点",
            "censor_rule": "截至数据结束未破坏：最后 bar >= 2026-08-01 记 RIGHT_CENSORED，否则 DATA_TRUNCATED",
            "pivot_rule": "2 左 2 右；低点左侧严格、右侧含等号；高点对称；确认索引=极值+2",
            "volume_rule": "最近 12 个完整周；上涨周与下跌周平均日成交额；各需 >=4 周",
            "outcome_rule": "主口径=结构终点；另报固定 20/60/120 交易日收盘收益与阶段内首次触及竞争",
            "path_rule": "mdd = 阶段内 (峰值-最低)/峰值 的最大值（负值）；er = |净位移|/路径长度（26 周）",
            "variant_rule": "S0=仅结构；A=结构+强度（主信号）；B=A+量能确认；C=同期对照",
        },
        "signal_counts": {
            "events": len(sig),
            "stocks": int(len(set(GA["stock_key"]))),
            "dates": len(sig_dates),
            "first_date": str(sig_dates[0].date()) if sig_dates else None,
            "last_date": str(sig_dates[-1].date()) if sig_dates else None,
        },
        "structure_only_counts": {
            "events": len(sig_struct),
            "tradable": n0,
            "rejected_by_strength": int(rejected),
            "main_events": len(sig_struct) - int(rejected),
            "main_tradable": na,
        },
        "groups": {
            "S0_structure_only": group_block(G0),
            "A_structure_strength": group_block(GA),
            "B_structure_vol": group_block(GA, volm),
        },
        "per_year": {},
        "per_regime": {},
        "concentration": {},
        "control": {},
    }

    years = sorted(set(GA["year"].tolist())) if GA["n"] else []
    for y in years:
        my = GA["year"] == y
        row = {}
        for name, m in (("A_structure_strength", my), ("B_structure_vol", my & volm)):
            c = m & GA["complete"]
            row[name] = {
                "n": int(m.sum()), "n_complete": int(c.sum()),
                "new_high_rate": float((c & GA["bph"]).sum() / c.sum()) if c.sum() else None,
                "mfe_median": float(np.median(GA["mfe"][c])) if c.sum() else None,
                "mfe_mean": float(np.mean(GA["mfe"][c])) if c.sum() else None,
                "gt10": float((GA["mfe"][c] > 0.10).mean()) if c.sum() else None,
            }
        summary["per_year"][str(y)] = row

    for reg in ("bull", "bear", "range", "unknown"):
        mr = np.array([r == reg for r in GA["regime"]], dtype=bool) if GA["n"] else np.zeros(0, bool)
        if mr.sum() == 0:
            continue
        c = mr & GA["complete"]
        summary["per_regime"][reg] = {
            "n": int(mr.sum()), "n_complete": int(c.sum()),
            "new_high_rate": float((c & GA["bph"]).sum() / c.sum()) if c.sum() else None,
            "mfe_median": float(np.median(GA["mfe"][c])) if c.sum() else None,
            "mfe_mean": float(np.mean(GA["mfe"][c])) if c.sum() else None,
            "gt10": float((GA["mfe"][c] > 0.10).mean()) if c.sum() else None,
        }

    # ---- 均线维度：同一事件集合上的事后过滤，不作为识别条件 ----
    base = ~GA["untradable"]
    ma_masks = {k: GA[k] for k in MA_KEYS}
    ma_masks["combo_ma20_60"] = (GA["m_ma20_above"] & GA["m_ma20_up"]
                                 & GA["w_ma60_above"] & GA["w_ma60_up"])
    summary["ma_modules"] = {}
    for name, m in [("none", base)] + [(k, v) for k, v in ma_masks.items()]:
        sel = m & base
        c = sel & GA["complete"]
        summary["ma_modules"][name] = {
            "retained_share": float(sel.sum() / base.sum()) if base.sum() else None,
            "n": int(sel.sum()), "n_complete": int(c.sum()),
            "new_high_rate": float((c & GA["bph"]).sum() / c.sum()) if c.sum() else None,
            "mfe_median": float(np.median(GA["mfe"][c])) if c.sum() else None,
            "mfe_mean": float(np.mean(GA["mfe"][c])) if c.sum() else None,
            "gt10": float((GA["mfe"][c] > 0.10).mean()) if c.sum() else None,
            "mae_median": float(np.median(GA["mae"][c])) if c.sum() else None,
            "dur_median": float(np.median(GA["dur_days"][c])) if c.sum() else None,
        }

    cc = np.array([str(x) for x in GA["stock_key"]]) if GA["n"] else np.array([])
    cnt = pd.Series(cc).value_counts() if cc.size else pd.Series(dtype=int)
    summary["concentration"] = ({
        "n_stocks": int(cnt.size),
        "top1_share": float(cnt.iloc[0] / cnt.sum()),
        "top5_share": float(cnt.iloc[:5].sum() / cnt.sum()),
        "top10_share": float(cnt.iloc[:10].sum() / cnt.sum()),
        "top_decile_share": float(cnt.iloc[:max(1, int(np.ceil(cnt.size * 0.1)))].sum() / cnt.sum()),
        "events_max_one_stock": int(cnt.iloc[0]),
        "hhi": float(((cnt / cnt.sum()) ** 2).sum()),
    } if cnt.size else {})

    # ---- 对照组比较 ----
    if ctl is not None:
        CG = pack_control(ctl, ctl.get("code_key"))
        summary["control"] = group_block(CG)

        def per_date(vals, dates, mask):
            dd: dict[int, list[float]] = {}
            for i, v in zip(dates[mask], vals[mask]):
                dd.setdefault(int(i), []).append(float(v))
            return dd

        def agg_map(vals, dates, mask, how):
            return {k: how(np.asarray(v)) for k, v in per_date(vals, dates, mask).items()}

        def fin(a: np.ndarray, m: np.ndarray) -> np.ndarray:
            return m & np.isfinite(a)

        m_a = ~GA["untradable"] & GA["complete"]
        m_a_bph = m_a & ~GA["bph_na"]
        m_c = CG["complete"]

        def both(vals_a, vals_c, how, name, use_finite=True):
            ma = fin(vals_a, m_a) if use_finite else m_a
            mc = fin(vals_c, m_c) if use_finite else m_c
            return paired(agg_map(vals_a, GA["date_idx"], ma, how),
                          agg_map(vals_c, CG["date_idx"], mc, how), name)

        summary["control"]["paired"] = {
            "mfe_median": both(GA["mfe"], CG["mfe"], np.median, "mfe_median"),
            "mfe_mean": both(GA["mfe"], CG["mfe"], np.mean, "mfe_mean"),
            "mae_median": both(GA["mae"], CG["mae"], np.median, "mae_median"),
            "gt10": both(GA["mfe"], CG["mfe"], lambda x: float((x > 0.10).mean()), "gt10"),
            "gt20": both(GA["mfe"], CG["mfe"], lambda x: float((x > 0.20).mean()), "gt20"),
            "dur_median": both(GA["dur_days"], CG["dur_days"], np.median, "dur_median"),
            "new_high": paired(agg_map(GA["bph"].astype(float), GA["date_idx"], m_a_bph, np.mean),
                               agg_map(CG["bph"].astype(float), CG["date_idx"], m_c, np.mean), "new_high"),
            "r20_mean": both(GA["r20"], CG["r20"], np.mean, "r20_mean"),
            "r60_mean": both(GA["r60"], CG["r60"], np.mean, "r60_mean"),
            "r120_mean": both(GA["r120"], CG["r120"], np.mean, "r120_mean"),
            "mdd_median": both(GA["mdd"], CG["mdd"], np.median, "mdd_median"),
            "ret_end_median": both(GA["ret_end"], CG["ret_end"], np.median, "ret_end_median"),
            "ret_end_mean": both(GA["ret_end"], CG["ret_end"], np.mean, "ret_end_mean"),
            "er_median": both(GA["er"], CG["er"], np.median, "er_median"),
            "up10_rate": both(GA["d_up10"], CG["d_up10"],
                              lambda x: float(np.isfinite(x).mean()), "up10_rate", use_finite=False),
            "dn10_rate": both(GA["d_dn10"], CG["d_dn10"],
                              lambda x: float(np.isfinite(x).mean()), "dn10_rate", use_finite=False),
        }

    out_json = os.path.join(OUT_DIR, "uptrend_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, ensure_ascii=False, indent=2)
    log(f"WROTE {out_json} elapsed={time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
