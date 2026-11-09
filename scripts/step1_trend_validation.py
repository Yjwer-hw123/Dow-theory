"""STEP 1：月线/周线趋势选股有效性验证（一次性研究脚本）。

严格约束
--------
- 不建设数据管道、不落 qfq/缓存/快照。
- 复权因子通过新浪接口在内存中取得；本地 raw 只读。
- 仅把统计结果写到 out/step1/。
- 观察时点：每月最后一个已完成交易日；信号只用当时已完成的月/周 K。
- 未来收益：信号次一交易日开盘买入，持有 5/20/60/120 个交易日。

数据口径
--------
- 股票池：data/universe/stock_codes.json 的 5,555 只沪深 A（5,222 存活 + 333 退市）。
- 本地 raw 为不复权；用新浪 qfq 累计因子在内存中算 qfq = raw / f。
- 基准：399317 国证A指（宽基全 A，2005-02-03 起）。
- 不做历史 ST 排除，不建行业映射。
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

from data.stock_loader import fetch_sina  # noqa: E402
from data.pit_universe import load_stock_codes  # noqa: E402
from scripts.build_qfq import fetch_factors  # noqa: E402

OUT_DIR = os.path.join(ROOT, "out", "step1")
RAW_DIR = os.path.join(ROOT, "cache", "raw")
FAC_DIR = os.path.join(ROOT, "cache", "factors")
UNIVERSE_DIR = os.path.join(ROOT, "data", "universe")

START = pd.Timestamp("2005-01-01")
END = pd.Timestamp("2026-09-30")
HORIZONS = [5, 20, 60, 120]
GROUPS = ["a_all", "a_m", "b", "a_w", "b_w", "c"]

# 本地缓存里已知的坏文件（非正价格 / OHLC 逻辑错 / 零价格）。只做内存级重取。
BAD_RAW = {
    "000001", "000333", "000003", "000013", "000015",
    "688089", "688143", "688173", "000725", "000938",
}


# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def as_ts(x) -> pd.Timestamp:
    return pd.Timestamp(x)


def is_finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """只保留可用于指标/成交计算的合法 bar。"""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    need = ["date", "open", "high", "low", "close", "volume"]
    for c in need:
        if c not in d.columns:
            return pd.DataFrame()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date"])
    d = d[(d["date"] >= pd.Timestamp("1990-01-01")) & (d["date"] <= END)]
    o, h, l, c, v = (d[k].astype(float) for k in ["open", "high", "low", "close", "volume"])
    ok = (
        (o > 0) & (h > 0) & (l > 0) & (c > 0) & (v > 0)
        & (h >= l) & (h >= o) & (h >= c) & (l <= o) & (l <= c)
    )
    d = d.loc[ok].copy()
    d = d.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return d


def read_local_raw(code: str) -> pd.DataFrame:
    p = os.path.join(RAW_DIR, f"{code}.parquet")
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame()


def load_factor(code: str) -> list[dict] | None:
    p = os.path.join(FAC_DIR, f"{code}.json")
    if os.path.exists(p):
        try:
            obj = json.load(open(p, encoding="utf-8"))
            if isinstance(obj, list):
                return obj
        except Exception:  # noqa: BLE001
            pass
    for attempt in range(3):
        try:
            f = fetch_factors(code)
        except Exception:  # noqa: BLE001
            f = None
        if f is not None:
            return f
        time.sleep(0.35 * (attempt + 1))
    return None


def apply_qfq(df: pd.DataFrame, facs: list[dict] | None) -> pd.DataFrame:
    """qfq = raw / f；f 取最后一个 <= 当日的因子。"""
    out = df.copy()
    if not facs:
        return out
    f = pd.Series(
        [float(x["f"]) for x in facs],
        index=pd.to_datetime([x["d"] for x in facs]),
    ).sort_index()
    idx = pd.to_datetime(out["date"])
    factor = f.reindex(idx, method="ffill").fillna(1.0).to_numpy()
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].astype(float) / factor
    return out


def merge_fresh(local: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """用接口新数据覆盖同日期旧数据；保留旧数据中更早的历史。"""
    if fresh is None or fresh.empty:
        return local
    fr = fresh.copy()
    fr["date"] = pd.to_datetime(fr["date"])
    fr = fr[fr["date"] <= END]
    if fr.empty:
        return local
    if local is None or local.empty:
        return fr
    lo = local.copy()
    lo["date"] = pd.to_datetime(lo["date"])
    start = fr["date"].min()
    keep = lo[lo["date"] < start]
    out = pd.concat([keep, fr], ignore_index=True)
    return out.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
def fetch_index_em(symbol: str) -> pd.DataFrame:
    """东财指数日线（单次返回全历史）。"""
    import akshare as ak

    df = ak.stock_zh_index_daily_em(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)


def build_market() -> tuple[list[pd.Timestamp], dict, dict, pd.Series, pd.Series,
                            dict, dict, list[pd.Timestamp], list[pd.Timestamp], dict]:
    """返回市场日历、基准、月/周末端、ISO 映射、观察时点。

    日历需要覆盖 2005 年前的均线预热期：
    - 本地 000300 缓存 2002-01-04 ~ 2026-09-24；
    - 东财 000300 2005-01-04 ~ 2026-09-30；
    - 东财 399317（基准）2005-02-03 ~ 2026-09-30。
    三者日期并集作为交易日历。基准自身行情不做任何前值填充。
    """
    local300 = pd.read_parquet(os.path.join(ROOT, "cache", "000300_daily.parquet"))
    local300["date"] = pd.to_datetime(local300["date"])
    local300 = local300[(local300["date"] >= pd.Timestamp("2002-01-01")) & (local300["date"] <= END)]

    try:
        em300 = fetch_index_em("sh000300")
    except Exception:  # noqa: BLE001
        em300 = fetch_sina("000300", 6000)
    em300 = em300[(em300["date"] >= pd.Timestamp("2002-01-01")) & (em300["date"] <= END)]

    try:
        bm = fetch_index_em("sz399317")
    except Exception:  # noqa: BLE001
        bm = fetch_sina("399317", 6000)
    bm["date"] = pd.to_datetime(bm["date"])
    bm = bm[bm["date"] <= END].sort_values("date").reset_index(drop=True)

    cal = sorted(set(
        local300["date"].dt.normalize().tolist()
        + em300["date"].dt.normalize().tolist()
        + bm["date"].dt.normalize().tolist()
    ))
    cal_pos = {d: i for i, d in enumerate(cal)}
    bm_open = bm.set_index("date")["open"].astype(float)
    bm_close = bm.set_index("date")["close"].astype(float)

    # 市场月/周末端（来自交易日历，不使用自然日强推）
    cser = pd.Series(cal)
    m_end = sorted(cser.groupby(cser.dt.to_period("M")).max().tolist())
    month_period_to_end = {pd.Timestamp(d).to_period("M"): pd.Timestamp(d) for d in m_end}

    week_end_by_iso: dict[tuple[int, int], pd.Timestamp] = {}
    for d in cal:
        iso = pd.Timestamp(d).isocalendar()
        week_end_by_iso[(int(iso.year), int(iso.week))] = pd.Timestamp(d)
    w_end = sorted(set(week_end_by_iso.values()))

    obs = [pd.Timestamp(d) for d in m_end if START <= pd.Timestamp(d) <= END]
    return cal, cal_pos, month_period_to_end, bm_open, bm_close, week_end_by_iso, {}, m_end, w_end, obs


def regime_at(t: pd.Timestamp, bm_close: pd.Series, ma200: pd.Series, slope: pd.Series) -> str:
    if t not in bm_close.index:
        return "unknown"
    c = float(bm_close.loc[t])
    m = float(ma200.loc[t]) if t in ma200.index else float("nan")
    s = float(slope.loc[t]) if t in slope.index else float("nan")
    if not (is_finite(c) and is_finite(m) and is_finite(s)):
        return "unknown"
    if c > m and m > s:
        return "bull"
    if c < m and m < s:
        return "bear"
    return "range"


# ---------------------------------------------------------------------------
def process_stock(
    code: str,
    raw: pd.DataFrame,
    facs: list[dict] | None,
    cal: list[pd.Timestamp],
    cal_pos: dict,
    month_period_to_end: dict,
    week_end_by_iso: dict,
    w_end: list[pd.Timestamp],
    obs: list[pd.Timestamp],
    bm_open: pd.Series,
    bm_close: pd.Series,
    regimes: dict,
    year_map: dict,
) -> list[tuple]:
    """单只股票：月/周指标 + 逐月观察 + 未来收益。"""
    d = clean_raw(raw)
    if d.empty or facs is None:
        return []
    d = apply_qfq(d, facs)
    d = clean_raw(d)  # 复权后再次检查
    if len(d) < 30:
        return []

    s = d.set_index("date").sort_index()
    close = s["close"].astype(float)
    open_ = s["open"].astype(float)

    # ---- 月线：自然月最后一个可用交易日收盘；对齐到市场月端 ----
    mm = pd.DataFrame({"close": close})
    mm["p"] = close.index.to_period("M")
    mclose = mm.groupby("p")["close"].last()
    mclose.index = pd.Index([month_period_to_end.get(p, pd.NaT) for p in mclose.index])
    mclose = mclose[~mclose.index.isna()].sort_index()
    mclose = mclose.reindex(sorted(month_period_to_end.values()))
    ma20 = mclose.rolling(20, min_periods=20).mean()
    ma20_lag3 = ma20.shift(3)
    b_cond = (mclose > ma20) & (ma20 > ma20_lag3)
    m_ready = ma20.notna() & ma20_lag3.notna()

    # ---- 周线：自然周最后一个可用交易日收盘；对齐到市场周末端 ----
    iso = close.index.isocalendar()
    ww = pd.DataFrame({
        "close": close.values,
        "iy": iso["year"].values,
        "iw": iso["week"].values,
    })
    wclose = ww.groupby(["iy", "iw"])["close"].last()
    wclose.index = pd.Index([
        week_end_by_iso.get((int(y), int(w),), pd.NaT) for y, w in wclose.index
    ])
    wclose = wclose[~wclose.index.isna()].sort_index()
    wclose = wclose.reindex(w_end)
    ma60 = wclose.rolling(60, min_periods=60).mean()
    ma60_lag4 = ma60.shift(4)
    c_cond = (wclose > ma60) & (ma60 > ma60_lag4)
    w_ready = ma60.notna() & ma60_lag4.notna()

    m_end = sorted(month_period_to_end.values())
    m_pos = {d: i for i, d in enumerate(m_end)}
    code_int = int(code)
    rows: list[tuple] = []
    has_open = set(d["date"].tolist())

    for t in obs:
        t = pd.Timestamp(t)
        if t not in has_open:
            continue  # 当日无 bar → 不在 PIT 池
        mi = m_pos.get(t)
        if mi is None:
            continue
        m_rdy = bool(m_ready.iloc[mi])
        b_on = bool(b_cond.iloc[mi]) if m_rdy else False
        wi = int(np.searchsorted(w_end, t, side="right") - 1)
        w_rdy = bool(w_ready.iloc[wi]) if wi >= 0 else False
        c_on = bool(c_cond.iloc[wi]) if (wi >= 0 and w_rdy) else False
        b_w = bool(b_on and w_rdy)
        c_flag = bool(b_w and c_on)

        ci = cal_pos.get(t)
        if ci is None or ci + 1 >= len(cal):
            continue
        entry = cal[ci + 1]
        o = open_.get(entry, np.nan)
        if not is_finite(o) or float(o) <= 0:
            continue

        vals: dict[int, tuple[float, float]] = {}
        for h in HORIZONS:
            xi = ci + h - 1
            if xi >= len(cal):
                vals[h] = (np.nan, np.nan)
                continue
            exit_d = cal[xi]
            c = close.get(exit_d, np.nan)
            if not is_finite(c):
                seg = close.loc[entry:exit_d].dropna()
                if seg.empty:
                    vals[h] = (np.nan, np.nan)
                    continue
                c = float(seg.iloc[-1])
            r = float(c) / float(o) - 1.0
            bo = bm_open.get(entry, np.nan)
            bc = bm_close.get(exit_d, np.nan)
            br = float(bc) / float(bo) - 1.0 if (is_finite(bo) and is_finite(bc) and float(bo) > 0) else np.nan
            vals[h] = (r, br)

        y = year_map[t]
        reg = regimes.get(t, "unknown")
        rows.append((
            code_int, int(t.strftime("%Y%m%d")), y, reg,
            1, int(m_rdy), int(b_on), int(w_rdy), int(b_w), int(c_flag),
            vals[5][0], vals[20][0], vals[60][0], vals[120][0],
            vals[5][1], vals[20][1], vals[60][1], vals[120][1],
        ))
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 只（冒烟用）")
    ap.add_argument("--factor-workers", type=int, default=20)
    ap.add_argument("--raw-workers", type=int, default=12)
    ap.add_argument("--stock-workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.time()

    codes = load_stock_codes()
    if args.limit:
        codes = codes[: args.limit]
    log(f"股票池 {len(codes)} 只")

    # 元数据（仅用于报告/命名，不影响 PIT 判定）
    live = json.load(open(os.path.join(UNIVERSE_DIR, "all_a.json"), encoding="utf-8"))
    dead = json.load(open(os.path.join(UNIVERSE_DIR, "delisted.json"), encoding="utf-8"))
    names = {str(x["code"]).zfill(6): x.get("name", "") for x in live}
    names.update({str(x["code"]).zfill(6): x.get("name", "") for x in dead})
    dead_codes = {str(x["code"]).zfill(6) for x in dead}

    # 市场日历 / 基准
    log("抓取市场日历与基准 399317 ...")
    cal, cal_pos, month_period_to_end, bm_open, bm_close, week_end_by_iso, _, m_end, w_end, obs = build_market()
    log(f"日历 {len(cal)} 日 {cal[0].date()}~{cal[-1].date()}；基准 {len(bm_close)} 日 "
        f"{bm_close.index[0].date()}~{bm_close.index[-1].date()}；观察 {len(obs)} 个月")

    # 每周观察的市场环境（只用截至当日已完成的行情）
    ma200 = bm_close.rolling(200, min_periods=200).mean()
    slope = ma200 - ma200.shift(20)
    regimes = {t: regime_at(pd.Timestamp(t), bm_close, ma200, slope) for t in obs}
    year_map = {pd.Timestamp(t): int(pd.Timestamp(t).year) for t in obs}

    # ---- 1) 复权因子：内存中取得，不落盘 ----
    log(f"抓取复权因子 {len(codes)} 只（并发 {args.factor_workers}）...")
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
                log(f"  因子 {done}/{len(codes)} fail={sum(1 for v in facs.values() if v is None)} "
                    f"{time.time()-t0:.0f}s")
    fac_ok = sum(1 for v in facs.values() if v is not None)
    fac_empty = sum(1 for v in facs.values() if v == [])
    fac_fail = [c for c, v in facs.items() if v is None]

    # ---- 2) 本地 raw 读取 + 已知坏/陈旧文件内存重取 ----
    log("读取本地 raw 并判定需刷新文件 ...")
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
    log(f"需内存刷新 {len(stale)} 只（不落盘）")
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
                    log(f"  raw 刷新 {done}/{len(stale)} empty={sum(1 for v in fresh.values() if v.empty)} "
                        f"{time.time()-t0:.0f}s")
    raw_fail = [c for c in stale if fresh.get(c) is None or fresh.get(c).empty]
    for c in stale:
        raw_local[c] = merge_fresh(raw_local.get(c, pd.DataFrame()), fresh.get(c, pd.DataFrame()))

    # ---- 3) 逐股票计算 ----
    log(f"逐股票计算（并发 {args.stock_workers}）...")
    rows_all: list[tuple] = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.stock_workers) as ex:
        futs = {
            ex.submit(
                process_stock, c, raw_local.get(c, pd.DataFrame()), facs.get(c),
                cal, cal_pos, month_period_to_end, week_end_by_iso, w_end, obs,
                bm_open, bm_close, regimes, year_map,
            ): c for c in codes
        }
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                rows_all.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                log(f"  [warn] {c} 计算失败: {type(e).__name__} {e}")
            done += 1
            if done % 500 == 0 or done == len(codes):
                log(f"  股票 {done}/{len(codes)} rows={len(rows_all)} {time.time()-t0:.0f}s")

    cols = [
        "code", "date_i", "year", "regime",
        "a_all", "a_m", "b", "a_w", "b_w", "c",
        "r5", "r20", "r60", "r120", "b5", "b20", "b60", "b120",
    ]
    df = pd.DataFrame(rows_all, columns=cols)
    df["date"] = pd.to_datetime(df["date_i"].astype(str), format="%Y%m%d")
    for g in GROUPS:
        df[g] = df[g].astype(bool)

    # ---- 4) 统计 ----
    log(f"统计：{len(df):,} 行 / {df['code'].nunique()} 只 / {df['date'].nunique()} 个月")

    def group_stats(g: str, h: int) -> dict:
        m = df[g] & df[f"r{h}"].notna()
        sub = df.loc[m]
        if sub.empty:
            return {"n": 0, "unique_stocks": 0, "months": 0}
        r = sub[f"r{h}"].astype(float)
        ex = (sub[f"r{h}"] - sub[f"b{h}"]).dropna()
        monthly = sub.groupby("date")[f"r{h}"].mean()
        return {
            "n": int(len(sub)),
            "unique_stocks": int(sub["code"].nunique()),
            "months": int(sub["date"].nunique()),
            "avg_stocks_per_month": round(float(len(sub) / max(sub["date"].nunique(), 1)), 2),
            "mean": round(float(r.mean()), 6),
            "median": round(float(r.median()), 6),
            "positive_rate": round(float((r > 0).mean()), 6),
            "monthly_equal_weight_mean": round(float(monthly.mean()), 6),
            "excess_mean": round(float(ex.mean()), 6) if len(ex) else None,
            "excess_median": round(float(ex.median()), 6) if len(ex) else None,
            "excess_positive_rate": round(float((ex > 0).mean()), 6) if len(ex) else None,
        }

    summary: dict = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "scripts/step1_trend_validation.py",
            "research_period": [str(START.date()), str(END.date())],
            "observation": "每月最后一个已完成交易日",
            "entry": "观察时点次一交易日开盘",
            "horizons_trading_days": HORIZONS,
            "groups": {
                "a_all": "全部当日有 bar 的 PIT 股票",
                "a_m": "月线可评估（月 MA20 与 3 个月前 MA20 均可用）",
                "b": "月收盘 > 月 MA20，且月 MA20 > 3 个月前 MA20",
                "a_w": "周线可评估（周 MA60 与 4 周前 MA60 均可用）",
                "b_w": "B 且周线可评估（用于隔离周线增量）",
                "c": "B 且周收盘 > 周 MA60，且周 MA60 > 4 周前 MA60",
            },
            "benchmark": {"code": "399317", "name": "国证A指", "source": "新浪财经日线"},
            "stock_source": "本地 cache/raw 不复权 raw + 新浪 qfq 累计因子（内存计算，不落盘）",
            "st_filter": "未做历史 ST 排除；STEP 1 包含 ST 股票，结果不是最终可交易股票池",
        },
        "coverage": {
            "codes": len(codes),
            "live_metadata": len(live),
            "delisted_metadata": len(dead),
            "factor_ok": fac_ok,
            "factor_empty_valid": fac_empty,
            "factor_fail_n": len(fac_fail),
            "factor_fail_sample": fac_fail[:30],
            "raw_refreshed_n": len(stale),
            "raw_refresh_fail_n": len(raw_fail),
            "raw_refresh_fail_sample": raw_fail[:30],
            "stocks_with_rows": int(df["code"].nunique()),
            "rows": int(len(df)),
            "months": int(df["date"].nunique()),
            "last_observation": str(df["date"].max().date()) if len(df) else None,
            "last_bar_20260930_stocks": None,
        },
        "benchmark_alignment": {
            "code": "399317",
            "first": str(bm_close.index[0].date()),
            "last": str(bm_close.index[-1].date()),
            "rows": int(len(bm_close)),
            "market_calendar_rows": len(cal),
            "missing_market_dates": int(len(set(cal) - set(bm_close.index))),
            "missing_before_20050203": int(sum(1 for d in cal if d < bm_close.index[0])),
            "frozen_end": str(END.date()),
            "forward_fill": False,
        },
        "group_horizon": {},
        "annual": [],
        "regime": [],
        "paired": [],
        "concentration": [],
        "yearly_counts": [],
    }

    # 最后 bar 新鲜度
    last_bars = []
    for c in codes:
        r = raw_local.get(c)
        if r is None or r.empty:
            continue
        last_bars.append(pd.to_datetime(r["date"]).max().normalize())
    if last_bars:
        summary["coverage"]["last_bar_20260930_stocks"] = int(sum(1 for x in last_bars if x == END))
        summary["coverage"]["last_bar_before_20260930_stocks"] = int(sum(1 for x in last_bars if x < END))

    for g in GROUPS:
        summary["group_horizon"][g] = {str(h): group_stats(g, h) for h in HORIZONS}

    # 年度
    years = sorted(df["year"].unique().tolist())
    for y in years:
        for g in GROUPS:
            for h in HORIZONS:
                m = df[g] & df[f"r{h}"].notna() & (df["year"] == y)
                sub = df.loc[m]
                if sub.empty:
                    continue
                ex = (sub[f"r{h}"] - sub[f"b{h}"]).dropna()
                summary["annual"].append({
                    "year": int(y), "ytd": bool(y == 2026), "group": g, "horizon": h,
                    "n": int(len(sub)), "unique_stocks": int(sub["code"].nunique()),
                    "mean": float(sub[f"r{h}"].mean()), "median": float(sub[f"r{h}"].median()),
                    "positive_rate": float((sub[f"r{h}"] > 0).mean()),
                    "excess_mean": float(ex.mean()) if len(ex) else None,
                })

    # 市场环境
    for reg in ["bull", "bear", "range", "unknown"]:
        for g in GROUPS:
            for h in HORIZONS:
                m = df[g] & df[f"r{h}"].notna() & (df["regime"] == reg)
                sub = df.loc[m]
                if sub.empty:
                    continue
                ex = (sub[f"r{h}"] - sub[f"b{h}"]).dropna()
                summary["regime"].append({
                    "regime": reg, "group": g, "horizon": h,
                    "n": int(len(sub)), "unique_stocks": int(sub["code"].nunique()),
                    "mean": float(sub[f"r{h}"].mean()), "median": float(sub[f"r{h}"].median()),
                    "positive_rate": float((sub[f"r{h}"] > 0).mean()),
                    "excess_mean": float(ex.mean()) if len(ex) else None,
                })

    # 配对比较（同月份组合均值差）
    pairs = [("b", "a_m"), ("b", "a_all"), ("c", "b_w"), ("c", "b")]
    for g1, g2 in pairs:
        for h in HORIZONS:
            s1 = df[df[g1] & df[f"r{h}"].notna()].groupby("date")[f"r{h}"].mean()
            s2 = df[df[g2] & df[f"r{h}"].notna()].groupby("date")[f"r{h}"].mean()
            diff = (s1 - s2).dropna()
            if len(diff) < 3:
                continue
            mean_d = float(diff.mean())
            sd = float(diff.std(ddof=1))
            tstat = mean_d / (sd / math.sqrt(len(diff))) if sd > 0 else None
            summary["paired"].append({
                "group_a": g1, "group_b": g2, "horizon": h,
                "months": int(len(diff)), "mean_diff": mean_d,
                "median_diff": float(diff.median()),
                "t_stat": tstat,
                "positive_month_share": float((diff > 0).mean()),
            })

    # 集中度：正收益中，前 1%/5%/10% 股票贡献占比
    for g in GROUPS:
        for h in HORIZONS:
            m = df[g] & df[f"r{h}"].notna()
            sub = df.loc[m]
            if sub.empty:
                continue
            by = sub.groupby("code")[f"r{h}"].sum()
            pos = by[by > 0].sum()
            if pos <= 0:
                continue
            k1 = max(1, int(round(len(by) * 0.01)))
            k5 = max(1, int(round(len(by) * 0.05)))
            k10 = max(1, int(round(len(by) * 0.10)))
            top = by.sort_values(ascending=False)
            summary["concentration"].append({
                "group": g, "horizon": h, "stocks": int(len(by)),
                "top1_share_of_positive": float(top.head(k1).clip(lower=0).sum() / pos),
                "top5_share_of_positive": float(top.head(k5).clip(lower=0).sum() / pos),
                "top10_share_of_positive": float(top.head(k10).clip(lower=0).sum() / pos),
                "top1_codes": [int(x) for x in top.head(k1).index.tolist()],
                "top1_names": [names.get(str(int(x)).zfill(6), "") for x in top.head(k1).index.tolist()],
            })

    # 每年实际可参与股票数量（A_all / A_m / B / C，按年内观察均值 + 年末）
    for y in years:
        sub = df[df["year"] == y]
        if sub.empty:
            continue
        last_date = sub["date"].max()
        last = sub[sub["date"] == last_date]
        summary["yearly_counts"].append({
            "year": int(y), "ytd": bool(y == 2026), "months": int(sub["date"].nunique()),
            "a_all_avg": float(sub.groupby("date")["a_all"].sum().mean()),
            "a_m_avg": float(sub.groupby("date")["a_m"].sum().mean()),
            "b_avg": float(sub.groupby("date")["b"].sum().mean()),
            "c_avg": float(sub.groupby("date")["c"].sum().mean()),
            "a_all_year_end": int(last["a_all"].sum()),
            "a_m_year_end": int(last["a_m"].sum()),
            "b_year_end": int(last["b"].sum()),
            "c_year_end": int(last["c"].sum()),
        })

    # ---- 5) 落盘（只写统计，不写行情） ----
    def to_csv(rows: list[dict], name: str) -> None:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, name), index=False, encoding="utf-8-sig")

    gh = []
    for g in GROUPS:
        for h in HORIZONS:
            x = dict(summary["group_horizon"][g][str(h)])
            x.update({"group": g, "horizon": h})
            gh.append(x)
    to_csv(gh, "group_horizon_stats.csv")
    to_csv(summary["annual"], "annual_stats.csv")
    to_csv(summary["regime"], "regime_stats.csv")
    to_csv(summary["paired"], "paired_tests.csv")
    to_csv(summary["concentration"], "concentration.csv")
    to_csv(summary["yearly_counts"], "yearly_counts.csv")

    json.dump(summary, open(os.path.join(OUT_DIR, "step1_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log(f"DONE rows={len(df):,} elapsed={time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
