# -*- coding: utf-8 -*-
"""EP-03 第一轮：B 类信号横截面质量识别的特征有效性研究。

严格约束：
- 只使用信号日及之前可获得的数据；
- 不改 B 信号定义、不做参数搜索、不做机器学习；
- 时间顺序切分：train 2005-2016 / validate 2017-2021 / test 2022-2026；
- 主指标为"同日横截面"IC 与同日分组收益（避免把择时收益误认为选股能力）。

用法：python scripts/entry_phase_03_features.py
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
OUT2 = os.path.join(ROOT, "out", "entry_phase_02")
OUT = os.path.join(ROOT, "out", "entry_phase_03")
QFQ = os.path.join(ROOT, "cache", "qfq")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02_l1 as l1  # noqa: E402
import scripts.uptrend_detector as ud  # noqa: E402
from scripts.step1_trend_validation import RAW_DIR  # noqa: E402

SEGS = ((2005, 2016, "train"), (2017, 2021, "validate"), (2022, 2026, "test"))
FEATURES = [
    "b0_len", "n_pb", "trend_age", "run_high_w", "w_since_h", "er26", "mdd_high",
    "depth", "retrace", "vol_ratio", "dist_high60", "since_high60",
    "mom20", "mom60", "vol20", "atr_pct20",
    "ud_vol20", "gap_freq60", "liq_log",
]
MIN_SI = 130
DAY_MIN_IC = 4
DAY_MIN_Q = 10
SEED = 20261107
# b0_len 来自整段 episode 的 bar 计数，episode 边界由全样本决定，包含信号日之后的 bar
PIT_VALID = {f: True for f in FEATURES}
PIT_VALID["b0_len"] = False


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def worker(code: str, rows: pd.DataFrame, mk_d: np.ndarray, mk_i: np.ndarray) -> list[dict]:
    p = os.path.join(QFQ, f"{code}.parquet")
    if not os.path.exists(p):
        return []
    q = pd.read_parquet(p)
    d = pd.to_datetime(q["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    o = q["open"].to_numpy(float)
    h = q["high"].to_numpy(float)
    lo = q["low"].to_numpy(float)
    c = q["close"].to_numpy(float)
    v = q["volume"].to_numpy(float)
    ret = np.full(c.size, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    out = []
    for row in rows.itertuples():
        sd = pd.to_datetime(row.signal_date)
        si = int(np.searchsorted(d, np.int64(sd.value), side="right") - 1)
        if si < 0 or pd.Timestamp(d[si]).normalize() != sd.normalize():
            continue
        if si < MIN_SI:
            continue
        f = dict(code=code, signal_date=sd, entry_date=pd.to_datetime(row.entry_date))
        f["r60q"] = float(row.qfq_exit / row.qfq_entry - 1.0)
        f["net60"] = f["r60q"] - float(row.cost_total_base)
        f["b0_len"] = float(row.b0_count)
        f["trend_age"] = float(row.age)
        f["depth"] = float(row.depth)
        f["vol_ratio"] = float(row.vol_ratio)

        f["mom20"] = c[si] / c[si - 20] - 1.0
        f["mom60"] = c[si] / c[si - 60] - 1.0
        seg60 = c[si - 59:si + 1]
        hi = float(np.nanmax(seg60))
        f["dist_high60"] = c[si] / hi - 1.0 if hi > 0 else np.nan
        f["since_high60"] = float((si - 59) + int(np.nanargmax(seg60)))

        f["vol20"] = float(np.nanstd(ret[si - 19:si + 1]) * np.sqrt(252.0))
        pc = c[si - 20:si + 1]
        tr = np.maximum(h[si - 19:si + 1] - lo[si - 19:si + 1],
                        np.maximum(np.abs(h[si - 19:si + 1] - pc[:-1]),
                                   np.abs(lo[si - 19:si + 1] - pc[:-1])))
        f["atr_pct20"] = float(np.nanmean(tr) / c[si]) if c[si] > 0 else np.nan

        gf = np.abs(o[si - 59:si + 1] / c[si - 60:si] - 1.0)
        f["gap_freq60"] = float(np.nanmean(gf >= 0.03))

        w_ret = ret[si - 19:si + 1]
        w_vol = v[si - 19:si + 1]
        up = w_ret > 0
        dn = w_ret < 0
        vu = float(np.nanmean(w_vol[up])) if up.any() else np.nan
        vd = float(np.nanmean(w_vol[dn])) if dn.any() else np.nan
        f["ud_vol20"] = vu / vd if (vd and np.isfinite(vd) and vd > 0) else np.nan

        adv = float(np.nanmean(v[si - 19:si + 1] * c[si - 19:si + 1]))
        f["liq_log"] = float(np.log(adv)) if adv > 0 else np.nan

        pi = int(np.searchsorted(mk_d, np.int64(sd.value), side="right") - 1)
        if pi >= 60:
            m20 = mk_i[pi] / mk_i[pi - 20] - 1.0
            m60 = mk_i[pi] / mk_i[pi - 60] - 1.0
            f["rel20"] = f["mom20"] - m20
            f["rel60"] = f["mom60"] - m60
        else:
            f["rel20"] = np.nan
            f["rel60"] = np.nan
        out.append(f)
    return out


def sp(a: pd.Series, b: pd.Series) -> float:
    """无 scipy 的 Spearman：秩化后做 Pearson。"""
    return float(a.rank().corr(b.rank()))


def qstats(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return dict(n=0)
    pos = x[x > 0]
    neg = x[x <= 0]
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()),
                win=float(len(pos) / len(x)),
                payoff=float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else np.nan)


def daily_cs(df: pd.DataFrame, feat: str) -> dict:
    """同日横截面：IC 与 5 组收益（每等权一天，再跨天平均）。"""
    ics, spread, rows = [], [], []
    days = 0
    bucket_stats: dict[int, list[float]] = {b: [] for b in range(1, 6)}
    bucket_n: dict[int, list[int]] = {b: [] for b in range(1, 6)}
    for d0, g in df.groupby("signal_date"):
        g = g.dropna(subset=[feat, "r60q"])
        if len(g) < DAY_MIN_IC:
            continue
        ic = sp(g[feat], g["r60q"])
        if np.isfinite(ic):
            ics.append(float(ic))
        if len(g) >= DAY_MIN_Q:
            days += 1
            r = g[feat].rank(method="first")
            b = np.ceil(r / len(g) * 5).clip(1, 5).astype(int)
            for k in range(1, 6):
                x = g.loc[b == k, "r60q"]
                if len(x) == 0:
                    continue
                bucket_stats[k].append(float(x.mean()))
                bucket_n[k].append(int(len(x)))
            m1 = g.loc[b == 1, "r60q"].mean()
            m5 = g.loc[b == 5, "r60q"].mean()
            spread.append(float(m5 - m1))
    ics = np.array(ics)
    spr = np.array(spread)
    t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(ics.size))) if ics.size > 1 and ics.std(ddof=1) > 0 else np.nan
    ts = float(spr.mean() / (spr.std(ddof=1) / np.sqrt(spr.size))) if spr.size > 1 and spr.std(ddof=1) > 0 else np.nan
    q = {b: dict(mean=float(np.nanmean(bucket_stats[b])) if bucket_stats[b] else np.nan,
                 sig=float(np.nansum(bucket_n[b]))) for b in range(1, 6)}
    return dict(n_days_ic=int(ics.size), ic_mean=float(ics.mean()) if ics.size else np.nan,
                ic_t=t, ic_pos_share=float((ics > 0).mean()) if ics.size else np.nan,
                n_days_q=days, spread=float(spr.mean()) if spr.size else np.nan, spread_t=ts,
                q=q, ics=ics.tolist())


def struct_worker(code: str) -> list[dict]:
    """复算 EP-01 信号结构特征（只用信号日及之前信息）。"""
    p = os.path.join(RAW_DIR, f"{code}.parquet")
    if not os.path.exists(p):
        return []
    raw = pd.read_parquet(p)
    A = ud.prepare_stock(raw, l1.load_facs(code))
    if A is None:
        return []
    s = ep.build_stock(code, A)
    rows = []
    for _e, a, b in ep.episode_groups(s):
        lab = s["LABEL_B"][a:b]
        obs = np.nonzero(lab)[0] + a
        if obs.size == 0:
            continue
        i0 = int(obs[0])
        rows.append(dict(
            code=code,
            signal_date=pd.Timestamp(ep.from_ns(int(s["W_END"][i0]))).normalize(),
            n_pb=int(s["N_PB"][i0]), run_high_w=int(s["RUN_HIGH_W"][i0]),
            w_since_h=int(s["W_SINCE_H"][i0]), er26=float(s["ER26"][i0]),
            mdd_high=float(s["MDD_HIGH"][i0]), retrace=float(s["RETRACE"][i0]),
            struct_ok=float(bool(s["STRUCT_OK"][i0])),
        ))
    return rows


def build_struct(codes) -> pd.DataFrame:
    p = os.path.join(OUT, "p3_struct.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
        return df
    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(struct_worker, c): c for c in codes}
        for i, fut in enumerate(as_completed(futs)):
            recs.extend(fut.result())
            if (i + 1) % 500 == 0 or (i + 1) == len(futs):
                log(f"  struct {i+1}/{len(futs)} {time.time()-t0:.0f}s")
    df = pd.DataFrame(recs)
    if len(df):
        df.to_csv(p, index=False, encoding="utf-8")
    return df


def main() -> None:
    ep.init_ctx()
    l1 = pd.read_csv(os.path.join(OUT2, "trades_b_l1.csv"), dtype={"code": str})
    l1["code"] = l1["code"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    l1["signal_date"] = pd.to_datetime(l1["signal_date"]).dt.normalize()
    l1["entry_date"] = pd.to_datetime(l1["entry_date"]).dt.normalize()

    mk = pd.read_csv(os.path.join(OUT2, "attr_market_proxy.csv"))
    mk["date"] = pd.to_datetime(mk["date"])
    mk_d = mk["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    mk_i = mk["mkt_idx"].to_numpy(float)

    t0 = time.time()
    recs: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(worker, c, g, mk_d, mk_i) for c, g in l1.groupby("code")]
        done = 0
        for fut in as_completed(futs):
            recs.extend(fut.result())
            done += 1
            if done % 500 == 0 or done == len(futs):
                log(f"  stock {done}/{len(futs)} {time.time()-t0:.0f}s")
    df = pd.DataFrame(recs)
    st = build_struct(sorted(l1["code"].unique()))
    df = df.merge(st, on=["code", "signal_date"], how="left")
    log(f"特征完成：{len(df)} / {len(l1)} 信号；结构特征命中 {int(df['n_pb'].notna().sum())}")
    df["seg"] = "na"
    for a, b, nm in SEGS:
        df.loc[df["signal_date"].dt.year.between(a, b), "seg"] = nm
    df = df[df["seg"] != "na"].copy()
    df.to_csv(os.path.join(OUT, "p3_features.csv"), index=False, encoding="utf-8")

    # 每日横截面统计
    res = {}
    ic_year = []
    for a, b, nm in SEGS:
        g = df[df["seg"] == nm]
        res[nm] = dict(n_signals=int(len(g)), n_stocks=int(g["code"].nunique()),
                       years=f"{a}-{b}", features={})
        for ft in FEATURES:
            r = daily_cs(g, ft)
            r.pop("ics")
            res[nm]["features"][ft] = r
            # 年度 IC（用于稳定性）
            for y, gy in g.groupby(g["signal_date"].dt.year):
                ic = gy[[ft, "r60q"]].dropna()
                if len(ic) >= 50:
                    ic_year.append(dict(feature=ft, year=int(y),
                                        ic=sp(ic[ft], ic["r60q"]),
                                        n=int(len(ic))))

    # 方向仅取自 train；val/test 报告同向表现
    table = []
    for ft in FEATURES:
        tr = res["train"]["features"][ft]
        va = res["validate"]["features"][ft]
        te = res["test"]["features"][ft]
        sgn = 1.0 if (tr.get("spread") or 0) >= 0 else -1.0
        table.append(dict(
            feature=ft, sign_train=sgn,
            ic_train=tr.get("ic_mean"), ic_val=va.get("ic_mean"), ic_test=te.get("ic_mean"),
            spread_train=tr.get("spread"), spread_val=va.get("spread"), spread_test=te.get("spread"),
            aligned_val=float(sgn * va.get("spread")) if np.isfinite(va.get("spread", np.nan)) else np.nan,
            aligned_test=float(sgn * te.get("spread")) if np.isfinite(te.get("spread", np.nan)) else np.nan,
            t_val=va.get("spread_t"), t_test=te.get("spread_t"),
            ic_t_train=tr.get("ic_t"), ic_t_val=va.get("ic_t"), ic_t_test=te.get("ic_t"),
            days_train=tr.get("n_days_q"), days_val=va.get("n_days_q"), days_test=te.get("n_days_q"),
        ))
    tab = pd.DataFrame(table)
    tab["pit_valid"] = tab["feature"].map(PIT_VALID)
    tab["pass_raw"] = tab["pit_valid"] & (tab["aligned_val"] > 0) & (tab["aligned_test"] > 0)
    tab["pass_strict"] = tab["pass_raw"] & (tab["t_val"] >= 2.5) & (tab["t_test"] >= 2.5)
    tab.to_csv(os.path.join(OUT, "p3_feature_table.csv"), index=False, encoding="utf-8")
    pd.DataFrame(ic_year).to_csv(os.path.join(OUT, "p3_ic_by_year.csv"), index=False, encoding="utf-8")

    # 分组明细（同日等权）
    qrows = []
    for nm in ("train", "validate", "test"):
        for ft in FEATURES:
            r = res[nm]["features"][ft]
            for k in range(1, 6):
                q = r["q"][k]
                qrows.append(dict(segment=nm, feature=ft, bucket=k, mean_r60=q["mean"], n_signals=q["sig"]))
    pd.DataFrame(qrows).to_csv(os.path.join(OUT, "p3_quintiles_daily.csv"), index=False, encoding="utf-8")

    # 组内合并五分组（补充胜率/盈亏比/净期望，供报告引用）
    prow = []
    for nm in ("train", "validate", "test"):
        g = df[df["seg"] == nm]
        for ft in FEATURES:
            x = g[[ft, "r60q", "net60"]].dropna()
            if len(x) < 100:
                continue
            qq = pd.qcut(x[ft], 5, labels=False, duplicates="drop") + 1
            for k in range(1, 6):
                y = x.loc[qq == k]
                s1 = qstats(y["r60q"])
                s2 = qstats(y["net60"])
                prow.append(dict(segment=nm, feature=ft, bucket=int(k), n=s1.get("n", 0),
                                 mean_r60=s1.get("mean", np.nan), median_r60=s1.get("median", np.nan),
                                 win=s1.get("win", np.nan), payoff=s1.get("payoff", np.nan),
                                 mean_net=s2.get("mean", np.nan)))
    pd.DataFrame(prow).to_csv(os.path.join(OUT, "p3_quintiles_pooled.csv"), index=False, encoding="utf-8")

    summary = dict(meta=dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                             n_signals=len(df), seeds=SEED,
                             segments=[dict(name=n, years=f"{a}-{b}") for a, b, n in SEGS],
                             features=FEATURES, day_min_ic=DAY_MIN_IC, day_min_q=DAY_MIN_Q),
                   results=res, table=table)
    with open(os.path.join(OUT, "p3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    log("完成")
    cols = ["feature", "sign_train", "ic_train", "ic_val", "ic_test",
            "aligned_val", "aligned_test", "t_val", "t_test", "pass_raw", "pass_strict"]
    print(tab[cols].to_string(index=False, float_format=lambda x: f"{x: .4f}"))


if __name__ == "__main__":
    main()
