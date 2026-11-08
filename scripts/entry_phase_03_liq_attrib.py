# -*- coding: utf-8 -*-
"""EP-03 归因：liq_log 的 2017—2021 失效是周期性的还是结构性的？

只回答三问（不改策略、不加新特征、不做 Phase B）：
  Q1 失效是否集中在特定年份 / 板块（行业数据缺失，用板块代理）/ ADV 层级？
  Q2 红利、大盘风格占优能否解释失效？
  Q3 低流动性优势在全市场层面是否稳定存在，还是本身就高度不稳定？

方法：
  A   B 信号按年、按板块分解 A 档（最低 ADV）与 E 档（最高 ADV）的 60 bar 毛收益差；
      并对各档做同段去均值，看是"A 档塌陷"还是"E 档抬升"。
  B   指数年度收益（沪深300 / 中证500 / 中证1000 / 上证），与 A−E 逐年对齐。
  C   全市场月度抽样：每月最后一个交易日，对全部可交易股票做
      rank IC(log ADV20, 未来 60 bar 收益) 与五分组价差，逐年汇总。
      这是 Q3 的关键：不依赖 B 信号，看效应本身。

输出 out/entry_phase_03/：
  p3a_year_ab.csv / p3a_style.csv / p3a_board.csv /
  p3a_market_ic.csv / p3a_market_quintile.csv / p3a_summary.json

用法：python scripts/entry_phase_03_liq_attrib.py
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
OUT3 = os.path.join(ROOT, "out", "entry_phase_03")
QFQ = os.path.join(ROOT, "cache", "qfq")
CACHE = os.path.join(ROOT, "cache")

FWD = 60
MIN_STOCKS_DAY = 50
MIN_BUCKET = 8


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sp(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def board_of(code: str) -> str:
    if code.startswith("60"):
        return "沪主板"
    if code.startswith("00"):
        return "深主板"
    if code.startswith("30"):
        return "创业板"
    if code.startswith("68"):
        return "科创板"
    if code[0] in "84":
        return "北交所"
    return "其他"


def b_signal_yearly() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(os.path.join(OUT3, "p3_features.csv"), dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    tr = df[df["seg"] == "train"]["liq_log"]
    cuts = [float(tr.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]
    names = ["A", "B", "C", "D", "E"]
    df["bucket"] = pd.cut(df["liq_log"], bins=[-np.inf, *cuts, np.inf], labels=names)
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year
    df["board"] = df["code"].map(board_of)

    rows = []
    for y, g in df.groupby("year"):
        a = g.loc[g["bucket"] == "A", "r60q"]
        e = g.loc[g["bucket"] == "E", "r60q"]
        if len(a) < 5 or len(e) < 5:
            continue
        rows.append(dict(year=int(y), n=len(g), n_a=len(a), n_e=len(e),
                         A=float(a.mean()), E=float(e.mean()),
                         diff=float(a.mean() - e.mean()),
                         A_win=float((a > 0).mean()), E_win=float((e > 0).mean())))
    yd = pd.DataFrame(rows)

    bs = []
    for seg in ("train", "validate", "test"):
        for bd, g in df[df["seg"] == seg].groupby("board"):
            a = g.loc[g["bucket"] == "A", "r60q"]
            e = g.loc[g["bucket"] == "E", "r60q"]
            if len(a) < MIN_BUCKET or len(e) < MIN_BUCKET:
                continue
            bs.append(dict(seg=seg, board=bd, n=len(g), n_a=len(a), n_e=len(e),
                           A=float(a.mean()), E=float(e.mean()),
                           diff=float(a.mean() - e.mean())))
    bd_df = pd.DataFrame(bs)
    return yd, bd_df


def demeaned_by_seg() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(OUT3, "p3_features.csv"), dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    tr = df[df["seg"] == "train"]["liq_log"]
    cuts = [float(tr.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]
    df["bucket"] = pd.cut(df["liq_log"], bins=[-np.inf, *cuts, np.inf], labels=list("ABCDE"))
    rows = []
    for seg in ("train", "validate", "test"):
        g = df[df["seg"] == seg]
        mean_all = float(g["r60q"].mean())
        for b in "ABCDE":
            x = g.loc[g["bucket"] == b, "r60q"]
            rows.append(dict(seg=seg, bucket=b, n=int(len(x)),
                             mean=float(x.mean()), excess=float(x.mean() - mean_all)))
    return pd.DataFrame(rows)


def index_style() -> pd.DataFrame:
    rows = []
    for code in ("000001", "000300", "000905", "000852"):
        p = os.path.join(CACHE, f"{code}_daily.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)[["date", "close"]].copy()
        d["year"] = d["date"].dt.year
        yc = d.groupby("year")["close"].last()
        for y in range(2006, 2027):
            if y in yc.index and (y - 1) in yc.index:
                rows.append(dict(index=code, year=y,
                                 ret=float(yc[y] / yc[y - 1] - 1.0)))
    st = pd.DataFrame(rows)
    wide = st.pivot(index="year", columns="index", values="ret")
    wide.columns.name = None
    if "000300" in wide.columns and "000852" in wide.columns:
        wide["big_minus_small"] = wide["000300"] - wide["000852"]
    if "000300" in wide.columns and "000905" in wide.columns:
        wide["big_minus_mid"] = wide["000300"] - wide["000905"]
    return wide.reset_index()


def sample_dates() -> list[pd.Timestamp]:
    d = pd.read_parquet(os.path.join(CACHE, "000001_daily.parquet"))[["date"]]
    d["ym"] = d["date"].dt.to_period("M")
    last = d.groupby("ym")["date"].max()
    return [pd.Timestamp(x) for x in last.values]


def mkt_worker(code: str, samp: np.ndarray) -> list[dict]:
    p = os.path.join(QFQ, f"{code}.parquet")
    if not os.path.exists(p):
        return []
    q = pd.read_parquet(p)
    d = pd.to_datetime(q["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    c = q["close"].to_numpy(float)
    v = q["volume"].to_numpy(float)
    n = c.size
    if n < 21:
        return []
    adv = v * c
    out = []
    for ts in samp:
        i = int(np.searchsorted(d, np.int64(ts), side="right") - 1)
        if i < 19 or i + FWD >= n:
            continue
        if pd.Timestamp(d[i]).normalize() != pd.Timestamp(ts).normalize():
            continue
        a20 = float(np.nanmean(adv[i - 19:i + 1]))
        if not np.isfinite(a20) or a20 <= 0:
            continue
        r = float(c[i + FWD] / c[i] - 1.0)
        if not np.isfinite(r):
            continue
        out.append(dict(date=str(pd.Timestamp(ts).date()), code=code,
                        liq=float(np.log(a20)), ret=r))
    return out


def market_liq() -> tuple[pd.DataFrame, pd.DataFrame]:
    samp = sample_dates()
    log(f"抽样日 {len(samp)} 个：{samp[0].date()} → {samp[-1].date()}")
    arr = np.array([np.int64(x.value) for x in samp], dtype=np.int64)

    recs = []
    files = [f[:-8] for f in os.listdir(QFQ) if f.endswith(".parquet")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(mkt_worker, f, arr) for f in files]
        done = 0
        for fut in as_completed(futs):
            recs.extend(fut.result())
            done += 1
            if done % 1000 == 0 or done == len(futs):
                log(f"  {done}/{len(futs)} {time.time()-t0:.0f}s")
    s = pd.DataFrame(recs)
    log(f"全市场样本点 {len(s)}")

    rows, qrows = [], []
    for d0, g in s.groupby("date"):
        if len(g) < MIN_STOCKS_DAY:
            continue
        ic = sp(g["liq"], g["ret"])
        r = g["liq"].rank(method="first")
        b = np.ceil(r / len(g) * 5).clip(1, 5).astype(int)
        q = {k: float(g.loc[b == k, "ret"].mean()) for k in range(1, 6)}
        rows.append(dict(date=d0, year=int(d0[:4]), n=len(g), ic=ic,
                         q1=q[1], q5=q[5], spread=q[5] - q[1]))
        qrows.append(dict(date=d0, **{f"q{k}": q[k] for k in range(1, 6)}))
    dd = pd.DataFrame(rows)
    yr = dd.groupby("year").agg(n_days=("ic", "size"), ic=("ic", "mean"),
                                t=("ic", lambda x: float(x.mean() / (x.std(ddof=1) / np.sqrt(x.size))
                                                          if x.size > 1 and x.std(ddof=1) > 0 else np.nan)),
                                spread=("spread", "mean"),
                                win_share=("ic", lambda x: float((x > 0).mean()))
                                ).reset_index()
    return yr, pd.DataFrame(qrows)


def main() -> None:
    global t0
    t0 = time.time()
    yd, bd = b_signal_yearly()
    yd.to_csv(os.path.join(OUT3, "p3a_year_ab.csv"), index=False, encoding="utf-8")
    bd.to_csv(os.path.join(OUT3, "p3a_board.csv"), index=False, encoding="utf-8")
    st = index_style()
    st.to_csv(os.path.join(OUT3, "p3a_style.csv"), index=False, encoding="utf-8")
    dm = demeaned_by_seg()
    dm.to_csv(os.path.join(OUT3, "p3a_seg_demeaned.csv"), index=False, encoding="utf-8")

    log("全市场抽样开始")
    my, mq = market_liq()
    my.to_csv(os.path.join(OUT3, "p3a_market_ic.csv"), index=False, encoding="utf-8")
    mq.to_csv(os.path.join(OUT3, "p3a_market_quintile.csv"), index=False, encoding="utf-8")

    # 逐年对齐：A−E、经济风格、全市场 liq 效应
    mym = my.rename(columns={"ic": "ic_mkt", "t": "t_mkt", "spread": "spread_mkt"})
    al = yd.merge(st[["year", "000300", "000852", "big_minus_small"]],
                  on="year", how="left")
    al = al.merge(mym[["year", "ic_mkt", "t_mkt", "spread_mkt"]],
                  on="year", how="left")
    corr_style = float(pd.Series(al["diff"]).corr(pd.Series(al["big_minus_small"])))
    corr_mkt = float(pd.Series(al["diff"]).corr(pd.Series(al["ic_mkt"])))

    summary = dict(
        n_years_b=len(yd),
        years_diff_pos=[int(y) for y in yd.loc[yd["diff"] > 0, "year"]],
        years_diff_neg=[int(y) for y in yd.loc[yd["diff"] <= 0, "year"]],
        corr_b_diff_vs_300_minus_1000=corr_style,
        corr_b_diff_vs_market_liq_ic=corr_mkt,
        market_liq_yearly=my.to_dict(orient="records"),
    )
    with open(os.path.join(OUT3, "p3a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    print("\n=== B 信号 A−E 逐年 ===")
    print(yd.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print("\n=== 指数年度收益 ===")
    print(st.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print("\n=== 板块 ===")
    print(bd.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print("\n=== 分段去均值 ===")
    print(dm.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\n=== 全市场 liq 效应（月度抽样）===")
    print(my.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"\n相关: A−E vs (300−1000) = {corr_style:.3f};  A−E vs 全市场 liq IC = {corr_mkt:.3f}")
    log("完成")


if __name__ == "__main__":
    t0 = time.time()
    main()
