# -*- coding: utf-8 -*-
"""EP-03 专项：liq_log（成交额档位）的可交易性 / 冲击成本分析。

只回答三问：
  Q1 不同 ADV 分档的 B 信号，未来 60 bar 收益是否存在稳定差异？
  Q2 5 万元下单，买卖价差、冲击、费用各是多少？
  Q3 扣除真实估计成本后，liq_log 是否仍有正向区分能力？

不产生资金曲线，不做组合，不做窗口敏感性，不改策略。

成本模型（全部只用信号日及之前数据）：
  费用      复用 entry_phase_02_l1.eval_trade（分交易所/分年份的佣金、印花税、过户费），
            这是可精确计算的部分。
  价差(主)  1 个最小变动价位 = 0.01 元，半价差 = 0.005/价格（保守下限，A 股无 L1 数据）。
  价差(压力) Corwin-Schultz (2012) 高低价估计，取信号日前 20 个 2 日窗口均值。
            注意：该估计在 A 股会被日内波动放大，且实测不随 ADV 递减，只能当上限。
  冲击      square-root law：impact = c * sigma_d * sqrt(Q / ADV)，c ∈ {0.5, 1.0, 2.0}
            主口径 c = 1.0。

输出 out/entry_phase_03/：
  p3c_signals.csv       逐信号：ADV / 两套价差 / 冲击 / 费用 / 净收益
  p3c_buckets.csv       分档 × 分段：毛收益、成本拆解、净收益（主口径与压力口径）
  p3c_ic.csv            liq_log 对毛收益 / 净收益的同日横截面 IC（三段）
  p3c_yearly.csv        逐年 低档 vs 高档 毛差与净差
  p3c_summary.json      结构化结论（含盈亏平衡）

用法：python scripts/entry_phase_03_cost_stage.py
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

import scripts.entry_phase_02_l1 as l1  # noqa: E402

OUT3 = os.path.join(ROOT, "out", "entry_phase_03")
QFQ = os.path.join(ROOT, "cache", "qfq")

ORDER_YUAN = 50000.0
C_MAIN = 1.0
C_SET = (0.5, 1.0, 2.0)
MIN_TICK = 0.01
CS_WIN = 20
DAY_MIN_IC = 4
TRADING_DAYS = 252.0
CAP_RATES = (0.005, 0.01)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sp(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def cs_spread(h: np.ndarray, lo: np.ndarray, si: int, win: int) -> float:
    est = []
    k = 3.0 - 2.0 * np.sqrt(2.0)
    for t in range(max(1, si - win + 1), si + 1):
        if not (np.isfinite(h[t]) and np.isfinite(lo[t])
                and np.isfinite(h[t - 1]) and np.isfinite(lo[t - 1])
                and lo[t] > 0 and lo[t - 1] > 0):
            continue
        beta = np.log(h[t] / lo[t]) ** 2 + np.log(h[t - 1] / lo[t - 1]) ** 2
        hi, lw = max(h[t], h[t - 1]), min(lo[t], lo[t - 1])
        if lw <= 0:
            continue
        gamma = np.log(hi / lw) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        if alpha <= 0:
            est.append(0.0)
        else:
            e = np.exp(alpha)
            est.append(float(2.0 * (e - 1.0) / (1.0 + e)))
    return float(np.mean(est)) if est else np.nan


def worker(code: str, g: pd.DataFrame) -> list[dict]:
    p = os.path.join(QFQ, f"{code}.parquet")
    if not os.path.exists(p):
        return []
    q = pd.read_parquet(p)
    d = pd.to_datetime(q["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    o = q["open"].to_numpy(float)
    h = q["high"].to_numpy(float)
    lo = q["low"].to_numpy(float)
    c = q["close"].to_numpy(float)

    facs = l1.load_facs(code)
    if facs is None:
        return []
    darr, farr = l1.factor_arrays(facs)

    out = []
    for r in g.itertuples():
        sd = pd.to_datetime(r.signal_date)
        si = int(np.searchsorted(d, np.int64(sd.value), side="right") - 1)
        if si < 0 or pd.Timestamp(d[si]).normalize() != sd.normalize():
            continue
        ed = pd.to_datetime(r.entry_date)
        ei = int(np.searchsorted(d, np.int64(ed.value), side="right") - 1)
        if ei < 0 or pd.Timestamp(d[ei]).normalize() != ed.normalize():
            continue
        xi = ei + 59
        if xi >= c.size:
            continue
        qe, qx = float(o[ei]), float(c[xi])
        if not (np.isfinite(qe) and qe > 0 and np.isfinite(qx) and qx > 0):
            continue
        de, dx = pd.Timestamp(d[ei]), pd.Timestamp(d[xi])
        raw_e = qe * l1.f_at(darr, farr, int(de.value))
        raw_x = qx * l1.f_at(darr, farr, int(dx.value))
        if not (np.isfinite(raw_e) and raw_e > 0
                and np.isfinite(raw_x) and raw_x > 0):
            continue

        t = l1.eval_trade(code, de, dx, raw_e, raw_x, "base")
        fees = t["comm"] + t["stamp"] + t["transfer"]

        adv = float(np.exp(r.liq_log))
        px = float(c[si])
        s_tick = MIN_TICK / px if px > 0 else np.nan
        s_cs = cs_spread(h, lo, si, CS_WIN)
        s_cs = float(s_cs) if np.isfinite(s_cs) else 0.0
        s_stress = max(s_cs, s_tick) if np.isfinite(s_tick) else s_cs
        half_tick = 0.5 * s_tick if np.isfinite(s_tick) else 0.0

        sig_d = float(r.vol20) / np.sqrt(TRADING_DAYS) if np.isfinite(r.vol20) else np.nan
        part = ORDER_YUAN / adv if adv > 0 else np.nan
        imp = C_MAIN * sig_d * np.sqrt(part) if (np.isfinite(part) and np.isfinite(sig_d)) else np.nan

        rec = dict(code=code, signal_date=str(sd.date()), entry_date=str(ed.date()),
                   seg=r.seg, liq_log=float(r.liq_log), adv=adv, price=px,
                   sigma_d=sig_d, participation=part,
                   gross=float(r.r60q), fees=fees,
                   spread_tick=s_tick, spread_cs=s_cs, spread_stress=s_stress,
                   cost_tick=2.0 * half_tick,
                   cost_cs=1.0 * s_stress,
                   impact=2.0 * imp)
        rec["net_main"] = rec["gross"] - fees - rec["cost_tick"] - rec["impact"]
        rec["net_stress"] = rec["gross"] - fees - rec["cost_cs"] - rec["impact"]
        for cc in C_SET:
            rec[f"net_c{cc}"] = rec["gross"] - fees - rec["cost_cs"] - cc * 2.0 * (
                sig_d * np.sqrt(part) if (np.isfinite(part) and np.isfinite(sig_d)) else np.nan)
        out.append(rec)
    return out


def daily_ic(df: pd.DataFrame, col: str) -> dict:
    ics = []
    for _d, g in df.groupby("signal_date"):
        g = g.dropna(subset=["liq_log", col])
        if len(g) < DAY_MIN_IC:
            continue
        v = sp(g["liq_log"], g[col])
        if np.isfinite(v):
            ics.append(v)
    a = np.array(ics)
    if a.size < 2:
        return dict(n_days=int(a.size), ic=np.nan, t=np.nan, pos_share=np.nan)
    t = float(a.mean() / (a.std(ddof=1) / np.sqrt(a.size))) if a.std(ddof=1) > 0 else np.nan
    return dict(n_days=int(a.size), ic=float(a.mean()), t=t, pos_share=float((a > 0).mean()))


def qs(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return dict(n=0)
    pos, neg = x[x > 0], x[x <= 0]
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()),
                win=float(len(pos) / len(x)),
                payoff=float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else np.nan)


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(os.path.join(OUT3, "p3_features.csv"), dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m-%d")
    log(f"信号 {len(df)}")

    recs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(worker, c, g) for c, g in df.groupby("code")]
        done = 0
        for fut in as_completed(futs):
            recs.extend(fut.result())
            done += 1
            if done % 1000 == 0 or done == len(futs):
                log(f"  {done}/{len(futs)} {time.time()-t0:.0f}s")
    s = pd.DataFrame(recs)
    log(f"完成 {len(s)} / {len(df)} 信号")
    s.to_csv(os.path.join(OUT3, "p3c_signals.csv"), index=False, encoding="utf-8")

    tr = s[s["seg"] == "train"]["liq_log"]
    cuts = [float(tr.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]
    names = ["A_最低", "B_低", "C_中", "D_高", "E_最高"]
    s["bucket"] = pd.cut(s["liq_log"], bins=[-np.inf, *cuts, np.inf], labels=names)
    log("分档 ADV 中位（万元）：" + ", ".join(
        f"{n}={np.exp(s.loc[s['bucket'] == n, 'liq_log']).median()/1e4:.0f}" for n in names))

    rows = []
    for seg in ("train", "validate", "test"):
        for n in names:
            b = s[(s["seg"] == seg) & (s["bucket"] == n)]
            if len(b) == 0:
                continue
            g1, m1, x1 = qs(b["gross"]), qs(b["net_main"]), qs(b["net_stress"])
            rows.append(dict(
                seg=seg, bucket=n, n=g1.get("n", 0),
                adv_median_wan=float(np.exp(b["liq_log"]).median() / 1e4),
                gross_mean=g1.get("mean", np.nan), gross_win=g1.get("win", np.nan),
                net_main_mean=m1.get("mean", np.nan), net_main_win=m1.get("win", np.nan),
                net_main_payoff=m1.get("payoff", np.nan),
                net_stress_mean=x1.get("mean", np.nan), net_stress_win=x1.get("win", np.nan),
                fees_bp=float(b["fees"].mean() * 1e4),
                spread_tick_bp=float(b["cost_tick"].mean() * 1e4),
                impact_bp=float(b["impact"].mean() * 1e4),
                cost_main_bp=float((b["fees"] + b["cost_tick"] + b["impact"]).mean() * 1e4),
                spread_cs_bp=float(b["cost_cs"].mean() * 1e4),
                cost_stress_bp=float((b["fees"] + b["cost_cs"] + b["impact"]).mean() * 1e4),
                flat_cost_bp=36.0,
            ))
    bk = pd.DataFrame(rows)
    bk.to_csv(os.path.join(OUT3, "p3c_buckets.csv"), index=False, encoding="utf-8")

    ic_rows = []
    for seg in ("train", "validate", "test"):
        g = s[s["seg"] == seg]
        for col in ("gross", "net_main", "net_stress", "net_c0.5", "net_c2.0"):
            ic_rows.append(dict(seg=seg, target=col, **daily_ic(g, col)))
    ic = pd.DataFrame(ic_rows)
    ic.to_csv(os.path.join(OUT3, "p3c_ic.csv"), index=False, encoding="utf-8")

    yr = []
    s["year"] = pd.to_datetime(s["signal_date"]).dt.year
    for y, g in s.groupby("year"):
        a, e = g[g["bucket"] == "A_最低"], g[g["bucket"] == "E_最高"]
        if len(a) < 5 or len(e) < 5:
            continue
        yr.append(dict(year=int(y), n_a=int(len(a)), n_e=int(len(e)),
                       gross_diff=float(a["gross"].mean() - e["gross"].mean()),
                       net_main_diff=float(a["net_main"].mean() - e["net_main"].mean()),
                       net_stress_diff=float(a["net_stress"].mean() - e["net_stress"].mean())))
    ydf = pd.DataFrame(yr)
    ydf.to_csv(os.path.join(OUT3, "p3c_yearly.csv"), index=False, encoding="utf-8")

    cap = {str(r0): float((s["adv"] < ORDER_YUAN / r0).mean()) for r0 in CAP_RATES}

    # 盈亏平衡：低档相对高档的增量成本，要到多少才能吃掉毛收益优势
    be = {}
    for seg in ("train", "validate", "test"):
        g = bk[bk["seg"] == seg].set_index("bucket")
        if "A_最低" not in g.index or "E_最高" not in g.index:
            continue
        be[seg] = dict(
            gross_diff_pp=float(g.loc["A_最低", "gross_mean"] - g.loc["E_最高", "gross_mean"]) * 100,
            cost_diff_main_bp=float(g.loc["A_最低", "cost_main_bp"] - g.loc["E_最高", "cost_main_bp"]),
            cost_diff_stress_bp=float(g.loc["A_最低", "cost_stress_bp"] - g.loc["E_最高", "cost_stress_bp"]),
            net_diff_main_pp=float(g.loc["A_最低", "net_main_mean"] - g.loc["E_最高", "net_main_mean"]) * 100,
            net_diff_stress_pp=float(g.loc["A_最低", "net_stress_mean"] - g.loc["E_最高", "net_stress_mean"]) * 100,
            breakeven_extra_cost_bp=float(
                (g.loc["A_最低", "gross_mean"] - g.loc["E_最高", "gross_mean"])
                - (g.loc["A_最低", "cost_main_bp"] - g.loc["E_最高", "cost_main_bp"]) / 1e4 * 0),
        )
        be[seg]["breakeven_extra_cost_bp"] = float(
            (g.loc["A_最低", "gross_mean"] - g.loc["E_最高", "gross_mean"]) * 1e4
            - (g.loc["A_最低", "cost_main_bp"] - g.loc["E_最高", "cost_main_bp"]))

    # 下单规模敏感性：三持仓时每仓约 33 万（= 5 万的 6.6 倍）；冲击 ∝ sqrt(Q)
    sizing = []
    for Q in (50000.0, 100000.0, 330000.0):
        sc = np.sqrt(Q / ORDER_YUAN)
        t2 = s.assign(net_q=s["gross"] - s["fees"] - s["cost_tick"] - s["impact"] * sc)
        capq = {str(r0): float((s["adv"] < Q / r0).mean()) for r0 in CAP_RATES}
        row = dict(order_yuan=Q, impact_scale=sc, capacity_share=capq)
        for seg in ("train", "validate", "test"):
            g = t2[t2["seg"] == seg]
            a = g.loc[g["bucket"] == "A_最低", "net_q"].mean()
            e = g.loc[g["bucket"] == "E_最高", "net_q"].mean()
            row[f"net_A_{seg}"] = float(a)
            row[f"net_E_{seg}"] = float(e)
            row[f"diff_{seg}_pp"] = float(a - e) * 100
            icq = daily_ic(g, "net_q")
            row[f"ic_{seg}"] = icq["ic"]
            row[f"ic_t_{seg}"] = icq["t"]
        sizing.append(row)

    summary = dict(
        sizing=sizing,
        n_signals=int(len(s)), order_yuan=ORDER_YUAN,
        cuts_adv_wan=[float(np.exp(x) / 1e4) for x in cuts],
        bucket_adv_median_wan={n: float(np.exp(s.loc[s["bucket"] == n, "liq_log"].median()
                                              ) / 1e4) for n in names},
        capacity_share_constrained=cap,
        ic_table=ic.to_dict(orient="records"),
        yearly_n=int(len(ydf)),
        yearly_gross_pos=int((ydf["gross_diff"] > 0).sum()),
        yearly_net_main_pos=int((ydf["net_main_diff"] > 0).sum()),
        yearly_net_stress_pos=int((ydf["net_stress_diff"] > 0).sum()),
        breakeven=be,
    )
    with open(os.path.join(OUT3, "p3c_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    print("\n=== 分档 (test) ===")
    print(bk[bk["seg"] == "test"].to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print("\n=== IC ===")
    print(ic.drop(columns=["pos_share"]).to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\n容量受限比例:", {k: round(v * 100, 2) for k, v in cap.items()}, "%")
    print("逐年 A-E 毛差为正 %d/%d，净差(主)为正 %d/%d，净差(压力)为正 %d/%d"
          % (int((ydf["gross_diff"] > 0).sum()), len(ydf),
             int((ydf["net_main_diff"] > 0).sum()),
             int((ydf["net_stress_diff"] > 0).sum()),
             int((ydf["net_stress_diff"] > 0).sum()), len(ydf)))
    print("\n=== 盈亏平衡 ===")
    for k, v in be.items():
        print(k, {kk: round(vv, 2) for kk, vv in v.items()})
    print("\n=== 下单规模敏感性 ===")
    for r0 in sizing:
        print(f"Q={int(r0['order_yuan']):>7d} 容量受限={ {k: round(v*100,2) for k,v in r0['capacity_share'].items()} }%"
              f"  A-E差={r0['diff_train_pp']:.2f}/{r0['diff_validate_pp']:.2f}/{r0['diff_test_pp']:.2f}pp"
              f"  IC t={r0['ic_t_train']:.2f}/{r0['ic_t_validate']:.2f}/{r0['ic_t_test']:.2f}")
    log("完成")


if __name__ == "__main__":
    main()
