"""EP-02c：B 类交易持仓期解剖（只描述，不模拟止损）

只回答三个问题：
  Q1 盈利兑现时间：MFE 出现在第几 bar，多少 bar 才能兑现主要盈利空间
  Q2 亏损演化：MAE 分布、首次跌破阈值的比例与时点、破位后继续恶化还是修复
  Q3 亏损恢复能力：不同浮亏程度下，最终恢复盈利的概率

口径：
  入场价 = 入场日开盘（qfq）；持有 60 bar（bar 0..59），bar 59 收盘卖出（与 L1 一致）。
  入场的信号集合与 p3_features.csv 相同（11,419 笔）。
  "盈利" 主口径用 net（= r60q - base 成本），毛收益口径同时给出。
  浮亏以入场开盘价为基准的 qfq 相对价计算；同时给出收盘口径（可确认）与盘中口径（极值）。
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out", "entry_phase_02")
QFQ = os.path.join(BASE, "cache", "qfq")
SIG = os.path.join(BASE, "out", "entry_phase_03", "p3_features.csv")
HOLD = 60
BREACH_TH = [4, 6, 8, 10]
POST_TH = [4, 6, 8]
SNAP_T = [5, 10, 15, 20, 30, 40]
SEGS = ["train", "validate", "test"]


def load_signals() -> pd.DataFrame:
    df = pd.read_csv(SIG, dtype={"code": str})
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["cost_total"] = df["r60q"] - df["net60"]
    return df


def worker(args):
    code, g = args
    p = os.path.join(QFQ, f"{code}.parquet")
    if not os.path.exists(p):
        return [{"code": code, "signal_date": sd, "seg": seg, "bad": "no_file"}
                for sd, seg in zip(g["signal_date"], g["seg"])]
    d = pd.read_parquet(p)
    dates = d["date"].values.astype("datetime64[D]")
    o = d["open"].values.astype(float)
    h = d["high"].values.astype(float)
    lo = d["low"].values.astype(float)
    c = d["close"].values.astype(float)
    n = dates.size
    pos = {dd: i for i, dd in enumerate(dates)}
    rows = []
    for sd, ed, seg, g0, nt in zip(g["signal_date"], g["entry_date"],
                                   g["seg"], g["r60q"], g["net60"]):
        di = pos.get(np.datetime64(ed, "D"))
        if di is None:
            rows.append({"code": code, "signal_date": sd, "seg": seg, "bad": "no_entry_bar"})
            continue
        if di + HOLD - 1 >= n:
            rows.append({"code": code, "signal_date": sd, "seg": seg, "bad": "short"})
            continue
        qe = o[di]
        if not np.isfinite(qe) or qe <= 0:
            rows.append({"code": code, "signal_date": sd, "seg": seg, "bad": "bad_price"})
            continue
        k = slice(di, di + HOLD)
        cr = c[k] / qe - 1.0
        hr = h[k] / qe - 1.0
        lr = lo[k] / qe - 1.0
        gross = float(cr[-1])
        mfe = float(hr.max())
        mfe_i = int(np.argmax(hr))
        mae = float(lr.min())
        mae_i = int(np.argmin(lr))
        cmfe = float(cr.max())
        rec = {
            "code": code, "signal_date": sd, "seg": seg, "bad": "",
            "gross": gross, "net": float(nt), "cost": float(g0 - nt),
            "mfe": mfe, "mfe_bar": mfe_i, "mae": mae, "mae_bar": mae_i,
            "cmfe": cmfe,
            "t50": int(np.argmax(hr >= 0.5 * mfe)) if mfe > 0 else -1,
            "t80": int(np.argmax(hr >= 0.8 * mfe)) if mfe > 0 else -1,
            "cmfe_by20": float(cr[:21].max()),
            "cmfe_by30": float(cr[:31].max()),
            "cmfe_by40": float(cr[:41].max()),
            "capture": float(gross / mfe) if mfe > 0 else np.nan,
            "final_pos": int(gross > 0),
            "net_pos": int(nt > 0),
        }
        for th in BREACH_TH:
            cbr = cr <= -th / 100.0
            lbr = lr <= -th / 100.0
            rec[f"breach_c{th}"] = int(np.argmax(cbr)) if cbr.any() else -1
            rec[f"breach_l{th}"] = int(np.argmax(lbr)) if lbr.any() else -1
        for th in POST_TH:
            b = rec[f"breach_c{th}"]
            if b < 0:
                rec[f"pb{th}_r0"] = np.nan
                rec[f"pb{th}_r5"] = np.nan
                rec[f"pb{th}_r10"] = np.nan
                rec[f"pb{th}_min"] = np.nan
                rec[f"pb{th}_max"] = np.nan
                rec[f"pb{th}_deepen2"] = np.nan
                rec[f"pb{th}_recover2"] = np.nan
                continue
            rec[f"pb{th}_r0"] = float(cr[b])
            rec[f"pb{th}_r5"] = float(cr[b + 5]) if b + 5 < HOLD else np.nan
            rec[f"pb{th}_r10"] = float(cr[b + 10]) if b + 10 < HOLD else np.nan
            rec[f"pb{th}_min"] = float(lr[b:].min())
            rec[f"pb{th}_max"] = float(hr[b:].max())
            rec[f"pb{th}_deepen2"] = int(float(lr[b:].min()) <= float(cr[b]) - 0.02)
            rec[f"pb{th}_recover2"] = int(float(hr[b:].max()) >= float(cr[b]) + 0.02)
        for t in SNAP_T:
            rec[f"r{t}"] = float(cr[t])
        rows.append(rec)
    return rows


def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else np.nan


def main():
    sig = load_signals()
    print(f"signals={len(sig)}  codes={sig['code'].nunique()}")
    groups = list(sig.groupby("code", sort=False))
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(worker, groups):
            out.extend(rows)
    raw = pd.DataFrame(out)
    bad = raw["bad"].fillna("") != ""
    bad_reasons = raw.loc[bad, "bad"].value_counts().to_dict()
    print(f"bad={int(bad.sum())} reasons={bad_reasons}")
    tr = raw.loc[~bad].copy()
    tr["signal_date"] = pd.to_datetime(tr["signal_date"])
    tr["seg"] = tr["seg"].astype(str)
    print(f"usable={len(tr)}")

    tr.to_csv(os.path.join(OUT, "p2c_trades.csv"), index=False, encoding="utf-8")

    summary = {"n_signals": int(len(sig)), "n_usable": int(len(tr)),
               "n_bad": int(bad.sum()), "bad_reasons": {k: int(v) for k, v in bad_reasons.items()}}

    # ---------------- Q1 ----------------
    q1_rows = []
    for seg in ["ALL"] + SEGS:
        g = tr if seg == "ALL" else tr[tr["seg"] == seg]
        w = g[g["net"] > 0]
        q1_rows.append({
            "seg": seg, "n": int(len(g)), "n_win": int(len(w)),
            "win_rate": float((g["net"] > 0).mean()),
            "net_med_pct": 100 * float(g["net"].median()),
            "mfe_med_pct": 100 * float(g["mfe"].median()),
            "mfe_p25_pct": 100 * q(g["mfe"], 25),
            "mfe_p75_pct": 100 * q(g["mfe"], 75),
            "mfe_bar_med": float(g["mfe_bar"].median()),
            "mfe_bar_p25": float(g["mfe_bar"].quantile(.25)),
            "mfe_bar_p75": float(g["mfe_bar"].quantile(.75)),
            "t50_med_win": float(w["t50"].median()),
            "t80_med_win": float(w["t80"].median()),
            "t80_p25_win": float(w["t80"].quantile(.25)),
            "t80_p75_win": float(w["t80"].quantile(.75)),
            "cmfe_by20_med_win": float(w["cmfe_by20"].median()) * 100,
            "cmfe_by30_med_win": float(w["cmfe_by30"].median()) * 100,
            "cmfe_by40_med_win": float(w["cmfe_by40"].median()) * 100,
            "capture_med_win": float(w["capture"].median()),
            "capture_mean_win": float(w["capture"].mean()),
        })
    q1 = pd.DataFrame(q1_rows)
    q1.to_csv(os.path.join(OUT, "p2c_q1_profit_timing.csv"), index=False, encoding="utf-8")
    print("\nQ1\n" + q1.round(3).to_string(index=False))

    edges = [-1, 5, 10, 20, 30, 40, 60]
    labels = ["0-5", "6-10", "11-20", "21-30", "31-40", "41-59"]
    dist = []
    for seg in ["ALL"] + SEGS:
        g = tr if seg == "ALL" else tr[tr["seg"] == seg]
        b = pd.cut(g["mfe_bar"], bins=edges, labels=labels)
        vc = b.value_counts().reindex(labels).fillna(0)
        for lb, cnt in vc.items():
            dist.append({"seg": seg, "mfe_bar_bucket": lb, "n": int(cnt),
                         "share": float(cnt / len(g))})
    pd.DataFrame(dist).to_csv(os.path.join(OUT, "p2c_q1_mfe_bar_dist.csv"),
                              index=False, encoding="utf-8")

    # ---------------- Q2 ----------------
    q2_rows = []
    for seg in ["ALL"] + SEGS:
        g = tr if seg == "ALL" else tr[tr["seg"] == seg]
        r = {"seg": seg, "n": int(len(g)),
             "mae_med_pct": 100 * float(g["mae"].median()),
             "mae_p25_pct": 100 * q(g["mae"], 25),
             "mae_p10_pct": 100 * q(g["mae"], 10),
             "mae_p05_pct": 100 * q(g["mae"], 5),
             "mae_p01_pct": 100 * q(g["mae"], 1),
             "mae_bar_med": float(g["mae_bar"].median())}
        for th in BREACH_TH:
            ec = g[f"breach_c{th}"] >= 0
            el = g[f"breach_l{th}"] >= 0
            r[f"ever_c{th}_share"] = float(ec.mean())
            r[f"ever_l{th}_share"] = float(el.mean())
            r[f"c{th}_bar_med"] = float(g.loc[ec, f"breach_c{th}"].median()) if ec.any() else np.nan
            r[f"c{th}_bar_p25"] = float(g.loc[ec, f"breach_c{th}"].quantile(.25)) if ec.any() else np.nan
            r[f"c{th}_bar_p75"] = float(g.loc[ec, f"breach_c{th}"].quantile(.75)) if ec.any() else np.nan
            if ec.any():
                sub = g.loc[ec]
                base = -th / 100.0
                r[f"c{th}_end_worse"] = float((sub["gross"] < base).mean())
                r[f"c{th}_end_up2"] = float((sub["gross"] > base + 0.02).mean())
                r[f"c{th}_end_pos"] = float((sub["gross"] > 0).mean())
                r[f"c{th}_end_net_pos"] = float((sub["net"] > 0).mean())
                r[f"c{th}_end_med_pct"] = 100 * float(sub["gross"].median())
            else:
                for k in ["end_worse", "end_up2", "end_pos", "end_net_pos", "end_med_pct"]:
                    r[f"c{th}_{k}"] = np.nan
        q2_rows.append(r)
    q2 = pd.DataFrame(q2_rows)
    q2.to_csv(os.path.join(OUT, "p2c_q2_loss_evolution.csv"), index=False, encoding="utf-8")
    print("\nQ2\n" + q2.round(3).T.to_string())

    pbs = []
    for th in POST_TH:
        for seg in ["ALL"] + SEGS:
            g = tr if seg == "ALL" else tr[tr["seg"] == seg]
            s = g[g[f"breach_c{th}"] >= 0]
            if not len(s):
                continue
            pbs.append({
                "x": th, "seg": seg, "n": int(len(s)),
                "at_breach_med_pct": 100 * float(s[f"pb{th}_r0"].median()),
                "post5_med_pct": 100 * float(s[f"pb{th}_r5"].median()) if s[f"pb{th}_r5"].notna().any() else np.nan,
                "post10_med_pct": 100 * float(s[f"pb{th}_r10"].median()) if s[f"pb{th}_r10"].notna().any() else np.nan,
                "exit_med_pct": 100 * float(s["gross"].median()),
                "deepen2_share": float(s[f"pb{th}_deepen2"].mean()),
                "recover2_share": float(s[f"pb{th}_recover2"].mean()),
                "post_min_med_pct": 100 * float(s[f"pb{th}_min"].median()),
                "post_max_med_pct": 100 * float(s[f"pb{th}_max"].median()),
                "net_win_share": float((s["net"] > 0).mean()),
            })
    pb = pd.DataFrame(pbs)
    pb.to_csv(os.path.join(OUT, "p2c_q2_post_breach.csv"), index=False, encoding="utf-8")
    print("\nQ2 post-breach\n" + pb.round(3).to_string(index=False))

    # ---------------- Q3 ----------------
    rows = []
    for seg in ["ALL"] + SEGS:
        g = tr if seg == "ALL" else tr[tr["seg"] == seg]
        rows.append({"kind": "baseline", "seg": seg, "t": -1, "x": 0, "n": int(len(g)),
                     "share": 1.0, "p_net_win": float((g["net"] > 0).mean()),
                     "p_gross_win": float((g["gross"] > 0).mean()),
                     "net_med_pct": 100 * float(g["net"].median()),
                     "net_mean_pct": 100 * float(g["net"].mean())})
        for t in SNAP_T:
            for th in BREACH_TH:
                col = f"breach_c{th}"
                hit = (g[col] >= 0) & (g[col] <= t)
                n = int(hit.sum())
                rows.append({"kind": "cum_breach", "seg": seg, "t": t, "x": th, "n": n,
                             "share": float(hit.mean()),
                             "p_net_win": float((g.loc[hit, "net"] > 0).mean()) if n else np.nan,
                             "p_gross_win": float((g.loc[hit, "gross"] > 0).mean()) if n else np.nan,
                             "net_med_pct": 100 * float(g.loc[hit, "net"].median()) if n else np.nan,
                             "net_mean_pct": 100 * float(g.loc[hit, "net"].mean()) if n else np.nan})
                nev = ~hit
                rows.append({"kind": "cum_never", "seg": seg, "t": t, "x": th, "n": int(nev.sum()),
                             "share": float(nev.mean()),
                             "p_net_win": float((g.loc[nev, "net"] > 0).mean()) if nev.any() else np.nan,
                             "p_gross_win": float((g.loc[nev, "gross"] > 0).mean()) if nev.any() else np.nan,
                             "net_med_pct": 100 * float(g.loc[nev, "net"].median()) if nev.any() else np.nan,
                             "net_mean_pct": 100 * float(g.loc[nev, "net"].mean()) if nev.any() else np.nan})
    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(OUT, "p2c_q3_recovery_grid.csv"), index=False, encoding="utf-8")

    buckets = [-np.inf, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, np.inf]
    blab = ["<-10", "-10~-8", "-8~-6", "-6~-4", "-4~-2", "-2~0", ">=0"]
    frows = []
    for seg in ["ALL"] + SEGS:
        g = tr if seg == "ALL" else tr[tr["seg"] == seg]
        for t in SNAP_T:
            bk = pd.cut(g[f"r{t}"], bins=buckets, labels=blab, right=False)
            for lb in blab:
                s = g.loc[bk == lb]
                frows.append({"seg": seg, "t": t, "bucket": lb, "n": int(len(s)),
                              "share": float(len(s) / len(g)) if len(g) else np.nan,
                              "p_net_win": float((s["net"] > 0).mean()) if len(s) else np.nan,
                              "p_gross_win": float((s["gross"] > 0).mean()) if len(s) else np.nan,
                              "net_med_pct": 100 * float(s["net"].median()) if len(s) else np.nan})
    fl = pd.DataFrame(frows)
    fl.to_csv(os.path.join(OUT, "p2c_q3_float_bucket.csv"), index=False, encoding="utf-8")
    sel = fl[(fl["seg"] == "ALL") & (fl["t"].isin([10, 20, 30])) & (fl["bucket"] != ">=0")]
    print("\nQ3 float bucket (ALL, bar 10 / 20 / 30)\n" +
          sel.round(3).to_string(index=False))

    # ---------------- 逐年 ----------------
    yrows = []
    tr["year"] = tr["signal_date"].dt.year
    for y, g in tr.groupby("year"):
        b6 = g["breach_c6"] >= 0
        b8 = g["breach_c8"] >= 0
        yrows.append({"year": int(y), "n": int(len(g)),
                      "net_win": float((g["net"] > 0).mean()),
                      "mfe_bar_med": float(g["mfe_bar"].median()),
                      "t80_med_win": float(g.loc[g["net"] > 0, "t80"].median()),
                      "mae_med_pct": 100 * float(g["mae"].median()),
                      "c6_share": float(b6.mean()),
                      "c6_recover_net": float((g.loc[b6, "net"] > 0).mean()) if b6.any() else np.nan,
                      "c8_share": float(b8.mean()),
                      "c8_recover_net": float((g.loc[b8, "net"] > 0).mean()) if b8.any() else np.nan})
    yr = pd.DataFrame(yrows)
    yr.to_csv(os.path.join(OUT, "p2c_yearly.csv"), index=False, encoding="utf-8")
    print("\nyearly\n" + yr.round(3).to_string(index=False))

    # ---------------- summary ----------------
    w = tr[tr["net"] > 0]
    b6 = tr["breach_c6"] >= 0
    b8 = tr["breach_c8"] >= 0
    summary.update({
        "win_rate_net": float((tr["net"] > 0).mean()),
        "net_mean_pct": 100 * float(tr["net"].mean()),
        "net_med_pct": 100 * float(tr["net"].median()),
        "mfe_med_pct": 100 * float(tr["mfe"].median()),
        "mfe_bar_med": float(tr["mfe_bar"].median()),
        "t80_med_win": float(w["t80"].median()),
        "t50_med_win": float(w["t50"].median()),
        "capture_med_win": float(w["capture"].median()),
        "mae_med_pct": 100 * float(tr["mae"].median()),
        "mae_p05_pct": 100 * q(tr["mae"], 5),
        "breach_c6_share": float(b6.mean()),
        "breach_c6_net_win": float((tr.loc[b6, "net"] > 0).mean()),
        "breach_c8_share": float(b8.mean()),
        "breach_c8_net_win": float((tr.loc[b8, "net"] > 0).mean()),
        "breach_c6_by10_net_win": float((tr.loc[(tr["breach_c6"] >= 0) & (tr["breach_c6"] <= 10), "net"] > 0).mean()),
        "breach_c6_by20_net_win": float((tr.loc[(tr["breach_c6"] >= 0) & (tr["breach_c6"] <= 20), "net"] > 0).mean()),
    })
    with open(os.path.join(OUT, "p2c_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsummary\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    # 自检：重算 gross 必须等于 p3_features 的 r60q
    key = sig.set_index(["code", "signal_date"])["r60q"]
    idx = pd.MultiIndex.from_arrays([tr["code"], tr["signal_date"]])
    d = tr["gross"].values - key.reindex(idx).values
    print(f"\nselfcheck gross vs r60q: max|diff|={np.nanmax(np.abs(d)):.2e} "
          f"n_nan={int(np.isnan(d).sum())}")


if __name__ == "__main__":
    main()
