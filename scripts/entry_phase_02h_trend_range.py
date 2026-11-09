"""EP-02h 趋势结构与震荡环境下的 B 类信号表现差异（诊断实验）

只回答一个问题：B 类信号在「趋势」与「震荡」环境下的 net15 是否存在显著差异，
以及该差异能否排除年份集中（E2）与定义偏差（E3）。

- 环境标签一律取 signal_date 收盘及之前的数据。
- 固定 H=15，无止损/止盈/移动止损；不比较其他期限。
- 判定门槛见 out/entry_phase_02/EP02H_TREND_RANGE_PROTOCOL.md（v1.4，已冻结）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02_l1 as l1  # noqa: E402
import scripts.entry_phase_02d_stoploss as d  # noqa: E402

OUT = os.path.join(ROOT, "out", "entry_phase_02")
PROTO = os.path.join(OUT, "EP02H_TREND_RANGE_PROTOCOL.md")
PROTO_SHA = "B79DE0B84D9F777CBAF173B23C505ABD8283290F670C22A78DD2017EEFBD1CBD"

H = 15
N_ITER = 10_000
MIN_VALID = 9_500
SEED_MAIN, SEED_T6 = 20261108, 20261109
TOL_ZERO, TOL_NEUTRAL = 1e-6, 1e-12
T0_MIN_N, T0_MIN_MONTHS = 1_000, 40
T4_TOP_SHARE = 0.60
REPRO_NAV, REPRO_FILLS = 22_335_139.61910044, 672


def log(m: str) -> None:
    print(f"[EP02H] {m}", flush=True)


def verify_protocol() -> None:
    if not os.path.exists(PROTO):
        raise SystemExit("协议缺失，拒绝运行")
    h = hashlib.sha256(open(PROTO, "rb").read()).hexdigest().upper()
    if h != PROTO_SHA:
        raise SystemExit(f"协议已被修改（{h}），拒绝运行")
    log(f"协议校验通过 SHA256={h[:16]}…")


# ---------------------------------------------------------------- 环境指标
def er_series(cl: np.ndarray, n_win: int) -> tuple[np.ndarray, int]:
    """ER_N[t] = |close[t]-close[t-N]| / Σ|Δclose|；分母为 0 时取 0 并计数。"""
    out = np.full(cl.size, np.nan)
    if cl.size <= n_win:
        return out, 0
    dif = np.abs(np.diff(cl))
    cs = np.concatenate(([0.0], np.cumsum(dif)))
    num = np.abs(cl[n_win:] - cl[:-n_win])
    den = cs[n_win:] - cs[:-n_win]
    zero = int(np.sum(den == 0))
    out[n_win:] = np.where(den == 0, 0.0, num / np.where(den == 0, 1.0, den))
    return out, zero


def ret_series(cl: np.ndarray, n_win: int) -> np.ndarray:
    out = np.full(cl.size, np.nan)
    if cl.size > n_win:
        out[n_win:] = cl[n_win:] / cl[:-n_win] - 1.0
    return out


def label_er(er: np.ndarray, dr: np.ndarray, thr: float) -> np.ndarray:
    lab = np.full(er.size, "unknown", dtype=object)
    ok = np.isfinite(er) & np.isfinite(dr)
    lab[ok & (er < thr)] = "range"
    st = ok & (er >= thr)
    lab[st & (dr > TOL_NEUTRAL)] = "up"
    lab[st & (dr < -TOL_NEUTRAL)] = "down"
    lab[st & (np.abs(dr) <= TOL_NEUTRAL)] = "neutral"
    return lab


def label_g2g(r: np.ndarray) -> np.ndarray:
    """EP-02g 方向口径（S-c），无 neutral。"""
    lab = np.full(r.size, "unknown", dtype=object)
    ok = np.isfinite(r)
    lab[ok & (r >= 0.10)] = "trend"      # 牛
    lab[ok & (r <= -0.10)] = "trend"     # 熊
    lab[ok & (r > -0.10) & (r < 0.10)] = "range"
    return lab


def index_env(cal_ns: np.ndarray) -> dict:
    ix = pd.read_parquet(os.path.join(ROOT, "cache", "000001_daily.parquet"))
    ix["date"] = pd.to_datetime(ix["date"])
    s = ix.drop_duplicates("date").set_index("date")["close"].sort_index()
    cl = s.reindex(pd.to_datetime(cal_ns), method="ffill").to_numpy(float)
    er20, z20 = er_series(cl, 20)
    er60, z60 = er_series(cl, 60)
    er120, z120 = er_series(cl, 120)
    d20, d60, d120 = ret_series(cl, 20), ret_series(cl, 60), ret_series(cl, 120)
    r120 = ret_series(cl, 120)
    return dict(
        close=cl, er20=er20, er60=er60, er120=er120,
        d20=d20, d60=d60, d120=d120, r120=r120,
        lab_main=label_er(er60, d60, 0.30),
        lab_sa=label_er(er20, d20, 0.35),
        lab_sb=label_er(er120, d120, 0.25),
        lab_sc=label_g2g(r120),
        zero_denom=dict(er20=z20, er60=z60, er120=z120),
    )


# ---------------------------------------------------------------- 单股票
def prep(args) -> list[dict]:
    code, g = args
    p = os.path.join(ROOT, "cache", "qfq", f"{code}.parquet")
    if not os.path.exists(p):
        return []
    df = pd.read_parquet(p)
    dates = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)
    qo = df["open"].to_numpy(float)
    qc = df["close"].to_numpy(float)
    facs = l1.load_facs(code)
    if facs is None:
        f = np.ones(dates.size)
    else:
        da, fa = l1.factor_arrays(facs)
        f = fa[np.maximum(np.searchsorted(da, dates, side="right") - 1, 0)].astype(float)
    n = dates.size
    pos = {int(x): i for i, x in enumerate(dates)}
    rc = qc * f

    out = []
    for row in g.itertuples():
        sd = pd.Timestamp(row.signal_date)
        ed = pd.Timestamp(row.entry_date)
        di = pos.get(int(ed.value))
        # 与 EP-02g 完全同一信号池：入场后至少 60 根 bar 可用
        if di is None or di == 0 or di + 59 >= n:
            continue
        qe = qo[di]
        if not np.isfinite(qe) or qe <= 0:
            continue
        xi = di + H - 1
        raw_e = float(qe * f[di])
        t = l1.eval_trade(code, ed, pd.Timestamp(dates[xi]), raw_e, float(rc[xi]), "base")
        gross = (rc[xi] / f[xi]) / qe - 1.0          # qfq 空间收益
        cost = (rc[xi] / raw_e - 1.0) - t["net_return"]  # raw 空间成本
        net15 = float(gross - cost)

        seg = qc[di:di + H] / qe - 1.0
        if not np.isfinite(seg).all():
            continue
        hit = np.nonzero(qc[di:di + H] >= qe)[0]
        rec = int(hit[0]) + 1 if hit.size else H

        # 个股结构：signal_date 收盘（入场前）
        sj = pos.get(int(sd.value))
        if sj is None:
            sj = int(np.searchsorted(dates, int(sd.value), side="right")) - 1
        ser = np.nan
        if sj is not None and sj >= 20:
            den = float(np.abs(np.diff(qc[sj - 20:sj + 1])).sum())
            ser = 0.0 if den == 0 else float(abs(qc[sj] - qc[sj - 20]) / den)
        mpos = np.nan
        if sj is not None and sj >= 19:
            mpos = float(qc[sj] / np.mean(qc[sj - 19:sj + 1]) - 1.0)

        out.append(dict(
            code=code, signal_date=str(sd.date()), entry_date=str(ed.date()),
            raw_entry=raw_e, day_ns=dates[di:di + 400],
            rc_raw=rc[di:di + 400], f=f[di:di + 400],
            net15=net15, mae15=float(seg.min()), mfe15=float(seg.max()), rec15=rec,
            stock_er20=ser, ma20_pos=mpos,
            exits={"H15": dict(off=H - 1, day=int(dates[xi]), raw=float(rc[xi]),
                               reason=f"fixed_{H}_bar", delay=0, f=float(f[xi]))},
        ))
    return out


# ---------------------------------------------------------------- 统计
def metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return dict(n=0)
    v = sub["net15"].to_numpy(float)
    pos = v[v > 0]
    neg = v[v < 0]
    pf = float(pos.mean() / abs(neg.mean())) if neg.size and neg.mean() < 0 else np.nan
    return dict(
        n=int(n), mean=float(v.mean()), median=float(np.median(v)),
        win_rate=float((v > 0).mean()), profit_factor=pf,
        mae_abs=float(np.abs(sub["mae15"]).mean()), mfe=float(sub["mfe15"].mean()),
        rec_share=float((sub["rec15"] < H).mean()),
        rec_median=float(sub["rec15"].median()),
        net_per_day=float(v.mean() / H),
    )


def boot_setup(sl: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = sl[sl[col].isin(["range", "up", "down", "trend"])]
    months = np.array(sorted(sub["month"].unique()))
    mi = {m: i for i, m in enumerate(months)}
    m_arr = sub["month"].map(mi).to_numpy()
    vals = sub["net15"].to_numpy(float)
    is_r = (sub[col] == "range").to_numpy()
    M = months.size
    cnt = np.zeros((M, 2))
    ssum = np.zeros((M, 2))
    cnt[:, 0] = np.bincount(m_arr[is_r], minlength=M)
    ssum[:, 0] = np.bincount(m_arr[is_r], weights=vals[is_r], minlength=M)
    cnt[:, 1] = np.bincount(m_arr[~is_r], minlength=M)
    ssum[:, 1] = np.bincount(m_arr[~is_r], weights=vals[~is_r], minlength=M)
    return months, cnt, ssum


def boot_delta(cnt: np.ndarray, ssum: np.ndarray, m: int, U: np.ndarray) -> dict:
    deltas = np.full(U.shape[0], np.nan)
    for r in range(U.shape[0]):
        idx = np.floor(U[r, :m] * m).astype(np.int64)
        b = np.bincount(idx, minlength=m).astype(float)
        nr, dr = float(b @ ssum[:, 0]), float(b @ cnt[:, 0])
        nt, dt = float(b @ ssum[:, 1]), float(b @ cnt[:, 1])
        if dr > 0 and dt > 0:
            deltas[r] = nr / dr - nt / dt
    valid = int(np.isfinite(deltas).sum())
    if valid == 0:
        return dict(valid=0, lo=np.nan, hi=np.nan)
    lo, hi = np.nanpercentile(deltas, [5, 95])
    return dict(valid=valid, lo=float(lo), hi=float(hi))


def iid_delta(sl: pd.DataFrame, col: str) -> dict:
    """逐笔 iid 重采样，仅用于展示聚类修正幅度，不作判定依据。"""
    sub = sl[sl[col].isin(["range", "up", "down", "trend"])]
    vals = sub["net15"].to_numpy(float)
    is_r = (sub[col] == "range").to_numpy()
    N = vals.size
    if N == 0:
        return dict(valid=0, lo=np.nan, hi=np.nan)
    rng = np.random.default_rng(SEED_MAIN)
    p = np.full(N, 1.0 / N)
    out = np.full(N_ITER, np.nan)
    chunk = 500
    for s in range(0, N_ITER, chunk):
        c = min(chunk, N_ITER - s)
        cnt = rng.multinomial(N, p, size=c).astype(float)
        nr = cnt @ (vals * is_r)
        dr = cnt @ is_r.astype(float)
        nt = cnt @ (vals * ~is_r)
        dt = cnt @ (~is_r).astype(float)
        ok = (dr > 0) & (dt > 0)
        res = np.full(c, np.nan)
        res[ok] = nr[ok] / dr[ok] - nt[ok] / dt[ok]
        out[s:s + c] = res
    valid = int(np.isfinite(out).sum())
    if valid == 0:
        return dict(valid=0, lo=np.nan, hi=np.nan)
    lo, hi = np.nanpercentile(out, [5, 95])
    return dict(valid=valid, lo=float(lo), hi=float(hi))


def t0_ok(sub: pd.DataFrame) -> dict:
    r = sub[sub["env_main"] == "range"]
    t = sub[sub["env_main"].isin(["up", "down"])]
    return dict(
        range_n=int(len(r)), range_months=int(r["month"].nunique()),
        trend_n=int(len(t)), trend_months=int(t["month"].nunique()),
        ok=bool(len(r) >= T0_MIN_N and r["month"].nunique() >= T0_MIN_MONTHS
                and len(t) >= T0_MIN_N and t["month"].nunique() >= T0_MIN_MONTHS),
    )


def delta_of(sub: pd.DataFrame, col: str) -> float:
    r = sub.loc[sub[col] == "range", "net15"]
    t = sub.loc[sub[col].isin(["up", "down", "trend"]), "net15"]
    if len(r) == 0 or len(t) == 0:
        return np.nan
    return float(r.mean() - t.mean())


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    verify_protocol()
    ep.init_ctx()
    cal_ns = pd.to_datetime(ep.CTX["cal"]).values.astype("datetime64[ns]").astype(np.int64)
    cal_pos = {pd.Timestamp(k): v for k, v in ep.CTX["cal_pos"].items()}
    days = pd.to_datetime(cal_ns)

    env = index_env(cal_ns)
    pd.DataFrame(dict(
        date=days.date, close=env["close"],
        er20=env["er20"], d20=env["d20"], er60=env["er60"], d60=env["d60"],
        er120=env["er120"], d120=env["d120"],
        label_main=env["lab_main"], label_sa=env["lab_sa"],
        label_sb=env["lab_sb"], label_sc=env["lab_sc"],
    )).to_csv(os.path.join(OUT, "p2h_env_daily.csv"), index=False, encoding="utf-8")
    log(f"环境序列 {days.size} 行；分母为零 er20/60/120 = "
        f"{env['zero_denom']['er20']}/{env['zero_denom']['er60']}/{env['zero_denom']['er120']}")

    sig = d.load_signals()
    groups = list(sig.groupby("code", sort=False))
    wins: list[dict] = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, part in enumerate(ex.map(prep, groups), 1):
            wins.extend(part)
            if i % 1000 == 0:
                log(f"{i}/{len(groups)} {time.time() - t0:.0f}s")
    log(f"信号 {len(wins)} 笔（协议门槛 11419）")
    if len(wins) != 11419:
        raise SystemExit(f"信号池不一致：{len(wins)}")

    # ---- 组合级一致性核对 ----
    pf = d.run_main_portfolio(wins, cal_ns, cal_pos, "H15", "base")
    dev = abs(pf["final_nav"] - REPRO_NAV) / REPRO_NAV
    log(f"H15 NAV {pf['final_nav']:,.2f} 成交 {len(pf['trades'])} 偏差 {dev:.2e}")
    if dev > 1e-9 or len(pf["trades"]) != REPRO_FILLS:
        raise SystemExit(f"H15 复现未通过：偏差 {dev:.2e}，成交 {len(pf['trades'])}")

    # ---- 信号级标签 ----
    def cal_of(sd: str) -> tuple[int | None, bool]:
        ts = pd.Timestamp(sd)
        g = cal_pos.get(ts)
        if g is not None:
            return int(g), False
        v = int(np.searchsorted(cal_ns, ts.value, side="right")) - 1
        return (v, True) if v >= 0 else (None, True)

    rows, unmapped = [], 0
    for s in wins:
        g, bad = cal_of(s["signal_date"])
        unmapped += int(bad)
        if g is None:
            continue
        sd = pd.Timestamp(s["signal_date"])
        ed = pd.Timestamp(s["entry_date"])
        rows.append(dict(
            code=s["code"], signal_date=s["signal_date"], entry_date=s["entry_date"],
            month=sd.strftime("%Y-%m"), year=int(ed.year),
            env_main=env["lab_main"][g], env_sa=env["lab_sa"][g],
            env_sb=env["lab_sb"][g], env_sc=env["lab_sc"][g],
            net15=s["net15"], mae15=s["mae15"], mfe15=s["mfe15"], rec15=s["rec15"],
            stock_er20=s["stock_er20"], ma20_pos=s["ma20_pos"],
        ))
    sl = pd.DataFrame(rows)
    sl.to_csv(os.path.join(OUT, "p2h_signal_level.csv"), index=False, encoding="utf-8")

    # ---- 边界计数 ----
    bnd = []
    for nm, col, neut in (("main", "env_main", True), ("S-a", "env_sa", True),
                          ("S-b", "env_sb", True), ("S-c", "env_sc", False)):
        z = sl[col]
        bnd.append(dict(definition=nm, total=len(sl),
                        neutral=int((z == "neutral").sum()) if neut else 0,
                        unknown=int((z == "unknown").sum()),
                        er_zero_denom=env["zero_denom"]["er60" if nm == "main" else
                                                          "er20" if nm == "S-a" else "er120"],
                        unmapped_signal_date=unmapped,
                        ma20_pos_missing=int(sl["ma20_pos"].isna().sum()),
                        stock_er20_missing=int(sl["stock_er20"].isna().sum())))
    pd.DataFrame(bnd).to_csv(os.path.join(OUT, "p2h_boundary_counts.csv"),
                             index=False, encoding="utf-8")

    # ---- 各环境指标 ----
    mrows = []
    for nm, col in (("main", "env_main"), ("S-a", "env_sa"),
                    ("S-b", "env_sb"), ("S-c", "env_sc")):
        for lab, z in sl.groupby(col):
            mrows.append(dict(definition=nm, env=lab, **metrics(z)))
    pd.DataFrame(mrows).to_csv(os.path.join(OUT, "p2h_env_metrics.csv"),
                               index=False, encoding="utf-8")

    # ---- 年份 × 环境 ----
    yrows = [dict(year=int(y), env=k, n=len(z), pnl=float(z["net15"].sum()),
                  mean=float(z["net15"].mean()), win_rate=float((z["net15"] > 0).mean()))
             for (y, k), z in sl.groupby(["year", "env_main"])]
    pd.DataFrame(yrows).sort_values(["year", "env"]).to_csv(
        os.path.join(OUT, "p2h_year.csv"), index=False, encoding="utf-8")

    # ---- Bootstrap：主定义 + 三个敏感性 ----
    months_main = np.array(sorted(sl["month"].unique()))
    U = np.random.default_rng(SEED_MAIN).random((N_ITER, months_main.size))
    bt_rows, sens_rows = [], []
    boot = {}
    for nm, col in (("main", "env_main"), ("S-a", "env_sa"),
                    ("S-b", "env_sb"), ("S-c", "env_sc")):
        months, cnt, ssum = boot_setup(sl, col)
        m = months.size
        b = boot_delta(cnt, ssum, m, U) if m else dict(valid=0, lo=np.nan, hi=np.nan)
        iid = iid_delta(sl, col)
        delta = delta_of(sl, col)
        eligible = (b["valid"] >= MIN_VALID and
                    len(sl[sl[col] == "range"]) >= T0_MIN_N and
                    sl.loc[sl[col] == "range", "month"].nunique() >= T0_MIN_MONTHS and
                    len(sl[sl[col].isin(["up", "down", "trend"])]) >= T0_MIN_N and
                    sl.loc[sl[col].isin(["up", "down", "trend"]), "month"].nunique() >= T0_MIN_MONTHS)
        boot[nm] = dict(delta=delta, **b, eligible=bool(eligible))
        bt_rows.append(dict(definition=nm, delta=delta, ci_lo_cluster=b["lo"],
                            ci_hi_cluster=b["hi"], valid_cluster=b["valid"],
                            ci_lo_iid=iid["lo"], ci_hi_iid=iid["hi"], valid_iid=iid["valid"],
                            months=m, seed=SEED_MAIN))
        sens_rows.append(dict(
            definition=nm,
            n_valid=int((sl[col] != "unknown").sum()),
            n_unknown=int((sl[col] == "unknown").sum()),
            n_range=int((sl[col] == "range").sum()),
            n_trend=int(sl[col].isin(["up", "down", "trend"]).sum()),
            months_valid=m,
            months_range=int(sl.loc[sl[col] == "range", "month"].nunique()),
            months_trend=int(sl.loc[sl[col].isin(["up", "down", "trend"]), "month"].nunique()),
            boot_valid=b["valid"], t1_eligible=bool(eligible)))
    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT, "p2h_sensitivity_samples.csv"),
                                   index=False, encoding="utf-8")

    # ---- T4：年份集中 ----
    yr = sl[sl["env_main"] == "range"].groupby("year")["net15"].sum().sort_values(ascending=False)
    p_range = float(yr.sum())
    if p_range <= 0:
        t4, t4_reason = "NOT_APPLICABLE", "P_NEG"
        t4_detail = dict(top3=np.nan, share=np.nan)
    elif int((yr > 0).sum()) < 3:
        t4, t4_reason = "NOT_APPLICABLE", "FEW_POS_YEARS"
        t4_detail = dict(top3=np.nan, share=np.nan)
    else:
        top3 = float(yr.iloc[:3].sum())
        share = top3 / p_range
        max_year = int(yr.index[0])
        sub2 = sl[sl["year"] != max_year].copy()
        d2 = delta_of(sub2, "env_main")
        months2, cnt2, ssum2 = boot_setup(sub2, "env_main")
        b2 = boot_delta(cnt2, ssum2, months2.size, U) if months2.size else dict(valid=0, lo=np.nan, hi=np.nan)
        if b2["valid"] < MIN_VALID:
            t4, t4_reason = "NOT_APPLICABLE", "INSUFFICIENT_SAMPLE"
        else:
            t4_reason = ""
            ok4 = (top3 > 0) and (abs(share) <= T4_TOP_SHARE) and (d2 > TOL_ZERO) and (b2["lo"] > 0)
            t4 = "PASS" if ok4 else "FAIL"
        t4_detail = dict(top3=top3, share=float(share), max_year=max_year,
                         delta_ex_max=d2, ci_lo_ex_max=b2["lo"], ci_hi_ex_max=b2["hi"],
                         valid_ex_max=b2["valid"])
    bt_rows.append(dict(definition="T4_ex_max_year", delta=t4_detail.get("delta_ex_max", np.nan),
                        ci_lo_cluster=t4_detail.get("ci_lo_ex_max", np.nan),
                        ci_hi_cluster=t4_detail.get("ci_hi_ex_max", np.nan),
                        valid_cluster=t4_detail.get("valid_ex_max", 0),
                        ci_lo_iid=np.nan, ci_hi_iid=np.nan, valid_iid=np.nan,
                        months=np.nan, seed=SEED_MAIN))
    pd.DataFrame(bt_rows).to_csv(os.path.join(OUT, "p2h_bootstrap.csv"),
                                 index=False, encoding="utf-8")

    # ---- T6：震荡内个股结构 ----
    rng = sl[(sl["env_main"] == "range") & sl["stock_er20"].notna()].copy()
    lt_rows = []
    t6 = dict(delta=np.nan, lo=np.nan, hi=np.nan, valid=0, ok=False, months=0)
    if len(rng) >= 30:
        q1, q2 = np.nanquantile(rng["stock_er20"], [1 / 3, 2 / 3])
        rng["grp"] = np.where(rng["stock_er20"] >= q2, "high",
                              np.where(rng["stock_er20"] <= q1, "low", "mid"))
        for grp, z in rng.groupby("grp"):
            lt_rows.append(dict(group=grp, n=len(z), months=int(z["month"].nunique()),
                                mean=float(z["net15"].mean()),
                                win_rate=float((z["net15"] > 0).mean())))
        months6 = np.array(sorted(rng["month"].unique()))
        mi = {mm: i for i, mm in enumerate(months6)}
        m_arr = rng["month"].map(mi).to_numpy()
        M = months6.size
        hi = (rng["grp"] == "high").to_numpy()
        lo = (rng["grp"] == "low").to_numpy()
        v = rng["net15"].to_numpy(float)
        ch, sh = np.bincount(m_arr[hi], minlength=M), np.bincount(m_arr[hi], weights=v[hi], minlength=M)
        cl_, sl_ = np.bincount(m_arr[lo], minlength=M), np.bincount(m_arr[lo], weights=v[lo], minlength=M)
        U6 = np.random.default_rng(SEED_T6).random((N_ITER, M))
        d6 = np.full(N_ITER, np.nan)
        for r in range(N_ITER):
            idx = np.floor(U6[r] * M).astype(np.int64)
            b = np.bincount(idx, minlength=M).astype(float)
            nh, dh = float(b @ sh), float(b @ ch)
            nl, dl = float(b @ sl_), float(b @ cl_)
            if dh > 0 and dl > 0:
                d6[r] = nh / dh - nl / dl
        valid6 = int(np.isfinite(d6).sum())
        if valid6:
            lo6, hi6 = np.nanpercentile(d6, [5, 95])
            t6 = dict(delta=float(d6[np.isfinite(d6)].mean()), lo=float(lo6), hi=float(hi6),
                      valid=valid6, months=int(M),
                      ok=bool(valid6 >= MIN_VALID and lo6 > 0), high_n=int(hi.sum()),
                      low_n=int(lo.sum()), q1=float(q1), q2=float(q2))
    pd.DataFrame(lt_rows).to_csv(os.path.join(OUT, "p2h_local_trend.csv"),
                                 index=False, encoding="utf-8")

    # ---- 组合级按环境（一致性核对）----
    env_by_key = {(r["code"], r["entry_date"]): r["env_main"] for _, r in sl.iterrows()}
    tr = pf["trades"].copy()

    def _key(code, ed) -> tuple[str, str]:
        return str(code).zfill(6), pd.Timestamp(ed).strftime("%Y-%m-%d")

    tr["env"] = [env_by_key.get(_key(r.code, r.entry_date), "unknown") for r in tr.itertuples()]
    prow = [dict(env=k, n=len(z), pnl=float(z["pnl"].sum()),
                 avg_ret=float(z["ret"].mean()), win_rate=float((z["ret"] > 0).mean()))
            for k, z in tr.groupby("env")]
    pd.DataFrame(prow).to_csv(os.path.join(OUT, "p2h_portfolio.csv"),
                              index=False, encoding="utf-8")

    # ---- 判定 ----
    t0d = t0_ok(sl)
    dm = boot["main"]
    t1_signs = [np.sign(boot[k]["delta"]) for k in ("S-a", "S-b", "S-c")
                if boot[k]["eligible"] and np.isfinite(boot[k]["delta"])
                and abs(boot[k]["delta"]) > TOL_ZERO]
    t1 = bool(abs(dm["delta"]) > TOL_ZERO and np.isfinite(dm["delta"])
              and sum(1 for s in t1_signs if s == np.sign(dm["delta"])) >= 2)
    m_main = sl[sl["env_main"] == "range"]
    m_trend = sl[sl["env_main"].isin(["up", "down"])]
    mm, mt = metrics(m_main), metrics(m_trend)
    t2 = bool(dm["delta"] > 0 and dm["lo"] > 0 and mm.get("win_rate", 0) > mt.get("win_rate", 0))
    t3 = bool(mm.get("mae_abs", np.inf) <= mt.get("mae_abs", -np.inf)
              and (not np.isfinite(mm.get("profit_factor", np.nan))
                   or mm.get("profit_factor", 0) >= mt.get("profit_factor", 0)))
    t5 = bool(m_main["year"].nunique() >= 6)
    insufficient = bool((not t0d["ok"]) or (not t5) or dm["valid"] < MIN_VALID
                        or t4_reason == "INSUFFICIENT_SAMPLE")

    if insufficient:
        label = "INSUFFICIENT_SAMPLE"
    elif t4 == "FAIL":
        label = "YEAR_CONCENTRATION"
    elif t4 == "NOT_APPLICABLE":
        label = f"CONCENTRATION_NOT_TESTABLE (T4:{t4_reason})"
    elif not t1:
        label = "DEFINITION_ARTIFACT"
    elif dm["delta"] < 0 and t2 is False and dm["lo"] < 0:
        label = "TREND_EDGE_REAL"
    elif not t2:
        label = "NO_ENVIRONMENT_EDGE"
    elif not t3:
        label = "NO_ENVIRONMENT_EDGE (T3)"
    elif t6["ok"]:
        label = "RANGE_LOCAL_TREND"
    else:
        label = "RANGE_EDGE_REAL"

    summary = dict(
        protocol_sha256=PROTO_SHA, phase="EP-02h", label=label,
        gates=dict(T0=t0d, T1=t1, T2=t2, T3=t3, T4=t4, T4_reason=t4_reason,
                   T5=t5, T6=t6),
        samples=dict(signals=int(len(sl)), unmapped_signal_date=unmapped,
                     months=int(sl["month"].nunique())),
        years_range=int(m_main["year"].nunique()),
        main=dict(delta=float(dm["delta"]), ci_lo=dm["lo"], ci_hi=dm["hi"],
                  valid=dm["valid"], seed=SEED_MAIN,
                  range=mm, trend=mt, up=metrics(sl[sl["env_main"] == "up"]),
                  down=metrics(sl[sl["env_main"] == "down"]),
                  neutral=metrics(sl[sl["env_main"] == "neutral"])),
        sensitivity={k: dict(delta=float(v["delta"]), lo=v["lo"], hi=v["hi"],
                             valid=v["valid"], eligible=v["eligible"])
                     for k, v in boot.items() if k != "main"},
        concentration=dict(p_range=p_range, result=t4, reason=t4_reason, **t4_detail,
                           by_year={int(k): float(v) for k, v in yr.items()}),
        local_trend=t6,
        portfolio=dict(final_nav=pf["final_nav"], fills=len(pf["trades"]),
                       mdd=pf["mdd"], reproduce_dev=dev),
        zero_denom=env["zero_denom"],
    )
    with open(os.path.join(OUT, "p2h_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    write_report(summary, pd.DataFrame(mrows), pd.DataFrame(bt_rows),
                 pd.DataFrame(yrows), pd.DataFrame(lt_rows), a.workers)
    log(f"结论标签 {label}；Δ={dm['delta']:+.4%} CI[{dm['lo']:+.4%}, {dm['hi']:+.4%}] "
        f"完成 {time.time() - t0:.0f}s")


def _pct(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x*100:+.2f}%"


def write_report(s: dict, env_m: pd.DataFrame, boot: pd.DataFrame,
                 year: pd.DataFrame, lt: pd.DataFrame, workers: int) -> None:
    m = s["main"]
    g = s["gates"]
    L = []
    L.append("# EP-02h 趋势结构与震荡环境下的 B 类信号（诊断报告）\n")
    L.append(f"- 协议：`EP02H_TREND_RANGE_PROTOCOL.md` v1.4，SHA256 `{PROTO_SHA[:16]}…`")
    L.append(f"- 结论标签：**{s['label']}**")
    L.append(f"- 信号 {s['samples']['signals']:,} 笔，月份 {s['samples']['months']} 个；"
             f"signal_date 未在日历中的映射异常 {s['samples']['unmapped_signal_date']} 笔")
    L.append(f"- 组合级 H15 复现：NAV {s['portfolio']['final_nav']:,.2f}（偏差 "
             f"{s['portfolio']['reproduce_dev']:.2e}），{s['portfolio']['fills']} 笔，"
             f"MDD {s['portfolio']['mdd']:.2%}\n")
    L.append("## 1. 主结果（信号级，等权）\n")
    L.append("| 环境 | 笔数 | 单笔均值 | 中位数 | 胜率 | 盈亏比 | \\|MAE\\| | 15 日内恢复比例 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in ("range", "up", "down", "neutral"):
        x = m.get(k) or {}
        if not x.get("n"):
            continue
        L.append(f"| {k} | {x['n']:,} | {_pct(x['mean'])} | {_pct(x['median'])} | "
                 f"{x['win_rate']:.2%} | {x['profit_factor']:.3f} | "
                 f"{x['mae_abs']*100:.2f}% | {x['rec_share']:.2%} |")
    t = m["trend"]
    L.append(f"| trend | {t['n']:,} | {_pct(t['mean'])} | {_pct(t['median'])} | "
             f"{t['win_rate']:.2%} | {t['profit_factor']:.3f} | {t['mae_abs']*100:.2f}% | "
             f"{t['rec_share']:.2%} |")
    L.append("")
    L.append(f"**Δ = range − trend = {_pct(m['delta'])}**，月度聚类 90% CI "
             f"[{_pct(m['ci_lo'])}, {_pct(m['ci_hi'])}]，有效迭代 {m['valid']:,}/10,000。\n")

    L.append("## 2. 判定闸门\n")
    L.append("| 闸门 | 结果 | 说明 |")
    L.append("|---|---|---|")
    L.append(f"| T0 样本量 | {'PASS' if g['T0']['ok'] else 'FAIL'} | "
             f"range {g['T0']['range_n']:,} 笔 / {g['T0']['range_months']} 月；"
             f"trend {g['T0']['trend_n']:,} 笔 / {g['T0']['trend_months']} 月 |")
    L.append(f"| T1 定义稳健 | {'PASS' if g['T1'] else 'FAIL'} | "
             f"S-a {_pct(s['sensitivity']['S-a']['delta'])}、"
             f"S-b {_pct(s['sensitivity']['S-b']['delta'])}、"
             f"S-c {_pct(s['sensitivity']['S-c']['delta'])} |")
    L.append(f"| T2 收益/胜率 | {'PASS' if g['T2'] else 'FAIL'} | "
             f"Δ {_pct(m['delta'])}，CI 下界 {_pct(m['ci_lo'])}；胜率 "
             f"{m['range']['win_rate']:.2%} vs {t['win_rate']:.2%} |")
    L.append(f"| T3 风险不劣化 | {'PASS' if g['T3'] else 'FAIL'} | "
             f"\\|MAE\\| {m['range']['mae_abs']*100:.2f}% vs {t['mae_abs']*100:.2f}%；"
             f"盈亏比 {m['range']['profit_factor']:.3f} vs {t['profit_factor']:.3f} |")
    share = s["concentration"].get("share", float("nan"))
    share_txt = "n/a" if not np.isfinite(share) else f"前 3 年占比 {share*100:.1f}%"
    L.append(f"| T4 年份集中 | {g['T4']} | {g['T4_reason'] or share_txt} |")
    L.append(f"| T5 年度覆盖 | {'PASS' if g['T5'] else 'FAIL'} | "
             f"range 覆盖 {s['years_range']} 个自然年 |")
    L.append(f"| T6 局部趋势 | {'PASS' if g['T6'].get('ok') else 'FAIL'} | "
             f"高−低 {_pct(g['T6'].get('delta', float('nan')))}，"
             f"CI 下界 {_pct(g['T6'].get('lo', float('nan')))}，"
             f"有效 {g['T6'].get('valid', 0):,} |")
    L.append("")

    L.append("## 3. Bootstrap 明细\n")
    L.append("| 定义 | Δ | 聚类 CI 下界 | 聚类 CI 上界 | 有效次数 | iid CI 下界 | iid CI 上界 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in boot.itertuples():
        L.append(f"| {r.definition} | {_pct(r.delta)} | {_pct(r.ci_lo_cluster)} | "
                 f"{_pct(r.ci_hi_cluster)} | {r.valid_cluster:,} | "
                 f"{_pct(r.ci_lo_iid)} | {_pct(r.ci_hi_iid)} |")
    L.append("")

    L.append("## 4. 震荡市内部个股结构（T6）\n")
    if len(lt):
        L.append("| 组 | 笔数 | 月份 | 单笔均值 | 胜率 |")
        L.append("|---|---|---|---|---|")
        for r in lt.itertuples():
            L.append(f"| {r.group} | {r.n:,} | {r.months} | {_pct(r.mean)} | {r.win_rate:.2%} |")
    L.append("")

    L.append("## 5. 年份分布\n")
    if len(year):
        piv = year.pivot_table(index="year", columns="env", values="pnl", aggfunc="sum").fillna(0)
        L.append("| 年份 | " + " | ".join(piv.columns) + " |")
        L.append("|---" * (len(piv.columns) + 1) + "|")
        for y, row in piv.iterrows():
            L.append(f"| {int(y)} | " + " | ".join(f"{v/1:,.0f}" for v in row) + " |")
    L.append("")

    L.append("## 6. 局限\n")
    L.append("- 环境标签有滞后；转折附近归类不稳定。")
    L.append("- 指数环境 ≠ 个股环境；个股结构只作为机制证据。")
    L.append("- 月度聚类 Bootstrap 只保留同月相关，不保留跨月连续相关。")
    L.append("- 15 日是 EP-02g 的样本内候选，沿用它是为了固定变量，不代表已验证。")
    L.append("- 组合级归因受 NAV 路径放大影响，只作核对。")
    L.append("- 全部结论来自同一样本，不构成样本外证据；本报告是诊断，不是 EP-02 的存续裁决。")
    with open(os.path.join(OUT, "P2H_TREND_RANGE_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
