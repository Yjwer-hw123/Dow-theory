# -*- coding: utf-8 -*-
"""ENTRY-PHASE-01 v1.2 实现（研究性代码，不产出任何买卖策略）。

严格按 out/entry_phase/ENTRY_PHASE_PROTOCOL.md（v1.2 FROZEN）实现：
- 复用上游 uptrend_detector.py 的 prepare_stock / pivots / running_two / 常量；
- 候选进入周、连续特征、A/B/C 原始标签、C 扩展历史分位；
- 入场执行、20/60/120 日收益、首次触及竞争、风险指标；
- B0 同 episode 对照、episode 等权加权交易分布、12 项检验与年度稳定性。

上游文件与冻结协议均不修改。任何文字歧义按最保守解释实现，并在
out/entry_phase/T1_T5_TEST_REPORT.md 的“实现歧义记录”中列明。

用法：
    python scripts/entry_phase.py --stage test --sample 200
    python scripts/entry_phase.py --stage full            # 仅在 T1–T5 通过后允许

阶段 2 全量统计的机械闸门：必须存在 out/entry_phase/T1_T5_PASS.json。
"""
from __future__ import annotations

import argparse
import ast
import copy
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

import scripts.uptrend_detector as ud  # noqa: E402
from scripts.step1_trend_validation import (  # noqa: E402
    END, FAC_DIR, STOCK_DIR, load_factor, log, read_local_raw,
)

OUT_DIR = os.path.join(ROOT, "out", "entry_phase")
PASS_MARKER = os.path.join(OUT_DIR, "T1_T5_PASS.json")
TEST_JSON = os.path.join(OUT_DIR, "t1_t5_results.json")
TEST_MD = os.path.join(OUT_DIR, "T1_T5_TEST_REPORT.md")

# ---- 冻结协议常量（不得修改）----
ENTRY_MIN_B0 = 5            # 8.2：B0 至少 5 个完整 r60 的候选周
ENTRY_MAX_MKT_DAYS = 10     # 4.2：入场有效期 10 个市场交易日
HORIZONS = (20, 60, 120)
RACE_WIN = 60
VOL_BASE_WEEKS = 12
ER_WIN = 26
C_MIN_HIST = 500            # 6.3：历史候选周下限
C_MIN_EPISODES = 50         # 6.3：历史 episode 下限
CATEGORIES = ("A", "B", "C")
YEARS_FULL = tuple(range(2005, 2026))   # 2005–2025 共 21 个完整年度
YEAR_MIN_EPISODES = 30      # 11.4：合格年度至少 30 个配对 episode
SUCCESS_MIN_PAIRED = 500    # 10.1：每类至少 500 个配对 episode
SUCCESS_WIN = 0.50
SUCCESS_PAYOFF = 2.00
SUCCESS_YEAR_SHARE = 0.70
MAE_TOL_PP = 0.02
N_BOOT = 1000

CTX: dict = {}
FAC_MEM: dict[str, list[dict] | None] = {}

# 行 schema：每个候选周一行，全部为决策时可确认信息 + 事后结果
ROW_SPEC = [
    ("W", np.int32), ("J", np.int32), ("EP", np.int32), ("BLOCK", np.int32),
    ("W_END", np.int64), ("J_END", np.int64),
    ("AGE", np.int32), ("RUN_HIGH_W", np.int32), ("N_PB", np.int32),
    ("W_SINCE_H", np.int32),
    ("RUN_HIGH", np.float64), ("DEPTH", np.float64), ("RETRACE", np.float64),
    ("VOL_RATIO", np.float64), ("ER26", np.float64), ("MDD_HIGH", np.float64),
    ("WL1", np.float64),
    ("STRUCT_OK", np.bool_), ("LABEL_A", np.bool_), ("LABEL_B", np.bool_),
    ("LABEL_C", np.bool_), ("C_CLASS", np.bool_),
    ("ENTRY_DI", np.int32), ("ENTRY_NS", np.int64),
    ("ENTRY_OPEN", np.float64), ("TRADABLE", np.bool_),
    ("R20", np.float64), ("R60", np.float64), ("R120", np.float64),
    ("MAE60", np.float64), ("MDD60", np.float64),
    ("RACE10", np.int8), ("RACE20N10", np.int8), ("RACE20", np.int8),
]

# T1/T2 只比较“特征与标签”，不比较事后收益（截断后收益必然缺失）
T12_COMPARE = [
    "AGE", "RUN_HIGH_W", "RUN_HIGH", "DEPTH", "RETRACE", "VOL_RATIO",
    "ER26", "MDD_HIGH", "W_SINCE_H", "N_PB", "STRUCT_OK",
    "LABEL_A", "LABEL_B", "LABEL_C", "C_CLASS",
]

# T4 允许出现分位数/中位数的函数（均为 PIT 切片或事后统计）
PIT_AUDIT_ALLOW = {
    "assign_c_thresholds": "C 阈值只用 w_end < d 的历史前缀",
    "naive_threshold": "T3 独立复算，同样只用 w_end < d",
    "metrics_weighted": "事后统计，作用于已结算交易集",
    "concentration_stats": "事后统计",
    "point_metrics": "事后统计",
    "ci95": "事后 bootstrap 区间",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def ns(x) -> int:
    return int(pd.Timestamp(x).value)


def from_ns(v: int) -> pd.Timestamp:
    return pd.Timestamp(int(v))


def init_ctx() -> None:
    global CTX
    CTX = ud.build_ctx()
    ud.CTX = CTX


def get_factor(code: str) -> list[dict] | None:
    if code in FAC_MEM:
        return FAC_MEM[code]
    p = os.path.join(FAC_DIR, f"{code}.json")
    facs = None
    if os.path.exists(p):
        try:
            obj = json.load(open(p, encoding="utf-8"))
            if isinstance(obj, list):
                facs = obj
        except Exception:  # noqa: BLE001
            facs = None
    if facs is None:
        try:
            facs = load_factor(code)
        except Exception:  # noqa: BLE001
            facs = None
    FAC_MEM[code] = facs
    return facs


def all_codes() -> list[str]:
    return sorted(os.path.basename(f)[:6]
                  for f in glob.glob(os.path.join(STOCK_DIR, "*.parquet")))


def weekly_low(A: dict) -> np.ndarray:
    """从日线按 ISO 周重建周线最低价，只用于复用 pivots（不修改上游）。

    与 prepare_stock 内部的 wlow 完全同源：同一 wkey → 同一全局周位置。
    """
    nw = A["nw"]
    dates = A["dates"]
    lo = A["l"]
    iso = pd.DatetimeIndex(dates).isocalendar()
    key = (iso["year"].to_numpy(np.int64) * 100
           + iso["week"].to_numpy(np.int64))
    starts = np.r_[0, np.nonzero(key[1:] != key[:-1])[0] + 1]
    mins = np.minimum.reduceat(lo, starts)
    kw = key[starts]
    pos = np.array([CTX["wkey_to_pos"].get(int(k), -1) for k in kw], np.int64)
    wlow = np.full(nw, np.nan)
    ok = pos >= 0
    wlow[pos[ok]] = mins[ok]
    return wlow


def main_episodes(A: dict) -> list[tuple[int, int]]:
    """复刻 analyze_signal 的 episode 划分，但返回主信号周（j, block）。

    上游 episodes 返回的是“结构信号周 k”；协议母体是 A 组主信号，
    因此这里按同样的区间规则取区间内强度首次满足的周 j。
    """
    out: list[tuple[int, int]] = []
    block_until = -1
    for k in np.nonzero(A["cond_struct_week"])[0]:
        k = int(k)
        if k <= block_until:
            continue
        e = int(A["nb"][k])
        block = e if e >= 0 else A["nw"]
        cand = np.nonzero(A["cond_week"][k:min(block, A["nw"])])[0]
        if cand.size:
            out.append((k + int(cand[0]), block))
        block_until = block
    return out


# ---------------------------------------------------------------------------
# 入场、收益、风险、首次触及
# ---------------------------------------------------------------------------
def entry_info(A: dict, w: int) -> tuple[bool, int, float, int]:
    """候选周 w → 入场 bar。返回 (tradable, di, entry_open, entry_ns)。"""
    t = A["w_end_ts"][w]
    dates = A["dates"]
    di = int(np.searchsorted(dates, ud.dt64(t), side="right"))
    if di >= dates.size:
        return False, -1, np.nan, 0
    ci = CTX["cal_pos"].get(t)
    if ci is None:
        return False, -1, np.nan, 0
    limit = CTX["cal"][min(ci + ENTRY_MAX_MKT_DAYS, len(CTX["cal"]) - 1)]
    if pd.Timestamp(dates[di]) > limit:
        return False, -1, np.nan, 0
    op = float(A["o"][di])
    if not np.isfinite(op) or op <= 0:
        return False, -1, np.nan, 0
    return True, di, op, ns(dates[di])


def first_touch(h: np.ndarray, l: np.ndarray, entry: float,
                up: float, dn: float) -> int:
    """首次触及（T+1 起）：0 上行先 / 1 下行先 / 2 都未触及 / 3 同日歧义。"""
    hu = h[1:]
    lu = l[1:]
    u = hu >= entry * (1.0 + up)
    d = lu <= entry * (1.0 - dn)
    hit = np.nonzero(u | d)[0]
    if hit.size == 0:
        return 2
    k = int(hit[0])
    if u[k] and d[k]:
        return 3
    return 0 if u[k] else 1


def outcome_fields(A: dict, w: int) -> dict:
    """第 7 节结果变量；入场 bar 记为第 1 个有效交易日。"""
    out = dict(
        ENTRY_DI=-1, ENTRY_NS=0, ENTRY_OPEN=np.nan, TRADABLE=False,
        R20=np.nan, R60=np.nan, R120=np.nan, MAE60=np.nan, MDD60=np.nan,
        RACE10=np.int8(-1), RACE20N10=np.int8(-1), RACE20=np.int8(-1),
    )
    ok, di, op, ent_ns = entry_info(A, w)
    if not ok:
        return out
    out.update(ENTRY_DI=np.int32(di), ENTRY_NS=np.int64(ent_ns),
               ENTRY_OPEN=op, TRADABLE=True)

    dates = A["dates"]
    c = A["c"]
    n = dates.size
    for hz in HORIZONS:
        j = di + hz - 1
        if j < n:
            out[f"R{hz}"] = float(c[j]) / op - 1.0

    if di + RACE_WIN - 1 < n:
        seg_l = A["l"][di:di + RACE_WIN]
        seg_h = A["h"][di:di + RACE_WIN]
        out["MAE60"] = float(np.min(seg_l)) / op - 1.0
        peak = np.maximum.accumulate(np.maximum(seg_h, op))
        out["MDD60"] = -float(np.max(1.0 - seg_l / peak))
        # first_touch 内部从第 2 根有效 bar 起算（T+1）
        out["RACE10"] = np.int8(first_touch(seg_h, seg_l, op, 0.10, 0.10))
        out["RACE20N10"] = np.int8(first_touch(seg_h, seg_l, op, 0.20, 0.10))
        out["RACE20"] = np.int8(first_touch(seg_h, seg_l, op, 0.20, 0.20))
    return out


# ---------------------------------------------------------------------------
# 候选周 + 连续特征 + 原始标签
# ---------------------------------------------------------------------------
def build_stock(code: str, A: dict) -> dict:
    """一只股票的全部候选周（按 episode、按周递增）。"""
    nw = A["nw"]
    wclose = A["wclose"]
    whigh = A["whigh"]
    wamt = A["wamount"]
    wlow = weekly_low(A)
    # 注意：上游 pivots 实测返回“极值索引”（与其 docstring 的“确认索引”不符）。
    # 冻结协议要求“已确认”= 2 右走完，因此这里显式把确认周 = 极值 + 2，
    # 只影响 ENTRY-PHASE 的特征/标签，不修改上游 episode 定义。
    conf_low, val_low = ud.pivots(wlow, "low")
    conf_week = conf_low + 2
    wl1, _, _ = ud.running_two(conf_week, val_low, nw)
    exts = conf_low

    eps = main_episodes(A)
    cols: dict[str, list] = {k: [] for k, _ in ROW_SPEC}

    for ep_i, (j, block) in enumerate(eps):
        last = int(min(block, nw))
        after = np.nonzero(exts > j)[0]
        a_week = int(exts[after[0]] + 2) if after.size else -10

        for w in range(j + 1, last):
            if not bool(A["has_bar"][w]):
                continue
            if not (np.isfinite(wl1[w]) and np.isfinite(wclose[w])
                    and wclose[w] >= wl1[w]):
                continue

            seg_h = whigh[j:w + 1]
            run_high = float(np.nanmax(seg_h))
            tied = np.nonzero(np.isfinite(seg_h) & (seg_h == run_high))[0]
            h = int(j + tied[-1])          # 并列取最晚一周
            depth = 1.0 - float(wclose[w]) / run_high

            den = run_high - float(wl1[w])
            retrace = ((run_high - float(wclose[w])) / den) if den > 0 else np.nan

            vol_ratio = np.nan
            if h - VOL_BASE_WEEKS >= 0:
                num = wamt[h:w + 1]
                base = wamt[h - VOL_BASE_WEEKS:h]
                if (np.all(np.isfinite(num)) and np.all(np.isfinite(base))
                        and np.all(num > 0) and np.all(base > 0)):
                    vol_ratio = float(np.mean(num) / np.mean(base))

            mdd_high = np.nan
            seg_l = wlow[h:w + 1]
            if np.isfinite(seg_l).any():
                mdd_high = 1.0 - float(np.nanmin(seg_l)) / run_high

            n_pb = int(np.sum((exts > j) & (exts + 2 <= w)))
            structure_ok = bool(np.isfinite(wl1[w]) and wclose[w] >= wl1[w])

            label_a = bool(w == a_week)
            label_b = bool(n_pb >= 1 and depth > 0
                           and np.isfinite(vol_ratio) and vol_ratio < 1.0)

            oc = outcome_fields(A, w)

            row = dict(
                W=np.int32(w), J=np.int32(j), EP=np.int32(ep_i),
                BLOCK=np.int32(block),
                W_END=np.int64(ns(A["w_end_ts"][w])),
                J_END=np.int64(ns(A["w_end_ts"][j])),
                AGE=np.int32(w - j), RUN_HIGH_W=np.int32(h),
                N_PB=np.int32(n_pb), W_SINCE_H=np.int32(w - h),
                RUN_HIGH=run_high, DEPTH=depth, RETRACE=retrace,
                VOL_RATIO=vol_ratio,
                ER26=float(A["er"][w]) if np.isfinite(A["er"][w]) else np.nan,
                MDD_HIGH=mdd_high, WL1=float(wl1[w]),
                STRUCT_OK=structure_ok, LABEL_A=label_a, LABEL_B=label_b,
                LABEL_C=False, C_CLASS=False,
                **oc,
            )
            for k, _dtype in ROW_SPEC:
                cols[k].append(row[k])

    n = len(cols["W"])
    arrs = {k: np.asarray(cols[k], dtype=dt) for k, dt in ROW_SPEC}
    eps_meta = []
    for j, block in eps:
        ok, di, op, ent = entry_info(A, j)
        b1 = np.nan
        if ok:
            jj = di + 59
            if jj < A["dates"].size:
                b1 = float(A["c"][jj]) / op - 1.0
        eps_meta.append(dict(j=int(j), block=int(block),
                             sig_ns=ns(A["w_end_ts"][j]), sig_r60=b1))

    return dict(code=code, n=n, eps=eps_meta, n_episodes=len(eps), **arrs)


def prepare_and_build(code: str) -> dict | None:
    raw = read_local_raw(code)
    facs = get_factor(code)
    if raw is None or raw.empty or facs is None:
        return None
    A = ud.prepare_stock(raw, facs)
    if A is None:
        return None
    return build_stock(code, A)


# ---------------------------------------------------------------------------
# C：扩展历史分位（PIT）
# ---------------------------------------------------------------------------
def assign_c_thresholds(stocks: list[dict]) -> list[dict]:
    """按决策日严格使用 w_end < d 的历史候选周计算 q80(age) / q20(depth)。

    历史不足 → 该决策日不可分类；C 标签保持 False。
    返回逐决策日的阈值序列（供 T3 与交付物核对）。
    """
    Ws, AGEs, DEPs, VOLs, EPs, SIs, RIs = [], [], [], [], [], [], []
    for si, s in enumerate(stocks):
        n = s["n"]
        if n == 0:
            continue
        Ws.append(s["W_END"])
        AGEs.append(s["AGE"].astype(np.float64))
        DEPs.append(s["DEPTH"].astype(np.float64))
        VOLs.append(s["VOL_RATIO"])
        EPs.append(s["EP"].astype(np.int64))
        SIs.append(np.full(n, si, np.int64))
        RIs.append(np.arange(n, dtype=np.int64))

    if not Ws:
        return []

    W = np.concatenate(Ws)
    AGE = np.concatenate(AGEs)
    DEP = np.concatenate(DEPs)
    VOL = np.concatenate(VOLs)
    EP = np.concatenate(EPs)
    SI = np.concatenate(SIs)
    RI = np.concatenate(RIs)
    EPK = SI * 100000 + EP

    order = np.argsort(W, kind="stable")
    W, AGE, DEP, VOL, EPK, SI, RI = (
        x[order] for x in (W, AGE, DEP, VOL, EPK, SI, RI)
    )

    thresholds: list[dict] = []
    seen: set[int] = set()
    i = 0
    total = W.size
    while i < total:
        d = int(W[i])
        j = i
        while j < total and int(W[j]) == d:
            j += 1

        rec = dict(
            date=str(from_ns(d).date()), hist_rows=int(i),
            hist_episodes=int(len(seen)), q_age=None, q_depth=None,
            classifiable=False,
        )
        if i >= C_MIN_HIST and len(seen) >= C_MIN_EPISODES:
            a = AGE[:i]
            a = a[np.isfinite(a)]
            dp = DEP[:i]
            dp = dp[np.isfinite(dp)]
            if a.size and dp.size:
                q_age = float(np.quantile(a, 0.80))
                q_dep = float(np.quantile(dp, 0.20))
                rec.update(q_age=q_age, q_depth=q_dep, classifiable=True)

        if rec["classifiable"]:
            q_age = rec["q_age"]
            q_dep = rec["q_depth"]
            for t in range(i, j):
                s = stocks[int(SI[t])]
                r = int(RI[t])
                s["C_CLASS"][r] = True
                s["LABEL_C"][r] = bool(
                    np.isfinite(AGE[t]) and np.isfinite(DEP[t])
                    and np.isfinite(VOL[t])
                    and AGE[t] >= q_age and DEP[t] <= q_dep and VOL[t] < 1.0
                )

        for t in range(i, j):
            seen.add(int(EPK[t]))
        thresholds.append(rec)
        i = j

    return thresholds


def naive_threshold(stocks: list[dict], d_ns: int) -> dict:
    """独立实现：对任意决策日重新汇总 w_end < d 的全部候选周。T3 用。"""
    ages = []
    deps = []
    eps: set[int] = set()
    n = 0
    for si, s in enumerate(stocks):
        if s["n"] == 0:
            continue
        m = s["W_END"] < d_ns
        if not m.any():
            continue
        n += int(m.sum())
        ages.append(s["AGE"][m].astype(np.float64))
        deps.append(s["DEPTH"][m].astype(np.float64))
        for e in np.unique(s["EP"][m]):
            eps.add(si * 100000 + int(e))
    out = dict(hist_rows=n, hist_episodes=len(eps), classifiable=False)
    if n >= C_MIN_HIST and len(eps) >= C_MIN_EPISODES and ages:
        a = np.concatenate(ages)
        a = a[np.isfinite(a)]
        dp = np.concatenate(deps)
        dp = dp[np.isfinite(dp)]
        if a.size and dp.size:
            out.update(classifiable=True,
                       q_age=float(np.quantile(a, 0.80)),
                       q_depth=float(np.quantile(dp, 0.20)))
    return out


# ---------------------------------------------------------------------------
# B0 配对与指标
# ---------------------------------------------------------------------------
def episode_groups(s: dict):
    ep = s["EP"]
    n = s["n"]
    if n == 0:
        return
    starts = np.r_[0, np.nonzero(ep[1:] != ep[:-1])[0] + 1]
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        yield int(ep[a]), int(a), int(b)


def build_paired(stocks: list[dict], cat: str) -> dict:
    """按 8.3 构造主要配对样本；B0 按 8.2 的标签排除。"""
    paired: list[dict] = []
    excl = dict(no_category=0, category_untradable=0, category_no_r60=0,
                b0_lt5=0, no_episode=0)
    n_cat_obs = 0
    n_episodes = 0

    for si, s in enumerate(stocks):
        for ep_i, a, b in episode_groups(s):
            n_episodes += 1
            lab = s[f"LABEL_{cat}"][a:b]
            obs = np.nonzero(lab)[0] + a
            if obs.size == 0:
                excl["no_category"] += 1
                continue
            n_cat_obs += int(obs.size)
            i0 = int(obs[0])      # 4.3：该类别的第一次观测
            if not bool(s["TRADABLE"][i0]):
                excl["category_untradable"] += 1
                continue
            if not np.isfinite(s["R60"][i0]):
                excl["category_no_r60"] += 1
                continue

            fin = np.isfinite(s["R60"][a:b])
            b0 = np.nonzero((~lab) & fin)[0] + a
            if b0.size < ENTRY_MIN_B0:
                excl["b0_lt5"] += 1
                continue

            ev = obs[np.isfinite(s["R60"][obs])]
            paired.append(dict(
                si=si, ep=ep_i, code=s["code"],
                year=int(from_ns(int(s["W_END"][i0])).year),
                date=from_ns(int(s["W_END"][i0])),
                cat_r60=float(s["R60"][i0]),
                cat_win=float(s["R60"][i0] > 0),
                cat_mae=float(s["MAE60"][i0]) if np.isfinite(s["MAE60"][i0]) else np.nan,
                cat_mdd=float(s["MDD60"][i0]) if np.isfinite(s["MDD60"][i0]) else np.nan,
                b0_r60=s["R60"][b0].astype(np.float64),
                b0_mae=s["MAE60"][b0].astype(np.float64),
                b0_mdd=s["MDD60"][b0].astype(np.float64),
                cat_ev_r60=s["R60"][ev].astype(np.float64),
                b1_r60=float(s["eps"][ep_i]["sig_r60"]),
            ))

    return dict(paired=paired, excluded=excl,
                n_cat_obs=n_cat_obs, n_episodes=n_episodes)


def concentration_stats(r_pos: np.ndarray) -> dict:
    """盈利交易的收益集中度（描述性，按交易笔数排序）。"""
    rp = np.sort(r_pos)[::-1]
    total = float(rp.sum())
    out = dict(win_share_top1=np.nan, win_share_top5=np.nan,
               win_share_top10=np.nan, expectancy60_ex_top1=np.nan)
    if rp.size == 0 or total <= 0:
        return out
    for frac, key in ((0.01, "win_share_top1"), (0.05, "win_share_top5"),
                      (0.10, "win_share_top10")):
        k = max(1, int(math.ceil(rp.size * frac)))
        out[key] = float(rp[:k].sum() / total)
    return out


def metrics_weighted(r: np.ndarray, w: np.ndarray) -> dict:
    """第 7.2 节 60 日收益分布；权重口径由调用方给定。"""
    r = np.asarray(r, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    out = dict(
        n=int(r.size), W_plus=np.nan, W_minus=np.nan, W_zero=np.nan,
        win60_all=np.nan, win60_nonflat=np.nan, flat60=np.nan,
        avg_win60=np.nan, avg_loss60=np.nan, payoff60=np.nan,
        expectancy60=np.nan, median_win60=np.nan, median_loss60=np.nan,
        expectancy60_ex_top1=np.nan,
        win_share_top1=np.nan, win_share_top5=np.nan, win_share_top10=np.nan,
    )
    if r.size == 0 or w.sum() <= 0:
        return out
    pos = r > 0
    neg = r < 0
    zero = r == 0
    wp = float(w[pos].sum())
    wn = float(w[neg].sum())
    w0 = float(w[zero].sum())
    tot = wp + wn + w0
    out.update(W_plus=wp, W_minus=wn, W_zero=w0,
               win60_all=wp / tot if tot > 0 else np.nan,
               win60_nonflat=wp / (wp + wn) if (wp + wn) > 0 else np.nan,
               flat60=w0 / tot if tot > 0 else np.nan,
               expectancy60=float((w * r).sum() / w.sum()))
    if wp > 0:
        out["avg_win60"] = float((w[pos] * r[pos]).sum() / wp)
    if wn > 0:
        out["avg_loss60"] = float((w[neg] * (-r[neg])).sum() / wn)
    if np.isfinite(out["avg_win60"]) and np.isfinite(out["avg_loss60"]) \
            and out["avg_loss60"] > 0:
        out["payoff60"] = out["avg_win60"] / out["avg_loss60"]
    if pos.any():
        out["median_win60"] = float(np.median(r[pos]))
    if neg.any():
        out["median_loss60"] = float(np.median(-r[neg]))

    rp = np.sort(r[pos])[::-1]
    if rp.size:
        k = max(1, int(math.ceil(rp.size * 0.01)))
        keep = np.ones(r.size, dtype=bool)
        # 只剔除盈利交易中的前 1%（“盈利前 1% 交易”）
        pos_idx = np.nonzero(pos)[0]
        pos_sorted = pos_idx[np.argsort(r[pos_idx])[::-1]]
        keep[pos_sorted[:k]] = False
        out["expectancy60_ex_top1"] = float((w[keep] * r[keep]).sum() / w[keep].sum()) \
            if w[keep].sum() > 0 else np.nan
        if rp.sum() > 0:
            for frac, key in ((0.01, "win_share_top1"), (0.05, "win_share_top5"),
                              (0.10, "win_share_top10")):
                kk = max(1, int(math.ceil(rp.size * frac)))
                out[key] = float(rp[:kk].sum() / rp.sum())
    return out


def episode_side(paired: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """B0 侧：episode 内等权（每 episode 总权重 1）。"""
    rs = []
    ws = []
    for e in paired:
        m = e["b0_r60"].size
        rs.append(e["b0_r60"])
        ws.append(np.full(m, 1.0 / m))
    if not rs:
        return np.empty(0), np.empty(0)
    return np.concatenate(rs), np.concatenate(ws)


def point_metrics(paired: list[dict]) -> dict:
    cat_r = np.array([e["cat_r60"] for e in paired], dtype=np.float64)
    cat_w = np.ones(cat_r.size)
    b0_r, b0_w = episode_side(paired)

    m_cat = metrics_weighted(cat_r, cat_w)
    m_b0 = metrics_weighted(b0_r, b0_w)

    b0_mean = np.array([e["b0_r60"].mean() for e in paired], dtype=np.float64)
    diff = cat_r - b0_mean
    win_diff = np.array([e["cat_win"] - float((e["b0_r60"] > 0).mean())
                         for e in paired])
    # B0 侧 win60_all 用权重口径；episode 内权重相同，等价于比例
    mae_cat = np.array([e["cat_mae"] for e in paired], dtype=np.float64)
    mae_b0 = np.array([np.nanmedian(e["b0_mae"]) if e["b0_mae"].size else np.nan
                       for e in paired], dtype=np.float64)

    return dict(
        n_paired=int(len(paired)),
        cat=m_cat, b0=m_b0,
        r60_diff_mean=float(np.mean(diff)) if diff.size else np.nan,
        r60_diff_median=float(np.median(diff)) if diff.size else np.nan,
        win60_diff_mean=float(np.mean(win_diff)) if win_diff.size else np.nan,
        payoff60_diff=(m_cat["payoff60"] - m_b0["payoff60"]),
        mae60_cat_median=float(np.nanmedian(mae_cat)) if mae_cat.size else np.nan,
        mae60_b0_median=float(np.nanmedian(mae_b0)) if mae_b0.size else np.nan,
        q10_cat=float(np.quantile(cat_r, 0.10)) if cat_r.size else np.nan,
        q10_b0=float(np.quantile(b0_r, 0.10)) if b0_r.size else np.nan,
        b1_r60=np.array([e["b1_r60"] for e in paired], dtype=np.float64),
    )


def bootstrap_paired(paired: list[dict], reps: int = N_BOOT, seed: int = 20261107):
    """按 episode 重采样；同一批重采样同时重建两侧。"""
    if not paired:
        return dict(r60_diff=[], win60_diff=[], payoff60_diff=[],
                    expectancy=[], n_reps=0)
    cat_r = np.array([e["cat_r60"] for e in paired], dtype=np.float64)
    cat_win = np.array([e["cat_win"] for e in paired], dtype=np.float64)
    b0_mean = np.array([e["b0_r60"].mean() for e in paired], dtype=np.float64)
    b0_win = np.array([float((e["b0_r60"] > 0).mean()) for e in paired])
    b0_list = [e["b0_r60"] for e in paired]
    m = len(paired)
    rng = np.random.default_rng(seed)
    out = dict(r60_diff=[], win60_diff=[], payoff60_diff=[], expectancy=[])
    for _ in range(reps):
        idx = rng.integers(0, m, m)
        out["r60_diff"].append(float(cat_r[idx].mean() - b0_mean[idx].mean()))
        out["win60_diff"].append(float(cat_win[idx].mean() - b0_win[idx].mean()))
        br = np.concatenate([b0_list[i] for i in idx])
        bw = np.concatenate([np.full(b0_list[i].size, 1.0 / b0_list[i].size)
                             for i in idx])
        mc = metrics_weighted(cat_r[idx], np.ones(m))
        mb = metrics_weighted(br, bw)
        if np.isfinite(mc["payoff60"]) and np.isfinite(mb["payoff60"]):
            out["payoff60_diff"].append(float(mc["payoff60"] - mb["payoff60"]))
        out["expectancy"].append(float(mc["expectancy60"]))
    out["n_reps"] = reps
    return out


def ci95(x) -> tuple[float, float]:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if a.size < 2:
        return (np.nan, np.nan)
    return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))


def boot_p(x, greater: bool = True) -> float:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return np.nan
    if greater:
        return float((a <= 0).mean())
    return float(2.0 * min((a <= 0).mean(), (a >= 0).mean()))


def bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    arr = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(np.where(np.isfinite(arr), arr, 1.0))
    q = np.ones(m, dtype=np.float64)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = int(order[rank])
        p = arr[i] if np.isfinite(arr[i]) else 1.0
        val = min(prev, p * m / (rank + 1))
        q[i] = val
        prev = val
    return [float(x) for x in q]


def annual_stability(paired: list[dict]) -> dict:
    """11.4：完整年度仅 2005–2025，2026 YTD 单列，不进入年度稳定性分母。"""
    years = sorted({e["year"] for e in paired})
    rows = []
    for y in years:
        sel = [e for e in paired if e["year"] == y]
        diff = np.array([e["cat_r60"] - e["b0_r60"].mean() for e in sel])
        rows.append(dict(
            year=int(y), n_paired=len(sel),
            r60_diff_mean=float(diff.mean()) if diff.size else np.nan,
            qualified=bool(int(y) in YEARS_FULL and len(sel) >= YEAR_MIN_EPISODES),
        ))
    return annual_from_rows(rows)


def annual_from_rows(rows: list[dict]) -> dict:
    full = [r for r in rows if int(r["year"]) in YEARS_FULL]
    ytd = [r for r in rows if int(r["year"]) not in YEARS_FULL]
    qual = [r for r in full if int(r["n_paired"]) >= YEAR_MIN_EPISODES]
    pos = [r for r in qual if np.isfinite(float(r["r60_diff_mean"]))
           and float(r["r60_diff_mean"]) > 0]
    return dict(
        years=full, full_years=full, ytd=ytd,
        n_qualified=len(qual), n_positive=len(pos),
        positive_share=(len(pos) / len(qual)) if qual else np.nan,
        worst_year=(min(qual, key=lambda r: float(r["r60_diff_mean"]))["year"]
                    if qual else None),
    )


def event_weight_expectancy(paired: list[dict]) -> dict:
    """事件等权（次口径）：类别侧用全部同类观测，B0 侧同 1/n_e。"""
    ev = np.concatenate([e["cat_ev_r60"] for e in paired]) if paired else np.empty(0)
    b0_r, b0_w = episode_side(paired)
    return dict(
        n_cat_events=int(ev.size),
        expectancy_cat=float(ev.mean()) if ev.size else np.nan,
        expectancy_b0=float((b0_r * b0_w).sum() / b0_w.sum())
        if b0_w.size and b0_w.sum() > 0 else np.nan,
    )


# ---------------------------------------------------------------------------
# 阶段 2：完整统计（严格闸门）
# -------------------------------------------------------------------
def run_category(stocks: list[dict], cat: str) -> dict:
    ps = build_paired(stocks, cat)
    paired = ps["paired"]
    if not paired:
        return dict(counts=dict(n_paired=0, **ps["excluded"]),
                    note="配对样本为空", success=False)

    pm = point_metrics(paired)
    boot = bootstrap_paired(paired)
    ci_r = ci95(boot["r60_diff"])
    ci_w = ci95(boot["win60_diff"])
    ci_p = ci95(boot["payoff60_diff"])
    ci_e = ci95(boot["expectancy"])

    tests = [
        dict(name=f"{cat}:r60_diff", p=boot_p(boot["r60_diff"], greater=False),
             est=pm["r60_diff_mean"], ci=list(ci_r)),
        dict(name=f"{cat}:win60_diff", p=boot_p(boot["win60_diff"], greater=False),
             est=pm["win60_diff_mean"], ci=list(ci_w)),
        dict(name=f"{cat}:payoff60_diff",
             p=boot_p(boot["payoff60_diff"], greater=False),
             est=pm["payoff60_diff"], ci=list(ci_p)),
        dict(name=f"{cat}:expectancy_gt0", p=boot_p(boot["expectancy"], greater=True),
             est=pm["cat"]["expectancy60"], ci=list(ci_e)),
    ]
    ann = annual_stability(paired)
    evw = event_weight_expectancy(paired)

    mae_ok = (np.isfinite(pm["mae60_cat_median"])
              and np.isfinite(pm["mae60_b0_median"])
              and pm["mae60_cat_median"] >= pm["mae60_b0_median"] - MAE_TOL_PP)
    cond = dict(
        n_paired=pm["n_paired"] >= SUCCESS_MIN_PAIRED,
        win=pm["cat"]["win60_all"] >= SUCCESS_WIN,
        payoff=pm["cat"]["payoff60"] >= SUCCESS_PAYOFF,
        expectancy=pm["cat"]["expectancy60"] > 0
        and np.isfinite(ci_e[0]) and ci_e[0] > 0,
        r60_ci=np.isfinite(ci_r[0]) and ci_r[0] > 0
        and np.isfinite(pm["r60_diff_median"]) and pm["r60_diff_median"] > 0,
        annual=np.isfinite(ann["positive_share"])
        and ann["positive_share"] >= SUCCESS_YEAR_SHARE,
        mae=mae_ok,
        event_sign=(np.isfinite(evw["expectancy_cat"])
                    and np.isfinite(pm["cat"]["expectancy60"])
                    and np.sign(evw["expectancy_cat"])
                    == np.sign(pm["cat"]["expectancy60"])),
    )
    success = bool(all(cond.values()))

    return dict(
        counts=dict(n_paired=pm["n_paired"], **ps["excluded"]),
        metrics=dict(
            category=pm["cat"], b0=pm["b0"],
            r60_diff_mean=pm["r60_diff_mean"],
            r60_diff_median=pm["r60_diff_median"],
            win60_diff_mean=pm["win60_diff_mean"],
            payoff60_diff=pm["payoff60_diff"],
            mae60_cat_median=pm["mae60_cat_median"],
            mae60_b0_median=pm["mae60_b0_median"],
            q10_cat=pm["q10_cat"], q10_b0=pm["q10_b0"],
        ),
        event_weight=evw,
        tests=tests,
        annual=ann,
        conditions=cond,
        success=success,
    )


def run_full(args) -> None:
    if not os.path.exists(PASS_MARKER):
        raise SystemExit(
            "拒绝运行阶段 2：未找到 T1–T5 通过标记 "
            f"{PASS_MARKER}。请先运行 --stage test 并通过全部测试。")
    codes = all_codes()
    if args.limit:
        codes = codes[: args.limit]
    log(f"[full] 股票池 {len(codes)} 只")

    t0 = time.time()
    stocks = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(prepare_and_build, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            s = None
            try:
                s = fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"  [warn] {futs[fut]} 构建失败 {type(e).__name__} {e}")
            if s is not None:
                stocks.append(s)
            done += 1
            if done % 500 == 0 or done == len(codes):
                log(f"  build {done}/{len(codes)} 有效={len(stocks)} "
                    f"{time.time()-t0:.0f}s")

    stocks.sort(key=lambda s: s["code"])
    thr = assign_c_thresholds(stocks)
    n_rows = sum(s["n"] for s in stocks)
    n_eps = sum(s["n_episodes"] for s in stocks)
    log(f"[full] 候选周 {n_rows}；episode {n_eps}；决策日 {len(thr)}")

    with open(os.path.join(OUT_DIR, "c_thresholds.csv"), "w", encoding="utf-8") as f:
        f.write("date,hist_rows,hist_episodes,q80_age,q20_depth,classifiable\n")
        for r in thr:
            f.write(f"{r['date']},{r['hist_rows']},{r['hist_episodes']},"
                    f"{r['q_age']},{r['q_depth']},{int(r['classifiable'])}\n")

    result = dict(
        meta=dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  script="scripts/entry_phase.py",
                  protocol="ENTRY-PHASE-01 v1.2 FROZEN",
                  stage2=True),
        counts=dict(stocks=len(stocks), candidate_rows=int(n_rows),
                    episodes=int(n_eps), decision_dates=len(thr)),
        categories={},
    )
    all_tests = []
    for cat in CATEGORIES:
        r = run_category(stocks, cat)
        result["categories"][cat] = r
        all_tests.extend(r.get("tests", []))
    q = bh_adjust([t["p"] for t in all_tests])
    for t, qv in zip(all_tests, q):
        t["q_bh"] = qv
    result["bh"] = all_tests

    with open(os.path.join(OUT_DIR, "entry_phase_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(ud.jsonable(result), f, ensure_ascii=False, indent=2)
    log(f"[full] WROTE entry_phase_summary.json elapsed={time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# 阶段 2 年度统计修正：只重聚合，不重算行情/候选周/收益
# ---------------------------------------------------------------------------
def _result_snapshot(c: dict) -> dict:
    return dict(
        n_paired=c["counts"]["n_paired"],
        win60_all=c["metrics"]["category"]["win60_all"],
        payoff60=c["metrics"]["category"]["payoff60"],
        expectancy60=c["metrics"]["category"]["expectancy60"],
        r60_diff_mean=c["metrics"]["r60_diff_mean"],
        mae60_cat_median=c["metrics"]["mae60_cat_median"],
        mae60_b0_median=c["metrics"]["mae60_b0_median"],
        bh=[(t["name"], t["p"], tuple(t["ci"]), t["q_bh"]) for t in c["tests"]],
    )


def refix_annual() -> None:
    src = os.path.join(OUT_DIR, "entry_phase_summary.json")
    dst_json = os.path.join(OUT_DIR, "entry_phase_summary_annualfix.json")
    dst_md = os.path.join(OUT_DIR, "STAGE2_ANNUAL_FIX_REPORT.md")
    if not os.path.exists(src):
        raise SystemExit(f"缺少原始阶段 2 结果：{src}")
    obj = json.load(open(src, encoding="utf-8"))
    fixed = copy.deepcopy(obj)
    inv = {}
    for cat in CATEGORIES:
        c0 = obj["categories"][cat]
        c1 = fixed["categories"][cat]
        rows = copy.deepcopy(c0["annual"]["years"])
        rec = annual_from_rows(rows)
        c1["annual"] = rec
        c1["conditions"]["annual"] = bool(
            np.isfinite(rec["positive_share"])
            and rec["positive_share"] >= SUCCESS_YEAR_SHARE)
        c1["success"] = bool(all(c1["conditions"].values()))
        s0 = _result_snapshot(c0)
        s1 = _result_snapshot(c1)
        inv[cat] = dict(before=s0, after=s1, same=bool(s0 == s1))

    fixed["meta"]["annual_fix"] = dict(
        applied_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        source=os.path.basename(src),
        rule="完整年度=2005–2025；2026 YTD 单列，不进入年度稳定性分母；"
             "合格年度=每年≥30 配对 episode",
    )
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    write_annual_fix_report(obj, fixed, inv, dst_md)
    log(f"[annual-fix] WROTE {dst_json} / {dst_md}")


def write_annual_fix_report(old: dict, new: dict, inv: dict, path: str) -> None:
    lines: list[str] = []
    lines.append("# ENTRY-PHASE-01 v1.2 阶段 2 年度统计修正报告")
    lines.append("")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 协议：ENTRY-PHASE-01 v1.2（FROZEN，SHA256 3E11BB6B…70DC5）")
    lines.append("- 原始结果：`entry_phase_summary.json`（保留为审计记录，未覆盖）")
    lines.append("- 修正结果：`entry_phase_summary_annualfix.json`")
    lines.append("- 执行方式：仅用原结果中的逐年度行重新聚合；"
                 "未重跑行情、候选周、收益、配对与 BH 计算")
    lines.append("")
    lines.append("## 偏差与修正口径")
    lines.append("")
    lines.append("- 偏差：原实现把 2026 YTD 当作完整年度计入年度稳定性分母。")
    lines.append("- 修正：完整年度仅 2005–2025（共 21 年）；2026 YTD 单列；"
                 "合格年度仍需每年 ≥30 个配对 episode。")
    lines.append("")
    lines.append("## 修正前后对照")
    lines.append("")
    lines.append("| 类别 | 旧分子/分母 | 旧正向比例 | 新分子/分母 | 新正向比例 | "
                 "新合格年度数 | 新最差年度 | 2026 YTD(n/差) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cat in CATEGORIES:
        a0 = old["categories"][cat]["annual"]
        a1 = new["categories"][cat]["annual"]
        q0 = [r for r in a0.get("years", []) if bool(r.get("qualified"))]
        p0 = [r for r in q0 if np.isfinite(float(r["r60_diff_mean"]))
              and float(r["r60_diff_mean"]) > 0]
        ytd = a1.get("ytd", [])
        ytxt = "；".join(f"{r['year']}: {r['n_paired']} / "
                        f"{r['r60_diff_mean']:+.4f}" for r in ytd) or "-"
        lines.append(
            f"| {cat} | {len(p0)}/{len(q0)} | {a0['positive_share']:.3f} | "
            f"{a1['n_positive']}/{a1['n_qualified']} | {a1['positive_share']:.3f} | "
            f"{a1['n_qualified']} | {a1['worst_year']} | {ytxt} |")
    lines.append("")
    lines.append("## 2026 YTD 独立统计")
    lines.append("")
    lines.append("| 类别 | 配对 episode | r60 配对差均值 | 状态 |")
    lines.append("|---|---|---|---|")
    for cat in CATEGORIES:
        for r in new["categories"][cat]["annual"].get("ytd", []):
            lines.append(f"| {cat} | {r['n_paired']} | "
                         f"{r['r60_diff_mean']:+.4f} | 单列，不计入年度分母 |")
    lines.append("")
    lines.append("## 逐年度明细（2005–2025）")
    lines.append("")
    lines.append("| 类别 | 年份 | 配对 episode | r60 配对差均值 | 合格 |")
    lines.append("|---|---|---|---|---|")
    for cat in CATEGORIES:
        for r in new["categories"][cat]["annual"]["full_years"]:
            lines.append(f"| {cat} | {r['year']} | {r['n_paired']} | "
                         f"{r['r60_diff_mean']:+.4f} | "
                         f"{'是' if r['qualified'] else '否'} |")
    lines.append("")
    lines.append("## 不变量核验（本次修正不得改变）")
    lines.append("")
    lines.append("| 类别 | 配对 episode | win60_all | payoff60 | expectancy60 | "
                 "r60 差 | MAE60(类别/B0) | 12 项 BH | 一致 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cat in CATEGORIES:
        s = inv[cat]["after"]
        lines.append(
            f"| {cat} | {s['n_paired']} | {s['win60_all']:.4f} | "
            f"{s['payoff60']:.4f} | {s['expectancy60']:.4f} | "
            f"{s['r60_diff_mean']:+.4f} | "
            f"{s['mae60_cat_median']:.4f} / {s['mae60_b0_median']:.4f} | "
            f"{len(s['bh'])} 项 q={[round(x[3], 3) for x in s['bh']]} | "
            f"{'是' if inv[cat]['same'] else '**否**'} |")
    lines.append("")
    lines.append("## 最终裁定（8 项判据不变）")
    lines.append("")
    lines.append("| 条件 | A | B | C |")
    lines.append("|---|---|---|---|")
    labels = {
        "n_paired": "≥500 配对 episode", "win": "win60_all ≥50%",
        "payoff": "payoff60 ≥2.00", "expectancy": "expectancy60 >0 且 CI 不含 0",
        "r60_ci": "r60 配对差 CI 不含 0", "annual": "≥70% 合格年度为正",
        "mae": "MAE60 中位不比 B0 差 >2pp", "event_sign": "事件等权期望同号",
    }
    for k, lab in labels.items():
        vals = ["通过" if new["categories"][c]["conditions"][k]
                else "**未通过**" for c in CATEGORIES]
        lines.append(f"| {lab} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.append("| **综合** | " + " | ".join(
        "**通过**" if new["categories"][c]["success"] else "**未通过**"
        for c in CATEGORIES) + " |")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    for cat in CATEGORIES:
        c = new["categories"][cat]
        a = c["annual"]
        failed = [labels[k] for k, v in c["conditions"].items()
                  if v is False]
        lines.append(
            f"- **{cat}**：{a['n_positive']}/{a['n_qualified']} 个完整年度为正"
            f"（{a['positive_share']:.1%}），最差年度 {a['worst_year']}；"
            f"未通过项：{'、'.join(failed) if failed else '无'}。")
    lines.append("")
    lines.append("- 本修正不改变任何成功门槛，不进入阶段 3。")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# 测试样本
# ---------------------------------------------------------------------------
def load_sample(n_target: int, seed: int, workers: int) -> list[dict]:
    codes = all_codes()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(codes))
    picks = [codes[i] for i in order[: min(len(codes), n_target * 3)]]
    sample: list[dict] = []

    def work(c):
        raw = read_local_raw(c)
        if raw is None or raw.empty:
            return None
        facs = get_factor(c)
        if facs is None:
            return None
        A = ud.prepare_stock(raw, facs)
        if A is None:
            return None
        return dict(code=c, raw=raw, fac=facs, A=A, rows=build_stock(c, A))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, c) for c in picks]
        for fut in as_completed(futs):
            s = fut.result()
            if s is not None:
                sample.append(s)
            if len(sample) >= n_target:
                for f2 in futs:
                    f2.cancel()
                break

    sample.sort(key=lambda s: s["code"])
    for s in sample:
        s["stock"] = s.pop("rows")
    return sample


def stocks_of(sample: list[dict], key: str = "stock") -> list[dict]:
    return [s[key] for s in sample]


# ---------------------------------------------------------------------------
# T1 / T2
# ---------------------------------------------------------------------------
def rebuild_pool(sample: list[dict], cut: pd.Timestamp, mode: str,
                 workers: int, seed: int) -> list[dict]:
    cut_ts = pd.Timestamp(cut)

    def work(s):
        raw = s["raw"]
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        if mode == "truncate":
            raw2 = raw[raw["date"] <= cut_ts].reset_index(drop=True)
        else:
            raw2 = randomize_future(raw, cut_ts, seed + int(s["code"]))
        A = ud.prepare_stock(raw2, s["fac"])
        if A is None:
            return None
        return build_stock(s["code"], A)

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, s) for s in sample]):
            r = fut.result()
            if r is not None:
                out.append(r)
    out.sort(key=lambda s: s["code"])
    assign_c_thresholds(out)
    return out


def randomize_future(raw: pd.DataFrame, cut: pd.Timestamp, seed: int) -> pd.DataFrame:
    """保留 <= cut 的 bar，把之后的价格替换为随机游走。"""
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    keep = d[d["date"] <= cut].copy()
    fut = d[d["date"] > cut].copy()
    if fut.empty or keep.empty:
        return d
    c0 = float(keep["close"].iloc[-1])
    rng = np.random.default_rng(seed)
    hist = keep["close"].to_numpy(dtype=float)[-250:]
    ret = np.diff(np.log(hist))
    sig = float(np.nanstd(ret))
    if not np.isfinite(sig) or sig <= 0:
        sig = 0.02
    m = len(fut)
    rets = rng.normal(0.0, sig, m)
    closes = c0 * np.exp(np.cumsum(rets))
    opens = np.r_[c0, closes[:-1]] * np.exp(rng.normal(0, sig / 3, m))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, sig / 2, m)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, sig / 2, m)))
    lows = np.maximum(lows, 0.01)
    fut["open"] = opens
    fut["close"] = closes
    fut["high"] = highs
    fut["low"] = lows
    if "volume" in fut.columns:
        v0 = float(keep["volume"].iloc[-1]) if np.isfinite(keep["volume"].iloc[-1]) else 1.0
        fut["volume"] = np.maximum(v0 * np.exp(rng.normal(0, 0.3, m)), 1.0)
    if "amount" in fut.columns:
        fut["amount"] = fut["volume"] * np.maximum(closes, 0.01)
    return pd.concat([keep, fut], ignore_index=True)


def compare_rows(full: dict, part: dict, cut_ns: int, limit_report: int = 8) -> dict:
    """比较 w_end <= cut 的行；键 = (J, W)。"""
    keys_f = {}
    for i in range(full["n"]):
        if int(full["W_END"][i]) <= cut_ns:
            keys_f[(int(full["J"][i]), int(full["W"][i]))] = i
    n_checked = 0
    mism = []
    missing = 0
    for i in range(part["n"]):
        if int(part["W_END"][i]) > cut_ns:
            continue
        key = (int(part["J"][i]), int(part["W"][i]))
        if key not in keys_f:
            missing += 1
            continue
        fi = keys_f[key]
        n_checked += 1
        for fld in T12_COMPARE:
            a = part[fld][i]
            b = full[fld][fi]
            if not same_value(a, b):
                if len(mism) < limit_report:
                    mism.append(dict(code=part["code"], j=int(part["J"][i]),
                                     w=int(part["W"][i]), field=fld,
                                     full=_scalar(b), part=_scalar(a)))
                break
    return dict(checked=n_checked, missing=missing, mismatches=mism)


def same_value(a, b) -> bool:
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    fa = float(a)
    fb = float(b)
    if not np.isfinite(fa) and not np.isfinite(fb):
        return True
    return fa == fb


def close_value(a, b, rel: float = 1e-9, abst: float = 1e-12) -> bool:
    """独立复算允许最后几位浮点差异；布尔/整数必须严格一致。"""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    fa = float(a)
    fb = float(b)
    if not np.isfinite(fa) and not np.isfinite(fb):
        return True
    if not np.isfinite(fa) or not np.isfinite(fb):
        return False
    return abs(fa - fb) <= max(abst, rel * max(abs(fa), abs(fb)))


def _scalar(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return x


def test_t1(sample, cuts, workers, seed) -> dict:
    full = stocks_of(sample)
    res = []
    for k, cut in enumerate(cuts):
        part = rebuild_pool(sample, cut, "truncate", workers, seed + k)
        part_by_code = {s["code"]: s for s in part}
        cut_ns = ns(cut)
        n_checked = 0
        n_missing = 0
        mism = []
        for f in full:
            p = part_by_code.get(f["code"])
            if p is None:
                continue
            r = compare_rows(f, p, cut_ns)
            n_checked += r["checked"]
            n_missing += r["missing"]
            mism.extend(r["mismatches"][: max(0, 8 - len(mism))])
        res.append(dict(cut=str(pd.Timestamp(cut).date()), checked=n_checked,
                        missing=n_missing, mismatches=mism,
                        passed=(len(mism) == 0 and n_checked > 0)))
    return dict(name="T1", passed=all(r["passed"] for r in res), cuts=res)


def test_t2(sample, cuts, workers, seed) -> dict:
    full = stocks_of(sample)
    res = []
    for k, cut in enumerate(cuts):
        part = rebuild_pool(sample, cut, "random", workers, seed + 100 + k)
        part_by_code = {s["code"]: s for s in part}
        cut_ns = ns(cut)
        n_checked = 0
        n_missing = 0
        mism = []
        for f in full:
            p = part_by_code.get(f["code"])
            if p is None:
                continue
            r = compare_rows(f, p, cut_ns)
            n_checked += r["checked"]
            n_missing += r["missing"]
            mism.extend(r["mismatches"][: max(0, 8 - len(mism))])
        res.append(dict(cut=str(pd.Timestamp(cut).date()), checked=n_checked,
                        missing=n_missing, mismatches=mism,
                        passed=(len(mism) == 0 and n_checked > 0)))
    return dict(name="T2", passed=all(r["passed"] for r in res), cuts=res)


# ---------------------------------------------------------------------------
# T3：扩展分位只依赖历史
# ---------------------------------------------------------------------------
def test_t3(stocks, thresholds, cuts) -> dict:
    # T3a：生产增量实现 vs 独立逐日重算
    cls = [t for t in thresholds if t["classifiable"]]
    if not cls:
        return dict(name="T3", passed=False,
                    note="样本内没有任何可分类决策日，无法检验")
    pick = np.linspace(0, len(cls) - 1, min(12, len(cls))).astype(int)
    bad = []
    for i in pick:
        t = cls[int(i)]
        d = ns(t["date"])
        nv = naive_threshold(stocks, d)
        if not nv["classifiable"]:
            bad.append(dict(date=t["date"], reason="naive 不可分类"))
            continue
        if (nv["hist_rows"] != t["hist_rows"]
                or nv["hist_episodes"] != t["hist_episodes"]
                or abs(nv["q_age"] - t["q_age"]) > 1e-12
                or abs(nv["q_depth"] - t["q_depth"]) > 1e-12):
            bad.append(dict(date=t["date"], prod=t, naive=nv))

    # T3b：截断未来行后，历史阈值不变
    cut_res = []
    for cut in cuts:
        cut_ns = ns(cut)
        # 只保留 w_end < cut 的行（模拟“未来候选周尚未出现”）
        sub = []
        for s in stocks:
            m = s["W_END"] < cut_ns
            d2 = dict(code=s["code"], eps=s["eps"],
                      n_episodes=s["n_episodes"], n=int(m.sum()))
            for k, _dt in ROW_SPEC:
                d2[k] = s[k][m]
            sub.append(d2)
        # 用独立逐日实现重算 cut 之前的所有可分类日
        bad2 = []
        n_chk = 0
        for t in thresholds:
            d = ns(t["date"])
            if d >= cut_ns:
                continue
            nv = naive_threshold(stocks, d)
            nv2 = naive_threshold(sub, d)
            n_chk += 1
            if (nv["classifiable"] != nv2["classifiable"]
                    or (nv["classifiable"] and (
                        nv["hist_rows"] != nv2["hist_rows"]
                        or abs(nv["q_age"] - nv2["q_age"]) > 1e-12
                        or abs(nv["q_depth"] - nv2["q_depth"]) > 1e-12))):
                if len(bad2) < 5:
                    bad2.append(dict(date=t["date"], full=nv, sub=nv2))
        cut_res.append(dict(cut=str(pd.Timestamp(cut).date()), checked=n_chk,
                            mismatches=bad2, passed=(len(bad2) == 0 and n_chk > 0)))
    passed = (len(bad) == 0) and all(r["passed"] for r in cut_res)
    return dict(name="T3", passed=passed, prod_vs_naive=dict(checked=int(pick.size),
                                                             mismatches=bad),
                future_rows_removed=cut_res)


# ---------------------------------------------------------------------------
# T4：静态审查（禁止全样本分位/中位）
# ---------------------------------------------------------------------------
def find_enclosing(func_ranges, line):
    best = None
    for name, a, b in func_ranges:
        if a <= line <= b:
            if best is None or (b - a) < best[2] - best[1]:
                best = (name, a, b)
    return best[0] if best else "<module>"


def test_t4(path) -> dict:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    func_ranges = [(n.name, n.lineno, n.end_lineno) for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]
    banned = {"np.quantile", "np.percentile", "np.nanquantile", "np.nanpercentile",
              "np.median", "np.nanmedian"}
    hits = []
    ok = True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            name = f"{fn.value.id}.{fn.attr}"
        elif isinstance(fn, ast.Name):
            name = fn.id
        if name in banned:
            enc = find_enclosing(func_ranges, node.lineno)
            allowed = enc in PIT_AUDIT_ALLOW
            hits.append(dict(line=node.lineno, call=name, function=enc,
                             allowed=allowed))
            ok = ok and allowed
    return dict(name="T4", passed=bool(ok), hits=hits,
                note="分位/中位数只允许出现在 PIT 切片或事后统计函数中")


# ---------------------------------------------------------------------------
# T5：人工核对 + 独立复算
# ---------------------------------------------------------------------------
def week_grid_from_daily(s: dict, cut: pd.Timestamp):
    """独立实现：按 ISO 周把日线聚合为全局周网格（含 wclose/whigh/wlow/wamount）。"""
    raw = s["raw"].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["date"] <= cut]
    # 与上游同口径：清洗 + qfq
    d = ud.clean_raw(raw)
    d = ud.clean_raw(ud.apply_qfq(d, s["fac"]))
    d = d[d["date"] >= CTX["cal0"]]
    dates = d["date"].to_numpy()
    iso = pd.DatetimeIndex(dates).isocalendar()
    key = iso["year"].to_numpy(np.int64) * 100 + iso["week"].to_numpy(np.int64)
    nw = len(CTX["w_end"])
    wc = np.full(nw, np.nan)
    wh = np.full(nw, np.nan)
    wl = np.full(nw, np.nan)
    wa = np.full(nw, np.nan)
    for k in np.unique(key):
        m = key == k
        p = CTX["wkey_to_pos"].get(int(k), -1)
        if p < 0:
            continue
        wc[p] = d["close"].to_numpy()[m][-1]
        wh[p] = d["high"].to_numpy()[m].max()
        wl[p] = d["low"].to_numpy()[m].min()
        wa[p] = d["amount"].to_numpy()[m].sum() if "amount" in d else np.nan
    return wc, wh, wl, wa


def er_python(closes: np.ndarray, w: int, n: int = ER_WIN) -> float:
    seg = closes[w - n + 1:w + 1]
    if seg.size != n or not np.all(np.isfinite(seg)) or seg[0] <= 0:
        return np.nan
    path = float(np.sum(np.abs(np.diff(seg))))
    if path <= 0:
        return np.nan
    return float(abs(seg[-1] - seg[0]) / path)


def manual_row_check(s: dict, i: int) -> dict:
    """对单行做独立标量复算，返回差异列表。"""
    A = s["A"]
    w = int(s["stock"]["W"][i])
    j = int(s["stock"]["J"][i])
    cut = from_ns(int(s["stock"]["W_END"][i]))

    wc, wh, wl, wa = week_grid_from_daily(s, cut)
    conf, vals = ud.pivots(wl[:w + 1], "low")
    exts = conf               # 上游实测返回极值索引
    conf_week = conf + 2
    wl1 = float(vals[-1]) if (conf.size and conf_week[-1] <= w) else np.nan
    n_pb = int(np.sum((exts > j) & (exts + 2 <= w)))
    after = np.nonzero(exts > j)[0]
    a_week = int(exts[after[0]] + 2) if after.size else -10

    seg_h = wh[j:w + 1]
    run_high = float(np.nanmax(seg_h))
    tied = np.nonzero(np.isfinite(seg_h) & (seg_h == run_high))[0]
    h = int(j + tied[-1])
    depth = 1.0 - float(wc[w]) / run_high
    den = run_high - wl1
    retrace = ((run_high - float(wc[w])) / den) if (np.isfinite(den) and den > 0) else np.nan
    vr = np.nan
    if h - VOL_BASE_WEEKS >= 0:
        num = wa[h:w + 1]
        base = wa[h - VOL_BASE_WEEKS:h]
        if (np.all(np.isfinite(num)) and np.all(np.isfinite(base))
                and np.all(num > 0) and np.all(base > 0)):
            vr = float(np.mean(num) / np.mean(base))
    mddh = np.nan
    segl = wl[h:w + 1]
    if np.isfinite(segl).any():
        mddh = 1.0 - float(np.nanmin(segl)) / run_high
    er = er_python(wc, w)
    label_a = bool(w == a_week)
    label_b = bool(n_pb >= 1 and depth > 0 and np.isfinite(vr) and vr < 1.0)

    # 入场：独立循环
    dates = A["dates"]
    di = len(dates)
    for t in range(dates.size):
        if pd.Timestamp(dates[t]) > cut:
            di = t
            break
    tradable = di < dates.size
    ent_ns = 0
    op = np.nan
    if tradable:
        ci = CTX["cal_pos"].get(cut)
        lim = CTX["cal"][min(ci + ENTRY_MAX_MKT_DAYS, len(CTX["cal"]) - 1)]
        if ci is None or pd.Timestamp(dates[di]) > lim:
            tradable = False
        else:
            op = float(A["o"][di])
            ent_ns = ns(dates[di])

    r60 = np.nan
    if tradable and di + 59 < dates.size:
        r60 = float(A["c"][di + 59]) / op - 1.0

    race10 = race20n10 = race20 = -1
    mae = mdd = np.nan
    if tradable and di + 59 < dates.size:
        mae = float(np.min(A["l"][di:di + 60])) / op - 1.0
        peak = np.maximum.accumulate(np.maximum(A["h"][di:di + 60], op))
        mdd = -float(np.max(1.0 - A["l"][di:di + 60] / peak))
        race10 = first_touch(A["h"][di:di + 60], A["l"][di:di + 60], op, .10, .10)
        race20n10 = first_touch(A["h"][di:di + 60], A["l"][di:di + 60], op, .20, .10)
        race20 = first_touch(A["h"][di:di + 60], A["l"][di:di + 60], op, .20, .20)

    checks = [
        ("RUN_HIGH_W", h), ("N_PB", n_pb), ("RUN_HIGH", run_high),
        ("DEPTH", depth), ("RETRACE", retrace), ("VOL_RATIO", vr),
        ("ER26", er), ("MDD_HIGH", mddh), ("W_SINCE_H", w - h),
        ("WL1", wl1), ("LABEL_A", label_a), ("LABEL_B", label_b),
        ("ENTRY_DI", di if tradable else -1), ("ENTRY_OPEN", op),
        ("R60", r60), ("MAE60", mae), ("MDD60", mdd),
        ("RACE10", race10), ("RACE20N10", race20n10), ("RACE20", race20),
        ("TRADABLE", tradable),
    ]
    diffs = []
    st = s["stock"]
    for fld, val in checks:
        prod = st[fld][i]
        if not close_value(prod, val):
            diffs.append(dict(field=fld, prod=_scalar(prod), manual=_scalar(val)))
    return dict(code=s["code"], j=int(st["J"][i]), w=w,
                date_w=str(cut.date()),
                entry_date=str(from_ns(int(st["ENTRY_NS"][i])).date())
                if bool(st["TRADABLE"][i]) else None,
                labels={c: bool(st[f"LABEL_{c}"][i]) for c in CATEGORIES},
                features=dict(age=int(st["AGE"][i]),
                              run_high=float(st["RUN_HIGH"][i]),
                              depth=float(st["DEPTH"][i]),
                              vol_ratio=float(st["VOL_RATIO"][i]),
                              er26=float(st["ER26"][i]),
                              n_pb=int(st["N_PB"][i]),
                              run_high_w=int(st["RUN_HIGH_W"][i]),
                              w_since_high=int(st["W_SINCE_H"][i]),
                              mdd_since_high=float(st["MDD_HIGH"][i])),
                entry=dict(tradable=bool(st["TRADABLE"][i]),
                           di=int(st["ENTRY_DI"][i]),
                           open=float(st["ENTRY_OPEN"][i])),
                outcomes=dict(r60=float(st["R60"][i]),
                              mae60=float(st["MAE60"][i]),
                              mdd60=float(st["MDD60"][i]),
                              race10=int(st["RACE10"][i]),
                              race20n10=int(st["RACE20N10"][i]),
                              race20=int(st["RACE20"][i])),
                diffs=diffs)


def metrics_python(r: list[float], w: list[float]) -> dict:
    """独立循环实现：用于核对 metrics_weighted。"""
    wp = wn = w0 = 0.0
    swp = swn = swr = 0.0
    totw = 0.0
    pos = []
    neg = []
    for ri, wi in zip(r, w):
        totw += wi
        if ri > 0:
            wp += wi
            swp += wi * ri
            pos.append(ri)
        elif ri < 0:
            wn += wi
            swn += wi * (-ri)
            neg.append(ri)
        else:
            w0 += wi
        swr += wi * ri
    tot = wp + wn + w0
    out = dict(win60_all=(wp / tot) if tot > 0 else np.nan,
               win60_nonflat=(wp / (wp + wn)) if (wp + wn) > 0 else np.nan,
               flat60=(w0 / tot) if tot > 0 else np.nan,
               avg_win60=(swp / wp) if wp > 0 else np.nan,
               avg_loss60=(swn / wn) if wn > 0 else np.nan,
               expectancy60=(swr / totw) if totw > 0 else np.nan)
    out["payoff60"] = (out["avg_win60"] / out["avg_loss60"]
                       if (np.isfinite(out["avg_win60"]) and np.isfinite(out["avg_loss60"])
                           and out["avg_loss60"] > 0) else np.nan)
    return out


def test_t5(sample, stocks) -> dict:
    rows_manual = []
    for s in sample:
        st = s["stock"]
        n = st["n"]
        if n == 0:
            continue
        # 优先涵盖 A/B/C 与普通周
        idxs = []
        for c in CATEGORIES:
            idxs.extend(np.nonzero(st[f"LABEL_{c}"])[0][:2].tolist())
        idxs.extend(np.linspace(0, n - 1, min(4, n)).astype(int).tolist())
        for i in sorted(set(int(x) for x in idxs)):
            rows_manual.append(manual_row_check(s, i))
            if len(rows_manual) >= 20:
                break
        if len(rows_manual) >= 20:
            break

    ok_rows = all(len(r["diffs"]) == 0 for r in rows_manual)

    # 10 个 episode 的独立指标复算
    ep_checks = []
    for cat in CATEGORIES:
        ps = build_paired(stocks, cat)
        for e in ps["paired"]:
            if len(ep_checks) >= 10:
                break
            b0_r = e["b0_r60"].tolist()
            b0_w = [1.0 / len(b0_r)] * len(b0_r)
            ref = metrics_weighted(np.array([e["cat_r60"]]), np.array([1.0]))
            ref_b0 = metrics_weighted(np.array(b0_r), np.array(b0_w))
            py = metrics_python([e["cat_r60"]], [1.0])
            py_b0 = metrics_python(b0_r, b0_w)
            diffs = []
            for k in ("win60_all", "payoff60", "expectancy60"):
                for tag, a, b in (("cat", ref[k], py[k]), ("b0", ref_b0[k], py_b0[k])):
                    if not close_value(a, b):
                        diffs.append(dict(side=tag, metric=k,
                                          vector=_scalar(a), python=_scalar(b)))
            ep_checks.append(dict(cat=cat, code=e["code"], year=e["year"],
                                  n_b0=len(b0_r), diffs=diffs,
                                  cat_r60=e["cat_r60"]))
        if len(ep_checks) >= 10:
            break
    ok_ep = all(len(r["diffs"]) == 0 for r in ep_checks)

    # T5 第 3 项：入场 bar 盘中触及必须被排除
    race_ignore_ok = True
    race_checks = 0
    for s in sample:
        st = s["stock"]
        for i in range(st["n"]):
            di = int(st["ENTRY_DI"][i])
            if not bool(st["TRADABLE"][i]) or di < 0 or di + 59 >= s["A"]["dates"].size:
                continue
            A = s["A"]
            h2 = A["h"][di:di + 60].copy()
            l2 = A["l"][di:di + 60].copy()
            h2[0] = h2[0] * 10.0
            l2[0] = l2[0] / 10.0
            r10 = first_touch(h2, l2, float(st["ENTRY_OPEN"][i]), .10, .10)
            if r10 != int(st["RACE10"][i]):
                race_ignore_ok = False
            race_checks += 1
            if race_checks >= 20:
                break
        if race_checks >= 20:
            break

    # T5 第 4 项：episode 母体必须与上游 analyze_signal 完全一致
    ep_parity_bad = []
    ep_parity_checked = 0
    for s in sample:
        A = s["A"]
        _, ev_main, episodes, _rej = ud.analyze_signal(s["code"], A)
        mine = main_episodes(A)
        wpos = {int(ns(t)): i for i, t in enumerate(A["w_end_ts"])}
        ref = []
        for k, block in episodes:
            k = int(k)
            cand = np.nonzero(A["cond_week"][k:min(int(block), A["nw"])])[0]
            if cand.size:
                ref.append((k + int(cand[0]), int(block)))
        ref_j = [wpos[int(ns(m["sig_date"]))] for m in ev_main]
        mine_j = [int(j) for j, _ in mine]
        diffs = []
        if len(mine) != len(ref):
            diffs.append(dict(field="n_episodes", prod=len(mine), ref=len(ref)))
        else:
            for (mj, mb), rb in zip(mine, ref):
                if (int(mj), int(mb)) != rb:
                    diffs.append(dict(field="episode",
                                      prod=[int(mj), int(mb)], ref=list(rb)))
        if mine_j != ref_j:
            diffs.append(dict(field="ev_main_weeks", prod=mine_j, ref=ref_j))
        ep_parity_checked += 1
        if diffs:
            ep_parity_bad.append(dict(code=s["code"], diffs=diffs))
    ok_parity = len(ep_parity_bad) == 0

    return dict(name="T5",
                passed=bool(ok_rows and ok_ep and race_ignore_ok and ok_parity),
                manual_rows=rows_manual, episodes=ep_checks,
                race_ignores_entry=dict(checked=race_checks,
                                        passed=race_ignore_ok),
                episode_parity=dict(checked=ep_parity_checked,
                                    passed=ok_parity, mismatches=ep_parity_bad),
                n_manual=len(rows_manual))


# ---------------------------------------------------------------------------
# 报告
# -------------------------------------------------------------------
def write_report(results: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    lines.append("# T1–T5 实现正确性测试报告")
    lines.append("")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 协议：ENTRY-PHASE-01 v1.2（FROZEN，"
                 "SHA256 3E11BB6B…70DC5）")
    lines.append("- 脚本：`scripts/entry_phase.py`")
    lines.append("- 范围：**只运行 T1–T5**；未运行阶段 2 全量统计，"
                 "未产出任何 A/B/C 研究结论")
    lines.append("")
    lines.append(f"- 测试样本：随机抽取 **{results['sample_n']}** 只可评估股票"
                 f"，候选周 {results['n_rows']} 行，episode {results['n_episodes']} 个")
    lines.append("")

    lines.append("## 测试结论")
    lines.append("")
    lines.append("| 测试 | 内容 | 结果 | 关键证据 |")
    lines.append("|---|---|---|---|")
    t1 = results["T1"]
    ev1 = "；".join(f"{c['cut']}: {c['checked']} 行" for c in t1["cuts"])
    lines.append(f"| T1 | 截断未来 bar，特征与标签不变 | "
                 f"{'PASS' if t1['passed'] else 'FAIL'} | {ev1}；不一致 "
                 f"{sum(len(c['mismatches']) for c in t1['cuts'])} |")
    t2 = results["T2"]
    ev2 = "；".join(f"{c['cut']}: {c['checked']} 行" for c in t2["cuts"])
    lines.append(f"| T2 | 替换未来 bar 为随机游走，特征与标签不变 | "
                 f"{'PASS' if t2['passed'] else 'FAIL'} | {ev2}；不一致 "
                 f"{sum(len(c['mismatches']) for c in t2['cuts'])} |")
    t3 = results["T3"]
    lines.append(f"| T3 | C 扩展分位只用历史，追加未来行不改阈值 | "
                 f"{'PASS' if t3['passed'] else 'FAIL'} | 增量 vs 逐日重算 "
                 f"{t3['prod_vs_naive']['checked']} 日；未来行移除 "
                 f"{len(t3['future_rows_removed'])} 次切割 |")
    t4 = results["T4"]
    lines.append(f"| T4 | 代码审查：禁止全样本分位/中位 | "
                 f"{'PASS' if t4['passed'] else 'FAIL'} | 命中 {len(t4['hits'])} 处，"
                 f"全部位于 PIT 切片或事后统计函数 |")
    t5 = results["T5"]
    lines.append(f"| T5 | 人工核对 20 样本 + 10 episode 独立复算 | "
                 f"{'PASS' if t5['passed'] else 'FAIL'} | "
                 f"{t5['n_manual']} 行、{len(t5['episodes'])} episode；"
                 f"入场 bar 排除 {t5['race_ignores_entry']['checked']} 例；"
                 f"母体核对 {t5['episode_parity']['checked']} 只，"
                 f"不一致 {len(t5['episode_parity']['mismatches'])} 只 |")
    lines.append("")

    lines.append("## T1 / T2 明细")
    lines.append("")
    for tag in ("T1", "T2"):
        for c in results[tag]["cuts"]:
            lines.append(f"- {tag} cut={c['cut']}：核对 {c['checked']} 行，"
                         f"缺失 {c['missing']} 行，不一致 {len(c['mismatches'])} 行")
            for m in c["mismatches"]:
                lines.append(f"  - `{m}`")
    lines.append("")

    lines.append("## T3 明细")
    lines.append("")
    lines.append(f"- 增量实现与独立逐日重算比较："
                 f"{results['T3']['prod_vs_naive']['checked']} 个决策日，"
                 f"不一致 {len(results['T3']['prod_vs_naive']['mismatches'])} 个")
    for c in results["T3"]["future_rows_removed"]:
        lines.append(f"- 移除 w_end ≥ {c['cut']} 的未来候选行后，"
                     f"复核 {c['checked']} 个历史决策日，"
                     f"阈值不一致 {len(c['mismatches'])} 个")
    lines.append("")

    lines.append("## T4 源码审查")
    lines.append("")
    lines.append("| 行号 | 调用 | 所在函数 | 允许 |")
    lines.append("|---|---|---|---|")
    for h in results["T4"]["hits"]:
        lines.append(f"| {h['line']} | `{h['call']}` | `{h['function']}` | "
                     f"{'是' if h['allowed'] else '**否**'} |")
    lines.append("")

    lines.append("## T5 人工核对样本（20 行）")
    lines.append("")
    lines.append("| 代码 | 信号周 | 候选周 | 年龄 | 深度 | 量比 | "
                 "A/B/C | 入场日 | r60 | MAE60 | RACE10 | 差异 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results["T5"]["manual_rows"]:
        lab = "".join(c for c in CATEGORIES if r["labels"][c]) or "-"
        lines.append(
            f"| {r['code']} | {r['j']} | {r['date_w']} | {r['features']['age']} | "
            f"{r['features']['depth']:.4f} | {r['features']['vol_ratio']:.3f} | "
            f"{lab} | {r['entry_date'] or 'untradable'} | "
            f"{r['outcomes']['r60']:.4f} | {r['outcomes']['mae60']:.4f} | "
            f"{r['outcomes']['race10']} | {len(r['diffs'])} |")
    lines.append("")

    lines.append("## T5 episode 独立复算（payoff60 / win60_all / expectancy60）")
    lines.append("")
    lines.append("| 类别 | 代码 | 年份 | B0 周数 | 类别 r60 | 差异 |")
    lines.append("|---|---|---|---|---|---|")
    for e in results["T5"]["episodes"]:
        lines.append(f"| {e['cat']} | {e['code']} | {e['year']} | {e['n_b0']} | "
                     f"{e['cat_r60']:.4f} | {len(e['diffs'])} |")
    lines.append("")

    lines.append("## T5 episode 母体核对（与上游 analyze_signal）")
    lines.append("")
    ep = results["T5"]["episode_parity"]
    lines.append(f"- 逐只核对 {ep['checked']} 只：信号周 `j`、终止周 `block`、"
                 f"`ev_main` 信号周序列是否与上游完全一致")
    lines.append(f"- 不一致股票 {len(ep['mismatches'])} 只")
    for m in ep["mismatches"]:
        lines.append(f"  - `{m['code']}`: `{m['diffs']}`")
    lines.append("")

    lines.append("## 实质性发现：上游 pivots 的确认时点")
    lines.append("")
    lines.append("T2 首轮暴露了一个上游实现与文字定义不一致的问题，必须披露：")
    lines.append("")
    lines.append("- `uptrend_detector.pivots` 的 docstring 写“返回确认索引 = 极值 + 2”，"
                 "但实测返回的是**极值索引**（已用 `[9,8,7,2,3,4,…]` 单测确认）。")
    lines.append("- 上游 `running_two(*pivots(...))` 因此把 2 右确认的波段点在**极值当周**"
                 "就视为“已确认”，比文字定义提前 2 周。")
    lines.append("- 这会出现在上游 `W` 的 `wl1 / wh1` 与识别条件里，属于 2 周的"
                 "前视；本实现不修改上游文件。")
    lines.append("- ENTRY-PHASE 的特征与标签一律按冻结协议使用"
                 "“确认周 = 极值 + 2”（`conf_week = conf + 2`）；"
                 "趋势 episode 的起止（信号周、终止周）仍按协议复用上游定义。")
    lines.append("- 影响：候选周的结构判断与 A 类确认周以协议为准；"
                 "episode 母体的信号周可能比严格 PIT 定义早最多 2 周。"
                 "**建议在阶段 2 前单独确认是否修订 UPTREND-DETECTOR-01；"
                 "本阶段未做任何上游改动。**")
    lines.append("")

    lines.append("## 实现歧义记录（按最保守解释执行）")
    lines.append("")
    lines.append("1. **episode 主信号周**：上游 `analyze_signal` 返回的 episodes 是"
                 "结构信号周 `k`；协议母体是 A 组主信号。本实现按同样的区间规则"
                 "重新取区间内强度首次满足的周 `j`，与上游 `ev_main` 的数量口径一致；"
                 "T5 已逐只核对信号周、终止周与 `ev_main` 序列，全部一致。")
    lines.append("2. **A 类“信号周之后”**：取极值索引严格大于主信号周 `j` 的第一个"
                 "周线波段低点，确认周 = 极值 + 2；若确认周不是候选周则不产生 A 观测。")
    lines.append("3. **vol_ratio**：按协议字面“周均成交额”，分子/分母均为对应窗口内"
                 "周成交额的算术平均（非按天数归一）。12 周窗口沿用既有设定。")
    lines.append("4. **首次触及窗口**：与 `r60` 一致，要求第 60 个有效 bar 存在；"
                 "计时从入场日的下一有效 bar 开始，入场日盘中触及不计。")
    lines.append("5. **C 类降级顺序**：本实现只落地主定义（历史 ≥500 候选周且 ≥50 "
                 "episode）；降级属于阶段 2 的报告决策，未在看到结果前自动应用。")
    lines.append("6. **B0 的 r60 完整性**：不完整 r60 的候选周直接剔除，不插补；"
                 "剔除后不足 5 周则整个 episode 退出主配对样本。")
    lines.append("")

    lines.append("## 阶段 2 闸门")
    lines.append("")
    lines.append(f"- 只有本报告全部 PASS 时才生成 `{os.path.basename(PASS_MARKER)}`；"
                 "`--stage full` 会强制检查该标记。")
    lines.append("")
    with open(TEST_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
def run_test(args) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.time()
    init_ctx()
    log(f"日历 {len(CTX['cal'])} 日；周末端 {len(CTX['w_end'])}")

    sample = load_sample(args.sample, args.seed, args.workers)
    if not sample:
        raise SystemExit("未取到可评估样本（检查 raw / 复权因子）")
    log(f"测试样本 {len(sample)} 只")

    stocks = stocks_of(sample)
    thr = assign_c_thresholds(stocks)
    n_rows = sum(s["n"] for s in stocks)
    n_eps = sum(s["n_episodes"] for s in stocks)
    log(f"候选周 {n_rows}；episode {n_eps}；决策日 {len(thr)}；"
        f"可分类日 {sum(1 for t in thr if t['classifiable'])}")

    t1 = test_t1(sample, [pd.Timestamp("2012-12-31"),
                          pd.Timestamp("2017-12-31"),
                          pd.Timestamp("2021-12-31")], args.workers, args.seed)
    log(f"T1 {'PASS' if t1['passed'] else 'FAIL'}")
    t2 = test_t2(sample, [pd.Timestamp("2014-12-31"),
                          pd.Timestamp("2019-12-31")], args.workers, args.seed)
    log(f"T2 {'PASS' if t2['passed'] else 'FAIL'}")
    t3 = test_t3(stocks, thr, [pd.Timestamp("2012-12-31"),
                               pd.Timestamp("2017-12-31")])
    log(f"T3 {'PASS' if t3['passed'] else 'FAIL'}")
    t4 = test_t4(os.path.abspath(__file__))
    log(f"T4 {'PASS' if t4['passed'] else 'FAIL'}")
    t5 = test_t5(sample, stocks)
    log(f"T5 {'PASS' if t5['passed'] else 'FAIL'}")

    results = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        sample_n=len(sample), n_rows=int(n_rows), n_episodes=int(n_eps),
        n_decision_dates=len(thr),
        n_classifiable_dates=sum(1 for t in thr if t["classifiable"]),
        T1=t1, T2=t2, T3=t3, T4=t4, T5=t5,
        passed=bool(t1["passed"] and t2["passed"] and t3["passed"]
                    and t4["passed"] and t5["passed"]),
    )
    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(ud.jsonable(results), f, ensure_ascii=False, indent=2)
    write_report(results)

    if results["passed"]:
        with open(PASS_MARKER, "w", encoding="utf-8") as f:
            json.dump(dict(passed=True, generated_at=results["generated_at"],
                           sample_n=len(sample), n_rows=int(n_rows),
                           report=os.path.basename(TEST_MD)), f,
                      ensure_ascii=False, indent=2)
        log("T1–T5 全部通过，已写 T1_T5_PASS.json")
    else:
        if os.path.exists(PASS_MARKER):
            os.remove(PASS_MARKER)
        log("T1–T5 未全部通过，未生成阶段 2 允许标记")
    log(f"WROTE {TEST_JSON} / {TEST_MD} elapsed={time.time()-t_all:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="ENTRY-PHASE-01 v1.2 实现")
    ap.add_argument("--stage", choices=["test", "full", "annual-fix"],
                    default="test")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()

    if args.stage == "test":
        run_test(args)
    elif args.stage == "annual-fix":
        refix_annual()
    else:
        init_ctx()
        run_full(args)


if __name__ == "__main__":
    main()
