"""P12-0 判据 4：量化 survivorship 到底把结论抬高了多少。

做法（严格对照，只换面板，其他一切冻结）
--------------------------------------
同一个 Alpha、同一套参数、同一套执行层（`research/overlay_regime` 的
build_targets / simulate / metrics 原样复用），跑在两个面板上：

  SURV 面板 = hs300_no_star.json 的 280 只（2026 年时点沪深300，P11-C 用的）
  PIT  面板 = 候选超集（当前存活全 A + 全部退市股），并按 PIT 定义掩膜：
              累计有 bar 天数 >= min_history 才计入

两者的差，就是"用今天的赢家回测历史"抬高的部分。

冻结条款（不允许为让结果好看而改）
----------------------------------
- Alpha 参数原样：mom=120, top_n=30, rebal=20, T+1, 单边 12bp, min_amount=5e7
- top_n=30 在 PIT 大池里占比变小（不再是"10% 池"）—— **这是刻意保留的**：
  本测的目标是"同一个策略在两个池子上的差"，不是"给大池重新寻优"。
  占比变化的影响单独在报告里标注。
- 等权基准口径原样：`ret.replace(0, nan).mean(axis=1)`，不计成本
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.stock_loader import load_universe_panel, cache_path  # noqa: E402
from research.overlay_regime import (  # noqa: E402
    ALPHA, COST_ONE_WAY, build_targets, metrics, simulate, _ret_matrix,
)

OUT = os.path.join(ROOT, "out")
UNIVERSE_DIR = os.path.join(ROOT, "data", "universe")
MIN_HISTORY = 60


# ---------------------------------------------------------------------------
def pit_mask(close: pd.DataFrame, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """PIT 掩膜：累计有 bar 天数 >= min_history 才保留，其余置 NaN.

    这一条同时解决：尚未上市（前面不够）、已退市（后面无 bar）、次新股。
    停牌当日 close=0 由下游 `_clean_close` 处理。
    """
    cum = close.where(close > 0).notna().cumsum()
    return close.where(cum >= min_history)


def pit_pool_size(close: pd.DataFrame, dates, min_history: int = MIN_HISTORY):
    """给定时点的 PIT 池大小（用于报告"样本宽度"一行）。"""
    m = pit_mask(close, min_history)
    out = {}
    for d in dates:
        d = pd.Timestamp(d)
        prior = m.index[m.index <= d]
        if len(prior) == 0:
            out[d] = 0
            continue
        out[d] = int(m.loc[prior[-1]].notna().sum())
    return out


def run_one(close: pd.DataFrame, amount: pd.DataFrame, label: str) -> dict:
    """在给定面板上跑 Alpha + 等权基准，返回指标。"""
    ret = _ret_matrix(close)
    tgt = build_targets(close, amount)
    net, expo, turn = simulate(close, tgt, ret)
    ew = ret.replace(0.0, np.nan).mean(axis=1).fillna(0.0)

    ann_turn = float(turn.sum() / max(len(turn), 1) * 243)
    m_alpha = metrics(net, expo, ann_turn)
    m_ew = metrics(ew)

    # 分区间（和 P11-C 同样的切法）
    per = {}
    for name, a, b in [("2005-2010", "2005-01-01", "2010-12-31"),
                       ("2011-2015", "2011-01-01", "2015-12-31"),
                       ("2016-2020", "2016-01-01", "2020-12-31"),
                       ("2021-2026", "2021-01-01", "2026-12-31")]:
        sl = slice(a, b)
        per[name] = {
            "alpha": metrics(net.loc[sl]),
            "ew": metrics(ew.loc[sl]),
        }
    return {"label": label, "alpha": m_alpha, "ew": m_ew, "periods": per,
            "net": net, "ew_ret": ew, "expo": expo}


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=250)
    ap.add_argument("--no-pit-superset", action="store_true",
                    help="PIT 池只用已缓存的 280+333（不读 all_a.json）")
    args = ap.parse_args()

    # ---- SURV 面板 ----
    _hs = json.load(open(os.path.join(UNIVERSE_DIR, "hs300_no_star.json"),
                         encoding="utf-8"))
    surv_codes = _hs["codes"] if isinstance(_hs, dict) else list(_hs)
    close_s, amt_s, rep_s = load_universe_panel(surv_codes, min_days=args.min_days)
    print(f"SURV 面板: {close_s.shape[1]} 只 × {len(close_s)} 天 "
          f"(requested {rep_s['requested']})", flush=True)

    # ---- PIT 面板 ----
    pit_codes = set(surv_codes)
    delisted = json.load(open(os.path.join(UNIVERSE_DIR, "delisted_codes.json"),
                             encoding="utf-8"))
    pit_codes |= set(delisted)
    if not args.no_pit_superset:
        all_a = json.load(open(os.path.join(UNIVERSE_DIR, "all_a.json"), encoding="utf-8"))
        for d in all_a:
            c = str(d["code"]).zfill(6)
            if c[0] in ("6", "0", "3") and c[:3] not in ("900", "200"):
                pit_codes.add(c)
    pit_codes = sorted(c for c in pit_codes if os.path.exists(cache_path(c)))
    close_p, amt_p, rep_p = load_universe_panel(pit_codes, min_days=args.min_days)
    print(f"PIT  面板: {close_p.shape[1]} 只 × {len(close_p)} 天", flush=True)

    # 统一日期轴（取交集，保证两条腿样本完全一致）
    # 口径纪律：起点对齐 P11-C 的 2005 年。个股数据可回溯到 1991 年，
    # 但那时 SURV 池只有个位数股票，留着会让"样本宽度"一行失真。
    common = close_s.index.intersection(close_p.index)
    common = common[common >= "2005-01-01"]
    close_s, amt_s = close_s.loc[common], amt_s.loc[common]
    close_p, amt_p = close_p.loc[common], amt_p.loc[common]
    print(f"共同交易日: {len(common)}  ({common[0].date()} ~ {common[-1].date()})", flush=True)

    # SURV 也要套同样的 PIT 掩膜吗？—— 不套。
    # SURV 就是 P11-C 的原样口径（无 PIT 过滤），这样差才是"完整的方法论差"。
    close_p_masked = pit_mask(close_p, MIN_HISTORY)

    print("跑 SURV ...", flush=True)
    rs = run_one(close_s, amt_s, "SURV(今日HS300)")
    print("跑 PIT ...", flush=True)
    rp = run_one(close_p_masked, amt_p, "PIT(全A+退市)")

    # ---- 样本宽度 ----
    probe = ["2005-12-30", "2010-12-31", "2015-12-31", "2020-12-31", "2025-12-31"]
    sz = pit_pool_size(close_p_masked, probe)
    sz_raw = {pd.Timestamp(d): int(close_s.loc[:d].iloc[-1].gt(0).sum()) for d in probe}

    # ---- 报表 ----
    rows = []
    for k in ("alpha", "ew"):
        a, b = rs[k], rp[k]
        rows.append({
            "腿": "Alpha(动量前30)" if k == "alpha" else "等权全池",
            "池": "SURV", "CAGR": a["CAGR"], "vol": a["vol"], "Sharpe": a["Sharpe"],
            "MaxDD": a["MaxDD"], "Calmar": a["Calmar"],
        })
        rows.append({
            "腿": "Alpha(动量前30)" if k == "alpha" else "等权全池",
            "池": "PIT", "CAGR": b["CAGR"], "vol": b["vol"], "Sharpe": b["Sharpe"],
            "MaxDD": b["MaxDD"], "Calmar": b["Calmar"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "p12_survivorship.csv"), index=False,
              encoding="utf-8-sig")

    per_rows = []
    for p in rs["periods"]:
        for k, nm in (("alpha", "Alpha(动量前30)"), ("ew", "等权全池")):
            sa, sp = rs["periods"][p][k], rp["periods"][p][k]
            per_rows.append({
                "区间": p, "腿": nm,
                "SURV_CAGR": sa.get("CAGR"), "PIT_CAGR": sp.get("CAGR"),
                "差pp": (None if sa.get("CAGR") is None or sp.get("CAGR") is None
                         else round((sa["CAGR"] - sp["CAGR"]) * 100, 2)),
                "SURV_Sharpe": sa.get("Sharpe"), "PIT_Sharpe": sp.get("Sharpe"),
            })
    dper = pd.DataFrame(per_rows)
    dper.to_csv(os.path.join(OUT, "p12_survivorship_periods.csv"), index=False,
                encoding="utf-8-sig")

    # 样本宽度统计也补进一张表
    pd.DataFrame([
        {"时点": pd.Timestamp(d).date(), "SURV在池": sz_raw.get(d, 0), "PIT在池": v}
        for d, v in sz.items()
    ]).to_csv(os.path.join(OUT, "p12_pit_pool_size.csv"), index=False,
              encoding="utf-8-sig")

    # 相关性（两条腿是不是同一条曲线 —— 若高度相关说明差别只在水位）
    corr_alpha = float(rs["net"].corr(rp["net"]))
    corr_ew = float(rs["ew_ret"].corr(rp["ew_ret"]))

    write_md(rs, rp, df, dper, sz, sz_raw, corr_alpha, corr_ew, close_s, close_p,
             common)

    print("\n" + df.to_string(index=False))
    print("\n" + dper.to_string(index=False))
    print(f"\ncorr(alpha) SURV~PIT = {corr_alpha:.3f}   corr(ew) = {corr_ew:.3f}")


def _p(x, nd=2):
    return "n/a" if x is None else f"{x*100:.{nd}f}%"


def write_md(rs, rp, df, dper, sz, sz_raw, corr_alpha, corr_ew, close_s, close_p,
             common) -> None:
    a_s, a_p = rs["alpha"], rp["alpha"]
    e_s, e_p = rs["ew"], rp["ew"]
    d_alpha_pp = (a_s["CAGR"] - a_p["CAGR"]) * 100
    d_ew_pp = (e_s["CAGR"] - e_p["CAGR"]) * 100
    gap_s = (a_s["CAGR"] - e_s["CAGR"]) * 100
    gap_p = (a_p["CAGR"] - e_p["CAGR"]) * 100

    L: list[str] = []
    L.append("# P12-0 判据 4：survivorship 把结论抬高了多少\n")
    L.append("同一 Alpha、同一参数、同一执行层（`research/overlay_regime` 原样复用），"
             "只换面板：\n")
    L.append(f"- **SURV** = `hs300_no_star.json` {close_s.shape[1]} 只"
             f"（2026 年时点沪深300）—— P11-C 原口径")
    L.append(f"- **PIT**  = 候选超集 {close_p.shape[1]} 只（当前存活全 A + 全部退市股），"
             f"按 PIT 定义掩膜（累计 bar ≥ {MIN_HISTORY}）\n")
    L.append(f"共同样本 {len(common)} 个交易日：{common[0].date()} ~ {common[-1].date()}\n")

    L.append("## 1. 主对照\n")
    L.append("| 腿 | 池 | CAGR | vol | Sharpe | MaxDD | Calmar |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        L.append(f"| {r['腿']} | {r['池']} | {_p(r['CAGR'])} | {_p(r['vol'])} | "
                 f"{r['Sharpe'] if r['Sharpe'] is not None else 'n/a'} | "
                 f"{_p(r['MaxDD'])} | {r['Calmar'] if r['Calmar'] is not None else 'n/a'} |")
    L.append("")

    L.append("## 2. survivorship 抬高了多少\n")
    L.append("| 腿 | SURV CAGR | PIT CAGR | 被抬高(pp) |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| Alpha(动量前30) | {_p(a_s['CAGR'])} | {_p(a_p['CAGR'])} | "
             f"**{d_alpha_pp:+.2f}** |")
    L.append(f"| 等权全池       | {_p(e_s['CAGR'])} | {_p(e_p['CAGR'])} | "
             f"**{d_ew_pp:+.2f}** |")
    L.append("")

    L.append("## 3. 关键：'Alpha 跑输等权' 这个结论本身还成立吗\n")
    L.append("| 池 | Alpha CAGR | 等权 CAGR | Alpha − 等权 (pp) |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| SURV（P11-C 口径） | {_p(a_s['CAGR'])} | {_p(e_s['CAGR'])} | "
             f"{gap_s:+.2f} |")
    L.append(f"| PIT（无偏池）     | {_p(a_p['CAGR'])} | {_p(e_p['CAGR'])} | "
             f"{gap_p:+.2f} |")
    L.append("")
    L.append(f"若两个池子里 `Alpha − 等权` **同号且量级接近**，说明"
             f"'动量跑输等权'这个**方向性结论不受 survivorship 影响**，"
             f"被污染的只是水位。")
    L.append(f"若**变号或量级差一倍以上**，则 P11-C 第 1 节的定性判断"
             f"**必须在 PIT 口径下重写**。\n")

    L.append("## 4. 样本宽度（PIT 池到底补了多少）\n")
    L.append("| 时点 | SURV 在池 | PIT 在池 | 倍数 |")
    L.append("|---|---:|---:|---:|")
    for d, v in sz.items():
        s = sz_raw.get(d, 0)
        L.append(f"| {d.date()} | {s} | {v} | {v/max(s,1):.1f}× |")
    L.append("")

    L.append("## 5. 分区间\n")
    L.append("| 区间 | 腿 | SURV CAGR | PIT CAGR | 差(pp) | SURV Sharpe | PIT Sharpe |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in dper.iterrows():
        f = lambda x: "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) \
            else f"{x*100:.2f}%"
        sh = lambda x: "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) \
            else f"{x:.3f}"
        L.append(f"| {r['区间']} | {r['腿']} | {f(r['SURV_CAGR'])} | "
                 f"{f(r['PIT_CAGR'])} | {r['差pp']} | {sh(r['SURV_Sharpe'])} | "
                 f"{sh(r['PIT_Sharpe'])} |")
    L.append("")

    L.append("## 6. 两条腿是不是同一条曲线\n")
    L.append(f"- corr(Alpha 日收益, SURV ~ PIT) = **{corr_alpha:.3f}**")
    L.append(f"- corr(等权 日收益, SURV ~ PIT) = **{corr_ew:.3f}**\n")
    L.append("相关性高（>0.9）说明差别主要在**水位**；"
             "相关性低说明连**择股结果**都被 survivorship 改写了。\n")

    L.append("## 7. 本测的口径声明（不能省略）\n")
    pct = 30 / max(close_p.shape[1], 1) * 100
    L.append(f"1. `top_n=30` 在 PIT 大池里不再是「10% 池」"
             f"（{close_p.shape[1]} 只里取 30 = {pct:.1f}%）。这是**刻意保留**的 —— "
             f"本测比的是「同一个策略在两个池子上的差」，不是给大池重新寻优。\n")
    L.append("2. 未剔除历史 ST（akshare 无历史 ST 序列）。PIT 池里含 ST 股，"
             "低价/反转类因子方向会偏乐观；本 Alpha 是动量，受 ST 影响相对小。\n")
    L.append("3. 未建模涨跌停可成交性 / 停牌无法卖出（属执行层，P12-5 处理）。\n")
    L.append("4. 新浪日线是否复权沿用 P11-C 口径，未重新校验 —— "
             "两侧口径相同，故**差值**不受影响。\n")
    L.append("5. 候选超集里「当前存活全 A」是**今天**的名单，加上退市股后，"
             "历史上曾上市的 A 股基本齐了；"
             "但 2005 年前上市、今天仍存活的公司若从未进过任何名单，仍可能遗漏。\n")

    path = os.path.join(OUT, "p12_survivorship.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
