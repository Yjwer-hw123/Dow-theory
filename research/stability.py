"""P11-A / P11-B：Dow Regime 的**时间稳定性**检验.

这一层不做任何优化，只回答一个问题：

    这个市场环境划分，是不是只在全样本平均意义上成立，
    而在某些年代完全反过来？

三种切法，互为对照：

- **P11-B 分区间**：2005-2010 / 2011-2015 / 2016-2020 / 2021-2026
- **P11-A 逐年**   ：每一年单独算，看符号翻转频率（walk-forward 的最小形式）
- **P11-A 滚动**   ：以 5 年滚动窗口算 RISK_ON−RISK_OFF 价差，看是否长期同号

判定标准（事先写死，不事后调整）：

1. `UP.fwd60 > DOWN.fwd60` 必须在 **每个** 区间成立。若某区间反号 → 该状态划分
   不具备跨期稳定性，不能进入组合层。
2. `TRANSITION` 的 fwd60 应当**不稳定**（时而高于 UP、时而低于 DOWN），
   这正是"未确认、不可交易"的形式化含义。若它长期稳定地介于两者之间，
   反而说明它是可用的中间态。
3. `RISK_ON − RISK_OFF` 价差的符号翻转次数是稳定性的直接度量：翻转越多，
   说明"事后好看"越可能来自全样本平均，而不是可复现的规律。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

HORIZONS = (20, 60, 120)
PERIODS = [
    ("2005-2010", "2005-01-01", "2010-12-31"),
    ("2011-2015", "2011-01-01", "2015-12-31"),
    ("2016-2020", "2016-01-01", "2020-12-31"),
    ("2021-2026", "2021-01-01", "2026-12-31"),
]
STATES = ("RISK_ON", "WAIT", "RISK_OFF")
TRENDS = ("UP", "TRANSITION", "DOWN")


# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """补 fwd 收益列（用 close_a，即该配对的 A 指数）."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    for h in horizons:
        d[f"fwd{h}"] = d["close_a"].shift(-h) / d["close_a"] - 1.0
    return d


def _stat(g: pd.DataFrame, col: str) -> tuple[float | None, float | None]:
    v = g[col].dropna()
    if not len(v):
        return None, None
    return round(float(v.mean()), 4), round(float((v > 0).mean()), 3)


def _slice(d: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return d[(d["date"] >= start) & (d["date"] <= end)]


# --------------------------------------------------------------------------
def period_state_table(d: pd.DataFrame) -> pd.DataFrame:
    """P11-B：分区间 × 状态 × 前瞻收益."""
    rows = []
    for label, s, e in PERIODS:
        sub = _slice(d, s, e)
        if not len(sub):
            continue
        for st in STATES:
            g = sub[sub["dow_signal"] == st]
            if not len(g):
                continue
            row = {"period": label, "state": st, "days": len(g),
                   "share": round(len(g) / len(sub), 3)}
            for h in HORIZONS:
                m, w = _stat(g, f"fwd{h}")
                row[f"fwd{h}_mean"] = m
                row[f"fwd{h}_win"] = w
            rows.append(row)
    return pd.DataFrame(rows)


def period_trend_table(d: pd.DataFrame) -> pd.DataFrame:
    """P11-B 核心：UP / TRANSITION / DOWN 在每区的 fwd60，检查单调性."""
    rows = []
    for label, s, e in PERIODS:
        sub = _slice(d, s, e)
        if not len(sub):
            continue
        row = {"period": label, "days": len(sub)}
        for tr in TRENDS:
            g = sub[sub["primary_trend"] == tr]
            m, w = _stat(g, "fwd60")
            m20, _ = _stat(g, "fwd20")
            m120, _ = _stat(g, "fwd120")
            row[f"{tr}_days"] = len(g)
            row[f"{tr}_fwd20"] = m20
            row[f"{tr}_fwd60"] = m
            row[f"{tr}_fwd120"] = m120
            row[f"{tr}_win"] = w
        up, dn = row.get("UP_fwd60"), row.get("DOWN_fwd60")
        row["up_minus_down"] = (round(up - dn, 4) if up is not None and dn is not None else None)
        row["up_gt_down"] = bool(up is not None and dn is not None and up > dn)
        trans = row.get("TRANSITION_fwd60")
        row["trans_between"] = (
            bool(up is not None and dn is not None and trans is not None and dn <= trans <= up)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_table(d: pd.DataFrame) -> pd.DataFrame:
    """P11-A 最小形式：逐年，看符号翻转."""
    d = d.copy()
    d["year"] = d["date"].dt.year
    rows = []
    for y, g in d.groupby("year"):
        if len(g) < 60:
            continue
        on = g[g["dow_signal"] == "RISK_ON"]["fwd60"].dropna()
        off = g[g["dow_signal"] == "RISK_OFF"]["fwd60"].dropna()
        wait = g[g["dow_signal"] == "WAIT"]["fwd60"].dropna()
        mkt = g["fwd60"].dropna()
        mkt_mean = float(mkt.mean()) if len(mkt) else None
        wait_mean = float(wait.mean()) if len(wait) else None
        row = {
            "year": int(y), "days": len(g),
            "RISK_ON_days": int((g["dow_signal"] == "RISK_ON").sum()),
            "RISK_OFF_days": int((g["dow_signal"] == "RISK_OFF").sum()),
            "RISK_ON_fwd60": round(float(on.mean()), 4) if len(on) else None,
            "WAIT_fwd60": round(wait_mean, 4) if wait_mean is not None else None,
            "RISK_OFF_fwd60": round(float(off.mean()), 4) if len(off) else None,
            "mkt_fwd60": round(mkt_mean, 4) if mkt_mean is not None else None,
        }
        # 逐年价差在大多数年份算不出来：RISK_ON / RISK_OFF 是按年交替的状态，
        # 同一年内极少共存。所以基准换成**同年的 WAIT**（每年都有），
        # 这才是"状态相对中性态有无信息"的正确对照。
        row["on_minus_wait"] = (round(float(on.mean()) - wait_mean, 4)
                                if len(on) and wait_mean is not None else None)
        row["off_minus_wait"] = (round(float(off.mean()) - wait_mean, 4)
                                 if len(off) and wait_mean is not None else None)
        row["spread"] = (round(float(on.mean() - off.mean()), 4)
                         if len(on) and len(off) else None)
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_spread(d: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """P11-A：滚动窗口的 RISK_ON−RISK_OFF fwd60 价差（用日频滚动平均近似）."""
    d = d.copy()
    win = int(years * 243)
    on = (d["dow_signal"] == "RISK_ON")
    off = (d["dow_signal"] == "RISK_OFF")
    f = d["fwd60"]
    # 滚动窗口内按状态分组求均值 → 用掩码后的滚动和/计数实现
    on_sum = (f.where(on, 0.0)).rolling(win, min_periods=win // 3).sum()
    on_cnt = on.astype(float).rolling(win, min_periods=win // 3).sum()
    off_sum = (f.where(off, 0.0)).rolling(win, min_periods=win // 3).sum()
    off_cnt = off.astype(float).rolling(win, min_periods=win // 3).sum()
    out = pd.DataFrame({
        "date": d["date"],
        "on_mean": on_sum / on_cnt.replace(0, np.nan),
        "off_mean": off_sum / off_cnt.replace(0, np.nan),
    })
    out["spread"] = out["on_mean"] - out["off_mean"]
    return out.dropna(subset=["spread"])


def half_sample(d: pd.DataFrame) -> pd.DataFrame:
    """前后半样本对照（最粗的稳定性检验）."""
    mid = d["date"].iloc[len(d) // 2]
    rows = []
    for label, sub in (("前半", d[d["date"] < mid]), ("后半", d[d["date"] >= mid])):
        row = {"half": label, "from": str(sub["date"].iloc[0].date()),
               "to": str(sub["date"].iloc[-1].date()), "days": len(sub)}
        for st in STATES:
            g = sub[sub["dow_signal"] == st]
            m, w = _stat(g, "fwd60")
            row[f"{st}_share"] = round(len(g) / len(sub), 3)
            row[f"{st}_fwd60"] = m
            row[f"{st}_win"] = w
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def render(d: pd.DataFrame, pair_label: str) -> str:
    ps = period_state_table(d)
    pt = period_trend_table(d)
    yr = yearly_table(d)
    rs = rolling_spread(d)
    hs = half_sample(d)

    L: list[str] = []
    L.append(f"# P11-A/B 市场环境的时间稳定性 —— {pair_label}\n")
    L.append("Dow-Literal-V0.1（冻结版）。本报告**不含任何参数调整**，只做分组统计。\n")
    L.append(f"样本：{d['date'].iloc[0].date()} ~ {d['date'].iloc[-1].date()}，"
             f"{len(d)} 个交易日。前瞻收益用 A 指数收盘价。\n")

    # ---- 1. 分区间：状态 ----
    L.append("## 1. 分区间：RISK_ON / WAIT / RISK_OFF 的条件收益（P11-B）\n")
    L.append("| 区间 | 状态 | 天数 | 占比 | fwd20 | fwd20胜率 | fwd60 | fwd60胜率 | fwd120 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ps.iterrows():
        L.append(
            f"| {r['period']} | {r['state']} | {r['days']} | {r['share']:.1%} | "
            f"{_f(r['fwd20_mean'])} | {_f(r['fwd20_win'],3)} | **{_f(r['fwd60_mean'])}** | "
            f"**{_f(r['fwd60_win'],3)}** | {_f(r['fwd120_mean'])} |"
        )
    L.append("")

    # ---- 1b. 相对中性态的超额：信息含量衰减 ----
    L.append("### 1b. 状态相对 WAIT 的超额 —— 信息含量是否在衰减（核心）\n")
    L.append("| 区间 | ON−WAIT (fwd60) | OFF−WAIT (fwd60) | 状态天数占比 ON/WAIT/OFF |")
    L.append("|---|---:|---:|---|")
    for label, _s, _e in PERIODS:
        sub = ps[ps["period"] == label]

        def _gv(st: str):
            r = sub[sub["state"] == st]
            if not len(r):
                return None
            return float(r["fwd60_mean"].iloc[0]), float(r["share"].iloc[0])

        on_v = _gv("RISK_ON")
        wt_v = _gv("WAIT")
        off_v = _gv("RISK_OFF")
        wt_mean = wt_v[0] if wt_v else None
        on_diff = (on_v[0] - wt_mean) if (on_v and wt_mean is not None) else None
        off_diff = (off_v[0] - wt_mean) if (off_v and wt_mean is not None) else None
        sh = "/".join(
            f"{(v[1]*100):.0f}%" if v else "-" for v in (on_v, wt_v, off_v)
        )
        L.append(f"| {label} | {_f(on_diff)} | {_f(off_diff)} | {sh} |")
    L.append("")
    L.append("> 若这一行从早期区间到近期区间单调缩小，说明\"状态有信息\"这件事本身"
             "在衰减 —— 那么它作为**组合风险控制变量**的价值也随之衰减。\n")
    L.append("## 2. 分区间：UP / TRANSITION / DOWN 的 fwd60（P11-B 核心）\n")
    L.append("| 区间 | UP | TRANSITION | DOWN | UP−DOWN | UP>DOWN | TRANS 居中 |")
    L.append("|---|---:|---:|---:|---:|:--:|:--:|")
    for _, r in pt.iterrows():
        L.append(
            f"| {r['period']} | {_f(r['UP_fwd60'])} | {_f(r['TRANSITION_fwd60'])} | "
            f"{_f(r['DOWN_fwd60'])} | {_f(r['up_minus_down'])} | "
            f"{'✅' if r['up_gt_down'] else '❌'} | {'是' if r['trans_between'] else '否'} |"
        )
    L.append("")
    n_ok = int(pt["up_gt_down"].sum())
    L.append(f"**UP>DOWN 成立区间数：{n_ok}/{len(pt)}**"
             f"{'  —— 跨期方向稳定' if n_ok == len(pt) else '  —— ⚠️ 存在反号区间'}\n")

    # ---- 3. 逐年 ----
    L.append("## 3. 逐年（P11-A 最小 walk-forward 形式）\n")
    L.append("> RISK_ON 与 RISK_OFF 是**按年交替**的状态，同一年内极少共存，"
             "所以\"同年价差\"多数年份算不出来。基准换成**同年的 WAIT**（每年都有）——"
             "这才是\"状态相对中性态有无信息\"的正确对照。\n")
    L.append("| 年份 | 天数 | ON 天 | OFF 天 | ON fwd60 | ON−WAIT | WAIT fwd60 | OFF fwd60 | OFF−WAIT |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in yr.iterrows():
        L.append(
            f"| {int(r['year'])} | {int(r['days'])} | {int(r['RISK_ON_days'])} | "
            f"{int(r['RISK_OFF_days'])} | {_f(r['RISK_ON_fwd60'])} | {_f(r['on_minus_wait'])} | "
            f"{_f(r['WAIT_fwd60'])} | {_f(r['RISK_OFF_fwd60'])} | {_f(r['off_minus_wait'])} |"
        )
    L.append("")
    onw = yr["on_minus_wait"].dropna()
    offw = yr["off_minus_wait"].dropna()
    if len(onw):
        L.append(f"- RISK_ON 与同年 WAIT 可比的年份 **{len(onw)}** 年："
                 f"ON 更高 **{int((onw>0).sum())}** 年，均值 {onw.mean():+.4f}，中位数 {onw.median():+.4f}")
    if len(offw):
        L.append(f"- RISK_OFF 与同年 WAIT 可比的年份 **{len(offw)}** 年："
                 f"OFF 更低 **{int((offw<0).sum())}** 年，均值 {offw.mean():+.4f}，中位数 {offw.median():+.4f}")
    L.append("")

    # ---- 4. 滚动 ----
    L.append("## 4. 5 年滚动窗口价差（P11-A）\n")
    if len(rs):
        L.append("| 窗口结束日 | RISK_ON 均值 | RISK_OFF 均值 | 价差 |")
        L.append("|---|---:|---:|---:|")
        step = max(1, len(rs) // 12)
        for _, r in rs.iloc[::step].iterrows():
            L.append(f"| {r['date'].date()} | {r['on_mean']:.4f} | {r['off_mean']:.4f} | "
                     f"{r['spread']:.4f} |")
        L.append("")
        pos = float((rs["spread"] > 0).mean())
        L.append(f"滚动窗口内价差为正的比例：**{pos:.1%}**（{len(rs)} 个窗口）\n")
    else:
        L.append("（样本不足，跳过）\n")

    # ---- 5. 前后半 ----
    L.append("## 5. 前后半样本对照\n")
    L.append("| 半样本 | 起 | 止 | 天数 | RISK_ON 占比 | RISK_ON fwd60 | RISK_OFF 占比 | RISK_OFF fwd60 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for _, r in hs.iterrows():
        L.append(
            f"| {r['half']} | {r['from']} | {r['to']} | {r['days']} | {r['RISK_ON_share']:.1%} | "
            f"{_f(r['RISK_ON_fwd60'])} | {r['RISK_OFF_share']:.1%} | {_f(r['RISK_OFF_fwd60'])} |"
        )
    L.append("")

    # ---- 6. 判定 ----
    L.append("## 6. 判定\n")
    checks = []
    checks.append(("UP>DOWN 在所有区间成立", n_ok == len(pt)))
    if len(onw):
        checks.append((f"RISK_ON 优于同年 WAIT（{int((onw>0).sum())}/{len(onw)} 年）",
                       float((onw > 0).mean()) > 0.5))
    if len(offw):
        checks.append((f"RISK_OFF 差于同年 WAIT（{int((offw<0).sum())}/{len(offw)} 年）",
                       float((offw < 0).mean()) > 0.5))
    if len(rs):
        checks.append(("5 年滚动窗口价差为正过半", float((rs["spread"] > 0).mean()) > 0.5))
    for name, ok in checks:
        L.append(f"- {'✅' if ok else '❌'} {name}")
    L.append("")
    L.append("> 注意：本节只回答\"时间上稳不稳\"，**不回答\"能不能赚钱\"**。"
             "能不能赚到钱是 P11-C 的问题。\n")
    return "\n".join(L)


def _f(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# --------------------------------------------------------------------------
def run(pair_index: int = 0, tag: str = "") -> tuple[str, dict]:
    from data.loader import load_index_pair  # noqa: PLC0415

    pairs = [
        ("000300", "000905", "沪深300 + 中证500"),
        ("000001", "399001", "上证指数 + 深证成指"),
        ("000300", "000510", "沪深300 + 中证A500"),
    ]
    a_code, b_code, label = pairs[pair_index]
    csv = os.path.join(OUT, f"regime_{a_code}_{b_code}_v0.1.csv")
    if not os.path.exists(csv):
        # 兜底：重新生成
        from config.dow_params import get_params  # noqa: PLC0415
        from core.regime_engine import analyze_pair  # noqa: PLC0415

        a, b = load_index_pair(a_code, b_code)
        df, _ = analyze_pair(a, b, get_params("v0.1"), strict_invariants=False)
    else:
        df = pd.read_csv(csv)

    d = prepare(df)
    md = render(d, label)
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(OUT, f"p11_stability{suffix}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)

    pt = period_trend_table(d)
    yr = yearly_table(d)
    rs = rolling_spread(d)
    onw = yr["on_minus_wait"].dropna()
    offw = yr["off_minus_wait"].dropna()
    meta = {
        "pair": label,
        "up_gt_down_periods": f"{int(pt['up_gt_down'].sum())}/{len(pt)}",
        "on_beat_wait_years": f"{int((onw>0).sum())}/{len(onw)}" if len(onw) else "-",
        "off_below_wait_years": f"{int((offw<0).sum())}/{len(offw)}" if len(offw) else "-",
        "rolling_pos_ratio": (round(float((rs['spread'] > 0).mean()), 3) if len(rs) else None),
    }
    return path, meta
