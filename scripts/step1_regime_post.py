"""STEP 1 后处理：按基准年度收益重建市场环境表。

原脚本的月度 MA200 状态分类把“MA200 水平”和“MA200 斜率变化”比较错了，
导致没有 bear 状态。这里不重跑全池，只用已落盘的年度统计 + 399317 年度收益，
按年度牛/熊/震荡重新聚合。年均值/正收益比例可按样本数加权还原；中位数只报告
“年度中位数均值”，不冒充全样本中位数。
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "step1")

# 基准年度收益（399317，国证A指）
import akshare as ak  # noqa: E402

bm = ak.stock_zh_index_daily_em(symbol="sz399317")
bm["date"] = pd.to_datetime(bm["date"])
bm = bm[bm["date"] <= pd.Timestamp("2026-09-30")].sort_values("date")
ye = bm.groupby(bm["date"].dt.year)["close"].last()
ret = (ye / ye.shift(1) - 1).to_dict()
# 2005 只有 2 月起的数据，不计入年度环境
ret.pop(2005, None)

def regime(y: int) -> str:
    r = ret.get(y)
    if r is None or not np.isfinite(r):
        return "unknown"
    if r > 0.10:
        return "bull"
    if r < -0.10:
        return "bear"
    return "range"

# 年度统计
a = pd.read_csv(os.path.join(OUT, "annual_stats.csv"))
a["regime"] = a["year"].map(regime)
a = a[a["regime"] != "unknown"]

rows = []
for (reg, g, h), sub in a.groupby(["regime", "group", "horizon"]):
    n = int(sub["n"].sum())
    if n <= 0:
        continue
    mean = float((sub["mean"] * sub["n"]).sum() / n)
    pos = float((sub["positive_rate"] * sub["n"]).sum() / n)
    ex_sub = sub.dropna(subset=["excess_mean"])
    ex = float((ex_sub["excess_mean"] * ex_sub["n"]).sum() / ex_sub["n"].sum()) if len(ex_sub) else None
    med_avg = float((sub["median"] * sub["n"]).sum() / n)
    rows.append({
        "regime": reg, "group": g, "horizon": int(h),
        "years": int(sub["year"].nunique()),
        "year_list": "|".join(str(int(y)) + ("YTD" if y == 2026 else "") for y in sorted(sub["year"].unique())),
        "n": n, "mean": mean, "avg_annual_median": med_avg,
        "positive_rate": pos, "excess_mean": ex,
    })

r = pd.DataFrame(rows)
r.to_csv(os.path.join(OUT, "regime_annual.csv"), index=False, encoding="utf-8-sig")

# 更新 summary：删除错误的月度 regime，写入年度 regime
s = json.load(open(os.path.join(OUT, "step1_summary.json"), encoding="utf-8"))
s.pop("regime", None)
s["regime_annual"] = rows
s["meta"]["market_regime"] = (
    "按 399317 年度收益划分：>+10% 牛、<-10% 熊、其余震荡；"
    "2005 年因基准仅有 2 月起数据不计入；2026 为 YTD。"
    "均值和正收益比例按样本数加权还原；中位数为年度中位数的样本加权均值，不是全样本中位数。"
)
json.dump(s, open(os.path.join(OUT, "step1_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# 删除错误的月度 regime 明细，避免误用
p = os.path.join(OUT, "regime_stats.csv")
if os.path.exists(p):
    os.remove(p)

print(r[r["horizon"].isin([20, 60, 120])][
    ["regime", "group", "horizon", "n", "mean", "positive_rate", "excess_mean", "year_list"]
].to_string(index=False))
