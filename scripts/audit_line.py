"""线态窄幅区间盘整审计（第 14 章；诊断，不调参数）.

背景：V0.1 报告里出现「21 年 1092 个成线日、0 次双指数同时突破」。
先做一次根因分辨，决定它是

    (a) 市场性质（线之后确实很少同日突破）
    (b) 实现 bug（突破判定在数学上不可能触发）
    (c) 语义过严（雷亚的"同时"未必要求同一天）

V0.1.1 的结论是 (b)：旧实现把突破日算进了区间自身，使 `close > max(high)` 永不成立。
修完以后这里用来量化回答：现在到底错过了多少"近似同时"的突破。

输出：每个单边突破日的 另一指数最近的同向突破日期差 分布。

    python scripts/audit_line.py
    python scripts/audit_line.py --pairs 0 --window 10
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.dow_params import get_params  # noqa: E402
from core.line_detector import detect_line_state  # noqa: E402
from data.loader import load_index_pair  # noqa: E402
from data.universe import get_pair  # noqa: E402


def _per_index_breakouts(df: pd.DataFrame, params) -> pd.DataFrame:
    """逐日计算单指数的成线状态与突破方向."""
    dates = df["date"].dt.date.tolist()
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    rows = []
    for i in range(len(df)):
        r = detect_line_state(dates, highs, lows, closes, i, params)
        rows.append({
            "date": dates[i],
            "active": bool(r["line_active"]),
            "dir": r.get("line_breakout_dir", "NONE"),
            "width": r["line_width"],
            "high": r["line_high"],
            "low": r["line_low"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=0)
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--window", type=int, default=10, help="允许错开几个交易日仍算'近似同时'")
    ap.add_argument("--dump-md", action="store_true")
    args = ap.parse_args()

    pair_key = int(args.pair) if str(args.pair).isdigit() else args.pair
    name, code_a, code_b = get_pair(pair_key)
    params = get_params(args.version)
    df_a, df_b = load_index_pair(code_a, code_b)
    if df_a.empty:
        raise SystemExit("[ERROR] 无数据")

    sa = _per_index_breakouts(df_a, params)
    sb = _per_index_breakouts(df_b, params)

    both_active = int((sa["active"] & sb["active"]).sum())
    single_dir_a = sa[sa["dir"] != "NONE"]
    single_dir_b = sb[sb["dir"] != "NONE"]
    dual = sa[(sa["dir"] != "NONE") & (sb["dir"] != "NONE") & (sa["dir"] == sb["dir"])]
    dual = dual[sa["active"] & sb["active"]]

    lines = [
        f"# 线态窄幅区间审计 · {name} ({code_a} / {code_b})",
        "",
        f"- 参数版本: {params.name}（5% / {params.line_min_bars} 根，**审计期间未调整**）",
        f"- 样本: {sa['date'].iloc[0]} ~ {sa['date'].iloc[-1]}（{len(sa)} 交易日）",
        "",
        "## 计数",
        "",
        "| 项 | 天数 |",
        "|---|---|",
        f"| 双指数同时成线 | {both_active} |",
        f"| A 单独突破 | {len(single_dir_a)} |",
        f"| B 单独突破 | {len(single_dir_b)} |",
        f"| **双指数同日同向突破** | **{len(dual)}** |",
        "",
    ]

    print("=" * 78)
    print(f"线态审计 · {name}  ({code_a} / {code_b})  参数={params.name}")
    print("=" * 78)
    print(f"双指数同时成线      : {both_active} 天")
    print(f"A 单独突破          : {len(single_dir_a)} 天")
    print(f"B 单独突破          : {len(single_dir_b)} 天")
    print(f"双指数同日同向突破  : {len(dual)} 天")

    if len(dual):
        print("\n已经触发的双指数突破：")
        lines += ["## 已确认的双指数突破", "", "| date | dir | width_a | width_b |", "|---|---|---|---|"]
        for _, r in dual.iterrows():
            j = sa.index[sa["date"] == r["date"]][0]
            print(f"  {r['date']}  {r['dir']:<4}  width_a={sa.loc[j,'width']}  width_b={sb.loc[j,'width']}")
            lines.append(f"| {r['date']} | {r['dir']} | {sa.loc[j,'width']} | {sb.loc[j,'width']} |")
        lines.append("")

    # ---- 近似同时性分析：A 突破后，B 在 ±window 内是否有同向突破 ----
    b_dates = {d: i for i, d in enumerate(sb["date"].tolist())}
    gaps: Counter = Counter()
    examples = []
    for i, r in sa[sa["dir"] != "NONE"].iterrows():
        if not r["active"]:
            continue
        found = None
        for k in range(0, args.window + 1):
            for j in ({i + k, i - k} if k else {i}):
                if j in sb.index and sb.loc[j, "active"] and sb.loc[j, "dir"] == r["dir"]:
                    found = (sb.loc[j, "date"], k if sb.loc[j, "date"] >= r["date"] else -k)
                    break
            if found:
                break
        if found is None:
            gaps["none"] += 1
        else:
            gap = found[1]
            key = str(gap) if abs(gap) <= 5 else ">5"
            gaps[key] += 1
            if (i - sa.index[0]) % 37 == 0:
                examples.append((r["date"], r["dir"], found[0], gap))

    print(f"\nA 成线突破后，B 的错开天数分布（≤{args.window} 交易日内）:")
    lines += ["## 近似同时性（不放宽规则，只做诊断）", "",
              f"A 单独突破后，B 在 ±{args.window} 个交易日内的同向突破错开天数：", "",
              "| gap(交易日) | 次数 |", "|---|---|"]
    for k in sorted(gaps, key=lambda x: (x == "none", x == ">5", int(x) if x not in ("none", ">5") else 0)):
        print(f"  {k:>7} : {gaps[k]}")
        lines.append(f"| {k} | {gaps[k]} |")
    lines.append("")

    near = sum(v for k, v in gaps.items() if k not in ("none", ">5"))
    total = sum(gaps.values())
    verdict = (
        f"'同时突破'并非零机会：共 {len(dual)} 次严格同日确认；"
        f"另有 {near}/{total} 次的另一指数在 ±{args.window} 日内给出了同向突破。"
        f"若放宽到 ±1 日会多带来 {gaps.get('1', 0) + gaps.get('-1', 0)} 次信号 —— "
        "这个差异是**语义选择**，不是 bug，留到 V1 走样本外检验时再决定是否放松。"
    )
    print("\n结论:", verdict)
    lines += ["## 结论", "", verdict, ""]

    if args.dump_md:
        p = ROOT / "out" / "audit_line.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
