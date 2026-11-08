"""次级反应 lineage 审计（诊断，不调参数）.

目的：回答一个问题 —— 一个超长 / 超大回撤的次级反应，到底是

    (a) 合理的持续状态（原趋势还在，回调只是拖得久），还是
    (b) 旧腿粘连（reaction 挂在一个早就该作废的主要运动行程上）

打印完整 lineage：

    PRIMARY_LEG #id
      trend / origin(date,price) / extreme(date,price) / structure_ref
      REACTION #id-R1
        start / extreme(date,price) / current / retracement / duration
        invalidated_by: date + 参照价

用法：
    python scripts/audit_lineage.py
    python scripts/audit_lineage.py --tag regime_000300_000905_v0.1 --top 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

OUT_DIR = ROOT / "out"


def _load(tag: str) -> pd.DataFrame:
    path = OUT_DIR / f"{tag}.csv"
    if not path.exists():
        raise SystemExit(f"[ERROR] 找不到事件库 {path}，请先跑 scripts/run_regime.py")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _d(v):
    if v is None or pd.isna(v):
        return "-"
    return pd.to_datetime(v).date()


def lineage_block(df: pd.DataFrame, row, close_now: float | None = None) -> str:
    return "\n".join(
        [
            f"PRIMARY_LEG #L{int(row['primary_leg_id'])}",
            f"  trend            : {row['primary_trend']}",
            f"  leg origin       : {_d(row['leg_origin_date'])}  @ {_fmt(row['leg_origin_price'])}",
            f"  leg extreme      : {_d(row['leg_extreme_date'])}  @ {_fmt(row['leg_extreme_price'])}",
            f"  structure_ref    : {_fmt(row['structure_ref'])}   (integrity 判定参照)",
            f"  dual_confirmation: {row['dual_confirmation']}   last_swing={row['last_confirmed_swing']}",
            f"  H1/H2            : {_fmt(row['H1'])} / {_fmt(row['H2'])}",
            f"  L1/L2            : {_fmt(row['L1'])} / {_fmt(row['L2'])}",
            "",
            f"REACTION #{row.get('secondary_id')}",
            f"  reaction start   : {_d(row['reaction_start'])}",
            f"  extreme          : {_d(row['reaction_extreme_date'])}  @ {_fmt(row['reaction_extreme'])}",
            f"  current close    : {_fmt(close_now)}",
            f"  retracement      : {float(row['retracement_ratio']):.2%}",
            f"  duration         : {int(row['secondary_duration'])} 天",
            f"  state            : {row['secondary_reaction']}  ({row['secondary_note']})",
            f"  invalidated_date : {_d(row['invalidated_date'])}",
            f"  dow_signal       : {row['dow_signal']}",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="regime_000300_000905_v0.1")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--dump-md", action="store_true", help="同时写出 out/audit_lineage.md")
    args = ap.parse_args()

    df = _load(args.tag)
    lines = [f"# 次级反应 lineage 审计 ({args.tag})", ""]

    print("=" * 78)
    print("当前 lineage")
    print("=" * 78)
    cur = df.iloc[-1]
    block = lineage_block(df, cur, close_now=float(cur["close_a"]))
    print(block)
    lines += ["## 当前 lineage", "", "```text", block, "```", ""]

    # ---- 最长的次级反应（每个 leg 取最后一天，避免同一 leg 重复计数） ----
    d1 = df.dropna(subset=["primary_leg_id", "secondary_duration"])
    d1 = d1[d1["secondary_duration"] > 0]
    per_leg = d1.sort_values("secondary_duration").groupby("primary_leg_id").tail(1)
    top = per_leg.nlargest(args.top, "secondary_duration")

    print()
    print("=" * 78)
    print(f"最长的 {args.top} 个次级反应（每个 leg 取峰值）")
    print("=" * 78)
    lines += [f"## 最长的 {args.top} 个次级反应", "",
              "| leg | reaction | trend | duration(天) | retracement | state | invalidated | leg extremes |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        row_txt = (
            f"| L{int(r['primary_leg_id'])} | {r['secondary_id']} | {r['primary_trend']} "
            f"| {int(r['secondary_duration'])} | {float(r['retracement_ratio']):.2%} "
            f"| {r['secondary_reaction']} | {_d(r['invalidated_date'])} "
            f"| {_fmt(r['leg_origin_price'])} → {_fmt(r['leg_extreme_price'])} "
            f"({_d(r['leg_origin_date'])} ~ {_d(r['leg_extreme_date'])}) |"
        )
        lines.append(row_txt)
        print(
            f"L{int(r['primary_leg_id']):>3} {r['secondary_id']:<10} {r['primary_trend']:<10} "
            f"dur={int(r['secondary_duration']):>4}  retr={float(r['retracement_ratio']):>8.2%}  "
            f"{r['secondary_reaction']:<12} invalidated={_d(r['invalidated_date'])}"
        )

    # ---- 旧腿粘连检测：同一 leg 内 duration 不应超过 max_days 太多却仍 ACTIVE ----
    print()
    print("=" * 78)
    print("旧腿粘连检测：duration > 180 天且仍未 INVALIDATED 的 reaction")
    print("=" * 78)
    stale = d1[(d1["secondary_duration"] > 180) & (d1["secondary_reaction"] != "INVALIDATED")]
    if stale.empty:
        print("无 —— 所有超长 reaction 都已经归到 INVALIDATED（结构改动或事件END）")
        verdict = "无旧腿粘连：所有 duration>180 天的次级反应均已 INVALIDATED"
    else:
        grp = stale.groupby("primary_leg_id").agg(
            days=("secondary_duration", "max"),
            state=("secondary_reaction", "last"),
        )
        print(grp.to_string())
        verdict = f"发现 {len(grp)} 个疑似旧腿粘连（leg id: {list(grp.index)}）"
    lines += ["", "## 旧腿粘连检测", "", verdict, ""]

    # ---- leg 换手是否跟上价格变化 ----
    print()
    print("=" * 78)
    print(f"全部 leg 数: {df['primary_leg_id'].nunique()}，"
          f"最长 reaction: {int(df['secondary_duration'].max())} 天，"
          f"retracement>1 的交易日: {int((df['retracement_ratio'] > 1).sum())}")
    print("=" * 78)

    if args.dump_md:
        p = OUT_DIR / "audit_lineage.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
