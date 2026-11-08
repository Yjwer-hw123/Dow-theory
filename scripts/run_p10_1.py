"""P10.1 驱动：空仓归因 / 闸门解除敏感性 / TRANSITION 子状态稳健性.

    python scripts/run_p10_1.py
    python scripts/run_p10_1.py 1        # 用第 2 组指数配对

产物：
    out/regime_<a>_<b>_v0.1.csv              含 blocker 列的事件库
    out/p10_1_attribution.md                 三块归因
    out/p10_1_attribution.csv                同上（机器可读）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.dow_params import get_params  # noqa: E402
from core.regime_engine import analyze_pair  # noqa: E402
from data.loader import load_index_pair  # noqa: E402
from data.universe import get_pair  # noqa: E402
from research.attribution import (  # noqa: E402
    blocker_table,
    gate_variants,
    render_attribution_md,
    transition_stability,
)
from research.event_store import save_event_store  # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="?", default=0)
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pair_key = int(args.pair) if str(args.pair).isdigit() else args.pair
    name, code_a, code_b = get_pair(pair_key)
    print(f"[P10.1] 指数配对：{name} ({code_a}/{code_b})")

    df_a, df_b = load_index_pair(code_a, code_b)
    if df_a.empty or df_b.empty:
        print("[ERROR] 数据为空")
        return 1

    params = get_params(args.version)
    engine, meta = analyze_pair(df_a, df_b, params)
    print(f"[P10.1] 参数版本 {params.name}  样本 {len(engine)} 交易日")
    print(f"[P10.1] 信号分布：{meta['signal_distribution']}")
    print(f"[P10.1] 阻断原因分布：{meta.get('blocker_distribution')}")
    print(f"[P10.1] 不变量违规：{meta['invariants'].get('total', 0)}")

    tag = args.tag or f"regime_{code_a}_{code_b}_{args.version}"
    save_event_store(engine, tag=tag)

    # 三块归因
    bt = blocker_table(engine, args.horizon)
    gv = gate_variants(engine, args.horizon)
    ts = transition_stability(engine, args.horizon)

    suffix = f"_{args.tag}" if args.tag else ""
    md_path = OUT / f"p10_1_attribution{suffix}.md"
    md_path.write_text(
        render_attribution_md(
            engine,
            title=f"P10.1 归因报告（{name}）",
            horizon=args.horizon,
            extra=(
                "## 读法\n\n"
                "- A 表回答「12% 的持仓到底是被谁压出来的」："
                "`share_of_no_position` 最大的那几行就是主要瓶颈。\n"
                "- B 表是**诊断**，不是策略候选。`vs_base_days` 说明撤掉某道闸门能"
                "多出多少天暴露，`fwd 均值` 说明这些多出来的天数质量如何。\n"
                "- D 表用分年度胜率检验 T2/T3 的正收益是否稳定："
                "如果 `years_win_gt50` 远小于 `years_with_data`，就是少数年份贡献的，不可写进规则。\n"
                "- **本报告不改变任何参数。** 任何规则改动都要走「规则语义研究」，"
                "不能从这张表里挑一行抄。\n"
            ),
        ),
        encoding="utf-8",
    )
    print(f"[OK] {md_path}")

    bt.to_csv(OUT / f"p10_1_blockers{suffix}.csv", index=False, encoding="utf-8-sig")
    gv.to_csv(OUT / f"p10_1_gates{suffix}.csv", index=False, encoding="utf-8-sig")
    ts.to_csv(OUT / f"p10_1_transition{suffix}.csv", index=False, encoding="utf-8-sig")

    print("\n--- A. 空仓归因 ---")
    print(bt.to_string(index=False))
    print("\n--- B. 闸门解除敏感性（诊断，非策略） ---")
    print(gv.to_string(index=False))
    print("\n--- D. TRANSITION 子状态稳健性 ---")
    print(ts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
