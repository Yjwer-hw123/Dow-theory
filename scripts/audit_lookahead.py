"""P10.1-C 驱动：对所有指数配对跑 Prefix Invariance 审计.

    python scripts/audit_lookahead.py            # 默认配对，24 个审计点
    python scripts/audit_lookahead.py --n 8      # 快速诊断
    python scripts/audit_lookahead.py --all      # 全部配对
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.dow_params import get_params  # noqa: E402
from data.loader import load_index_pair  # noqa: E402
from data.universe import PAIRS, get_pair  # noqa: E402
from research.lookahead_audit import (  # noqa: E402
    format_report,
    run_prefix_invariance,
)


def audit_one(name: str, code_a: str, code_b: str, n: int, version: str,
              seed: int, quiet: bool = False) -> dict:
    df_a, df_b = load_index_pair(code_a, code_b)
    if df_a.empty or df_b.empty:
        print(f"[WARN] {name}: 数据为空，跳过")
        return {"pair": name, "skipped": True}
    params = get_params(version)

    def _progress(done, total, rec):
        if quiet:
            return
        flag = "OK " if not rec["n_diffs"] and not rec["error"] else "DIFF"
        detail = ""
        if rec["error"]:
            detail = f" ERROR {rec['error'][:60]}"
        elif rec["diffs"]:
            fd = rec["first_diff"]
            detail = f" {fd['field']}: full={fd['full']!r} prefix={fd['prefix']!r}"
        print(f"  [{done:>3}/{total}] {rec['date']} {flag}{detail}")

    rows, summary = run_prefix_invariance(
        df_a, df_b, params, n_samples=n, seed=seed, progress=_progress
    )
    print(f"\n### {name} ({code_a}/{code_b})  {len(df_a)} 交易日")
    print(format_report(rows, summary))
    print()
    summary["pair"] = name
    summary["code_a"] = code_a
    summary["code_b"] = code_b
    summary["rows"] = rows
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="?", default=0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    targets = PAIRS if args.all else [get_pair(
        int(args.pair) if str(args.pair).isdigit() else args.pair)]

    all_bad = 0
    for name, ca, cb in targets:
        s = audit_one(name, ca, cb, args.n, args.version, args.seed, quiet=args.quiet)
        if not s.get("skipped"):
            all_bad += s["n_inconsistent"]

    print("=" * 72)
    if all_bad == 0:
        print("[PASS] 所有配对、所有审计点：prefix == full（无第二层前瞻偏差）")
        return 0
    print(f"[FAIL] 合计 {all_bad} 个审计点不一致 —— 存在前瞻偏差，需定位并修复")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
