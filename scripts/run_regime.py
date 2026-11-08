"""跑道氏 V0.1 原书复刻版，输出事件库 CSV + Markdown 报告.

    python scripts/run_regime.py                       # 默认 沪深300 + 中证500
    python scripts/run_regime.py 1                     # 用第 2 组指数配对
    python scripts/run_regime.py 0 --refresh           # 强制刷新数据
    python scripts/run_regime.py 0 --version v0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from adapters.dow_literal import freeze_report  # noqa: E402
from config.dow_params import get_params  # noqa: E402
from core.regime_engine import analyze_pair, latest_state  # noqa: E402
from data.loader import load_index_pair  # noqa: E402
from data.universe import get_pair  # noqa: E402
from report.regime_report import render_report  # noqa: E402
from research.event_store import save_event_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="?", default=0, help="指数配对序号或名称")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--no-volume", action="store_true")
    args = ap.parse_args()

    pair_key = int(args.pair) if str(args.pair).isdigit() else args.pair
    name, code_a, code_b = get_pair(pair_key)
    print(f"[INFO] 指数配对: {name} ({code_a} / {code_b})")

    df_a, df_b = load_index_pair(code_a, code_b, refresh=args.refresh)
    if df_a.empty or df_b.empty:
        print("[ERROR] 指数数据为空，请检查网络或缓存")
        return 1

    params = get_params(args.version)
    print(f"[INFO] 参数版本: {params.name}")
    if params.is_literal:
        print("[INFO] 已确认为原书复刻冻结参数（V0.1），未做任何调优")

    df, meta = analyze_pair(df_a, df_b, params, with_volume=not args.no_volume)
    meta.update({"pair": name, "code_a": code_a, "code_b": code_b})

    csv_path = save_event_store(df, tag=f"regime_{code_a}_{code_b}_{args.version}")
    print(f"[OK] 事件库已保存: {csv_path}  ({len(df)} 行)")

    md_path = ROOT / "out" / f"report_{code_a}_{code_b}_{args.version}.md"
    render_report(df, meta, out_path=str(md_path))
    print(f"[OK] 报告已保存: {md_path}")

    st = latest_state(df, params.name)
    print("\n=== 最新市场状态 ===")
    print(json.dumps(st.as_row(), ensure_ascii=False, indent=2, default=str))

    print("\n=== 冻结参数 ===")
    print(freeze_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
