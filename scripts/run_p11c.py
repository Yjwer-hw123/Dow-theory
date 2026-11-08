"""P11-C 驱动：Alpha vs Alpha + Dow Regime."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.overlay_regime import run  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--universe", default="hs300")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    path, meta = run(args.pair, args.universe, args.tag)
    print("=" * 78)
    print("P11-C 组合覆盖层")
    print("=" * 78)
    print(f"配对:        {meta['pair']}")
    print(f"股票池:      {meta['universe']}")
    print(f"Alpha(满仓): CAGR {meta['alpha_CAGR']}  Sharpe {meta['alpha_Sharpe']}  "
          f"MaxDD {meta['alpha_MaxDD']}   (等权全池 CAGR {meta.get('ew_CAGR')})")
    print(f"Dow 择时 Sharpe 优于同暴露对照: {meta['dow_sharpe_wins']}")
    print(f"MA200 参照 Sharpe 增益:          {meta['ma200_sharpe_gain']}")
    print(f"Dow 增益超过 MA200 的映射数:      {meta['dow_beats_ma200']}")
    m1 = meta.get("m1") or {}
    if m1:
        print(f"M1 映射: 暴露 {m1.get('avg_exposure')}  "
              f"Dow CAGR {m1.get('Dow_CAGR')} / Sharpe {m1.get('Dow_Sharpe')} | "
              f"等暴露 CAGR {m1.get('等暴露_CAGR')} / Sharpe {m1.get('等暴露_Sharpe')}")
        print(f"  → CAGR 增益 {m1.get('CAGR_增益pp')}pp  Sharpe 增益 {m1.get('Sharpe_增益')}  "
              f"MaxDD 改善 {m1.get('MaxDD_改善pp')}pp")
    print(f"\n报告: {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
