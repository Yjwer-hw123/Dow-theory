"""P11-A/B 驱动：对多个指数配对做市场环境的时间稳定性检验."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.stability import run  # noqa: E402

PAIRS = [
    (0, "沪深300 + 中证500", ""),
    (1, "上证指数 + 深证成指", "pair1"),
    (2, "沪深300 + 中证A500", "pair2"),
]


def main() -> None:
    print("=" * 78)
    print("P11-A/B  Dow Regime 时间稳定性")
    print("=" * 78)
    rows = []
    for idx, label, tag in PAIRS:
        try:
            path, meta = run(idx, tag)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
            continue
        rows.append((label, meta))
        print(f"\n--- {label} ---")
        print(f"  报告: {os.path.relpath(path)}")
        print(f"  UP>DOWN 成立区间:      {meta['up_gt_down_periods']}")
        print(f"  RISK_ON > 同年 WAIT:   {meta['on_beat_wait_years']}")
        print(f"  RISK_OFF < 同年 WAIT:  {meta['off_below_wait_years']}")
        print(f"  滚动窗口价差为正比例:   {meta['rolling_pos_ratio']}")

    print("\n" + "=" * 78)
    print("结论汇总")
    print("=" * 78)
    print(f"{'配对':22s} {'UP>DOWN':>8s} {'ON>WAIT':>8s} {'OFF<WAIT':>9s} {'滚动正':>7s}")
    for label, meta in rows:
        print(f"{label:22s} {meta['up_gt_down_periods']:>8s} {meta['on_beat_wait_years']:>8s} "
              f"{meta['off_below_wait_years']:>9s} {str(meta['rolling_pos_ratio']):>7s}")


if __name__ == "__main__":
    main()
