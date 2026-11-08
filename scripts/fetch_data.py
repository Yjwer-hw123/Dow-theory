"""抓取/刷新指数日线数据到本地缓存.

    python scripts/fetch_data.py            # 刷新默认候选的所有指数
    python scripts/fetch_data.py 000300     # 只刷一个
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import coverage_report, load_index  # noqa: E402
from data.universe import INDICES  # noqa: E402

DEFAULT = ["000300", "000905", "000510", "000001", "399001"]


def main(codes: list[str] | None = None, refresh: bool = True) -> int:
    codes = codes or DEFAULT
    print(f"{'code':<8}{'name':<10}{'rows':>7}  {'start':<12}{'end':<12}{'last':>10}  zero_vol")
    for c in codes:
        df = load_index(c, refresh=refresh)
        meta = coverage_report(df)
        name = INDICES.get(c).name if c in INDICES else ""
        if not meta:
            print(f"{c:<8}{name:<10}{'FAILED':>7}")
            continue
        print(
            f"{c:<8}{name:<10}{meta['rows']:>7}  {meta['start']:<12}{meta['end']:<12}"
            f"{meta['last_close']:>10.2f}  {meta['zero_volume_days']}"
        )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args if args else None))
