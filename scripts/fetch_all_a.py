"""抓取全部存活 A 股历史日线（断点续抓）。

为什么在退市股之外还要抓这个
----------------------------
PIT 池的定义是"当日真实在交易的全部 A 股"。退市股只补上了"死掉的"一侧；
如果存活侧仍然只有 `hs300_no_star.json` 的 280 只（2026 年时点沪深300），
那么"今天不在沪深300 里的公司"依然缺席 —— 幸存偏差只修了一半。

所以候选超集必须是：all_a.json（当前存活全 A，5,569）+ delisted.json（退市 333）。

过滤规则
--------
- 排除 B 股：沪市 900xxx、深市 200xxx
- 排除北交所：4xx / 8xx 开头
- 保留：60/68（沪主板、科创板）、00/30/301（深主板、中小板、创业板）
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.stock_loader import (  # noqa: E402
    CACHE_DIR, cache_path, fetch_sina,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_A = os.path.join(ROOT, "data", "universe", "all_a.json")


def keep(code: str) -> bool:
    """只保留 A 股主板/创业板/科创板，剔除 B 股与北交所。"""
    if not (code.isdigit() and len(code) == 6):
        return False
    if code[:3] in ("900", "200"):        # B 股
        return False
    if code[0] in ("4", "8"):             # 北交所 / 老三板
        return False
    return code[0] in ("6", "0", "3")


def main() -> None:
    raw = json.load(open(ALL_A, encoding="utf-8"))
    codes = sorted({str(d["code"]).zfill(6) for d in raw})
    codes = [c for c in codes if keep(c)]

    os.makedirs(CACHE_DIR, exist_ok=True)
    todo = [c for c in codes if not os.path.exists(cache_path(c))]
    print(f"[all_a] candidates={len(codes)}  cached={len(codes) - len(todo)}  "
          f"todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    fail: list = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_sina, c): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            try:
                df = fut.result()
            except Exception as e:  # noqa: BLE001
                df = None
                fail.append((c, f"EXC:{type(e).__name__}"))
            if df is None or df.empty:
                if not fail or fail[-1][0] != c:
                    fail.append((c, "EMPTY"))
            else:
                df.to_parquet(cache_path(c), index=False)
            if done % 200 == 0 or done == len(todo):
                print(f"[{done}/{len(todo)}] elapsed={time.time() - t0:.0f}s "
                      f"fail={len(fail)}", flush=True)

    print(f"\nDONE todo={len(todo)} fail={len(fail)} elapsed={time.time() - t0:.0f}s",
          flush=True)
    print("failed sample:", fail[:30], flush=True)


if __name__ == "__main__":
    main()
