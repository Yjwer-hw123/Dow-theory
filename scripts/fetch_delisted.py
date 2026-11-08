"""抓取退市股历史日线（断点续抓）。退市股名单在 data/universe/delisted_codes.json。
新浪对退市股仍返回完整历史，末根 bar 就是退市日（实测 6/6 命中）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.stock_loader import (  # noqa: E402
    CACHE_DIR, UA, SINA_URL, cache_path, fetch_sina, sina_symbol,
)

CODES = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "universe", "delisted_codes.json"), encoding="utf-8"))

os.makedirs(CACHE_DIR, exist_ok=True)
todo = [c for c in CODES if not os.path.exists(cache_path(c))]
print(f"[delisted] total={len(CODES)}  cached={len(CODES)-len(todo)}  todo={len(todo)}",
      flush=True)
if not todo:
    print("nothing to do"); sys.exit(0)

t0 = time.time()
fail = []
done = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch_sina, c): c for c in todo}
    for fut in as_completed(futs):
        c = futs[fut]
        done += 1
        try:
            df = fut.result()
        except Exception as e:
            df = None; fail.append((c, f"EXC:{e}"))
        if df is None or df.empty:
            fail.append(c)
        else:
            df.to_parquet(cache_path(c), index=False)
        if done % 20 == 0 or done == len(todo):
            print(f"[{done}/{len(todo)}] elapsed={time.time()-t0:.0f}s "
                  f"last={c} rows={0 if df is None else len(df)}", flush=True)

ok = len(todo) - len([f for f in fail if isinstance(f, str)])
print(f"\nDONE ok={ok}/{len(todo)}  fail={len(fail)}  elapsed={time.time()-t0:.0f}s",
      flush=True)
print("failed:", fail[:30])