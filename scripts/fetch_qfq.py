"""全量重抓**前复权**日线，落地到 cache/qfq/{code}.parquet（不破坏现有缓存）。

为什么要重抓（P12-0 数据地基审计结论）
------------------------------------
现有 `cache/stocks/` 里的面板经实证是**新浪不复权**数据：

  - 600519 缓存值 == 新浪 raw 值（分毫不差），且 2016 年仍是 263.79（未除权）
  - 全样本里出现 948 次日收益 < -30%、1515 次 < -20%
    —— 主板跌停只有 -10%，创业板/科创板 -20%，**物理上不可能**
  - 000333 更糟：缓存来自东财且数值错乱（2016-06-01 = 1.28，真实 22.86），
    该股缓存里有 76 次日收益 < -20%

不复权的后果：送转/分红当天价格除权下调，被当成真实暴跌，
且这个"损失"**永不恢复**（10送10 → 序列上永久 -50%，实际持仓价值不变）。
这是系统性、单向的低估，量级足以淹没任何 Alpha 结论。

修复用的源：`akshare.stock_zh_a_daily(symbol, adjust="qfq")`
----------------------------------------------------------
- 走**新浪**通道（东财已对本机限流：ConnectionError / 20 只只成功 2 只）
- 实测 600519 返回 6015 行 —— 比 `fetch_sina` 的 datalen=6000 截断更**完整**
- 实测日收益 < -20% 次数 = 0（600519）/ 0（000333），除权已消除

已知边界
--------
- **退市股取不到 qfq**（新浪复权接口对已退市代码返回空 JSON）。
  实测 600001 / 000003 均失败。退市股仍沿用不复权，
  在报告里显式声明；等东财限流恢复再补。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QFQ_DIR = os.path.join(ROOT, "cache", "qfq")
STOCK_CACHE = os.path.join(ROOT, "cache", "stocks")
REPORT = os.path.join(ROOT, "cache", "qfq_report.json")


def sina_symbol(code: str) -> str:
    return ("sh" if code[0] in "65" else "sz") + code


def fetch_qfq(code: str) -> pd.DataFrame | None:
    import akshare as ak

    try:
        df = ak.stock_zh_a_daily(symbol=sina_symbol(code), adjust="qfq")
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty or "close" not in df.columns:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    if "amount" not in df.columns:
        df["amount"] = df.get("volume", 0.0) * df["close"]
    keep = ["date", "open", "high", "low", "close", "volume", "amount"]
    return df[[c for c in keep if c in df.columns]]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 只（冒烟用）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(QFQ_DIR, exist_ok=True)
    codes = sorted(os.path.basename(f)[:6]
                   for f in glob.glob(os.path.join(STOCK_CACHE, "*.parquet")))
    if args.limit:
        codes = codes[:args.limit]
    todo = [c for c in codes if not os.path.exists(os.path.join(QFQ_DIR, f"{c}.parquet"))]
    print(f"[qfq] cached={len(codes) - len(todo)}  todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    ok, fail, done = 0, [], 0

    def work(c: str):
        return c, fetch_qfq(c)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, c) for c in todo]
        for fut in as_completed(futs):
            c, df = fut.result()
            done += 1
            if df is None or df.empty:
                fail.append(c)
            else:
                df.to_parquet(os.path.join(QFQ_DIR, f"{c}.parquet"), index=False)
                ok += 1
            if done % 200 == 0 or done == len(todo):
                print(f"[{done}/{len(todo)}] ok={ok} fail={len(fail)} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)

    json.dump({"ok": ok, "fail": fail, "elapsed_s": round(time.time() - t0)},
              open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nDONE ok={ok} fail={len(fail)} elapsed={time.time() - t0:.0f}s", flush=True)
    print("fail sample:", fail[:20], flush=True)


if __name__ == "__main__":
    main()
