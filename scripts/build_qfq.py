"""重铺数据地基：新浪不复权日线 + 新浪前复权因子 → 前复权面板。

背景（P12-0 数据地基审计）
------------------------
现有 `cache/stocks/` 里的面板经实证有两类缺陷：

1. **主体是不复权数据**。600519 缓存值与新浪 raw 分毫不差（2016-06-01 = 263.79，
   未除权）。全样本出现 948 次日收益 < -30%、1515 次 < -20% ——
   主板跌停 -10%、创业板/科创板 -20%，**物理上不可能**，全是除权/送转。
   后果：送转当天序列上永久 -50%（实际持仓价值不变），单向低估收益。

2. **部分文件来自东财且数值错乱**。000333 缓存 2016-06-01 = 1.28，
   真实不复权 22.86；该股缓存里还有 76 次日收益 < -20%。

修复方案（不重抓价格，只抓因子）
------------------------------
新浪提供前复权**累计因子**端点（每只 ~1KB）：

    https://finance.sina.com.cn/realstock/company/{symbol}/qfq.js
    → var sz000333qfq={"total":16,"data":[{"d":"2026-06-29","f":"1.0000"}, ...]}

语义经实证确认（对 600519 三个日期全部命中 akshare 前复权值）：

    qfq(t) = raw(t) / f   ,  f = 最后一个 d <= t 的因子

该端点**对退市股也有效**（600001 邯郸钢铁 total=13），所以退市股同样能复权。
成交额 amount 不受复权影响（是真金白银），保持原值 —— 流动性门槛口径不变。

产出
----
    cache/raw/{code}.parquet      新浪不复权全量重抓（覆盖旧的坏文件）
    cache/factors/{code}.json     前复权因子
    cache/qfq/{code}.parquet      前复权面板（P12 起唯一可用口径）
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from data.stock_loader import fetch_sina  # noqa: E402
from data.pit_universe import load_stock_codes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "cache", "raw")
FAC_DIR = os.path.join(ROOT, "cache", "factors")
QFQ_DIR = os.path.join(ROOT, "cache", "qfq")
REPORT_JSON = os.path.join(ROOT, "cache", "qfq_build_report.json")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Referer": "https://finance.sina.com.cn/"}


def sina_symbol(code: str) -> str:
    return ("sh" if code[0] in "65" else "sz") + code


def fetch_factors(code: str) -> list[dict] | None:
    url = f"https://finance.sina.com.cn/realstock/company/{sina_symbol(code)}/qfq.js"
    try:
        t = requests.get(url, headers=UA, timeout=20).text
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"=\s*(\{.*\})", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except Exception:  # noqa: BLE001
        return None
    data = d.get("data") or []
    return [{"d": x["d"], "f": float(x["f"])} for x in data]


def apply_factors(df: pd.DataFrame, facs: list[dict] | None) -> pd.DataFrame:
    """qfq = raw / f，f 取最后一个 d <= t 的因子。"""
    out = df.copy()
    if not facs:
        return out
    f = pd.Series([x["f"] for x in facs],
                  index=pd.to_datetime([x["d"] for x in facs])).sort_index()
    idx = pd.to_datetime(out["date"])
    fac = f.reindex(idx, method="ffill").fillna(1.0).to_numpy()
    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out[c] = out[c].astype(float) / fac
    # amount 是成交额，复权不影响；volume 按因子放大（股本变动后的等效股数）
    if "volume" in out.columns:
        out["volume"] = out["volume"].astype(float) * fac
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-raw", action="store_true", help="已有 cache/raw 时跳过重抓")
    args = ap.parse_args()

    for d in (RAW_DIR, FAC_DIR, QFQ_DIR):
        os.makedirs(d, exist_ok=True)

    codes = load_stock_codes()
    if args.limit:
        codes = codes[:args.limit]
    print(f"[build_qfq] codes={len(codes)}", flush=True)

    # ---- 1) 重抓不复权（覆盖坏文件） ----
    if args.skip_raw:
        todo_raw = []
    else:
        todo_raw = [c for c in codes
                    if not os.path.exists(os.path.join(RAW_DIR, f"{c}.parquet"))]
    print(f"[raw] todo={len(todo_raw)}", flush=True)
    raw_fail: list[str] = []
    if todo_raw:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_sina, c): c for c in todo_raw}
            for fut in as_completed(futs):
                c = futs[fut]
                done += 1
                try:
                    df = fut.result()
                except Exception:  # noqa: BLE001
                    df = None
                if df is None or df.empty:
                    raw_fail.append(c)
                else:
                    df.to_parquet(os.path.join(RAW_DIR, f"{c}.parquet"), index=False)
                if done % 500 == 0 or done == len(todo_raw):
                    print(f"  [raw {done}/{len(todo_raw)}] fail={len(raw_fail)} "
                          f"{time.time()-t0:.0f}s", flush=True)

    # ---- 2) 抓因子 ----
    todo_fac = [c for c in codes
                if not os.path.exists(os.path.join(FAC_DIR, f"{c}.json"))]
    print(f"[fac] todo={len(todo_fac)}", flush=True)
    fac_fail: list[str] = []
    if todo_fac:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_factors, c): c for c in todo_fac}
            for fut in as_completed(futs):
                c = futs[fut]
                done += 1
                try:
                    fs = fut.result()
                except Exception:  # noqa: BLE001
                    fs = None
                if fs is None:
                    fac_fail.append(c)
                else:
                    json.dump(fs, open(os.path.join(FAC_DIR, f"{c}.json"), "w",
                                       encoding="utf-8"), ensure_ascii=False)
                if done % 500 == 0 or done == len(todo_fac):
                    print(f"  [fac {done}/{len(todo_fac)}] fail={len(fac_fail)} "
                          f"{time.time()-t0:.0f}s", flush=True)

    # ---- 3) 合成前复权面板 ----
    print("[build] 合成 qfq 面板 ...", flush=True)
    n_ok, n_nofac, stats = 0, 0, {"lt_20": 0, "lt_30": 0, "obs": 0}
    for c in codes:
        rp = os.path.join(RAW_DIR, f"{c}.parquet")
        fp = os.path.join(FAC_DIR, f"{c}.json")
        if not os.path.exists(rp):
            continue
        df = pd.read_parquet(rp)
        facs = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else None
        if not facs:
            n_nofac += 1
        q = apply_factors(df, facs)
        q.to_parquet(os.path.join(QFQ_DIR, f"{c}.parquet"), index=False)
        n_ok += 1
        r = q["close"].astype(float).pct_change()
        stats["obs"] += int(r.notna().sum())
        stats["lt_20"] += int((r < -0.20).sum())
        stats["lt_30"] += int((r < -0.30).sum())

    rep = {"codes": len(codes), "raw_fail": raw_fail, "fac_fail": fac_fail,
           "qfq_built": n_ok, "no_factor": n_nofac, "return_stats": stats}
    json.dump(rep, open(REPORT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nDONE qfq={n_ok} no_factor={n_nofac} raw_fail={len(raw_fail)} "
          f"fac_fail={len(fac_fail)}", flush=True)
    print(f"qfq 面板日收益 < -20%: {stats['lt_20']} / {stats['obs']} "
          f"({stats['lt_20']/max(stats['obs'],1)*1e4:.2f} 每万日)", flush=True)
    print(f"qfq 面板日收益 < -30%: {stats['lt_30']}", flush=True)


if __name__ == "__main__":
    main()
