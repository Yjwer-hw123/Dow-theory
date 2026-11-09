"""个股日线数据层（P11-C 组合覆盖层用）.

数据源优先级
------------
1. **新浪** `CN_MarketData.getKLineData`，`datalen=6000` —— 实测可回溯到 2001 年，
   单次 ~5.6s。注意：datalen 上限不是 1023（那是默认值），给多少它返多少（≤6000）。
2. **东方财富** push2his —— 单次更快（~0.9s）且带真实成交额，但**本机实测会断连**
   （`RemoteDisconnected`，连续请求几十次后被限流），所以只作兜底。

本机实测不可用的源（不要再试）
------------------------------
- `mootdx`：三个参数组合全部返回 0 行（~6s 超时）。
- `quotes.money.163.com`：502 Bad Gateway。
- 腾讯 `ifzq.gtimg.cn`：不带区间只给 320 根，带上区间后返回结构变化，不足以支撑长回测。

成交额
------
新浪只给成交量（股），所以 `amount = volume × close`（用收盘价近似 VWAP）。
只用于流动性门槛，不影响收益计算。

符号映射（与指数不同，务必区分）
--------------------------------
- 个股 `000001` = 平安银行 = **深市** `sz000001`
- 指数 `000001` = 上证指数 = 沪市（本项目 `data/loader.py` 的 `_sina_symbol` 是按
  指数写的，直接拿来抓个股会把 `000001` 抓成上证指数 —— 这是个静默错误）
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "cache", "stocks")

# 股票池已在项目内自包含（2026-09-25 从 stock-analysis-skill 复制进来），
# 不再跨工作空间引用 —— 换机器/迁移目录时不会断。
UNIVERSE_DIR = os.path.join(ROOT, "data", "universe")
UNIVERSES = {
    "hs300": os.path.join(UNIVERSE_DIR, "hs300_no_star.json"),
    "zz500": os.path.join(UNIVERSE_DIR, "zz500_no_star.json"),
    "wide": os.path.join(UNIVERSE_DIR, "wide_no_star.json"),
}

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SINA_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
EM_UT = "7eea3edcaed734bea9cbfc24409ed989"
EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
SINA_DATALEN = 6000


# ---------------------------------------------------------------------------
def stock_secid(code: str) -> str:
    """东财个股 secid（沪 1. / 深 0.）"""
    code = str(code).strip().zfill(6)
    return f"1.{code}" if code.startswith(("6", "9")) else f"0.{code}"


def sina_symbol(code: str) -> str:
    """新浪个股符号（沪 sh / 深 sz）"""
    code = str(code).strip().zfill(6)
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def cache_path(code: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)  # 目录已于 2026-11-09 清空释放，按需重建
    return os.path.join(CACHE_DIR, f"{code}.parquet")


def load_universe_codes(name: str = "hs300") -> list[str]:
    with open(UNIVERSES[name], encoding="utf-8") as f:
        return list(json.load(f)["codes"])


def _http(url: str, retries: int = 3, timeout: int = 25, referer: str = "") -> str | None:
    hdrs = {"User-Agent": UA, "Connection": "close"}
    if referer:
        hdrs["Referer"] = referer
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            time.sleep(0.4 * (attempt + 1))
    return None


def _finalize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    df["amount"] = df["volume"] * df["close"]     # 近似成交额
    return df[COLS]


# ---------------------------------------------------------------------------
def fetch_sina(code: str, datalen: int = SINA_DATALEN) -> pd.DataFrame:
    url = f"{SINA_URL}?symbol={sina_symbol(code)}&scale=240&ma=no&datalen={datalen}"
    raw = _http(url, referer="https://finance.sina.com.cn/")
    if not raw:
        return pd.DataFrame()
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if not isinstance(data, list):
        return pd.DataFrame()
    rows = [
        {
            "date": r["day"],
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume") or 0.0),
        }
        for r in data
        if isinstance(r, dict) and all(k in r for k in ("day", "open", "high", "low", "close"))
    ]
    return _finalize(rows)


def fetch_em(code: str, retries: int = 2, timeout: int = 25) -> pd.DataFrame:
    params = {
        "secid": stock_secid(code),
        "ut": EM_UT,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101", "fqt": "1", "beg": "0", "end": "20500101", "lmt": "1000000",
    }
    url = f"{EM_KLINE_URL}?{urllib.parse.urlencode(params)}"
    raw = _http(url, retries=retries, timeout=timeout,
                referer="https://quote.eastmoney.com/")
    if not raw:
        return pd.DataFrame()
    try:
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    klines = ((payload or {}).get("data") or {}).get("klines") or []
    rows = []
    for line in klines:
        p = line.split(",")
        if len(p) < 7:
            continue
        try:
            rows.append({"date": p[0], "open": float(p[1]), "high": float(p[3]),
                         "low": float(p[4]), "close": float(p[2]),
                         "volume": float(p[5]), "amount": float(p[6])})
        except (ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)[COLS]


def fetch_stock(code: str) -> pd.DataFrame:
    """新浪 → 东财，逐级兜底."""
    df = fetch_sina(code)
    if not df.empty and len(df) > 60:
        return df
    df2 = fetch_em(code)
    if not df2.empty:
        return df2
    return df


# ---------------------------------------------------------------------------
def load_stock(code: str, refresh: bool = False) -> pd.DataFrame:
    """带 parquet 缓存的个股日线加载（只读缓存；缺失才联网）."""
    path = cache_path(code)
    if os.path.exists(path) and not refresh:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    df = fetch_stock(code)
    if df.empty:
        return df
    try:
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001
        df.to_csv(path.replace(".parquet", ".csv"), index=False, encoding="utf-8-sig")
    return df


def load_universe_panel(codes: list[str] | None = None, universe: str = "hs300",
                        min_days: int = 250):
    """读缓存 → 组装宽表面板.

    返回 (close_wide, amount_wide, report)
    """
    codes = codes or load_universe_codes(universe)
    closes, amounts, kept, skipped = {}, {}, [], []
    for c in codes:
        df = load_stock(c)
        if df.empty or len(df) < min_days:
            skipped.append(c)
            continue
        s = df.set_index("date")
        closes[c] = s["close"]
        amounts[c] = s["amount"]
        kept.append(c)
    close_wide = pd.DataFrame(closes).sort_index()
    amount_wide = pd.DataFrame(amounts).sort_index()
    report = {"requested": len(codes), "kept": len(kept), "skipped": skipped}
    return close_wide, amount_wide, report


def coverage_report(codes: list[str] | None = None, universe: str = "hs300") -> pd.DataFrame:
    codes = codes or load_universe_codes(universe)
    rows = []
    for c in codes:
        p = cache_path(c)
        if not os.path.exists(p):
            rows.append({"code": c, "rows": 0, "start": None, "end": None,
                         "zero_vol": 0, "cached": False})
            continue
        df = pd.read_parquet(p)
        rows.append({
            "code": c, "rows": len(df),
            "start": str(pd.to_datetime(df["date"]).iloc[0].date()),
            "end": str(pd.to_datetime(df["date"]).iloc[-1].date()),
            "zero_vol": int((df["volume"] <= 0).sum()),
            "cached": True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="hs300", choices=list(UNIVERSES))
    ap.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    ap.add_argument("--check", action="store_true", help="只体检缓存，不联网")
    ap.add_argument("--workers", type=int, default=6, help="并发线程数")
    args = ap.parse_args()

    codes = load_universe_codes(args.universe)
    print(f"[stock_loader] universe={args.universe} codes={len(codes)}", flush=True)

    if args.check:
        rep = coverage_report(codes, args.universe)
        have = rep[rep["rows"] > 0]
        print(f"cached={len(have)}/{len(codes)}  "
              f"rows min/med/max = {have['rows'].min() if len(have) else 0}/"
              f"{int(have['rows'].median()) if len(have) else 0}/"
              f"{have['rows'].max() if len(have) else 0}  "
              f"zero_vol_total={int(have['zero_vol'].sum()) if len(have) else 0}")
        if len(have) < len(codes):
            miss = rep[rep["rows"] == 0]["code"].tolist()
            print(f"missing {len(miss)}:", ", ".join(miss[:40]))
        return

    todo = [c for c in codes
            if args.force or not os.path.exists(cache_path(c))]
    print(f"[stock_loader] todo={len(todo)} workers={args.workers}", flush=True)
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    fail: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_stock, c): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            try:
                df = fut.result()
            except Exception:  # noqa: BLE001
                df = pd.DataFrame()
            if df.empty:
                fail.append(c)
                print(f"[{done}/{len(todo)}] {c} FAIL", flush=True)
                continue
            df.to_parquet(cache_path(c), index=False)
            if done % 20 == 0 or done == len(todo):
                print(f"[{done}/{len(todo)}] elapsed={time.time()-t0:.0f}s "
                      f"last={c} rows={len(df)}", flush=True)

    print(f"\nDONE ok={len(todo)-len(fail)}/{len(todo)} failed={len(fail)} "
          f"elapsed={time.time()-t0:.0f}s")
    if fail:
        print("failed:", ", ".join(fail))
    rep = coverage_report(codes, args.universe)
    have = rep[rep["rows"] > 0]
    if len(have):
        print(f"coverage={len(have)}/{len(codes)}  "
              f"rows med={int(have['rows'].median())}  "
              f"range {have['start'].min()} -> {have['end'].max()}")


if __name__ == "__main__":
    _main()
