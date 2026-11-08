"""指数日线数据加载与本地缓存.

数据源：东方财富 push2his（含 OHLCV + 成交额，可回溯到 2005 年）。
设计为只读、可缓存：第一次联网抓取，之后读本地 parquet。

设计原则
--------
道氏引擎只吃两个东西：**两个指数的日收盘价 + 日成交量**（amount）。
所以这一层刻意做得极薄，任何市场（A股/美股/期货指数）只要能产出同样的
DataFrame 就能接进来，引擎本身不需要改。

返回列约定（严格）：
    date, open, high, low, close, volume, amount
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Iterable

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
EM_UT = "7eea3edcaed734bea9cbfc24409ed989"
SINA_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
TX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
COLS = ["date", "open", "high", "low", "close", "volume", "amount"]


def cache_path(code: str) -> str:
    return os.path.join(CACHE_DIR, f"{code}_daily.parquet")


def _secid(code: str) -> str:
    code = str(code)
    return f"1.{code}" if code.startswith(("000", "88", "999")) else f"0.{code}"


def _sina_symbol(code: str) -> str:
    code = str(code)
    return ("sh" if code.startswith(("000", "88", "999")) else "sz") + code


def _http(url: str, retries: int = 3, timeout: int = 25) -> str | None:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    print(f"[WARN] fetch failed ({last_err}): {url[:90]}")
    return None


def fetch_sina(code: str, datalen: int = 6000) -> pd.DataFrame:
    """新浪日线（首选数据源，含成交量，可回溯多年）."""
    url = (
        f"{SINA_URL}?symbol={_sina_symbol(code)}&scale=240&ma=no&datalen={datalen}"
    )
    raw = _http(url)
    if not raw:
        return pd.DataFrame()
    try:
        data = json.loads(raw)
    except Exception:
        return pd.DataFrame()
    rows = [
        {
            "date": r["day"],
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume") or 0),
            "amount": float(r.get("volume") or 0),
        }
        for r in data
        if all(k in r for k in ("day", "open", "high", "low", "close"))
    ]
    return _finalize(rows)


def fetch_tencent(code: str, datalen: int = 6000) -> pd.DataFrame:
    """腾讯日线（备用）."""
    symbol = _sina_symbol(code)
    url = f"{TX_URL}?param={symbol},day,,,{datalen},qfq"
    raw = _http(url)
    if not raw:
        return pd.DataFrame()
    try:
        payload = json.loads(raw)
        node = payload["data"][symbol]
        klines = node.get("day") or node.get("qfqday")
    except Exception:
        return pd.DataFrame()
    rows = [
        {
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]),
            "amount": float(k[5]),
        }
        for k in klines
        if len(k) >= 6
    ]
    return _finalize(rows)


def _finalize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return (
        df.sort_values("date")
        .drop_duplicates(subset="date")
        .reset_index(drop=True)[COLS]
    )


def fetch_index_daily(code: str, secid: str | None = None, lmt: int = 8000,
                      retries: int = 4) -> pd.DataFrame:
    """联网抓取指数日线：新浪 → 腾讯 → 东财，逐级兜底."""
    for fn in (fetch_sina, fetch_tencent):
        df = fn(code, datalen=min(lmt, 6000))
        if not df.empty:
            df.attrs["source"] = fn.__name__
            return df
    return _fetch_em(code, secid or _secid(code), lmt=lmt, retries=retries)


def _fetch_em(code: str, secid: str, lmt: int = 8000, retries: int = 4) -> pd.DataFrame:
    """东方财富 push2his（第三级兜底，带重试）."""
    secid = secid or _secid(code)
    params = {
        "secid": secid,
        "ut": EM_UT,
        "fields1": "f1,f2,f3,f4,f5,f6",
        # f51 日期 / f52 开 / f53 收 / f54 高 / f55 低 / f56 量 / f57 额
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101",
        "fqt": "0",
        "beg": "0",
        "end": "20500101",
        "lmt": str(lmt),
    }
    url = f"{EM_KLINE_URL}?{urllib.parse.urlencode(params)}"
    payload = None
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.6 * (attempt + 1))
    if payload is None:
        print(f"[WARN] fetch failed {code}: {last_err}")
        return pd.DataFrame()
    data = (payload or {}).get("data") or {}
    klines = data.get("klines") or []
    rows = []
    for line in klines:
        p = line.split(",")
        if len(p) < 7:
            continue
        rows.append(
            {
                "date": p[0],
                "open": float(p[1]),
                "close": float(p[2]),
                "high": float(p[3]),
                "low": float(p[4]),
                "volume": float(p[5]),
                "amount": float(p[6]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)[COLS]


def load_index(code: str, refresh: bool = False, secid: str | None = None) -> pd.DataFrame:
    """带 parquet 缓存的指数日线加载."""
    path = cache_path(code)
    if os.path.exists(path) and not refresh:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df[COLS].sort_values("date").reset_index(drop=True)
    df = fetch_index_daily(code, secid=secid)
    if df.empty:
        return df
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.replace(".parquet", ".csv"), index=False, encoding="utf-8-sig")
    return df


def load_index_pair(
    code_a: str, code_b: str, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载一对指数并对齐交易日（内连接）."""
    a = load_index(code_a, refresh=refresh)
    b = load_index(code_b, refresh=refresh)
    if a.empty or b.empty:
        return a, b
    a = a.set_index("date")
    b = b.set_index("date").add_suffix("_b")
    joined = a.join(b, how="inner").reset_index()
    out_a = joined[["date", "open", "high", "low", "close", "volume", "amount"]]
    out_b = joined[["date", "open_b", "high_b", "low_b", "close_b", "volume_b", "amount_b"]].rename(
        columns={
            "open_b": "open",
            "high_b": "high",
            "low_b": "low",
            "close_b": "close",
            "volume_b": "volume",
            "amount_b": "amount",
        }
    )
    return out_a.reset_index(drop=True), out_b.reset_index(drop=True)


def load_many(codes: Iterable[str], refresh: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for c in codes:
        df = load_index(c, refresh=refresh)
        if not df.empty:
            out[c] = df
        time.sleep(0.3)
    return out


def coverage_report(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "rows": int(len(df)),
        "start": str(df["date"].iloc[0].date()),
        "end": str(df["date"].iloc[-1].date()),
        "last_close": float(df["close"].iloc[-1]),
        "zero_volume_days": int((df["volume"] <= 0).sum()),
        "nan_close_days": int(df["close"].isna().sum()),
    }
