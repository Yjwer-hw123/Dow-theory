"""Point-in-Time（PIT）股票池构造 —— P12-0 研究地基.

为什么需要这个模块
------------------
P11-C 用的是 `hs300_no_star.json`：**2026 年时点的**沪深300 成分股（280 只），
拿它回测 2005 年。这有两个不可接受的偏差：

1. **生存者偏差（survivorship）** —— 只有活到今天、且今天仍在沪深300 里的公司进入样本。
   退市、被剔出指数的公司全部缺席。历史 Alpha 会被系统性抬高。
2. **样本宽度随时间是假的** —— 实测这 280 只里，2005 年前有数据的只有 105 只（37.5%）。
   "2005-2026 的 280 只回测"在 2005 年其实只有 105 只，且这 105 只是**从今天的赢家里挑出来的**。

解决方案：不绑指数成分，用"当日真实在交易的全部 A 股"
----------------------------------------------------
PIT 池的定义只用价格面板，不用任何事后信息：

    在日期 t，股票 c 属于研究池  ⟺  c 在 t 日有成交（有 bar）
                                且 c 截至 t 日的累计有 bar 天数 ≥ min_history（剔除次新股）

这个定义**天然是 point-in-time 的**，而且免费解决四件事：

- 尚未上市 → 无 bar → 排除
- 已退市   → 退市后无 bar → 排除
- 停牌     → 当日无 bar → 排除
- 次新股   → 历史 bar 数不足 → 排除

代价是它**不覆盖**：
- 指数成分（本项目不需要 —— 用全 A 反而更无偏）
- 历史 ST/*ST 状态（见"已知未解决"）
- 一字涨跌停的可成交性（属于执行层，P12-5 成本环节处理）

退市股必须抓进来，否则 PIT 定义仍然残缺
----------------------------------------
已实测：新浪对退市股仍返回完整历史，且**末根 bar 就是退市日**
（600001 邯郸钢铁 → 2009-12-15；000004 国华退 → 2026-07-13）。
所以只要把退市股抓进缓存，上面的面板定义就是完整的。

已知未解决
----------
- **历史 ST 状态拿不到**。akshare 只有当前 ST 快照，没有历史。ST 股涨跌停 ±5%、
  且长期跑输，不剔除会污染低价/反转类因子的结论。当前只能：(a) 用当前名含 ST
  做一次敏感性检查；(b) 在报告里显式声明。**不要**用"当前是否 ST"当基础过滤 ——
  那本身又引入一层事后信息。
- 退市股名单可能有遗漏（沪 159 + 深 208，未含北交所）。
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_CACHE = os.path.join(ROOT, "cache", "stocks")
UNIVERSE_DIR = os.path.join(ROOT, "data", "universe")
DELIST_JSON = os.path.join(UNIVERSE_DIR, "delisted.json")
ALL_A_JSON = os.path.join(UNIVERSE_DIR, "all_a.json")


# ---------------------------------------------------------------------------
# 1. 候选集：全 A（存活）+ 退市股
# ---------------------------------------------------------------------------
def fetch_all_a() -> list[dict]:
    """全部 A 股当前清单（akshare）。返回 [{code, name}]。"""
    import akshare as ak

    df = ak.stock_info_a_code_name()
    code_col = "code" if "code" in df.columns else df.columns[0]
    name_col = "name" if "name" in df.columns else df.columns[1]
    out = []
    for _, r in df.iterrows():
        c = str(r[code_col]).strip().zfill(6)
        if c.isdigit() and len(c) == 6:
            out.append({"code": c, "name": str(r[name_col]).strip().replace(" ", "")})
    return out


def fetch_delisted() -> list[dict]:
    """退市股清单（沪 + 深）。带上市日与终止上市日，用于交叉校验价格数据边界。"""
    import akshare as ak

    out: list[dict] = []
    try:
        sh = ak.stock_info_sh_delist()
        for _, r in sh.iterrows():
            out.append({
                "code": str(r["公司代码"]).strip().zfill(6),
                "name": str(r["公司简称"]).strip(),
                "exchange": "SH",
                "list_date": str(r.get("上市日期"))[:10],
                "delist_date": str(r.get("暂停上市日期"))[:10],
            })
    except Exception as e:
        print(f"  [warn] 沪市退市清单失败: {type(e).__name__} {e}")
    try:
        sz = ak.stock_info_sz_delist()
        for _, r in sz.iterrows():
            out.append({
                "code": str(r["证券代码"]).strip().zfill(6),
                "name": str(r["证券简称"]).strip(),
                "exchange": "SZ",
                "list_date": str(r.get("上市日期"))[:10],
                "delist_date": str(r.get("终止上市日期"))[:10],
            })
    except Exception as e:
        print(f"  [warn] 深市退市清单失败: {type(e).__name__} {e}")

    # 去重（同一 code 可能两边都出现）
    seen: dict[str, dict] = {}
    for d in out:
        if d["code"] not in seen:
            seen[d["code"]] = d
    return sorted(seen.values(), key=lambda x: x["code"])


def build_candidate_lists(force: bool = False) -> dict:
    """构造并落盘候选清单：all_a.json（存活全A）+ delisted.json（退市股）。

    两者合起来才是"历史上曾经存在过的全部 A 股" —— 这是 PIT 池的候选超集。
    """
    os.makedirs(UNIVERSE_DIR, exist_ok=True)

    if not force and os.path.exists(ALL_A_JSON):
        live = json.load(open(ALL_A_JSON, encoding="utf-8"))
        print(f"  all_a.json 已存在: {len(live)} 只")
    else:
        print("  抓取全 A 清单 ...")
        live = fetch_all_a()
        json.dump(live, open(ALL_A_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  all_a.json 落盘: {len(live)} 只")

    if not force and os.path.exists(DELIST_JSON):
        dead = json.load(open(DELIST_JSON, encoding="utf-8"))
        print(f"  delisted.json 已存在: {len(dead)} 只")
    else:
        print("  抓取退市清单 ...")
        dead = fetch_delisted()
        json.dump(dead, open(DELIST_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  delisted.json 落盘: {len(dead)} 只")

    live_codes = {d["code"] for d in live}
    dead_codes = {d["code"] for d in dead}
    return {
        "live": live,
        "delisted": dead,
        "all": sorted(live_codes | dead_codes),
        "overlap": sorted(live_codes & dead_codes),
    }


# ---------------------------------------------------------------------------
# 2. 从已缓存 parquet 读价格覆盖区间（用于审计 + PIT 池）
# ---------------------------------------------------------------------------
def price_coverage() -> pd.DataFrame:
    """每只已缓存个股的首末交易日与 bar 数。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(STOCK_CACHE, "*.parquet"))):
        code = os.path.basename(f)[:6]
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            d = pd.Series(pd.to_datetime(df.index))
        else:
            continue
        d = d.dropna()
        if d.empty:
            continue
        rows.append({
            "code": code,
            "first_bar": d.min().date(),
            "last_bar": d.max().date(),
            "n_bars": int(len(d)),
        })
    return pd.DataFrame(rows).sort_values("code").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. PIT 池（核心）
# ---------------------------------------------------------------------------
def pit_pool(panel: pd.DataFrame, date, min_history: int = 60) -> list[str]:
    """在 `date` 当日，属于研究池的股票代码列表。

    PIT 定义（只用 t 日及之前的信息）：
        c ∈ pool  ⟺  panel.loc[date, c] 非空
                  且 panel.loc[:date, c] 非空 bar 数 ≥ min_history

    panel: index=交易日(DatetimeIndex, 升序), columns=股票代码, values=收盘价(复权)
    """
    d = pd.Timestamp(date)
    if d not in panel.index:
        prior = panel.index[panel.index <= d]
        if len(prior) == 0:
            return []
        d = prior[-1]
    hist = panel.loc[:d]
    traded_today = hist.iloc[-1].notna()
    long_enough = hist.notna().sum() >= min_history
    return sorted(panel.columns[traded_today & long_enough])


def pit_pool_series(panel: pd.DataFrame, dates, min_history: int = 60) -> pd.Series:
    """多个日期的 PIT 池大小（用于画"样本宽度随时间"曲线）。"""
    return pd.Series(
        {pd.Timestamp(d): len(pit_pool(panel, d, min_history)) for d in dates},
        name="pit_size",
    )


def coverage_by_year(cov: pd.DataFrame, years=None) -> pd.DataFrame:
    """按年末统计"已上市可选"只数 —— 直接暴露样本宽度衰减。"""
    if years is None:
        years = list(range(2000, 2027))
    rows = []
    for y in years:
        end = pd.Timestamp(f"{y}-12-31")
        first = pd.to_datetime(cov["first_bar"])
        last = pd.to_datetime(cov["last_bar"])
        alive = int(((first <= end) & (last >= end)).sum())
        listed = int((first <= end).sum())
        rows.append({
            "year": y,
            "listed_by_year_end": listed,
            "alive_at_year_end": alive,
            "delisted_by_year_end": listed - alive,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PIT 股票池地基")
    ap.add_argument("--build-lists", action="store_true", help="抓取并落盘 all_a / delisted 清单")
    ap.add_argument("--force", action="store_true", help="强制重新抓清单")
    ap.add_argument("--coverage", action="store_true", help="统计已缓存个股的价格覆盖")
    args = ap.parse_args()

    if args.build_lists:
        r = build_candidate_lists(force=args.force)
        print(f"\n存活 A 股 : {len(r['live'])}")
        print(f"退市股    : {len(r['delisted'])}")
        print(f"候选超集  : {len(r['all'])}")
        print(f"两边重叠  : {len(r['overlap'])} {r['overlap'][:10]}")

    if args.coverage:
        cov = price_coverage()
        print(f"\n已缓存个股: {len(cov)}")
        print(coverage_by_year(cov).to_string(index=False))
