# -*- coding: utf-8 -*-
"""EP-02 L1-6 资金曲线专项纠正：组合逐日盯市账本。

只重算资金曲线与 L1-6，不改协议、不改信号、不改交易规则、不删亏损交易：
- 交易集 = trades_b_l1.csv 的 11,419 笔（买卖日期、名义本金、三档成本均不变）；
- 现金账本：买入日支出（本金+滑点+佣金+过户费），卖出日回款（扣佣金/印花税/过户费）；
- 持仓市值：`N × qfq_t / qfq_entry`（与逐笔收益口径一致，含分红再投资）；
- NAV = 最小初始资金 + 累计现金流 + 全部未平仓市值（逐日盯市）；
- 另算最低初始资金需求、最大同时持仓、最大资金占用、最大回撤及起止日；
- 连亏：保留原逐笔口径，另按入场月份批次与实现月份统计。

用法：
    python scripts/entry_phase_02_l1_ledger.py --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.entry_phase as ep  # noqa: E402
import scripts.entry_phase_02_l1 as l1  # noqa: E402
import scripts.uptrend_detector as ud  # noqa: E402
from scripts.step1_trend_validation import RAW_DIR  # noqa: E402

OUT_DIR = os.path.join(ROOT, "out", "entry_phase_02")
TRADES = os.path.join(OUT_DIR, "trades_b_l1.csv")
N = l1.NOTIONAL
SCEN = "base"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cash_flows(code, d_entry, d_exit, raw_entry, raw_exit, shares):
    slip = l1.SLIP[SCEN]
    comm = l1.COMM[SCEN]
    p_buy = raw_entry * (1.0 + slip)
    buy_amt = shares * p_buy
    buy_cash = (buy_amt + max(buy_amt * comm, l1.MIN_COMM)
                + l1.transfer_fee(code, d_entry, "buy", buy_amt, shares, SCEN))
    p_sell = raw_exit * (1.0 - slip)
    sell_amt = shares * p_sell
    sell_cash = (sell_amt - max(sell_amt * comm, l1.MIN_COMM)
                 - sell_amt * l1.stamp_rate(d_exit, "sell")
                 - l1.transfer_fee(code, d_exit, "sell", sell_amt, shares, SCEN))
    return buy_cash, sell_cash


CAL_NS: np.ndarray | None = None
NCAL = 0
CAL_POS: dict = {}


def build_code(code: str, tl: pd.DataFrame) -> dict:
    """一只股票：持仓逐日市值 + 现金流（返回 3 个长度 NCAL 的数组）。"""
    rp = os.path.join(RAW_DIR, f"{code}.parquet")
    raw = pd.read_parquet(rp)
    facs = l1.load_facs(code)
    A = ud.prepare_stock(raw, facs)
    pos = np.zeros(NCAL, dtype=np.float64)
    cash = np.zeros(NCAL, dtype=np.float64)
    cnt = np.zeros(NCAL, dtype=np.float64)
    if A is None:
        return dict(code=code, pos=pos, cash=cash, cnt=cnt, ok=False)

    dns = A["dates"].astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(CAL_NS, dns)
    ok = idx < NCAL
    ok[ok] &= CAL_NS[idx[ok]] == dns[ok]
    qclose = np.full(NCAL, np.nan)
    qclose[idx[ok]] = A["c"][ok]
    qclose = pd.Series(qclose).ffill().to_numpy()

    for row in tl.itertuples():
        di = CAL_POS.get(pd.Timestamp(row.entry_date))
        xi = CAL_POS.get(pd.Timestamp(row.exit_date))
        if di is None or xi is None or xi <= di:
            continue
        qe = float(row.qfq_entry)
        vals = N * qclose[di:xi] / qe
        pos[di:xi] += vals
        cnt[di] += 1
        cnt[xi] -= 1
        shares = N / float(row.raw_entry)
        b, s = cash_flows(code, row.entry_date, row.exit_date,
                          float(row.raw_entry), float(row.raw_exit), shares)
        cash[di] -= b
        cash[xi] += s
    return dict(code=code, pos=pos, cash=cash, cnt=cnt, ok=True)


def max_dd(nav: np.ndarray) -> tuple[float, int, int]:
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    i = int(np.argmax(dd))
    j = int(np.argmax(nav[:i + 1]))
    return float(dd[i]), j, i


def month_key(s) -> str:
    return str(s)[:7]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    global CAL_NS, NCAL, CAL_POS
    ep.init_ctx()
    cal = ep.CTX["cal"]
    NCAL = len(cal)
    CAL_NS = np.array([pd.Timestamp(x).value for x in cal], dtype=np.int64)
    CAL_POS = ep.CTX["cal_pos"]
    log(f"[ledger] 市场日历 {NCAL} 天 {cal[0].date()} ~ {cal[-1].date()}")

    df = pd.read_csv(TRADES, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    df["net_amount"] = df["net_return_base"] * N
    log(f"[ledger] 交易 {len(df)} 笔，涉及 {df['code'].nunique()} 只股票")
    groups = {c: g for c, g in df.groupby("code")}

    t0 = time.time()
    pos_m = np.zeros(NCAL)
    cash_m = np.zeros(NCAL)
    cnt_m = np.zeros(NCAL)
    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_code, c, g): c for c, g in groups.items()}
        for fut in as_completed(futs):
            r = fut.result()
            if not r["ok"]:
                failed.append(r["code"])
            pos_m += r["pos"]
            cash_m += r["cash"]
            cnt_m += r["cnt"]
            done += 1
            if done % 500 == 0 or done == len(groups):
                log(f"  ledger {done}/{len(groups)} {time.time()-t0:.0f}s")

    cash_cum = np.cumsum(cash_m)
    c0 = float(max(0.0, -cash_cum.min()))
    nav = c0 + cash_cum + pos_m
    dd, ds, de = max_dd(nav)
    dates = pd.to_datetime(cal)
    dd_peak = float(nav[ds]) if ds < NCAL else np.nan
    dd_trough = float(nav[de])

    open_pos = np.maximum.accumulate(cnt_m.cumsum())
    max_open = int(cnt_m.cumsum().max())
    max_pos_value = float(pos_m.max())
    min_cash = float((c0 + cash_cum).min())

    # ---- 2015 股灾窗口 ----
    y15 = (dates >= pd.Timestamp("2015-06-01")) & (dates <= pd.Timestamp("2015-10-31"))
    nav15 = nav[y15]
    i15 = int(np.argmax(nav15)) if nav15.size else -1
    j15 = int(np.argmin(nav15[i15:])) + i15 if nav15.size else -1
    trades15 = df[df["entry_date"].astype(str).str[:7] == "2015-06"]
    exit15 = df[(df["exit_date"].astype(str) >= "2015-06-01")
                & (df["exit_date"].astype(str) <= "2015-10-31")]

    # ---- 批次与月份连亏 ----
    df["batch"] = df["entry_date"].astype(str).str[:7]
    df["rmonth"] = df["exit_date"].astype(str).str[:7]
    df["net_amount"] = df["net_return_base"] * N

    def streak(series: pd.Series) -> tuple[int, list[str]]:
        best = cur = 0
        best_seq: list[str] = []
        cur_seq: list[str] = []
        for k, v in series.items():
            if v < 0:
                cur += 1
                cur_seq.append(k)
                if cur > best:
                    best, best_seq = cur, list(cur_seq)
            else:
                cur = 0
                cur_seq = []
        return best, best_seq

    batch_pl = df.groupby("batch")["net_amount"].sum().sort_index()
    month_pl = df.groupby("rmonth")["net_amount"].sum().sort_index()
    batch_streak, batch_seq = streak(batch_pl)
    month_streak, month_seq = streak(month_pl)

    worst_batch = batch_pl.idxmin()
    worst_batch_pl = float(batch_pl.min())

    # 原逐笔连亏
    ts = df.sort_values(["exit_date", "code"])
    r = ts["net_return_base"].to_numpy()
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)

    # ---- L1-6 重算 ----
    l16 = dict(
        mdd_portfolio=dd,
        mdd_le_30pct=bool(dd <= 0.30),
        streak_original_trades=int(best),
        streak_entry_batch=int(batch_streak),
        streak_losing_month=int(month_streak),
        streak_le_12=(int(batch_streak) <= 12),
        pass_=bool(dd <= 0.30 and int(batch_streak) <= 12),
    )

    summary = dict(
        meta=dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  script="scripts/entry_phase_02_l1_ledger.py",
                  trades=len(df), notional=N, scenario=SCEN,
                  nav_method="NAV = C0 + 累计现金流 + 未平仓市值（逐日盯市）",
                  position_value="N × qfq_t / qfq_entry"),
        cash=dict(min_initial_capital=c0,
                  min_cash_balance=min_cash,
                  max_position_value=max_pos_value,
                  max_open_positions=max_open,
                  final_nav=float(nav[-1]),
                  total_net_amount=float(df["net_amount"].sum())),
        nav=dict(max_drawdown=dd,
                 dd_peak_date=str(dates[ds].date()),
                 dd_trough_date=str(dates[de].date()),
                 dd_peak_nav=dd_peak,
                 dd_trough_nav=dd_trough,
                 dd_loss_amount=float(dd_peak - dd_trough)),
        crash_2015=dict(
            window="2015-06-01~2015-10-31",
            nav_start=float(nav15[i15]) if i15 >= 0 else np.nan,
            nav_trough=float(nav15[j15]) if j15 >= 0 else np.nan,
            loss_amount=float(nav15[i15] - nav15[j15]) if i15 >= 0 else np.nan,
            loss_pct=float(nav15[j15] / nav15[i15] - 1.0) if i15 >= 0 else np.nan,
            peak_date=str(dates[y15][i15].date()) if i15 >= 0 else "",
            trough_date=str(dates[y15][j15].date()) if j15 >= 0 else "",
            entry_201506_n=len(trades15),
            entry_201506_net_amount=float(trades15["net_amount"].sum()),
            entry_201506_win_rate=float((trades15["net_return_base"] > 0).mean()),
            exit_window_n=len(exit15),
            exit_window_net_amount=float(exit15["net_amount"].sum()),
            exit_window_win_rate=float((exit15["net_return_base"] > 0).mean()),
        ),
        streaks=dict(original_trades=int(best),
                     entry_batch=int(batch_streak), entry_batch_seq=batch_seq,
                     losing_month=int(month_streak), losing_month_seq=month_seq,
                     worst_entry_batch=worst_batch,
                     worst_entry_batch_loss=worst_batch_pl),
        l1_6=l16,
        failed_codes=failed,
    )

    with open(os.path.join(OUT_DIR, "summary_b_l1_ledger.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    out = pd.DataFrame(dict(
        date=[str(d.date()) for d in dates],
        cash=c0 + cash_cum,
        position_value=pos_m,
        nav=nav,
        drawdown=(np.maximum.accumulate(nav) - nav) / np.maximum.accumulate(nav),
        n_open=cnt_m.cumsum(),
    ))
    out.to_csv(os.path.join(OUT_DIR, "daily_nav_b_l1.csv"), index=False,
               encoding="utf-8")

    # 年度/批次表
    batch_pl.rename("net_amount").reset_index().to_csv(
        os.path.join(OUT_DIR, "batch_entry_month_pl.csv"), index=False,
        encoding="utf-8")
    month_pl.rename("net_amount").reset_index().to_csv(
        os.path.join(OUT_DIR, "month_realized_pl.csv"), index=False,
        encoding="utf-8")

    write_report(summary, dates, nav, out)
    log(f"[ledger] MDD={dd:.2%} ({dates[ds].date()}~{dates[de].date()})；"
        f"最低初始资金={c0:,.0f} 元；最大同时持仓={max_open}；"
        f"原连亏={best} 批次连亏={batch_streak} 月连亏={month_streak}")


def write_report(s: dict, dates, nav: np.ndarray, out: pd.DataFrame) -> None:
    c = s["cash"]
    n = s["nav"]
    k = s["crash_2015"]
    st = s["streaks"]
    l = s["l1_6"]
    lines = [
        "# EP-02 L1-6 资金曲线专项纠正报告", "",
        f"- 生成：{s['meta']['generated_at']}",
        f"- 交易：{s['meta']['trades']:,} 笔；名义本金 {N:,.0f} 元/笔；基准成本",
        f"- 账本：{s['meta']['nav_method']}；持仓市值 = {s['meta']['position_value']}",
        "",
        "## 直接回答", "",
        "| 问题 | 结果 |",
        "|---|---|",
        f"| 1. 组合最大回撤 | **{n['max_drawdown']:.2%}**"
        f"（{n['dd_peak_date']} → {n['dd_trough_date']}，"
        f"回撤金额 {n['dd_loss_amount']:,.0f} 元） |",
        f"| 2. 2015 股灾窗口组合损失 | **{k['loss_amount']:,.0f} 元**"
        f"（{k['loss_pct']:.2%}，{k['peak_date']} → {k['trough_date']}） |",
        f"| 3. 最大资金需求 | 最低初始资金 **{c['min_initial_capital']:,.0f} 元**；"
        f"峰值持仓市值 {c['max_position_value']:,.0f} 元；最大同时持仓 {c['max_open_positions']} 笔 |",
        f"| 4. L1-6 是否通过 | **{'通过' if l['pass_'] else '未通过'}**"
        f"（MDD {l['mdd_portfolio']:.2%} {'≤' if l['mdd_le_30pct'] else '>'} 30%；"
        f"连续亏损批次 {l['streak_entry_batch']} {'≤' if l['streak_le_12'] else '>'} 12） |",
        "",
        "## 资金账本", "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 最低初始资金需求 | {c['min_initial_capital']:,.0f} 元 |",
        f"| 最低现金余额（初始资金下） | {c['min_cash_balance']:,.0f} 元 |",
        f"| 峰值持仓市值（最大资金占用） | {c['max_position_value']:,.0f} 元 |",
        f"| 最大同时持仓 | {c['max_open_positions']} 笔 |",
        f"| 期末 NAV | {c['final_nav']:,.0f} 元 |",
        f"| 全部实现净收益 | {c['total_net_amount']:,.0f} 元 |",
        "",
        "## 最大回撤", "",
        f"- 峰值 NAV {n['dd_peak_nav']:,.0f} 元（{n['dd_peak_date']}）",
        f"- 谷底 NAV {n['dd_trough_nav']:,.0f} 元（{n['dd_trough_date']}）",
        f"- 回撤幅度 {n['max_drawdown']:.2%}，金额 {n['dd_loss_amount']:,.0f} 元",
        "",
        "## 2015 年股灾（纠正“60 笔连亏”的解读）", "",
        f"- 窗口 2015-06-01~2015-10-31：NAV {k['nav_start']:,.0f} → "
        f"{k['nav_trough']:,.0f}，损失 {k['loss_amount']:,.0f} 元（{k['loss_pct']:.2%}）",
        f"- 2015-06 入场批次：{k['entry_201506_n']} 笔，合计净盈亏 "
        f"{k['entry_201506_net_amount']:,.0f} 元，胜率 {k['entry_201506_win_rate']:.2%}",
        f"- 2015-06~10 退出批次：{k['exit_window_n']} 笔，合计净盈亏 "
        f"{k['exit_window_net_amount']:,.0f} 元，胜率 {k['exit_window_win_rate']:.2%}",
        "",
        "## 连亏口径对照", "",
        "| 口径 | 最长连续亏损 | 说明 |",
        "|---|---|---|",
        f"| 原逐笔（按退出日排序） | {st['original_trades']} 笔 | 含大量同一轮回撤的重叠持仓，不能视为独立风险事件 |",
        f"| 入场月份批次 | {st['entry_batch']} 个 | 连续亏损的入场批次 |",
        f"| 实现月份 | {st['losing_month']} 个月 | 组合月度已实现净收益连续为负 |",
        f"| 最差入场批次 | {st['worst_entry_batch']} | 净亏损 {st['worst_entry_batch_loss']:,.0f} 元 |",
        "",
        "## L1-6 重算", "",
        "| 项 | 结果 |",
        "|---|---|",
        f"| 组合 MDD ≤ 30% | {'通过' if l['mdd_le_30pct'] else '未通过'}（{l['mdd_portfolio']:.2%}） |",
        f"| 连续亏损批次 ≤ 12 | {'通过' if l['streak_le_12'] else '未通过'}（{l['streak_entry_batch']}） |",
        f"| L1-6 总判定 | **{'通过' if l['pass_'] else '未通过'}** |",
        "",
        "## 口径声明", "",
        "- 交易集、买卖日期、成本与名义本金与冻结协议完全一致，未删除任何亏损交易。",
        "- 持仓市值按 `N × qfq_t / qfq_entry` 逐日盯市（与逐笔收益口径一致，含分红再投资）。",
        "- 初始资金取“完整执行全部 11,419 笔所需的最低现金”，现金不允许为负（无杠杆）。",
        "- 回撤基于逐日 NAV（含未实现盈亏），不是逐笔等额账本。",
    ]
    with open(os.path.join(OUT_DIR, "L1_LEDGER_RISK_REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
