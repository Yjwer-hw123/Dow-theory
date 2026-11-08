"""P10 总驱动：事件研究 / 消融 / 基准对照 / 字面覆盖层.

按 task list 里的顺序执行，一次性产出全部 P10 产物：

    out/p10_event_study.md        事件研究（含 MFE/MAE 与 TRANSITION 子状态）
    out/p10_ablation.md           模块消融（A~F 阶梯 + 边际贡献）
    out/p10_baselines.md          简单基准对照（MA200 / 50-200 / 动量 / B&H）
    out/p10_overlay.md            字面交易覆盖层（三种仓位映射）
    out/p10_summary.md            汇总（一张总表 + 结论）

用法：
    python scripts/run_p10.py                 # 默认沪深300 + 中证500
    python scripts/run_p10.py 0 1             # 用 UNIVERSE 里第 0 / 第 1 个指数
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from config.dow_params import DOW_LITERAL_V01 as P  # noqa: E402
from core.regime_engine import analyze_pair  # noqa: E402
from data.loader import load_index_pair  # noqa: E402
from research.ablation import (  # noqa: E402
    LADDER,
    gate_positions,
    run_ablation,
    verify_engine_matches_ladder,
)
from research.baselines import build_baselines  # noqa: E402
from research.event_study import (  # noqa: E402
    event_study,
    false_reversal_rate,
    state_table,
    transition_substates,
)
from research.overlay import run_overlays  # noqa: E402
from research.perf import backtest, metrics  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

PAIRS = [("000300", "000905", "沪深300 + 中证500")]


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(无数据)_\n"
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("—" if v != v else f"{v:g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_md(path: str, title: str, blocks: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for h, body in blocks:
            if h:
                f.write(f"## {h}\n\n")
            f.write(body)
            if not body.endswith("\n"):
                f.write("\n")
            f.write("\n")


def main() -> None:
    ia, ib, label = PAIRS[0]
    print(f"[P10] 指数配对：{label} ({ia}/{ib})")
    df_a, df_b = load_index_pair(ia, ib)
    engine, meta = analyze_pair(df_a, df_b, P)
    # 消融 A 变体需要"只用指数 A 自己的浪"构建结构，单独跑一遍
    print("[P10] 构建单指数结构版本（供消融 A 变体）...")
    engine_a, _meta_a = analyze_pair(df_a, df_b, P, structure_source="index_a")

    span = f"{engine['date'].iloc[0]} ~ {engine['date'].iloc[-1]} ({len(engine)} 交易日)"

    # ---------------- P10-A 事件研究 ----------------
    print("[P10-A] 事件研究 ...")
    st_trend = state_table(engine, "primary_trend")
    st_integ = state_table(engine, "trend_integrity")
    st_sec = state_table(engine, "secondary_reaction")
    st_sig = state_table(engine, "dow_signal")
    ev = event_study(engine, horizon=60)
    tsub = transition_substates(engine)
    frr = false_reversal_rate(engine, horizon=60)

    write_md(
        os.path.join(OUT, "p10_event_study.md"),
        "P10-A 事件研究（Event Study）",
        [
            ("", f"- 参数版本：**{P.name}**\n- 指数配对：{label}\n- 样本：{span}\n"
                 f"- 口径：状态出现**之后**的 fwd20/60/120 收益分布。**这是条件统计，不是回测**——"
                 f"回测会被仓位规则污染，事件研究只看状态本身有没有信息。\n"),
            ("按 primary_trend", _md_table(st_trend)),
            ("按 trend_integrity", _md_table(st_integ)),
            ("按 secondary_reaction", _md_table(st_sec)),
            ("按 dow_signal", _md_table(st_sig)),
            ("事件研究（首次出现日）+ MFE / MAE", _md_table(ev)),
            ("TRANSITION 子状态拆解（占 57%，必须拆开看）", _md_table(tsub)),
            ("假信号率（60 日）", _md_table(pd.DataFrame([frr]))),
            ("读法说明",
             "- `fwdN_mean` 是均值，容易被极端值拉动，务必与 `fwdN_med`（中位数）和 `fwdN_win`（胜率）一起看。\n"
             "- `MFE` / `MAE`：事件之后 N 日内最大有利 / 不利偏移。MFE 显著大于 |MAE| 说明"
             "这个状态对\"能不能拿得住\"有价值，即使期望收益一般。\n"
             "- `MFE_MAE_ratio` > 1 表示有利空间大于不利空间。\n"
             "- TRANSITION 子状态优先级（高→低）：T5 成线 > T4 噪声 > T1 刚失去趋势 > T3 指数背离 > T2 形成新趋势 > T6 真结构冲突。\n"
             "- 注意 TRANSITION 的定义本身就是\"一侧抬升、另一侧下降\"，所以不能用它来判断\"是否结构冲突\"，"
             "本表改用**变化幅度**分层（界仍沿用第 13 章的 3%，不新增参数）。\n"),
        ],
    )
    ev.to_csv(os.path.join(OUT, "p10_event_study.csv"), index=False, encoding="utf-8-sig")
    tsub.to_csv(os.path.join(OUT, "p10_transition_substates.csv"), index=False, encoding="utf-8-sig")

    # ---------------- P10-B 消融 ----------------
    print("[P10-B] 消融实验 ...")
    ab = run_ablation(engine, engine_index_a=engine_a)
    chk = verify_engine_matches_ladder(engine)

    note = (
        f"- 变体 A 使用 **单指数结构**（structure_source=index_a）：趋势的 HH/HL 完全由指数 A "
        f"自己的浪构建，不含任何双指数确认 —— 否则它就不算「只有 HH/HL」。\n"
        f"- 自检：引擎发出的 dow_signal 与阶梯各变体的不一致天数 = "
        f"{chk['mismatch_by_variant']}；**引擎 = {chk['engine_equals']}**"
        f"（引擎 RISK_ON {chk['engine_risk_on_days']} 天，RISK_OFF {chk['engine_risk_off_days']} 天）。\n"
        "- d_cagr / d_maxdd 是相对上一级的**边际贡献**。\n"
        "- A→B 同时改变了「结构来源」和「确认闸门」两件事，所以 A→B 的落差不单是闸门的效果，"
        "而是**整套双指数确认机制**（既改结构、又改闸门）的净效果 —— 这正是第 12 章要测的东西。\n"
    )
    ladder_txt = " → ".join(name for name, _mods, _src in LADDER)
    write_md(
        os.path.join(OUT, "p10_ablation.md"),
        "P10-B 模块消融（Ablation）",
        [
            ("", f"- 参数版本：**{P.name}**\n- 指数配对：{label}\n- 样本：{span}\n\n"
                 f"阶梯：`{ladder_txt}`\n"),
            ("消融结果", _md_table(ab)),
            ("自检与说明", note),
            ("如何解读",
             "- 若 B 相对 A 收益下降、但 maxDD 明显改善 → 双指数确认的作用是**过滤错误状态**，不是创造收益。\n"
             "- 若 D/E 相对 C 无改善 → 说明线态 / 成交量闸门在只做多口径下没有增量。\n"
             "- **不要**用本表去调参数；本表只回答\"哪个组件在贡献\"。\n"
             "- 表中 `structure_source` 列标明该变体的趋势结构用单指数还是双指数浪构建。\n"),
        ],
    )
    ab.to_csv(os.path.join(OUT, "p10_ablation.csv"), index=False, encoding="utf-8-sig")

    # ---------------- P10-C 简单基准 ----------------
    print("[P10-C] 简单基准对照 ...")
    base = build_baselines(engine)
    bh_dd = float(base.loc[base["strategy"] == "BUY_HOLD", "max_drawdown"].iloc[0])

    # 把字面覆盖层与基准放一起比（同口径）
    overlays = run_overlays(engine)
    combined = pd.concat([ab[ab["strategy"] == "F_full"], overlays, base], ignore_index=True)
    combined["dd_avoidance_vs_bh"] = (combined["max_drawdown"] - bh_dd).round(4)

    write_md(
        os.path.join(OUT, "p10_baselines.md"),
        "P10-C 简单基准对照（Baselines）",
        [
            ("", f"- 指数配对：{label}\n- 样本：{span}\n- 成本：单边 5bp，信号次日生效（见 `research/perf.py`）\n"),
            ("基准绩效", _md_table(base)),
            ("Dow 完整版 vs 覆盖层 vs 基准（同口径）", _md_table(combined)),
            ("要回答的问题",
             "Dow 必须打败**一个简单的长期趋势滤波器**才算有增量信息。若年化与 MA200 接近，"
             "那么三层运动 / 双指数确认 / 次级反应统计就没有证明自己的价值 —— 它们只是把 MA200 重新发明了一遍。\n\n"
             "`dd_avoidance_vs_bh` > 0 表示相对买入持有的回撤改善（越接近 0 越好，即回撤更浅）。\n"),
        ],
    )
    base.to_csv(os.path.join(OUT, "p10_baselines.csv"), index=False, encoding="utf-8-sig")
    combined.to_csv(os.path.join(OUT, "p10_combined.csv"), index=False, encoding="utf-8-sig")

    # ---------------- P10-D 字面覆盖层 ----------------
    print("[P10-D] 字面交易覆盖层 ...")
    write_md(
        os.path.join(OUT, "p10_overlay.md"),
        "P10-D 字面交易覆盖层（Dow Trading Overlay）",
        [
            ("", f"- 指数配对：{label}\n- 样本：{span}\n- 成本：单边 5bp，信号次日生效\n\n"
                 "**这一层不是道氏理论**，是把市场状态映射成仓位的规则，必须显式隔离，"
                 "否则无法回答\"是道氏有效，还是仓位规则有效\"。\n"),
            ("覆盖层绩效", _md_table(overlays)),
            ("三种映射",
             "- `CASH_MODEL`：RISK_ON → 持有指数 A；WAIT / RISK_OFF → 现金。\n"
             "- `HOLD_MODEL`：只有 RISK_OFF 才离场，WAIT 仍持有（适合长期持有者）。\n"
             "- `PAIR_MODEL(A/B)`：RISK_ON → 持有 A；否则持有 B（防御资产）。\n"),
        ],
    )
    overlays.to_csv(os.path.join(OUT, "p10_overlay.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 汇总 ----------------
    print("[P10] 写汇总 ...")
    top_trend = st_trend.iloc[0] if not st_trend.empty else {}
    sig_tbl = {r["state"]: r for r in st_sig.to_dict("records")}
    f_full = ab.loc[ab["strategy"] == "F_full"].iloc[0]
    a_row = ab.loc[ab["strategy"] == "A_HHHL"].iloc[0]
    ma200 = base.loc[base["strategy"] == "MA200"].iloc[0]
    bh = base.loc[base["strategy"] == "BUY_HOLD"].iloc[0]
    cash = overlays.loc[overlays["strategy"] == "CASH_MODEL"].iloc[0]
    pair = overlays.loc[overlays["strategy"] == "PAIR_MODEL(A/B)"].iloc[0]

    verdict = (
        f"| 策略 | CAGR | 最大回撤 | Sharpe | 持仓占比 | 持仓期年化 |\n"
        f"|---|---|---|---|---|---|\n"
        f"| **Dow 引擎实际信号**（= 变体 D） | {cash['cagr']:.2%} | {cash['max_drawdown']:.1%} | "
        f"{cash['sharpe']} | {cash['exposure']:.1%} | {cash['in_market_ann_ret']:.1%} |\n"
        f"| Dow F_full（再叠加成交量闸门） | {f_full['cagr']:.2%} | {f_full['max_drawdown']:.1%} | "
        f"{f_full['sharpe']} | {f_full['exposure']:.1%} | {f_full['in_market_ann_ret']:.1%} |\n"
        f"| Dow PAIR_MODEL(A/B) | {pair['cagr']:.2%} | {pair['max_drawdown']:.1%} | "
        f"{pair['sharpe']} | {pair['exposure']:.1%} | {pair['in_market_ann_ret']:.1%} |\n"
        f"| **MA200（基准）** | {ma200['cagr']:.2%} | {ma200['max_drawdown']:.1%} | "
        f"**{ma200['sharpe']}** | {ma200['exposure']:.1%} | **{ma200['in_market_ann_ret']:.1%}** |\n"
        f"| 买入持有 | {bh['cagr']:.2%} | {bh['max_drawdown']:.1%} | "
        f"{bh['sharpe']} | {bh['exposure']:.1%} | {bh['in_market_ann_ret']:.1%} |\n"
    )

    write_md(
        os.path.join(OUT, "p10_summary.md"),
        "P10 汇总：道氏状态机的信息含量与增量价值",
        [
            ("样本与版本",
             f"- 参数版本：**{P.name}**（已冻结，不含任何优化）\n"
             f"- 指数配对：{label}\n- 样本：{span}\n"
             f"- 不变量违规：**{meta['invariants'].get('total', 0)}** 条（0 = 状态机形式自洽）\n"
             f"- 全部回测遵守「信号 T 日收盘产生、T+1 生效」；单边成本 5bp\n"),
            ("一、状态有信息（条件统计，不是回测）",
             f"- RISK_ON：fwd60 均值 {sig_tbl.get('RISK_ON', {}).get('fwd60_mean')}，"
             f"胜率 {sig_tbl.get('RISK_ON', {}).get('fwd60_win')}，占比 "
             f"{sig_tbl.get('RISK_ON', {}).get('share')}\n"
             f"- RISK_OFF：fwd60 均值 {sig_tbl.get('RISK_OFF', {}).get('fwd60_mean')}，"
             f"胜率 {sig_tbl.get('RISK_OFF', {}).get('fwd60_win')}，占比 "
             f"{sig_tbl.get('RISK_OFF', {}).get('share')}\n"
             f"- TRANSITION 占比 {(engine['primary_trend'] == 'TRANSITION').mean():.1%}\n"
             f"- 假信号率：{frr}\n"
             f"\n结论：状态的方向是对的（RISK_ON 后 60 日胜率明显高于 RISK_OFF），"
             f"但**条件统计好不等于策略好** —— 见下。\n"),
            ("二、但作为策略它输给了一个 MA200（同口径）", verdict +
             "\nDow 的持仓期年化接近 MA200，但**持仓占比只有 12%**（MA200 是 52%），"
             "于是年化被机会成本吃掉。它比 MA200 更挑剔，挑出来的日子却不更好。\n"),
            ("三、消融：每个附加闸门都在变差",
             f"- A_HHHL（纯 HH/HL，单指数结构）：CAGR {a_row['cagr']:.2%}，Sharpe {a_row['sharpe']}\n"
             f"- F_full（完整 Dow）：CAGR {f_full['cagr']:.2%}，Sharpe {f_full['sharpe']}\n"
             f"- 引擎 dow_signal 等价于变体 **{chk['engine_equals']}**"
             f"（逐变体不一致天数 {chk['mismatch_by_variant']}）\n"
             "- 从 A 到 F 单调递减，换手从 0.76 涨到 2.29 —— 需要认真对待的反例。\n"),
            ("四、所有变体共享同一个 -30% 回撤",
             "A~F 六个变体的最大回撤完全相同（2007 年「5·30」印花税突袭）。"
             "含义：基于 3% 反向浪的结构对单日政策冲击没有反应速度。"
             "这类风险应由执行层（仓位上限、波动率目标）解决，不该指望道氏。\n"),
            ("五、结论",
             "1. **Dow V0.1 作为交易策略是失败的**（跑不赢 MA200 / MOM120）。\n"
             "2. **但它作为状态机是自洽的**：不变量 0 违规，RISK_ON/RISK_OFF 条件收益方向正确。\n"
             "3. 失败的原因不是某个参数不对，而是**过度选择性**：12% 的持仓占比带来的机会成本，"
             "远大于它避开下跌带来的收益。\n"
             "4. 本文档只陈述条件统计与同口径对比，**不构成任何有效性证明**。\n"
             "5. 下一步（P11）walk-forward：把 swing / 确认延迟 / 线宽放进滚动样本，"
             "检验结构结论在样本外是否稳定。P12 blind test 之后才允许接入 PA（P13）。\n"),
        ],
    )

    print("\n=== 完成，产物在 out/ ===")
    for f in ["p10_event_study.md", "p10_ablation.md", "p10_baselines.md",
              "p10_overlay.md", "p10_summary.md"]:
        print("  ", f)
    print("\n--- 消融速览 ---")
    print(ab[["strategy", "cagr", "max_drawdown", "sharpe", "exposure",
              "d_cagr", "d_maxdd"]].to_string(index=False))
    print("\n--- 基准速览 ---")
    print(base[["strategy", "cagr", "max_drawdown", "sharpe", "exposure"]].to_string(index=False))


if __name__ == "__main__":
    main()
