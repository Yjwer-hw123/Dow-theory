"""P10.1-C 全局无前瞻审计 —— Prefix Invariance Test.

原理
----
如果引擎在 t 日没有使用任何 t 日之后才存在的信息，那么下面两个状态必须完全一致：

    full[t]    : 用**全部**历史数据跑引擎，取第 t 日的输出
    prefix[t]  : 只用**截至 t** 的数据跑引擎，取最后一天的输出

    full[t] == prefix[t]   对每一个 t 都必须成立

只要存在一个 t 使两者不等，就证明引擎在 t 日消费了未来信息。

为什么这比逐个模块写针对性测试强
--------------------------------
它**不预设泄漏发生在哪里**。数据流是

    Swing → Confirmation → Trend → Secondary → Line → Volume → Regime

任何一个环节含未来函数，都会在这次比对里暴露，不需要事先猜到是哪个模块。

最容易被忽略的一类泄漏（本测试的主要目标）
----------------------------------------
浪识别是"极值确认"式的：一条浪的 `end_date` 是**极值点当天**，但这条浪
只有在价格从极值反向走够 3% 之后**才会被创建出来**。

    高点在 3/01  →  直到 3/20 跌够 3%，这条上升浪才被 add()

如果任何下游（确认、趋势、行程）拿 `end_date`（3/01）当作"这条信息何时可知"，
引擎就会在 3/01 那天使用 3/20 才知道的事 —— 这是**极值日 ≠ 可知日**型的前瞻，
和 P10 修掉的"确认日"型前瞻是两个独立的 bug。
"""

from __future__ import annotations

import random
from typing import Any, Optional

import pandas as pd

# 参与比对的全部状态字段（覆盖数据流的每一层输出）
COMPARE_FIELDS: list[str] = [
    # trend engine
    "primary_trend",
    "trend_integrity",
    "reversal_candidate",
    "H1", "H2", "L1", "L2",
    # confirmation
    "last_confirmed_swing",
    "dual_confirmation",
    # secondary reaction（含 lineage）
    "secondary_reaction",
    "retracement_ratio",
    "secondary_duration",
    "primary_leg_id",
    "leg_origin_date",
    "leg_origin_price",
    "leg_extreme_date",
    "leg_extreme_price",
    "structure_ref",
    # line / volume
    "line_state",
    "line_active",
    "line_breakout",
    "volume_state",
    # regime
    "dow_signal",
]

FLOAT_TOL = 1e-6


def _norm(v: Any) -> Any:
    """把 None / NaN / NaT 统一成 None，避免把'缺失'误判成'不一致'."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _eq(a: Any, b: Any) -> bool:
    a, b = _norm(a), _norm(b)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        return abs(fa - fb) <= FLOAT_TOL * max(1.0, abs(fa), abs(fb))
    return a == b


def diff_rows(full_row: pd.Series, prefix_row: pd.Series,
              fields: Optional[list[str]] = None) -> list[dict]:
    """逐字段比对两行，返回差异清单（保持 COMPARE_FIELDS 的顺序 → 第一个即最上游）."""
    out = []
    for f in (fields or COMPARE_FIELDS):
        a, b = full_row.get(f), prefix_row.get(f)
        if not _eq(a, b):
            out.append({"field": f, "full": _norm(a), "prefix": _norm(b)})
    return out


def sample_indices(n_rows: int, n_samples: int, seed: int = 0,
                   min_prefix: int = 300) -> list[int]:
    """抽取审计点：等距铺满 + 随机补充，保证覆盖不同市场阶段."""
    hi = n_rows - 1
    if hi <= min_prefix:
        return []
    # 等距点：把样本期切成 n_samples 段
    step = max(1, (hi - min_prefix) // max(1, (n_samples // 2)))
    even = list(range(min_prefix, hi + 1, step))
    rng = random.Random(seed)
    rand = rng.sample(range(min_prefix, hi + 1), min(n_samples, hi - min_prefix))
    return sorted(set(even + rand))


def run_prefix_invariance(df_a: pd.DataFrame,
                          df_b: pd.DataFrame,
                          params,
                          n_samples: int = 24,
                          seed: int = 0,
                          min_prefix: int = 300,
                          with_volume: bool = True,
                          analyze_fn=None,
                          fields: Optional[list[str]] = None,
                          progress=None) -> tuple[list[dict], dict]:
    """执行 Prefix Invariance 审计.

    Returns
    -------
    (rows, summary)
        rows    : 每个审计点的结果（含差异明细）
        summary : 汇总（不一致点数、按字段统计的不一致频次）
    """
    if analyze_fn is None:
        from core.regime_engine import analyze_pair as analyze_fn  # type: ignore

    full, _ = analyze_fn(df_a, df_b, params, with_volume=with_volume,
                         strict_invariants=False)
    idxs = sample_indices(len(full), n_samples, seed=seed, min_prefix=min_prefix)

    rows: list[dict] = []
    for k, t in enumerate(idxs):
        pa = df_a.iloc[: t + 1].reset_index(drop=True)
        pb = df_b.iloc[: t + 1].reset_index(drop=True)
        rec: dict = {"t": t, "date": full.iloc[t]["date"], "n_diffs": 0,
                     "diffs": [], "error": None}
        try:
            pre, _ = analyze_fn(pa, pb, params, with_volume=with_volume,
                                strict_invariants=False)
            d = diff_rows(full.iloc[t], pre.iloc[-1], fields=fields)
            rec["n_diffs"] = len(d)
            rec["diffs"] = d
            rec["first_diff"] = d[0] if d else None
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
        if progress is not None:
            progress(k + 1, len(idxs), rec)

    # ---- 汇总 ----
    bad = [r for r in rows if r["n_diffs"] or r["error"]]
    field_hist: dict[str, int] = {}
    for r in rows:
        for d in (r["diffs"] or [])[:1]:     # 只看首个差异 → 定位最上游的泄漏点
            field_hist[d["field"]] = field_hist.get(d["field"], 0) + 1

    summary = {
        "n_checked": len(rows),
        "n_consistent": len(rows) - len(bad),
        "n_inconsistent": len(bad),
        "first_diff_field_hist": dict(sorted(field_hist.items(),
                                             key=lambda kv: -kv[1])),
        "invariant_ok": len(bad) == 0,
    }
    return rows, summary


def format_report(rows: list[dict], summary: dict, max_show: int = 12) -> str:
    """把审计结果渲染成可读文本."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("P10.1-C  Prefix Invariance（全局无前瞻审计）")
    lines.append("=" * 72)
    lines.append(f"审计点: {summary['n_checked']}   "
                 f"一致: {summary['n_consistent']}   "
                 f"不一致: {summary['n_inconsistent']}")
    lines.append("")

    if summary["invariant_ok"]:
        lines.append("[PASS] 全部审计点 prefix == full —— 未发现第二层前瞻偏差")
        return "\n".join(lines)

    lines.append("[FAIL] 存在 prefix != full —— 引擎在 t 日消费了未来信息")
    lines.append("")
    lines.append("首个差异字段频次（越靠上越是泄漏源头）:")
    for f, c in summary["first_diff_field_hist"].items():
        lines.append(f"    {f:<24} {c}")
    lines.append("")

    bad = [r for r in rows if r["n_diffs"] or r["error"]]
    lines.append(f"不一致点明细（最多显示 {max_show} 个）:")
    for r in bad[:max_show]:
        if r["error"]:
            lines.append(f"  {r['date']}  ERROR {r['error']}")
            continue
        fd = r["first_diff"]
        lines.append(
            f"  {r['date']}  差异 {r['n_diffs']} 项 | 首个: {fd['field']}"
            f"  full={fd['full']!r}  prefix={fd['prefix']!r}"
        )
    if len(bad) > max_show:
        lines.append(f"  ... 另有 {len(bad) - max_show} 个")
    return "\n".join(lines)
