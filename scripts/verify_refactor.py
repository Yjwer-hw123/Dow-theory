"""重构验证：新引擎输出必须与 P10 冻结时的输出**逐字段一致**.

line/volume 的数组化只是性能优化（消除主循环里的 O(n²) tolist/slice）。
它只允许改变速度，不允许改变任何一个数字或标签。本脚本以 P10 阶段冻结的
`out/regime_000300_000905_v0.1.csv` 为基准做逐列回归比对。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.dow_params import get_params  # noqa: E402
from core.regime_engine import analyze_pair  # noqa: E402
from data.loader import load_index_pair  # noqa: E402

OLD_CSV = ROOT / "out" / "regime_000300_000905_v0.1.csv"


def norm_val(v) -> str:
    """把 None / NaN / 空串统一成 'NA'，日期统一成 YYYY-MM-DD."""
    if v is None:
        return "NA"
    try:
        if pd.isna(v):
            return "NA"
    except (TypeError, ValueError):
        pass
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    return s if s else "NA"


def main() -> int:
    if not OLD_CSV.exists():
        print(f"[ERROR] 找不到基准输出 {OLD_CSV}")
        return 1
    old = pd.read_csv(OLD_CSV)

    a, b = load_index_pair("000300", "000905")
    t = time.time()
    df, _ = analyze_pair(a, b, get_params("v0.1"), strict_invariants=False)
    secs = time.time() - t
    print(f"[INFO] 新引擎耗时 {secs:.2f}s（重构前 26.1s）  rows={len(df)}")
    print(f"[INFO] 基准 {old.shape}  →  新 {df.shape}")

    ok = True
    if len(old) != len(df):
        print(f"[FAIL] 行数不同: {len(old)} vs {len(df)}")
        ok = False
    n = min(len(old), len(df))

    for c in df.columns:
        if c not in old.columns:
            print(f"[WARN] 基准缺少列 {c}，跳过比对")
            continue
        o = old[c].iloc[:n]
        nw = df[c].iloc[:n]
        if pd.api.types.is_numeric_dtype(nw) and not pd.api.types.is_bool_dtype(nw):
            oa = pd.to_numeric(o, errors="coerce").to_numpy(dtype=float)
            na = pd.to_numeric(nw, errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(oa) & np.isnan(na)
            close = np.isclose(oa, na, rtol=1e-8, atol=1e-10) | both_nan
            if not close.all():
                k = int((~close).sum())
                bad = np.where(~close)[0][:3]
                print(f"[FAIL] {c}: {k} 个数值不一致，例 idx={list(bad)} "
                      f"{[oa[i] for i in bad]} vs {[na[i] for i in bad]}")
                ok = False
        else:
            os_ = np.array([norm_val(v) for v in o])
            ns_ = np.array([norm_val(v) for v in nw])
            neq = os_ != ns_
            if neq.any():
                k = int(neq.sum())
                i0 = int(np.argmax(neq))
                print(f"[FAIL] {c}: {k} 行不一致，例 idx={i0} "
                      f"基准={os_[i0]!r} 新={ns_[i0]!r}")
                ok = False

    print("-" * 64)
    if ok:
        print("[PASS] 重构后输出与 P10 冻结输出逐字段一致 —— 纯性能优化，无数值改动")
        return 0
    print("[FAIL] 存在差异 —— 重构改变了结果，不可接受")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
