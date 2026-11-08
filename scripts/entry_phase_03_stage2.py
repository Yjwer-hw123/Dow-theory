# -*- coding: utf-8 -*-
"""EP-03 第二轮：按冻结的预注册规则复核特征（不做组合、不做资金曲线）。

规则来源：out/entry_phase_03/EP03_ROUND2_PREREGISTER.md
（SHA256 EA98DBFDAAB09E05FC39941557E6ACFEFA8092D62596B83282EAFBD900E9609A）

输入（均为第一轮已产出，不重新计算特征）：
  out/entry_phase_03/p3_summary.json     三段 IC / t / 天数
  out/entry_phase_03/p3_ic_by_year.csv   年度 IC（用于年度一致性与 walk-forward）

输出：
  p3r2_selection.csv            选择段判定 + 最终 OOS
  p3r2_walkforward.csv          逐年 walk-forward 明细
  p3r2_walkforward_summary.csv  walk-forward 汇总
  p3r2_summary.json             结构化结果

用法：python scripts/entry_phase_03_stage2.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "entry_phase_03")

YEAR_SEL = (2017, 2021)
YEAR_OOS = (2022, 2026)
IC_MIN = 0.03          # |IC_S|
T_MIN = 2.5            # |t_S| 与 |t_OOS|
YEAR_MIN = 3           # 选择段同号年数下限
YEAR_IC_MIN = 0.02     # 计入"同号年"的年度 |IC| 下限
MAX_CAND = 3


def load_summary() -> dict:
    with open(os.path.join(OUT, "p3_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def wmean_ic(sub: pd.DataFrame) -> float:
    n = sub["n"].to_numpy(float)
    ic = sub["ic"].to_numpy(float)
    return float((ic * n).sum() / n.sum())


def main() -> None:
    sm = load_summary()
    feats = [f for f in sm["meta"]["features"] if f != "b0_len"]
    yr = pd.read_csv(os.path.join(OUT, "p3_ic_by_year.csv"))
    yr = yr[yr["feature"].isin(feats)].copy()

    rows, wf_rows = [], []
    for ft in feats:
        tr = sm["results"]["train"]["features"][ft]
        va = sm["results"]["validate"]["features"][ft]
        te = sm["results"]["test"]["features"][ft]

        ic_s, t_s = float(va["ic_mean"]), float(va["ic_t"])
        ic_o, t_o = float(te["ic_mean"]), float(te["ic_t"])
        d = 1.0 if ic_s >= 0 else -1.0

        ys = yr[(yr["feature"] == ft) & yr["year"].between(*YEAR_SEL)]
        ys = ys[ys["n"] >= 50]
        y_match = int(((np.sign(ys["ic"]) == d) & (ys["ic"].abs() >= YEAR_IC_MIN)).sum())
        y_elig = int(len(ys))

        pass_sel = (abs(ic_s) >= IC_MIN) and (abs(t_s) >= T_MIN) and (y_match >= YEAR_MIN)
        pass_oos = (np.sign(ic_o) == d) and (abs(t_o) >= T_MIN)
        pass_final = bool(pass_sel and pass_oos)

        rows.append(dict(
            feature=ft, ic_train=float(tr["ic_mean"]),
            ic_sel=ic_s, t_sel=t_s, days_sel=int(va["n_days_ic"]), dir=d,
            years_match=y_match, years_elig=y_elig, pass_sel=bool(pass_sel),
            ic_oos=ic_o, t_oos=t_o, days_oos=int(te["n_days_ic"]),
            pass_oos=bool(pass_oos), pass_final=pass_final,
        ))

        # 逐年 walk-forward：方向只用当年之前的年度 IC（按信号数加权）
        all_y = yr[yr["feature"] == ft].sort_values("year")
        for _, r in all_y.iterrows():
            prior = all_y[all_y["year"] < r["year"]]
            if len(prior) == 0:
                continue
            dp = 1.0 if wmean_ic(prior) >= 0 else -1.0
            wf_rows.append(dict(
                feature=ft, year=int(r["year"]), n=int(r["n"]),
                dir_prior=dp, ic_year=float(r["ic"]),
                aligned_ic=float(dp * r["ic"]), n_prior=int(len(prior)),
            ))

    sel = pd.DataFrame(rows).sort_values(["pass_final", "ic_sel"],
                                         ascending=[False, False])
    sel.to_csv(os.path.join(OUT, "p3r2_selection.csv"), index=False, encoding="utf-8")

    wf = pd.DataFrame(wf_rows)
    wf.to_csv(os.path.join(OUT, "p3r2_walkforward.csv"), index=False, encoding="utf-8")

    wfs = []
    for ft, g in wf.groupby("feature"):
        a = g["aligned_ic"].to_numpy(float)
        share = float((a > 0).mean())
        mean_a = float(a.mean())
        t_a = (float(mean_a / (a.std(ddof=1) / np.sqrt(a.size))
                     ) if a.size > 1 and a.std(ddof=1) > 0 else np.nan)
        wfs.append(dict(feature=ft, n_years=int(a.size), same_sign_share=share,
                        mean_aligned_ic=mean_a, t_aligned=t_a,
                        mean_raw_ic=float(g["ic_year"].mean()),
                        years=",".join(str(int(y)) for y in g["year"])))
    wfw = pd.DataFrame(wfs).sort_values("mean_aligned_ic", ascending=False)
    wfw.to_csv(os.path.join(OUT, "p3r2_walkforward_summary.csv"),
               index=False, encoding="utf-8")

    cand = sel[sel["pass_final"]].head(MAX_CAND)["feature"].tolist()

    summary = dict(
        rule=dict(ic_min=IC_MIN, t_min=T_MIN, year_min=YEAR_MIN,
                  year_ic_min=YEAR_IC_MIN, year_sel=list(YEAR_SEL),
                  year_oos=list(YEAR_OOS), max_candidates=MAX_CAND,
                  prereg_sha256="EA98DBFDAAB09E05FC39941557E6ACFEFA8092D62596B83282EAFBD900E9609A"),
        n_features=len(feats),
        n_pass_sel=int(sel["pass_sel"].sum()),
        n_pass_final=int(sel["pass_final"].sum()),
        candidates=cand,
        selection=sel.to_dict(orient="records"),
        walkforward=wfs,
    )
    with open(os.path.join(OUT, "p3r2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    cols = ["feature", "ic_sel", "t_sel", "years_match", "pass_sel",
            "ic_oos", "t_oos", "pass_oos", "pass_final"]
    print(sel[cols].to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\n候选：", cand if cand else "无")
    print("\nwalk-forward 汇总（按对齐后 IC 均值降序，前 8）：")
    print(wfw.head(8).to_string(index=False, float_format=lambda x: f"{x: .4f}"))


if __name__ == "__main__":
    main()
