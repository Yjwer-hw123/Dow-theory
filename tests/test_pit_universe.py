"""pit_universe 的单元测试（P12-0 完成判据 3）。

用**合成面板**验证，不依赖网络与真实缓存 —— 这样任何时刻都能跑，
结论完全由断言决定，不受数据源漂移影响。

要钉死的行为
------------
设股票 c 在 t1 上市、t2 退市（t1 < t2）。pit_pool 必须给出：

    t < t1            -> c 不在池（尚未上市）
    t1 <= t <= t2     -> c 在池（但受 min_history 约束，上市后前 min_history-1 天仍不在）
    t > t2            -> c 不在池（已退市）

另外三条容易被写错的分支：
    - 停牌（区间内出现 NaN）      -> 当日不在池
    - min_history 恰好边界        -> off-by-one 必须正确
    - 传入非交易日（如周日）      -> 回退到前一个交易日，不能返回空
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.pit_universe import pit_pool, pit_pool_series  # noqa: E402


def make_panel() -> pd.DataFrame:
    """合成面板：index=2005-01-03..2005-04-29 的交易日（周一到周五）。

    三只股票：
      A 全程有价（2005-01-03 ~ 末日）
      B 中途上市（LIST_DAY 起有价）——模拟"2005-03-01 上市"
      C 中途退市（DELIST_DAY 后为 NaN）——模拟"2005-03-01 退市"
      D 全程有价但 2005-02-14 ~ 2005-02-18 停牌（NaN）
    """
    dates = pd.bdate_range("2005-01-03", "2005-04-29")  # 约 82 个交易日
    n = len(dates)

    def series(first_idx: int, last_idx: int, halt: tuple[int, int] | None = None):
        s = pd.Series(np.nan, index=dates, dtype=float)
        s.iloc[first_idx:last_idx + 1] = 10.0
        if halt is not None:
            s.iloc[halt[0]:halt[1] + 1] = np.nan
        return s

    LIST_DAY = pd.Timestamp("2005-03-01")
    DELIST_DAY = pd.Timestamp("2005-03-01")
    i_list = int(dates.get_indexer([LIST_DAY], method="bfill")[0])
    i_delist = int(dates.get_indexer([DELIST_DAY], method="ffill")[0])

    i_halt0 = int(dates.get_indexer([pd.Timestamp("2005-02-14")], method="bfill")[0])
    i_halt1 = int(dates.get_indexer([pd.Timestamp("2005-02-18")], method="ffill")[0])

    return pd.DataFrame({
        "A": series(0, n - 1),
        "B": series(i_list, n - 1),            # 2005-03-01 上市
        "C": series(0, i_delist),              # 2005-03-01 退市（当日仍有 bar）
        "D": series(0, n - 1, halt=(i_halt0, i_halt1)),  # 全程 + 停牌一周
    })


def test_上市前_区间内_退市后() -> None:
    p = make_panel()
    MH = 1  # 关掉次新过滤，只看"有没有 bar"

    # 1) 上市前：B 不在池，A/C/D 在
    pool = pit_pool(p, "2005-02-01", min_history=MH)
    assert "B" not in pool, f"上市前 B 不该在池: {pool}"
    assert set(pool) == {"A", "C", "D"}, pool

    # 2) 上市后（且已够 min_history）：B 进池
    pool = pit_pool(p, "2005-03-02", min_history=MH)
    assert "B" in pool, f"上市后 B 应在池: {pool}"

    # 3) 退市后：C 出池
    pool = pit_pool(p, "2005-04-01", min_history=MH)
    assert "C" not in pool, f"退市后 C 不该在池: {pool}"
    assert set(pool) == {"A", "B", "D"}, pool

    print("  [ok] 上市前 / 区间内 / 退市后 三段正确")


def test_退市当日仍在池() -> None:
    """退市日当天仍有 bar，且此时仍属于可选池（当日还能卖）。"""
    p = make_panel()
    pool = pit_pool(p, "2005-03-01", min_history=1)
    assert "C" in pool, "退市当日 C 仍有 bar，应在池"
    print("  [ok] 退市当日仍在池（末根 bar 即退市日）")


def test_min_history_边界() -> None:
    """累计 bar 数 >= min_history 才进池，off-by-one 必须对。

    面板只有 85 根（2005-01-03 ~ 2005-04-29），B 从 2005-03-01 起才有 bar，
    所以用 min_history=20 在面板内精确验证边界 —— 边界行为与 60 完全一致。
    """
    p = make_panel()
    dates = p.index
    MH = 20
    i_list = int(dates.get_indexer([pd.Timestamp("2005-03-01")], method="bfill")[0])
    assert i_list + MH < len(dates), "面板太短，无法验证边界"

    d_19 = dates[i_list + MH - 2]   # 累计 MH-1 根
    d_20 = dates[i_list + MH - 1]   # 累计 MH 根

    # 先确认前置条件：B 在这两天的累计 bar 数确实是我们以为的值
    assert int(p.loc[:d_19, "B"].notna().sum()) == MH - 1
    assert int(p.loc[:d_20, "B"].notna().sum()) == MH

    assert "B" not in pit_pool(p, d_19, min_history=MH), f"累计 {MH-1} 根不该进池"
    assert "B" in pit_pool(p, d_20, min_history=MH), f"累计 {MH} 根应进池"
    print(f"  [ok] min_history 边界精确（{d_19.date()} 累计{MH-1}不在 / "
          f"{d_20.date()} 累计{MH}在）")


def test_停牌当日不在池() -> None:
    p = make_panel()
    during = pit_pool(p, "2005-02-16", min_history=1)     # 停牌中
    before = pit_pool(p, "2005-02-11", min_history=1)     # 停牌前
    after = pit_pool(p, "2005-02-21", min_history=1)      # 复牌后
    assert "D" not in during, "停牌期间 D 不该在池"
    assert "D" in before and "D" in after, "停牌前后 D 应在池"
    print("  [ok] 停牌当日不在池，复牌后回池")


def test_非交易日回退() -> None:
    """传入周日（非交易日）应回退到前一交易日，而不是返回空池。"""
    p = make_panel()
    sunday = pd.Timestamp("2005-02-06")   # 周日
    assert sunday not in p.index
    pool_sun = pit_pool(p, sunday, min_history=1)
    pool_fri = pit_pool(p, "2005-02-04", min_history=1)   # 同周五
    assert pool_sun == pool_fri, (pool_sun, pool_fri)
    assert len(pool_sun) > 0
    print("  [ok] 非交易日回退到前一交易日")


def test_早于面板起点返回空() -> None:
    p = make_panel()
    assert pit_pool(p, "2004-01-01") == []
    print("  [ok] 早于面板起点返回空池")


def test_pool_series_单调合理() -> None:
    p = make_panel()
    s = pit_pool_series(p, ["2005-01-31", "2005-02-28", "2005-03-31", "2005-04-29"],
                        min_history=1)
    assert list(s.index) == [pd.Timestamp(x) for x in
                             ["2005-01-31", "2005-02-28", "2005-03-31", "2005-04-29"]]
    assert s.loc["2005-01-31"] == 3, s        # A C D
    assert s.loc["2005-04-29"] == 3, s        # A B D（C 已退）
    print(f"  [ok] pit_pool_series: {dict(s.astype(int))}")


if __name__ == "__main__":
    print("PIT 池单元测试（合成面板，不依赖网络）")
    for fn in [
        test_上市前_区间内_退市后,
        test_退市当日仍在池,
        test_min_history_边界,
        test_停牌当日不在池,
        test_非交易日回退,
        test_早于面板起点返回空,
        test_pool_series_单调合理,
    ]:
        fn()
    print("\nALL PASSED")
