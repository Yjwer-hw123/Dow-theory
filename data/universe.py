"""指数候选 Universe —— 用于做鲁棒性比较.

核心纪律（README 第五节）：
    不能因为某一组指数回测最好，就把它永久定成答案。
    必须问：如果换一组相近的市场指数，道氏信号还成立吗？

A 股里 *没有* 天然的"铁路/工业"二分。我们在找的是：
    两个相互独立、又共同反映同一经济体运行的宽基平均价格指数。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexMeta:
    code: str        # 东方财富 secid 中的数字部分
    secid: str       # 完整 secid
    name: str
    role: str        # 类比角色说明


def _secid(code: str) -> str:
    code = str(code)
    # 中证/上证系列 → 1.xxx；深证系列 → 0.xxx
    return f"1.{code}" if code.startswith(("000", "88", "999")) else f"0.{code}"


INDICES: dict[str, IndexMeta] = {
    m.code: m
    for m in [
        IndexMeta("000300", _secid("000300"), "沪深300", "大盘核心（工业指数类比角色）"),
        IndexMeta("000905", _secid("000905"), "中证500", "中小盘（铁路指数类比角色）"),
        IndexMeta("000510", _secid("000510"), "中证A500", "新一代宽基核心"),
        IndexMeta("000001", _secid("000001"), "上证指数", "沪市全样本"),
        IndexMeta("399001", _secid("399001"), "深证成指", "深市核心"),
        IndexMeta("000852", _secid("000852"), "中证1000", "小盘"),
    ]
}


# 候选配对：(名字, A, B)
PAIRS: list[tuple[str, str, str]] = [
    ("沪深300 + 中证500", "000300", "000905"),
    ("中证A500 + 中证500", "000510", "000905"),
    ("上证指数 + 深证成指", "000001", "399001"),
    ("沪深300 + 中证A500", "000300", "000510"),
    ("沪深300 + 中证1000", "000300", "000852"),
]

DEFAULT_PAIR = PAIRS[0]


def get_pair(name: str | int = 0) -> tuple[str, str, str]:
    if isinstance(name, int):
        return PAIRS[name]
    for p in PAIRS:
        if p[0] == name:
            return p
    raise KeyError(f"unknown pair: {name}")
