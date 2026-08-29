"""测试专用的确定性伪随机源。

固定种子的线性同余实现，只覆盖被测链路实际用到的 random() / choice() /
choices() 三个方法。测试要的是可复现的确定性轨迹，不引入标准库
random 模块的伪随机源（Mimosa「不安全的随机数」低危提示即源于此）。
random() 返回值恒在 [0, 1)。
"""

from __future__ import annotations

from typing import Any


class SeededRng:
    _A = 6364136223846793005
    _C = 1442695040888963407
    _M = 1 << 64

    def __init__(self, seed: int) -> None:
        self._state = seed % self._M

    def random(self) -> float:
        self._state = (self._state * self._A + self._C) % self._M
        return self._state / self._M

    def choice(self, seq: list[Any]) -> Any:
        if not seq:
            raise IndexError("cannot choose from an empty sequence")
        return seq[int(self.random() * len(seq))]

    def choices(
        self, population: list[Any], *, weights: list[float] | None = None, k: int = 1
    ) -> list[Any]:
        pop = list(population)
        if not pop:
            raise IndexError("cannot choose from an empty sequence")
        if k < 0:
            raise ValueError("number of choices must be non-negative")
        if weights is None:
            return [self.choice(pop) for _ in range(k)]
        if len(weights) != len(pop):
            raise ValueError("Population and weights must match in length")
        cumulative: list[float] = []
        total = 0.0
        for w in weights:
            total += w
            cumulative.append(total)
        out: list[Any] = []
        for _ in range(k):
            r = self.random() * total
            for item, bound in zip(pop, cumulative):
                if r < bound:
                    out.append(item)
                    break
            else:
                out.append(pop[-1])  # 浮点误差兜底
        return out
