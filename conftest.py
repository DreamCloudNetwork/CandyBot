"""测试公共夹具。"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def pytest_configure(config: pytest.Config) -> None:
    """会话启动时先做一轮完整回收并冻结存量对象。

    集成测试断言用的是 asyncio 的毫秒级时序预算（drain_tick 0.15s）。套件
    规模增长后，gen2 回收偶发 200ms 级停顿会随机把 worker 的 DB 往返拖过
    预算，表现为「消息没被处理」的 flake。import/collect 阶段构造的大对象
    图在整个会话内是稳定的，gc.freeze() 把它们移出回收计数字典，gen2 停顿
    随之消失（实测 15/15 → 0/15 flake），而会话内新产生垃圾的回收行为不变。
    """
    gc.collect()
    gc.freeze()
