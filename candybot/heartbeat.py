"""主动发言心跳调度（任务 4）：群聊空闲一段时间后，自主决定要不要冒个泡。

CandyBot 本是纯消息驱动的——没有新消息就永远不会说话。本模块参照 MaiBot
「心流」空闲循环的轻量版：每群维护最后一条他人消息的入库时刻，静默满一个
随机空闲窗口后，把一次「空闲评估」作为特殊条目投进该群**现有的串行队列**
（关键设计：与真实消息同队列排队，天然避免「群里恰好来了新消息」的竞态，
队列顺序保证不叠发）。评估本身在 bot 层执行（查护栏 → judge 角色决定说不
说 → reply 管线生成 → 现有发送链路），本模块只负责「什么时候醒来看一眼」
与行为节制：

- 退避：实际发出过主动发言后进入 respond_reset_minutes 的观察窗——窗内群里
  有任何他人消息＝「有人接/场子热了」，退避倍数重置为 1；观察窗到点仍没人
  接，下轮空闲窗口 ×2，封顶 ×8（越没人理越安静）。
- 活跃门槛：only_active_today 时，仅当天有过 ≥min_today_messages 条他人
  消息的群参与心跳（死群不冒泡）。
- 每群每日发言上限由 bot 层在调 LLM 前检查（本模块只记账 spoken_today）。

enabled=false（默认）时调度器**不创建任何 asyncio 任务**，仅随消息入库顺带
维护 last_inbound/seq 等纯内存记账（无行为差异）；热重载打开后 sync() 拉起
循环任务，关闭时立即取消任务并清空全部待触发窗口——「改 false 立即停止全
部待触发心跳」（已投递未执行的评估条目由 bot 层出队时按开关作废）。

时钟与随机源都是实例属性（_clock/_rng），测试可以注入假时钟驱动调度、固定
种子验证窗口分布；tick 循环体单独拆成 `await self.tick()`，测试可直接步进。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅注解引用，避免运行时循环导入
    from .models import Settings

logger = logging.getLogger(__name__)

# 调度循环的步进秒数（任务书「每秒/每 N 秒 tick 一次」）：到期判定与
# 观察窗到点都在这里完成，秒级精度对 600 秒量级的空闲窗口绰绰有余。
TICK_SECONDS = 1.0
# 无人接话时的退避倍数上限（窗口 ×2 递增，封顶 ×8）
BACKOFF_CAP = 8


@dataclass(slots=True)
class IdleEvaluation:
    """投进群串行队列的「空闲评估」条目（与 NormalizedMessage 并列的队列载荷）。

    seq 是入队时刻该群的他人消息入库计数：出队执行时计数若已变大，说明
    空闲已终结，本轮直接放弃（不要拿旧上下文说话），重新排程。
    enqueued_at 记录入队时刻，仅作日志/调试用。
    """

    enqueued_at: float
    seq: int


@dataclass(slots=True)
class _GroupState:
    """单群的心跳记账（全内存，重启后从第一条新消息重新计起）。"""

    seq: int = 0  # 他人消息入库计数（竞态检查用）
    last_inbound: float = 0.0  # 最近一条他人消息的入库时刻（0=启动以来没见过消息）
    today: date | None = None  # today_inbound 所属自然日
    today_inbound: int = 0  # 当天他人消息数（only_active_today 门槛）
    spoken_day: date | None = None  # spoken_count 所属自然日
    spoken_count: int = 0  # 当天已实际发出的主动发言数（每日上限记账）
    multiplier: int = 1  # 空闲窗口退避倍数（1/2/4/8）
    watch_since: float | None = None  # 发出主动发言后、观察窗未结算的时刻
    next_due: float | None = None  # 下一次到期时刻；None=还没排（或消息刚到、重新计）
    pending: bool = False  # 是否有一条空闲评估在队列里排队/执行中
    base_override: float | None = None  # 评估结束后排程的基准时刻（从本轮收尾起算）

    def add_inbound(self, today: date, now: float) -> None:
        self.seq += 1
        self.last_inbound = now
        if self.today != today:
            self.today = today
            self.today_inbound = 0
        self.today_inbound += 1
        self.next_due = None
        self.base_override = None  # 真实活动面前压倒本轮收尾的排程基准

    def today_ok(self, today: date, min_messages: int) -> bool:
        return self.today == today and self.today_inbound >= min_messages

    def spoken_today(self, today: date) -> int:
        return self.spoken_count if self.spoken_day == today else 0


class HeartbeatScheduler:
    """单个 asyncio 循环任务服务全部群（不是每群一个常驻任务，避免群多时资源散）。"""

    def __init__(
        self,
        settings: Callable[[], Settings],
        enqueue: Callable[[int, IdleEvaluation], Awaitable[None]],
        *,
        clock: Callable[[], float] = time.time,
        tick_seconds: float = TICK_SECONDS,
    ):
        # settings/enqueue 都走回调：热重载后取到新快照，入队复用 bot 的串行队列
        self._settings = settings
        self._enqueue = enqueue
        self._clock = clock
        self._tick_seconds = tick_seconds
        self._states: dict[int, _GroupState] = {}
        self._task: asyncio.Task[None] | None = None
        # 空闲窗口掷点的随机源（加密安全，与 ai/bot 的掷点约定一致）；
        # 测试替换为固定种子的 random.Random 即可复现窗口分布
        self._rng: random.Random = random.SystemRandom()

    # ------------------------------------------------------------ 记账入口

    def _state(self, group_id: int) -> _GroupState:
        state = self._states.get(group_id)
        if state is None:
            state = _GroupState()
            self._states[group_id] = state
        return state

    @property
    def _today(self) -> date:
        return datetime.fromtimestamp(self._clock()).date()

    def note_inbound(self, group_id: int) -> None:
        """一条他人消息入库（在 bot 的归一化入库处调用）：空闲计时重新起算。

        同步 O(1) 纯内存操作；观察窗内进来＝「有人接话」，退避倍数当场重置。
        """
        state = self._state(group_id)
        now = self._clock()
        state.add_inbound(self._today, now)
        reset = self._settings().proactive.respond_reset_minutes * 60.0
        if state.watch_since is not None and now <= state.watch_since + reset:
            logger.debug(
                "群 %d 主动发言后有人接话，退避重置（倍数恢复 ×1）", group_id
            )
            state.multiplier = 1
            state.watch_since = None

    def seq_of(self, group_id: int) -> int:
        """当前他人消息入库计数（出队竞态检查用）。"""
        return self._state(group_id).seq

    def spoken_today(self, group_id: int) -> int:
        """该群当天已实际发出的主动发言数（跨天自动归零）。"""
        return self._state(group_id).spoken_today(self._today)

    def note_spoken(self, group_id: int) -> None:
        """一条主动发言实际发出（≥1 条正文成功）：记当日上限账、开启观察窗。

        上限计数跨天自动归零（spoken_today 按日期判定）；退避倍数**不**跟着
        跨天归零——「最近没人接我的话」这个信号跨天仍然成立，只有观察窗内
        有人接话才重置。"""
        state = self._state(group_id)
        today = self._today
        if state.spoken_day != today:
            state.spoken_day = today
            state.spoken_count = 0
        state.spoken_count += 1
        state.watch_since = self._clock()
        logger.debug("群 %d 主动发言已记账（今日第 %d 次）", group_id, state.spoken_count)

    # ------------------------------------------------------------ 生命周期

    def sync(self) -> None:
        """按当前配置对齐循环任务：enabled 拉起，关闭立即取消并清空待触发窗口。

        bot.start()（启动时）与 reload_settings()（热重载时）各调一次；
        enabled=false 时不创建任何任务——与引入本功能前零差异。
        """
        enabled = self._settings().proactive.enabled
        if enabled and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(
                self._loop(), name="proactive-heartbeat"
            )
            logger.info("主动发言心跳已启动（enabled=true）")
        elif not enabled and self._task is not None:
            if not self._task.done():
                self._task.cancel()
            self._task = None
            self._clear_pending()
            logger.info("主动发言心跳已停止（enabled=false，待触发窗口全部清空）")

    def _clear_pending(self) -> None:
        """清空全部待触发窗口（排程与在途标记）；观察窗与退避倍数保留——
        已在队列里的评估条目出队时按开关作废，届时 finish 会重新排程。"""
        for state in self._states.values():
            state.next_due = None
            state.base_override = None

    async def stop(self) -> None:
        """停机：取消循环任务与队列中未执行的空闲评估条目（后者随队列 worker
        一起被取消，条目本身随进程回收）。"""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._clear_pending()

    async def _loop(self) -> None:
        while True:
            # CancelledError 原样传播（停机/sync 关闭时靠它退出并进入 cancelled 态）
            await asyncio.sleep(self._tick_seconds)
            try:
                await self.tick()
            except Exception:  # pragma: no cover —— 保险丝：调度循环绝不带崩退出
                logger.exception("心跳调度 tick 异常（忽略，下轮继续）")

    # ------------------------------------------------------------ 调度

    def _window(self, state: _GroupState) -> float:
        """下一次空闲窗口：[idle_min, idle_max] 均匀随机 × 当前退避倍数。"""
        ps = self._settings().proactive
        low, high = ps.idle_min_seconds, ps.idle_max_seconds
        if high < low:  # 热重载改出非法区间时保守夹住（配置校验在启动/解析期）
            low, high = high, low
        return self._rng.uniform(low, high) * state.multiplier

    async def tick(self) -> None:
        """单步调度：结算观察窗、排程到期群、投递空闲评估。

        公开为方法便于测试用假时钟直接步进，无需等待真实 sleep。
        """
        ps = self._settings().proactive
        if not ps.enabled:
            return
        now = self._clock()
        today = self._today
        reset_seconds = ps.respond_reset_minutes * 60.0
        for group_id, state in self._states.items():
            if (
                state.watch_since is not None
                and now >= state.watch_since + reset_seconds
            ):
                state.watch_since = None
                if state.multiplier < BACKOFF_CAP:
                    state.multiplier = min(state.multiplier * 2, BACKOFF_CAP)
                    logger.debug(
                        "群 %d 主动发言没人接，下轮空闲窗口退避为 ×%d",
                        group_id,
                        state.multiplier,
                    )
            if state.last_inbound <= 0 or state.pending:
                continue
            if state.next_due is None:
                if state.watch_since is not None:
                    continue  # 观察窗未结算：退避倍数还没定，不提前掷窗口
                if ps.only_active_today and not state.today_ok(today, ps.min_today_messages):
                    continue  # 当天不够活跃：不排程（门槛只拦排程与到期两处）
                base = state.base_override if state.base_override is not None else state.last_inbound
                state.base_override = None
                window = self._window(state)
                state.next_due = base + window
                logger.debug(
                    "群 %d 空闲评估排在 %.0f 秒后（窗口 %.0fs × 退避 %d）",
                    group_id,
                    window,
                    window / max(state.multiplier, 1),
                    state.multiplier,
                )
            if now >= state.next_due:
                if ps.only_active_today and not state.today_ok(today, ps.min_today_messages):
                    continue  # 排好程后跨了天/当天不够活跃：到期也不投
                state.pending = True
                await self._enqueue(
                    group_id, IdleEvaluation(enqueued_at=now, seq=state.seq)
                )
                logger.debug("群 %d 空闲到点，已投递空闲评估", group_id)

    def finish(self, group_id: int) -> None:
        """本轮空闲评估处理完毕（执行或放弃）：解除在途标记、从当下重新排程。

        新窗口此刻就掷定基准（本轮收尾时刻）；若观察窗尚未结算，tick 会等到
        倍数落定后才真正排到期时间——「没人接则下轮窗口 ×2」由此生效。
        """
        state = self._state(group_id)
        state.pending = False
        state.next_due = None
        state.base_override = self._clock()
