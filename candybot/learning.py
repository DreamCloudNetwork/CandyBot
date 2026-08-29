"""后台学习服务：每日群印象（任务 A）、表达学习（任务 B）、黑话学习（任务 C）。

三类学习任务的全部 LLM 调用都发生在后台 asyncio 任务里，绝不阻塞每群
决策队列；任一环节失败只记 warning 日志并跳过本次，不重试、不堆积
（这是「后台任务可以容忍失败」的例外条款——主链路仍按错误完整暴露处理）。

学习成果经 candy.db 持久化（group_impression / expressions / jargons 三表），
使用侧约定：
- 群印象注入提示词 L2（天内字节级稳定，快照缓存在 GroupRuntime）；
- 表达与黑话注入 L4（每次回复现取现注入，属易变信息）。
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, timedelta

from .ai import AIClient
from .database import ExpressionEntry, JargonEntry
from .memory import MemoryManager, seconds_until_next_midnight
from .models import ChatRecord, Settings
from .prompts import learning_chat_text

logger = logging.getLogger(__name__)

# 调度与限流参数（被淘汰缓冲倍数、印象文本预算、每批黑话候选上限、
# 黑话含义入库长度上限）原先写死在本模块，现统一移到 learning 配置段
# （models.LearningSettings），默认值与提取前的字面量一致。


def day_bounds(day: date) -> tuple[float, float]:
    """某自然日的 [start_ts, end_ts) Unix 时间戳界（bot 的竞态防护共用）。"""
    start = datetime.combine(day, datetime.min.time()).timestamp()
    return start, start + 86400.0


def _clip_records_for_impression(
    records: Sequence[ChatRecord], budget: int
) -> list[ChatRecord]:
    """从最新一条向前截取，使聊天文本总长不超预算（至少保留一条）。"""
    kept: list[ChatRecord] = []
    total = 0
    for record in reversed(records):
        cost = len(record.text) + 24  # 昵称前缀与换行的粗略开销
        if kept and total + cost > budget:
            break
        total += cost
        kept.append(record)
    return kept[::-1]


class LearningService:
    """群印象/表达/黑话三类学习任务的调度者（全部后台执行）。

    ai_provider / settings_provider 以回调注入：配置热重载会重建
    AIClient 与 Settings 快照，学习任务每次现取最新实例。
    """

    def __init__(
        self,
        memory: MemoryManager,
        ai_provider: Callable[[], AIClient],
        settings_provider: Callable[[], Settings],
    ):
        self._memory = memory
        self._ai = ai_provider
        self._settings = settings_provider
        # 每群的「被淘汰消息」积累区与在跑的批次任务
        self._pending: dict[int, list[ChatRecord]] = {}
        self._batch_tasks: dict[int, asyncio.Task[None]] = {}
        self._impression_task: asyncio.Task[None] | None = None
        self._stopping = False
        # 表达加权随机抽取的随机源；测试可替换为固定种子的 random.Random。
        self.rng: random.Random = random.SystemRandom()
        # 黑话机械匹配的正则缓存（term → pattern），条目有上限防泄漏。
        self._jargon_patterns: dict[str, re.Pattern[str]] = {}

    # ------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        """启动每日印象循环（幂等）。先补一次昨天：跨过零点停机重启时
        错过的那份印象在启动时补上。"""
        if self._impression_task is None or self._impression_task.done():
            self._impression_task = asyncio.create_task(
                self._daily_impression_loop(), name="daily-group-impression"
            )

    async def stop(self) -> None:
        """取消每日印象循环与在跑的批次任务（不等待其完成）。"""
        self._stopping = True
        tasks: list[asyncio.Task[None]] = []
        if self._impression_task is not None:
            tasks.append(self._impression_task)
            self._impression_task = None
        tasks.extend(self._batch_tasks.values())
        self._batch_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------ 触发入口
    #
    # 表达与黑话学习共用溢出触发时机：GroupMemory 热缓存装满后，每次
    # append 挤出的最旧消息经 note_evicted 交进来，同群攒够一批即在后台
    # 跑一次学习任务。

    def note_evicted(self, group_id: int, record: ChatRecord) -> None:
        """热缓存淘汰回调（memory 的 append 锁内同步调用）：只允许 O(1)
        的记账与 create_task，绝不 await、绝不做重活。"""
        ls = self._settings().learning
        if self._stopping or not ls.enabled:
            return
        if not ls.expression_enabled and not ls.jargon_enabled:
            return
        batch_size = max(ls.expression_batch_size, 1)
        pending = self._pending.setdefault(group_id, [])
        pending.append(record)
        overflow = len(pending) - batch_size * ls.pending_buffer_factor
        if overflow > 0:
            del pending[:overflow]
        if len(pending) < batch_size:
            return
        running = self._batch_tasks.get(group_id)
        if running is not None and not running.done():
            return  # 该群已有批次在跑：继续积累，完成后的下次淘汰再触发
        batch = pending[:batch_size]
        del pending[:batch_size]
        logger.debug(
            "群 %d 攒够 %d 条被淘汰消息，后台触发表达/黑话学习", group_id, len(batch)
        )
        task = asyncio.create_task(
            self._learn_batch(group_id, batch), name=f"group-learning-{group_id}"
        )
        self._batch_tasks[group_id] = task
        task.add_done_callback(self._batch_done)

    def _batch_done(self, task: asyncio.Task[None]) -> None:
        """批次收尾：异常已在任务内消化，这里只负责摘除登记。"""
        for gid, registered in list(self._batch_tasks.items()):
            if registered is task:
                del self._batch_tasks[gid]

    async def _learn_batch(self, group_id: int, batch: list[ChatRecord]) -> None:
        """一批被淘汰消息：表达与黑话学习各自独立执行、互不牵连。"""
        ls = self._settings().learning
        if ls.expression_enabled:
            await self._guarded(self._learn_expressions(group_id, batch))
        if ls.jargon_enabled:
            await self._guarded(self._learn_jargons(group_id, batch))

    async def _guarded(self, coro: Awaitable[None]) -> None:
        """后台学习任务统一容错：失败只记 warning 并跳过本次。"""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("后台学习任务失败，跳过本次", exc_info=True)

    # ------------------------------------------------------------ 任务 B：表达学习

    async def _learn_expressions(self, group_id: int, batch: list[ChatRecord]) -> None:
        ls = self._settings().learning
        chat_text = learning_chat_text(batch)
        if not chat_text.strip():
            return
        candidates = await self._ai().learn_expressions(chat_text)
        if not candidates:
            logger.debug("群 %d 表达学习：本批无可学内容", group_id)
            return
        logger.debug("群 %d 表达学习候选：%r", group_id, candidates)
        ts = time.time()
        written = 0
        for situation, style in candidates:
            if ls.expression_self_review and not await self._ai().review_expression(
                situation, style
            ):
                logger.debug("群 %d 表达自审拒收：当%s时→%s", group_id, situation, style)
                continue
            await self._memory.db.record_expression(group_id, situation, style, ts)
            written += 1
        if written:
            logger.info("群 %d 表达学习：入库 %d 条（候选 %d 条）", group_id, written, len(candidates))

    # ------------------------------------------------------------ 任务 C：黑话学习

    async def _learn_jargons(self, group_id: int, batch: list[ChatRecord]) -> None:
        ls = self._settings().learning
        chat_text = learning_chat_text(batch)
        if not chat_text.strip():
            return
        terms = await self._ai().extract_jargon_terms(chat_text)
        if not terms:
            logger.debug("群 %d 黑话学习：本批无候选词条", group_id)
            return
        logger.debug("群 %d 黑话候选：%r", group_id, terms)
        saved = 0
        for term in terms[: ls.jargon_candidates_per_batch]:
            meaning = await self._infer_jargon(term, chat_text)
            if meaning is None:
                continue
            await self._memory.db.record_jargon(
                group_id, term, meaning, time.time(), ls.jargon_max_entries
            )
            saved += 1
            logger.debug("群 %d 黑话入库：%r → %r", group_id, term, meaning)
        if saved:
            logger.info("群 %d 黑话学习：入库 %d 条（候选 %d 条）", group_id, saved, len(terms))

    async def _infer_jargon(self, term: str, context_text: str) -> str | None:
        """双路含义推断：一次带上下文、一次只看词条，两次一致才认为
        「真的理解」并返回入库含义；任何一路不足/不一致返回 None。"""
        ai = self._ai()
        meaning_with_context, no_info = await ai.infer_jargon_with_context(
            term, context_text
        )
        if no_info or not meaning_with_context:
            logger.debug("黑话 %r 带上下文推断信息不足，跳过", term)
            return None
        meaning_alone = await ai.infer_jargon_alone(term)
        if not meaning_alone:
            logger.debug("黑话 %r 仅词条推断无结果，跳过", term)
            return None
        if not await ai.compare_jargon_inference(meaning_with_context, meaning_alone):
            logger.debug("黑话 %r 双路推断不一致，视为未理解，跳过", term)
            return None
        return meaning_with_context[
            : self._settings().learning.jargon_meaning_max_chars
        ]

    # ------------------------------------------------------------ 任务 A：每日印象

    async def _daily_impression_loop(self) -> None:
        # 启动补一次昨天：跨过零点期间进程不在，错过的印象补上（已有则跳过）
        await self._guarded(self.summarize_day(date.today() - timedelta(days=1)))
        while not self._stopping:
            await asyncio.sleep(seconds_until_next_midnight())
            if self._stopping:
                return
            # 零点后立刻总结刚过去的那一天
            await self._guarded(self.summarize_day(date.today() - timedelta(days=1)))

    async def summarize_day(self, day: date) -> int:
        """为白名单内有聊天记录、且当天尚未总结过的群各生成一条群印象。

        返回本次实际生成的群数。单群失败只跳过该群，不影响其他群。
        """
        ls = self._settings().learning
        if not ls.enabled or not ls.impression_enabled:
            return 0
        settings = self._settings()
        db = self._memory.db
        await db.create_tables()
        day_str = day.isoformat()
        start_ts, end_ts = day_bounds(day)
        generated = 0
        for group_id in await db.list_group_ids():
            if settings.profile_for(group_id) is None:
                continue  # 白名单外的群不做中期记忆
            try:
                if await db.has_impression(group_id, day_str):
                    continue
                records = await db.load_day_records(group_id, start_ts, end_ts)
                if not records:
                    continue
                chat_text = learning_chat_text(
                    _clip_records_for_impression(records, ls.impression_text_budget)
                )
                summary = await self._ai().summarize_impression(
                    day_str, chat_text, ls.impression_max_chars
                )
                if not summary:
                    continue
                summary = summary.strip()[: ls.impression_max_chars]
                await db.save_impression(group_id, day_str, summary)
                generated += 1
                logger.info("群 %d %s 群印象已生成（%d 字）", group_id, day_str, len(summary))
                logger.debug("群 %d %s 群印象内容：%s", group_id, day_str, summary)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "群 %d %s 群印象生成失败，跳过本次", group_id, day_str, exc_info=True
                )
        # 只保留最近 impression_days 天所需（再新一些的天），更旧的清理
        try:
            cutoff = (date.today() - timedelta(days=max(ls.impression_days, 1))).isoformat()
            removed = await db.prune_impressions(cutoff)
            if removed:
                logger.debug("清理过期群印象 %d 条（早于 %s）", removed, cutoff)
        except Exception:
            logger.warning("群印象清理失败", exc_info=True)
        return generated

    # ------------------------------------------------------------ 使用侧（注入）

    async def pick_expressions(
        self, group_id: int, limit: int
    ) -> list[tuple[str, str]]:
        """从该群候选表达中按权重（学习次数）随机抽 ≤limit 条，并刷新
        被选中条目的 last_active_time。"""
        if limit <= 0:
            return []
        candidates = await self._memory.db.load_expressions(group_id)
        if not candidates:
            return []
        pool: list[ExpressionEntry] = list(candidates)
        weights = [max(c.weight, 1) for c in pool]
        picked: list[ExpressionEntry] = []
        while pool and len(picked) < limit:
            chosen = self.rng.choices(pool, weights=weights, k=1)[0]
            index = pool.index(chosen)
            pool.pop(index)
            weights.pop(index)
            picked.append(chosen)
        await self._memory.db.touch_expressions([p.id for p in picked], time.time())
        return [(p.situation, p.style) for p in picked]

    async def match_jargons(
        self, group_id: int, context: str, limit: int
    ) -> list[tuple[str, str]]:
        """对当前上下文做黑话机械匹配（中文按包含、西文按词边界），
        命中的取前 limit 条并刷新 last_hit_time。"""
        if limit <= 0 or not context.strip():
            return []
        entries = await self._memory.db.load_jargons(group_id)
        if not entries:
            return []
        hits: list[JargonEntry] = [
            entry for entry in entries if self._term_pattern(entry.term).search(context)
        ]
        hits = hits[:limit]
        if hits:
            await self._memory.db.touch_jargons([h.id for h in hits], time.time())
        return [(h.term, h.meaning) for h in hits]

    def _term_pattern(self, term: str) -> re.Pattern[str]:
        """黑话词条的匹配正则：含中文按子串包含；纯西文/数字按词边界
        （前后不接字母数字，避免 nb 误命中 unbalanced），大小写不敏感。"""
        pattern = self._jargon_patterns.get(term)
        if pattern is None:
            escaped = re.escape(term)
            if re.search(r"[\u4e00-\u9fff]", term):
                pattern = re.compile(escaped)
            else:
                pattern = re.compile(
                    r"(?<![0-9A-Za-z])" + escaped + r"(?![0-9A-Za-z])", re.IGNORECASE
                )
            if len(self._jargon_patterns) > 512:  # 防御性封顶，词条被删后缓存可重学
                self._jargon_patterns.clear()
            self._jargon_patterns[term] = pattern
        return pattern
