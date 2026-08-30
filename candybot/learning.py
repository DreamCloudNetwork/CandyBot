"""后台学习服务：每日群印象（任务 A）、表达学习（任务 B）、黑话学习（任务 C）。

三类学习任务的全部 LLM 调用都发生在后台 asyncio 任务里，绝不阻塞每群
决策队列；任一环节失败只记 warning 日志并跳过本次，不重试、不堆积
（这是「后台任务可以容忍失败」的例外条款——主链路仍按错误完整暴露处理）。

学习成果经 candy.db 持久化（group_impression / expressions / jargons 三表，
表达条目的语义向量另存 expression_embedding 新表），使用侧约定：
- 群印象注入提示词 L2（天内字节级稳定，快照缓存在 GroupRuntime）；
- 表达与黑话注入 L4（每次回复现取现注入，属易变信息）。

表达注入前的选取有两种方式（learning.expression_selection_mode）：
weighted_random（默认）按权重随机抽取；vector 用 embedding 按当前聊天语境
做语义召回（top_k + 相似度下限过滤后仍在存活候选内加权随机）。vector 模式
需配置 models.embedding（未配置启动即报错，见 models.load_settings）；
embedding 的批量计算都在后台进行（学习入库后补算、启动懒补），检索时的
单次查询向量化按 (群, 触发消息) 缓存——运行期 embed 调用失败只记 WARNING
并退回加权随机，这是容错，不是掩盖配置错误。
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, timedelta

from .ai import AIClient
from .database import ExpressionEntry, JargonEntry, pack_vector
from .memory import MemoryManager, seconds_until_next_midnight
from .models import (
    EXPRESSION_SELECTION_VECTOR,
    ChatRecord,
    LearningSettings,
    Settings,
)
from .prompts import learning_chat_text

logger = logging.getLogger(__name__)

# 调度与限流参数（被淘汰缓冲倍数、印象文本预算、每批黑话候选上限、
# 黑话含义入库长度上限）原先写死在本模块，现统一移到 learning 配置段
# （models.LearningSettings），默认值与提取前的字面量一致。

# 每次 embedding API 调用携带的条目数（表达向量批量补算的分片大小）
_EMBED_BATCH_SIZE = 32
# 查询向量缓存的条目上限（FIFO 淘汰）：只为同一条触发消息的多次生成服务，
# 消息一过就再无复用价值，有界即可
_QUERY_VECTOR_CACHE_MAX = 512
# 向量检索查询文本的字符预算（取最近内容的尾部）：作为实例属性暴露
# （LearningService.query_text_budget），便于测试用小预算验证截断行为。
EXPRESSION_QUERY_TEXT_BUDGET = 600


def expression_embed_text(situation: str, style: str) -> str:
    """表达条目的 embedding 文本：与 L4 注入同款句式，
    与查询文本（聊天流 learning_chat_text）语义空间尽量对齐。"""
    return f'当"{situation}"时，可以用"{style}"'


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（纯 Python：表达条目量级只有几百，不值得引入 numpy）。
    维数不一致或任一为零向量时返回 0.0（视为语境无关，交给阈值过滤）。"""
    if not a or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


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
        # ---- 表达向量检索（expression_selection_mode=vector）的进程内状态 ----
        # 每群 {expression_id: 向量} 缓存：首次检索从库装载，此后只随学习
        # 入库/补算刷新（见 _embed_missing），检索不再查库。
        self._expr_vectors: dict[int, dict[int, list[float]]] = {}
        # 查询向量缓存：(群, 触发消息 id, 模型名) → 向量，FIFO 封顶。
        # 缓存键带上模型名：模型热重载变更后同一条消息要按新模型重算。
        self._query_vectors: OrderedDict[tuple[int, int, str], list[float]] = (
            OrderedDict()
        )
        # 当前生效的 embedding 模型名（None=未配置）；变化时全部向量缓存
        # 整体失效——旧模型的向量不能与新查询混用（见 _embedding_model_name）。
        self._embedding_model: str | None = None
        # 启动懒补任务（start 里挂上，stop 里取消）
        self._backfill_task: asyncio.Task[None] | None = None
        # 查询文本的截断预算（取尾部）；测试可替换以验证截断行为。
        self.query_text_budget: int = EXPRESSION_QUERY_TEXT_BUDGET

    # ------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        """启动每日印象循环（幂等）。先补一次昨天：跨过零点停机重启时
        错过的那份印象在启动时补上。同时挂一次表达向量的后台懒补
        （不阻塞启动；embedding 未配置时整个补算静默跳过）。"""
        if self._impression_task is None or self._impression_task.done():
            self._impression_task = asyncio.create_task(
                self._daily_impression_loop(), name="daily-group-impression"
            )
        if self._backfill_task is None or self._backfill_task.done():
            self._backfill_task = asyncio.create_task(
                self._guarded(self._backfill_embeddings()),
                name="expression-embedding-backfill",
            )

    async def stop(self) -> None:
        """取消每日印象循环、在跑的批次任务与向量懒补任务（不等待其完成）。"""
        self._stopping = True
        tasks: list[asyncio.Task[None]] = []
        if self._impression_task is not None:
            tasks.append(self._impression_task)
            self._impression_task = None
        if self._backfill_task is not None:
            tasks.append(self._backfill_task)
            self._backfill_task = None
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
            model = self._embedding_model_name()
            if model is not None:
                # 新入库条目紧接着批量补算向量（每批一次 embed 调用）。
                # 本方法整体已在后台批次任务里，再经 _guarded 包一层：
                # 失败只记 WARNING，不影响这批已成功的入库与后续黑话学习。
                await self._guarded(self._embed_missing(group_id, model))

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
        self,
        group_id: int,
        limit: int,
        recent: Sequence[ChatRecord] = (),
        *,
        trigger: ChatRecord | None = None,
    ) -> list[tuple[str, str]]:
        """从该群候选表达中抽 ≤limit 条注入用，并刷新被选中条目的
        last_active_time。

        weighted_random 模式（默认，行为与引入语义检索之前逐字节一致）：
        按权重（学习次数）随机抽取，recent/trigger 不参与任何逻辑。
        vector 模式：先用当前聊天语境（recent + trigger）做 embedding 语义
        召回，过滤相似度低于 expression_min_similarity 的条目后，在存活候选
        内仍按现有加权随机抽取（保留随机性，防止每次都抽到同一批）；一个
        候选都不过时返回空——想不起贴切的说法就不硬用。召回失败（未配置、
        无查询文本、embed 调用异常）记 WARNING 并退回本次加权随机——
        运行期容错，不阻塞决策队列；配置错误另在启动校验时直接报错。
        """
        if limit <= 0:
            return []
        ls = self._settings().learning
        candidates = await self._memory.db.load_expressions(group_id)
        if not candidates:
            return []
        sims: dict[int, float] = {}
        if ls.expression_selection_mode == EXPRESSION_SELECTION_VECTOR:
            recall = await self._vector_recall(group_id, candidates, recent, trigger, ls)
            if recall is not None:
                if not recall:
                    logger.debug(
                        "群 %d 表达向量召回：全部候选低于相似度下限 %.2f，本次不注入",
                        group_id,
                        ls.expression_min_similarity,
                    )
                    return []
                sims = recall
                candidates = [c for c in candidates if c.id in recall]
        picked = self._weighted_pick(candidates, limit)
        await self._memory.db.touch_expressions([p.id for p in picked], time.time())
        if sims:
            logger.debug(
                "群 %d 表达 L4 注入（向量召回，含相似度）：%s",
                group_id,
                "、".join(
                    f'{p.situation}→{p.style}={sims[p.id]:.3f}' for p in picked
                ),
            )
        return [(p.situation, p.style) for p in picked]

    def _weighted_pick(
        self, pool_src: Sequence[ExpressionEntry], limit: int
    ) -> list[ExpressionEntry]:
        """按权重随机不放回抽 ≤limit 条（pick_expressions 的既有核心，
        原样抽出共用；改动它会同步改变两种模式的抽取轨迹）。"""
        pool: list[ExpressionEntry] = list(pool_src)
        weights = [max(c.weight, 1) for c in pool]
        picked: list[ExpressionEntry] = []
        while pool and len(picked) < limit:
            chosen = self.rng.choices(pool, weights=weights, k=1)[0]
            index = pool.index(chosen)
            pool.pop(index)
            weights.pop(index)
            picked.append(chosen)
        return picked

    # ------------------------------------------------ 表达向量检索（embedding 基建）

    def _embedding_model_name(self) -> str | None:
        """当前配置的 embedding 模型名（None=未配置）。与上次生效值不同时
        把每群向量缓存与查询向量缓存整体作废——旧模型的向量不能与新查询
        放进同一个相似度空间。"""
        embedding = self._settings().models.embedding
        model = embedding.model if embedding is not None else None
        if model != self._embedding_model:
            self._expr_vectors.clear()
            self._query_vectors.clear()
            self._embedding_model = model
        return model

    async def _backfill_embeddings(self) -> None:
        """启动懒补：跨全部群给缺向量/模型过期的表达条目分批补算。
        embedding 未配置时整体静默跳过（默认配置没有该角色，不是错误）。"""
        model = self._embedding_model_name()
        if model is None:
            return
        await self._memory.db.create_tables()  # 新表由 create_all 补建（幂等）
        await self._embed_missing(None, model)

    async def _embed_missing(self, group_id: int | None, model: str) -> int:
        """给缺向量（或向量出自旧模型）的表达条目分批计算 embedding 并落库，
        返回补算条数。每批一次 embed 调用；调用失败原样上抛，由
        _guarded 记 WARNING。已建好的每群缓存随手刷新。"""
        missing = await self._memory.db.list_expressions_missing_embedding(
            model, group_id
        )
        if not missing:
            return 0
        done = 0
        for start in range(0, len(missing), _EMBED_BATCH_SIZE):
            if self._stopping:
                break
            chunk = missing[start : start + _EMBED_BATCH_SIZE]
            texts = [
                expression_embed_text(situation, style)
                for _id, _gid, situation, style in chunk
            ]
            vectors = await self._ai().embed(texts)
            if len(vectors) != len(chunk):
                raise RuntimeError(
                    f"embedding 返回 {len(vectors)} 条向量，与本次发送的 {len(chunk)} 条文本不符"
                )
            entries: list[tuple[int, bytes, int, str]] = []
            for (expression_id, gid, _situation, _style), vector in zip(chunk, vectors):
                entries.append((expression_id, pack_vector(vector), len(vector), model))
                cache = self._expr_vectors.get(gid)
                if cache is not None:
                    cache[expression_id] = vector  # 群缓存已建立时同步刷新
            await self._memory.db.upsert_expression_embeddings(entries)
            done += len(entries)
        logger.debug(
            "表达向量补算 %d 条（群 %s，模型 %s）",
            done,
            group_id if group_id is not None else "全部",
            model,
        )
        return done

    async def _group_vectors(self, group_id: int, model: str) -> dict[int, list[float]]:
        """某群 {expression_id: 向量} 的内存缓存：首次检索从库装载一次，
        之后只随学习入库/补算刷新。"""
        vectors = self._expr_vectors.get(group_id)
        if vectors is None:
            vectors = await self._memory.db.load_expression_vectors(group_id, model)
            self._expr_vectors[group_id] = vectors
        return vectors

    async def _query_embedding(
        self, group_id: int, message_id: int | None, model: str, text: str
    ) -> list[float] | None:
        """查询文本的向量：同一条触发消息的多次生成（新鲜度重生成、观望
        重评后的再抽取）只请求一次 embed；message_id 为 None 时不缓存。
        返回 None 表示端点给了空向量。异常原样上抛，由调用方降级。"""
        key: tuple[int, int, str] | None = (
            (group_id, message_id, model) if message_id is not None else None
        )
        if key is not None:
            cached = self._query_vectors.get(key)
            if cached is not None:
                self._query_vectors.move_to_end(key)
                return cached
        vectors = await self._ai().embed([text])
        if not vectors:
            return None
        query_vector = vectors[0]
        if key is not None:
            self._query_vectors[key] = query_vector
            while len(self._query_vectors) > _QUERY_VECTOR_CACHE_MAX:
                self._query_vectors.popitem(last=False)
        return query_vector

    async def _vector_recall(
        self,
        group_id: int,
        candidates: Sequence[ExpressionEntry],
        recent: Sequence[ChatRecord],
        trigger: ChatRecord | None,
        ls: LearningSettings,
    ) -> dict[int, float] | None:
        """按当前聊天语境做语义召回：返回过滤后存活的
        {expression_id: 相似度}；None 表示本次检索不可用（未配置、无查询
        文本、embed 调用失败），调用方退回加权随机。"""
        model = self._embedding_model_name()
        if model is None:
            logger.warning(
                "群 %d 表达选取模式为 vector 但未配置 models.embedding，本次退回加权随机",
                group_id,
            )
            return None
        records = list(recent)
        if trigger is not None and all(r.message_id != trigger.message_id for r in records):
            records.append(trigger)  # 触发消息理应已在上下文尾部，防御性补上
        text = learning_chat_text(records)
        if len(text) > self.query_text_budget:
            text = text[-self.query_text_budget :]  # 保留最近内容的尾部
        if not text.strip():
            logger.warning("群 %d 表达向量检索：无可用查询文本，本次退回加权随机", group_id)
            return None
        message_id = (
            trigger.message_id
            if trigger is not None
            else (records[-1].message_id if records else None)
        )
        try:
            query_vector = await self._query_embedding(group_id, message_id, model, text)
        except Exception:
            logger.warning(
                "群 %d 表达向量检索：embedding 调用失败，本次退回加权随机",
                group_id,
                exc_info=True,
            )
            return None
        if not query_vector:
            logger.warning(
                "群 %d 表达向量检索：embedding 返回空向量，本次退回加权随机", group_id
            )
            return None
        vectors = await self._group_vectors(group_id, model)
        scored: list[tuple[float, ExpressionEntry]] = []
        for entry in candidates:
            vector = vectors.get(entry.id)
            if vector is None:
                continue  # 缺向量的条目本次不参与召回（补算任务会跟进）
            scored.append((_cosine_similarity(query_vector, vector), entry))
        scored.sort(key=lambda item: -item[0])
        top = scored[: ls.expression_vector_top_k]
        logger.debug(
            "群 %d 表达向量召回 top_%d（阈值 %.2f）：%s",
            group_id,
            ls.expression_vector_top_k,
            ls.expression_min_similarity,
            "、".join(f'{entry.situation}→{entry.style}={sim:.3f}' for sim, entry in top)
            or "（无可比对向量）",
        )
        return {
            entry.id: sim
            for sim, entry in top
            if sim >= ls.expression_min_similarity
        }

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
