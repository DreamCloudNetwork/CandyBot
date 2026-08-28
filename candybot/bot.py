"""核心编排：事件过滤链、决策、回复与发送。

每群一个串行 asyncio 队列保证决策顺序。@ 必答与「对方正在和我说话」的
消息（judge 判定为延续与本机器人的对话）不受冷却和护栏限制；其余主动
插话需依次通过冷却、发言间隔、热闹静默三道护栏及判定门槛。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import date

import aiohttp

from .ai import AIClient, ImageOp, ReplyDraft, split_image_ops
from .dedup import MessageDedup
from .events_server import EventsServer
from .memory import MemoryManager
from .models import (
    ChatRecord,
    Decision,
    GroupProfile,
    NormalizedMessage,
    Settings,
    load_settings,
)
from .normalize import normalize_group_message
from .prompts import (
    nickname_list_from_history,
    now_text as fmt_now_text,
    record_to_turn,
    runtime_system_prompt,
    static_system_prompt,
)
from .snowluma import SnowlumaClient

logger = logging.getLogger(__name__)


class GroupRuntime:
    """单个群的运行时状态。"""

    def __init__(self) -> None:
        self.last_proactive_ts: float = 0.0
        # 反插嘴护栏的记账：上次自己发言以来的他人消息数（含当前待判定消息），
        # 初值为大数表示「从未发言」不被间隔约束，发送后归零重新累积；
        # 近一分钟的群消息时间戳（含 @ 必答的消息，反映真实热闹程度）。
        self.msgs_since_reply: int = 10**9
        self.recent_msg_times: deque[float] = deque()
        self._static_cache_key: tuple[int, str] | None = None
        self._static_cache: str = ""

    def static_system(self, kind: str, persona: str, *, via_tool: bool = True) -> str:
        """L1 静态层按 (kind, persona, via_tool) 缓存；persona 改热重载时自动重建。

        via_tool 决定 reply 守则里输出契约的措辞，必须与该角色请求是否
        携带 tools 参数一致（取自 AIClient 的实时角色状态）。
        """
        key = (hash(kind), persona, via_tool)
        if key != self._static_cache_key:
            self._static_cache = static_system_prompt(persona, kind, via_tool=via_tool)
            self._static_cache_key = key
        return self._static_cache


class CandyBot:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._dedup = MessageDedup()
        # 内存热缓存容量取全局最大上下文需求的 2 倍（仍有界）；文本历史
        # 在 candy.db 里全量保留，图片按 storage.image_retention_days 回收
        max_context = max(
            [p.context_size for p in settings.groups.values()]
            + [settings.groups_default.context_size]
        )
        self._memory = MemoryManager(
            settings.bot.data_dir,
            default_capacity=max(max_context * 2, 64),
            image_retention_days=settings.storage.image_retention_days,
        )
        self._http = aiohttp.ClientSession()
        self._ai = AIClient(
            models=settings.models,
            generation=settings.generation,
            multimodal_mode=settings.multimodal.mode,
        )
        self._snowluma = SnowlumaClient(settings.snowluma)
        self._server = EventsServer(
            self._on_event,
            host=settings.bot.listen_host,
            port=settings.bot.listen_port,
            secret=settings.bot.event_secret,
        )
        self._runtimes: dict[int, GroupRuntime] = defaultdict(GroupRuntime)
        self._group_queues: dict[int, asyncio.Queue[NormalizedMessage]] = {}
        self._queue_workers: dict[int, asyncio.Task[None]] = {}
        self._daily_date: date = date.today()
        self._daily_replies: int = 0
        self._last_self_message_id: int = 0
        self._stopping = False

    # ------------------------------------------------------------ 生命周期

    @property
    def log_level(self) -> str:
        """配置的日志级别名，main.py 在启动时应用。"""
        return self._settings.bot.log_level

    async def start(self) -> None:
        await self._memory.start()  # 建表 + 每日图片回收循环
        await self._snowluma.start()
        await self._snowluma.probe()
        login = await self._snowluma.query_login_info()
        if login is not None:
            reported = str(login.get("user_id", ""))
            if reported and reported != str(self._settings.bot.self_qq):
                logger.warning(
                    "SnowLuma 登录账号是 %s，而 bot.self_qq 配置为 %s；@ 识别可能失效",
                    reported,
                    self._settings.bot.self_qq,
                )
        else:
            logger.info("未能查询登录账号（read 权限或 action 缺失），跳过自检")
        await self._server.start()

    async def stop(self) -> None:
        self._stopping = True
        for task in list(self._queue_workers.values()):
            task.cancel()
        await self._server.stop()
        await self._http.close()
        await self._snowluma.stop()
        await self._memory.close()

    # ------------------------------------------------------------ 事件入口

    async def _on_event(self, event: dict) -> None:
        if self._stopping:
            return
        post_type = event.get("post_type")
        if post_type == "meta_event":
            return  # 心跳/生命周期
        if post_type == "notice":
            await self._on_notice(event)
            return
        if post_type != "message":
            return

        try:
            mid = int(event.get("message_id"))
        except (TypeError, ValueError):
            return
        if self._dedup.check_and_mark(mid):
            return

        try:
            assess = getattr(self._ai, "assess_image", None)
            normalized = await normalize_group_message(
                event,
                self_qq=self._settings.bot.self_qq,
                multimodal=self._settings.multimodal,
                # 传协程回调，normalize 解析 reply 段时会 await 它
                find_by_message_id=lambda ref_id: self._find_record(event, ref_id),
                http_session=self._http,
                describe_image=(
                    self._ai.describe_image
                    if self._settings.multimodal.mode == "describe"
                    else None
                ),
                assess_image=(
                    assess if self._settings.multimodal.mode == "direct" else None
                ),
            )
        except Exception:
            logger.exception("归一化事件失败，已忽略：%r", event.get("message_id"))
            return
        if normalized is None:
            return

        group_id = normalized.record.group_id
        if self._settings.profile_for(group_id) is None:
            logger.debug("群 %d 不在白名单，忽略", group_id)
            return

        memory = await self._memory.get(group_id)
        await memory.append(normalized.record)
        logger.debug("收到消息 %s : %s",group_id,normalized)
        await self._enqueue(group_id, normalized)

    async def _find_record(self, event: dict, ref_id: int) -> ChatRecord | None:
        try:
            group_id = int(event.get("group_id"))
        except (TypeError, ValueError):
            return None
        memory = await self._memory.get(group_id)
        return await memory.find_by_message_id(ref_id)

    async def _on_notice(self, event: dict) -> None:
        """通知类事件：目前只处理群撤回——删除本地记录的对应消息。"""
        if event.get("notice_type") != "group_recall":
            return
        try:
            group_id = int(event["group_id"])
            message_id = int(event["message_id"])
        except (KeyError, TypeError, ValueError):
            logger.debug("缺少字段的撤回事件，忽略：%r", event)
            return
        if self._settings.profile_for(group_id) is None:
            return
        memory = await self._memory.get(group_id)
        if await memory.remove(message_id):
            logger.info("群 %d：消息 %d 已撤回，已删除本地记录", group_id, message_id)
        else:
            logger.debug(
                "群 %d：撤回的消息 %d 不在本地记忆中", group_id, message_id
            )

    # ------------------------------------------------------------ 群内串行队列

    async def _enqueue(self, group_id: int, msg: NormalizedMessage) -> None:
        queue = self._group_queues.get(group_id)
        if queue is None:
            queue: asyncio.Queue[NormalizedMessage] = asyncio.Queue()
            self._group_queues[group_id] = queue
            self._queue_workers[group_id] = asyncio.create_task(
                self._group_worker(group_id, queue)
            )
        await queue.put(msg)

    async def _group_worker(
        self, group_id: int, queue: asyncio.Queue[NormalizedMessage]
    ) -> None:
        while not self._stopping:
            try:
                msg = await queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._decide_and_reply(group_id, msg)
            except Exception:
                logger.exception(
                    "处理群 %d 消息 %d 时出错", group_id, msg.record.message_id
                )

    # ------------------------------------------------------------ 决策与回复

    async def _decide_and_reply(self, group_id: int, msg: NormalizedMessage) -> None:
        profile = self._settings.profile_for(group_id)
        assert profile is not None
        runtime = self._runtimes[group_id]

        decision = await self._make_decision(group_id, msg, profile, runtime)
        if not decision.should_reply:
            return

        reply_text = await self._compose_reply(group_id, msg, profile, runtime, decision)
        if not reply_text:
            logger.info("群 %d 回复生成为空，放弃发送", group_id)
            return
        if not decision.forced and not self._consume_daily_quota():
            logger.info("达到 global_daily_limit，本次主动回复被拦截")
            return
        try:
            await self._send_with_retry(group_id, reply_text)
        except Exception as exc:
            logger.error("群 %d 发送失败：%s", group_id, exc)
            return

        # 只有主动插话消耗并刷新冷却；@ 必答和对话延续都不该把正在进行的
        # 交流掐断。无论哪种触发，间隔都从本条发言后重新累计。
        if not decision.forced and not decision.engaged:
            runtime.last_proactive_ts = time.time()
        runtime.msgs_since_reply = 0

        sent_record = ChatRecord(
            message_id=self._next_self_message_id(),
            group_id=group_id,
            user_id=self._settings.bot.self_qq,
            nickname="糖糖",
            text=reply_text,
            ts=time.time(),
            is_self=True,
        )
        memory = await self._memory.get(group_id)
        await memory.append(sent_record)

    async def _compose_reply(
        self,
        group_id: int,
        msg: NormalizedMessage,
        profile: GroupProfile,
        runtime: GroupRuntime,
        decision: Decision,
    ) -> str | None:
        """生成回复并处理其中的图片生命周期操作。

        首稿通过 send_reply 工具参数携带 drop/recall 操作：立即把操作落到
        记忆；若发生过成功的召回（模型想重新查看某张旧图），重建上下文后
        再生成一次，让召回的原图进入本轮对话。二稿不再重试，保证每条消息
        至多两次生成调用。作为最终防线，正文中任何形似 <drop_img>/<recall_img>
        的残留标记仍会被剥除收编，绝不发进群里。
        """
        memory = await self._memory.get(group_id)
        for attempt in range(2):
            recent = memory.tail(profile.context_size)
            # 跟随 reply 角色实时的工具调用状态：降级后 L1 守则换回纯文本措辞
            static_system = runtime.static_system(
                "reply", profile.persona, via_tool=self._ai.reply_tool_use
            )
            nicknames = nickname_list_from_history(
                [record_to_turn(r) for r in recent[:-1]]
            )
            runtime_system = runtime_system_prompt(
                group_id, date.today().isoformat(), nicknames
            )
            draft = await self._generate_with_retry(
                static_system,
                runtime_system,
                recent,
                msg.record,
                fmt_now_text(),
                forced=decision.forced,
                engaged=decision.engaged,
                score=decision.score,
                reason=decision.reason,
            )
            if draft is None or not draft.text:
                return None
            text, tag_ops = split_image_ops(draft.text)
            ops: list[ImageOp] = [*draft.ops, *tag_ops]
            changed = await self._apply_image_ops(group_id, memory, ops)
            recalled = any(op.action == "recall_img" for op in ops)
            if attempt == 0 and changed and recalled:
                logger.info("群 %d 模型召回历史图片，基于新上下文重写回复", group_id)
                continue
            return text
        return None

    async def _apply_image_ops(
        self, group_id: int, memory, ops: list[ImageOp]
    ) -> bool:
        """把回复里的图片降级/召回操作写入记忆；非 direct 模式一律忽略。"""
        if not ops or self._settings.multimodal.mode != "direct":
            return False
        changed = False
        for op in ops:
            direction = "recall" if op.action == "recall_img" else "drop"
            try:
                applied = await memory.transition_images(op.message_id, direction)
            except Exception:
                logger.exception("群 %d 图片状态切换失败：%r", group_id, op)
                continue
            if applied:
                changed = True
                logger.info(
                    "群 %d 消息 %d 的图片已按模型指令%s",
                    group_id,
                    op.message_id,
                    "召回为原图展示" if direction == "recall" else "收起（转为总结/占位符）",
                )
        return changed

    async def _make_decision(
        self, group_id: int, msg: NormalizedMessage, profile: GroupProfile, runtime: GroupRuntime
    ) -> Decision:
        # 结构性护栏的记账先于一切判断：即使消息最终不触发回复，
        # 也要计入间隔与热闹统计
        runtime.msgs_since_reply += 1
        now = time.time()
        window = runtime.recent_msg_times
        window.append(now)
        while window and window[0] < now - 60.0:
            window.popleft()

        if msg.mentioned_me:
            return Decision(should_reply=True, forced=True)

        # 每条普通消息都过一遍 judge：除了打分，还要识别「这条消息是否在对
        # 我说」——正和我聊天的场景里，任何时间窗口都不应该把对话掐断。
        memory = await self._memory.get(group_id)
        recent = memory.tail(profile.context_size)
        static_system = runtime.static_system("judge", profile.persona)
        nicknames = nickname_list_from_history([record_to_turn(r) for r in recent[:-1]])
        runtime_system = runtime_system_prompt(
            group_id, date.today().isoformat(), nicknames
        )
        try:
            verdict = await self._ai.judge_interest(
                static_system,
                runtime_system,
                recent,
                msg.record,
                fmt_now_text(),
                threshold=profile.proactivity_threshold,
            )
        except Exception as exc:
            logger.warning("judge 调用失败，按不发言处理：%s", exc)
            return Decision(should_reply=False)
        logger.info(
            "群 %d 回复判定 %d/阈值 %d%s（%s）：%s",
            group_id,
            verdict.score,
            profile.proactivity_threshold,
            "[与我对话]" if verdict.to_me else "",
            msg.record.nickname,
            verdict.reason,
        )

        # 对方在延续与我的对话 → 这是接话而不是插话：绕过全部护栏放行，
        # 且不刷新主动冷却，否则下一句对话又会被掐断
        if verdict.to_me:
            return Decision(
                should_reply=True,
                engaged=True,
                score=verdict.score,
                reason=verdict.reason,
            )

        elapsed = time.time() - runtime.last_proactive_ts
        if runtime.last_proactive_ts > 0 and elapsed < profile.cooldown_seconds:
            logger.debug(
                "群 %d 冷却中（剩 %.0fs），跳过判断", group_id, profile.cooldown_seconds - elapsed
            )
            return Decision(should_reply=False, score=verdict.score, reason=verdict.reason)

        # 护栏一：刚主动发过言，需攒够 min_gap_messages 条他人消息后再评估
        if (
            profile.min_gap_messages > 0
            and runtime.msgs_since_reply <= profile.min_gap_messages
        ):
            logger.debug(
                "群 %d 距上次发言仅 %d 条他人消息（要求超过 %d 条），跳过判断",
                group_id,
                runtime.msgs_since_reply,
                profile.min_gap_messages,
            )
            return Decision(should_reply=False, score=verdict.score, reason=verdict.reason)

        # 护栏二：近一分钟消息量达到阈值说明群里正热闹（多人在接龙），
        # 此时插话最容易被嫌弃，保持安静
        if profile.busy_rate_per_min > 0 and len(window) >= profile.busy_rate_per_min:
            logger.debug(
                "群 %d 近 60 秒已有 %d 条消息（≥%d），热闹期保持安静",
                group_id,
                len(window),
                profile.busy_rate_per_min,
            )
            return Decision(should_reply=False, score=verdict.score, reason=verdict.reason)

        # 首评时模型不知道本群门槛，可能把「自己有点想插话」高估到门槛下方
        # 却没过线。这类分数不直接采信：把门槛告知 judge 请其复核，确信值得
        # 开口才维持高分区，否则如实下调（复评的 to_me 不再单独放行——首评已
        # 认定并非在与我对话，即使翻转也按普通插话处理）。复核开关与触发下限
        # 由 generation.recheck_enabled / recheck_min_score 配置。
        gen = self._settings.generation
        if (
            gen.recheck_enabled
            and gen.recheck_min_score < verdict.score < profile.proactivity_threshold
        ):
            try:
                rechecked = await self._ai.judge_interest(
                    static_system,
                    runtime_system,
                    recent,
                    msg.record,
                    fmt_now_text(),
                    threshold=profile.proactivity_threshold,
                    prev_verdict=verdict,
                    min_score=gen.recheck_min_score,
                )
            except Exception as exc:
                logger.warning("群 %d judge 复核失败，按首评不达标处理：%s", group_id, exc)
            else:
                logger.info(
                    "群 %d 回复复核 首评 %d → 复评 %d/阈值 %d%s：%s",
                    group_id,
                    verdict.score,
                    rechecked.score,
                    profile.proactivity_threshold,
                    "[与我对话]" if rechecked.to_me else "",
                    rechecked.reason,
                )
                if rechecked.score >= profile.proactivity_threshold:
                    return Decision(
                        should_reply=True,
                        score=rechecked.score,
                        reason=rechecked.reason,
                    )
                return Decision(
                    should_reply=False,
                    score=rechecked.score,
                    reason=rechecked.reason,
                )

        if verdict.score >= profile.proactivity_threshold:
            return Decision(
                should_reply=True, score=verdict.score, reason=verdict.reason
            )
        return Decision(should_reply=False, score=verdict.score, reason=verdict.reason)

    # ------------------------------------------------------------ 重试包装

    async def _generate_with_retry(self, *args, **kwargs) -> ReplyDraft | None:
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await self._ai.generate_reply(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                logger.warning("生成回复第 %d 次失败：%s", attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2
        logger.error("生成回复最终失败：%s", last_exc)
        return None

    async def _send_with_retry(self, group_id: int, text: str) -> None:
        delay = 1.5
        for attempt in range(3):
            try:
                await self._snowluma.send_group_msg(group_id, text)
                return
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning("发送第 %d 次失败：%s", attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2

    def _next_self_message_id(self) -> int:
        """自发言的合成负 id：绝不与真实消息的正 id 冲突，且进程内严格
        递减——时钟回拨时退化为减一序列，避免撞 (group_id, message_id)
        唯一键导致已发送的回复进不了历史。"""
        candidate = -time.time_ns()
        if candidate >= self._last_self_message_id:
            candidate = self._last_self_message_id - 1
        self._last_self_message_id = candidate
        return candidate

    def _consume_daily_quota(self) -> bool:
        today = date.today()
        if today != self._daily_date:
            self._daily_date = today
            self._daily_replies = 0
        limit = self._settings.rate_limit.global_daily_limit
        if limit is not None and self._daily_replies >= limit:
            return False
        self._daily_replies += 1
        return True


def build_bot(config_file: str = "config.json5") -> CandyBot:
    """从项目根目录 config.py 的 ConfigClass 构造 CandyBot。"""
    from config import Config  # 用户已有的 JSON5 配置单例

    Config._config_file = config_file
    Config.load_config()
    return CandyBot(load_settings(Config))
