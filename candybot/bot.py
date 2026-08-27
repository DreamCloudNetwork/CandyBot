"""核心编排：事件过滤链、决策、回复与发送。

每群一个串行 asyncio 队列保证决策顺序；@ 必答不受冷却与每日限额约束之外
的判断环节，直接进入回复流程。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import date

import aiohttp

from .ai import AIClient
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
        self._static_cache_key: tuple[int, str] | None = None
        self._static_cache: str = ""

    def static_system(self, kind: str, persona: str) -> str:
        """L1 静态层按 (kind, persona) 缓存；persona 改热重载时自动重建。"""
        key = (hash(kind), persona)
        if key != self._static_cache_key:
            self._static_cache = static_system_prompt(persona, kind)
            self._static_cache_key = key
        return self._static_cache


class CandyBot:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._dedup = MessageDedup()
        # 记忆文件容量取全局最大上下文需求的 2 倍（服务后续调参，仍有界）
        max_context = max(
            [p.context_size for p in settings.groups.values()]
            + [settings.groups_default.context_size]
        )
        self._memory = MemoryManager(
            settings.bot.data_dir, default_capacity=max(max_context * 2, 64)
        )
        self._http = aiohttp.ClientSession()
        self._ai = AIClient(
            base_url=settings.ai_backend.base_url,
            api_key=settings.ai_backend.api_key,
            judge_model=settings.models.judge,
            reply_model=settings.models.reply,
            vision_model=settings.models.vision,
            generation=settings.generation,
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
        self._stopping = False

    # ------------------------------------------------------------ 生命周期

    @property
    def log_level(self) -> str:
        """配置的日志级别名，main.py 在启动时应用。"""
        return self._settings.bot.log_level

    async def start(self) -> None:
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

    # ------------------------------------------------------------ 事件入口

    async def _on_event(self, event: dict) -> None:
        if self._stopping:
            return
        post_type = event.get("post_type")
        if post_type == "meta_event":
            return  # 心跳/生命周期
        if post_type != "message":
            return

        try:
            mid = int(event.get("message_id"))
        except (TypeError, ValueError):
            return
        if self._dedup.check_and_mark(mid):
            return

        try:
            normalized = await normalize_group_message(
                event,
                self_qq=self._settings.bot.self_qq,
                multimodal=self._settings.multimodal,
                find_by_message_id=lambda ref_id: self._find_record(event, ref_id),
                http_session=self._http,
                describe_image=(
                    self._ai.describe_image
                    if self._settings.multimodal.mode == "describe"
                    else None
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

        memory = self._memory.get(group_id)
        memory.append(normalized.record)
        logger.debug("收到消息 %s : %s",group_id,normalized)
        await self._enqueue(group_id, normalized)

    def _find_record(self, event: dict, ref_id: int) -> ChatRecord | None:
        try:
            group_id = int(event.get("group_id"))
        except (TypeError, ValueError):
            return None
        return self._memory.get(group_id).find_by_message_id(ref_id)

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

        memory = self._memory.get(group_id)
        recent = memory.tail(profile.context_size)
        static_system = runtime.static_system("reply", profile.persona)
        nicknames = nickname_list_from_history([record_to_turn(r) for r in recent[:-1]])
        runtime_system = runtime_system_prompt(
            group_id, date.today().isoformat(), nicknames
        )

        reply_text = await self._generate_with_retry(
            static_system,
            runtime_system,
            recent,
            msg.record,
            fmt_now_text(),
            forced=decision.forced,
            score=decision.score,
            reason=decision.reason,
        )
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

        if not decision.forced:
            runtime.last_proactive_ts = time.time()  # 只有主动发言消耗冷却

        sent_record = ChatRecord(
            message_id=-time.time_ns(),  # 合成负 id，绝不与他人冲突
            group_id=group_id,
            user_id=self._settings.bot.self_qq,
            nickname="糖糖",
            text=reply_text,
            ts=time.time(),
            is_self=True,
        )
        memory.append(sent_record)

    async def _make_decision(
        self, group_id: int, msg: NormalizedMessage, profile: GroupProfile, runtime: GroupRuntime
    ) -> Decision:
        if msg.mentioned_me:
            return Decision(should_reply=True, forced=True)
        elapsed = time.time() - runtime.last_proactive_ts
        if runtime.last_proactive_ts > 0 and elapsed < profile.cooldown_seconds:
            logger.debug(
                "群 %d 冷却中（剩 %.0fs），跳过判断", group_id, profile.cooldown_seconds - elapsed
            )
            return Decision(should_reply=False)

        recent = self._memory.get(group_id).tail(profile.context_size)
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
            "群 %d 回复判定 %d/阈值 %d（%s）：%s",
            group_id,
            verdict.score,
            profile.proactivity_threshold,
            msg.record.nickname,
            verdict.reason,
        )
        if verdict.score >= profile.proactivity_threshold:
            return Decision(
                should_reply=True, score=verdict.score, reason=verdict.reason
            )
        return Decision(should_reply=False, score=verdict.score, reason=verdict.reason)

    # ------------------------------------------------------------ 重试包装

    async def _generate_with_retry(self, *args, **kwargs) -> str | None:
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


def build_bot(config_file: str = "config.json") -> CandyBot:
    """从项目根目录 config.py 的 ConfigClass 构造 CandyBot。"""
    from config import Config  # 用户已有的 JSON 配置单例

    Config._config_file = config_file
    Config.load_config()
    return CandyBot(load_settings(Config))
