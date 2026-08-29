"""核心编排：事件过滤链、决策、回复与发送。

每群一个串行 asyncio 队列保证决策顺序。@ 必答与「对方正在和我说话」的
消息（judge 判定为延续与本机器人的对话）不受冷却和护栏限制；其余主动
插话需依次通过冷却、发言间隔、热闹静默三道护栏及判定门槛。

连发期间（生成中或打字延迟中）一旦有他人新消息进入记忆，下一条发出前
会先让 reply 模型对剩余腹稿重想一次：可以放弃、改写或照原样继续，防止
别人已经插话、AI 却把打好的字一条条硬发完。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable
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
from .postprocess import (
    ProcessedReply,
    ensure_indexes,
    estimate_typing_time,
    process_reply,
)
from .prompts import (
    nickname_list_from_history,
    now_text as fmt_now_text,
    record_to_turn,
    runtime_system_prompt,
    static_system_prompt,
)
from .snowluma import SnowlumaClient

logger = logging.getLogger(__name__)

# 一轮连发最多几次「被打断后重想」：每次重想是一回额外的 reply 模型调用，
# 预算用尽后剩下的腹稿按原计划发完，防止病态刷屏把发送环节变成无限往返。
_MAX_RECONSIDER_PER_BURST = 2


def _verbatim_match(text: str, pending: list[str]) -> bool:
    """重想输出是否与腹稿一字不差（忽略行间空白差异）。

    模型的复读常带尾部空白或空行，也可能沿用「一行一条」之外的换行习惯；
    不规范化就会误判成改写，让已经掷好的错别字与更正被 process_reply
    重新加工一遍。
    """

    def norm(s: str) -> str:
        return "\n".join(line.strip() for line in s.splitlines() if line.strip())

    return norm(text) == norm("\n".join(pending))


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
    def __init__(
        self,
        settings: Settings,
        settings_loader: Callable[[], Settings] | None = None,
    ):
        self._settings = settings
        # 热重载时重建 Settings 快照的来源（由 build_bot 传入
        # lambda: load_settings(Config)）；不传则该实例不支持热重载
        # （如测试里手工注入 settings）。
        self._settings_loader = settings_loader
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
        # 输出层后处理的随机源（拆条兜底/错别字/更正掷点）；与 ai.py 的
        # 约定一致走加密安全随机，测试可替换为固定种子的 random.Random。
        self._pp_rng: random.Random = random.SystemRandom()

    # ------------------------------------------------------------ 生命周期

    @property
    def log_level(self) -> str:
        """配置的日志级别名，main.py 在启动时应用。"""
        return self._settings.bot.log_level

    async def start(self) -> None:
        await self._memory.start()  # 建表 + 每日图片回收循环
        pp = self._settings.response_post_process
        if pp.enabled and (pp.typo_error_rate > 0 or pp.typo_word_replace_rate > 0):
            # 拼音反查表构建一次约 0.6 秒：在后台线程预热，避免首条带错字
            # 的回复把事件循环卡住（所有群的队列和事件接收一起冻半秒）
            await asyncio.to_thread(ensure_indexes)
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
        workers = list(self._queue_workers.values())
        for task in workers:
            task.cancel()
        # 必须等 worker 真正退出再关连接池：连发写回内存里，被取消的 worker
        # 可能正卡在一次数据库写入上（sqlalchemy aiosqlite 用 shield 保护
        # 在途操作），此时并行 dispose 会让双方互相等待，进程退出被挂死。
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        await self._server.stop()
        await self._http.close()
        await self._snowluma.stop()
        await self._memory.close()

    # ------------------------------------------------------------ 配置热重载

    def reload_settings(self) -> bool:
        """重新解析 config.json5 并原子替换运行时配置（热重载入口）。

        由 main.py 经 loop.call_soon_threadsafe 调度到事件循环线程调用，与
        消息处理天然串行，无需额外加锁。启动时已烘进监听/子进程会话的字段
        无法就地更换、改动仍需重启：bot.listen_host / listen_port /
        event_secret（aiohttp 监听与签名校验已绑定）、bot.data_dir、
        snowluma.*（MCP 子进程会话）；热缓存容量也在启动时按全局最大
        context_size 定死（新配置超出时记警告）。其余（白名单、人设、护栏
        阈值、模型与生成参数、多模态、输出后处理、限速、图片保留天数）
        即时生效。

        解析失败（典型场景：配置写坏）完整记日志但不拖垮服务：沿用旧配置，
        下一次保存自动重试。返回是否实际完成了替换。
        """
        if self._settings_loader is None:
            logger.warning("未携带 settings_loader 的 CandyBot 实例，配置热重载不可用")
            return False
        try:
            new_settings = self._settings_loader()
        except Exception:
            logger.exception("配置重载失败，继续使用旧配置")
            return False
        self._settings = new_settings
        # 模型端点、生成参数与多模态模式都烘在 AIClient 构造函数里，须随
        # 配置重建；进行中的请求持有的是旧实例引用，会正常完成。工具协议
        # 降级状态随之重置，最多多一次降级往返，不影响正确性。
        self._ai = AIClient(
            models=new_settings.models,
            generation=new_settings.generation,
            multimodal_mode=new_settings.multimodal.mode,
        )
        # 每日回收循环每轮现读该属性，替换后即生效
        self._memory.image_retention_days = new_settings.storage.image_retention_days
        # 与 __init__ 里容量推导同式的口径：新配置要求更长的上下文时热缓存
        # 不会跟着扩，历史会偏短——提示重启而不是静默降容
        max_context = max(
            [p.context_size for p in new_settings.groups.values()]
            + [new_settings.groups_default.context_size]
        )
        if max_context * 2 > self._memory.default_capacity:
            logger.warning(
                "新配置的 context_size（%d）超出启动时确定的热缓存容量（%d），"
                "历史会偏短，如需完全生效请重启",
                max_context,
                self._memory.default_capacity,
            )
        logger.info("配置已热重载")
        return True

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

        # 以决策时刻的消息集为基线：之后（生成中或连发打字中）只要记忆里
        # 新进了他人的消息，就算「被打断」——下一条发出前先让 reply 模型对
        # 剩下的腹稿重想一次（见 _send_reply_segments 与 reconsider）。
        memory = await self._memory.get(group_id)
        seen_ids = {r.message_id for r in memory.tail(len(memory))}
        reconsider_left = _MAX_RECONSIDER_PER_BURST

        async def reconsider(sent: list[str], pending: list[str]) -> ProcessedReply | None:
            """被打断后的重想调用。

            返回新发送计划；返回 None 表示没法重想（预算用尽或调用失败，
            调用方按原计划继续）；返回空计划表示模型决定放弃剩余消息。
            """
            nonlocal reconsider_left
            if reconsider_left <= 0:
                logger.debug("群 %d 本轮连发的重想预算用尽，按原计划继续", group_id)
                return None
            reconsider_left -= 1
            recent = memory.tail(profile.context_size)
            static_system = runtime.static_system(
                "reply", profile.persona, via_tool=self._ai.reply_tool_use
            )
            nicknames = nickname_list_from_history([record_to_turn(r) for r in recent])
            runtime_system = runtime_system_prompt(
                group_id, date.today().isoformat(), nicknames
            )
            draft = await self._generate_with_retry(
                self._ai.reconsider_reply,
                static_system,
                runtime_system,
                recent,
                fmt_now_text(),
                sent_segments=tuple(sent),
                pending_segments=tuple(pending),
            )
            if draft is None:
                logger.warning("群 %d 被打断后重想调用失败，按原计划继续", group_id)
                return None
            if draft.ops:
                await self._apply_image_ops(group_id, memory, list(draft.ops))
            if not draft.text:
                return ProcessedReply([], [])  # 空计划＝放弃剩余
            if _verbatim_match(draft.text, pending):
                # 一字不改地要继续：沿用原计划本身（连同已经掷好的错别字
                # 与更正），别让「重想」把该发的话再重新加工一遍
                logger.debug("群 %d 重想后决定照原样继续", group_id)
                return None
            return process_reply(
                draft.text, self._settings.response_post_process, rng=self._pp_rng
            )

        reply_text = await self._compose_reply(group_id, msg, profile, runtime, decision)
        if not reply_text:
            logger.info("群 %d 回复生成为空，放弃发送", group_id)
            return
        if not decision.forced and not self._consume_daily_quota():
            logger.info("达到 global_daily_limit，本次主动回复被拦截")
            return
        # 输出层拟人化后处理：拆条 + 打字延迟 + 错别字/更正（见 postprocess）
        processed = process_reply(
            reply_text, self._settings.response_post_process, rng=self._pp_rng
        )
        # 插话重想链路同受后处理总开关约束：enabled=False 时发送链路必须
        # 与未引入后处理前完全一致（整条单发、不放弃不改写、不多花调用）。
        sent_count = 0
        try:
            sent_count = await self._send_reply_segments(
                group_id,
                processed,
                seen_ids=seen_ids,
                reconsider=(
                    reconsider if self._settings.response_post_process.enabled else None
                ),
            )
        except Exception:
            # 发送链路内部已消化重试耗尽的失败；走到这里说明是预期外的
            # 错误（如重想闭包里的缺陷），不记账也不退配额，保守放过。
            logger.exception("群 %d 回复发送环节异常", group_id)
            return
        if not sent_count:
            # 一条正文也没发出去（重想后全部放弃，或首条即发送失败）：
            # 退还本次主动回复的日配额、不刷新冷却与发言间隔——什么都没
            # 说却消耗节制，会把 prompts 里承诺「稍后照常回应」的插话本身
            # 拦在护栏之外。
            if not decision.forced:
                self._refund_daily_quota()
            return
        # 哪怕后续条目发送失败，只要已经开口就该记账：护栏的语义是「距
        # 上次发言」，已发出的消息在群里是真实存在的。只有主动插话消耗并
        # 刷新冷却；@ 必答和对话延续都不该把正在进行的交流掐断。
        # 自发言的记忆写回不在这里：_send_reply_segments 每成功发出一条就
        # 立即把该条写回，连发期间穿插进来的他人消息才能落在真实位置。
        if not decision.forced and not decision.engaged:
            runtime.last_proactive_ts = time.time()
        runtime.msgs_since_reply = 0

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
                self._ai.generate_reply,
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

    async def _generate_with_retry(self, fn, *args, **kwargs) -> ReplyDraft | None:
        """带指数退避地调用一个 reply 模型的生成函数（generate/reconsider）。

        两次都失败返回 None——调用方须把它与「模型主动输出空正文」区分开：
        前者按原计划继续，后者才是模型的明确决定。
        """
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                logger.warning("生成回复第 %d 次失败：%s", attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2
        logger.error("生成回复最终失败：%s", last_exc)
        return None

    async def _send_reply_segments(
        self,
        group_id: int,
        processed: ProcessedReply,
        *,
        seen_ids: set[int],
        reconsider=None,
    ) -> int:
        """逐条发送拆条结果，制造真人连发多条的打字节奏。

        返回实际成功发出的正文条数（更正不计）：调用方据此决定护栏记账与
        日配额退还——「一条都没发出去」和「发过言」是两种必须区分的结局。
        第一条不等待：LLM 生成本身的耗时就是自然延迟。后续每条（含错别字
        更正）先按本条文本估算打字时长 sleep，再走现有的 3 次重试发送；
        任何一条最终失败都记错误日志并放弃剩余条目（不再向上抛出，
        已经发出的条数不能随异常一起丢失）。
        发出每一条之前先看有没有被打断：基线（seen_ids）之后新入库的他人
        消息即插话，此时先请 reconsider 对剩余腹稿重想——放弃（直接返回，
        旧计划的更正也随之作废）、换上新计划继续，或（重想不可得时）按原
        计划照发。被打断却对插话视若无睹地刷屏，比闭嘴更像机器人。
        每条正文发送成功后立即以对应的无错字原文单独写回记忆
        （memory_segments 与 messages 下标对齐）：写回时机跟着发送走，
        插话才会落在真实的时间位置，自己的多条发言在历史里也是各自独立
        的 assistant 回合；被放弃的腹稿从未发出，也就从不进入记忆。
        错别字更正只是面向群友的表层噪音，不进记忆。
        更正消息用 OneBot v11 reply 消息段引用最后一条正文（SnowLuma 的
        send_group_msg 返回 message_id）；拿不到 id 时退回不带引用的纯文本，
        并记警告日志。
        """
        memory = await self._memory.get(group_id)
        speed = self._settings.response_post_process.typing_speed
        plan = processed
        index = 0
        sent_clean: list[str] = []
        last_sent_id: int | None = None
        while True:
            if sent_clean:  # 本轮连发已开口：每条之前都要有打字间隔
                await self._typing_delay(
                    group_id,
                    len(sent_clean) + 1,
                    len(sent_clean) + len(plan.messages) - index,
                    plan.messages[index],
                    speed,
                )
            if reconsider is not None:
                interrupted = [
                    r
                    for r in memory.tail(len(memory))
                    if not r.is_self and r.message_id not in seen_ids
                ]
                if interrupted:
                    logger.info(
                        "群 %d 连发被 %d 条新消息打断，重想剩下的 %d 条腹稿",
                        group_id,
                        len(interrupted),
                        len(plan.messages) - index,
                    )
                    seen_ids.update(r.message_id for r in interrupted)
                    outcome = await reconsider(
                        sent_clean, plan.memory_segments[index:]
                    )
                    if outcome is not None:
                        if not outcome.messages:
                            logger.info(
                                "群 %d 重想后决定到此为止：剩余腹稿不再发送", group_id
                            )
                            return len(sent_clean)
                        plan, index = outcome, 0
                        continue  # 新计划同样先延迟、再查一次是否又被插话
            try:
                mid = await self._send_with_retry(group_id, plan.messages[index])
            except Exception as exc:
                logger.error(
                    "群 %d 发送失败：%s（已发出 %d 条，放弃剩余条目）",
                    group_id,
                    exc,
                    len(sent_clean),
                )
                return len(sent_clean)
            await memory.append(
                self._self_record(group_id, plan.memory_segments[index])
            )
            sent_clean.append(plan.memory_segments[index])
            if mid is not None:
                last_sent_id = mid  # message_id 0 是合法 id，不能走 or 短路
            index += 1
            if index >= len(plan.messages):
                break
        if plan.correction:
            total = len(sent_clean) + 1
            await self._typing_delay(group_id, total, total, plan.correction, speed)
            segments: list[dict] = []
            if last_sent_id is not None:
                segments.append({"type": "reply", "data": {"id": str(last_sent_id)}})
            else:
                logger.warning(
                    "群 %d 发送响应未返回 message_id，错别字更正不带引用直接发送", group_id
                )
            segments.append({"type": "text", "data": {"text": plan.correction}})
            try:
                await self._send_with_retry(group_id, segments)
            except Exception as exc:
                # 正文已全部发出，只是更正这条噪音没送出去：记日志即可，
                # 不能让已发言的记账跟着丢失。
                logger.error("群 %d 发送失败：%s（更正消息作废）", group_id, exc)
        return len(sent_clean)

    async def _typing_delay(
        self,
        group_id: int,
        index: int,
        total: int,
        message: str,
        speed: float,
    ) -> None:
        """按下一条消息的估算打字时长等待；speed 为 0 时直接跳过。"""
        delay = estimate_typing_time(message, speed)
        logger.debug(
            "群 %d 第 %d/%d 条预计打字 %.1f 秒", group_id, index, total, delay
        )
        if delay > 0:
            await asyncio.sleep(delay)

    async def _send_with_retry(
        self, group_id: int, message: str | list[dict]
    ) -> int | None:
        """发送一条消息（文本或 OneBot 段数组），返回其 message_id。"""
        delay = 1.5
        for attempt in range(3):
            try:
                return await self._snowluma.send_group_msg(group_id, message)
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning("发送第 %d 次失败：%s", attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2
        return None  # pragma: no cover —— 最后一次失败必抛

    def _self_record(self, group_id: int, text: str) -> ChatRecord:
        """自己发出的一条消息对应的记忆记录（发送成功后逐条写回）。"""
        return ChatRecord(
            message_id=self._next_self_message_id(),
            group_id=group_id,
            user_id=self._settings.bot.self_qq,
            nickname="糖糖",
            text=text,
            ts=time.time(),
            is_self=True,
        )

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

    def _refund_daily_quota(self) -> None:
        """退还一次主动回复的日配额：配额语义是「每日主动发言数」，
        重想全部放弃或首条即发送失败时其实一句话都没说。跨天重置后计数
        已属于新的一天，不退昨天的账。"""
        if date.today() == self._daily_date and self._daily_replies > 0:
            self._daily_replies -= 1


def build_bot(config_file: str = "config.json5") -> CandyBot:
    """从项目根目录 config.py 的 ConfigClass 构造 CandyBot。

    同时传入 settings_loader：配置热重载时基于 Config 单例当前内存中的
    原始配置重新解析与校验，无需重启进程。
    """
    from config import Config  # 用户已有的 JSON5 配置单例

    Config._config_file = config_file
    Config.load_config()
    return CandyBot(load_settings(Config), settings_loader=lambda: load_settings(Config))
