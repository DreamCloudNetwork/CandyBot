"""核心编排：事件过滤链、决策、回复与发送。

每群一个串行 asyncio 队列保证决策顺序。@ 必答与「对方正在和我说话」的
消息（judge 判定为延续与本机器人的对话）不受冷却和护栏限制；其余主动
插话需依次通过冷却、发言间隔、热闹静默三道护栏及判定门槛。

连发期间（生成中或打字延迟中）一旦有他人新消息进入记忆，下一条发出前
会先让 reply 模型对剩余腹稿重想一次：可以放弃、改写或照原样继续，防止
别人已经插话、AI 却把打好的字一条条硬发完。

决策层另有三项增强（配置都在 generation 段，关闭时行为与引入前一致）：

- 发送前新鲜度检查（freshness_check_enabled）：回复生成期间群里若进了
  明确指向 bot 的新消息（@我/回复我），把新消息并入上下文重生成一次
  （每条回复至多一次）；普通新话题不触发——宁可稍旧也不无限拖延。
- 观望（observe_band / observe_delay_seconds）：终评分差一点点没过门槛
  且未被护栏直接终止的消息，不再直接放弃，延迟一段时间后连同届时的
  最新上下文重新走一遍判定（每条消息至多一次，护栏与配额路径不变）。
- 重复抑制（repetition_guard_enabled）：生成前若发现目标消息之后已有
  自己的发言且对方没再开口，在 L4 注入「不要和之前的发言重复」的提醒。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import aiohttp

from .ai import AIClient, ImageOp, ReplyDraft, split_image_ops
from .commandline import CommandUsageError, detect_command_name, parse_invocation
from .dedup import MessageDedup
from .events_server import EventsServer
from .memory import GroupMemory, MemoryManager
from .migrations import pending_migrations
from .models import (
    ChatRecord,
    Decision,
    GroupProfile,
    NormalizedMessage,
    Settings,
    load_settings,
)
from .learning import LearningService, day_bounds
from .normalize import normalize_group_message
from .plugin_api import (
    CommandContext,
    CommandRegistry,
    CommandSpec,
    build_registry,
)
from .postprocess import (
    ProcessedReply,
    ensure_indexes,
    estimate_typing_time,
    process_reply,
    typing_policy_of,
)
from .prompts import (
    nickname_list_from_history,
    now_text as fmt_now_text,
    record_to_turn,
    runtime_system_prompt,
    static_system_prompt,
)
from .snowluma import SnowlumaClient
from .stickers import STICKER_RECORD_TEXT, StickerStore

logger = logging.getLogger(__name__)

# 「已观望过」记账的进程内历史上限：只为防同一条消息被反复观望，
# 长期运行的活跃群里旧条目没有保留价值，超出按 FIFO 遗忘。
# （纯内存记账容量，不进配置；重想预算等可调参数已移到 generation 段。）
_OBSERVED_HISTORY_MAX = 4096


def _verbatim_match(text: str, pending: list[str]) -> bool:
    """重想输出是否与腹稿一字不差（忽略行间空白差异）。

    模型的复读常带尾部空白或空行，也可能沿用「一行一条」之外的换行习惯；
    不规范化就会误判成改写，让已经掷好的错别字与更正被 process_reply
    重新加工一遍。
    """

    def norm(s: str) -> str:
        return "\n".join(line.strip() for line in s.splitlines() if line.strip())

    return norm(text) == norm("\n".join(pending))


def _index_of(records: list[ChatRecord], message_id: int) -> int:
    """在时间正序的快照里按 message_id 定位记录；找不到返回 -1。"""
    for i, record in enumerate(records):
        if record.message_id == message_id:
            return i
    return -1


def _self_reply_after(
    records: list[ChatRecord], target_id: int
) -> tuple[bool, bool]:
    """目标消息是否还在上下文里、其后是否已有自己的发言。

    返回 (目标存在, 其后有 is_self 记录)。观望到点时据此决定要不要取消
    重评：连发写回紧跟发送，目标之后出现的 is_self 记录即「这一带已经
    回复过了」的信号——无论那条发言实际是在回哪条消息（哪怕是回另一条
    无关的 @），都按有意保留的保守规则取消，避免同一话题被连说两遍。
    目标不存在（被撤回或被淘汰出热缓存）时同样按取消处理。
    """
    index = _index_of(records, target_id)
    if index < 0:
        return False, False
    return True, any(record.is_self for record in records[index + 1 :])


def _already_replied_to(records: list[ChatRecord], target: ChatRecord) -> bool:
    """重复回复判定（重复抑制 / 任务 C）：本次生成是否很可能在重复自己
    针对同一条消息刚发过的话。

    判定规则（records 为热缓存快照、时间正序，逐条扫描）：
    1. 目标消息不在快照里（已撤回或被淘汰出热缓存）→ False；
    2. 目标消息之后没有任何 is_self 记录 → False（还没回过它）；
    3. 取目标之后最后一条 is_self 记录：若它之后还出现目标发送者
       （user_id 相同且非 self）的新消息，说明对话已经往前走了 → False；
    4. 其余情况 → True：自己在这条消息之后发过言、对方也没再开口，
       此时若再生成，极可能把同一句话说第二遍。
    """
    index = _index_of(records, target.message_id)
    if index < 0:
        return False
    after = records[index + 1 :]
    last_self = -1
    for i, record in enumerate(after):
        if record.is_self:
            last_self = i
    if last_self < 0:
        return False
    return not any(
        record.user_id == target.user_id and not record.is_self
        for record in after[last_self + 1 :]
    )


def _directed_new_messages(
    memory: GroupMemory,
    runtime: GroupRuntime,
    seen_ids: set[int],
    *,
    include_commands: bool = True,
) -> list[ChatRecord]:
    """找出决策基线之后新入记忆、且明确指向 bot（@我/回复我）的他人消息。

    「生成期间新增」以 seen_ids（决策时刻的记忆快照 id 集）为基线，而不是
    与目标消息比时间戳：决策前已在上下文里的消息模型本来就看得见，不该
    触发重生成。「指向 bot」取自 GroupRuntime 登记的近期 mentioned_me 消息
    id（ChatRecord 不携带该信息，_on_event 在 memory.append 的同时登记）。
    include_commands=False 时跳过命令插件产生的消息——它们没进模型上下文
    （见 GroupMemory.model_tail），不该触发新鲜度重生成。
    """
    mentions = set(runtime.recent_mentions)
    if not mentions:
        return []
    return [
        record
        for record in memory.tail(len(memory))
        if not record.is_self
        and record.message_id not in seen_ids
        and record.message_id in mentions
        and (include_commands or not record.is_command)
    ]


@dataclass(slots=True)
class CommandInvocation:
    """_on_event 命中命令注册表后随队列元素携带的调用意图。

    参数解析与 handler 调用都推迟到 worker 消费时：命令执行与正常回复共用
    每群串行队列，保证两条链路的发言在群里按到达顺序出现。
    """

    spec: CommandSpec
    text: str  # 命令全文（已 lstrip，供 shlex 解析）


def _command_memory_text(result: str | list[dict]) -> str:
    """命令输出写回群记忆用的纯文本：str 原样；段数组拼接文本段，
    纯媒体段落一个占位（与表情包的「[表情包]」同理，图片不进历史）。"""
    if isinstance(result, str):
        return result
    parts = [
        str(seg.get("data", {}).get("text", ""))
        for seg in result
        if isinstance(seg, dict) and seg.get("type") == "text"
    ]
    return " ".join(p for p in parts if p) or "[命令消息]"


class GroupRuntime:
    """单个群的运行时状态。"""

    def __init__(self) -> None:
        self.last_proactive_ts: float = 0.0
        # 反插嘴护栏的记账：上次自己发言以来的他人消息数（含当前待判定消息），
        # 初值为大数表示「从未发言」不被间隔约束，发送后归零重新累积；
        # 近一分钟的群消息时间戳（含 @ 必答的消息，反映真实热闹程度）。
        self.msgs_since_reply: int = 10**9
        self.recent_msg_times: deque[float] = deque()
        # 近期「明确指向自己」（@我/回复我的消息，即 normalize 的
        # mentioned_me）的消息 id：ChatRecord 不带这个信息，发送前新鲜度
        # 检查据此判断生成期间新进的消息是不是在等自己回话。有界即可，
        # 只需与本轮决策基线之后的新消息做交集。
        self.recent_mentions: deque[int] = deque(maxlen=256)
        self._static_cache_key: tuple[int, str] | None = None
        self._static_cache: str = ""
        # L2 群印象快照：按 (group_id, 日期) 缓存，天内字节级不变
        # （KV Cache 硬纪律），跨过零点才从库里重新取一次。
        self._impressions_day: str | None = None
        self._impressions: tuple[tuple[str, str], ...] = ()

    def impressions_snapshot(
        self, today: str
    ) -> tuple[tuple[str, str], ...] | None:
        """该日缓存的印象快照；未缓存或已跨天返回 None（需要刷新）。"""
        return self._impressions if self._impressions_day == today else None

    def remember_impressions(
        self, today: str, snapshot: tuple[tuple[str, str], ...]
    ) -> None:
        """写入某日的印象快照（tuple 固化，保证天内重建字节级相同）。"""
        self._impressions_day = today
        self._impressions = snapshot

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
        registry: CommandRegistry | None = None,
    ):
        self._settings = settings
        # 热重载时重建 Settings 快照的来源（由 build_bot 传入
        # lambda: load_settings(Config)）；不传则该实例不支持热重载
        # （如测试里手工注入 settings）。
        self._settings_loader = settings_loader
        # 命令插件注册表：内置命令 + plugins.enabled 时扫描插件目录。
        # 不传则装到进程共享的 default_registry（生产路径）；测试传入
        # 独立实例即可与真实 plugins/ 目录隔离。构建期装载，改插件需重启。
        self._commands = build_registry(settings, registry)
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
        # 后台学习服务（每日群印象 / 表达 / 黑话）：热缓存淘汰的消息交给它
        # 攒批触发学习；配置与 AI 客户端热重载后仍取到最新实例（回调注入）。
        self._learning = LearningService(
            self._memory, lambda: self._ai, lambda: self._settings
        )
        self._memory.evict_listener = self._learning.note_evicted
        self._http = aiohttp.ClientSession()
        self._ai = AIClient(
            models=settings.models,
            generation=settings.generation,
            multimodal_mode=settings.multimodal.mode,
        )
        self._snowluma = SnowlumaClient(settings.snowluma)
        # 表情包（任务 C 最小版）：收图时在 _on_event 收集，文字回复发送
        # 成功后按小概率跟发；配置经回调现取，热重载即时生效。
        sticker_root = Path(settings.bot.data_dir) / "stickers"
        self._stickers = StickerStore(
            sticker_root,
            self._memory.db,
            lambda: self._settings,
        )
        # 跟发掷点与随机抽图的随机源；测试可替换为固定种子的 random.Random。
        self._sticker_rng: random.Random = random.SystemRandom()
        self._server = EventsServer(
            self._on_event,
            host=settings.bot.listen_host,
            port=settings.bot.listen_port,
            secret=settings.bot.event_secret,
            max_body_bytes=settings.bot.max_event_body_bytes,
            # 表情包 HTTP 模式的供图路由（send_mode=base64/file 时用不到，
            # 但挂载着不影响任何东西，热重载切到 http 即刻可发外链）
            stickers_dir=sticker_root,
        )
        self._runtimes: dict[int, GroupRuntime] = defaultdict(GroupRuntime)
        # 队列元素为 (消息, 是否观望重评, 命令调用)：观望到点后把原消息
        # 重新入队，以 observe=True 复用同一串行队列与判定路径，绝不并发
        # 生成；命中命令注册表的消息带 invocation 走 _run_command，同样
        # 经串行队列与正常回复保持同群时序
        self._group_queues: dict[
            int, asyncio.Queue[tuple[NormalizedMessage, bool, CommandInvocation | None]]
        ] = {}
        self._queue_workers: dict[int, asyncio.Task[None]] = {}
        # 观望的记账：(group_id, message_id) → 未决任务（停机时取消）；
        # 已观望过的消息 id 集合（每条至多观望一次，防循环），按 FIFO 封顶
        self._observe_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}
        self._observed_once: set[tuple[int, int]] = set()
        self._observed_order: deque[tuple[int, int]] = deque()
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
        ps = self._settings.plugins
        # 迁移不在启动时自动执行（见 migrations.py）：库结构落后（旧库缺
        # is_command 列，消息入库会直接报错）就拒绝启动，提示手动迁移。
        gaps = await pending_migrations(self._memory.db)
        if gaps:
            raise SystemExit(
                f"数据库结构落后于当前版本（缺：{'、'.join(gaps)}）。"
                "请先停止机器人，运行 python -m candybot.migrations 完成迁移，"
                "再重新启动。"
            )
        logger.info(
            "命令插件%s，注册命令：%s",
            "已启用" if ps.enabled else "已禁用（/ 消息照常走大模型）",
            "、".join(f"/{n}" for n in self._commands.names()) or "（无）",
        )
        await self._learning.start()  # 每日群印象循环（启动时先补昨天的）
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
            logger.info("未能查询登录账号（action 缺失或鉴权失败），跳过自检")
        await self._server.start()

    async def stop(self) -> None:
        self._stopping = True
        # 未决的观望任务先取消：_observe_task 的取消语义是静默退出，
        # 取消后不会再向队列投递二次判定
        observe_tasks = list(self._observe_tasks.values())
        for task in observe_tasks:
            task.cancel()
        if observe_tasks:
            await asyncio.gather(*observe_tasks, return_exceptions=True)
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
        # 学习任务可能正卡在数据库写入上：先停它们，再释放连接池
        await self._learning.stop()
        await self._memory.close()

    # ------------------------------------------------------------ 配置热重载

    def reload_settings(self) -> bool:
        """重新解析 config.json5 并原子替换运行时配置（热重载入口）。

        由 main.py 经 loop.call_soon_threadsafe 调度到事件循环线程调用，与
        消息处理天然串行，无需额外加锁。启动时已烘进监听/客户端会话的字段
        无法就地更换、改动仍需重启：bot.listen_host / listen_port /
        event_secret / max_event_body_bytes（aiohttp 监听与签名校验、请求体
        上限已绑定）、bot.data_dir、snowluma 的连接类字段（endpoint /
        api_key / timeout_ms / allow_private_endpoint，HTTP 客户端会话建好；
        但 send_max_attempts / send_retry_delay_seconds 是发送时现读，
        即时生效）；热缓存容量也在
        启动时按全局最大 context_size 定死（新配置超出时记警告）。其余
        （白名单、人设与 bot.self_nickname、护栏阈值、模型与生成参数、
        多模态、输出后处理、限速、图片保留天数、表情包识别启发式与跟发图片
        引用方式 stickers.send_mode / http_base_url——供图路由在事件服务上
        常驻挂载，切到 http 即刻可发外链）即时生效。

        解析失败（典型场景：配置写坏）完整记日志但不拖垮服务：沿用旧配置，
        下一次保存自动重试。返回是否实际完成了替换。

        learning 段与 models.learning 随快照/AI 客户端重建即时生效；已缓存
        的 L2 印象快照当天不刷新（天内字节级不变），次日自然按新配置重建。
        models.embedding 与表达选取方式（learning.expression_selection_mode
        等）同样即时生效：embedding 模型变更后旧的表达向量缓存整体作废，
        由学习入库/下一次启动的后台补算按新模型重算（见 learning.py）；
        改成 vector 却没配 embedding 的新配置会在解析阶段就报错、被沿用旧配置。
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
                self_nickname=self._settings.bot.self_nickname,
                multimodal=self._settings.multimodal,
                # 表情包识别启发式参数（尺寸上限与总结关键词）现取现用，
                # 热重载即时生效
                stickers=self._settings.stickers,
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

        invocation = self._detect_command(normalized)
        memory = await self._memory.get(group_id)
        if invocation is not None:
            # 插件产生的命令消息照常入库（审计与印象统计可用），只打
            # is_command 标记：plugins.include_commands_in_history=false 时
            # 由 model_tail 把它与命令回复一起过滤出模型的历史上下文。
            normalized.record.is_command = True
        await memory.append(normalized.record)
        if normalized.mentioned_me:
            # 新鲜度检查的记账：ChatRecord 不带 mentioned_me，这里登记
            # 「明确指向自己」的消息 id，供发送前比对生成期间的新消息
            self._runtimes[group_id].recent_mentions.append(normalized.record.message_id)
        logger.debug("收到消息 %s : %s",group_id,normalized)
        if normalized.sticker_flags:
            # 表情包收集（任务 C）：辅助能力，失败只记日志，不挡这条消息的决策
            try:
                await self._stickers.collect(
                    normalized.record, normalized.sticker_flags
                )
            except Exception:
                logger.warning("群 %d 表情包收集失败", group_id, exc_info=True)
        if invocation is not None:
            # 命中注册表的 / 命令：取消大模型自主回复，直接交给插件执行
            # （仍走本群串行队列，与正常回复保持到达顺序）
            await self._enqueue(group_id, normalized, invocation=invocation)
            return
        await self._enqueue(group_id, normalized)

    def _detect_command(self, msg: NormalizedMessage) -> CommandInvocation | None:
        """判断这条消息是否命中命令注册表（未命中一律按普通消息走原链路）。

        只看「/ + 第一个空白前的命令名」能否查到 spec；未知命令（含插件
        总开关关闭时）不作否决，照常交给大模型。enabled 现取现读，热重载
        即时生效。
        """
        if not self._settings.plugins.enabled:
            return None
        name = detect_command_name(msg.record.text)
        if name is None:
            return None
        spec = self._commands.get(name)
        if spec is None:
            return None
        return CommandInvocation(spec=spec, text=msg.record.text.lstrip())

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

    async def _enqueue(
        self,
        group_id: int,
        msg: NormalizedMessage,
        *,
        observe: bool = False,
        invocation: CommandInvocation | None = None,
    ) -> None:
        queue = self._group_queues.get(group_id)
        if queue is None:
            queue: asyncio.Queue[
                tuple[NormalizedMessage, bool, CommandInvocation | None]
            ] = asyncio.Queue()
            self._group_queues[group_id] = queue
            self._queue_workers[group_id] = asyncio.create_task(
                self._group_worker(group_id, queue)
            )
        await queue.put((msg, observe, invocation))

    async def _group_worker(
        self,
        group_id: int,
        queue: asyncio.Queue[
            tuple[NormalizedMessage, bool, CommandInvocation | None]
        ],
    ) -> None:
        while not self._stopping:
            try:
                msg, observe, invocation = await queue.get()
            except asyncio.CancelledError:
                return
            try:
                if invocation is not None:
                    await self._run_command(group_id, msg, invocation)
                else:
                    await self._decide_and_reply(group_id, msg, observe=observe)
            except Exception:
                logger.exception(
                    "处理群 %d 消息 %d 时出错", group_id, msg.record.message_id
                )

    # ------------------------------------------------------------ 命令插件

    async def _run_command(
        self, group_id: int, msg: NormalizedMessage, invocation: CommandInvocation
    ) -> None:
        """命令分发：解析参数 → 调用插件 handler → 原样发送返回值 → 写回记忆。

        完全绕开判定/生成/冷却/日配额/后处理链路：命令说的就是插件返回的
        原文，不拆条、不打字延迟、不注入错别字。失败（用法错误、超时、
        handler 崩溃）回一句中文提示——机器人对一条 /命令 完全沉默会被
        当成死机；handler 返回 None 或空串才真正不发。
        plugins.include_commands_in_history=false 时命令消息与这条回复
        仍照常写回记忆（带 is_command 标记），只是不再送入模型历史上下文。
        """
        spec = invocation.spec
        memory = await self._memory.get(group_id)
        result: str | list[dict] | None
        try:
            args = parse_invocation(spec, invocation.text)
        except CommandUsageError as exc:
            result = str(exc)
            logger.info("群 %d 命令 /%s 用法错误：%s", group_id, spec.name, exc)
        else:
            ctx = CommandContext(
                group_id=group_id,
                user_id=msg.record.user_id,
                nickname=msg.record.nickname,
                text=invocation.text,
                args=args,
                registry=self._commands,
                settings=self._settings,
                db=self._memory.db,
            )
            try:
                result = await self._invoke_handler(spec, ctx)
            except TimeoutError:
                result = f"/{spec.name} 执行超时，稍后再试吧。"
                logger.warning("群 %d 命令 /%s 执行超时", group_id, spec.name)
            except Exception:
                result = f"/{spec.name} 执行失败了（内部错误）。"
                logger.exception("群 %d 命令 /%s 执行异常", group_id, spec.name)
        if not result:
            return
        try:
            await self._send_with_retry(group_id, result)
        except Exception as exc:
            # 与主链路一致：发送失败只记日志，不写回记忆（群里没说过）
            logger.error("群 %d 命令回复发送失败：%s", group_id, exc)
            return
        await memory.append(
            self._self_record(
                group_id, _command_memory_text(result), is_command=True
            )
        )

    async def _invoke_handler(
        self, spec: CommandSpec, ctx: CommandContext
    ) -> str | list[dict] | None:
        """调用 handler：同步返回即时生效，协程受 plugins.timeout_seconds 约束。

        返回值类型不合法（插件写了不该写的东西）按「不发消息」处理并记
        warning——异常永远不许从这里漏到 worker 顶层以外。
        """
        outcome = spec.handler(ctx)
        if inspect.isawaitable(outcome):
            timeout = max(float(self._settings.plugins.timeout_seconds), 1.0)
            outcome = await asyncio.wait_for(outcome, timeout)
        if outcome is None or isinstance(outcome, str):
            return outcome
        if isinstance(outcome, list) and all(isinstance(seg, dict) for seg in outcome):
            return outcome
        logger.warning(
            "命令 /%s 返回了非法类型 %s，忽略不发送",
            spec.name,
            type(outcome).__name__,
        )
        return None

    # ------------------------------------------------------------ 决策与回复

    async def _impressions_for(
        self, group_id: int, runtime: GroupRuntime
    ) -> tuple[tuple[str, str], ...]:
        """L2 群印象注入：按 (group_id, 日期) 快照缓存。

        同一天内直接返回缓存的 tuple，天内任意次重建 runtime_system_prompt
        字节级相同（KV Cache 硬纪律）；跨过零点快照键变化，才从库里重取
        一次——那时昨日印象已由每日任务生成。

        零点竞态防护：每日印象任务是零点后逐群串行生成的，当天第一条消息
        可能赶在昨日印象就位之前进来。此时暂不固化快照（下次回复前重查），
        印象就位后才固化——代价是 L2 前缀当天可能多刷新一次，好过整日缺失
        昨日的印象（见 _yesterday_impression_pending）。
        """
        ls = self._settings.learning
        if not ls.enabled or not ls.impression_enabled:
            return ()
        today = date.today().isoformat()
        cached = runtime.impressions_snapshot(today)
        if cached is not None:
            return cached
        try:
            rows = await self._memory.db.load_impressions(group_id, ls.impression_days)
        except Exception:
            # 注入是辅助能力：读库失败按「今天没有印象」处理并缓存空快照，
            # 既不阻断决策，也保证天内字节级稳定
            logger.warning("群 %d 读取群印象失败，本次运行日内不注入", group_id, exc_info=True)
            rows = []
        else:
            if await self._yesterday_impression_pending(group_id, {row.day for row in rows}):
                # 昨日印象还在生成途中：返回当前不完整的快照但暂不固化
                return tuple((row.day, row.summary) for row in rows)
        snapshot = tuple((row.day, row.summary) for row in rows)
        runtime.remember_impressions(today, snapshot)
        return snapshot

    async def _yesterday_impression_pending(
        self, group_id: int, present_days: set[str]
    ) -> bool:
        """判断「昨天有聊天、但印象还没生成」——即 L2 快照暂不完整。

        昨日印象缺席且该群昨天有过消息时返回 True，调用方据此跳过快照
        固化。存在性检查自身失败时按「无待生成」处理（照常固化）：竞态
        防护不能反过来变成天内不稳定的来源。
        """
        yesterday = date.today() - timedelta(days=1)
        if yesterday.isoformat() in present_days:
            return False
        try:
            start_ts, end_ts = day_bounds(yesterday)
            return await self._memory.db.has_day_records(group_id, start_ts, end_ts)
        except Exception:
            logger.warning(
                "群 %d 检查昨日消息是否存在失败，跳过竞态防护照常固化快照",
                group_id,
                exc_info=True,
            )
            return False

    async def _learning_hints(
        self,
        group_id: int,
        recent: list[ChatRecord],
        *,
        trigger: ChatRecord | None = None,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """回复前的 L4 注入准备：抽表达 + 匹配黑话（都只进指令层）。

        由 _decide_and_reply 每轮调用一次，结果喂给该轮全部 _compose_reply
        生成（含新鲜度重生成）：抽中会刷新表达条目的最近使用时间，同一轮
        内重复调用等于重复采样，必须避免。

        recent/trigger 只在表达 vector 检索模式下参与语境构造（weighted_random
        模式忽略，行为与引入前一致）：trigger 为触发本轮决策的消息，其 id
        同时是查询向量缓存的键；黑话匹配沿用 recent 的纯文本拼接。

        辅助能力：任何失败只记日志、退化为不注入，绝不阻断回复本身。
        """
        ls = self._settings.learning
        expression_hints: list[tuple[str, str]] = []
        jargon_hints: list[tuple[str, str]] = []
        if not ls.enabled:
            return expression_hints, jargon_hints
        try:
            if ls.expression_enabled:
                expression_hints = await self._learning.pick_expressions(
                    group_id, ls.expression_max_inject, recent, trigger=trigger
                )
            if ls.jargon_enabled:
                context = "\n".join(
                    r.text for r in recent if not r.is_self and r.text
                )
                jargon_hints = await self._learning.match_jargons(
                    group_id, context, ls.jargon_max_inject
                )
        except Exception:
            logger.warning("群 %d 学习注入准备失败，本次跳过注入", group_id, exc_info=True)
            return [], []
        if expression_hints or jargon_hints:
            logger.debug(
                "群 %d L4 注入：表达 %r，黑话 %r", group_id, expression_hints, jargon_hints
            )
        return expression_hints, jargon_hints

    async def _decide_and_reply(
        self, group_id: int, msg: NormalizedMessage, *, observe: bool = False
    ) -> None:
        """对一条消息走完「判定 → 生成 → 发送 → 记账」的完整链路。

        observe=True 表示这是观望到点后的二次处理：护栏与配额路径与首评
        完全相同（见 _make_decision），但不再安排新的观望。

        L4 的学习注入（抽表达 + 匹配黑话）每轮只准备一次：抽中会刷新
        表达条目的最近使用时间，若让新鲜度重生成再抽一遍就是二次采样、
        二次刷新，纯属多余；提前的代价是初稿与重生成稿共享同一组 hints，
        反而让两轮生成看到一致的风格参考，语义更自洽。
        """
        profile = self._settings.profile_for(group_id)
        assert profile is not None
        runtime = self._runtimes[group_id]

        decision = await self._make_decision(
            group_id, msg, profile, runtime, observe=observe
        )
        if not decision.should_reply:
            return

        # 以决策时刻的消息集为基线：之后（生成中或连发打字中）只要记忆里
        # 新进了他人的消息，就算「被打断」——下一条发出前先让 reply 模型对
        # 剩下的腹稿重想一次（见 _send_reply_segments 与 reconsider）。
        memory = await self._memory.get(group_id)
        seen_ids = {r.message_id for r in memory.tail(len(memory))}
        # L4 学习注入每轮决策只算一次（见本方法 docstring），初稿与
        # 新鲜度重生成共享同一组 hints；失败容错在 _learning_hints 内部，
        # 退化为不注入
        expression_hints, jargon_hints = await self._learning_hints(
            group_id,
            memory.model_tail(
                profile.context_size,
                include_commands=self._settings.plugins.include_commands_in_history,
            ),
            trigger=msg.record,  # 表达 vector 检索的语境与缓存键（加权随机模式忽略）
        )
        # 一轮连发最多几次「被打断后重想」（generation.max_reconsider_per_burst）：
        # 每次重想是一回额外的 reply 模型调用，预算用尽后剩下的腹稿按原计划
        # 发完，防止病态刷屏把发送环节变成无限往返。
        reconsider_left = max(
            int(self._settings.generation.max_reconsider_per_burst), 0
        )

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
            recent = memory.model_tail(
                profile.context_size,
                include_commands=self._settings.plugins.include_commands_in_history,
            )
            static_system = runtime.static_system(
                "reply", profile.persona, via_tool=self._ai.reply_tool_use
            )
            nicknames = nickname_list_from_history([record_to_turn(r) for r in recent])
            runtime_system = runtime_system_prompt(
                group_id,
                date.today().isoformat(),
                nicknames,
                impressions=await self._impressions_for(group_id, runtime),
                commands_enabled=self._settings.plugins.enabled,
                commands_in_history=self._settings.plugins.include_commands_in_history,
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

        reply_text = await self._compose_reply(
            group_id,
            msg,
            profile,
            runtime,
            decision,
            expression_hints,
            jargon_hints,
        )
        if not reply_text:
            logger.info("群 %d 回复生成为空，放弃发送", group_id)
            return
        # 发送前新鲜度检查（打断的低成本等价物）：_compose_reply 生成期间
        # 群里可能又进了明确指向 bot 的新消息（@我/回复我），原稿基于旧
        # 上下文、可能说的正是那些消息已经回答过的话。此时把新消息并入
        # 上下文重生成一次（走现有生成与重试路径，每条回复至多一次，
        # 防止循环）；普通新话题不触发——宁可稍旧也不要无限拖延。
        if self._settings.generation.freshness_check_enabled:
            directed = _directed_new_messages(
                memory,
                runtime,
                seen_ids,
                include_commands=self._settings.plugins.include_commands_in_history,
            )
            if directed:
                logger.info(
                    "群 %d 生成期间来了 %d 条明确指向自己的新消息（%s），并入最新上下文重生成一次",
                    group_id,
                    len(directed),
                    "、".join(f"{r.nickname}({r.message_id})" for r in directed),
                )
                regenerated = await self._compose_reply(
                    group_id,
                    msg,
                    profile,
                    runtime,
                    decision,
                    expression_hints,
                    jargon_hints,
                )
                if regenerated:
                    reply_text = regenerated
                    # 基线刷新到重生成时刻：新消息已进上下文，发送环节不必
                    # 再把它们当「插话」触发连发重想
                    seen_ids = {r.message_id for r in memory.tail(len(memory))}
                else:
                    logger.warning("群 %d 新鲜度重生成未产出内容，按原稿发送", group_id)
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
        # 任务 C：文字回复成功发出后，按小概率跟发一张表情包（辅助能力，
        # 内部消化全部失败，不影响本轮记账）
        await self._maybe_send_sticker(group_id, memory)

    async def _maybe_send_sticker(self, group_id: int, memory: GroupMemory) -> None:
        """表情包跟发（任务 C 最小版）：每条文字回复后独立掷点，命中且该群
        收藏非空时随机抽一张，以 OneBot v11 image 消息段跟发（图片引用方式
        由 stickers.send_mode 决定，见 stickers.image_segment）；不做模型选择。

        发送成功后写回一条 is_self 的「[表情包]」占位记录，让模型在历史里
        知道自己发过图（路径与 base64 都不进历史）。任何失败（含 image 段
        被端点拒绝、重试耗尽）只记日志——文字已经发出去了，不能让跟发的
        失败反过来动摇本轮发言的记账。

        选中图片后不秒发：先按占位文本「[表情包]」估算一段「挑图 + 打字」
        时长 sleep（复用 postprocess.estimate_typing_time 与全局
        typing_speed 倍率，约 1 秒量级；倍率为 0 时估算即 0、自然不延迟），
        仿真人挑图要一会儿的节奏。
        """
        st = self._settings.stickers
        if not st.enabled or st.send_probability <= 0:
            return
        try:
            if self._sticker_rng.random() >= st.send_probability:
                return
            picked = await self._stickers.pick_for_send(group_id, self._sticker_rng)
            if picked is None:
                logger.debug("群 %d 掷点命中但收藏为空，不跟发表情包", group_id)
                return
            speed = self._settings.response_post_process.typing_speed
            delay = estimate_typing_time(
                STICKER_RECORD_TEXT,
                speed,
                typing_policy_of(self._settings.response_post_process),
            )
            logger.debug("群 %d 跟发表情包前预计挑图打字 %.1f 秒", group_id, delay)
            if delay > 0:
                await asyncio.sleep(delay)
            await self._send_with_retry(group_id, [self._stickers.image_segment(picked)])
            await memory.append(self._self_record(group_id, STICKER_RECORD_TEXT))
            await self._stickers.mark_used(picked)
            logger.info(
                "群 %d 跟发表情包：%s（该图累计使用 %d 次）",
                group_id,
                picked.path,
                picked.use_count + 1,
            )
        except Exception:
            logger.warning("群 %d 表情包跟发失败", group_id, exc_info=True)

    async def _compose_reply(
        self,
        group_id: int,
        msg: NormalizedMessage,
        profile: GroupProfile,
        runtime: GroupRuntime,
        decision: Decision,
        expression_hints: list[tuple[str, str]],
        jargon_hints: list[tuple[str, str]],
    ) -> str | None:
        """生成回复并处理其中的图片生命周期操作。

        expression_hints / jargon_hints 是本轮决策的 L4 学习注入内容，由
        _decide_and_reply 调 _learning_hints 计算一次后传入：新鲜度重生成
        再次进入本方法时复用同一组 hints，绝不二次抽样、二次刷新表达条目。

        首稿通过 send_reply 工具参数携带 drop/recall 操作：立即把操作落到
        记忆；若发生过成功的召回（模型想重新查看某张旧图），重建上下文后
        再生成一次，让召回的原图进入本轮对话。二稿不再重试，保证每条消息
        至多两次生成调用。作为最终防线，正文中任何形似 <drop_img>/<recall_img>
        的残留标记仍会被剥除收编，绝不发进群里。

        生成前先做重复抑制检查（任务 C，repetition_guard_enabled=False 时
        跳过）：目标消息之后是否已有自己的发言、且对方没再开口（判定规则
        见 _already_replied_to）。命中时不拦截发送，只在 L4 注入重复提醒，
        把取舍交给模型。
        """
        memory = await self._memory.get(group_id)
        repetition = self._settings.generation.repetition_guard_enabled and (
            _already_replied_to(memory.tail(len(memory)), msg.record)
        )
        if repetition:
            logger.info(
                "群 %d 消息 %d 之后已有自己的发言，本次生成注入 L4 重复提醒",
                group_id,
                msg.record.message_id,
            )
        for attempt in range(2):
            recent = memory.model_tail(
                profile.context_size,
                include_commands=self._settings.plugins.include_commands_in_history,
            )
            # 跟随 reply 角色实时的工具调用状态：降级后 L1 守则换回纯文本措辞
            static_system = runtime.static_system(
                "reply", profile.persona, via_tool=self._ai.reply_tool_use
            )
            nicknames = nickname_list_from_history(
                [record_to_turn(r) for r in recent[:-1]]
            )
            runtime_system = runtime_system_prompt(
                group_id,
                date.today().isoformat(),
                nicknames,
                impressions=await self._impressions_for(group_id, runtime),
                commands_enabled=self._settings.plugins.enabled,
                commands_in_history=self._settings.plugins.include_commands_in_history,
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
                expression_hints=expression_hints,
                jargon_hints=jargon_hints,
                repetition_warning=repetition,
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
        self,
        group_id: int,
        msg: NormalizedMessage,
        profile: GroupProfile,
        runtime: GroupRuntime,
        *,
        observe: bool = False,
    ) -> Decision:
        """判定一条消息要不要回。judge 打分 + to_me 识别 + 三道护栏 + 门槛复核。

        observe=True 是观望到点后的二次判定：复用完全相同的护栏与配额路径，
        只去掉两处差异——消息到达时的记账不做第二遍（同一条消息不重复计入
        发言间隔与热闹统计），以及不再为它安排新的观望（每条消息至多一次）。
        """
        # 结构性护栏的记账先于一切判断：即使消息最终不触发回复，
        # 也要计入间隔与热闹统计。观望的二次判定是 45 秒前的旧消息：
        # 不重复记账，但过期时间戳照常清理（热闹程度按当下流量判定）。
        now = time.time()
        window = runtime.recent_msg_times
        if not observe:
            runtime.msgs_since_reply += 1
            window.append(now)
        while window and window[0] < now - 60.0:
            window.popleft()

        if msg.mentioned_me:
            return Decision(should_reply=True, forced=True)

        # 每条普通消息都过一遍 judge：除了打分，还要识别「这条消息是否在对
        # 我说」——正和我聊天的场景里，任何时间窗口都不应该把对话掐断。
        memory = await self._memory.get(group_id)
        recent = memory.model_tail(
            profile.context_size,
            include_commands=self._settings.plugins.include_commands_in_history,
        )
        static_system = runtime.static_system("judge", profile.persona)
        nicknames = nickname_list_from_history([record_to_turn(r) for r in recent[:-1]])
        runtime_system = runtime_system_prompt(
            group_id,
            date.today().isoformat(),
            nicknames,
            impressions=await self._impressions_for(group_id, runtime),
            commands_enabled=self._settings.plugins.enabled,
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
                # 复核失败时手里的终评仍是首评：按首评分数决定是否观望
                self._schedule_observe(
                    group_id, msg, profile, runtime, verdict.score, observe
                )
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
                self._schedule_observe(
                    group_id, msg, profile, runtime, rechecked.score, observe
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
        self._schedule_observe(group_id, msg, profile, runtime, verdict.score, observe)
        return Decision(should_reply=False, score=verdict.score, reason=verdict.reason)

    def _schedule_observe(
        self,
        group_id: int,
        msg: NormalizedMessage,
        profile: GroupProfile,
        runtime: GroupRuntime,
        score: int | None,
        observe: bool,
    ) -> None:
        """观望（wait 的低成本等价物）：差一点点没过门槛的消息先看一会儿再说。

        触发条件（缺一不可）：

        - 终评分（含复评后的最终分数）落在 [门槛 - observe_band, 门槛) 的
          观望带内；observe_band=0 整体关闭；
        - 未被护栏（冷却/发言间隔/热闹静默）直接终止——本方法只在护栏
          检查全部通过、纯因分数不达标而静默的路径上被调用；
        - 该消息（(group_id, message_id) 记账）从未被观望过——二次判定
          仍差一点也不会再安排第三次，杜绝观望循环；
        - 不是观望的二次处理（observe=True 时永不再安排）。

        到点后经 _observe_task 把原消息重新投进该群的串行队列，取届时
        最新的上下文再走一遍完全相同的判定与护栏路径。
        """
        gen = self._settings.generation
        if observe or gen.observe_band <= 0 or score is None:
            return
        threshold = profile.proactivity_threshold
        if not threshold - gen.observe_band <= score < threshold:
            return
        key = (group_id, msg.record.message_id)
        if key in self._observed_once:
            return
        self._observed_once.add(key)
        self._observed_order.append(key)
        while len(self._observed_order) > _OBSERVED_HISTORY_MAX:
            self._observed_once.discard(self._observed_order.popleft())
        logger.info(
            "群 %d 消息 %d 终评 %d 差一点点没过门槛 %d，安排 %.0f 秒后观望重评",
            group_id,
            msg.record.message_id,
            score,
            threshold,
            gen.observe_delay_seconds,
        )
        task = asyncio.create_task(
            self._observe_task(group_id, msg),
            name=f"observe-{group_id}-{msg.record.message_id}",
        )
        self._observe_tasks[key] = task
        task.add_done_callback(lambda _t, k=key: self._observe_tasks.pop(k, None))

    async def _observe_task(self, group_id: int, msg: NormalizedMessage) -> None:
        """观望任务的延时体：到点检查取消条件，满足则重新入队二次判定。"""
        message_id = msg.record.message_id
        try:
            await asyncio.sleep(self._settings.generation.observe_delay_seconds)
            if self._stopping:
                return
            memory = await self._memory.get(group_id)
            exists, answered = _self_reply_after(
                memory.tail(len(memory)), message_id
            )
            if not exists:
                logger.info(
                    "群 %d 消息 %d 观望期间已不在上下文（撤回或被淘汰），取消重评",
                    group_id,
                    message_id,
                )
                return
            if answered:
                logger.info(
                    "群 %d 消息 %d 观望期间已通过其他路径回复，取消重评",
                    group_id,
                    message_id,
                )
                return
            logger.info(
                "群 %d 消息 %d 观望到点，取最新上下文重新判定", group_id, message_id
            )
            await self._enqueue(group_id, msg, observe=True)
        except asyncio.CancelledError:
            # 停机或消息已被处理：观望静默作废，不报错也不二次判定
            logger.debug("群 %d 消息 %d 的观望任务已取消", group_id, message_id)

    # ------------------------------------------------------------ 重试包装

    async def _generate_with_retry(self, fn, *args, **kwargs) -> ReplyDraft | None:
        """带指数退避地调用一个 reply 模型的生成函数（generate/reconsider）。

        尝试次数与首次退避来自 generation.generate_max_attempts /
        generate_retry_base_delay（每次 ×2 倍增）。全部失败返回 None——
        调用方须把它与「模型主动输出空正文」区分开：前者按原计划继续，
        后者才是模型的明确决定。
        """
        gen = self._settings.generation
        attempts = max(int(gen.generate_max_attempts), 1)
        delay = gen.generate_retry_base_delay
        last_exc: Exception | None = None
        for attempt in range(attempts):
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
        delay = estimate_typing_time(
            message, speed, typing_policy_of(self._settings.response_post_process)
        )
        logger.debug(
            "群 %d 第 %d/%d 条预计打字 %.1f 秒", group_id, index, total, delay
        )
        if delay > 0:
            await asyncio.sleep(delay)

    async def _send_with_retry(
        self, group_id: int, message: str | list[dict]
    ) -> int | None:
        """发送一条消息（文本或 OneBot 段数组），返回其 message_id。

        尝试次数与首次退避来自 snowluma.send_max_attempts /
        send_retry_delay_seconds（每次 ×2 倍增）；最后一次失败原样上抛。
        """
        snow = self._settings.snowluma
        attempts = max(int(snow.send_max_attempts), 1)
        delay = snow.send_retry_delay_seconds
        for attempt in range(attempts):
            try:
                return await self._snowluma.send_group_msg(group_id, message)
            except Exception as exc:
                if attempt == attempts - 1:
                    raise
                logger.warning("发送第 %d 次失败：%s", attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2
        return None  # pragma: no cover —— 最后一次失败必抛

    def _self_record(
        self, group_id: int, text: str, *, is_command: bool = False
    ) -> ChatRecord:
        """自己发出的一条消息对应的记忆记录（发送成功后逐条写回）。

        is_command=True 仅命令回复使用（见 _run_command）。"""
        return ChatRecord(
            message_id=self._next_self_message_id(),
            group_id=group_id,
            user_id=self._settings.bot.self_qq,
            nickname=self._settings.bot.self_nickname,
            text=text,
            ts=time.time(),
            is_self=True,
            is_command=is_command,
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
