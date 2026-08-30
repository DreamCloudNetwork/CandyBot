"""表情包（任务 C 最小版 + 任务 2 v2）：审核收藏 + 描述入库 + 小概率跟发。

收集：bot._on_event 在消息入记忆后调用 collect()，按 normalize 给出的
sticker_flags（与 record.images 下标对齐）把命中的图片存到
`data/stickers/<群号>/<内容指纹>.<ext>`，并在 candy.db 的 sticker 表登记
使用统计（use_count / last_used_time）。识别来源按 multimodal 模式而异
（见 normalize.py）：direct 用视觉模型入库评估的 is_sticker 判定，describe
按总结文本关键词，placeholder 按「图片尺寸小」启发式（解析文件头取宽高，
不引入额外依赖）。全局数量超 stickers.max_count（默认 64）时替换最久未
使用的条目——删表记录与描述 meta（sticker_meta 级联）的同时删图片文件。

收藏审核与描述（stickers.moderation_enabled，默认开）：vision 可用时对
候选图过一次 ai.assess_sticker——不合格（截图/广告等）不收藏；通过则同
一次调用产出的「描述 + 情绪」入 sticker_meta 表。direct 模式的合并入库
评估已在 normalize 一次调用里给出结论（sticker_metas 传入），不再重复
请求；vision 未配置或审核失败时维持现状收藏、无 meta（只参与随机兜底）。

发送：成功发出一段文字回复之后（bot._maybe_send_sticker），每条按
stickers.send_probability（默认 0.05）掷点。命中后选图按
stickers.select_mode：random（默认）从收藏随机挑一张；smart 取该群有
meta 的条目为候选（≤smart_max_candidates、最久未使用优先轮换、乱序
编号），连同最近聊天与刚发出的回复交给 ai.pick_sticker 按语境选一张，
模型可以回答「不发」（本次跟发作罢），调用失败退回随机抽选。选中后以
OneBot v11 image 消息段跟发。发送成功后写回一条 is_self 的 ChatRecord
占位「[表情包]」，让模型在历史里知道自己发过图——路径与 base64 都不进
历史。

图片怎么交给 OneBot 端由 stickers.send_mode 决定（见 image_segment）：
base64（默认）把图片字节内嵌进请求，SnowLuma 与 CandyBot 不同机也能发；
http 发 events_server 上注册的只读外链（`GET /stickers/<群号>/<指纹>`，
文件名即内容指纹、无从枚举），基址 stickers.http_base_url 需对 SnowLuma
可达；file 仍是本机绝对路径 URI，只在端点能读到本机磁盘（同机或共享
磁盘）时可用。端点不支持时 image 段发送失败只记错误日志，收集与文字
回复不受影响。
"""

from __future__ import annotations

import base64
import binascii
import logging
import random
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from .database import (
    CandyDatabase,
    StickerEntry,
    StickerMetaEntry,
    image_fingerprint,
)
from .models import (
    STICKER_SUMMARY_KEYWORDS_DEFAULT,
    ChatRecord,
    Settings,
    StickerSettings,
)

if TYPE_CHECKING:  # 仅注解与鸭子类型协议引用，静态层不依赖 LLM 层
    from .ai import StickerAssessment

logger = logging.getLogger(__name__)

# 写回记忆的占位文本：模型据此知道「这张图是我发的」，不带路径与数据
STICKER_RECORD_TEXT = "[表情包]"

# describe 模式的文本启发式：视觉模型的总结里出现任一配置关键词
# （stickers.summary_keywords，忽略大小写整词包含）即视为表情包类；
# 默认词表见 models.STICKER_SUMMARY_KEYWORDS_DEFAULT。显式配置空列表
# 表示该启发式永不命中。编译结果按词组缓存复用。
_STICKER_SUMMARY_RES: dict[tuple[str, ...], re.Pattern[str]] = {}


def _sticker_summary_re(keywords: Sequence[str]) -> re.Pattern[str] | None:
    if not keywords:
        return None
    key = tuple(keywords)
    regex = _STICKER_SUMMARY_RES.get(key)
    if regex is None:
        regex = re.compile("|".join(re.escape(k) for k in key), re.IGNORECASE)
        _STICKER_SUMMARY_RES[key] = regex
    return regex


def is_sticker_by_summary(
    summary: str | None,
    keywords: Sequence[str] = STICKER_SUMMARY_KEYWORDS_DEFAULT,
) -> bool:
    """按视觉模型给的一句话总结判断是否表情包类（describe 模式来源）。"""
    regex = _sticker_summary_re(keywords)
    return bool(summary) and regex is not None and regex.search(summary) is not None

_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}

# 表情包文件的 HTTP 路由前缀，以及收藏命名规则的严格校验（群号纯数字 +
# 内容指纹 64 位小写十六进制 + 已知图片后缀）：发送侧拼 URL 与事件服务
# 供图侧共用同一套规则，指纹即访问凭证（无从枚举，也不接受任何越界路径）。
STICKER_URL_PREFIX = "/stickers"
_STICKER_GROUP_RE = re.compile(r"^\d{1,20}$")
_STICKER_FILE_RE = re.compile(r"^[0-9a-f]{64}\.(?:png|jpg|gif|webp|bmp)$")
# 反向映射供图用；剔除 image/jpg 别名（注册 MIME 只有 image/jpeg），
# 否则同一个 .jpg 后缀会被别名覆盖成非标准的 MIME。
_STICKER_CONTENT_TYPES = {
    suffix: mime
    for mime, suffix in _MIME_SUFFIXES.items()
    if mime != "image/jpg"
}


def sticker_url(base_url: str, rel_path: str) -> str:
    """HTTP 模式下表情包的外链：{基址}/stickers/{群号}/{指纹}.{ext}。"""
    return f"{base_url.rstrip('/')}{STICKER_URL_PREFIX}/{quote(rel_path)}"


def sticker_content_type(rel_path: str) -> str | None:
    """按表情包文件后缀给 Content-Type；后缀不认识返回 None。"""
    return _STICKER_CONTENT_TYPES.get(Path(rel_path).suffix.lower())


def resolve_sticker_file(root: Path, group_part: str, name_part: str) -> Path | None:
    """事件服务供图的路径解析：群号与文件名不合收藏命名规则返回 None。

    只做命名校验（顺带排除 `..` 等越界写法），不检查文件是否存在。
    """
    if not _STICKER_GROUP_RE.match(group_part) or not _STICKER_FILE_RE.match(name_part):
        return None
    path = (root / group_part / name_part).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def parse_data_url(data_url: str) -> tuple[str, bytes] | None:
    """data URL → (mime, 原始字节)；非 base64 数据 URL 或解码失败返回 None。"""
    if not data_url.startswith("data:"):
        return None
    header, sep, payload = data_url.partition(",")
    if not sep or not payload:
        return None
    mime = header[5:].split(";", 1)[0].strip().lower()
    if "base64" not in header.lower():
        return None
    try:
        data = base64.b64decode(payload)
    except (binascii.Error, ValueError):
        return None
    return mime or "image/png", data


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """从文件头解析图片宽高（PNG/GIF/JPEG/WebP/BMP），解析不出返回 None。

    只为「尺寸小」启发式服务，刻意不做完整解码（不引入图像库依赖）；
    格式不认识时宁可返回 None（不收集）也不猜测。
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return (
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
        )
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return (
            int.from_bytes(data[6:8], "little"),
            int.from_bytes(data[8:10], "little"),
        )
    if data[:2] == b"BM" and len(data) >= 26:
        return (
            abs(int.from_bytes(data[18:22], "little", signed=True)),
            abs(int.from_bytes(data[22:26], "little", signed=True)),
        )
    if data[:2] == b"\xff\xd8":  # JPEG：扫描 SOF 段（含常见扩展编号）
        return _jpeg_dimensions(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        return _webp_dimensions(data)
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    i = 2
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in sof_markers:
            return (
                int.from_bytes(data[i + 7 : i + 9], "big"),
                int.from_bytes(data[i + 5 : i + 7], "big"),
            )
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            i += 2  # 无长度字段的独立标记
        else:
            if i + 4 > len(data):
                return None
            i += 2 + int.from_bytes(data[i + 2 : i + 4], "big")
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    fourcc = data[12:16]
    if fourcc == b"VP8X":
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return w, h
    if fourcc == b"VP8 ":  # 有损：帧头里 14bit 宽 + 14bit 高
        return (
            int.from_bytes(data[26:28], "little") & 0x3FFF,
            int.from_bytes(data[28:30], "little") & 0x3FFF,
        )
    if fourcc == b"VP8L":  # 无损：1 字节签名 0x2F 后跟 4 字节宽高位打包
        if data[21] != 0x2F:
            return None
        bits = int.from_bytes(data[22:26], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


# placeholder 模式「尺寸小」启发式的默认边长上限（像素，即参数提取前的写死
# 值，可用 stickers.max_side_px 覆盖）：QQ 表情包普遍远小于截图与照片，
# 较长边不超过该值才收集；宽高解析不出的格式一律不收集。
_STICKER_MAX_SIDE = 512


def is_small_image(data_url: str, max_side_px: int = _STICKER_MAX_SIDE) -> bool:
    """placeholder 模式的尺寸启发式：宽高可解析且较长边 ≤ max_side_px。"""
    parsed = parse_data_url(data_url)
    if parsed is None:
        return False
    dims = image_dimensions(parsed[1])
    if dims is None or dims[0] <= 0 or dims[1] <= 0:
        return False
    return max(dims) <= max_side_px


def _shuffle_in_place(items: list[Any], rng: random.Random) -> None:
    """就地 Fisher–Yates 洗牌：只用 rng.random()，不依赖 stdlib 特有方法
    （测试注入的确定性随机源只实现 random/choice 等少数接口）。"""
    for i in range(len(items) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        items[i], items[j] = items[j], items[i]


class StickerStore:
    """表情包文件与 sticker 表的管理者（收集、审核、上限替换、抽发、记账）。

    ai/bot 只在事件处理与发送链路里现取现用；配置经 settings_provider
    回调读取，热重载后自动取到最新值。表情包根目录在构造时定死
    （`<data_dir>/stickers`，data_dir 本就需重启生效）。ai_provider 回调
    返回当前 AIClient（热重载会重建客户端，与 LearningService 同一注入
    手法）；只按鸭子类型用其 assess_sticker / pick_sticker 两个方法，
    不配置（测试等场合）则审核与 smart 选图都静默退回现状。
    """

    def __init__(
        self,
        root: Path,
        db: CandyDatabase,
        settings_provider: Callable[[], Settings],
        ai_provider: Callable[[], Any] | None = None,
    ):
        self._root = root
        self._db = db
        self._settings = settings_provider
        self._ai = ai_provider

    # ------------------------------------------------------------ 收集

    async def collect(
        self,
        record: ChatRecord,
        flags: Sequence[bool],
        metas: Sequence["StickerAssessment | None"] = (),
    ) -> int:
        """按 sticker_flags 收藏该消息里命中表情包类的图片，返回本次入库数。

        flags 与 record.images 下标对齐；False、原图缺失（空串）与解析
        失败的槽位一律跳过。

        metas 同样与 images 下标对齐，携带 direct 模式合并入库评估已给出的
        审核结论（免重复调用 vision）；该槽位无现成结论而
        stickers.moderation_enabled 开着时，这里单独调 ai.assess_sticker
        审核：不合格不收藏（DEBUG 带理由），通过则「描述 + 情绪」入
        sticker_meta 表。vision 未配置或审核失败维持现状收藏、无 meta，
        该条目后续只参与随机兜底。任何异常原样上抛，由调用方按辅助能力
        处理（审核调用自身的失败在这里就地消化）。
        """
        st = self._settings().stickers
        if not st.enabled or not flags:
            return 0
        saved = 0
        for index, flag in enumerate(flags):
            if not flag or index >= len(record.images):
                continue
            data_url = record.images[index]
            if not data_url:
                continue  # 原图已被保留期回收（收图当次不会发生，防御）
            parsed = parse_data_url(data_url)
            if parsed is None:
                logger.warning(
                    "群 %d 表情包收集跳过：data URL 无法解码（槽位 %d）",
                    record.group_id,
                    index,
                )
                continue
            meta = metas[index] if index < len(metas) else None
            if meta is None and st.moderation_enabled:
                meta = await self._assess_sticker(record.group_id, data_url)
            if meta is not None and not meta.acceptable:
                logger.debug(
                    "群 %d 表情包审核未通过，不收藏：%s",
                    record.group_id,
                    meta.description or "模型未给出理由",
                )
                continue
            mime, data = parsed
            sha = image_fingerprint(data_url)
            rel_path = f"{record.group_id}/{sha}{_MIME_SUFFIXES.get(mime, '.png')}"
            path = self._root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            ts = record.ts if record.ts > 0 else time.time()
            sticker_id = await self._db.insert_sticker(
                record.group_id, sha, rel_path, record.summary_of(index) or "", ts
            )
            if sticker_id is None:
                continue  # 同群同图只收藏一次
            saved += 1
            if meta is not None:
                await self._db.insert_sticker_meta(
                    sticker_id, meta.description, meta.emotion, ts
                )
            logger.info(
                "群 %d 收藏表情包：%s（总结：%s%s）",
                record.group_id,
                rel_path,
                record.summary_of(index) or "无",
                f"，描述：{meta.description}【{meta.emotion}】" if meta else "",
            )
        if saved:
            evicted = await self._db.evict_stickers_over(st.max_count)
            for rel_path in evicted:
                try:
                    (self._root / rel_path).unlink(missing_ok=True)
                except OSError:
                    # 记录已删，文件残留不影响正确性：如实记日志
                    logger.warning("删除被淘汰表情包文件失败：%s", rel_path, exc_info=True)
                logger.info(
                    "表情包收藏超过上限 %d，已替换最久未使用：%s", st.max_count, rel_path
                )
        return saved

    async def _assess_sticker(
        self, group_id: int, data_url: str
    ) -> "StickerAssessment | None":
        """收集路径上的独立审核调用；任何「拿不到结论」都返回 None。

        未注入 AI 客户端、vision 未配置或输出不可解析（assess_sticker 自行
        记日志返回 None）记 DEBUG；网络/端点异常记 WARNING 后同样退回无
        meta 收藏——审核是辅助能力，绝不拦住收集本身。
        """
        if self._ai is None:
            return None
        ai = self._ai()
        assess = getattr(ai, "assess_sticker", None) if ai is not None else None
        if assess is None:
            return None  # 未注入或假客户端不带审核能力：按无结论处理
        try:
            assessment = await assess(data_url)
        except Exception:
            logger.warning(
                "群 %d 表情包审核调用失败，维持收藏、无 meta", group_id, exc_info=True
            )
            return None
        if assessment is None:
            logger.debug(
                "群 %d 表情包审核未产出结论（vision 未配置或输出不可解析），按无 meta 收藏",
                group_id,
            )
        return assessment

    # ------------------------------------------------------------ 抽发

    async def pick_for_send(self, group_id: int, rng: random.Random) -> StickerEntry | None:
        """从该群收藏中随机抽一张待跟发；收藏为空返回 None。"""
        entries = await self._db.load_stickers(group_id)
        if not entries:
            return None
        return rng.choice(entries)

    async def pick_for_send_smart(
        self, group_id: int, context_text: str, rng: random.Random
    ) -> StickerEntry | None:
        """smart 选图（任务 2）：掷点命中后把候选交给模型按语境挑，可以不发。

        候选取该群有描述 meta（审核通过入库）的收藏：按「最久未使用优先」
        截取 ≤smart_max_candidates 张再乱序编号（轮换避免总用同一批；无
        meta 的条目不参与 smart 选择、仍可在随机兜底里被抽到）。

        模型明确选择不发 → INFO 带理由并返回 None；无 meta 候选、无 AI
        客户端、选图调用或解析失败（WARNING）都退回一次随机抽选，与现状
        一致；随机池本身为空时返回 None（调用方按收藏为空处理）。
        """
        st = self._settings().stickers
        candidates: list[StickerMetaEntry] = await self._db.load_stickers_with_meta(
            group_id
        )
        ai = self._ai() if self._ai is not None else None
        if ai is None or not candidates:
            if not candidates:
                logger.debug("群 %d 无带描述 meta 的收藏，退回随机抽选", group_id)
            return await self.pick_for_send(group_id, rng)
        candidates = candidates[: st.smart_max_candidates]
        _shuffle_in_place(candidates, rng)
        entries = [(item.description, item.emotion) for item in candidates]
        try:
            index, reason = await ai.pick_sticker(context_text, entries)
        except Exception:
            logger.warning(
                "群 %d smart 选图失败，退回随机抽选", group_id, exc_info=True
            )
            return await self.pick_for_send(group_id, rng)
        if index is None:
            logger.info(
                "群 %d 模型判断语境不合，本次不跟发表情包：%s",
                group_id,
                reason or "未给出理由",
            )
            return None
        return candidates[index].entry

    def image_segment(self, entry: StickerEntry) -> dict:
        """OneBot v11 image 消息段：按 stickers.send_mode 引用图片。

        - base64（默认）：`base64://<内嵌数据>`，图片字节随发送请求交给
          SnowLuma，跨机也能发，代价是每次多传一份文件（表情包本身很小）；
        - http：`{http_base_url}/stickers/<群号>/<指纹>.<ext>`，由本进程事件
          服务的只读路由供图（见 events_server.EventsServer），基址需对
          SnowLuma 可达；
        - file：`file://` 绝对路径 URI，要求端点能读到本机磁盘（同机或共享
          磁盘）。

        读取失败（文件被外部删除等）原样上抛，由发送链路按辅助能力处理。
        """
        st = self._settings().stickers
        return {"type": "image", "data": {"file": self.image_reference(entry, st)}}

    def image_reference(self, entry: StickerEntry, st: StickerSettings) -> str:
        """按 send_mode 生成 image 段的 file 字段值（见 image_segment）。"""
        if st.send_mode == "http":
            if not st.http_base_url:
                # 配置校验会拦住这种组合；热重载窗口期按更稳的 base64 兜底
                logger.warning(
                    "stickers.send_mode=http 但未配 http_base_url，本次退回 base64"
                )
            else:
                return sticker_url(st.http_base_url, entry.path)
        if st.send_mode == "file":
            return self.absolute_path(entry).resolve().as_uri()
        data = self.absolute_path(entry).read_bytes()
        return "base64://" + base64.b64encode(data).decode("ascii")

    async def mark_used(self, entry: StickerEntry) -> None:
        """发送成功后记账：使用数 +1、刷新最近使用时间（LRU 依据）。"""
        await self._db.touch_sticker(entry.id, time.time())

    # ------------------------------------------------------------ 内部

    def absolute_path(self, entry: StickerEntry) -> Path:
        return self._root / entry.path
