"""表情包（最小版，任务 C）：收集「表情包类图片」+ 小概率跟发。

收集：bot._on_event 在消息入记忆后调用 collect()，按 normalize 给出的
sticker_flags（与 record.images 下标对齐）把命中的图片存到
`data/stickers/<群号>/<内容指纹>.<ext>`，并在 candy.db 的 sticker 表登记
使用统计（use_count / last_used_time）。识别来源按 multimodal 模式而异
（见 normalize.py）：direct 用视觉模型入库评估的 is_sticker 判定，describe
按总结文本关键词，placeholder 按「图片尺寸小」启发式（解析文件头取宽高，
不引入额外依赖）。全局数量超 stickers.max_count（默认 64）时替换最久未
使用的条目——删表记录的同时删图片文件。

发送：成功发出一段文字回复之后（bot._maybe_send_sticker），每条按
stickers.send_probability（默认 0.05）掷点，命中且该群收藏非空时随机挑
一张，以 OneBot v11 image 消息段（file:// 绝对路径 URI）跟发；不做模型
选择（模型参与留作后续迭代）。发送成功后写回一条 is_self 的 ChatRecord
占位「[表情包]」，让模型在历史里知道自己发过图——路径与 base64 都不进
历史。

限制：发送依赖 OneBot 服务端能读到本机表情包文件（SnowLuma 与 CandyBot
同机或共享磁盘）；端点不支持时 image 段发送失败只记错误日志，收集与
文字回复不受影响。
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

from .database import CandyDatabase, StickerEntry, image_fingerprint
from .models import STICKER_SUMMARY_KEYWORDS_DEFAULT, ChatRecord, Settings

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


class StickerStore:
    """表情包文件与 sticker 表的管理者（收集、上限替换、抽发、记账）。

    ai/bot 只在事件处理与发送链路里现取现用；配置经 settings_provider
    回调读取，热重载后自动取到最新值。表情包根目录在构造时定死
    （`<data_dir>/stickers`，data_dir 本就需重启生效）。
    """

    def __init__(
        self,
        root: Path,
        db: CandyDatabase,
        settings_provider: Callable[[], Settings],
    ):
        self._root = root
        self._db = db
        self._settings = settings_provider

    # ------------------------------------------------------------ 收集

    async def collect(self, record: ChatRecord, flags: Sequence[bool]) -> int:
        """按 sticker_flags 收藏该消息里命中表情包类的图片，返回本次入库数。

        flags 与 record.images 下标对齐；False、原图缺失（空串）与解析
        失败的槽位一律跳过。任何异常原样上抛，由调用方按辅助能力处理。
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
            mime, data = parsed
            sha = image_fingerprint(data_url)
            rel_path = f"{record.group_id}/{sha}{_MIME_SUFFIXES.get(mime, '.png')}"
            path = self._root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            ts = record.ts if record.ts > 0 else time.time()
            inserted = await self._db.insert_sticker(
                record.group_id, sha, rel_path, record.summary_of(index) or "", ts
            )
            if not inserted:
                continue  # 同群同图只收藏一次
            saved += 1
            logger.info(
                "群 %d 收藏表情包：%s（总结：%s）",
                record.group_id,
                rel_path,
                record.summary_of(index) or "无",
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

    # ------------------------------------------------------------ 抽发

    async def pick_for_send(self, group_id: int, rng: random.Random) -> StickerEntry | None:
        """从该群收藏中随机抽一张待跟发；收藏为空返回 None。"""
        entries = await self._db.load_stickers(group_id)
        if not entries:
            return None
        return rng.choice(entries)

    def image_segment(self, entry: StickerEntry) -> dict:
        """OneBot v11 image 消息段：file 取表情包文件的 file:// 绝对 URI。"""
        file_uri = self.absolute_path(entry).resolve().as_uri()
        return {"type": "image", "data": {"file": file_uri}}

    async def mark_used(self, entry: StickerEntry) -> None:
        """发送成功后记账：使用数 +1、刷新最近使用时间（LRU 依据）。"""
        await self._db.touch_sticker(entry.id, time.time())

    # ------------------------------------------------------------ 内部

    def absolute_path(self, entry: StickerEntry) -> Path:
        return self._root / entry.path
