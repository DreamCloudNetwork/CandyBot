"""OneBot v11 群消息事件 → 内部归一化消息。

text/face/at/reply/image 等 segment 转为内部 ChatRecord；
@ 或回复机器人时标记 mentioned_me；图片按配置的 multimodal.mode 处理：
- placeholder：文本只留 ``[图片]`` 占位符（默认），但 base64 仍随记录落盘；
- direct：下载图片转 base64 data URL，附在 record.images 供直传模型，
  入库时由视觉模型判定后续对话继续展示原图还是只用总结；
- describe：下载后调用视觉模型转文字描述，写进正文；base64 同样随记录落盘。

三种模式还会为每张下载成功的图产出 sticker_flags（是否像表情包类，
供表情包收集，见 stickers.py）。
"""

from __future__ import annotations

import base64
import logging

import aiohttp

from .models import (
    IMAGE_STATE_SHOW,
    IMAGE_STATE_SUMMARIZED,
    ChatRecord,
    MultimodalSettings,
    NormalizedMessage,
    validate_request_url,
)
from .stickers import is_small_image, is_sticker_by_summary

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = ("png", "jpg", "jpeg", "gif", "webp", "bmp")
_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=15)
_MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _sender_nickname(event: dict) -> str:
    sender = event.get("sender") or {}
    nickname = sender.get("card") or sender.get("nickname")
    if isinstance(nickname, str) and nickname.strip():
        return nickname.strip()
    uid = event.get("user_id", 0)
    try:
        return f"QQ{int(uid)}"
    except (TypeError, ValueError):
        return "匿名"


def _extract_image_urls(segment_data: dict) -> list[str]:
    """从 image 段取可下载的 http(s) 地址；拒绝 file:// 与无 host 的路径。"""
    urls: list[str] = []
    for key in ("url", "file"):
        value = segment_data.get(key)
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        if lowered.startswith(("http://", "https://")):
            urls.append(value)
        elif lowered.startswith("base64://"):
            continue  # 内联数据，无需下载，direct 模式直接用不了的情况走占位符
    return urls


async def _download_as_data_url(
    session: aiohttp.ClientSession, url: str
) -> str | None:
    """下载图片并编码为 data URL；任何失败都只记日志返回 None。"""
    try:
        validate_request_url(url)
    except ValueError as exc:
        logger.warning("跳过不安全的图片地址：%s", exc)
        return None
    try:
        async with session.get(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            data = await resp.read()
    except Exception as exc:  # 网络/HTTP 错误一律降级为占位符
        logger.warning("下载图片失败 %s：%s", url, exc)
        return None
    if len(data) > _MAX_IMAGE_BYTES:
        logger.warning("图片过大，放弃下载：%s (%d bytes)", url, len(data))
        return None
    mime = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
    if not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def normalize_group_message(
    event: dict,
    *,
    self_qq: int,
    multimodal: MultimodalSettings,
    find_by_message_id,
    http_session: aiohttp.ClientSession | None = None,
    describe_image=None,
    assess_image=None,
) -> NormalizedMessage | None:
    """把一条 group 消息事件转为 NormalizedMessage。

    find_by_message_id: Callable[[int], Awaitable[ChatRecord | None]]，通常传
    GroupMemory.find_by_message_id（库里有全量历史，早于启动的消息也可引用）；
    describe_image: 仅 describe 模式需要，Callable[[str dataurl], Awaitable[str]]；
    assess_image: 仅 direct 模式需要，Callable[[str dataurl], Awaitable[ImageAssessment]]，
    用于入库时判定该图后续继续展示原图还是只保留总结。
    """
    if event.get("post_type") != "message" or event.get("message_type") != "group":
        return None
    try:
        group_id = int(event["group_id"])
        user_id = int(event["user_id"])
        message_id = int(event["message_id"])
    except (KeyError, TypeError, ValueError):
        logger.debug("忽略缺少关键字段的事件：%r", event.get("post_type"))
        return None
    if user_id == self_qq:
        return None  # 自己发出的回显

    segments = event.get("message")
    if isinstance(segments, str):  # 个别实现给 CQ 码字符串
        segments = [{"type": "text", "data": {"text": segments}}]
    if not isinstance(segments, list):
        segments = []

    mentioned_me = False
    parts: list[str] = []
    image_urls: list[str] = []

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_type = str(seg.get("type", ""))
        data = seg.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        if seg_type == "text":
            text = str(data.get("text", ""))
            if text.strip():
                parts.append(text.strip())
        elif seg_type == "at":
            target = str(data.get("qq", ""))
            if target in (str(self_qq), "all"):
                if target == str(self_qq):
                    mentioned_me = True
                    parts.append("@糖糖")
                else:
                    parts.append("@全体成员")
            elif target.isdigit():
                parts.append(f"@QQ{target}")
        elif seg_type == "reply":
            ref_text, is_self_ref = await _resolve_reply(
                data, find_by_message_id, self_qq
            )
            if ref_text:
                parts.append(ref_text)
                mentioned_me = mentioned_me or is_self_ref
        elif seg_type == "image":
            if multimodal.mode != "describe":
                parts.append("[图片]")
            image_urls.extend(_extract_image_urls(data))
        elif seg_type == "face":
            parts.append(f"[表情{data.get('id', '')}]".replace("]", "]"))
        elif seg_type in ("record", "video"):
            parts.append("[语音]" if seg_type == "record" else "[视频]")
        elif seg_type == "json":
            parts.append("[卡片消息]")
        elif seg_type == "forward":
            parts.append("[合并转发]")
        # 其余类型静默丢弃

    text = "\n".join(p for p in parts if p)

    # 图片增强：direct 下载转 data URL 并入库判定；describe 转文字并入正文；
    # placeholder 不给模型看图。三种模式下载到的 base64 都随记忆落盘备查。
    # sticker_flags 与 images 下标对齐：标记每张图是否像「表情包类」，供
    # bot 的表情包收集（来源分模式：direct 用入库评估的 is_sticker 判定，
    # describe 按总结文本关键词，placeholder 按「尺寸小」启发式）。
    images: tuple[str, ...] = ()
    image_states: tuple[str, ...] = ()
    image_summaries: dict[int, str] | None = None
    sticker_flags: tuple[bool, ...] = ()
    if image_urls and multimodal.download_media and http_session is not None:
        data_urls: list[str] = []
        descriptions: list[str] = []
        describe_flags: list[bool] = []  # describe 模式：与 data_urls 一一对应
        for url in image_urls[:4]:  # 至多取 4 张，防刷屏
            data_url = await _download_as_data_url(http_session, url)
            if data_url is None:
                continue
            data_urls.append(data_url)
            if multimodal.mode == "describe":
                if describe_image is not None:
                    try:
                        desc = await describe_image(data_url)
                    except Exception as exc:
                        logger.warning("图片转述失败：%s", exc)
                        desc = None
                else:
                    desc = None
                if desc:
                    descriptions.append(desc)
                describe_flags.append(is_sticker_by_summary(desc))

        if multimodal.mode == "describe":
            images = tuple(data_urls)
            sticker_flags = tuple(describe_flags)
            for desc in descriptions:
                text += f"\n[图片：{desc}]"
            if not descriptions:
                text += "\n[图片]"
        elif multimodal.mode == "direct":
            images = tuple(data_urls)
            states: list[str] = []
            summaries: dict[int, str] = {}
            flags: list[bool] = []
            for index, data_url in enumerate(images):
                summary: str | None = None
                keep_raw = True
                if assess_image is not None:
                    try:
                        assessment = await assess_image(data_url)
                        summary = assessment.summary
                        keep_raw = assessment.keep_raw
                        sticker_is_sticker = assessment.is_sticker
                    except Exception as exc:
                        logger.warning("图片入库评估失败，默认保留原图：%s", exc)
                        sticker_is_sticker = is_small_image(data_url)
                else:
                    # 未配置 vision：没有入库评估结论可复用，退回尺寸启发式
                    sticker_is_sticker = is_small_image(data_url)
                state = IMAGE_STATE_SHOW
                # 判定无需继续展示原图：能总结就转为总结，否则保守保留原图
                if not keep_raw and summary:
                    state = IMAGE_STATE_SUMMARIZED
                if summary:
                    summaries[index] = summary
                states.append(state)
                flags.append(sticker_is_sticker)
            image_states = tuple(states)
            image_summaries = summaries or None
            sticker_flags = tuple(flags)
            if not images:
                text = text.replace("[图片]", "").strip()
                text = (text + "\n[图片]").strip()
        else:  # placeholder：一律只展示占位符，图本身仍要写进本地记忆
            images = tuple(data_urls)
            sticker_flags = tuple(is_small_image(url) for url in data_urls)

    if not text.strip() and not images:
        return None

    record = ChatRecord(
        message_id=message_id,
        group_id=group_id,
        user_id=user_id,
        nickname=_sender_nickname(event),
        text=text.strip(),
        ts=float(event.get("time") or 0),
        images=images,
        image_states=image_states,
        image_summaries=image_summaries,
    )
    return NormalizedMessage(
        record=record, mentioned_me=mentioned_me, sticker_flags=sticker_flags
    )


async def _resolve_reply(
    reply_data: dict, find_by_message_id, self_qq: int
) -> tuple[str | None, bool]:
    """构造回复引用文本，返回 (文本, 是否引用了机器人自己的消息)。"""
    try:
        ref_id = int(reply_data.get("id"))
    except (TypeError, ValueError):
        return None, False
    try:
        referenced = await find_by_message_id(ref_id)
    except Exception:  # 记忆层异常不该影响消息解析
        referenced = None
    if referenced is not None:
        if referenced.is_self:
            return "[回复糖糖]", True
        snippet = referenced.text.replace("\n", " ")[:50]
        ref_label = (
            f"{referenced.nickname}({referenced.user_id})"
            if referenced.nickname
            else f"QQ{referenced.user_id}"
        )
        return f"[回复 {ref_label}：{snippet}]", False
    # 记忆里找不到（可能早于启动），只能靠 user_id 判断
    ref_user = str(reply_data.get("user_id", ""))
    if ref_user.isdigit():
        is_self = int(ref_user) == self_qq
        label = "糖糖" if is_self else f"QQ{ref_user}"
        return f"[回复 {label}]", is_self
    return "[回复消息]", False
