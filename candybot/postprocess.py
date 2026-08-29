"""输出层拟人化后处理：拆条、打字延迟、错别字与敷衍兜底。

LLM 生成的整段回复在这里被加工成更接近真人的发送形态：

- split_reply_segments：剥掉括号旁白后按中英文标点拆条，颜文字与 emoji
  序列绝不拆断；超出 max_split 时把剩余句子并入最后一条。
- estimate_typing_time：按字符类型估算打字秒数（中文/全角 0.3 秒/字、
  英文数字 0.15 秒/字、emoji 与颜文字各 1 秒、单字回复 3 倍），再乘全局
  typing_speed 倍率。
- TypoGenerator：基于 pypinyin 的同音字/整词替换生成错别字，附带供更正
  消息使用的正确词。整词替换用 pypinyin 自带的最大正向匹配分词，不引入
  额外分词依赖。多音字（如「银行」的行）按 typo_polyphone_mode 取词内
  读音替换或整体跳过，读音不确定时绝不替换。
- process_reply：串起以上各步的总入口。回复过长（单条超 max_length，
  按显示字数计：emoji/颜文字块算 1 字）或清洗后无内容时，改为从敷衍池
  随机抽一条发送。

错别字只是面向群友的表层噪音：process_reply 同时返回与逐条发送内容下标
对齐的无错字分条原文（memory_segments），发送链路每成功发出一条就用对应
的一条写回记忆，既避免错字污染 L3 历史，也保证历史顺序与群里实际发言
顺序一致。
"""

from __future__ import annotations

import logging
import random
import re
import threading
import unicodedata
from dataclasses import dataclass

from .models import ResponsePostProcessSettings

logger = logging.getLogger(__name__)

# 基础 emoji 字符范围。刻意不含几何符号（U+25A0~25FF）等普通文本符号区，
# 颜文字 (=^ω^=)、(・ω・) 与箭头 ↑ 不会误伤。
EMOJI_CHARS = (
    "\U0001F000-\U0001FAFF"  # 象形符号/表情/补充表情各区
    "\u2600-\u27BF"          # 杂项符号与装饰符（☀ ✅ ❤）
    "\u2B00-\u2BFF"          # 星形与常见聊天箭头（⭐ ⭕ ⬛）
)
# emoji 序列：国旗按成对区域指示符算一个，键帽含数字本身，
# 其余为单个基础字符 + 变体选择符/肤色修饰 + ZWJ 组合的完整序列。
EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF]{2}"
    "|[1-9#*]\uFE0F?\u20E3"
    f"|[{EMOJI_CHARS}](?:[\uFE0F\U0001F3FB-\U0001F3FF]|\u200D[{EMOJI_CHARS}])*"
)

# 颜文字：括号形态（内部至少含一个非中文/非字母数字/非空白字符，因此
# 「（笑）」这类纯文字旁白不会被误保护），以及无括号的符号串（^^、T_T 风格）。
KAOMOJI_RE = re.compile(
    r"[(\[（【][^()\[\]（）【】]*?"
    r"[^()\[\]（）【】一-龥a-zA-Z0-9\s]"
    r"[^()\[\]（）【】]*?[)\]）】]"
    r"|[▼▽・ᴥω･﹏^><≧≦￣｀´∀ヮДд︿﹀へ｡ﾟ╥╯╰︶︹•⁄]{2,15}"
)

# 括号旁白/内心戏：全角（）与半角() 成对才删，内部不再嵌套括号。
_NARRATION_RES = (re.compile(r"\([^()]*\)"), re.compile(r"（[^（）]*）"))
# 嵌套括号（如「（（笑））」）一次剥不净，内层删完后剩下的空括号对——
# 带着它们发言会原样发出「（）」这样的残段，一并清掉。
_EMPTY_PAREN_RES = (re.compile(r"\(\s*\)"), re.compile(r"（\s*）"))


def _strip_narration(text: str) -> str:
    """剥离括号旁白，再收敛清理嵌套写法剥完后剩下的空括号对。

    旁白正则保持既有的单遍语义（「a(b(c)d)e」这类真内容最多损失最内一层，
    不会被二次剥成「ae」）；空括号对则循环清除直到稳定，覆盖「（（（哭）））」
    这类一次剥不净的多层嵌套。每轮都严格缩短文本，循环必然收敛。
    """
    for narration_re in _NARRATION_RES:
        text = narration_re.sub("", text)
    while True:
        cleaned = text
        for pattern in _EMPTY_PAREN_RES:
            cleaned = pattern.sub("", cleaned)
        if cleaned == text:
            return cleaned
        text = cleaned

# 拆条的断点标点：中英句子成分隔符与句末标点（英文句号刻意排除，避免拆坏
# 「3.14」「e.g.」之类文本），连续标点算一个断点。
_SENTENCE_DELIMS = "。！？!?，,；;…\n\r"
_DELIM_RUN_RE = re.compile(f"[{re.escape(_SENTENCE_DELIMS)}]+")

# 打字时长估算的单字耗时（秒）：中文/全角、英文与数字等其他、emoji/颜文字。
_CJK_CHAR_TIME = 0.3
_ENGLISH_CHAR_TIME = 0.15
_SPECIAL_UNIT_TIME = 1.0
# 单条打字延迟的封顶：估算时长乘 typing_speed 后也可能大得离谱（长回复
# 配高倍率会把所在群的串行队列堵几分钟，后续消息的判断与回复全部排队），
# 封顶兜住最坏情况，拟人节奏不受影响（正常配置远达不到上限）。
_MAX_TYPING_DELAY_SECONDS = 60.0

# 先 emoji 后颜文字地整体扫描（两类的字符集互不重叠，顺序不影响结果）。
_SPECIAL_TOKEN_RE = re.compile(f"{EMOJI_RE.pattern}|{KAOMOJI_RE.pattern}")


def display_len(text: str) -> int:
    """按「显示字数」计数：每个 emoji 序列/颜文字块整体算 1 个字。

    ZWJ 组合 emoji（如 👨‍👩‍👧 = 5 个码点）直接 len() 会虚高，导致
    max_length 超长兜底对 emoji 密集的回复误伤。
    """
    return len(_SPECIAL_TOKEN_RE.sub("\u4e00", text))

# 占位符用私用区字符包裹序号：不含任何标点与括号，绝不会在拆条/去旁白阶段
# 被拆断或误删，最后统一还原。
_PH_HEAD = "\ue500"
_PH_TAIL = "\ue501"
_PH_RE = re.compile(f"{_PH_HEAD}(\\d+){_PH_TAIL}")


# ---------------------------------------------------------------- 拆条


def _protect_special(text: str) -> tuple[str, list[str]]:
    """把 emoji 序列与颜文字替换成占位符，返回（处理后文本, 原文列表）。"""
    originals: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        originals.append(match.group())
        return f"{_PH_HEAD}{len(originals) - 1}{_PH_TAIL}"

    return _SPECIAL_TOKEN_RE.sub(_sub, text), originals


def _restore_special(text: str, originals: list[str]) -> str:
    return _PH_RE.sub(lambda m: originals[int(m.group(1))], text)


def _strong_punctuation(delim_run: str, keep: bool) -> str:
    """句末标点处理：! ?（全半角 !?！？）按配置保留，其余一律去掉。"""
    if not keep:
        return ""
    return "".join(ch for ch in delim_run if ch in "!?！？")


def split_reply_segments(
    text: str,
    max_split: int = 3,
    *,
    keep_strong_punctuation: bool = True,
) -> list[str]:
    """把整段回复拆成至多 max_split 条更像人打的短消息。

    处理顺序：保护颜文字/emoji → 去掉括号旁白 → 按标点分句（句末标点默认
    去掉，! ? 由 keep_strong_punctuation 决定）→ 超出条数上限时把剩余句子
    并入最后一条 → 还原占位符。不做长度截断：拆出的条若仍超长，由
    process_reply 走敷衍兜底。返回空列表表示没有可发送的内容。
    """
    protected, originals = _protect_special(text)
    cleaned = _strip_narration(protected)

    # 切成 (正文, 紧随其后的标点串) 序列；空正文（连续断点）直接丢弃
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in _DELIM_RUN_RE.finditer(cleaned):
        chunk = cleaned[cursor : match.start()]
        if chunk.strip():
            parts.append((chunk, match.group()))
        cursor = match.end()
    tail = cleaned[cursor:]
    if tail.strip():
        parts.append((tail, ""))
    if not parts:
        return []

    # 分组：前 max_split-1 组各一句，其余句子并进最后一组
    max_split = max(1, max_split)
    if len(parts) <= max_split:
        groups = [[part] for part in parts]
    else:
        groups = [[part] for part in parts[: max_split - 1]]
        groups.append(parts[max_split - 1 :])

    segments: list[str] = []
    for group in groups:
        # 组内相邻句子保留原有标点（它们现在处于句中）；仅整条末尾的
        # 句末标点按策略处理
        body = "".join(content + delim for content, delim in group[:-1])
        last_content, last_delim = group[-1]
        raw = body + last_content + _strong_punctuation(last_delim, keep_strong_punctuation)
        piece = _restore_special(raw, originals)
        piece = re.sub(r"\s+", " ", piece).strip()
        if piece:
            segments.append(piece)
    return segments


# ---------------------------------------------------------------- 打字延迟


def estimate_typing_time(text: str, speed: float = 1.0) -> float:
    """估算「打出这段字」所需秒数，供发送间隔制造打字感。

    中文/全角字符 0.3 秒、英文数字等半角字符 0.15 秒；emoji 序列与颜文字
    各按整块 1 秒；没有任何特殊块的单字符回复按 3 倍计（「嗯」不是秒回的）；
    总时长乘全局 typing_speed 倍率，speed <= 0 表示关闭延迟，
    结果封顶在 _MAX_TYPING_DELAY_SECONDS。
    """
    # `not speed > 0` 而非 `speed <= 0`：连 NaN（异常配置漏网时）也按关闭处理
    if not speed > 0:
        return 0.0
    if not text:
        return 0.0

    special_units = 0
    plain_chars = 0
    total = 0.0
    positions = {m.span() for m in _SPECIAL_TOKEN_RE.finditer(text)}
    covered = [False] * len(text)
    for start, end in positions:
        special_units += 1
        for i in range(start, end):
            covered[i] = True
    for i, ch in enumerate(text):
        if covered[i]:
            continue
        plain_chars += 1
        total += (
            _CJK_CHAR_TIME
            if unicodedata.east_asian_width(ch) in ("W", "F")
            else _ENGLISH_CHAR_TIME
        )
    if special_units == 0 and plain_chars == 1:
        total *= 3
    total += special_units * _SPECIAL_UNIT_TIME
    return min(total * speed, _MAX_TYPING_DELAY_SECONDS)


# ---------------------------------------------------------------- 错别字

# pypinyin 的词典反查表（拼音 → 字 / 拼音串 → 词）构建一次约 0.5 秒，
# 进程内共享；未配置任何错字率时完全不会触碰（见 process_reply 的短路），
# 启用时由 bot.start() 在后台线程预热，避免首条回复卡事件循环。
_CHAR_INDEX: dict[str, list[str]] | None = None
_WORD_INDEX: dict[tuple[str, ...], list[str]] | None = None
_CHAR_READINGS: dict[str, str] | None = None
_POLYPHONES: frozenset[str] | None = None
_WORD_READINGS: dict[str, tuple[str, ...]] | None = None
_INDEX_LOCK = threading.Lock()

IndexTables = tuple[
    dict[str, str],  # 字 → 首选拼音（tone3）
    dict[str, list[str]],  # 拼音 → 常用同音字
    dict[tuple[str, ...], list[str]],  # 拼音串 → 同音词
    frozenset[str],  # 多音字集合（词典里有不止一个读音的常用字）
    dict[str, tuple[str, ...]],  # 词 → 词内逐字拼音（phrases_dict 首选读音）
]


def ensure_indexes() -> IndexTables:
    """惰性构建：字→拼音、拼音→常用同音字、拼音串→同音词、多音字、词→词内读音。

    候选字限定在 pypinyin 内置词库出现过的「常用字」内，避免从四万字全量
    表里选出「趑」「魋」之类的生僻字穿帮。
    """
    global _CHAR_INDEX, _WORD_INDEX, _CHAR_READINGS, _POLYPHONES, _WORD_READINGS
    if (
        _CHAR_INDEX is not None
        and _WORD_INDEX is not None
        and _CHAR_READINGS is not None
        and _POLYPHONES is not None
        and _WORD_READINGS is not None
    ):
        return _CHAR_READINGS, _CHAR_INDEX, _WORD_INDEX, _POLYPHONES, _WORD_READINGS
    with _INDEX_LOCK:
        if (
            _CHAR_INDEX is None
            or _WORD_INDEX is None
            or _CHAR_READINGS is None
            or _POLYPHONES is None
            or _WORD_READINGS is None
        ):
            from pypinyin.contrib.tone_convert import tone_to_tone3
            from pypinyin.phrases_dict import phrases_dict
            from pypinyin.pinyin_dict import pinyin_dict

            common = {ch for word in phrases_dict for ch in word}
            char_index: dict[str, list[str]] = {}
            readings: dict[str, str] = {}
            polyphones: set[str] = set()
            for codepoint, pys in pinyin_dict.items():
                ch = chr(codepoint)
                if ch not in common:
                    continue
                options = pys.split(",")
                if len(options) > 1:
                    polyphones.add(ch)
                first = tone_to_tone3(options[0])
                readings[ch] = first
                char_index.setdefault(first, []).append(ch)
            word_index: dict[tuple[str, ...], list[str]] = {}
            word_readings: dict[str, tuple[str, ...]] = {}
            for word, pys in phrases_dict.items():
                if len(word) < 2 or len(pys) != len(word):
                    continue
                key = tuple(tone_to_tone3(p[0]) for p in pys)
                word_index.setdefault(key, []).append(word)
                word_readings[word] = key
            _CHAR_INDEX, _WORD_INDEX, _CHAR_READINGS = char_index, word_index, readings
            _POLYPHONES, _WORD_READINGS = frozenset(polyphones), word_readings
    return _CHAR_READINGS, _CHAR_INDEX, _WORD_INDEX, _POLYPHONES, _WORD_READINGS


def _segment_words(text: str) -> list[str]:
    """用 pypinyin 自带的最大正向匹配词典分词（不引入额外分词依赖）。"""
    from pypinyin.seg.simpleseg import seg

    return list(seg(text))


def _is_han(text: str) -> bool:
    return bool(text) and "\u4e00" <= text[0] <= "\u9fff"


class TypoGenerator:
    """基于拼音的同音错别字生成器（可注入 rng 以便测试固定随机种子）。"""

    def __init__(
        self,
        *,
        error_rate: float = 0.05,
        tone_error_rate: float = 0.3,
        word_replace_rate: float = 0.2,
        polyphone_mode: str = "word_reading",
        rng: random.Random | None = None,
    ):
        self._error_rate = error_rate
        self._tone_error_rate = tone_error_rate
        self._word_replace_rate = word_replace_rate
        self._polyphone_mode = polyphone_mode
        # 默认加密安全随机源（与 ai.py 的掷点约定一致）；测试可传入
        # random.Random(seed) 获得可复现输出。
        self._rng = rng if rng is not None else random.SystemRandom()

    @property
    def active(self) -> bool:
        """两个替换概率都为 0 时不产生任何错字，可整体跳过。"""
        return self._error_rate > 0 or self._word_replace_rate > 0

    def create_typo(self, text: str) -> tuple[str, list[str]]:
        """生成同音错字版文本，返回（错字文本, 被替换掉的正确字/词列表）。

        多字词按 word_replace_rate 尝试整体换成同音词；单字（含词内字）按
        error_rate 尝试换同音字，其中 tone_error_rate 的概率改用声调标错的
        拼音找候选（打拼音时漏点/错点声调的典型手滑）。找不到合法候选字时
        保持原字，绝不产出无意义生僻字。

        多音字处理（如「银行」的行读 háng 而首选音是 xíng）由 polyphone_mode
        决定："word_reading" 取词典词内读音照常替换；"skip" 则多音字整体跳过。
        无论哪种模式，无法确定读音时（多音字单独成词、非词条多字块中的多音
        字）一律不替换，绝不产出读音对不上的「假同音」错字。
        """
        if not self.active:
            return text, []
        readings, char_index, word_index, polyphones, word_readings = ensure_indexes()
        rng = self._rng
        out: list[str] = []
        corrections: list[str] = []
        for token in _segment_words(text):
            if not _is_han(token):
                out.append(token)
                continue
            entry = word_readings.get(token)
            contextual = entry is not None and len(entry) == len(token)
            py_tuple = entry if contextual else tuple(
                readings.get(ch, "") for ch in token
            )
            # 字读音可信 = 取自词内读音表，或该字在词典里根本没有第二个读音
            trustworthy = [
                bool(py) and (contextual or ch not in polyphones)
                for ch, py in zip(token, py_tuple)
            ]
            if self._polyphone_mode == "skip":
                trustworthy = [
                    ok and ch not in polyphones
                    for ok, ch in zip(trustworthy, token)
                ]
            replaced = False
            # 整词替换：拼音串完全相同的其他词（同音词都是词库里的高频真词）
            if (
                len(token) >= 2
                and all(trustworthy)
                and rng.random() < self._word_replace_rate
            ):
                candidates = [w for w in word_index.get(py_tuple, ()) if w != token]
                if candidates:
                    out.append(rng.choice(candidates))
                    corrections.append(token)
                    replaced = True
            if replaced:
                continue
            # 逐字同音替换
            chars: list[str] = []
            for ch, py, ok in zip(token, py_tuple, trustworthy):
                typo = self._try_typo_char(ch, py, char_index) if ok else None
                chars.append(typo or ch)
                if typo:
                    corrections.append(ch)
            out.append("".join(chars))
        # 去重保序：同一个字出现多次只更正一次
        return "".join(out), list(dict.fromkeys(corrections))

    def _try_typo_char(
        self, ch: str, py: str, char_index: dict[str, list[str]]
    ) -> str | None:
        rng = self._rng
        if rng.random() >= self._error_rate:
            return None
        base = py[:-1] if py[-1].isdigit() else py
        tone = int(py[-1]) if py[-1].isdigit() else 0
        pool: list[str] = []
        # 先掷声调错误：命中则从错声调的同音字里选（选不到退回本音）
        if self._tone_error_rate > 0 and rng.random() < self._tone_error_rate:
            wrong_tone = rng.choice([t for t in (1, 2, 3, 4) if t != tone])
            pool = char_index.get(f"{base}{wrong_tone}", [])
        if not pool:
            pool = char_index.get(py, [])
        candidates = [c for c in pool if c != ch]
        return rng.choice(candidates) if candidates else None


# ---------------------------------------------------------------- 总入口


@dataclass(frozen=True)
class ProcessedReply:
    """一次后处理的产出：最终发送的消息序列 + 逐条对齐的记忆原文。

    messages 可能带错别字；memory_segments 是同样拆条、未加错字的原文，
    下标与 messages 一一对应——bot 每成功发出一条，就把对应的一条单独写回
    记忆，让连发期间穿插进来的他人消息落在真实的时间位置上，而不是被挤到
    整段回复之前。correction 非空时是一条「＊正确词」更正，由 bot 用
    OneBot v11 reply 消息段引用最后一条正文发送（snowluma 的
    send_group_msg 支持段数组并返回 message_id，见 snowluma.py）。
    错字与更正只是表层噪音，不进入 L3 历史消耗模型注意力。
    """

    messages: list[str]
    memory_segments: list[str]
    correction: str | None = None

    @property
    def memory_text(self) -> str:
        """拆条合并后的无错字原文（\n 连接），供整体查看/测试比对。"""
        return "\n".join(self.memory_segments)


def _lazy_reply(settings: ResponsePostProcessSettings, rng: random.Random) -> ProcessedReply:
    """敷衍兜底：不发送原文，从配置池里随机抽一条。"""
    piece = rng.choice(settings.lazy_replies)
    logger.debug("回复过长或无有效内容，改用敷衍回复：%s", piece)
    return ProcessedReply([piece], [piece])


def process_reply(
    text: str,
    settings: ResponsePostProcessSettings,
    rng: random.Random | None = None,
) -> ProcessedReply:
    """把一条生成的回复整理成拟人化的发送计划（见模块 docstring）。"""
    if not settings.enabled:
        return ProcessedReply([text], [text])
    if rng is None:
        rng = random.SystemRandom()

    segments = split_reply_segments(
        text,
        settings.max_split,
        keep_strong_punctuation=settings.keep_strong_punctuation,
    )
    if not segments:
        return _lazy_reply(settings, rng)
    if any(display_len(seg) > settings.max_length for seg in segments):
        return _lazy_reply(settings, rng)
    logger.debug("回复拆为 %d 条：%s", len(segments), segments)

    generator = TypoGenerator(
        error_rate=settings.typo_error_rate,
        tone_error_rate=settings.typo_tone_error_rate,
        word_replace_rate=settings.typo_word_replace_rate,
        polyphone_mode=settings.typo_polyphone_mode,
        rng=rng,
    )
    if not generator.active:
        return ProcessedReply(list(segments), list(segments))

    messages: list[str] = []
    corrections: list[str] = []
    for seg in segments:
        typoed, fixed = generator.create_typo(seg)
        messages.append(typoed)
        corrections.extend(fixed)
    # 更正消息只带「＊正确词」文本：引用最后一条正文的 reply 消息段由 bot
    # 拼（需要 send_group_msg 返回的 message_id，见 snowluma.py）。
    correction: str | None = None
    if corrections and rng.random() < settings.typo_correction_probability:
        correction = f"＊{rng.choice(corrections)}"
    return ProcessedReply(messages, list(segments), correction)
