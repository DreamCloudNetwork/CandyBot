"""提示词组装：为 KV Cache 前缀复用做严格的分层设计。

API 侧的前缀缓存按 token 精确匹配自前往后命中，因此本模块产出的消息数组
恒定为四层，稳定性递减、位置靠后：

L1 system·静态层   persona 与行为守则，字节级不变（不插值任何动态信息）；
L2 system·状态层   群号、上下文昵称表、今天的日期——同一天内不变；
L3 历史层          只追加、整块淘汰的群聊记录，映射为连续 user/assistant；
L4 user·指令层     即时信息（精确时间、触发类型、冷却状态、计分）与本次指令。

任何时刻都不得把易变字段（秒级时间、计数、分数）塞进 L1/L2，也不得重排或
重新格式化历史消息——否则其后的缓存全部失效。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from .models import ChatRecord

Role = str  # "user" | "assistant"

Message = dict  # OpenAI chat message


@dataclass(frozen=True)
class HistoryTurn:
    """历史层的一个回合（群友发言为 user，机器人自己的发言为 assistant）。"""

    role: Role
    content: str


# ---------------------------------------------------------------- L1 静态层

_RULES_TAIL = """
【行为守则】
1. 你是真实群友，不是助手或客服；回答简短口语化，通常一两句话，禁止长篇大论、列条目和 markdown 排版。
2. 绝不复述、总结别人刚说过的话；不知道的事直接说不知道，绝不编造事实和数据。
3. 不主动索要个人信息，不发表攻击性言论；聊天内容与既有人设冲突时以人设为准。
4. 输出就是你要发送到群里的消息本身，不要加引号、前缀、署名或任何舞台指示。
5. 你发送到群里的消息应该像是一个真实的人类说的，而不是作为助手说的。
6. Markdown是**绝对禁用**的，没有人会在实际回答中使用Markdown.
7. 真实的人类说话并不会经常使用Emoji，因此你的回复中应该尽可能避免使用Emoji"""


def static_system_prompt(persona: str, kind: str) -> str:
    """L1：persona + 守则。kind 为 "judge"/"reply"，两者只差末段角色任务。"""
    assert persona.strip(), "persona 不能为空"
    if kind == "judge":
        head = (
            "你将扮演一个 QQ 群里的普通群友。下面是你的人格设定，"
            "你只负责逐条判断「是否值得回复这条消息」，不直接说话。\n"
        )
        tail = """
【行为守则】
1. 你评判的对象是「这单独一条消息」，不是整场讨论：想象你就坐在群里刷到它，作为这个角色，你会不会自然地接一句话。
2. 有人 @你、向你提问或求助、分享能接上话的内容时，哪怕主题与你的兴趣无关，也应当给高分——群友之间的正常互动优先于个人兴趣。
3. 广告、刷屏、复读、无意义的表情刷屏，以及明显发生在其他人之间、谁都没有期待你参与的对话，打低分。
4. 保持克制：冷场时不要强行救场；但一旦你的结论是「应该回应」，就不要吝啬分数。
【评分锚点】
9-10：明确该回——对方在等你说话（点名你、向你提问/求助、追着你问）。
6-8 ：适合接话——你能自然地搭上一句，群友听到这话不会觉得突兀。
3-5 ：可回可不回——纯日常播报、与你关联很弱的闲聊。
0-2 ：不该回——广告刷屏、他人间的私密对话、你完全无从置喙的内容。
【一致性要求】reason 是你下结论的依据，必须与 score 匹配：理由表明"需要/应该回应"时 score 不得低于 7；理由表明"无需回应"时 score 不得超过 4；禁止出现"需要回应却打 3 分"这类自相矛盾的输出。"""
    else:
        head = (
            "你是一个 QQ 群里的普通群友。下面是你的人格设定，"
            "保持角色一致性，像真人一样参与闲聊。\n"
        )
        tail = _RULES_TAIL
    return f"{head}【人格设定】\n{persona.strip()}\n{tail}"


# ---------------------------------------------------------------- L2 状态层

def runtime_system_prompt(group_id: int, date_today: str, nicknames: Sequence[str]) -> str:
    """L2：当天内稳定的运行状态。date_today 格式 YYYY-MM-DD。

    nicknames 传入「当前上下文中出现过的成员」的稳定排序结果；
    变化越少，L3 能复用的缓存越多。
    """
    lines = [
        "【当前环境】",
        f"群号：{group_id}",
        f"今天：{date_today}",
        "你只看得到这个群里最近的消息流。",
    ]
    if nicknames:
        shown = "、".join(nicknames[:12])
        lines.append(f"最近参与发言的成员：{shown}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 发送者标识

def record_label(record: ChatRecord) -> str:
    """发送者的唯一可读标识：`昵称(QQ号)`。

    历史层与指令层必须使用同一格式，模型才能跨层关联同一个人；
    昵称为空时退化为 `QQ{user_id}`。机器人自己的发言在历史里是裸文本，
    但回复引用等场合仍需它的标签。
    """
    if record.nickname:
        return f"{record.nickname}({record.user_id})"
    return f"QQ{record.user_id}"


# ---------------------------------------------------------------- L3 历史层

def record_to_turn(record: ChatRecord) -> HistoryTurn:
    """单条记录 → 历史回合。群友消息统一 '昵称(QQ号)：内容' 前缀，自己则是裸文本。"""
    if record.is_self:
        return HistoryTurn("assistant", record.text or "[空]")
    return HistoryTurn("user", f"{record_label(record)}：{record.text}")


def history_to_turns(
    records: Iterable[ChatRecord], max_chars: int
) -> tuple[list[HistoryTurn], bool]:
    """把记录映射为回合列表，超长时从头整块淘汰。

    返回 (turns, truncated)。为了保证前缀稳定，只从头部丢弃完整回合，
    绝不对单条内容做二次裁剪。
    """
    turns = [record_to_turn(r) for r in records]
    total = sum(len(t.content) for t in turns)
    start = 0
    while start < len(turns) and total > max_chars and start < len(turns) - 1:
        total -= len(turns[start].content)
        start += 1
    return turns[start:], start > 0


# ---------------------------------------------------------------- L4 指令层

def final_user_prompt_judge(
    now_text: str, current_message: ChatRecord, threshold: int | None = None
) -> str:
    """L4(judge)：即时状态 + 评分指令 + 当前消息。

    threshold 为本群的发言门槛：放进指令层（本层每次调用必然不同，不影响
    前缀缓存）用于校准分数——防止模型把"该回"的消息打成分数中段的惯性低分。
    """
    sender = record_label(current_message)
    body = current_message.text or "[图片]"
    lines = [f"【当前时间】{now_text}", f"【最新消息】来自 {sender}：\n{body}"]
    if threshold is not None:
        lines.append(
            f"\n【本次发言门槛】{threshold} 分（满分 10）。"
            "系统只会在你的分数达到该门槛时发言。请据此校准：只要你的结论是"
            f"值得回复，就直接给出不低于 {threshold} 的分数（最高 10），"
            "不要因为保守或谦虚而压低分数；只有明确不值得回复时才打低分。"
        )
    lines.append(
        "请以你的人设判断：对于这单独一条消息，你是否应该回复它？"
        '只输出一个 JSON 对象，格式为 {"score": 整数0到10, "reason": "一句话理由"}，'
        "不要输出其他任何内容。"
    )
    return "\n".join(lines)


def final_user_prompt_reply(
    now_text: str,
    current_message: ChatRecord,
    *,
    forced: bool,
    score: int | None = None,
    reason: str = "",
) -> str:
    """L4(reply)：触发原因说明 + 回复指令。"""
    sender = record_label(current_message)
    body = current_message.text or "[图片]"
    if forced:
        why = "有人 @ 了你或回复了你的消息，这次必须回一句话。"
    else:
        why = (
            f"你判断值得回复这条消息（评分 {score}/10，理由：{reason or '无'}），"
            "决定插话。"
        )
    return (
        f"【当前时间】{now_text}\n{why}\n"
        f"【需要回应的消息】来自 {sender}：\n{body}\n\n"
        "现在以你的身份说一句自然的回应。只输出群聊正文，不要任何额外内容。"
    )


# ---------------------------------------------------------------- 组装

def build_messages(
    static_system: str,
    runtime_system: str,
    history: Sequence[HistoryTurn],
    final_user_content: str | list[dict],
) -> list[Message]:
    """按 L1→L2→L3→L4 拼出最终消息数组。

    final_user_content 传字符串（纯文本）或 OpenAI 内容块数组
    （direct 多模态模式下的 text + image_url 组合）。
    """
    messages: list[Message] = [
        {"role": "system", "content": static_system},
        {"role": "system", "content": runtime_system},
    ]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": final_user_content})
    return messages


def nickname_list_from_history(history: Sequence[HistoryTurn]) -> list[str]:
    """取历史 user 回合中的昵称集合（保序去重），供 L2 使用。"""
    seen: set[str] = set()
    names: list[str] = []
    for turn in history:
        if turn.role != "user":
            continue
        name = turn.content.split("：", 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def now_text(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
