"""Audio parser prompts (Chinese).

Chinese counterpart to ``prompts/en/audio.py``, same constant and same message split: a system message ending with
the shared English ``<reply_format>`` block, the audio part alone as the user message.
"""

from __future__ import annotations

__all__ = ["PROMPT_FOR_AUDIO"]

from everos.algo.utils import paragraphs
from everos.component.parser.backend.prompts._reply_format import TEXT_REPLY_FORMAT

_AUDIO_RULES = """这是用户在聊天应用里发送的一段语音消息。请逐字转写。

关键指令:
- 仅输出转写出的文本,不要任何开场白或结束语。
- 保持与音频相同的语言。如果音频是中文,输出简体中文。
- 如果出现以下任一情况,返回 '##UNKNOWN':
  1. 没有提供音频文件
  2. 音频文件没有内容
  3. 音频文件损坏或无法解析"""

PROMPT_FOR_AUDIO = paragraphs(_AUDIO_RULES, TEXT_REPLY_FORMAT)
