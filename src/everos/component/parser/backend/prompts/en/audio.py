"""Audio parser prompt: the system message of the transcription call.

The user message is the audio part alone. The system message ends with ``<reply_format>``: the transcription goes in
``{"text": "..."}``.
"""

from __future__ import annotations

__all__ = ["PROMPT_FOR_AUDIO"]

from everos.algo.utils import paragraphs
from everos.component.parser.backend.prompts._reply_format import TEXT_REPLY_FORMAT

_AUDIO_RULES = """This is a voice message sent by a user in a chat application. Transcribe it word by word.

CRITICAL INSTRUCTIONS:
- Output ONLY the transcribed text directly. Do NOT include any preamble or postscript.
- Maintain the same language as the audio. If the audio is in Chinese, output in Simplified Chinese.
- If any of the following situations occur, return '##UNKNOWN':
  1. No audio file is provided
  2. The audio file contains no content
  3. The audio file is corrupted or cannot be parsed"""

PROMPT_FOR_AUDIO = paragraphs(_AUDIO_RULES, TEXT_REPLY_FORMAT)
