"""Audio parser — ASR via multimodal LLM.

Sends the audio bytes as a single ``image_url`` data URI (OpenRouter / Gemini accepts
audio MIMEs through the same content-part shape) as the user message, the rules as
the system message, and reads the transcription from the reply's ``text``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everos.algo.llm import image_url_part_from_bytes
from everos.algo.types import Modality, ParsedContent
from everos.algo.utils import InvalidInputError
from everos.component.parser.backend._replies import _call_llm_for_text
from everos.component.parser.backend.prompts.en import PROMPT_FOR_AUDIO

if TYPE_CHECKING:
    from everos.algo.llm import LLMClient
    from everos.algo.types import RawFile

__all__ = ["aparse"]

logger = logging.getLogger(__name__)


_MIME_MAP: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "amr": "audio/amr",
    "aiff": "audio/aiff",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Parse an audio ``RawFile`` into ``ParsedContent`` (transcription).

    Raises:
        InvalidInputError: If ``raw_file`` has no content or an unsupported extension.
        LLMOutputError: If the reply is not a valid ``{"text": "..."}`` object.
    """
    if not raw_file.content:
        raise InvalidInputError(
            "RawFile.content is empty, there is nothing to transcribe: pass the audio bytes in content"
        )

    extension = raw_file.extension.lower().lstrip(".")
    mime = _MIME_MAP.get(extension)
    if mime is None:
        raise InvalidInputError(
            f"RawFile.extension {raw_file.extension!r} is not a supported audio extension: "
            f"set it to one of {sorted(_MIME_MAP)}"
        )

    logger.debug(
        "parsing audio extension=%s bytes=%d", extension, len(raw_file.content)
    )
    text = await _call_llm_for_text(
        llm,
        system=PROMPT_FOR_AUDIO,
        user=[image_url_part_from_bytes(raw_file.content, mime)],
        operator="parser audio",
    )
    return ParsedContent(text=text, modality=Modality.AUDIO, mime=raw_file.mime or mime)
