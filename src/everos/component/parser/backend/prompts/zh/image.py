"""Image parser prompts (Chinese).

Chinese counterpart to ``prompts/en/image.py``. Wire identity is preserved
(``PROMPT_FOR_PICTURE`` + ``PROMPT_FOR_MERGE`` + ``MERGE_USER_PROMPT``) so callers can swap languages
by changing the import: each system message ends with the shared English ``<reply_format>`` block, and
``MERGE_USER_PROMPT`` is the English one, the markup holding no language.
"""

from __future__ import annotations

__all__ = ["MERGE_USER_PROMPT", "PROMPT_FOR_MERGE", "PROMPT_FOR_PICTURE"]

from everos.algo.utils import paragraphs
from everos.component.parser.backend.prompts._reply_format import TEXT_REPLY_FORMAT
from everos.component.parser.backend.prompts.en.image import MERGE_USER_PROMPT

_PICTURE_RULES = """请阅读这张图片,像自然阅读一样返回它的内容。

关键指令:
- 仅输出提取出来的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。
- 输出格式为 Markdown。
- 图片中的所有文字必须**逐字**提取。**绝不**总结、改写、缩略或修改任何文字。**绝不**以任何方式改变原内容。

要求:
1. 精确、完整地提取所有文字,**逐字**保留原始格式与版面。**不要**跳过或总结任何文字,不论长短。
2. 表格转为 HTML 格式
3. 公式转为 LaTeX 格式
4. 如有图形或图表,用文字描述其内容与数据
5. 对于非文字元素(照片、插图等),详细描述其主要内容与关键元素
6. 保留原始结构与层级"""

PROMPT_FOR_PICTURE = paragraphs(_PICTURE_RULES, TEXT_REPLY_FORMAT)


_MERGE_RULES = """下方是同一张长图片不同片段的 OCR 结果。相邻片段之间有重叠区域。
请智能地把这些片段合并成完整的文本。

关键指令:
- 仅输出合并后的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。

要求:
1. 识别并去除重叠区域里的重复内容
2. 保证文本的连续性与完整性
3. 保留原始格式与结构"""

PROMPT_FOR_MERGE = paragraphs(_MERGE_RULES, TEXT_REPLY_FORMAT)
