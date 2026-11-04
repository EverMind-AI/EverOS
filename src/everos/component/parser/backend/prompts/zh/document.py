"""Document parser prompts (Chinese).

Chinese counterpart to ``prompts/en/document.py``: PDF / Office (after the
LibreOffice → PDF conversion) share ``PROMPT_FOR_FILE``; the HTML parser
uses ``PROMPT_FOR_HTML``. Same constants and same message split as the English set: each system message ends with
the shared English ``<reply_format>`` block; ``HTML_USER_PROMPT`` is the English one, the markup holding no language.
"""

from __future__ import annotations

__all__ = ["HTML_USER_PROMPT", "PROMPT_FOR_FILE", "PROMPT_FOR_HTML"]

from everos.algo.utils import paragraphs
from everos.component.parser.backend.prompts._reply_format import TEXT_REPLY_FORMAT
from everos.component.parser.backend.prompts.en.document import HTML_USER_PROMPT

_FILE_RULES = """请阅读这份文档,像自然阅读一样返回它的内容。

关键指令:
- 仅输出提取出来的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。
- 输出格式为 Markdown。
- 所有文字必须**逐字**提取。**绝不**总结、改写、缩略或修改任何文字。**绝不**以任何方式改变原内容。即便看上去有错也保持原样复现。

要求:
1. 精确、完整地从每一页提取所有内容。不要跳过或合并任何内容。如果一页里包含多个并排放置(左右排列)的表格或区块,它们是**独立**的表格 —— 每一个都作为单独的表格提取,各自带标题和表头
2. 忽略页眉与页脚
3. 表格转为 HTML 格式
4. 公式转为 LaTeX 格式
5. 如有图形或图表,用文字描述其内容与数据
6. 对于非文字元素(照片、插图等),详细描述其主要内容与关键元素
7. 保留原始结构与层级"""

PROMPT_FOR_FILE = paragraphs(_FILE_RULES, TEXT_REPLY_FORMAT)


_HTML_RULES = """请阅读这段从网页上抽取出的 HTML 内容,返回主要正文。

关键指令:
- 仅输出提取出来的内容,不要任何开场白或结束语。
- 保持与原内容相同的语言。如果无法判断语言,使用英文。
- 输出格式为 Markdown。
- 所有文字必须**逐字**提取。**绝不**总结、改写、缩略或修改任何文字。

要求:
1. 提取主要正文(文章主体、核心文本)。忽略非内容元素,如导航、广告、Cookie 提示、侧边栏文字、页脚链接
2. 按网页的自然阅读顺序输出内容,保留原始顺序
3. 表格保留 HTML 表格格式
4. 公式转为 LaTeX 格式
5. 保留原始结构与层级(标题、段落、列表)
6. 对于图片,如有 alt 文本或描述则描述其内容"""

PROMPT_FOR_HTML = paragraphs(_HTML_RULES, TEXT_REPLY_FORMAT)
