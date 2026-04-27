"""
Text processing utility module

Provides general-purpose utility functions for text processing, including smart truncation, formatting, and other features.
"""

from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass


class TokenType(Enum):
    """Token type enumeration"""

    CJK_CHAR = "cjk_char"  # CJK characters
    ENGLISH_WORD = "english_word"  # English word
    CONTINUOUS_NUMBER = "continuous_number"  # Continuous numbers
    PUNCTUATION = "punctuation"  # Punctuation
    WHITESPACE = "whitespace"  # Whitespace characters
    OTHER = "other"  # Other characters


@dataclass
class Token:
    """Text Token"""

    type: TokenType
    content: str
    start_pos: int
    end_pos: int
    score: float = 0.0


@dataclass
class TokenConfig:
    """Token configuration"""

    cjk_char_score: float = 1.0
    english_word_score: float = 1.5
    continuous_number_score: float = 0.8
    punctuation_score: float = 0.5
    whitespace_score: float = 0.3
    other_score: float = 0.5


class SmartTextParser:
    """Smart text parser

    Capable of distinguishing different types of tokens, supports configurable score calculation,
    provides left-to-right traversal and intelligent truncation based on total score.
    """

    def __init__(self, config: TokenConfig = None):
        """Initialize parser

        Args:
            config: Token configuration, use default if None
        """
        self.config = config or TokenConfig()

        # CJK character ranges
        self._cjk_ranges = [
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Extension A
            (0x20000, 0x2A6DF),  # CJK Extension B
            (0x2A700, 0x2B73F),  # CJK Extension C
            (0x2B740, 0x2B81F),  # CJK Extension D
            (0x2B820, 0x2CEAF),  # CJK Extension E
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0xAC00, 0xD7AF),  # Hangul Syllables
        ]

    def _is_cjk_char(self, char: str) -> bool:
        """Check if character is a CJK character"""
        if not char:
            return False
        code = ord(char)
        return any(start <= code <= end for start, end in self._cjk_ranges)

    def _is_english_char(self, char: str) -> bool:
        """Check if character is an English character"""
        return char.isalpha() and ord(char) < 128

    def _is_punctuation(self, char: str) -> bool:
        """Check if character is punctuation"""
        # Common punctuation characters
        punctuation_chars = set('.,!?;:"\'()[]{}+-*/%=<>@#$&|~`^_\\/')

        return char in punctuation_chars or (
            0x2000 <= ord(char) <= 0x206F  # General Punctuation
            or 0x3000 <= ord(char) <= 0x303F  # CJK Symbols and Punctuation
            or 0xFF00 <= ord(char) <= 0xFFEF  # Fullwidth ASCII and halfwidth Katakana
        )

    def parse_tokens(self, text: str, max_score: float = None) -> List[Token]:
        """Parse text into a list of Tokens

        Args:
            text: Text to parse
            max_score: Maximum score limit, stop parsing early when this score is reached

        Returns:
            List[Token]: List of Tokens
        """
        if not text:
            return []

        tokens = []
        current_score = 0.0
        i = 0
        text_len = len(text)

        while i < text_len:
            char = text[i]
            start_pos = i

            # Handle CJK characters
            if self._is_cjk_char(char):
                token = Token(
                    type=TokenType.CJK_CHAR,
                    content=char,
                    start_pos=start_pos,
                    end_pos=i + 1,
                    score=self.config.cjk_char_score,
                )
                tokens.append(token)
                current_score += token.score
                i += 1

                # Check if early truncation is needed
                if max_score is not None and current_score > max_score:
                    # Remove the last added token as it exceeds the limit
                    tokens.pop()
                    break

            # Handle English words
            elif self._is_english_char(char):
                word_end = i
                while word_end < text_len and (
                    self._is_english_char(text[word_end]) or text[word_end] in "'-"
                ):
                    word_end += 1

                token = Token(
                    type=TokenType.ENGLISH_WORD,
                    content=text[i:word_end],
                    start_pos=start_pos,
                    end_pos=word_end,
                    score=self.config.english_word_score,
                )
                tokens.append(token)
                current_score += token.score
                i = word_end

                # Check if early truncation is needed
                if max_score is not None and current_score > max_score:
                    # Remove the last added token as it exceeds the limit
                    tokens.pop()
                    break

            # Handle continuous numbers
            elif char.isdigit():
                num_end = i
                while num_end < text_len and (
                    text[num_end].isdigit() or text[num_end] in ".,"
                ):
                    num_end += 1

                token = Token(
                    type=TokenType.CONTINUOUS_NUMBER,
                    content=text[i:num_end],
                    start_pos=start_pos,
                    end_pos=num_end,
                    score=self.config.continuous_number_score,
                )
                tokens.append(token)
                current_score += token.score
                i = num_end

                # Check if early truncation is needed
                if max_score is not None and current_score > max_score:
                    # Remove the last added token as it exceeds the limit
                    tokens.pop()
                    break

            # Handle punctuation
            elif self._is_punctuation(char):
                token = Token(
                    type=TokenType.PUNCTUATION,
                    content=char,
                    start_pos=start_pos,
                    end_pos=i + 1,
                    score=self.config.punctuation_score,
                )
                tokens.append(token)
                current_score += token.score
                i += 1

                # Check if early truncation is needed
                if max_score is not None and current_score > max_score:
                    # Remove the last added token as it exceeds the limit
                    tokens.pop()
                    break

            # Handle whitespace
            elif char.isspace():
                # Merge consecutive whitespace characters
                space_end = i
                while space_end < text_len and text[space_end].isspace():
                    space_end += 1

                token = Token(
                    type=TokenType.WHITESPACE,
                    content=text[i:space_end],
                    start_pos=start_pos,
                    end_pos=space_end,
                    score=self.config.whitespace_score,
                )
                tokens.append(token)
                current_score += token.score
                i = space_end

                # Check if early truncation is needed
                if max_score is not None and current_score > max_score:
                    # Remove the last added token as it exceeds the limit
                    tokens.pop()
                    break

            # Handle other characters
            else:
                token = Token(
                    type=TokenType.OTHER,
                    content=char,
                    start_pos=start_pos,
                    end_pos=i + 1,
                    score=self.config.other_score,
                )
                tokens.append(token)
                current_score += token.score
                i += 1

                # Check if early truncation is needed
                if max_score is not None and current_score > max_score:
                    # Remove the last added token as it exceeds the limit
                    tokens.pop()
                    break

        return tokens

    def calculate_total_score(self, tokens: List[Token]) -> float:
        """Calculate total score of token list

        Args:
            tokens: List of Tokens

        Returns:
            float: Total score
        """
        return sum(token.score for token in tokens)

    def smart_truncate_by_score(
        self,
        text: str,
        max_score: float,
        suffix: str = "...",
        enable_fallback: bool = True,
    ) -> str:
        """Smartly truncate text based on score

        Args:
            text: Text to truncate
            max_score: Maximum allowed score
            suffix: Suffix to append after truncation
            enable_fallback: Whether to enable fallback mode, fall back to character length truncation if parsing fails

        Returns:
            str: Truncated text
        """
        if not text:
            return text or ""

        if max_score <= 0:
            return text  # Maintain backward compatibility, return original text if limit <= 0

        try:
            # First parse the full text
            all_tokens = self.parse_tokens(text)

            if not all_tokens:
                return text

            # Calculate actual score, no truncation needed if within limit
            total_score = self.calculate_total_score(all_tokens)
            if total_score <= max_score:
                return text

            # Use full tokens for truncation calculation
            tokens = all_tokens

            # Need truncation, find appropriate position
            current_score = 0.0
            truncate_pos = len(text)

            for token in tokens:
                if current_score + token.score > max_score:
                    # If it's an English word or continuous number and the overflow is small, allow full inclusion to avoid breaking
                    if (
                        token.type
                        in [TokenType.ENGLISH_WORD, TokenType.CONTINUOUS_NUMBER]
                        and current_score + token.score
                        <= max_score * 1.05  # Allow up to 5% overflow
                        and current_score > 0
                    ):  # Must have other tokens already, cannot exceed on first token
                        current_score += token.score
                        truncate_pos = token.end_pos
                    else:
                        truncate_pos = token.start_pos
                    break
                current_score += token.score
                truncate_pos = token.end_pos

            # If truncation is needed
            if truncate_pos < len(text):
                result = text[:truncate_pos].rstrip()
                return result + suffix if result else text

            return text

        except Exception as e:
            # Fallback mode: use simple character length truncation if parsing fails
            if enable_fallback:
                # Estimate truncation length: assume average 1 point per character
                estimated_length = int(max_score * 0.8)  # Conservative estimate
                if len(text) <= estimated_length:
                    return text

                # Simple character-based truncation, avoid breaking in the middle of words
                truncate_pos = estimated_length

                # Look backward for a suitable truncation point (whitespace or punctuation)
                for i in range(
                    min(estimated_length + 10, len(text) - 1),
                    max(estimated_length - 10, 0),
                    -1,
                ):
                    if text[i].isspace() or text[i] in '.,!?;:':
                        truncate_pos = i + 1
                        break

                result = text[:truncate_pos].rstrip()
                return result + suffix if result else text
            else:
                # Raise exception if fallback is disabled
                raise e

    def get_text_analysis(self, text: str) -> Dict[str, Any]:
        """Get text analysis result

        Args:
            text: Text to analyze

        Returns:
            Dict: Dictionary containing various statistics
        """
        tokens = self.parse_tokens(text)

        # Count tokens by type
        type_counts = {token_type: 0 for token_type in TokenType}
        type_scores = {token_type: 0.0 for token_type in TokenType}

        for token in tokens:
            type_counts[token.type] += 1
            type_scores[token.type] += token.score

        return {
            "total_tokens": len(tokens),
            "total_score": self.calculate_total_score(tokens),
            "type_counts": {t.value: count for t, count in type_counts.items()},
            "type_scores": {t.value: score for t, score in type_scores.items()},
            "tokens": tokens,
        }


def smart_truncate_text(
    text: str,
    max_count: int,
    chinese_weight: float = 1.0,
    english_word_weight: float = 1.0,
    suffix: str = "...",
) -> str:
    """
    Smartly truncate text based on word/character count

    Uses the new SmartTextParser for more accurate token parsing and score calculation.
    English words count as one unit, Chinese characters count as one unit, with different weights assignable.

    Args:
        text: Text to truncate
        max_count: Maximum count (total after weight accumulation)
        chinese_weight: Weight for Chinese characters, default 1.0
        english_word_weight: Weight for English words, default 1.0
        suffix: Suffix to add when truncating, default "..."

    Returns:
        str: Truncated text

    Examples:
        >>> smart_truncate_text("Hello World 你好世界", 4)
        'Hello World 你好...'  # 2 English words + 2 Chinese characters = 4
        >>> smart_truncate_text("Hello World 你好世界", 4, chinese_weight=0.5)
        'Hello World 你好世界'  # 2 English words + 4*0.5 Chinese characters = 4
    """
    if not text or max_count <= 0:
        return text or ""

    if not isinstance(text, str):
        text = str(text)

    # Use the new smart parser for truncation
    config = TokenConfig(
        cjk_char_score=chinese_weight,
        english_word_score=english_word_weight,
        continuous_number_score=english_word_weight,  # Use English word weight for numbers
        punctuation_score=0.0,  # Punctuation not counted, maintain backward compatibility
        whitespace_score=0.0,  # Whitespace not counted, maintain backward compatibility
        other_score=0.0,  # Other characters not counted, maintain backward compatibility
    )

    parser = SmartTextParser(config)
    return parser.smart_truncate_by_score(text, max_count, suffix)


def clean_whitespace(text: str) -> str:
    """
    Clean extra whitespace characters in text

    Uses SmartTextParser for more accurate whitespace handling,
    preserving the integrity of other tokens.

    Args:
        text: Text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return text

    if not isinstance(text, str):
        text = str(text)

    # Use smart parser to handle whitespace
    parser = SmartTextParser()
    tokens = parser.parse_tokens(text)

    if not tokens:
        return text.strip()

    # Reconstruct text, merging consecutive whitespaces into a single space
    result_parts = []
    prev_was_whitespace = False

    for token in tokens:
        if token.type == TokenType.WHITESPACE:
            if not prev_was_whitespace:
                result_parts.append(' ')  # Use single space uniformly
            prev_was_whitespace = True
        else:
            result_parts.append(token.content)
            prev_was_whitespace = False

    # Strip leading and trailing whitespace
    return ''.join(result_parts).strip()
