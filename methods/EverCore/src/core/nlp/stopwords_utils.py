"""
Stopwords utility class

Provides functionality for loading and managing stopwords, supports Harbin Institute of Technology stopwords list.
"""

import os
import logging
from typing import Set, Optional
from common_utils.project_path import CURRENT_DIR

logger = logging.getLogger(__name__)


class StopwordsManager:
    """Stopwords manager"""

    def __init__(self, stopwords_file_path: Optional[str] = None):
        """Initialize the stopwords manager

        Args:
            stopwords_file_path: Path to the stopwords file. If None, use default path
        """
        self.stopwords_file_path = (
            stopwords_file_path or self._get_default_stopwords_path()
        )
        self._stopwords: Optional[Set[str]] = None
        self.load_stopwords()

    def _get_default_stopwords_path(self) -> str:
        """Get default stopwords file path"""
        return str(CURRENT_DIR / "config" / "stopwords" / "hit_stopwords.txt")

    def load_stopwords(self) -> Set[str]:
        """Load stopwords

        Returns:
            Set of stopwords
        """
        if self._stopwords is not None:
            return self._stopwords

        stopwords = set()

        # Check if file exists
        if not os.path.exists(self.stopwords_file_path):
            logger.warning(f"Stopwords file does not exist: {self.stopwords_file_path}")
            logger.info("An empty stopwords set will be used")
            self._stopwords = stopwords
            return stopwords

        try:
            with open(self.stopwords_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:  # Skip empty lines
                        stopwords.add(word)

            logger.info(
                f"Successfully loaded stopwords, total {len(stopwords)} stopwords"
            )
            self._stopwords = stopwords
            return stopwords

        except Exception as e:
            logger.error(f"Failed to load stopwords: {e}")
            logger.info("An empty stopwords set will be used")
            self._stopwords = set()
            return set()

    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword

        Args:
            word: Word to check

        Returns:
            True if the word is a stopword, otherwise False
        """
        return word in self._stopwords

    def filter_stopwords(self, words: list, min_length: int = 1) -> list:
        """Filter out stopwords

        Args:
            words: List of words
            min_length: Minimum word length, words shorter than this will also be filtered

        Returns:
            List of words after filtering
        """

        filtered_words = []
        for word in words:
            if (
                word not in self._stopwords and len(word) >= min_length and word.strip()
            ):  # Filter whitespace characters
                filtered_words.append(word)

        return filtered_words


# Global stopwords manager instance
_stopwords_manager: Optional[StopwordsManager] = StopwordsManager()


def filter_stopwords(words: list, min_length: int = 1) -> list:
    """Convenience function: filter stopwords

    Args:
        words: List of words
        min_length: Minimum word length

    Returns:
        List of words after filtering
    """
    return _stopwords_manager.filter_stopwords(words, min_length)
