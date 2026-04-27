"""Tests for ContentItem field rename and normalization (content → text).

Covers three compatibility layers:
1. ContentItem Pydantic model: model_validator maps legacy 'content' → 'text'
2. _normalize_content_items(): normalizes raw dicts at ingestion time
3. get_text_from_content_items(): reads both 'text' and 'content' fields
"""

import pytest

from api_specs.dtos.memory import ContentItem
from api_specs.memory_types import get_text_from_content_items
from api_specs.request_converter import _normalize_content_items


# =============================================================================
# ContentItem Pydantic model — model_validator compat
# =============================================================================


class TestContentItemCompatValidator:
    """ContentItem model_validator: legacy 'content' field maps to 'text'."""

    def test_canonical_text_field(self):
        item = ContentItem(type="text", text="hello")
        assert item.text == "hello"

    def test_legacy_content_field_mapped_to_text(self):
        """Callers passing 'content' instead of 'text' should be accepted."""
        item = ContentItem.model_validate({"type": "text", "content": "hello"})
        assert item.text == "hello"

    def test_text_takes_precedence_over_content(self):
        """If both 'text' and 'content' are provided, 'text' wins (no remapping)."""
        item = ContentItem.model_validate(
            {"type": "text", "text": "canonical", "content": "legacy"}
        )
        assert item.text == "canonical"

    def test_non_text_type_with_content_field(self):
        """Even non-text types with 'content' should get it mapped to 'text'."""
        item = ContentItem.model_validate({"type": "image", "content": "a cat"})
        assert item.text == "a cat"

    def test_no_text_no_content_is_none(self):
        item = ContentItem(type="text")
        assert item.text is None

    def test_source_field_preserved(self):
        item = ContentItem.model_validate(
            {"type": "text", "content": "doc body", "source": "google_doc"}
        )
        assert item.text == "doc body"
        assert item.source == "google_doc"

    def test_type_field_required(self):
        with pytest.raises(Exception):
            ContentItem.model_validate({"text": "hello"})  # missing 'type'


# =============================================================================
# MessageItem.content — str | List[ContentItem] coercion
# =============================================================================


class TestMessageItemContentCoercion:
    """MessageItem.content accepts plain string, coerced to [{type: text, text: ...}]."""

    def _base(self, content):
        return {"role": "user", "timestamp": 1700000000000, "content": content}

    def test_plain_string_coerced_to_array(self):
        from api_specs.dtos.memory import MessageItem

        item = MessageItem.model_validate(self._base("hello world"))
        assert len(item.content) == 1
        assert item.content[0].type == "text"
        assert item.content[0].text == "hello world"

    def test_array_format_unchanged(self):
        from api_specs.dtos.memory import MessageItem

        item = MessageItem.model_validate(
            self._base([{"type": "text", "text": "hello"}])
        )
        assert len(item.content) == 1
        assert item.content[0].text == "hello"

    def test_array_with_legacy_content_field(self):
        from api_specs.dtos.memory import MessageItem

        item = MessageItem.model_validate(
            self._base([{"type": "text", "content": "hello"}])
        )
        assert item.content[0].text == "hello"

    def test_empty_string_fails_validation(self):
        from api_specs.dtos.memory import MessageItem

        with pytest.raises(Exception):
            MessageItem.model_validate(self._base(""))

    def test_empty_array_fails_validation(self):
        from api_specs.dtos.memory import MessageItem

        with pytest.raises(Exception):
            MessageItem.model_validate(self._base([]))


# =============================================================================
# request_converter — raw dict path string content coercion
# =============================================================================


class TestRequestConverterStringContent:
    """Raw dict path: string content coerced before _normalize_content_items."""

    def _personal_request(self, content):
        return {
            "user_id": "user_001",
            "messages": [
                {"role": "user", "timestamp": 1700000000000, "content": content}
            ],
        }

    def _group_request(self, content):
        return {
            "group_id": "group_001",
            "messages": [
                {
                    "role": "user",
                    "sender_id": "user_001",
                    "timestamp": 1700000000000,
                    "content": content,
                }
            ],
        }

    def test_personal_add_string_content(self):
        from api_specs.request_converter import convert_personal_add_to_memorize_request

        req = convert_personal_add_to_memorize_request(self._personal_request("hello"))
        raw = req.new_raw_data_list[0]
        items = raw.content["content"]
        assert items == [{"type": "text", "text": "hello"}]

    def test_group_add_string_content(self):
        from api_specs.request_converter import convert_group_add_to_memorize_request

        req = convert_group_add_to_memorize_request(self._group_request("hello group"))
        raw = req.new_raw_data_list[0]
        items = raw.content["content"]
        assert items == [{"type": "text", "text": "hello group"}]

    def test_personal_add_empty_string_raises(self):
        from api_specs.request_converter import convert_personal_add_to_memorize_request

        with pytest.raises(ValueError):
            convert_personal_add_to_memorize_request(self._personal_request(""))

    def test_personal_add_array_content_still_works(self):
        from api_specs.request_converter import convert_personal_add_to_memorize_request

        req = convert_personal_add_to_memorize_request(
            self._personal_request([{"type": "text", "text": "array content"}])
        )
        raw = req.new_raw_data_list[0]
        items = raw.content["content"]
        assert items == [{"type": "text", "text": "array content"}]


# =============================================================================
# _normalize_content_items() — raw dict ingestion normalization
# =============================================================================


class TestNormalizeContentItems:
    """_normalize_content_items() renames 'content' → 'text' for type='text' items."""

    def test_canonical_format_unchanged(self):
        items = [{"type": "text", "text": "hello"}]
        result = _normalize_content_items(items)
        assert result == [{"type": "text", "text": "hello"}]

    def test_legacy_content_renamed_to_text(self):
        items = [{"type": "text", "content": "hello"}]
        result = _normalize_content_items(items)
        assert result == [{"type": "text", "text": "hello"}]
        assert "content" not in result[0]

    def test_both_text_and_content_not_modified(self):
        """If 'text' already exists, do not overwrite with 'content'."""
        items = [{"type": "text", "text": "canonical", "content": "legacy"}]
        result = _normalize_content_items(items)
        assert result[0]["text"] == "canonical"
        # 'content' is NOT removed when 'text' already present (no-op case)
        assert "content" in result[0]

    def test_non_text_type_content_not_renamed(self):
        """Only type='text' items have their 'content' field renamed."""
        items = [{"type": "image", "content": "some_url"}]
        result = _normalize_content_items(items)
        # For non-text type, 'content' stays as-is (no rename)
        assert result[0].get("content") == "some_url"
        assert "text" not in result[0]

    def test_multiple_items_mixed(self):
        items = [
            {"type": "text", "content": "first"},
            {"type": "text", "text": "second"},
            {"type": "image", "content": "img_url"},
        ]
        result = _normalize_content_items(items)
        assert result[0] == {"type": "text", "text": "first"}
        assert result[1] == {"type": "text", "text": "second"}
        assert result[2] == {"type": "image", "content": "img_url"}

    def test_non_dict_items_pass_through(self):
        items = ["raw_string", 42, None]
        result = _normalize_content_items(items)
        assert result == ["raw_string", 42, None]

    def test_empty_list(self):
        assert _normalize_content_items([]) == []

    def test_original_dict_not_mutated(self):
        """Function must not mutate the input dicts in place."""
        original = {"type": "text", "content": "hello"}
        items = [original]
        _normalize_content_items(items)
        assert "content" in original  # original is untouched
        assert "text" not in original

    def test_item_without_type_field_pass_through(self):
        items = [{"content": "no type"}]
        result = _normalize_content_items(items)
        # No 'type' field → no rename
        assert result[0].get("content") == "no type"


# =============================================================================
# get_text_from_content_items() — read-side compat for stored data
# =============================================================================


class TestGetTextFromContentItems:
    """get_text_from_content_items() reads from 'text' (new) and 'content' (legacy)."""

    def test_canonical_text_field(self):
        items = [{"type": "text", "text": "hello world"}]
        assert get_text_from_content_items(items) == "hello world"

    def test_legacy_content_field_fallback(self):
        """Historical data stored with 'content' field should still be readable."""
        items = [{"type": "text", "content": "hello world"}]
        assert get_text_from_content_items(items) == "hello world"

    def test_text_takes_priority_over_content(self):
        items = [{"type": "text", "text": "canonical", "content": "legacy"}]
        assert get_text_from_content_items(items) == "canonical"

    def test_multiple_text_items_joined(self):
        items = [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]
        assert get_text_from_content_items(items) == "line one line two"

    def test_non_text_type_formatted_as_metadata(self):
        """Non-text items are formatted as [TYPE] metadata, not skipped."""
        items = [
            {"type": "image", "text": "image description"},
            {"type": "text", "text": "actual text"},
        ]
        assert get_text_from_content_items(items) == "[IMAGE] actual text"

    def test_empty_text_item_skipped(self):
        items = [{"type": "text", "text": ""}, {"type": "text", "text": "has content"}]
        assert get_text_from_content_items(items) == "has content"

    def test_plain_string_returned_as_is(self):
        """Legacy path: if a plain string is passed, return it directly."""
        assert get_text_from_content_items("plain text") == "plain text"

    def test_empty_list_returns_empty_string(self):
        assert get_text_from_content_items([]) == ""

    def test_non_list_non_string_returns_empty(self):
        assert get_text_from_content_items(None) == ""
        assert get_text_from_content_items(42) == ""

    def test_mixed_legacy_and_canonical(self):
        """Mix of old 'content' and new 'text' field items across list."""
        items = [
            {"type": "text", "content": "from legacy"},
            {"type": "text", "text": "from canonical"},
        ]
        result = get_text_from_content_items(items)
        assert result == "from legacy from canonical"
