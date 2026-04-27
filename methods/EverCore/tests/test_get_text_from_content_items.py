"""Tests for get_text_from_content_items multimodal text extraction."""

import pytest
from api_specs.memory_types import get_text_from_content_items


class TestPureText:
    def test_single_text_item(self):
        items = [{"type": "text", "text": "hello"}]
        assert get_text_from_content_items(items) == "hello"

    def test_multiple_text_items(self):
        items = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert get_text_from_content_items(items) == "first second"

    def test_empty_text_skipped(self):
        items = [{"type": "text", "text": ""}, {"type": "text", "text": "hello"}]
        assert get_text_from_content_items(items) == "hello"


class TestNonTextWithNameAndSummary:
    def test_pdf_with_name_and_summary(self):
        items = [
            {"type": "pdf", "name": "Q2_report.pdf", "uri": "minio://docs/Q2_report.pdf"}
        ]
        parse_info = {"minio://docs/Q2_report.pdf": {"summary": "Revenue grew 15%"}}
        assert (
            get_text_from_content_items(items, parse_info)
            == "[PDF: Q2_report.pdf | Summary: Revenue grew 15%]"
        )

    def test_image_with_name_and_summary(self):
        items = [
            {"type": "image", "name": "vacation.jpg", "uri": "minio://img/vacation.jpg"}
        ]
        parse_info = {"minio://img/vacation.jpg": {"summary": "A beach sunset photo"}}
        assert (
            get_text_from_content_items(items, parse_info)
            == "[IMAGE: vacation.jpg | Summary: A beach sunset photo]"
        )


class TestNonTextWithNameOnly:
    def test_image_name_no_parse_info(self):
        items = [{"type": "image", "name": "vacation.jpg"}]
        assert get_text_from_content_items(items) == "[IMAGE: vacation.jpg]"

    def test_parse_failed_with_name(self):
        items = [
            {"type": "image", "name": "photo.png", "uri": "minio://img/photo.png"}
        ]
        parse_info = {"minio://img/photo.png": {"status": "failed"}}
        assert get_text_from_content_items(items, parse_info) == "[IMAGE: photo.png]"


class TestNonTextWithSummaryOnly:
    def test_audio_summary_no_name(self):
        items = [{"type": "audio", "uri": "minio://audio/rec.mp3"}]
        parse_info = {"minio://audio/rec.mp3": {"summary": "Discussion about timeline"}}
        assert (
            get_text_from_content_items(items, parse_info)
            == "[AUDIO | Summary: Discussion about timeline]"
        )


class TestNonTextNoNameNoSummary:
    def test_no_parse_info(self):
        items = [{"type": "image"}]
        assert get_text_from_content_items(items) == "[IMAGE]"

    def test_audio_no_info(self):
        items = [{"type": "audio"}]
        assert get_text_from_content_items(items) == "[AUDIO]"

    def test_uri_not_in_parse_info(self):
        items = [{"type": "image", "uri": "minio://img/unknown.png"}]
        parse_info = {"minio://img/other.png": {"summary": "something"}}
        assert get_text_from_content_items(items, parse_info) == "[IMAGE]"


class TestMixedContent:
    def test_text_and_pdf(self):
        items = [
            {"type": "text", "text": "帮我总结"},
            {"type": "pdf", "name": "Q2_report.pdf", "uri": "minio://docs/Q2.pdf"},
        ]
        parse_info = {"minio://docs/Q2.pdf": {"summary": "Revenue grew 15%"}}
        assert (
            get_text_from_content_items(items, parse_info)
            == "帮我总结 [PDF: Q2_report.pdf | Summary: Revenue grew 15%]"
        )

    def test_text_and_multiple_files(self):
        items = [
            {"type": "text", "text": "会议材料都在这了"},
            {"type": "pdf", "name": "agenda.pdf", "uri": "minio://docs/agenda.pdf"},
            {"type": "image", "name": "whiteboard.png", "uri": "minio://img/wb.png"},
            {"type": "audio"},
        ]
        parse_info = {
            "minio://docs/agenda.pdf": {"summary": "Topics: budget review"},
            "minio://img/wb.png": {"summary": "Architecture diagram"},
        }
        result = get_text_from_content_items(items, parse_info)
        assert result == (
            "会议材料都在这了 "
            "[PDF: agenda.pdf | Summary: Topics: budget review] "
            "[IMAGE: whiteboard.png | Summary: Architecture diagram] "
            "[AUDIO]"
        )


class TestLegacyFallback:
    def test_string_input(self):
        assert get_text_from_content_items("plain text") == "plain text"

    def test_empty_list(self):
        assert get_text_from_content_items([]) == ""

    def test_none_input(self):
        assert get_text_from_content_items(None) == ""

    def test_non_dict_items_skipped(self):
        items = ["not a dict", 123, {"type": "text", "text": "hello"}]
        assert get_text_from_content_items(items) == "hello"
