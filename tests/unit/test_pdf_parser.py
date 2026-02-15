"""Tests for PyMuPDF PDF parser.

Uses mock fitz.Page objects to avoid real PDF dependencies in unit tests.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

from rag_sdk.document.pdf_models import (
    BBox,
    Checkbox,
    CheckboxState,
    PageElement,
    ParsedDocument,
    ParsedPage,
    Table,
    TextLine,
    TextSpan,
)
from rag_sdk.document.pdf_parser import PyMuPDFParser


class FitzRect:
    """Mock fitz.Rect."""

    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1 - x0
        self.height = y1 - y0


class FitzPoint:
    """Mock fitz.Point."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def make_mock_page(
    text_dict: Dict[str, Any],
    drawings: List[Dict[str, Any]] | None = None,
    width: float = 612,
    height: float = 792,
) -> MagicMock:
    """Create a mock fitz.Page object."""
    page = MagicMock()
    page.rect = FitzRect(0, 0, width, height)
    page.get_text.return_value = text_dict
    page.get_cdrawings.return_value = drawings or []
    return page


def make_text_dict(
    spans: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a fitz-style text dict from span definitions."""
    return {
        "blocks": [
            {
                "type": 0,
                "lines": [{"spans": spans}],
            }
        ]
    }


class TestPyMuPDFParserExtraction:
    def test_extract_text_spans(self) -> None:
        parser = PyMuPDFParser()
        text_dict: Dict[str, Any] = make_text_dict(
            [
                {
                    "text": "Hello",
                    "bbox": (10, 20, 50, 35),
                    "font": "Arial",
                    "size": 12.0,
                    "flags": 0,
                },
                {
                    "text": "World",
                    "bbox": (60, 20, 100, 35),
                    "font": "Arial-Bold",
                    "size": 12.0,
                    "flags": 16,  # bold flag
                },
            ]
        )

        page = make_mock_page(text_dict)
        result = parser.parse_page(page, page_number=0)

        assert isinstance(result, ParsedPage)
        assert result.page_number == 0
        assert len(result.text_lines) >= 1

        # Check that both spans were found
        all_text = " ".join(line.text for line in result.text_lines)
        assert "Hello" in all_text
        assert "World" in all_text

    def test_empty_page(self) -> None:
        parser = PyMuPDFParser()
        text_dict: Dict[str, Any] = {"blocks": []}
        page = make_mock_page(text_dict)
        result = parser.parse_page(page, page_number=0)

        assert result.page_number == 0
        assert len(result.text_lines) == 0
        assert len(result.tables) == 0
        assert len(result.checkboxes) == 0

    def test_skips_empty_text_spans(self) -> None:
        parser = PyMuPDFParser()
        text_dict: Dict[str, Any] = make_text_dict(
            [
                {
                    "text": "",
                    "bbox": (0, 0, 10, 10),
                    "font": "Arial",
                    "size": 12,
                    "flags": 0,
                },
                {
                    "text": "   ",
                    "bbox": (0, 0, 10, 10),
                    "font": "Arial",
                    "size": 12,
                    "flags": 0,
                },
                {
                    "text": "Real",
                    "bbox": (10, 0, 40, 10),
                    "font": "Arial",
                    "size": 12,
                    "flags": 0,
                },
            ]
        )
        page = make_mock_page(text_dict)
        result = parser.parse_page(page, page_number=0)

        all_text = " ".join(line.text for line in result.text_lines)
        assert all_text.strip() == "Real"


class TestPyMuPDFParserSegments:
    def test_extract_line_segments(self) -> None:
        parser = PyMuPDFParser()
        drawings = [
            {
                "items": [
                    ("l", FitzPoint(0, 50), FitzPoint(200, 50)),  # horizontal line
                    ("l", FitzPoint(100, 0), FitzPoint(100, 100)),  # vertical line
                ],
            }
        ]

        text_dict: Dict[str, Any] = {"blocks": []}
        page = make_mock_page(text_dict, drawings)
        segments = parser._extract_segments(page)

        assert len(segments) == 2
        assert segments[0].is_horizontal
        assert segments[1].is_vertical

    def test_extract_rect_as_four_segments(self) -> None:
        parser = PyMuPDFParser()
        drawings = [
            {
                "items": [
                    ("re", FitzRect(10, 20, 110, 70)),
                ],
            }
        ]

        text_dict: Dict[str, Any] = {"blocks": []}
        page = make_mock_page(text_dict, drawings)
        segments = parser._extract_segments(page)

        assert len(segments) == 4


class TestPyMuPDFParserCheckboxes:
    def test_detect_unchecked_checkbox(self) -> None:
        parser = PyMuPDFParser(checkbox_min_size=6.0, checkbox_max_size=24.0)

        # A small square rect (12x12)
        drawings = [
            {
                "items": [("re", FitzRect(10, 10, 22, 22))],
                "fill": None,  # no fill = unchecked
            }
        ]

        # A text line to the right as label
        text_dict: Dict[str, Any] = make_text_dict(
            [
                {
                    "text": "Accept",
                    "bbox": (30, 10, 80, 22),
                    "font": "Arial",
                    "size": 12,
                    "flags": 0,
                },
            ]
        )

        page = make_mock_page(text_dict, drawings)
        result = parser.parse_page(page, page_number=0)

        assert len(result.checkboxes) == 1
        assert result.checkboxes[0].state == CheckboxState.UNCHECKED
        assert result.checkboxes[0].label == "Accept"

    def test_detect_checked_checkbox(self) -> None:
        parser = PyMuPDFParser()

        drawings = [
            {
                "items": [("re", FitzRect(10, 10, 22, 22))],
                "fill": (0, 0, 0),  # black fill = checked
            }
        ]

        text_dict: Dict[str, Any] = {"blocks": []}
        page = make_mock_page(text_dict, drawings)
        result = parser.parse_page(page, page_number=0)

        assert len(result.checkboxes) == 1
        assert result.checkboxes[0].state == CheckboxState.CHECKED


class TestPyMuPDFParserTableFiltering:
    def test_filter_text_lines_in_table_region(self) -> None:
        """Text lines inside table bbox should be filtered from free text."""
        parser = PyMuPDFParser()

        table = Table(
            headers=["A", "B"],
            rows=[["1", "2"]],
            bbox=BBox(x0=0, y0=0, x1=200, y1=100),
        )

        inside = TextLine(
            spans=[TextSpan(text="inside", bbox=BBox(x0=10, y0=10, x1=50, y1=20))],
            bbox=BBox(x0=10, y0=10, x1=50, y1=20),
        )
        outside = TextLine(
            spans=[TextSpan(text="outside", bbox=BBox(x0=10, y0=200, x1=80, y1=210))],
            bbox=BBox(x0=10, y0=200, x1=80, y1=210),
        )

        filtered = parser._filter_table_overlaps([inside, outside], [table])
        assert len(filtered) == 1
        assert filtered[0].text == "outside"


class TestToDocuments:
    def test_one_doc_per_page(self) -> None:
        parser = PyMuPDFParser()

        page0 = ParsedPage(
            page_number=0,
            width=612,
            height=792,
            elements=[],
            text_lines=[
                TextLine(
                    spans=[
                        TextSpan(
                            text="Page 1 text", bbox=BBox(x0=0, y0=0, x1=100, y1=10)
                        )
                    ],
                    bbox=BBox(x0=0, y0=0, x1=100, y1=10),
                )
            ],
        )
        page1 = ParsedPage(
            page_number=1,
            width=612,
            height=792,
            elements=[],
            text_lines=[
                TextLine(
                    spans=[
                        TextSpan(
                            text="Page 2 text", bbox=BBox(x0=0, y0=0, x1=100, y1=10)
                        )
                    ],
                    bbox=BBox(x0=0, y0=0, x1=100, y1=10),
                )
            ],
        )

        parsed = ParsedDocument(source="test.pdf", total_pages=2, pages=[page0, page1])

        docs = parser.to_documents(parsed, one_doc_per_page=True)
        assert len(docs) == 2
        assert docs[0].metadata["page_number"] == 0
        assert docs[1].metadata["page_number"] == 1

    def test_single_document_mode(self) -> None:
        parser = PyMuPDFParser()

        page = ParsedPage(
            page_number=0,
            width=612,
            height=792,
            elements=[],
            text_lines=[],
        )

        parsed = ParsedDocument(source="test.pdf", total_pages=1, pages=[page])

        docs = parser.to_documents(parsed, one_doc_per_page=False)
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "test.pdf"
        assert docs[0].metadata["total_pages"] == 1

    def test_checkbox_rendering(self) -> None:
        parser = PyMuPDFParser()
        cb_bbox = BBox(x0=10, y0=10, x1=22, y1=22)
        checkbox = Checkbox(state=CheckboxState.CHECKED, bbox=cb_bbox, label="Done")
        element = PageElement(element_type="checkbox", checkbox=checkbox, bbox=cb_bbox)

        page = ParsedPage(
            page_number=0,
            width=612,
            height=792,
            elements=[element],
            checkboxes=[checkbox],
        )
        parsed = ParsedDocument(source="test.pdf", total_pages=1, pages=[page])

        docs = parser.to_documents(parsed)
        assert "[x] Done" in docs[0].content

    def test_table_rendering_in_text(self) -> None:
        parser = PyMuPDFParser()
        table_bbox = BBox(x0=0, y0=0, x1=200, y1=100)
        table = Table(
            headers=["Name", "Age"],
            rows=[["Alice", "30"]],
            bbox=table_bbox,
        )
        element = PageElement(element_type="table", table=table, bbox=table_bbox)

        page = ParsedPage(
            page_number=0,
            width=612,
            height=792,
            elements=[element],
            tables=[table],
        )
        parsed = ParsedDocument(source="test.pdf", total_pages=1, pages=[page])

        docs = parser.to_documents(parsed)
        assert "Name | Age" in docs[0].content
        assert "Alice | 30" in docs[0].content
        assert '"headers"' in docs[0].metadata.get("tables", "")
