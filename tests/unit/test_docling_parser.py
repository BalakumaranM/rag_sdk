"""Tests for Docling PDF parser.

Uses mock docling objects to avoid requiring the optional docling dependency.
All mock modules are patched into sys.modules for the duration of each test.
"""

import builtins
import sys
from types import ModuleType
from typing import Any, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from rag_sdk.document.pdf_models import ParsedDocument


# ---------------------------------------------------------------------------
# Mock Docling types — these simulate the real docling_core/docling classes
# ---------------------------------------------------------------------------


class MockBoundingBox:
    """Mock docling_core BoundingBox."""

    def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
        self.l = left
        self.t = top
        self.r = right
        self.b = bottom


class MockProvenanceItem:
    """Mock docling_core ProvenanceItem."""

    def __init__(self, bbox: MockBoundingBox, page_no: int = 1) -> None:
        self.bbox = bbox
        self.page_no = page_no


class MockTextItem:
    """Mock docling_core TextItem."""

    def __init__(
        self, text: str, prov: Optional[List[MockProvenanceItem]] = None
    ) -> None:
        self.text = text
        self.prov = prov or []
        self.label = MagicMock(value="paragraph")


class MockSectionHeaderItem(MockTextItem):
    """Mock docling_core SectionHeaderItem (extends TextItem)."""

    def __init__(
        self, text: str, prov: Optional[List[MockProvenanceItem]] = None
    ) -> None:
        super().__init__(text, prov)
        self.label = MagicMock(value="section_header")


class MockDataFrame:
    """Minimal mock of pandas DataFrame for table tests."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        rows: Optional[List[List[Any]]] = None,
    ) -> None:
        self.columns = MagicMock()
        self.columns.tolist.return_value = columns or []
        self._rows = rows or []
        self.values = MagicMock()
        self.values.tolist.return_value = self._rows
        self.empty = not bool(columns)


class MockTableItem:
    """Mock docling_core TableItem."""

    def __init__(
        self,
        df: Any = None,
        prov: Optional[List[MockProvenanceItem]] = None,
    ) -> None:
        self._df = df
        self.prov = prov or []
        self.label = MagicMock(value="table")

    def export_to_dataframe(self, doc: Any = None) -> Any:
        if self._df is not None:
            return self._df
        return MockDataFrame()


class MockPageSize:
    """Mock page size object."""

    def __init__(self, width: float = 612.0, height: float = 792.0) -> None:
        self.width = width
        self.height = height


class MockPage:
    """Mock page entry in DoclingDocument.pages."""

    def __init__(self, width: float = 612.0, height: float = 792.0) -> None:
        self.size = MockPageSize(width, height)


class MockDoclingDocument:
    """Mock DoclingDocument returned by Docling's converter."""

    def __init__(
        self,
        items: Optional[List[Tuple[Any, int]]] = None,
        num_pages: int = 1,
        page_width: float = 612.0,
        page_height: float = 792.0,
    ) -> None:
        self._items = items or []
        self._num_pages = num_pages
        self.pages = {
            i: MockPage(page_width, page_height) for i in range(1, num_pages + 1)
        }

    def num_pages(self) -> int:
        return self._num_pages

    def iterate_items(
        self, page_no: Optional[int] = None, **kwargs: Any
    ) -> List[Tuple[Any, int]]:
        if page_no is not None:
            return [
                (item, level)
                for item, level in self._items
                if self._item_on_page(item, page_no)
            ]
        return self._items

    def _item_on_page(self, item: Any, page_no: int) -> bool:
        if hasattr(item, "prov") and item.prov:
            return item.prov[0].page_no == page_no
        return True


class MockConversionResult:
    """Mock Docling ConversionResult."""

    def __init__(self, document: MockDoclingDocument) -> None:
        self.document = document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_mock_docling_modules() -> dict[str, ModuleType]:
    """Create mock modules for docling and docling_core imports."""
    doc_types = ModuleType("docling_core.types.doc")
    doc_types.TextItem = MockTextItem  # type: ignore[attr-defined]
    doc_types.SectionHeaderItem = MockSectionHeaderItem  # type: ignore[attr-defined]
    doc_types.TableItem = MockTableItem  # type: ignore[attr-defined]

    pipeline_options = ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.PdfPipelineOptions = MagicMock  # type: ignore[attr-defined]
    pipeline_options.TableStructureOptions = MagicMock  # type: ignore[attr-defined]
    pipeline_options.TableFormerMode = MagicMock(  # type: ignore[attr-defined]
        ACCURATE="accurate", FAST="fast"
    )

    base_models = ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = MagicMock(PDF="pdf")  # type: ignore[attr-defined]

    converter_module = ModuleType("docling.document_converter")
    converter_module.DocumentConverter = MagicMock  # type: ignore[attr-defined]
    converter_module.PdfFormatOption = MagicMock  # type: ignore[attr-defined]

    return {
        "docling": ModuleType("docling"),
        "docling.datamodel": ModuleType("docling.datamodel"),
        "docling.datamodel.pipeline_options": pipeline_options,
        "docling.datamodel.base_models": base_models,
        "docling.document_converter": converter_module,
        "docling_core": ModuleType("docling_core"),
        "docling_core.types": ModuleType("docling_core.types"),
        "docling_core.types.doc": doc_types,
    }


@pytest.fixture(autouse=True)
def _patch_docling_modules() -> Generator[None, None, None]:
    """Patch docling/docling_core into sys.modules for every test."""
    mock_modules = _create_mock_docling_modules()
    # Clear cached parser module so it re-imports with our mocks
    sys.modules.pop("rag_sdk.document.docling_parser", None)
    with patch.dict(sys.modules, mock_modules):
        yield
    # Clean up so other test files aren't affected
    sys.modules.pop("rag_sdk.document.docling_parser", None)


def _build_parser(converter_result: MockConversionResult) -> Any:
    """Build a DoclingParser with a mocked converter."""
    from rag_sdk.document.docling_parser import DoclingParser

    parser = DoclingParser()
    mock_converter = MagicMock()
    mock_converter.convert.return_value = converter_result
    parser.converter = mock_converter
    return parser


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDoclingParserBasic:
    """Basic parse_file tests."""

    def test_parse_file_basic(self) -> None:
        """Verify ParsedDocument structure from a simple document."""
        prov = MockProvenanceItem(MockBoundingBox(10, 20, 200, 35), page_no=1)
        items = [(MockTextItem("Hello world", prov=[prov]), 0)]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        assert isinstance(parsed, ParsedDocument)
        assert parsed.source == "test.pdf"
        assert parsed.total_pages == 1
        assert len(parsed.pages) == 1

    def test_parse_file_multi_page(self) -> None:
        """Verify multi-page document produces correct number of pages."""
        items = [
            (
                MockTextItem(
                    "Page 1 text",
                    prov=[
                        MockProvenanceItem(MockBoundingBox(0, 0, 100, 20), page_no=1)
                    ],
                ),
                0,
            ),
            (
                MockTextItem(
                    "Page 2 text",
                    prov=[
                        MockProvenanceItem(MockBoundingBox(0, 0, 100, 20), page_no=2)
                    ],
                ),
                0,
            ),
        ]
        doc = MockDoclingDocument(items=items, num_pages=2)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        assert parsed.total_pages == 2
        assert len(parsed.pages) == 2
        assert parsed.pages[0].page_number == 0
        assert parsed.pages[1].page_number == 1


class TestTextItemMapping:
    """TextItem -> TextLine -> PageElement mapping."""

    def test_text_item_mapping(self) -> None:
        """Verify TextItem maps to a text_line PageElement."""
        prov = MockProvenanceItem(MockBoundingBox(10, 20, 200, 35), page_no=1)
        items = [(MockTextItem("Sample paragraph text", prov=[prov]), 0)]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        page = parsed.pages[0]
        assert len(page.elements) == 1
        assert len(page.text_lines) == 1
        assert len(page.tables) == 0

        elem = page.elements[0]
        assert elem.element_type == "text_line"
        assert elem.text_line is not None
        assert elem.text_line.text == "Sample paragraph text"
        assert elem.text_line.spans[0].is_bold is False

        assert elem.bbox.x0 == 10
        assert elem.bbox.y0 == 20
        assert elem.bbox.x1 == 200
        assert elem.bbox.y1 == 35

    def test_empty_text_skipped(self) -> None:
        """Verify empty/whitespace TextItems are skipped."""
        items = [
            (MockTextItem("", prov=[]), 0),
            (MockTextItem("   ", prov=[]), 0),
        ]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        assert len(parsed.pages[0].elements) == 0

    def test_text_item_no_provenance(self) -> None:
        """TextItem without provenance gets zero bbox."""
        items = [(MockTextItem("No bbox text", prov=[]), 0)]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        elem = parsed.pages[0].elements[0]
        assert elem.bbox.x0 == 0
        assert elem.bbox.y0 == 0
        assert elem.bbox.x1 == 0
        assert elem.bbox.y1 == 0


class TestSectionHeaderMapping:
    """SectionHeaderItem maps to bold TextLine."""

    def test_section_header_bold(self) -> None:
        """SectionHeaderItem should produce is_bold=True on the TextSpan."""
        prov = MockProvenanceItem(MockBoundingBox(10, 50, 300, 70), page_no=1)
        items = [(MockSectionHeaderItem("Introduction", prov=[prov]), 0)]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        elem = parsed.pages[0].elements[0]
        assert elem.element_type == "text_line"
        assert elem.text_line is not None
        assert elem.text_line.spans[0].is_bold is True
        assert elem.text_line.text == "Introduction"


class TestTableItemMapping:
    """TableItem -> Table -> PageElement mapping."""

    def test_table_item_mapping(self) -> None:
        """Verify TableItem with dataframe maps to Table with headers/rows."""
        df = MockDataFrame(
            columns=["Name", "Score"],
            rows=[["Alice", 95], ["Bob", 87]],
        )
        prov = MockProvenanceItem(MockBoundingBox(50, 100, 400, 250), page_no=1)
        items = [(MockTableItem(df=df, prov=[prov]), 0)]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        page = parsed.pages[0]
        assert len(page.tables) == 1
        assert len(page.text_lines) == 0

        table = page.tables[0]
        assert table.headers == ["Name", "Score"]
        assert table.rows == [["Alice", "95"], ["Bob", "87"]]

        elem = page.elements[0]
        assert elem.element_type == "table"
        assert elem.bbox.x0 == 50

    def test_empty_table_skipped(self) -> None:
        """Empty dataframe table should be skipped."""
        items = [(MockTableItem(df=MockDataFrame(), prov=[]), 0)]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")

        assert len(parsed.pages[0].tables) == 0
        assert len(parsed.pages[0].elements) == 0


class TestPageFiltering:
    """Page filtering via the pages parameter."""

    def test_page_filtering(self) -> None:
        """Passing pages=[0, 2] only processes those pages."""
        items = [
            (
                MockTextItem(
                    f"Page {i + 1}",
                    prov=[
                        MockProvenanceItem(
                            MockBoundingBox(0, 0, 100, 20), page_no=i + 1
                        )
                    ],
                ),
                0,
            )
            for i in range(3)
        ]
        doc = MockDoclingDocument(items=items, num_pages=3)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf", pages=[0, 2])

        assert parsed.total_pages == 3
        assert len(parsed.pages) == 2
        assert parsed.pages[0].page_number == 0
        assert parsed.pages[1].page_number == 2

    def test_out_of_range_pages_skipped(self) -> None:
        """Pages beyond the document range are silently skipped."""
        doc = MockDoclingDocument(items=[], num_pages=2)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf", pages=[0, 5, 10])

        assert len(parsed.pages) == 1
        assert parsed.pages[0].page_number == 0


class TestMissingDependency:
    """Verify clear error when docling is not installed."""

    def test_missing_dependency_error(self) -> None:
        """ImportError with helpful message when docling not installed."""
        # This test needs to undo the autouse fixture's mock modules
        # so we can simulate docling not being installed.
        blocked = {
            "docling",
            "docling.datamodel",
            "docling.datamodel.base_models",
            "docling.datamodel.pipeline_options",
            "docling.document_converter",
        }

        # Remove all mock docling modules from sys.modules
        for key in list(sys.modules):
            if key.startswith("docling"):
                del sys.modules[key]
        sys.modules.pop("rag_sdk.document.docling_parser", None)

        real_import = builtins.__import__

        def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in blocked:
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=blocking_import):
            from rag_sdk.document.docling_parser import DoclingParser

            with pytest.raises(ImportError, match="pip install rag-sdk\\[docling\\]"):
                DoclingParser()


class TestToDocuments:
    """Verify the inherited to_documents() method works with Docling output."""

    def test_to_documents_one_per_page(self) -> None:
        """to_documents produces one Document per page."""
        prov = MockProvenanceItem(MockBoundingBox(10, 20, 200, 35), page_no=1)
        items = [(MockTextItem("Hello from Docling", prov=[prov]), 0)]
        doc = MockDoclingDocument(items=items, num_pages=1)

        parser = _build_parser(MockConversionResult(doc))
        parsed = parser.parse_file("test.pdf")
        documents = parser.to_documents(parsed)

        assert len(documents) == 1
        assert "Hello from Docling" in documents[0].content
        assert documents[0].metadata["source"] == "test.pdf"
        assert documents[0].metadata["page_number"] == 0
