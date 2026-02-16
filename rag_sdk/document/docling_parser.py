"""Docling-based PDF parser.

Uses IBM's Docling library for ML-powered PDF understanding including
layout analysis (RT-DETR), table structure extraction (TableFormer),
and optional OCR. Handles multi-column layouts, borderless tables,
scanned PDFs, and complex academic layouts.

Requires the optional `docling` dependency: pip install rag-sdk[docling]
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from .base import BasePDFParser
from .pdf_models import (
    BBox,
    PageElement,
    ParsedDocument,
    ParsedPage,
    Table,
    TextLine,
    TextSpan,
)

logger = logging.getLogger(__name__)


class DoclingParser(BasePDFParser):
    """PDF parser using Docling as the extraction backend.

    Pipeline per document:
    1. Convert PDF via Docling's DocumentConverter (layout + table + OCR)
    2. Iterate items in reading order per page
    3. Map TextItem/SectionHeaderItem to TextLine, TableItem to Table
    4. Build ParsedDocument with structured page data

    Best suited for complex PDFs: multi-column, borderless tables, scanned.
    """

    def __init__(
        self,
        do_ocr: bool = True,
        do_table_structure: bool = True,
        table_mode: str = "accurate",
        timeout: Optional[float] = None,
    ) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableStructureOptions,
                TableFormerMode,
            )
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
            )
        except ImportError:
            raise ImportError(
                "Docling is required for the docling PDF parser backend. "
                "Install it with: pip install rag-sdk[docling]"
            )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=(
                TableFormerMode.ACCURATE
                if table_mode == "accurate"
                else TableFormerMode.FAST
            ),
        )
        if timeout is not None:
            pipeline_options.document_timeout = timeout

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    def parse_file(
        self, file_path: str, pages: Optional[List[int]] = None
    ) -> ParsedDocument:
        """Parse a PDF file into a structured ParsedDocument.

        Args:
            file_path: Path to the PDF file.
            pages: Optional list of 0-indexed page numbers to parse.
                If None, parse all pages.

        Returns:
            ParsedDocument with structured page data.
        """
        result = self.converter.convert(file_path)
        doc = result.document

        total_pages = doc.num_pages()

        parsed_pages: List[ParsedPage] = []
        pages_to_process = pages if pages is not None else list(range(total_pages))

        for page_num in pages_to_process:
            if 0 <= page_num < total_pages:
                parsed_page = self.parse_page(doc, page_num)
                parsed_pages.append(parsed_page)

        return ParsedDocument(
            source=str(Path(file_path).name),
            total_pages=total_pages,
            pages=parsed_pages,
        )

    def parse_page(self, page: Any, page_number: int) -> ParsedPage:
        """Parse a single PDF page from a DoclingDocument.

        Args:
            page: A DoclingDocument object (the full document).
            page_number: Zero-indexed page number.

        Returns:
            ParsedPage with extracted elements.
        """
        from docling_core.types.doc import SectionHeaderItem, TableItem, TextItem

        doc = page  # page param receives the DoclingDocument
        docling_page_no = page_number + 1  # Docling uses 1-indexed pages

        # Get page dimensions from document
        page_size = doc.pages.get(docling_page_no)
        width = page_size.size.width if page_size and page_size.size else 0.0
        height = page_size.size.height if page_size and page_size.size else 0.0

        elements: List[PageElement] = []
        text_lines: List[TextLine] = []
        tables: List[Table] = []

        for item, _level in doc.iterate_items(page_no=docling_page_no):
            if isinstance(item, TableItem):
                table_element = self._map_table_item(item, doc)
                if table_element:
                    tables.append(table_element.table)  # type: ignore[arg-type]
                    elements.append(table_element)
            elif isinstance(item, (TextItem, SectionHeaderItem)):
                text_element = self._map_text_item(item)
                if text_element:
                    text_lines.append(text_element.text_line)  # type: ignore[arg-type]
                    elements.append(text_element)

        return ParsedPage(
            page_number=page_number,
            width=width,
            height=height,
            elements=elements,
            text_lines=text_lines,
            tables=tables,
            checkboxes=[],
        )

    def _extract_bbox(self, item: Any) -> BBox:
        """Extract bounding box from a Docling item's provenance."""
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0]
            bbox = prov.bbox
            return BBox(x0=bbox.l, y0=bbox.t, x1=bbox.r, y1=bbox.b)
        return BBox(x0=0, y0=0, x1=0, y1=0)

    def _map_text_item(self, item: Any) -> Optional[PageElement]:
        """Map a Docling TextItem or SectionHeaderItem to a PageElement."""
        from docling_core.types.doc import SectionHeaderItem

        text = item.text
        if not text or not text.strip():
            return None

        bbox = self._extract_bbox(item)
        is_bold = isinstance(item, SectionHeaderItem)

        span = TextSpan(
            text=text,
            bbox=bbox,
            font_name="",
            font_size=0.0,
            is_bold=is_bold,
            is_italic=False,
        )
        text_line = TextLine(spans=[span], bbox=bbox)

        return PageElement(
            element_type="text_line",
            text_line=text_line,
            bbox=bbox,
        )

    def _map_table_item(self, item: Any, doc: Any) -> Optional[PageElement]:
        """Map a Docling TableItem to a PageElement with Table data."""
        bbox = self._extract_bbox(item)

        try:
            df = item.export_to_dataframe(doc=doc)
            if df.empty:
                return None

            headers = [str(col) for col in df.columns.tolist()]
            rows = [[str(cell) for cell in row] for row in df.values.tolist()]
        except Exception:
            logger.warning("Failed to export table to dataframe, skipping")
            return None

        table = Table(headers=headers, rows=rows, bbox=bbox)

        return PageElement(
            element_type="table",
            table=table,
            bbox=bbox,
        )
