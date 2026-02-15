"""PyMuPDF-based PDF parser.

Orchestrates text extraction, geometry clustering, table detection,
and checkbox detection into a structured ParsedDocument.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # type: ignore[import-untyped]

from .base import BasePDFParser
from .pdf_geometry import (
    classify_segments,
    cluster_by_y,
    merge_touching_segments,
    sort_reading_order,
)
from .pdf_models import (
    BBox,
    Checkbox,
    CheckboxState,
    LineSegment,
    PageElement,
    ParsedDocument,
    ParsedPage,
    Table,
    TextLine,
    TextSpan,
)
from .pdf_table_extractor import TableExtractor, TableGridDetector

logger = logging.getLogger(__name__)


class PyMuPDFParser(BasePDFParser):
    """PDF parser using PyMuPDF (fitz) as the extraction backend.

    Pipeline per page:
    1. Extract text spans with bbox and font info
    2. Cluster spans into text lines by Y-proximity
    3. Extract vector line segments from drawings
    4. Detect checkboxes from small square rects
    5. Detect table grids from line intersections
    6. Extract table content from spans within grid cells
    7. Filter text lines that overlap table regions
    8. Build page elements in reading order
    """

    def __init__(
        self,
        line_y_tolerance: float = 2.0,
        min_segment_length: float = 10.0,
        grid_snap_tolerance: float = 3.0,
        min_table_rows: int = 2,
        min_table_cols: int = 2,
        segment_merge_gap: float = 2.0,
        checkbox_min_size: float = 6.0,
        checkbox_max_size: float = 24.0,
        checkbox_aspect_ratio_tolerance: float = 0.3,
        include_tables_in_text: bool = True,
    ) -> None:
        self.line_y_tolerance = line_y_tolerance
        self.min_segment_length = min_segment_length
        self.grid_snap_tolerance = grid_snap_tolerance
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        self.segment_merge_gap = segment_merge_gap
        self.checkbox_min_size = checkbox_min_size
        self.checkbox_max_size = checkbox_max_size
        self.checkbox_aspect_ratio_tolerance = checkbox_aspect_ratio_tolerance
        self.include_tables_in_text = include_tables_in_text

        self.grid_detector = TableGridDetector(
            grid_snap_tolerance=grid_snap_tolerance,
            min_table_rows=min_table_rows,
            min_table_cols=min_table_cols,
        )
        self.table_extractor = TableExtractor()

    def parse_file(
        self, file_path: str, pages: Optional[List[int]] = None
    ) -> ParsedDocument:
        """Parse a PDF file into a structured ParsedDocument.

        Args:
            file_path: Path to the PDF file.
            pages: Optional list of 0-indexed page numbers to parse.

        Returns:
            ParsedDocument with all extracted data.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(file_path)
        parsed_pages: List[ParsedPage] = []

        page_indices = pages if pages is not None else range(doc.page_count)
        for page_num in page_indices:
            if page_num < 0 or page_num >= doc.page_count:
                logger.warning(
                    f"Page {page_num} out of range (0-{doc.page_count - 1}), skipping."
                )
                continue
            fitz_page = doc[page_num]
            parsed_pages.append(self.parse_page(fitz_page, page_num))

        doc.close()

        return ParsedDocument(
            source=str(path),
            total_pages=doc.page_count,
            pages=parsed_pages,
            metadata={"filename": path.name, "extension": path.suffix},
        )

    def parse_page(self, page: Any, page_number: int) -> ParsedPage:
        """Parse a single PDF page.

        Args:
            page: A fitz.Page object.
            page_number: Zero-indexed page number.

        Returns:
            ParsedPage with extracted elements.
        """
        width = page.rect.width
        height = page.rect.height

        # Step 1: Extract text spans
        spans = self._extract_spans(page)

        # Step 2: Cluster into text lines
        text_lines = cluster_by_y(spans, tolerance=self.line_y_tolerance)

        # Step 3: Extract line segments from drawings
        segments = self._extract_segments(page)

        # Step 4: Classify segments
        horizontal, vertical = classify_segments(
            segments, min_length=self.min_segment_length
        )

        # Step 5: Merge touching segments
        horizontal = merge_touching_segments(horizontal, self.segment_merge_gap)
        vertical = merge_touching_segments(vertical, self.segment_merge_gap)

        # Step 6: Detect checkboxes
        checkboxes = self._detect_checkboxes(page, text_lines)

        # Step 7: Detect table grids
        grids = self.grid_detector.detect_grids(horizontal, vertical)

        # Step 8: Extract tables
        tables = self.table_extractor.extract_tables(grids, spans)

        # Step 9: Filter text lines that overlap table regions
        free_text_lines = self._filter_table_overlaps(text_lines, tables)

        # Step 10: Build page elements in reading order
        elements = self._build_elements(free_text_lines, tables, checkboxes)
        elements = sort_reading_order(elements)

        return ParsedPage(
            page_number=page_number,
            width=width,
            height=height,
            elements=elements,
            text_lines=free_text_lines,
            tables=tables,
            checkboxes=checkboxes,
        )

    def _extract_spans(self, page: Any) -> List[TextSpan]:
        """Extract TextSpans from a fitz page using get_text('dict')."""
        spans: List[TextSpan] = []
        text_dict: Dict[str, Any] = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    bbox_data = span.get("bbox", (0, 0, 0, 0))
                    font_name = span.get("font", "")
                    font_size = span.get("size", 0.0)
                    flags = span.get("flags", 0)

                    spans.append(
                        TextSpan(
                            text=text,
                            bbox=BBox(
                                x0=bbox_data[0],
                                y0=bbox_data[1],
                                x1=bbox_data[2],
                                y1=bbox_data[3],
                            ),
                            font_name=font_name,
                            font_size=font_size,
                            is_bold=bool(flags & 2**4),
                            is_italic=bool(flags & 2**1),
                        )
                    )

        return spans

    def _extract_segments(self, page: Any) -> List[LineSegment]:
        """Extract LineSegments from fitz page drawings.

        Handles both line items and rect items (decomposed into 4 segments).
        """
        segments: List[LineSegment] = []

        for drawing in page.get_cdrawings():
            for item in drawing.get("items", []):
                kind = item[0]

                if kind == "l":  # line
                    p1, p2 = item[1], item[2]
                    segments.append(LineSegment(x0=p1.x, y0=p1.y, x1=p2.x, y1=p2.y))
                elif kind == "re":  # rectangle
                    rect = item[1]
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                    # Decompose into 4 line segments
                    segments.extend(
                        [
                            LineSegment(x0=x0, y0=y0, x1=x1, y1=y0),  # top
                            LineSegment(x0=x1, y0=y0, x1=x1, y1=y1),  # right
                            LineSegment(x0=x0, y0=y1, x1=x1, y1=y1),  # bottom
                            LineSegment(x0=x0, y0=y0, x1=x0, y1=y1),  # left
                        ]
                    )
                elif kind == "c":  # curve — skip for now
                    pass

        return segments

    def _detect_checkboxes(
        self, page: Any, text_lines: List[TextLine]
    ) -> List[Checkbox]:
        """Detect checkboxes from small square rectangles in drawings."""
        checkboxes: List[Checkbox] = []

        for drawing in page.get_cdrawings():
            for item in drawing.get("items", []):
                if item[0] != "re":
                    continue

                rect = item[1]
                w = abs(rect.x1 - rect.x0)
                h = abs(rect.y1 - rect.y0)

                if w < self.checkbox_min_size or w > self.checkbox_max_size:
                    continue
                if h < self.checkbox_min_size or h > self.checkbox_max_size:
                    continue

                aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                if aspect_ratio < (1.0 - self.checkbox_aspect_ratio_tolerance):
                    continue

                bbox = BBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

                # Check if there's a mark inside (fill or inner drawing)
                fill = drawing.get("fill")
                has_fill = fill is not None and fill != (1, 1, 1)
                state = CheckboxState.CHECKED if has_fill else CheckboxState.UNCHECKED

                # Find label text to the right
                label = self._find_checkbox_label(bbox, text_lines)

                checkboxes.append(Checkbox(state=state, bbox=bbox, label=label))

        return checkboxes

    def _find_checkbox_label(self, cb_bbox: BBox, text_lines: List[TextLine]) -> str:
        """Find the text label immediately to the right of a checkbox."""
        best_line = None
        best_distance = float("inf")

        for line in text_lines:
            # Label should be roughly at the same Y and to the right
            if not cb_bbox.overlaps_y(line.bbox, tolerance=3.0):
                continue
            x_distance = line.bbox.x0 - cb_bbox.x1
            if 0 <= x_distance < 50 and x_distance < best_distance:
                best_distance = x_distance
                best_line = line

        return best_line.text if best_line else ""

    def _filter_table_overlaps(
        self, text_lines: List[TextLine], tables: List[Table]
    ) -> List[TextLine]:
        """Remove text lines whose bbox falls inside a table region."""
        if not tables:
            return text_lines

        free_lines: List[TextLine] = []
        for line in text_lines:
            in_table = any(table.bbox.contains(line.bbox) for table in tables)
            if not in_table:
                free_lines.append(line)

        return free_lines

    def _build_elements(
        self,
        text_lines: List[TextLine],
        tables: List[Table],
        checkboxes: List[Checkbox],
    ) -> List[PageElement]:
        """Build PageElement list from all detected content."""
        elements: List[PageElement] = []

        for line in text_lines:
            elements.append(
                PageElement(element_type="text_line", text_line=line, bbox=line.bbox)
            )

        if self.include_tables_in_text:
            for table in tables:
                elements.append(
                    PageElement(element_type="table", table=table, bbox=table.bbox)
                )

        for checkbox in checkboxes:
            elements.append(
                PageElement(
                    element_type="checkbox", checkbox=checkbox, bbox=checkbox.bbox
                )
            )

        return elements
