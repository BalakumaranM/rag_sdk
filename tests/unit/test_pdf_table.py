"""Tests for PDF table grid detection and extraction."""

from rag_sdk.document.pdf_models import BBox, LineSegment, TextSpan
from rag_sdk.document.pdf_table_extractor import (
    TableExtractor,
    TableGridDetector,
)


def _h_seg(x0: float, x1: float, y: float) -> LineSegment:
    """Create a horizontal segment."""
    return LineSegment(x0=x0, y0=y, x1=x1, y1=y)


def _v_seg(x: float, y0: float, y1: float) -> LineSegment:
    """Create a vertical segment."""
    return LineSegment(x0=x, y0=y0, x1=x, y1=y1)


def _span(text: str, x0: float, y0: float, x1: float, y1: float) -> TextSpan:
    """Create a TextSpan."""
    return TextSpan(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))


class TestTableGridDetector:
    def test_simple_2x2_grid(self) -> None:
        """A 2x2 table has 3 horizontal + 3 vertical lines."""
        horizontal = [
            _h_seg(0, 200, 0),  # top
            _h_seg(0, 200, 50),  # middle
            _h_seg(0, 200, 100),  # bottom
        ]
        vertical = [
            _v_seg(0, 0, 100),  # left
            _v_seg(100, 0, 100),  # middle
            _v_seg(200, 0, 100),  # right
        ]

        detector = TableGridDetector(
            grid_snap_tolerance=3.0, min_table_rows=2, min_table_cols=2
        )
        grids = detector.detect_grids(horizontal, vertical)
        assert len(grids) == 1
        assert grids[0].num_rows == 2
        assert grids[0].num_cols == 2

    def test_3x3_grid(self) -> None:
        horizontal = [_h_seg(0, 300, y) for y in [0, 50, 100, 150]]
        vertical = [_v_seg(x, 0, 150) for x in [0, 100, 200, 300]]

        detector = TableGridDetector()
        grids = detector.detect_grids(horizontal, vertical)
        assert len(grids) == 1
        assert grids[0].num_rows == 3
        assert grids[0].num_cols == 3

    def test_no_grid_insufficient_lines(self) -> None:
        """Only one horizontal and one vertical — not enough for a grid."""
        horizontal = [_h_seg(0, 100, 50)]
        vertical = [_v_seg(50, 0, 100)]

        detector = TableGridDetector(min_table_rows=2, min_table_cols=2)
        grids = detector.detect_grids(horizontal, vertical)
        assert len(grids) == 0

    def test_no_grid_empty_input(self) -> None:
        detector = TableGridDetector()
        assert detector.detect_grids([], []) == []
        assert detector.detect_grids([_h_seg(0, 100, 0)], []) == []
        assert detector.detect_grids([], [_v_seg(0, 0, 100)]) == []

    def test_cell_bbox(self) -> None:
        """Verify cell bboxes are correct in a detected grid."""
        horizontal = [_h_seg(0, 200, y) for y in [0, 50, 100]]
        vertical = [_v_seg(x, 0, 100) for x in [0, 100, 200]]

        detector = TableGridDetector()
        grids = detector.detect_grids(horizontal, vertical)
        assert len(grids) == 1

        cell_00 = grids[0].get_cell_bbox(0, 0)
        assert cell_00.x0 == 0
        assert cell_00.y0 == 0
        assert cell_00.x1 == 100
        assert cell_00.y1 == 50

    def test_snap_tolerance(self) -> None:
        """Slightly misaligned lines should still form a grid."""
        horizontal = [
            _h_seg(0, 200, 0),
            _h_seg(0, 200, 49.5),  # slightly off from 50
            _h_seg(0, 200, 100.5),  # slightly off from 100
        ]
        vertical = [
            _v_seg(0, 0, 101),
            _v_seg(99.5, 0, 101),  # slightly off from 100
            _v_seg(200, 0, 101),
        ]

        detector = TableGridDetector(grid_snap_tolerance=3.0)
        grids = detector.detect_grids(horizontal, vertical)
        assert len(grids) == 1


class TestTableExtractor:
    def test_extract_from_2x2_grid(self) -> None:
        """Map text spans to cells and produce a Table."""
        horizontal = [_h_seg(0, 200, y) for y in [0, 30, 60]]
        vertical = [_v_seg(x, 0, 60) for x in [0, 100, 200]]

        detector = TableGridDetector()
        grids = detector.detect_grids(horizontal, vertical)
        assert len(grids) == 1

        # Place spans inside cells
        spans = [
            _span("Name", 10, 5, 50, 25),  # row 0, col 0
            _span("Age", 110, 5, 150, 25),  # row 0, col 1
            _span("Alice", 10, 35, 60, 55),  # row 1, col 0
            _span("30", 110, 35, 130, 55),  # row 1, col 1
        ]

        extractor = TableExtractor()
        tables = extractor.extract_tables(grids, spans)
        assert len(tables) == 1

        table = tables[0]
        assert table.headers == ["Name", "Age"]
        assert table.rows == [["Alice", "30"]]

    def test_no_headers_mode(self) -> None:
        horizontal = [_h_seg(0, 200, y) for y in [0, 30, 60]]
        vertical = [_v_seg(x, 0, 60) for x in [0, 100, 200]]

        detector = TableGridDetector()
        grids = detector.detect_grids(horizontal, vertical)

        spans = [
            _span("A1", 10, 5, 40, 25),
            _span("B1", 110, 5, 140, 25),
            _span("A2", 10, 35, 40, 55),
            _span("B2", 110, 35, 140, 55),
        ]

        extractor = TableExtractor()
        tables = extractor.extract_tables(grids, spans, detect_headers=False)
        assert len(tables) == 1
        assert tables[0].headers == ["col_0", "col_1"]
        assert len(tables[0].rows) == 2

    def test_empty_cells(self) -> None:
        """Cells with no text spans should have empty strings."""
        horizontal = [_h_seg(0, 200, y) for y in [0, 30, 60]]
        vertical = [_v_seg(x, 0, 60) for x in [0, 100, 200]]

        detector = TableGridDetector()
        grids = detector.detect_grids(horizontal, vertical)

        # Only one span, rest of cells empty
        spans = [_span("Only", 10, 5, 40, 25)]

        extractor = TableExtractor()
        tables = extractor.extract_tables(grids, spans)
        assert len(tables) == 1
        assert tables[0].headers == ["Only", ""]
        assert tables[0].rows == [["", ""]]
