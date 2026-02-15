"""Tests for PDF geometry utilities."""

from rag_sdk.document.pdf_models import (
    BBox,
    LineSegment,
    PageElement,
    TextLine,
    TextSpan,
)
from rag_sdk.document.pdf_geometry import (
    classify_segments,
    cluster_by_y,
    merge_touching_segments,
    sort_reading_order,
)


def _make_span(text: str, x0: float, y0: float, x1: float, y1: float) -> TextSpan:
    return TextSpan(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))


class TestClusterByY:
    def test_single_line(self) -> None:
        spans = [
            _make_span("Hello", 0, 10, 30, 20),
            _make_span("world", 35, 10, 65, 20),
        ]
        lines = cluster_by_y(spans, tolerance=2.0)
        assert len(lines) == 1
        assert lines[0].text == "Hello world"

    def test_two_lines(self) -> None:
        spans = [
            _make_span("Line1", 0, 10, 30, 20),
            _make_span("Line2", 0, 40, 30, 50),
        ]
        lines = cluster_by_y(spans, tolerance=2.0)
        assert len(lines) == 2
        assert lines[0].text == "Line1"
        assert lines[1].text == "Line2"

    def test_tolerance_merges_close_lines(self) -> None:
        spans = [
            _make_span("A", 0, 10, 10, 20),
            _make_span("B", 15, 12, 25, 22),  # close Y to first
        ]
        lines = cluster_by_y(spans, tolerance=5.0)
        assert len(lines) == 1
        assert lines[0].text == "A B"

    def test_tolerance_splits_far_lines(self) -> None:
        spans = [
            _make_span("A", 0, 10, 10, 20),
            _make_span("B", 15, 12, 25, 22),
        ]
        lines = cluster_by_y(spans, tolerance=0.5)
        assert len(lines) == 2

    def test_empty_input(self) -> None:
        assert cluster_by_y([]) == []

    def test_x_sorting_within_line(self) -> None:
        """Spans should be sorted left-to-right within each line."""
        spans = [
            _make_span("world", 50, 10, 80, 20),
            _make_span("Hello", 0, 10, 30, 20),
        ]
        lines = cluster_by_y(spans, tolerance=2.0)
        assert lines[0].text == "Hello world"

    def test_bbox_covers_all_spans(self) -> None:
        spans = [
            _make_span("A", 10, 5, 30, 15),
            _make_span("B", 50, 8, 80, 18),
        ]
        lines = cluster_by_y(spans, tolerance=5.0)
        assert len(lines) == 1
        bbox = lines[0].bbox
        assert bbox.x0 == 10
        assert bbox.x1 == 80
        assert bbox.y0 == 5
        assert bbox.y1 == 18


class TestMergeTouchingSegments:
    def test_merge_overlapping_horizontal(self) -> None:
        segments = [
            LineSegment(x0=0, y0=50, x1=60, y1=50),
            LineSegment(x0=50, y0=50, x1=120, y1=50),
        ]
        merged = merge_touching_segments(segments, gap_tolerance=2.0)
        assert len(merged) == 1
        assert merged[0].x0 == 0
        assert merged[0].x1 == 120

    def test_merge_adjacent_horizontal(self) -> None:
        segments = [
            LineSegment(x0=0, y0=50, x1=60, y1=50),
            LineSegment(x0=61, y0=50, x1=120, y1=50),
        ]
        merged = merge_touching_segments(segments, gap_tolerance=2.0)
        assert len(merged) == 1

    def test_no_merge_far_horizontal(self) -> None:
        segments = [
            LineSegment(x0=0, y0=50, x1=30, y1=50),
            LineSegment(x0=100, y0=50, x1=130, y1=50),
        ]
        merged = merge_touching_segments(segments, gap_tolerance=2.0)
        assert len(merged) == 2

    def test_merge_vertical(self) -> None:
        segments = [
            LineSegment(x0=50, y0=0, x1=50, y1=60),
            LineSegment(x0=50, y0=55, x1=50, y1=120),
        ]
        merged = merge_touching_segments(segments, gap_tolerance=2.0)
        assert len(merged) == 1
        assert merged[0].y0 == 0
        assert merged[0].y1 == 120

    def test_empty_input(self) -> None:
        assert merge_touching_segments([]) == []

    def test_different_y_not_merged(self) -> None:
        segments = [
            LineSegment(x0=0, y0=50, x1=100, y1=50),
            LineSegment(x0=0, y0=100, x1=100, y1=100),
        ]
        merged = merge_touching_segments(segments, gap_tolerance=2.0)
        assert len(merged) == 2


class TestClassifySegments:
    def test_basic_classification(self) -> None:
        segments = [
            LineSegment(x0=0, y0=50, x1=100, y1=50),  # horizontal
            LineSegment(x0=50, y0=0, x1=50, y1=100),  # vertical
            LineSegment(x0=0, y0=0, x1=100, y1=100),  # diagonal
        ]
        h, v = classify_segments(segments, min_length=10.0)
        assert len(h) == 1
        assert len(v) == 1

    def test_filters_short_segments(self) -> None:
        segments = [
            LineSegment(x0=0, y0=50, x1=5, y1=50),  # too short
            LineSegment(x0=0, y0=50, x1=50, y1=50),  # long enough
        ]
        h, v = classify_segments(segments, min_length=10.0)
        assert len(h) == 1
        assert len(v) == 0

    def test_empty_input(self) -> None:
        h, v = classify_segments([])
        assert h == []
        assert v == []


class TestSortReadingOrder:
    def test_top_to_bottom_left_to_right(self) -> None:
        # Create elements at different positions
        def _elem(x: float, y: float) -> PageElement:
            bbox = BBox(x0=x, y0=y, x1=x + 20, y1=y + 10)
            line = TextLine(
                spans=[TextSpan(text=f"({x},{y})", bbox=bbox)],
                bbox=bbox,
            )
            return PageElement(element_type="text_line", text_line=line, bbox=bbox)

        elements = [
            _elem(100, 10),  # top-right
            _elem(0, 50),  # bottom-left
            _elem(0, 10),  # top-left
            _elem(100, 50),  # bottom-right
        ]

        sorted_elems = sort_reading_order(elements, y_tolerance=5.0)
        positions = [(e.bbox.x0, e.bbox.y0) for e in sorted_elems]
        assert positions == [(0, 10), (100, 10), (0, 50), (100, 50)]

    def test_empty_input(self) -> None:
        assert sort_reading_order([]) == []

    def test_single_element(self) -> None:
        bbox = BBox(x0=0, y0=0, x1=20, y1=10)
        elem = PageElement(element_type="text_line", bbox=bbox)
        result = sort_reading_order([elem])
        assert len(result) == 1
