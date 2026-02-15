"""Tests for PDF data models."""

from rag_sdk.document.pdf_models import (
    BBox,
    Checkbox,
    CheckboxState,
    GridCell,
    LineSegment,
    PageElement,
    ParsedDocument,
    ParsedPage,
    Table,
    TextLine,
    TextSpan,
)


class TestBBox:
    def test_width_and_height(self) -> None:
        bbox = BBox(x0=10, y0=20, x1=50, y1=60)
        assert bbox.width == 40
        assert bbox.height == 40

    def test_mid_y_and_mid_x(self) -> None:
        bbox = BBox(x0=0, y0=10, x1=100, y1=30)
        assert bbox.mid_y == 20.0
        assert bbox.mid_x == 50.0

    def test_overlaps_y(self) -> None:
        a = BBox(x0=0, y0=10, x1=100, y1=20)
        b = BBox(x0=0, y0=15, x1=100, y1=25)
        c = BBox(x0=0, y0=25, x1=100, y1=35)

        assert a.overlaps_y(b)
        assert not a.overlaps_y(c)
        assert a.overlaps_y(c, tolerance=5.0)

    def test_contains(self) -> None:
        outer = BBox(x0=0, y0=0, x1=100, y1=100)
        inner = BBox(x0=10, y0=10, x1=50, y1=50)
        outside = BBox(x0=110, y0=110, x1=200, y1=200)

        assert outer.contains(inner)
        assert not inner.contains(outer)
        assert not outer.contains(outside)

    def test_contains_point(self) -> None:
        bbox = BBox(x0=10, y0=20, x1=50, y1=60)
        assert bbox.contains_point(30, 40)
        assert bbox.contains_point(10, 20)  # edge
        assert not bbox.contains_point(5, 40)


class TestTextSpan:
    def test_creation(self) -> None:
        span = TextSpan(
            text="hello",
            bbox=BBox(x0=0, y0=0, x1=30, y1=10),
            font_name="Arial",
            font_size=12.0,
            is_bold=True,
        )
        assert span.text == "hello"
        assert span.is_bold is True
        assert span.is_italic is False


class TestTextLine:
    def test_text_property(self) -> None:
        spans = [
            TextSpan(text="Hello", bbox=BBox(x0=0, y0=0, x1=30, y1=10)),
            TextSpan(text="world", bbox=BBox(x0=35, y0=0, x1=65, y1=10)),
        ]
        line = TextLine(spans=spans, bbox=BBox(x0=0, y0=0, x1=65, y1=10))
        assert line.text == "Hello world"

    def test_empty_spans(self) -> None:
        line = TextLine(spans=[], bbox=BBox(x0=0, y0=0, x1=0, y1=0))
        assert line.text == ""


class TestLineSegment:
    def test_horizontal(self) -> None:
        seg = LineSegment(x0=0, y0=50, x1=100, y1=50)
        assert seg.is_horizontal is True
        assert seg.is_vertical is False

    def test_vertical(self) -> None:
        seg = LineSegment(x0=50, y0=0, x1=50, y1=100)
        assert seg.is_horizontal is False
        assert seg.is_vertical is True

    def test_diagonal(self) -> None:
        seg = LineSegment(x0=0, y0=0, x1=100, y1=100)
        assert seg.is_horizontal is False
        assert seg.is_vertical is False

    def test_length(self) -> None:
        seg = LineSegment(x0=0, y0=0, x1=3, y1=4)
        assert seg.length == 5.0


class TestCheckboxState:
    def test_enum_values(self) -> None:
        assert CheckboxState.CHECKED == "checked"
        assert CheckboxState.UNCHECKED == "unchecked"


class TestTable:
    def test_creation(self) -> None:
        table = Table(
            headers=["Name", "Age"],
            rows=[["Alice", "30"], ["Bob", "25"]],
            bbox=BBox(x0=0, y0=0, x1=200, y1=100),
            confidence=0.95,
        )
        assert len(table.headers) == 2
        assert len(table.rows) == 2
        assert table.confidence == 0.95


class TestPageElement:
    def test_text_line_element(self) -> None:
        line = TextLine(
            spans=[TextSpan(text="test", bbox=BBox(x0=0, y0=0, x1=20, y1=10))],
            bbox=BBox(x0=0, y0=0, x1=20, y1=10),
        )
        elem = PageElement(
            element_type="text_line",
            text_line=line,
            bbox=BBox(x0=0, y0=0, x1=20, y1=10),
        )
        assert elem.element_type == "text_line"
        assert elem.text_line is not None
        assert elem.table is None


class TestParsedDocument:
    def test_creation(self) -> None:
        page = ParsedPage(page_number=0, width=612, height=792)
        doc = ParsedDocument(source="test.pdf", total_pages=1, pages=[page])
        assert doc.total_pages == 1
        assert len(doc.pages) == 1


class TestModelSerialization:
    def test_bbox_round_trip(self) -> None:
        bbox = BBox(x0=1.5, y0=2.5, x1=3.5, y1=4.5)
        data = bbox.model_dump()
        restored = BBox(**data)
        assert restored == bbox

    def test_grid_cell_serialization(self) -> None:
        cell = GridCell(row=0, col=1, bbox=BBox(x0=0, y0=0, x1=50, y1=20), text="value")
        data = cell.model_dump()
        assert data["row"] == 0
        assert data["text"] == "value"

    def test_checkbox_serialization(self) -> None:
        cb = Checkbox(
            state=CheckboxState.CHECKED,
            bbox=BBox(x0=10, y0=10, x1=20, y1=20),
            label="Accept terms",
        )
        data = cb.model_dump()
        assert data["state"] == "checked"
        restored = Checkbox(**data)
        assert restored.state == CheckboxState.CHECKED
