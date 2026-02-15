"""Pydantic data models for PDF parsing.

All models are pure data — no fitz/PyMuPDF dependency.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box with (x0, y0) as top-left and (x1, y1) as bottom-right."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def mid_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def mid_x(self) -> float:
        return (self.x0 + self.x1) / 2

    def overlaps_y(self, other: "BBox", tolerance: float = 0.0) -> bool:
        """Check if two bboxes overlap vertically within tolerance."""
        return self.y0 - tolerance <= other.y1 and other.y0 - tolerance <= self.y1

    def contains(self, other: "BBox") -> bool:
        """Check if this bbox fully contains another."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point falls inside this bbox."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


class TextSpan(BaseModel):
    """A single word or text span extracted from a PDF page."""

    text: str
    bbox: BBox
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False


class TextLine(BaseModel):
    """A logical line of text = Y-clustered TextSpans sorted by X."""

    spans: List[TextSpan]
    bbox: BBox

    @property
    def text(self) -> str:
        return " ".join(span.text for span in self.spans)


class LineSegment(BaseModel):
    """A vector line segment from PDF drawing commands."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def is_horizontal(self) -> bool:
        return abs(self.y1 - self.y0) < 1.0

    @property
    def is_vertical(self) -> bool:
        return abs(self.x1 - self.x0) < 1.0

    @property
    def length(self) -> float:
        return ((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2) ** 0.5


class GridCell(BaseModel):
    """A single cell in a detected table grid."""

    row: int
    col: int
    bbox: BBox
    text: str = ""


class Table(BaseModel):
    """A structured table extracted from a PDF page."""

    headers: List[str]
    rows: List[List[str]]
    bbox: BBox
    confidence: float = 1.0


class CheckboxState(str, Enum):
    """State of a detected checkbox."""

    CHECKED = "checked"
    UNCHECKED = "unchecked"


class Checkbox(BaseModel):
    """A detected checkbox with state and label."""

    state: CheckboxState
    bbox: BBox
    label: str = ""


class PageElement(BaseModel):
    """Union wrapper for reading-order traversal of page content."""

    element_type: str  # "text_line", "table", "checkbox"
    text_line: Optional[TextLine] = None
    table: Optional[Table] = None
    checkbox: Optional[Checkbox] = None
    bbox: BBox


class ParsedPage(BaseModel):
    """Full parsed result of a single PDF page."""

    page_number: int
    width: float
    height: float
    elements: List[PageElement] = Field(default_factory=list)
    text_lines: List[TextLine] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    checkboxes: List[Checkbox] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Full parsed result of a PDF document."""

    source: str
    total_pages: int
    pages: List[ParsedPage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
