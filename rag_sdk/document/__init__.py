from .models import Document
from .loader import DocumentLoader
from .base import BaseTextSplitter, BasePDFParser
from .splitter import TextSplitter
from .agentic_splitter import AgenticSplitter
from .proposition_splitter import PropositionSplitter
from .semantic_splitter import SemanticSplitter
from .late_splitter import LateSplitter
from .pdf_models import (
    BBox,
    TextSpan,
    TextLine,
    LineSegment,
    GridCell,
    Table,
    CheckboxState,
    Checkbox,
    PageElement,
    ParsedPage,
    ParsedDocument,
)
from .pdf_geometry import (
    cluster_by_y,
    merge_touching_segments,
    classify_segments,
    sort_reading_order,
)
from .pdf_table_extractor import TableGridDetector, TableExtractor
from .pdf_parser import PyMuPDFParser

__all__ = [
    "Document",
    "DocumentLoader",
    "BaseTextSplitter",
    "BasePDFParser",
    "TextSplitter",
    "AgenticSplitter",
    "PropositionSplitter",
    "SemanticSplitter",
    "LateSplitter",
    "BBox",
    "TextSpan",
    "TextLine",
    "LineSegment",
    "GridCell",
    "Table",
    "CheckboxState",
    "Checkbox",
    "PageElement",
    "ParsedPage",
    "ParsedDocument",
    "cluster_by_y",
    "merge_touching_segments",
    "classify_segments",
    "sort_reading_order",
    "TableGridDetector",
    "TableExtractor",
    "PyMuPDFParser",
]

try:
    from .docling_parser import DoclingParser  # noqa: F401

    __all__.append("DoclingParser")
except ImportError:
    pass
