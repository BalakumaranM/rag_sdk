# API Reference: Document, Splitters & PDF

```python
from rag_sdk.document import (
    Document, DocumentLoader,
    BaseTextSplitter, BasePDFParser,
    TextSplitter, AgenticSplitter, PropositionSplitter, SemanticSplitter, LateSplitter,
    PyMuPDFParser,
    # PDF data models
    ParsedDocument, ParsedPage, PageElement,
    TextLine, TextSpan, Table, Checkbox, CheckboxState, BBox,
    LineSegment, GridCell,
    # PDF geometry utilities
    cluster_by_y, merge_touching_segments, classify_segments, sort_reading_order,
    # PDF table extraction
    TableGridDetector, TableExtractor,
)
```

---

## Document

```python
class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

## DocumentLoader

### load_file

```python
@staticmethod
def load_file(
    file_path: str,
    pdf_parser: Optional[BasePDFParser] = None,
    one_doc_per_page: bool = True,
) -> Union[Document, List[Document]]
```

Returns `Document` for `.txt`/`.md`, `List[Document]` for `.pdf`.

### load_directory

```python
@classmethod
def load_directory(
    cls,
    directory_path: str,
    extensions: List[str] = [".txt", ".md"],
    pdf_parser: Optional[BasePDFParser] = None,
) -> List[Document]
```

Recursively loads files matching the given extensions.

---

## Base Classes

### BaseTextSplitter (ABC)

```python
class BaseTextSplitter(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]: ...
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]: ...
```

### BasePDFParser (ABC)

```python
class BasePDFParser(ABC):
    @abstractmethod
    def parse_file(self, file_path: str, pages: Optional[List[int]] = None) -> ParsedDocument: ...
    @abstractmethod
    def parse_page(self, page: Any, page_number: int) -> ParsedPage: ...
    def to_documents(self, parsed: ParsedDocument, one_doc_per_page: bool = True) -> List[Document]: ...
```

`to_documents()` is concrete — converts `ParsedDocument` to `List[Document]`, serializing tables as JSON in metadata and rendering checkboxes as `[x]`/`[ ]`.

---

## Splitters

### TextSplitter (Recursive)

```python
class TextSplitter(BaseTextSplitter):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,  # default: ["\n\n", "\n", " ", ""]
    ): ...
```

### AgenticSplitter

```python
class AgenticSplitter(BaseTextSplitter):
    def __init__(
        self,
        llm_provider: LLMProvider,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.5,
    ): ...
```

Adds `"chunking_strategy": "agentic"` to chunk metadata.

### PropositionSplitter

```python
class PropositionSplitter(BaseTextSplitter):
    def __init__(
        self,
        llm_provider: LLMProvider,
        max_propositions_per_chunk: int = 5,
    ): ...
```

Adds `"chunking_strategy": "proposition"` to chunk metadata.

### SemanticSplitter

```python
class SemanticSplitter(BaseTextSplitter):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: SemanticChunkingConfig,
    ): ...
```

### LateSplitter

```python
class LateSplitter(BaseTextSplitter):
    def __init__(self, config: LateChunkingConfig): ...
```

Requires `transformers` and `torch`. Adds `"late_embedding": List[float]` to chunk metadata.

---

## PDF Parsers

### PyMuPDFParser

```python
class PyMuPDFParser(BasePDFParser):
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
    ): ...
```

### DoclingParser

```python
class DoclingParser(BasePDFParser):
    def __init__(
        self,
        do_ocr: bool = True,
        do_table_structure: bool = True,
        table_mode: str = "accurate",  # "accurate" | "fast"
        timeout: Optional[float] = None,
    ): ...
```

Requires `pip install rag_sdk[docling]`.

---

## PDF Data Models

### ParsedDocument

```python
class ParsedDocument(BaseModel):
    source: str
    total_pages: int
    pages: List[ParsedPage] = []
    metadata: Dict[str, Any] = {}
```

### ParsedPage

```python
class ParsedPage(BaseModel):
    page_number: int
    width: float
    height: float
    elements: List[PageElement] = []      # reading order
    text_lines: List[TextLine] = []
    tables: List[Table] = []
    checkboxes: List[Checkbox] = []
```

### PageElement

```python
class PageElement(BaseModel):
    element_type: str  # "text_line" | "table" | "checkbox"
    text_line: Optional[TextLine] = None
    table: Optional[Table] = None
    checkbox: Optional[Checkbox] = None
    bbox: BBox
```

### TextLine / TextSpan

```python
class TextLine(BaseModel):
    spans: List[TextSpan]
    bbox: BBox
    @property
    def text(self) -> str: ...  # joined span texts

class TextSpan(BaseModel):
    text: str
    bbox: BBox
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
```

### Table

```python
class Table(BaseModel):
    headers: List[str]
    rows: List[List[str]]
    bbox: BBox
    confidence: float = 1.0
```

### Checkbox / CheckboxState

```python
class CheckboxState(str, Enum):
    CHECKED = "checked"
    UNCHECKED = "unchecked"

class Checkbox(BaseModel):
    state: CheckboxState
    bbox: BBox
    label: str = ""
```

### BBox

```python
class BBox(BaseModel):
    x0: float; y0: float; x1: float; y1: float

    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...
    @property
    def mid_y(self) -> float: ...
    @property
    def mid_x(self) -> float: ...
    def overlaps_y(self, other: BBox, tolerance: float = 0.0) -> bool: ...
    def contains(self, other: BBox) -> bool: ...
    def contains_point(self, x: float, y: float) -> bool: ...
```

## See Also

- [Document Loading](20-document-loading.md) — usage guide
- [Text Splitting](21-text-splitting.md) — splitter guide
