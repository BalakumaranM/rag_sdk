# Document Loading

## The Document Model

```python
from rag_sdk.document import Document

doc = Document(
    content="Some text...",
    metadata={"source": "file.txt", "author": "Alice"},
)
print(doc.id)       # auto-generated UUID
print(doc.content)  # "Some text..."
print(doc.metadata) # {"source": "file.txt", "author": "Alice"}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | `uuid4()` | Unique identifier |
| `content` | `str` | required | Text content |
| `metadata` | `Dict[str, Any]` | `{}` | Arbitrary metadata |

## DocumentLoader

`DocumentLoader` provides static methods for loading files and directories.

### load_file

```python
from rag_sdk.document import DocumentLoader

# Text/Markdown files → single Document
doc = DocumentLoader.load_file("notes.txt")

# PDF files → list of Documents (one per page by default)
docs = DocumentLoader.load_file("report.pdf")
```

**Signature:**

```python
@staticmethod
def load_file(
    file_path: str,
    pdf_parser: Optional[BasePDFParser] = None,
    one_doc_per_page: bool = True,
) -> Union[Document, List[Document]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | required | Path to the file |
| `pdf_parser` | `Optional[BasePDFParser]` | `None` | PDF parser instance; creates default `PyMuPDFParser` if None |
| `one_doc_per_page` | `bool` | `True` | For PDFs: one Document per page vs. single Document |

**Returns:** `Document` for `.txt`/`.md` files, `List[Document]` for `.pdf` files.

**Metadata for text files:**

```python
{"source": "/path/to/file.txt", "filename": "file.txt", "extension": ".txt"}
```

**Metadata for PDF pages:**

```python
{"source": "report.pdf", "page_number": 0, "tables": "[...]"}
```

### load_directory

```python
docs = DocumentLoader.load_directory(
    "./data",
    extensions=[".txt", ".md", ".pdf"],
    pdf_parser=my_parser,
)
```

**Signature:**

```python
@classmethod
def load_directory(
    cls,
    directory_path: str,
    extensions: List[str] = [".txt", ".md"],
    pdf_parser: Optional[BasePDFParser] = None,
) -> List[Document]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory_path` | `str` | required | Directory to scan recursively |
| `extensions` | `List[str]` | `[".txt", ".md"]` | File extensions to include |
| `pdf_parser` | `Optional[BasePDFParser]` | `None` | PDF parser for `.pdf` files |

Recursively scans the directory. Files that fail to load are logged and skipped.

## PDF Parsing

The SDK provides two PDF parser backends:

### PyMuPDF (Default)

Fast, rule-based parser built on PyMuPDF/fitz. Extracts text lines, tables (via line segment detection), and checkboxes.

```yaml
document_processing:
  pdf_parser:
    backend: "pymupdf"
    one_document_per_page: true
    include_tables_in_text: true
```

### Docling

ML-powered parser from IBM using RT-DETR for layout analysis and TableFormer for table extraction. Better for complex layouts, borderless tables, and scanned PDFs.

```bash
pip install rag_sdk[docling]
```

```yaml
document_processing:
  pdf_parser:
    backend: "docling"
    docling_do_ocr: true
    docling_do_table_structure: true
    docling_table_mode: "accurate"
```

### Comparison

| Feature | PyMuPDF | Docling |
|---------|---------|---------|
| Speed | Fast | Slower (ML models) |
| Install size | Small (included) | Large (~2GB models) |
| Simple layouts | Excellent | Excellent |
| Complex layouts | Good | Better |
| Borderless tables | Limited | Good |
| Scanned PDFs (OCR) | No | Yes |
| Checkboxes | Yes | No |

### Using via RAG class

```python
# The RAG class handles PDF parsing automatically
stats = rag.ingest_pdf("report.pdf")
```

The `ingest_pdf` method uses the parser configured in `document_processing.pdf_parser`, loads the file via `DocumentLoader`, then feeds the resulting documents through the full ingestion pipeline (split → embed → store).

## PDF Data Models

The PDF parser produces structured data:

- `ParsedDocument` — full document with list of `ParsedPage`
- `ParsedPage` — page with `elements` (reading order), `text_lines`, `tables`, `checkboxes`
- `PageElement` — union of `TextLine`, `Table`, or `Checkbox` with bounding box
- `Table` — `headers: List[str]`, `rows: List[List[str]]`
- `Checkbox` — `state: CheckboxState` (checked/unchecked), `label: str`
- `BBox` — bounding box with `x0, y0, x1, y1` and helper methods

Tables are serialized as JSON in document metadata for downstream use.

## See Also

- [Text Splitting](21-text-splitting.md) — chunking strategies
- [API: Document](32-api-document.md) — full API reference
