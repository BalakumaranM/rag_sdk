from abc import ABC, abstractmethod
import json
from typing import Any, List, Optional
from .models import Document
from .pdf_models import ParsedDocument, ParsedPage


class BaseTextSplitter(ABC):
    """Base class for all text splitting strategies."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split a single text string into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        pass

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks, preserving metadata.

        Args:
            documents: Documents to split.

        Returns:
            List of chunked documents with parent metadata.
        """
        pass


class BasePDFParser(ABC):
    """Base class for PDF parsing strategies.

    Subclasses implement the extraction logic; this base provides the
    concrete `to_documents()` conversion for the RAG pipeline.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def parse_page(self, page: Any, page_number: int) -> ParsedPage:
        """Parse a single PDF page.

        Args:
            page: Backend-specific page object (e.g., fitz.Page).
            page_number: Zero-indexed page number.

        Returns:
            ParsedPage with extracted elements.
        """
        pass

    def to_documents(
        self, parsed: ParsedDocument, one_doc_per_page: bool = True
    ) -> List[Document]:
        """Convert ParsedDocument to List[Document] for the RAG pipeline.

        Tables are serialized as JSON in metadata. Checkboxes are rendered
        as [x] or [ ] in the content text.

        Args:
            parsed: The parsed PDF document.
            one_doc_per_page: If True, create one Document per page.
                If False, create a single Document for the whole PDF.

        Returns:
            List of Document objects ready for splitting and embedding.
        """
        if one_doc_per_page:
            return [
                self._page_to_document(page, parsed.source) for page in parsed.pages
            ]
        else:
            all_text_parts: List[str] = []
            all_tables: List[dict[str, Any]] = []
            for page in parsed.pages:
                all_text_parts.append(self._page_to_text(page))
                all_tables.extend(self._tables_to_dicts(page))

            metadata: dict[str, Any] = {
                "source": parsed.source,
                "total_pages": parsed.total_pages,
            }
            if all_tables:
                metadata["tables"] = json.dumps(all_tables)

            return [
                Document(
                    content="\n\n".join(all_text_parts),
                    metadata=metadata,
                )
            ]

    def _page_to_document(self, page: ParsedPage, source: str) -> Document:
        """Convert a single ParsedPage to a Document."""
        content = self._page_to_text(page)
        tables = self._tables_to_dicts(page)

        metadata: dict[str, Any] = {
            "source": source,
            "page_number": page.page_number,
        }
        if tables:
            metadata["tables"] = json.dumps(tables)

        return Document(content=content, metadata=metadata)

    def _page_to_text(self, page: ParsedPage) -> str:
        """Render a ParsedPage as plain text."""
        parts: List[str] = []

        for element in page.elements:
            if element.element_type == "text_line" and element.text_line:
                parts.append(element.text_line.text)
            elif element.element_type == "table" and element.table:
                table = element.table
                if table.headers:
                    parts.append(" | ".join(table.headers))
                    parts.append(" | ".join("---" for _ in table.headers))
                for row in table.rows:
                    parts.append(" | ".join(row))
            elif element.element_type == "checkbox" and element.checkbox:
                cb = element.checkbox
                marker = "[x]" if cb.state.value == "checked" else "[ ]"
                parts.append(f"{marker} {cb.label}")

        return "\n".join(parts)

    def _tables_to_dicts(self, page: ParsedPage) -> List[dict[str, Any]]:
        """Convert tables from a page to serializable dicts."""
        return [
            {
                "page": page.page_number,
                "headers": table.headers,
                "rows": table.rows,
            }
            for table in page.tables
        ]
