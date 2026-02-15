from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union
from pathlib import Path
import logging
from .models import Document

if TYPE_CHECKING:
    from .base import BasePDFParser

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from files and directories.

    Supported extensions: .txt, .md, .pdf
    """

    @staticmethod
    def load_file(
        file_path: str,
        pdf_parser: Optional[BasePDFParser] = None,
        one_doc_per_page: bool = True,
    ) -> Union[Document, List[Document]]:
        """Load a single file.

        For .txt and .md files, returns a single Document.
        For .pdf files, returns a list of Documents (one per page by default).

        Args:
            file_path: Path to the file to load.
            pdf_parser: Optional PDF parser instance for .pdf files.
                If not provided and file is .pdf, a default PyMuPDFParser is created.
            one_doc_per_page: For PDFs, whether to create one Document per page.

        Returns:
            A Document for text files, or a list of Documents for PDFs.
        """
        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            return DocumentLoader._load_pdf(file_path, pdf_parser, one_doc_per_page)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "extension": path.suffix,
            },
        )

    @staticmethod
    def _load_pdf(
        file_path: str,
        pdf_parser: Optional[BasePDFParser] = None,
        one_doc_per_page: bool = True,
    ) -> List[Document]:
        """Load a PDF file using the PDF parser.

        Args:
            file_path: Path to the PDF file.
            pdf_parser: Optional parser instance. Creates default if None.
            one_doc_per_page: Whether to create one Document per page.

        Returns:
            List of Documents extracted from the PDF.
        """
        if pdf_parser is None:
            from .pdf_parser import PyMuPDFParser

            pdf_parser = PyMuPDFParser()

        parsed = pdf_parser.parse_file(file_path)
        return pdf_parser.to_documents(parsed, one_doc_per_page=one_doc_per_page)

    @classmethod
    def load_directory(
        cls,
        directory_path: str,
        extensions: List[str] = [".txt", ".md"],
        pdf_parser: Optional[BasePDFParser] = None,
    ) -> List[Document]:
        """Load all supported files from a directory recursively.

        Args:
            directory_path: Path to the directory.
            extensions: File extensions to load.
            pdf_parser: Optional PDF parser for .pdf files.

        Returns:
            List of loaded Documents.
        """
        documents: List[Document] = []
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    result = cls.load_file(str(file_path), pdf_parser=pdf_parser)
                    if isinstance(result, list):
                        documents.extend(result)
                    else:
                        documents.append(result)
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")

        return documents
