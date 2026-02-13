from typing import List
from pathlib import Path
import logging
from .models import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads documents from a directory.
    Supported extensions: .txt, .md
    """

    @staticmethod
    def load_file(file_path: str) -> Document:
        """
        Load a single file.
        """
        path = Path(file_path)
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

    @classmethod
    def load_directory(
        cls, directory_path: str, extensions: List[str] = [".txt", ".md"]
    ) -> List[Document]:
        """
        Load all supported files from a directory recursively.
        """
        documents = []
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    doc = cls.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")

        return documents
