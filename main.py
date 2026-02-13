import os
import logging
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader
from rag_sdk.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 1. Load Configuration
    # Ensure you have OPENAI_API_KEY in your environment or .env file
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("Please set OPENAI_API_KEY environment variable.")
        return

    logger.info("Loading configuration...")
    # For this example, we'll use the default config which uses InMemoryVectorStore
    try:
        config = ConfigLoader.from_yaml("config.yaml")
    except FileNotFoundError:
        logger.warning("config.yaml not found, using default env config")
        config = ConfigLoader.from_env()

    # 2. Initialize RAG
    logger.info("Initializing RAG SDK...")
    rag = RAG(config)

    # 3. Create/Load Documents
    logger.info("Loading documents...")
    # Option A: specific file
    # doc = DocumentLoader.load_file("sample.txt")
    # documents = [doc]

    # Option B: Manual creation (for demo simplicity)
    with open("sample.txt", "r") as f:
        text = f.read()

    documents = [Document(content=text, metadata={"source": "sample.txt"})]

    # 4. Ingest Documents
    logger.info("Ingesting documents...")
    stats = rag.ingest_documents(documents)
    logger.info(f"Ingestion stats: {stats}")

    # 5. Query
    query = "What are the key components of the RAG SDK?"
    logger.info(f"Querying: {query}")

    result = rag.query(query)

    logger.info("--- Answer ---")
    logger.info(result["answer"])

    logger.info("--- Sources ---")
    for doc in result["sources"]:
        logger.info(f"- {doc.metadata.get('source', 'unknown')}: {doc.content[:50]}...")


if __name__ == "__main__":
    main()
