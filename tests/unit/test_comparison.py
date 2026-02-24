import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Load example document
text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([text])
logger.debug("texts: %s", texts)
logger.debug("texts[0]: %s", texts[0])
logger.debug("texts[1]: %s", texts[1])
