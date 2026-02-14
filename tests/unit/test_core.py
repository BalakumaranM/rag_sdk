from unittest.mock import MagicMock, patch
import pytest
from rag_sdk.config import ConfigLoader, Config
from rag_sdk.document import Document
from rag_sdk.document.base import BaseTextSplitter
from rag_sdk.document.splitter import TextSplitter
from rag_sdk.document.agentic_splitter import AgenticSplitter
from rag_sdk.document.proposition_splitter import PropositionSplitter
from rag_sdk.retrieval.base import BaseRetriever
from rag_sdk.retrieval.retriever import Retriever
from rag_sdk.retrieval.graph_rag import GraphRAGRetriever, Entity
from rag_sdk.retrieval.raptor import RAPTORRetriever, _kmeans
from rag_sdk.retrieval.corrective_rag import CorrectiveRAGRetriever
from rag_sdk.generation.standard import StandardGeneration
from rag_sdk.generation.cove import ChainOfVerificationGeneration
from rag_sdk.generation.attributed import AttributedGeneration


def test_dummy():
    config = ConfigLoader.from_env()
    assert config is not None


def test_rag_init():
    assert True


# --- Config Tests ---


def test_config_defaults():
    config = Config()
    assert config.document_processing.chunking.strategy == "recursive"
    assert config.retrieval.strategy == "dense"
    assert config.retrieval.corrective_rag_enabled is False
    assert config.generation.strategy == "standard"


def test_config_new_fields():
    config = Config()
    assert config.retrieval.graph_rag.max_entities_per_chunk == 10
    assert config.retrieval.raptor.num_levels == 3
    assert config.retrieval.corrective_rag.relevance_threshold == 0.7
    assert config.cove.max_verification_questions == 3
    assert config.attributed_generation.citation_style == "numeric"
    assert config.document_processing.agentic_chunking.max_chunk_size == 1000
    assert (
        config.document_processing.proposition_chunking.max_propositions_per_chunk == 5
    )


def test_config_from_dict_with_new_fields():
    config = Config(
        **{
            "document_processing": {
                "chunking": {"strategy": "agentic"},
                "agentic_chunking": {"max_chunk_size": 500},
            },
            "retrieval": {
                "strategy": "graph_rag",
                "corrective_rag_enabled": True,
            },
            "generation": {"strategy": "cove"},
            "cove": {"max_verification_questions": 5},
        }
    )
    assert config.document_processing.chunking.strategy == "agentic"
    assert config.document_processing.agentic_chunking.max_chunk_size == 500
    assert config.retrieval.strategy == "graph_rag"
    assert config.retrieval.corrective_rag_enabled is True
    assert config.generation.strategy == "cove"
    assert config.cove.max_verification_questions == 5


def test_config_yaml_with_new_fields(tmp_path):
    yaml_content = """
project_name: "test"
document_processing:
  chunking:
    strategy: "proposition"
retrieval:
  strategy: "raptor"
  corrective_rag_enabled: true
generation:
  strategy: "attributed"
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)
    config = ConfigLoader.from_yaml(str(yaml_file))
    assert config.document_processing.chunking.strategy == "proposition"
    assert config.retrieval.strategy == "raptor"
    assert config.retrieval.corrective_rag_enabled is True
    assert config.generation.strategy == "attributed"


# --- Base ABC Tests ---


def test_text_splitter_inherits_base():
    assert issubclass(TextSplitter, BaseTextSplitter)


def test_retriever_inherits_base():
    assert issubclass(Retriever, BaseRetriever)


# --- TextSplitter Tests ---


def test_text_splitter_basic():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = splitter.split_text(text)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) > 0


def test_text_splitter_documents():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
    docs = [Document(content="A short text.", metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "test"
    assert "parent_id" in chunks[0].metadata


# --- Agentic Splitter Tests ---


def test_agentic_splitter_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[0, 2]"

    splitter = AgenticSplitter(llm_provider=mock_llm, max_chunk_size=500)
    text = "First topic sentence. More on first topic. Second topic begins here. And continues."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1


def test_agentic_splitter_fallback_on_failure():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("LLM failed")

    splitter = AgenticSplitter(llm_provider=mock_llm, max_chunk_size=50)
    text = "A" * 100
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1


def test_agentic_splitter_short_text():
    mock_llm = MagicMock()
    splitter = AgenticSplitter(llm_provider=mock_llm, max_chunk_size=500)
    chunks = splitter.split_text("Short text.")
    assert chunks == ["Short text."]
    mock_llm.generate.assert_not_called()


def test_agentic_splitter_split_documents():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[0]"

    splitter = AgenticSplitter(llm_provider=mock_llm, max_chunk_size=5000)
    docs = [
        Document(
            content="Some content here. More content.", metadata={"source": "test"}
        )
    ]
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunking_strategy"] == "agentic"


# --- Proposition Splitter Tests ---


def test_proposition_splitter_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '["Prop one.", "Prop two.", "Prop three."]'

    splitter = PropositionSplitter(llm_provider=mock_llm, max_propositions_per_chunk=2)
    chunks = splitter.split_text("Some text about multiple things.")
    assert len(chunks) == 2  # 3 propositions / 2 per chunk = 2 chunks


def test_proposition_splitter_fallback():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("LLM failed")

    splitter = PropositionSplitter(llm_provider=mock_llm, max_propositions_per_chunk=2)
    text = "First sentence. Second sentence. Third sentence."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1


def test_proposition_splitter_split_documents():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '["Fact one.", "Fact two."]'

    splitter = PropositionSplitter(llm_provider=mock_llm, max_propositions_per_chunk=5)
    docs = [Document(content="Some content.", metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunking_strategy"] == "proposition"


# --- Graph RAG Tests ---


def test_graph_rag_build_graph():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        '{"entities": [{"name": "Python", "type": "Language"}, '
        '{"name": "Guido", "type": "Person"}], '
        '"relationships": [{"source": "Guido", "target": "Python", "relation": "created"}]}'
    )

    mock_embedding = MagicMock()
    mock_vectorstore = MagicMock()
    config = Config().retrieval

    retriever = GraphRAGRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vectorstore,
        llm_provider=mock_llm,
        config=config,
    )

    docs = [Document(content="Guido created Python.", metadata={})]
    retriever.build_graph(docs)

    assert "python" in retriever.entities
    assert "guido" in retriever.entities
    assert len(retriever.relationships) == 1
    assert "python" in retriever.adjacency.get("guido", set())


def test_graph_rag_retrieve():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        '["python"]',  # query entity extraction
    ]

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 10

    doc = Document(content="Python is great.", metadata={})
    mock_vectorstore = MagicMock()
    mock_vectorstore.search.return_value = [(doc, 0.9)]

    config = Config().retrieval
    retriever = GraphRAGRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vectorstore,
        llm_provider=mock_llm,
        config=config,
    )
    # Pre-populate graph
    retriever.entities["python"] = Entity(
        name="python", entity_type="Language", document_ids=[doc.id]
    )
    retriever.adjacency["python"] = set()

    results = retriever.retrieve("Tell me about Python", top_k=5)
    assert len(results) >= 1


# --- RAPTOR Tests ---


def test_kmeans_basic():
    import numpy as np

    vectors = np.array(
        [[1, 0], [1, 1], [0, 1], [10, 10], [10, 11], [11, 10]], dtype=float
    )
    assignments = _kmeans(vectors, k=2)
    assert len(assignments) == 6
    # Points 0-2 should be in one cluster, 3-5 in another
    assert assignments[0] == assignments[1] == assignments[2]
    assert assignments[3] == assignments[4] == assignments[5]
    assert assignments[0] != assignments[3]


def test_raptor_build_tree():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Summary of cluster."

    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    mock_vectorstore = MagicMock()
    config = Config(
        retrieval={
            "strategy": "raptor",
            "raptor": {"num_levels": 1, "max_clusters_per_level": 2},
        }
    ).retrieval

    retriever = RAPTORRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vectorstore,
        llm_provider=mock_llm,
        config=config,
    )

    docs = [
        Document(content="Doc one.", metadata={}),
        Document(content="Doc two.", metadata={}),
        Document(content="Doc three.", metadata={}),
    ]
    retriever.build_tree(docs)

    # Should have called add_documents for summary docs
    assert mock_vectorstore.add_documents.called


def test_raptor_retrieve():
    mock_llm = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1, 0.2]

    leaf_doc = Document(content="Leaf doc.", metadata={})
    summary_doc = Document(content="Summary.", metadata={"raptor_level": 1})
    mock_vectorstore = MagicMock()
    mock_vectorstore.search.return_value = [(leaf_doc, 0.9), (summary_doc, 0.8)]

    config = Config().retrieval
    retriever = RAPTORRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vectorstore,
        llm_provider=mock_llm,
        config=config,
    )

    results = retriever.retrieve("test query", top_k=2)
    assert len(results) == 2
    # Leaf docs should come first
    assert results[0].content == "Leaf doc."


# --- Corrective RAG Tests ---


def test_corrective_rag_passes_through_relevant_docs():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        '[{"index": 0, "relevant": true}, {"index": 1, "relevant": true}]'
    )

    doc1 = Document(content="Relevant doc 1.", metadata={})
    doc2 = Document(content="Relevant doc 2.", metadata={})

    mock_base_retriever = MagicMock(spec=BaseRetriever)
    mock_base_retriever.retrieve.return_value = [doc1, doc2]

    config = Config().retrieval.corrective_rag
    retriever = CorrectiveRAGRetriever(
        base_retriever=mock_base_retriever,
        llm_provider=mock_llm,
        config=config,
    )

    results = retriever.retrieve("test query", top_k=5)
    assert len(results) == 2


def test_corrective_rag_refines_query():
    mock_llm = MagicMock()
    # First call: evaluate relevance (none relevant)
    # Second call: refine query
    # Third call: evaluate relevance (all relevant)
    mock_llm.generate.side_effect = [
        '[{"index": 0, "relevant": false}]',  # First evaluation
        "refined query about topic",  # Query refinement
        '[{"index": 0, "relevant": true}]',  # Second evaluation
    ]

    doc = Document(content="Some doc.", metadata={})
    mock_base_retriever = MagicMock(spec=BaseRetriever)
    mock_base_retriever.retrieve.return_value = [doc]

    config = Config().retrieval.corrective_rag
    retriever = CorrectiveRAGRetriever(
        base_retriever=mock_base_retriever,
        llm_provider=mock_llm,
        config=config,
    )

    results = retriever.retrieve("vague query", top_k=5)
    assert len(results) >= 1
    assert mock_base_retriever.retrieve.call_count >= 2


# --- Standard Generation Tests ---


def test_standard_generation():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "The answer is 42."

    gen = StandardGeneration(llm_provider=mock_llm)
    docs = [Document(content="The meaning of life is 42.", metadata={})]
    result = gen.generate("What is the answer?", docs)

    assert result["answer"] == "The answer is 42."
    mock_llm.generate.assert_called_once()


# --- CoVe Generation Tests ---


def test_cove_generation():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        "Initial answer about Python.",  # Initial answer
        '["Is Python a programming language?", "Was Python created in 1991?"]',  # Verification questions
        "Yes, Python is a programming language.",  # Verification answer 1
        "Yes, Python was released in 1991.",  # Verification answer 2
        "Refined answer: Python is a programming language created in 1991.",  # Refined answer
    ]

    gen = ChainOfVerificationGeneration(
        llm_provider=mock_llm, max_verification_questions=2
    )
    docs = [Document(content="Python was created in 1991.", metadata={})]
    result = gen.generate("Tell me about Python", docs)

    assert "answer" in result
    assert "initial_answer" in result
    assert "verification_qa" in result
    assert len(result["verification_qa"]) == 2


def test_cove_generation_no_verification_questions():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        "Simple answer.",  # Initial answer
        "invalid response",  # Failed to generate verification questions
    ]

    gen = ChainOfVerificationGeneration(llm_provider=mock_llm)
    docs = [Document(content="Context.", metadata={})]
    result = gen.generate("Question?", docs)

    assert result["answer"] == "Simple answer."
    assert result["verification_qa"] == []


# --- Attributed Generation Tests ---


def test_attributed_generation():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Python is a language [1] created by Guido [2]."

    gen = AttributedGeneration(llm_provider=mock_llm)
    docs = [
        Document(
            content="Python is a programming language.", metadata={"source": "wiki.txt"}
        ),
        Document(
            content="Guido van Rossum created Python.", metadata={"source": "bio.txt"}
        ),
    ]
    result = gen.generate("Tell me about Python", docs)

    assert "answer" in result
    assert "citations" in result
    assert len(result["citations"]) == 2
    assert result["citations"][0]["citation_number"] == 1
    assert result["citations"][0]["source"] == "wiki.txt"
    assert result["citations"][1]["citation_number"] == 2


def test_attributed_generation_no_citations():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "I don't know the answer."

    gen = AttributedGeneration(llm_provider=mock_llm)
    docs = [Document(content="Unrelated content.", metadata={})]
    result = gen.generate("Unknown question?", docs)

    assert result["answer"] == "I don't know the answer."
    assert result["citations"] == []


# --- Strategy Selection in Core Tests ---


def test_strategy_selection_defaults():
    """Test that default config selects correct strategies."""
    config = Config()
    assert config.document_processing.chunking.strategy == "recursive"
    assert config.retrieval.strategy == "dense"
    assert config.generation.strategy == "standard"


def test_invalid_chunking_strategy():
    """Test that invalid chunking strategy raises ValueError."""
    from rag_sdk.core import RAG

    config = Config(document_processing={"chunking": {"strategy": "invalid"}})
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        # Need to mock LLM/embedding init to get to splitter init
        with (
            patch.object(RAG, "_init_embeddings"),
            patch.object(RAG, "_init_vectorstore"),
            patch.object(RAG, "_init_llm"),
        ):
            rag = RAG.__new__(RAG)
            rag.config = config
            rag.llm_provider = MagicMock()
            rag._init_splitter()


def test_invalid_retrieval_strategy():
    """Test that invalid retrieval strategy raises ValueError."""
    from rag_sdk.core import RAG

    config = Config(retrieval={"strategy": "invalid"})
    with pytest.raises(ValueError, match="Unsupported retrieval strategy"):
        rag = RAG.__new__(RAG)
        rag.config = config
        rag.embedding_provider = MagicMock()
        rag.vector_store = MagicMock()
        rag.llm_provider = MagicMock()
        rag._init_retriever()


def test_invalid_generation_strategy():
    """Test that invalid generation strategy raises ValueError."""
    from rag_sdk.core import RAG

    config = Config(generation={"strategy": "invalid"})
    with pytest.raises(ValueError, match="Unsupported generation strategy"):
        rag = RAG.__new__(RAG)
        rag.config = config
        rag.llm_provider = MagicMock()
        rag._init_generation()
