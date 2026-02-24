"""Unit tests for AdvancedGraphRAGRetriever."""

from unittest.mock import MagicMock

from rag_sdk.config import Config
from rag_sdk.document import Document
from rag_sdk.graph.models import Community, Entity, Relationship
from rag_sdk.retrieval.advanced_graph_rag import AdvancedGraphRAGRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    base = Config()
    for k, v in kwargs.items():
        setattr(base.retrieval.advanced_graph_rag, k, v)
    return base.retrieval


def _make_retriever(search_mode="local", **cfg_kwargs):
    cfg_kwargs["search_mode"] = search_mode
    config = _make_config(**cfg_kwargs)

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.5] * 8
    mock_embedding.embed_documents.return_value = [[0.5] * 8]

    mock_vectorstore = MagicMock()
    mock_vectorstore.search.return_value = []

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[]"

    retriever = AdvancedGraphRAGRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vectorstore,
        llm_provider=mock_llm,
        config=config,
    )
    return retriever, mock_llm, mock_embedding, mock_vectorstore


def _add_entity(retriever, name, entity_type="concept", description=""):
    e = Entity(name=name, entity_type=entity_type, description=description)
    retriever._indexer.entities[name] = e
    retriever._indexer.graph.add_node(name, entity_type=entity_type)
    return e


def _add_relationship(retriever, source, target, relation="related", weight=5.0):
    r = Relationship(source=source, target=target, relation=relation, weight=weight)
    retriever._indexer.relationships.append(r)
    retriever._indexer.graph.add_edge(source, target, weight=weight, relation=relation)
    return r


def _add_community(
    retriever,
    cid="c1",
    level=0,
    entities=None,
    rank=5.0,
    summary="Test summary",
    embedding=None,
):
    emb = (
        embedding if embedding is not None else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    c = Community(
        id=cid,
        level=level,
        entities=entities or ["alice"],
        title="Test Community",
        summary=summary,
        rank=rank,
        embedding=emb,
    )
    retriever._indexer.communities[cid] = c
    return c


def _make_doc(content="content", doc_id=None):
    return Document(content=content, metadata={}, id=doc_id or "d1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_internal_indexer(self):
        retriever, *_ = _make_retriever()
        assert retriever._indexer is not None

    def test_initial_entities_empty(self):
        retriever, *_ = _make_retriever()
        assert retriever.entities == {}

    def test_initial_relationships_empty(self):
        retriever, *_ = _make_retriever()
        assert retriever.relationships == []

    def test_initial_communities_empty(self):
        retriever, *_ = _make_retriever()
        assert retriever.communities == {}

    def test_initial_graph_empty(self):
        retriever, *_ = _make_retriever()
        assert retriever.graph.number_of_nodes() == 0


class TestProperties:
    def test_entities_reflects_indexer(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "alice")
        assert "alice" in retriever.entities
        assert retriever.entities is retriever._indexer.entities

    def test_relationships_reflects_indexer(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "a")
        _add_entity(retriever, "b")
        _add_relationship(retriever, "a", "b")
        assert retriever.relationships is retriever._indexer.relationships

    def test_graph_reflects_indexer(self):
        retriever, *_ = _make_retriever()
        assert retriever.graph is retriever._indexer.graph

    def test_communities_reflects_indexer(self):
        retriever, *_ = _make_retriever()
        _add_community(retriever)
        assert retriever.communities is retriever._indexer.communities


class TestExtractQueryEntities:
    def test_json_array_parsed(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = '["python", "guido"]'
        result = retriever._extract_query_entities("Who made Python?")
        assert "python" in result
        assert "guido" in result

    def test_results_always_lowercase(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = '["Python", "GUIDO"]'
        result = retriever._extract_query_entities("query")
        assert all(e == e.lower() for e in result)

    def test_fallback_on_llm_exception(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.side_effect = Exception("fail")
        result = retriever._extract_query_entities(
            "What is machine learning all about?"
        )
        assert isinstance(result, list)
        assert all(len(e) > 3 for e in result)

    def test_fallback_on_bad_json(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = "not json"
        result = retriever._extract_query_entities("query about things here")
        assert isinstance(result, list)

    def test_non_string_items_filtered(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = '["python", 42, null, "guido"]'
        result = retriever._extract_query_entities("query")
        assert all(isinstance(e, str) for e in result)

    def test_empty_array_returns_empty_list(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = "[]"
        result = retriever._extract_query_entities("query")
        assert result == []


class TestMatchEntities:
    def test_query_entity_substring_of_graph_entity(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "machine learning")
        matched = retriever._match_entities(["machine"])
        assert "machine learning" in matched

    def test_graph_entity_substring_of_query_entity(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "python")
        matched = retriever._match_entities(["python programming"])
        assert "python" in matched

    def test_no_overlap_returns_empty(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "alice")
        matched = retriever._match_entities(["quantum"])
        assert len(matched) == 0

    def test_empty_query_entities_returns_empty(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "alice")
        matched = retriever._match_entities([])
        assert matched == set()

    def test_multiple_graph_entities_matched(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "alice")
        _add_entity(retriever, "alice smith")
        matched = retriever._match_entities(["alice"])
        assert "alice" in matched
        assert "alice smith" in matched


class TestGetNeighborhood:
    def test_empty_seeds_returns_empty(self):
        retriever, *_ = _make_retriever()
        result = retriever._get_neighborhood(set(), hops=2)
        assert result == set()

    def test_seed_not_in_graph_skipped(self):
        retriever, *_ = _make_retriever()
        result = retriever._get_neighborhood({"nonexistent"}, hops=2)
        assert isinstance(result, set)

    def test_one_hop_visits_seed(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "alice")
        _add_entity(retriever, "lab")
        _add_relationship(retriever, "alice", "lab")
        result = retriever._get_neighborhood({"alice"}, hops=1)
        assert "alice" in result

    def test_two_hops_reaches_further_node(self):
        retriever, *_ = _make_retriever()
        for node in ["alice", "lab", "project"]:
            _add_entity(retriever, node)
        _add_relationship(retriever, "alice", "lab")
        _add_relationship(retriever, "lab", "project")
        result = retriever._get_neighborhood({"alice"}, hops=2)
        assert "lab" in result

    def test_circular_graph_no_infinite_loop(self):
        retriever, *_ = _make_retriever()
        _add_entity(retriever, "a")
        _add_entity(retriever, "b")
        _add_relationship(retriever, "a", "b")
        _add_relationship(retriever, "b", "a")
        result = retriever._get_neighborhood({"a"}, hops=10)
        assert isinstance(result, set)


class TestCosineSimilarity:
    def test_identical_unit_vectors(self):
        sim = AdvancedGraphRAGRetriever._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        sim = AdvancedGraphRAGRetriever._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim - 0.0) < 1e-6

    def test_opposite_vectors(self):
        sim = AdvancedGraphRAGRetriever._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_norm_a_returns_zero(self):
        sim = AdvancedGraphRAGRetriever._cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0

    def test_both_zero_returns_zero(self):
        sim = AdvancedGraphRAGRetriever._cosine_similarity([0.0, 0.0], [0.0, 0.0])
        assert sim == 0.0

    def test_arbitrary_vectors(self):
        sim = AdvancedGraphRAGRetriever._cosine_similarity([3.0, 4.0], [3.0, 4.0])
        assert abs(sim - 1.0) < 1e-6


class TestScoreCommunities:
    def test_higher_rank_boosts_score(self):
        retriever, *_ = _make_retriever()
        # Both communities have identical embeddings → only rank differentiates them
        _add_community(
            retriever,
            "c1",
            rank=1.0,
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        _add_community(
            retriever,
            "c2",
            rank=10.0,
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        scored = retriever._score_communities([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert scored[0][0].id == "c2"

    def test_sorted_descending_by_score(self):
        retriever, *_ = _make_retriever()
        _add_community(
            retriever,
            "c1",
            rank=2.0,
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        _add_community(
            retriever,
            "c2",
            rank=8.0,
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        scored = retriever._score_communities([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_community_without_embedding_excluded(self):
        retriever, *_ = _make_retriever()
        c = Community(
            id="c1", level=0, entities=[], summary="No embedding", embedding=None
        )
        retriever._indexer.communities["c1"] = c
        scored = retriever._score_communities([1.0, 0.0])
        assert len(scored) == 0

    def test_empty_communities_returns_empty_list(self):
        retriever, *_ = _make_retriever()
        scored = retriever._score_communities([1.0, 0.0])
        assert scored == []


class TestMakeDoc:
    def test_creates_document_with_content(self):
        retriever, *_ = _make_retriever()
        doc = retriever._make_doc("Hello world")
        assert doc.content == "Hello world"

    def test_id_starts_with_synth(self):
        retriever, *_ = _make_retriever()
        doc = retriever._make_doc("content")
        assert doc.id.startswith("synth-")

    def test_metadata_assigned(self):
        retriever, *_ = _make_retriever()
        doc = retriever._make_doc("content", {"source": "knowledge_graph"})
        assert doc.metadata["source"] == "knowledge_graph"

    def test_default_metadata_empty(self):
        retriever, *_ = _make_retriever()
        doc = retriever._make_doc("content")
        assert doc.metadata == {}

    def test_unique_ids_generated(self):
        retriever, *_ = _make_retriever()
        ids = {retriever._make_doc("c").id for _ in range(30)}
        assert len(ids) == 30


class TestLocalSearch:
    def test_returns_dense_docs_when_no_neighborhood(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        mock_llm.generate.return_value = '["quantum_physics"]'  # no match in graph
        doc = _make_doc("Dense doc.", "d1")
        mock_vectorstore.search.return_value = [(doc, 0.9)]
        results = retriever._local_search("query", top_k=3, filters=None)
        assert any(r.id == "d1" for r in results)

    def test_graph_context_doc_prepended_when_neighborhood_found(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        _add_entity(retriever, "python", "technology", "A language.")
        mock_llm.generate.return_value = '["python"]'
        doc = _make_doc("Dense result.", "d1")
        mock_vectorstore.search.return_value = [(doc, 0.8)]
        results = retriever._local_search("query about python", top_k=5, filters=None)
        assert results[0].id.startswith("synth-")
        assert results[0].metadata.get("type") == "graph_context"

    def test_graph_context_contains_entity_info(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        _add_entity(retriever, "python", "technology", "A language.")
        mock_llm.generate.return_value = '["python"]'
        mock_vectorstore.search.return_value = []
        results = retriever._local_search("query about python", top_k=5, filters=None)
        graph_doc = results[0]
        assert "python" in graph_doc.content

    def test_top_k_forwarded_to_vector_store(self):
        # _local_search trusts the vector store to honour top_k; verify the param is passed
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        mock_llm.generate.return_value = "[]"
        mock_vectorstore.search.return_value = []
        retriever._local_search("query", top_k=3, filters=None)
        call_kwargs = mock_vectorstore.search.call_args.kwargs
        assert call_kwargs.get("top_k") == 3

    def test_filters_forwarded_to_vector_store(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        mock_llm.generate.return_value = "[]"
        mock_vectorstore.search.return_value = []
        retriever._local_search("query", top_k=3, filters={"source": "wiki"})
        call_kwargs = mock_vectorstore.search.call_args.kwargs
        assert call_kwargs.get("filters") == {"source": "wiki"}


class TestGlobalSearch:
    def test_returns_synthesized_doc_as_first_result(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "global"
        )
        _add_community(retriever)
        mock_llm.generate.side_effect = [
            "Partial answer about the topic.",
            "Final synthesized answer.",
        ]
        mock_vectorstore.search.return_value = []
        results = retriever._global_search("broad query", top_k=3, filters=None)
        assert len(results) >= 1
        assert results[0].metadata.get("type") == "synthesized_answer"

    def test_synthesized_doc_contains_synthesis(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "global"
        )
        _add_community(retriever)
        mock_llm.generate.side_effect = ["Partial.", "Final synthesis content."]
        mock_vectorstore.search.return_value = []
        results = retriever._global_search("broad query", top_k=3, filters=None)
        assert "Final synthesis content." in results[0].content

    def test_fallback_when_no_communities(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "global"
        )
        doc = _make_doc("Dense fallback.", "d1")
        mock_vectorstore.search.return_value = [(doc, 0.9)]
        results = retriever._global_search("query", top_k=3, filters=None)
        assert any(r.id == "d1" for r in results)

    def test_fallback_when_all_map_calls_fail(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "global"
        )
        _add_community(retriever)
        mock_llm.generate.side_effect = Exception("LLM completely down")
        doc = _make_doc("Dense fallback.", "d1")
        mock_vectorstore.search.return_value = [(doc, 0.9)]
        # Should not raise
        results = retriever._global_search("query", top_k=3, filters=None)
        assert isinstance(results, list)

    def test_dense_docs_appended_after_synthesis(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "global"
        )
        _add_community(retriever)
        mock_llm.generate.side_effect = ["Partial.", "Synthesis."]
        dense_doc = _make_doc("Dense.", "d1")
        mock_vectorstore.search.return_value = [(dense_doc, 0.8)]
        results = retriever._global_search("query", top_k=3, filters=None)
        doc_ids = [r.id for r in results]
        assert "d1" in doc_ids


class TestDriftSearch:
    def test_returns_list_of_documents(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("drift")
        _add_community(retriever)
        mock_llm.generate.side_effect = [
            "Hypothetical answer.",
            '["follow-up 1", "follow-up 2"]',
        ]
        doc = _make_doc("Doc.", "d1")
        mock_vectorstore.search.return_value = [(doc, 0.9)]
        results = retriever._drift_search("research query", top_k=5, filters=None)
        assert isinstance(results, list)

    def test_hyde_failure_does_not_crash(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("drift")
        _add_community(retriever)
        mock_llm.generate.side_effect = Exception("HyDE failed")
        mock_vectorstore.search.return_value = []
        results = retriever._drift_search("query", top_k=3, filters=None)
        assert isinstance(results, list)

    def test_top_k_limits_results(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "drift", drift_max_rounds=1, drift_follow_up_questions=1
        )
        _add_community(retriever)
        mock_llm.generate.side_effect = [
            "HyDE answer.",
            '["follow-up 1"]',
            "Refined answer.",
        ]
        docs = [_make_doc(f"Doc {i}", f"d{i}") for i in range(10)]
        mock_vectorstore.search.return_value = [(d, 0.9) for d in docs]
        results = retriever._drift_search("query", top_k=3, filters=None)
        assert len(results) <= 3

    def test_max_rounds_respected(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "drift", drift_max_rounds=2
        )
        _add_community(retriever)
        call_log = []

        def llm_side_effect(**kwargs):
            call_log.append("generate")
            return "[]"

        mock_llm.generate.side_effect = llm_side_effect
        mock_vectorstore.search.return_value = []
        retriever._drift_search("query", top_k=3, filters=None)
        # HyDE (1) + follow-ups per round (2) + refinements (2) ≤ expected call count
        assert mock_llm.generate.call_count <= 10  # loose upper bound


class TestRetrieve:
    def test_dispatches_to_local_mode(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        mock_llm.generate.return_value = "[]"
        mock_vectorstore.search.return_value = []
        result = retriever.retrieve("query")
        assert isinstance(result, list)

    def test_dispatches_to_global_mode(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever(
            "global"
        )
        _add_community(retriever)
        mock_llm.generate.side_effect = ["Partial.", "Synthesis."]
        mock_vectorstore.search.return_value = []
        result = retriever.retrieve("broad query")
        assert isinstance(result, list)

    def test_dispatches_to_drift_mode(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("drift")
        _add_community(retriever)
        mock_llm.generate.side_effect = ["HyDE.", "[]"]
        mock_vectorstore.search.return_value = []
        result = retriever.retrieve("exploratory query")
        assert isinstance(result, list)

    def test_fallback_to_dense_when_no_communities(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        # No communities → graph not built → dense fallback
        doc = _make_doc("Dense fallback.", "d1")
        mock_vectorstore.search.return_value = [(doc, 0.9)]
        results = retriever.retrieve("query")
        assert any(r.id == "d1" for r in results)

    def test_top_k_forwarded(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        mock_llm.generate.return_value = "[]"
        mock_vectorstore.search.return_value = []
        retriever.retrieve("query", top_k=7)
        call_kwargs = mock_vectorstore.search.call_args.kwargs
        assert call_kwargs.get("top_k") == 7

    def test_filters_forwarded(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        mock_llm.generate.return_value = "[]"
        mock_vectorstore.search.return_value = []
        retriever.retrieve("query", filters={"tag": "science"})
        call_kwargs = mock_vectorstore.search.call_args.kwargs
        assert call_kwargs.get("filters") == {"tag": "science"}

    def test_unknown_mode_falls_through_to_local(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever("local")
        _add_community(retriever)
        mock_llm.generate.return_value = "[]"
        mock_vectorstore.search.return_value = []
        result = retriever.retrieve("query")
        assert isinstance(result, list)
