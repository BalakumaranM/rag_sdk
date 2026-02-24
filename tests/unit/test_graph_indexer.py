"""Unit tests for rag_sdk.graph.indexer.GraphIndexer."""

import json
from unittest.mock import MagicMock


from rag_sdk.config import Config
from rag_sdk.document import Document
from rag_sdk.graph.indexer import GraphIndexer
from rag_sdk.graph.models import Community, Entity, Relationship


def _make_config(**kwargs):
    base = Config()
    for k, v in kwargs.items():
        setattr(base.retrieval.advanced_graph_rag, k, v)
    return base.retrieval


def _make_indexer(**cfg_kwargs):
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
    mock_llm = MagicMock()
    config = _make_config(**cfg_kwargs)
    indexer = GraphIndexer(
        embedding_provider=mock_embedding,
        llm_provider=mock_llm,
        config=config,
    )
    return indexer, mock_llm, mock_embedding


def _entity_json(*names):
    return [{"name": n, "type": "concept", "description": f"About {n}."} for n in names]


def _rel_json(source, target, relation="related", weight=5):
    return [
        {
            "source": source,
            "target": target,
            "relation": relation,
            "description": f"{source} {relation} {target}.",
            "weight": weight,
        }
    ]


def _extraction(entities=None, rels=None):
    return json.dumps({"entities": entities or [], "relationships": rels or []})


def _community_report(title="Test Community", rank=5.0):
    return json.dumps(
        {
            "title": title,
            "summary": f"A community about {title}.",
            "findings": [{"explanation": "Key finding.", "data_refs": "doc1"}],
            "rank": rank,
        }
    )


class TestGraphIndexerInit:
    def test_empty_entities(self):
        indexer, *_ = _make_indexer()
        assert indexer.entities == {}

    def test_empty_relationships(self):
        indexer, *_ = _make_indexer()
        assert indexer.relationships == []

    def test_empty_communities(self):
        indexer, *_ = _make_indexer()
        assert indexer.communities == {}

    def test_empty_graph(self):
        indexer, *_ = _make_indexer()
        assert indexer.graph.number_of_nodes() == 0

    def test_providers_stored(self):
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1]]
        mock_llm = MagicMock()
        config = Config().retrieval
        indexer = GraphIndexer(mock_emb, mock_llm, config)
        assert indexer.embedding_provider is mock_emb
        assert indexer.llm_provider is mock_llm


class TestExtractEntitiesAndRelationships:
    def test_entities_extracted_with_all_fields(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _extraction(
            entities=[
                {"name": "Alice", "type": "person", "description": "A scientist."}
            ]
        )
        entities, _ = indexer._extract_entities_and_relationships("text", "doc1")
        assert len(entities) == 1
        assert entities[0].name == "alice"
        assert entities[0].entity_type == "person"
        assert entities[0].description == "A scientist."
        assert entities[0].document_ids == ["doc1"]

    def test_relationships_extracted_with_all_fields(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _extraction(
            rels=[
                {
                    "source": "Alice",
                    "target": "Lab",
                    "relation": "works_at",
                    "description": "Alice works there.",
                    "weight": 7,
                }
            ]
        )
        _, rels = indexer._extract_entities_and_relationships("text", "doc1")
        assert len(rels) == 1
        assert rels[0].source == "alice"
        assert rels[0].target == "lab"
        assert rels[0].relation == "works_at"
        assert rels[0].description == "Alice works there."
        assert rels[0].weight == 7.0

    def test_name_lowercased(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _extraction(
            entities=[{"name": "UPPER_CASE", "type": "concept", "description": ""}]
        )
        entities, _ = indexer._extract_entities_and_relationships("text", "doc1")
        assert entities[0].name == "upper_case"

    def test_max_entities_capped(self):
        indexer, mock_llm, _ = _make_indexer(max_entities_per_chunk=3)
        mock_llm.generate.return_value = _extraction(
            entities=_entity_json(*[f"e{i}" for i in range(10)])
        )
        entities, _ = indexer._extract_entities_and_relationships("text", "doc1")
        assert len(entities) <= 3

    def test_max_relationships_capped(self):
        indexer, mock_llm, _ = _make_indexer(max_relationships_per_chunk=2)
        rels = [
            {
                "source": "a",
                "target": f"b{i}",
                "relation": "r",
                "description": "",
                "weight": 1,
            }
            for i in range(10)
        ]
        mock_llm.generate.return_value = _extraction(rels=rels)
        _, result_rels = indexer._extract_entities_and_relationships("text", "doc1")
        assert len(result_rels) <= 2

    def test_llm_exception_returns_empty(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.side_effect = Exception("LLM down")
        entities, rels = indexer._extract_entities_and_relationships("text", "doc1")
        assert entities == []
        assert rels == []

    def test_invalid_json_returns_empty(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = "no json here"
        entities, rels = indexer._extract_entities_and_relationships("text", "doc1")
        assert entities == []
        assert rels == []

    def test_missing_optional_fields_use_defaults(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _extraction(
            entities=[{"name": "alice"}],
            rels=[{"source": "alice", "target": "lab", "relation": "knows"}],
        )
        entities, rels = indexer._extract_entities_and_relationships("text", "doc1")
        assert entities[0].entity_type == ""
        assert entities[0].description == ""
        assert rels[0].description == ""
        assert rels[0].weight == 1.0

    def test_weight_defaults_to_one_when_absent(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _extraction(
            rels=[{"source": "a", "target": "b", "relation": "r"}]
        )
        _, rels = indexer._extract_entities_and_relationships("text", "doc1")
        assert rels[0].weight == 1.0

    def test_entity_types_in_prompt(self):
        indexer, mock_llm, _ = _make_indexer(entity_types=["drug", "disease"])
        mock_llm.generate.return_value = _extraction()
        indexer._extract_entities_and_relationships("text", "doc1")
        prompt = mock_llm.generate.call_args[1]["prompt"]
        assert "drug" in prompt
        assert "disease" in prompt


class TestMergeDescriptions:
    def test_calls_llm_with_descriptions(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = "Merged description."
        result = indexer._merge_descriptions("entity", "alice", ["Desc 1.", "Desc 2."])
        mock_llm.generate.assert_called_once()
        assert result == "Merged description."

    def test_strips_whitespace_from_result(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = "  Trimmed.  "
        result = indexer._merge_descriptions("entity", "alice", ["Desc."])
        assert result == "Trimmed."

    def test_llm_failure_returns_joined_fallback(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.side_effect = Exception("fail")
        result = indexer._merge_descriptions("entity", "alice", ["A.", "B."])
        assert "A." in result
        assert "B." in result

    def test_single_description_still_calls_llm(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = "Single merged."
        result = indexer._merge_descriptions("entity", "alice", ["Single desc."])
        assert result == "Single merged."

    def test_relationship_kind_in_prompt(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = "merged"
        indexer._merge_descriptions("relationship", "a → b", ["Desc."])
        prompt = mock_llm.generate.call_args[1]["prompt"]
        assert "relationship" in prompt
        assert "a → b" in prompt


class TestDetectCommunities:
    def test_empty_graph_returns_empty(self):
        indexer, *_ = _make_indexer()
        assert indexer._detect_communities() == []

    def test_single_node_produces_community(self):
        indexer, *_ = _make_indexer()
        indexer.graph.add_node("alice")
        result = indexer._detect_communities()
        assert len(result) >= 1
        for level, members in result:
            assert isinstance(level, int)
            assert isinstance(members, set)

    def test_connected_nodes_grouped(self):
        indexer, *_ = _make_indexer()
        indexer.graph.add_node("a")
        indexer.graph.add_node("b")
        indexer.graph.add_edge("a", "b", weight=5.0)
        result = indexer._detect_communities()
        assert len(result) >= 1
        all_nodes = set()
        for _, members in result:
            all_nodes.update(members)
        assert "a" in all_nodes and "b" in all_nodes

    def test_two_levels_config(self):
        indexer, *_ = _make_indexer(community_levels=2)
        for i in range(6):
            indexer.graph.add_node(f"n{i}")
        for i in range(5):
            indexer.graph.add_edge(f"n{i}", f"n{i + 1}", weight=3.0)
        result = indexer._detect_communities()
        assert len(result) >= 1

    def test_one_level_config(self):
        indexer, *_ = _make_indexer(community_levels=1)
        indexer.graph.add_node("a")
        indexer.graph.add_node("b")
        indexer.graph.add_edge("a", "b", weight=1.0)
        result = indexer._detect_communities()
        levels = {level for level, _ in result}
        assert 0 in levels

    def test_result_is_list_of_tuples(self):
        indexer, *_ = _make_indexer()
        indexer.graph.add_node("x")
        result = indexer._detect_communities()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestSummarizeCommunities:
    def test_creates_community_for_each_raw(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report()
        indexer.entities = {"alice": Entity(name="alice"), "bob": Entity(name="bob")}
        indexer._summarize_communities([(0, {"alice"}), (1, {"bob"})])
        assert len(indexer.communities) == 2

    def test_community_title_from_llm(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report(title="Physics Team")
        indexer.entities = {"alice": Entity(name="alice")}
        indexer._summarize_communities([(0, {"alice"})])
        comm = next(iter(indexer.communities.values()))
        assert comm.title == "Physics Team"

    def test_community_rank_from_llm(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report(rank=9.0)
        indexer.entities = {"alice": Entity(name="alice")}
        indexer._summarize_communities([(0, {"alice"})])
        comm = next(iter(indexer.communities.values()))
        assert comm.rank == 9.0

    def test_community_findings_from_llm(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report()
        indexer.entities = {"alice": Entity(name="alice")}
        indexer._summarize_communities([(0, {"alice"})])
        comm = next(iter(indexer.communities.values()))
        assert len(comm.findings) >= 1

    def test_community_level_stored_correctly(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report()
        indexer.entities = {"a": Entity(name="a"), "b": Entity(name="b")}
        indexer._summarize_communities([(2, {"a"}), (0, {"b"})])
        levels = {c.level for c in indexer.communities.values()}
        assert 2 in levels
        assert 0 in levels

    def test_full_content_contains_title(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report(title="Special Topic")
        indexer.entities = {"alice": Entity(name="alice")}
        indexer._summarize_communities([(0, {"alice"})])
        comm = next(iter(indexer.communities.values()))
        assert "Special Topic" in comm.full_content

    def test_llm_failure_uses_fallback_values(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.side_effect = Exception("LLM down")
        indexer.entities = {"alice": Entity(name="alice")}
        indexer._summarize_communities([(0, {"alice"})])
        assert len(indexer.communities) == 1
        comm = next(iter(indexer.communities.values()))
        assert comm.title  # fallback title non-empty
        assert comm.rank == 5.0  # default fallback rank

    def test_empty_raw_communities_noop(self):
        indexer, *_ = _make_indexer()
        indexer._summarize_communities([])
        assert indexer.communities == {}

    def test_internal_relationships_included_in_prompt(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report()
        indexer.entities = {
            "alice": Entity(name="alice"),
            "lab": Entity(name="lab"),
        }
        indexer.relationships = [
            Relationship(source="alice", target="lab", relation="works_at")
        ]
        indexer._summarize_communities([(0, {"alice", "lab"})])
        prompt = mock_llm.generate.call_args[1]["prompt"]
        assert "works_at" in prompt

    def test_external_relationships_excluded_from_prompt(self):
        indexer, mock_llm, _ = _make_indexer()
        mock_llm.generate.return_value = _community_report()
        indexer.entities = {"alice": Entity(name="alice")}
        indexer.relationships = [
            Relationship(source="alice", target="outsider", relation="knows")
        ]
        # Only alice in community — outsider is external
        indexer._summarize_communities([(0, {"alice"})])
        prompt = mock_llm.generate.call_args[1]["prompt"]
        # The relationship should not appear (target not in community)
        assert "knows" not in prompt


class TestEmbedCommunitySummaries:
    def test_embeddings_assigned_to_all_communities(self):
        indexer, _, mock_embedding = _make_indexer()
        mock_embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        indexer.communities = {
            "c1": Community(id="c1", level=0, entities=[], summary="Summary 1"),
            "c2": Community(id="c2", level=0, entities=[], summary="Summary 2"),
        }
        indexer._embed_community_summaries()
        for comm in indexer.communities.values():
            assert comm.embedding is not None

    def test_no_communities_does_not_call_embed(self):
        indexer, _, mock_embedding = _make_indexer()
        indexer._embed_community_summaries()
        mock_embedding.embed_documents.assert_not_called()

    def test_embedding_failure_does_not_raise(self):
        indexer, _, mock_embedding = _make_indexer()
        mock_embedding.embed_documents.side_effect = Exception("embed fail")
        indexer.communities = {
            "c1": Community(id="c1", level=0, entities=[], summary="Summary")
        }
        indexer._embed_community_summaries()  # must not raise
        assert indexer.communities["c1"].embedding is None

    def test_summaries_passed_to_embed(self):
        indexer, _, mock_embedding = _make_indexer()
        mock_embedding.embed_documents.return_value = [[0.5, 0.5]]
        indexer.communities = {
            "c1": Community(id="c1", level=0, entities=[], summary="My summary")
        }
        indexer._embed_community_summaries()
        texts = mock_embedding.embed_documents.call_args[0][0]
        assert "My summary" in texts


class TestBuildGraph:
    def _full_pipeline_side_effects(self, extraction_response, community_response=""):
        effects = [extraction_response]
        if community_response:
            effects.append(community_response)
        return effects

    def test_entities_populated_after_build(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        mock_llm.generate.side_effect = [
            _extraction(entities=_entity_json("alice")),
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        indexer.build_graph([Document(content="Alice is here.", metadata={})])
        assert "alice" in indexer.entities

    def test_relationships_populated_after_build(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        mock_llm.generate.side_effect = [
            _extraction(
                entities=_entity_json("alice", "lab"),
                rels=_rel_json("alice", "lab", "works_at", weight=7),
            ),
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        indexer.build_graph([Document(content="text", metadata={})])
        assert len(indexer.relationships) == 1
        assert indexer.relationships[0].relation == "works_at"

    def test_graph_nodes_created(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        mock_llm.generate.side_effect = [
            _extraction(entities=_entity_json("alice")),
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        indexer.build_graph([Document(content="text", metadata={})])
        assert "alice" in indexer.graph.nodes

    def test_graph_edges_created(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        mock_llm.generate.side_effect = [
            _extraction(rels=_rel_json("alice", "lab", weight=5)),
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        indexer.build_graph([Document(content="text", metadata={})])
        assert indexer.graph.has_edge("alice", "lab")

    def test_communities_created_after_build(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        mock_llm.generate.side_effect = [
            _extraction(entities=_entity_json("alice")),
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        indexer.build_graph([Document(content="text", metadata={})])
        assert len(indexer.communities) >= 1

    def test_duplicate_entity_merges_document_ids(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        entity_response = _extraction(entities=_entity_json("alice"))
        mock_llm.generate.side_effect = [
            entity_response,  # doc1 extraction
            entity_response,  # doc2 extraction (same entity)
            "Merged desc.",  # description merge call
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        doc1 = Document(content="Alice A.", metadata={}, id="d1")
        doc2 = Document(content="Alice B.", metadata={}, id="d2")
        indexer.build_graph([doc1, doc2])
        entity = indexer.entities["alice"]
        assert "d1" in entity.document_ids
        assert "d2" in entity.document_ids

    def test_edge_weight_uses_max_on_duplicate_relationship(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        mock_llm.generate.side_effect = [
            _extraction(rels=_rel_json("a", "b", weight=3)),
            _extraction(rels=_rel_json("a", "b", weight=9)),
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        doc1 = Document(content="text1", metadata={})
        doc2 = Document(content="text2", metadata={})
        indexer.build_graph([doc1, doc2])
        assert indexer.graph["a"]["b"]["weight"] == 9.0

    def test_empty_documents_list(self):
        indexer, mock_llm, mock_embedding = _make_indexer()
        mock_embedding.embed_documents.return_value = []
        indexer.build_graph([])
        assert indexer.entities == {}
        assert indexer.relationships == []

    def test_relationship_weight_disabled_uses_one(self):
        indexer, mock_llm, mock_embedding = _make_indexer(
            relationship_weight_in_graph=False
        )
        mock_llm.generate.side_effect = [
            _extraction(rels=_rel_json("a", "b", weight=9)),
            _community_report(),
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
        indexer.build_graph([Document(content="text", metadata={})])
        # Weight should be 1.0 when relationship_weight_in_graph is False
        assert indexer.graph["a"]["b"]["weight"] == 1.0
