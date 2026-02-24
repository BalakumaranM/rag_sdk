"""Unit tests for rag_sdk.graph.models — Entity, Relationship, Community dataclasses."""

from rag_sdk.graph.models import Community, Entity, Relationship


class TestEntity:
    def test_required_fields(self):
        e = Entity(name="alice")
        assert e.name == "alice"

    def test_defaults(self):
        e = Entity(name="alice")
        assert e.entity_type == ""
        assert e.description == ""
        assert e.document_ids == []

    def test_all_fields(self):
        e = Entity(
            name="alice",
            entity_type="person",
            description="A scientist.",
            document_ids=["doc1", "doc2"],
        )
        assert e.name == "alice"
        assert e.entity_type == "person"
        assert e.description == "A scientist."
        assert e.document_ids == ["doc1", "doc2"]

    def test_document_ids_are_independent_per_instance(self):
        e1 = Entity(name="a")
        e2 = Entity(name="b")
        e1.document_ids.append("doc1")
        assert "doc1" not in e2.document_ids

    def test_equality(self):
        e1 = Entity(name="alice", entity_type="person")
        e2 = Entity(name="alice", entity_type="person")
        assert e1 == e2

    def test_inequality_different_name(self):
        assert Entity(name="alice") != Entity(name="bob")

    def test_name_is_mutable(self):
        e = Entity(name="Alice")
        e.name = e.name.lower()
        assert e.name == "alice"

    def test_description_is_mutable(self):
        e = Entity(name="alice")
        e.description = "Updated."
        assert e.description == "Updated."

    def test_document_ids_append(self):
        e = Entity(name="alice", document_ids=["doc1"])
        e.document_ids.append("doc2")
        assert e.document_ids == ["doc1", "doc2"]

    def test_entity_type_stored(self):
        e = Entity(name="acme", entity_type="organization")
        assert e.entity_type == "organization"


class TestRelationship:
    def test_required_fields(self):
        r = Relationship(source="alice", target="lab", relation="works_at")
        assert r.source == "alice"
        assert r.target == "lab"
        assert r.relation == "works_at"

    def test_defaults(self):
        r = Relationship(source="a", target="b", relation="related")
        assert r.description == ""
        assert r.weight == 1.0
        assert r.document_ids == []

    def test_all_fields(self):
        r = Relationship(
            source="alice",
            target="lab",
            relation="works_at",
            description="Alice works at the lab.",
            weight=8.0,
            document_ids=["doc1"],
        )
        assert r.description == "Alice works at the lab."
        assert r.weight == 8.0
        assert r.document_ids == ["doc1"]

    def test_weight_accepts_int(self):
        r = Relationship(source="a", target="b", relation="r", weight=7)
        assert r.weight == 7

    def test_document_ids_are_independent_per_instance(self):
        r1 = Relationship(source="a", target="b", relation="r")
        r2 = Relationship(source="c", target="d", relation="r")
        r1.document_ids.append("doc1")
        assert "doc1" not in r2.document_ids

    def test_equality(self):
        r1 = Relationship(source="a", target="b", relation="r", weight=5.0)
        r2 = Relationship(source="a", target="b", relation="r", weight=5.0)
        assert r1 == r2

    def test_inequality_different_source(self):
        r1 = Relationship(source="a", target="b", relation="r")
        r2 = Relationship(source="x", target="b", relation="r")
        assert r1 != r2

    def test_weight_mutation_max(self):
        r = Relationship(source="a", target="b", relation="r", weight=3.0)
        r.weight = max(r.weight, 9.0)
        assert r.weight == 9.0

    def test_description_mutation(self):
        r = Relationship(source="a", target="b", relation="r")
        r.description = "New description"
        assert r.description == "New description"


class TestCommunity:
    def test_required_fields(self):
        c = Community(id="c1", level=0, entities=["alice", "bob"])
        assert c.id == "c1"
        assert c.level == 0
        assert c.entities == ["alice", "bob"]

    def test_defaults(self):
        c = Community(id="c1", level=0, entities=[])
        assert c.title == ""
        assert c.summary == ""
        assert c.full_content == ""
        assert c.findings == []
        assert c.rank == 0.0
        assert c.embedding is None

    def test_all_fields(self):
        emb = [0.1, 0.2, 0.3]
        c = Community(
            id="c1",
            level=1,
            entities=["alice", "bob"],
            title="Science Leaders",
            summary="Key researchers in physics.",
            full_content="# Science Leaders\n\nKey researchers.",
            findings=[{"explanation": "Alice leads the team.", "data_refs": "doc1"}],
            rank=8.5,
            embedding=emb,
        )
        assert c.title == "Science Leaders"
        assert c.summary == "Key researchers in physics."
        assert "Science Leaders" in c.full_content
        assert len(c.findings) == 1
        assert c.findings[0]["explanation"] == "Alice leads the team."
        assert c.rank == 8.5
        assert c.embedding == emb

    def test_level_zero_is_coarsest(self):
        coarse = Community(id="c1", level=0, entities=[])
        fine = Community(id="c2", level=2, entities=[])
        assert coarse.level < fine.level

    def test_findings_are_independent_per_instance(self):
        c1 = Community(id="c1", level=0, entities=[])
        c2 = Community(id="c2", level=0, entities=[])
        c1.findings.append({"explanation": "test"})
        assert len(c2.findings) == 0

    def test_embedding_starts_none(self):
        c = Community(id="c1", level=0, entities=[])
        assert c.embedding is None

    def test_embedding_assignment(self):
        c = Community(id="c1", level=0, entities=[])
        c.embedding = [0.5, 0.5]
        assert c.embedding == [0.5, 0.5]

    def test_rank_default_zero(self):
        c = Community(id="c1", level=0, entities=[])
        assert c.rank == 0.0

    def test_rank_set(self):
        c = Community(id="c1", level=0, entities=[], rank=10.0)
        assert c.rank == 10.0

    def test_entities_mutation(self):
        c = Community(id="c1", level=0, entities=["alice"])
        c.entities.append("bob")
        assert "bob" in c.entities

    def test_equality(self):
        c1 = Community(id="c1", level=0, entities=["alice"], rank=5.0)
        c2 = Community(id="c1", level=0, entities=["alice"], rank=5.0)
        assert c1 == c2

    def test_inequality_different_id(self):
        c1 = Community(id="c1", level=0, entities=[])
        c2 = Community(id="c2", level=0, entities=[])
        assert c1 != c2
