"""GraphRAG ingestion pipeline.

Handles entity/relationship extraction from document chunks, cross-chunk
merging of duplicate descriptions, community detection, structured community
report generation, and community embedding.

The populated ``GraphIndexer`` is consumed by ``AdvancedGraphRAGRetriever``
at query time.
"""

import json
import logging
import re
import uuid
from typing import Dict, List, Set, Tuple

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community

    _NETWORKX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NETWORKX_AVAILABLE = False

from ..config import RetrievalConfig
from ..document import Document
from ..embeddings import EmbeddingProvider
from ..llm import LLMProvider
from .models import Community, Entity, Relationship

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Few-shot examples for the entity/relationship extraction prompt.
# Generic across domains; entity_types config provides domain focus.
# ---------------------------------------------------------------------------

_EXTRACTION_EXAMPLE_1 = """\
Text:
Marie Curie discovered polonium and radium in 1898 while working at the University of Paris.
She collaborated closely with her husband Pierre Curie throughout her research career.
------------------------
Output:
{
  "entities": [
    {"name": "marie curie", "type": "person",
     "description": "Marie Curie was a pioneering physicist and chemist who discovered polonium and radium and conducted research at the University of Paris."},
    {"name": "pierre curie", "type": "person",
     "description": "Pierre Curie was a physicist and close research collaborator of Marie Curie throughout her scientific career."},
    {"name": "polonium", "type": "concept",
     "description": "Polonium is a radioactive element discovered by Marie Curie in 1898."},
    {"name": "radium", "type": "concept",
     "description": "Radium is a radioactive element discovered by Marie Curie in 1898."},
    {"name": "university of paris", "type": "organization",
     "description": "The University of Paris was the institution where Marie Curie conducted her research."}
  ],
  "relationships": [
    {"source": "marie curie", "target": "polonium",
     "relation": "discovered", "description": "Marie Curie discovered polonium in 1898 during her research.", "weight": 9},
    {"source": "marie curie", "target": "radium",
     "relation": "discovered", "description": "Marie Curie discovered radium in 1898 during her research.", "weight": 9},
    {"source": "marie curie", "target": "university of paris",
     "relation": "worked at", "description": "Marie Curie conducted her scientific research at the University of Paris.", "weight": 7},
    {"source": "marie curie", "target": "pierre curie",
     "relation": "collaborated with", "description": "Marie Curie collaborated closely with Pierre Curie throughout her research career.", "weight": 8}
  ]
}"""

_EXTRACTION_EXAMPLE_2 = """\
Text:
OpenAI released GPT-4 in March 2023. The model powers ChatGPT and is accessed via the OpenAI API.
Microsoft integrated GPT-4 into Azure OpenAI Service, allowing enterprise customers to deploy it securely.
------------------------
Output:
{
  "entities": [
    {"name": "openai", "type": "organization",
     "description": "OpenAI is an AI research company that developed and released the GPT-4 language model."},
    {"name": "gpt-4", "type": "technology",
     "description": "GPT-4 is a large language model released by OpenAI in March 2023, powering ChatGPT and available via API."},
    {"name": "chatgpt", "type": "product",
     "description": "ChatGPT is a conversational AI product powered by GPT-4 and developed by OpenAI."},
    {"name": "openai api", "type": "system",
     "description": "The OpenAI API is the programmatic interface through which GPT-4 is accessed by developers."},
    {"name": "microsoft", "type": "organization",
     "description": "Microsoft integrated GPT-4 into its Azure OpenAI Service for enterprise customers."},
    {"name": "azure openai service", "type": "system",
     "description": "Azure OpenAI Service is Microsoft's enterprise offering that integrates GPT-4 for secure deployment."}
  ],
  "relationships": [
    {"source": "openai", "target": "gpt-4",
     "relation": "released", "description": "OpenAI released GPT-4 in March 2023.", "weight": 10},
    {"source": "gpt-4", "target": "chatgpt",
     "relation": "powers", "description": "GPT-4 is the underlying model that powers ChatGPT.", "weight": 9},
    {"source": "gpt-4", "target": "openai api",
     "relation": "accessed via", "description": "GPT-4 is accessed programmatically through the OpenAI API.", "weight": 8},
    {"source": "microsoft", "target": "azure openai service",
     "relation": "provides", "description": "Microsoft provides Azure OpenAI Service as an enterprise deployment platform.", "weight": 9},
    {"source": "azure openai service", "target": "gpt-4",
     "relation": "integrates", "description": "Azure OpenAI Service integrates GPT-4 to allow secure enterprise use.", "weight": 8}
  ]
}"""


class GraphIndexer:
    """Ingestion pipeline that builds the knowledge graph from documents.

    Call ``build_graph(documents)`` once after chunking and embedding.
    The resulting ``entities``, ``relationships``, ``graph``, and
    ``communities`` attributes are consumed by ``AdvancedGraphRAGRetriever``
    at query time.

    Example::

        indexer = GraphIndexer(embedding_provider, llm_provider, config)
        indexer.build_graph(chunked_documents)
        # indexer.entities, indexer.communities now populated
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        config: RetrievalConfig,
    ):
        if not _NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required for GraphIndexer. "
                "Install it with: pip install rag_sdk[advanced-graph-rag]"
            )
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.config = config

        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.graph: "nx.Graph" = nx.Graph()
        self.communities: Dict[str, Community] = {}

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def build_graph(self, documents: List[Document]) -> None:
        """Run the full ingestion pipeline.

        Phases:
        1. Extract entities + relationships from each chunk.
        2. Merge duplicate descriptions across chunks via LLM.
        3. Build a weighted networkx graph.
        4. Detect hierarchical communities (Leiden → Louvain → greedy).
        5. Generate a structured LLM report per community.
        6. Embed community summaries for semantic scoring at query time.
        """
        cfg = self.config.advanced_graph_rag

        entity_descs: Dict[str, List[str]] = {}
        rel_map: Dict[Tuple[str, str], Relationship] = {}
        rel_descs: Dict[Tuple[str, str], List[str]] = {}

        # --- Phase 1: Extraction ---
        for doc in documents:
            entities, rels = self._extract_entities_and_relationships(
                doc.content, doc.id
            )

            for entity in entities:
                if entity.name in self.entities:
                    self.entities[entity.name].document_ids.append(doc.id)
                else:
                    self.entities[entity.name] = entity

                if entity.name not in entity_descs:
                    entity_descs[entity.name] = []
                if entity.description:
                    entity_descs[entity.name].append(entity.description)

                if entity.name not in self.graph:
                    self.graph.add_node(entity.name, entity_type=entity.entity_type)

            for rel in rels:
                key = (rel.source, rel.target)
                if key not in rel_map:
                    rel_map[key] = rel
                    rel_descs[key] = []
                else:
                    rel_map[key].document_ids.append(doc.id)

                if rel.description:
                    rel_descs[key].append(rel.description)

                for node in (rel.source, rel.target):
                    if node not in self.graph:
                        self.graph.add_node(node)

                edge_weight = rel.weight if cfg.relationship_weight_in_graph else 1.0
                if self.graph.has_edge(rel.source, rel.target):
                    existing = self.graph[rel.source][rel.target].get("weight", 1.0)
                    self.graph[rel.source][rel.target]["weight"] = max(
                        existing, edge_weight
                    )
                else:
                    self.graph.add_edge(
                        rel.source,
                        rel.target,
                        relation=rel.relation,
                        weight=edge_weight,
                    )

        # --- Phase 2: Merge entity descriptions ---
        for name, descs in entity_descs.items():
            if name not in self.entities:
                continue
            if len(descs) > 1:
                self.entities[name].description = self._merge_descriptions(
                    "entity", name, descs
                )
            elif len(descs) == 1:
                self.entities[name].description = descs[0]

        # --- Phase 3: Merge relationship descriptions ---
        for key, descs in rel_descs.items():
            rel = rel_map[key]
            if len(descs) > 1:
                rel.description = self._merge_descriptions(
                    "relationship", f"{key[0]} → {key[1]}", descs
                )
            elif len(descs) == 1:
                rel.description = descs[0]

        self.relationships = list(rel_map.values())
        logger.info(
            "Extracted %d entities, %d relationships",
            len(self.entities),
            len(self.relationships),
        )

        # --- Phase 4: Community detection ---
        raw_communities = self._detect_communities()
        logger.info("Detected %d communities across all levels", len(raw_communities))

        # --- Phase 5: Structured community reports ---
        self._summarize_communities(raw_communities)

        # --- Phase 6: Embed summaries ---
        self._embed_community_summaries()

        logger.info("Graph index ready: %d communities", len(self.communities))

    # ---------------------------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------------------------

    def _extract_entities_and_relationships(
        self, text: str, document_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a chunk using a few-shot JSON prompt."""
        cfg = self.config.advanced_graph_rag
        max_ents = cfg.max_entities_per_chunk
        max_rels = cfg.max_relationships_per_chunk
        entity_types_str = ", ".join(cfg.entity_types)

        prompt = (
            "-Goal-\n"
            "Given a text document, identify all entities of the specified types and all "
            "relationships among the identified entities.\n\n"
            "-Steps-\n"
            "1. Identify all entities. For each entity extract:\n"
            "   - name: entity name, lowercased\n"
            "   - type: one of the entity types listed below\n"
            "   - description: comprehensive 1-2 sentence description of the entity's "
            "attributes and role in the text\n"
            f"   Extract at most {max_ents} entities.\n\n"
            "2. For each pair of clearly related entities extract:\n"
            "   - source: name of the source entity\n"
            "   - target: name of the target entity\n"
            "   - relation: short label for the relationship type\n"
            "   - description: explanation of why and how the entities are related\n"
            "   - weight: integer 1–10 indicating relationship strength\n"
            f"   Extract at most {max_rels} relationships.\n\n"
            f"-Entity types-\n{entity_types_str}\n\n"
            "-Examples-\n"
            "######################\n"
            f"Example 1:\n{_EXTRACTION_EXAMPLE_1}\n"
            "######################\n"
            f"Example 2:\n{_EXTRACTION_EXAMPLE_2}\n"
            "######################\n\n"
            "-Real Data-\n"
            "######################\n"
            f"Text:\n{text}\n"
            "------------------------\n"
            "Output (JSON only, no prose):"
        )

        try:
            response = self.llm_provider.generate(prompt=prompt)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                entities = [
                    Entity(
                        name=e["name"].lower(),
                        entity_type=e.get("type", ""),
                        description=e.get("description", ""),
                        document_ids=[document_id],
                    )
                    for e in data.get("entities", [])[:max_ents]
                ]
                rels = [
                    Relationship(
                        source=r["source"].lower(),
                        target=r["target"].lower(),
                        relation=r.get("relation", ""),
                        description=r.get("description", ""),
                        weight=float(r.get("weight", 1.0)),
                        document_ids=[document_id],
                    )
                    for r in data.get("relationships", [])[:max_rels]
                ]
                return entities, rels
        except Exception as e:
            logger.warning("Entity/relationship extraction failed: %s", e)

        return [], []

    def _merge_descriptions(self, kind: str, name: str, descriptions: List[str]) -> str:
        """Merge multiple chunk-level descriptions into one canonical description via LLM."""
        desc_text = "\n".join(f"- {d}" for d in descriptions)
        prompt = (
            f"Merge these {kind} descriptions for '{name}' into one comprehensive description.\n\n"
            f"Descriptions:\n{desc_text}\n\n"
            "Write a single description (2-4 sentences) capturing all key information:"
        )
        try:
            return self.llm_provider.generate(prompt=prompt).strip()
        except Exception as e:
            logger.warning("Description merging failed for %s '%s': %s", kind, name, e)
            return " ".join(descriptions)

    # ---------------------------------------------------------------------------
    # Community Detection
    # ---------------------------------------------------------------------------

    def _detect_communities(self) -> List[Tuple[int, Set[str]]]:
        """Run hierarchical community detection on the weighted networkx graph.

        Tries Leiden (leidenalg + igraph) when configured, falls back to
        Louvain (networkx ≥ 3.0) or greedy modularity (older networkx).
        Runs at multiple resolutions to produce ``community_levels`` hierarchy
        levels (0 = coarsest, higher = finer).

        Returns:
            List of (level, member_set) tuples across all hierarchy levels.
        """
        if self.graph.number_of_nodes() == 0:
            return []

        cfg = self.config.advanced_graph_rag
        levels = max(cfg.community_levels, 1)
        results: List[Tuple[int, Set[str]]] = []

        if levels == 1:
            resolutions = [1.0]
        else:
            step = 1.0 / max(levels - 1, 1)
            resolutions = [0.5 + i * step for i in range(levels)]

        # --- Try Leiden ---
        if cfg.community_detection_algorithm == "leiden":
            try:
                import igraph as ig  # noqa: PLC0415
                import leidenalg  # noqa: PLC0415

                nodes = list(self.graph.nodes())
                node_idx = {n: i for i, n in enumerate(nodes)}
                edges = [(node_idx[u], node_idx[v]) for u, v in self.graph.edges()]
                weights = [
                    self.graph[u][v].get("weight", 1.0) for u, v in self.graph.edges()
                ]
                ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
                if edges:
                    ig_graph.es["weight"] = weights

                for level, resolution in enumerate(resolutions):
                    partition = leidenalg.find_partition(
                        ig_graph,
                        leidenalg.RBConfigurationVertexPartition,
                        weights="weight" if edges else None,
                        resolution_parameter=resolution,
                        seed=42,
                    )
                    for comm in partition:
                        results.append((level, {nodes[i] for i in comm}))

                logger.info("Used Leiden algorithm for community detection")
                return results

            except ImportError:
                logger.info("leidenalg/igraph not available, falling back to Louvain")
            except Exception as e:
                logger.warning(
                    "Leiden community detection failed: %s — falling back to Louvain", e
                )

        # --- Louvain / greedy fallback ---
        for level, resolution in enumerate(resolutions):
            try:
                partition = nx_community.louvain_communities(
                    self.graph, seed=42, weight="weight", resolution=resolution
                )
                for community_set in partition:
                    results.append((level, set(community_set)))
            except TypeError:
                try:
                    partition = nx_community.louvain_communities(self.graph, seed=42)
                    for community_set in partition:
                        results.append((level, set(community_set)))
                except AttributeError:
                    partition = nx_community.greedy_modularity_communities(self.graph)
                    for community_set in partition:
                        results.append((0, set(community_set)))
                    return results
            except AttributeError:
                partition = nx_community.greedy_modularity_communities(self.graph)
                for community_set in partition:
                    results.append((0, set(community_set)))
                return results
            except Exception as e:
                logger.warning("Community detection failed at level %d: %s", level, e)

        return results

    # ---------------------------------------------------------------------------
    # Community Summarization
    # ---------------------------------------------------------------------------

    def _summarize_communities(
        self, raw_communities: List[Tuple[int, Set[str]]]
    ) -> None:
        """Generate a structured LLM report for each detected community."""
        for level, member_set in raw_communities:
            entity_names = list(member_set)
            internal_rels = [
                r
                for r in self.relationships
                if r.source in member_set and r.target in member_set
            ]

            ent_lines = []
            for name in entity_names[:20]:
                if name in self.entities:
                    e = self.entities[name]
                    desc_suffix = f": {e.description}" if e.description else ""
                    ent_lines.append(f"- {name} ({e.entity_type}){desc_suffix}")
                else:
                    ent_lines.append(f"- {name}")

            rel_lines = [
                f"- {r.source} → [{r.relation}] → {r.target}"
                + (f" — {r.description}" if r.description else "")
                for r in internal_rels[:15]
            ]

            prompt = (
                "Analyze this community of entities in a knowledge graph and generate a structured report.\n\n"
                "Entities:\n" + "\n".join(ent_lines) + "\n\n"
                "Relationships:\n"
                + ("\n".join(rel_lines) if rel_lines else "None")
                + "\n\n"
                "Return ONLY a JSON object with:\n"
                '- "title": short topic label (5-10 words)\n'
                '- "summary": 1-2 sentence overview of what this community represents\n'
                '- "findings": array of {"explanation": str, "data_refs": str} objects (2-5 findings)\n'
                '- "rank": float 1-10 indicating how important/central this topic is\n'
                "Response:"
            )

            fallback_name = entity_names[0] if entity_names else "entities"
            title = f"Community: {fallback_name}"
            summary = f"A community of related entities: {', '.join(entity_names[:5])}"
            full_content = summary
            findings: List[Dict[str, str]] = []
            rank = 5.0

            try:
                response = self.llm_provider.generate(prompt=prompt).strip()
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    title = data.get("title", title)
                    summary = data.get("summary", summary)
                    findings = data.get("findings", [])
                    rank = float(data.get("rank", rank))

                    parts = [f"# {title}", f"\n{summary}"]
                    if findings:
                        parts.append("\n## Findings")
                        for finding in findings:
                            explanation = finding.get("explanation", "")
                            data_refs = finding.get("data_refs", "")
                            parts.append(f"- {explanation}")
                            if data_refs:
                                parts.append(f"  - *References: {data_refs}*")
                    full_content = "\n".join(parts)

            except Exception as e:
                logger.warning("Community summarization failed: %s", e)

            community_id = str(uuid.uuid4())
            self.communities[community_id] = Community(
                id=community_id,
                level=level,
                entities=entity_names,
                title=title,
                summary=summary,
                full_content=full_content,
                findings=findings,
                rank=rank,
            )

    def _embed_community_summaries(self) -> None:
        """Embed all community summaries for semantic scoring at query time."""
        communities = list(self.communities.values())
        summaries = [c.summary for c in communities]
        if not summaries:
            return
        try:
            embeddings = self.embedding_provider.embed_documents(summaries)
            for community, emb in zip(communities, embeddings):
                community.embedding = emb
        except Exception as e:
            logger.warning("Community embedding failed: %s", e)
