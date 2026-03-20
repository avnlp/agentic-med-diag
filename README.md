<h1 align="center"> <a href="https://github.com/avnlp/agentic-med-diag"> Agentic Graph RAG for Medical Diagnosis </a> </h1>

An implementation of multi-hop, agentic Graph Retrieval-Augmented Generation (Graph RAG) for medical question answering. The system combines semantic and relational retrieval channels - backed by **Neo4j** and **Milvus** -  to answer complex clinical questions that require multi-hop reasoning across a medical knowledge graph.

Code coming soon.

## Features

| Feature | Description |
|---|---|
| **Agentic multi-hop reasoning** | Iterative sub-query decomposition with back-references across hops |
| **Dual retrieval channels** | Parallel semantic (text) and relational (KG triple) channels |
| **Graph RAG backends** | LightRAG, MiniRAG, PathRAG, HyperGraphRAG — switchable at runtime |
| **Knowledge graph storage** | Neo4j for entity/relationship graphs |
| **Vector storage** | Milvus for dense hybrid search over medical documents |
| **LangGraph orchestration** | Stateful subgraphs with conditional loop edges and fan-out/fan-in |
| **Comprehensive evaluation** | DeepEval metrics; comparison against LightRAG, MiniRAG, PathRAG |
| **Medical benchmarks** | HealthBench, MedCaseReasoning, MetaMedQA, PubMedQA |

## Architecture

The system is implemented as a hierarchical **LangGraph** state machine. A parent graph fans out to two parallel subgraphs — the semantic channel and the relational channel — then synthesizes their outputs into a final answer.

### Reasoning Pipelines

**Full agentic multi-hop pipeline:**

- Retrieves initial context via a GraphRAG backend
- Decomposes the question into semantic sub-queries (with `#N` back-references) and relational SPO triple queries (with `Entity#N` placeholders)
- Runs two parallel iterative retrieval-and-reasoning loops:
  - **Semantic channel**: sub-query grounding → GraphRAG retrieval → semantic filter → text summary → sub-answer → logic drafting → evidence verification → conditional expansion
  - **Relational channel**: SPO triple queries → KG filter → triplet list summaries → sub-answer
- Synthesizes a final answer combining both channels

### Storage Layer

- **Neo4j** — stores the entity/relationship knowledge graph built from medical documents
- **Milvus** — stores dense vector embeddings for semantic retrieval over document chunks

## Backends

### LightRAG

Paper: [https://arxiv.org/abs/2410.05779](https://arxiv.org/abs/2410.05779)  
GitHub: [https://github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)

[LightRAG](https://github.com/HKUDS/LightRAG) builds a **dual-level knowledge graph** — a local graph of fine-grained entity mentions and a global graph of high-level concept clusters — and retrieves context from both simultaneously in **hybrid mode**.

**Indexing:**

- Documents are chunked and passed to an LLM to extract named entities and their pairwise relationships
- Entities and relationships are stored in **Neo4j**, with embeddings stored in a vector index alongside raw document chunks
- Parallel async inserts and async LLM calls accelerate large-scale ingestion

**Retrieval (hybrid mode):**

- Performs both keyword-based KG traversal and dense vector search over document chunks
- Returns a structured JSON context with four sections: `Knowledge Graph Data (Entity)`, `Knowledge Graph Data (Relationship)`, `Document Chunks`, and a `Reference Document List`
- The `context_filter` splits this into the semantic channel (document chunks + reference list) and the relational channel (entity + relationship JSON)

### MiniRAG

Paper: [https://arxiv.org/abs/2501.06713](https://arxiv.org/abs/2501.06713)  
GitHub: [https://github.com/HKUDS/MiniRAG](https://github.com/HKUDS/MiniRAG)

[MiniRAG](https://github.com/HKUDS/MiniRAG) is a **lightweight Graph RAG** designed for resource-constrained settings. It uses a simpler graph construction process and a reduced `"light"` retrieval mode that prioritises efficiency over exhaustive graph traversal.

**Indexing:**

- Same chunking parameters as LightRAG (400 tokens, 50-token overlap)
- Extracts entities and relationships into a flat graph stored in the working directory (no Neo4j dependency)
- Lower memory and LLM-call overhead than heavier backends

**Retrieval (light mode):**

- Focuses on the most immediately relevant neighbourhood of the query entity rather than full hybrid traversal
- Returns a CSV-formatted context with three sections: `Entities`, `Relationships`, and `Sources`
- The `context_filter` splits this into the semantic channel (`Sources` CSV) and the relational channel (`Entities` + `Relationships` CSVs)

### PathRAG

Paper: [https://arxiv.org/abs/2502.14902](https://arxiv.org/abs/2502.14902)  
GitHub: [https://github.com/BUPT-GAMMA/PathRAG](https://github.com/BUPT-GAMMA/PathRAG)

[PathRAG](https://github.com/BUPT-GAMMA/PathRAG) organises the knowledge graph into a **two-tier hierarchy** — a global layer of abstract, high-level entities and relationships, and a local layer of fine-grained, low-level entities and relationships — and retrieves reasoning **paths** between entities rather than isolated nodes or edges.

**Indexing:**

- Constructs both high-level (thematic/concept) and low-level (mention-level) entity and relationship layers during ingestion
- Stores the hierarchical graph in the working directory

**Retrieval (hybrid mode):**

- Traverses connecting paths between query-relevant entities, returning chains of evidence rather than individual triples
- Returns a CSV context with five sections: `high-level entity information`, `high-level relationship information`, `Sources`, `low-level entity information`, and `low-level relationship information`
- The `context_filter` separates the semantic channel (`Sources` CSV) from the relational channel (both high-level and low-level entity/relationship CSVs)

### HyperGraphRAG

Paper: [https://arxiv.org/abs/2503.21322](https://arxiv.org/abs/2503.21322)  
GitHub: [https://github.com/LHRLAB/HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG)

[HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG) extends the standard pairwise-edge graph model with **hyperedges** — edges that connect more than two entities at once. This allows a single relationship to capture group interactions among multiple medical concepts simultaneously (e.g., a clinical syndrome that jointly implicates multiple symptoms, biomarkers, and treatments).

**Indexing:**

- Extracts entities and higher-order relationships (hyperedges) from documents during ingestion
- Represents multi-entity relationships as hyperedges in the graph, stored in the working directory

**Retrieval (hybrid mode):**

- Combines dense vector search with hypergraph traversal to surface both document chunks and multi-entity relational context
- Returns a CSV context with three sections: `Entities`, `Relationships` (including hyperedges), and `Sources`
- The `context_filter` is structurally identical to MiniRAG's split — `Sources` for the semantic channel, `Entities` + `Relationships` for the relational channel

## Evaluation

We evaluate all four pipelines against LightRAG, MiniRAG, PathRAG, and HyperGraphRAG baselines using [DeepEval](https://deepeval.com/) metrics.

**Metrics**: Contextual Recall, Contextual Precision, Contextual Relevancy, Answer Relevancy, Faithfulness

**Datasets**:

| Dataset | Description |
|---|---|
| [HealthBench](https://openai.com/index/healthbench/) | Multi-turn medical AI benchmark with expert rubric evaluations |
| [MedCaseReasoning](https://github.com/kevinwu23/Stanford-MedCaseReasoning) | Medical case studies with detailed reasoning processes |
| [MetaMedQA](https://github.com/maximegmd/MetaMedQA-benchmark) | Medical QA based on USMLE textbook content |
| [PubMedQA](https://pubmedqa.github.io/) | Biomedical QA based on PubMed articles |

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv
pip install uv

# Install dependencies
uv sync
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [GraphSearch](https://arxiv.org/abs/2509.22009)
- [LangGraph](https://www.langchain.com/langgraph)
- [Neo4j](https://neo4j.com/)
- [Milvus](https://milvus.io/)
- [DeepEval](https://deepeval.com/)
- [LightRAG](https://arxiv.org/abs/2410.05779) - [GitHub](https://github.com/HKUDS/LightRAG)
- [MiniRAG](https://arxiv.org/abs/2501.06713) - [GitHub](https://github.com/HKUDS/MiniRAG)
- [PathRAG](https://arxiv.org/abs/2502.14902) - [GitHub](https://github.com/BUPT-GAMMA/PathRAG)
- [HyperGraphRAG](https://arxiv.org/abs/2503.21322) - [GitHub](https://github.com/LHRLAB/HyperGraphRAG)
