<h1 align="center"><a href="https://github.com/avnlp/agentic-med-diag">Agentic GraphRAG for Medical Diagnosis</a></h1>

<div align="center">

[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/avnlp/agentic-med-diag)
[![CI](https://img.shields.io/github/actions/workflow/status/avnlp/agentic-med-diag/ci.yml?branch=main&label=CI&logo=githubactions)](https://github.com/avnlp/agentic-med-diag/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/github/actions/workflow/status/avnlp/agentic-med-diag/ci.yml?branch=main&label=Ruff&logo=ruff)](https://github.com/avnlp/agentic-med-diag/actions/workflows/ci.yml)
[![ty](https://img.shields.io/github/actions/workflow/status/avnlp/agentic-med-diag/ci.yml?branch=main&label=ty&logo=python)](https://github.com/avnlp/agentic-med-diag/actions/workflows/ci.yml)
[![Bandit](https://img.shields.io/github/actions/workflow/status/avnlp/agentic-med-diag/ci.yml?branch=main&label=Bandit&logo=owasp)](https://github.com/avnlp/agentic-med-diag/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/avnlp/agentic-med-diag/ci.yml?branch=main&label=Tests&logo=pytest)](https://github.com/avnlp/agentic-med-diag/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/github/avnlp/agentic-med-diag/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/avnlp/agentic-med-diag)
[![License](https://img.shields.io/github/license/avnlp/agentic-med-diag?color=green)](https://github.com/avnlp/agentic-med-diag/blob/main/LICENSE)

</div>

This repository implements an Agentic GraphRAG system for Medical Diagnosis. It ingests medical literature, extracts structured clinical knowledge into a typed Neo4j knowledge graph with hierarchical communities, and answers diagnostic questions through multi-strategy retrieval and an agentic plan–research–verify reasoning loop.

The system is built using:

- [**LangGraph**](https://www.langchain.com/langgraph) for orchestrating the extraction, embedding, and search pipelines.
- [**BAML**](https://www.boundaryml.com/) for type-safe, schema-injected LLM functions.
- [**DeepAgents**](https://github.com/langchain-ai/deepagents) for the multi-role agentic reasoning loop.
- [**Neo4j**](https://neo4j.com/) for the persistent knowledge graph with typed nodes, edges, and hierarchical communities.
- [**Qdrant**](https://qdrant.io/) / [**Weaviate**](https://weaviate.io/) for vector search and hybrid (dense + BM25) retrieval.
- [**GLiNER**](https://github.com/urchade/GLiNER) and [**GLiREL**](https://github.com/jackboyla/GLiREL) for local zero-shot entity and relation extraction.
- [**graspologic**](https://github.com/microsoft/graspologic) / **Neo4j GDS** for hierarchical Leiden community detection.
- [**ZeroEntropy**](https://www.zeroentropy.dev) for embeddings (`zembed-1`) and cross-encoder reranking (`zerank-2`).

## Features

- **Schema-driven knowledge graph** built from a runtime-injectable `GraphSchema` of medical entity and relation types that drives every extractor, resolver, summarizer, and prompt.
- **Three-extractor fusion** combining GLiNER NER, GLiREL relation extraction, and LLM extraction, merged with configurable union, intersection, max-score, or GLiNER-primary strategies.
- **Two-stage entity resolution** using a deterministic SemHash/MinHash pre-filter followed by clustering, BM25 + cosine candidate retrieval, and LLM deduplication.
- **Hierarchical community detection** via Leiden (graspologic-native or Neo4j GDS), with LLM-generated community reports summarising each cluster.
- **Four vector collections** (entity, relation, chunk, community) for complementary semantic search.
- **Layered retrieval** combining atomic search methods, pluggable rerankers, and data-only recipes, fused with Reciprocal Rank Fusion or a cross-encoder.
- **Agentic plan–research–verify loop** that decomposes the question, runs parallel researchers over retrieval tools, and gates synthesis on a deterministic sufficiency check.
- **Pluggable storage** with Neo4j for the graph and Qdrant or Weaviate for vectors.

## Knowledge Graph

The system stores everything in a single typed property graph in Neo4j with five node labels (`Document`, `Chunk`, `Entity`, `Community`, `CommunityReport`) and seven edge types:

```
(:Document) <-[:PART_OF]-  (:Chunk) -[:HAS_ENTITY]-> (:Entity:<Type>)
(:Chunk)    -[:NEXT_CHUNK]-> (:Chunk)
(:Entity)   -[:RELATES_TO {type, description, score}]-> (:Entity)
(:Entity)   -[:IN_COMMUNITY]->     (:Community)
(:Community)-[:PARENT_COMMUNITY]-> (:Community)
(:Community)-[:HAS_REPORT]->       (:CommunityReport)
```

### Entities

- An entity carries its `name`, medical `label` (Disease, Drug, …), an optional `description`, an extraction `score`, and free-form `schema_properties`. Its identity is `(name, label)`.
- Resolution later fills in a `canonical_name` and a list of `aliases` (for example, *metformin → metformin HCl, Glucophage*).
- A `provenance` record tracks which extractors produced the entity, the surface forms seen in the text, and the source chunk ids and offsets.

### Relations

- A relation is a directed subject–predicate–object triple (`head → type → tail`) with a `description`, `score`, and properties.
- Before resolution, endpoints are known only by name; resolution links them to canonical entity ids.
- The schema constrains each relation's valid head and tail types — `TREATED_BY` only connects `Disease → {Drug, DrugClass, Procedure}` — which the extractor uses to reject implausible triples.
- All relations persist as generic `:RELATES_TO` edges with the medical type in the `type` property.

### Communities

- After resolution, hierarchical Leiden partitions the resolved relation graph into nested communities. Each `Community` records its level, its parent community, and the entities and relations it contains.
- An LLM generates a `CommunityReport` for each community, bottom-up by level: a title, a summary, structured findings, and a clinical-importance rating. Lower-level reports roll up into higher-level ones.
- Community reports give the agent a thematic, cluster-level view, so a broad question can be answered from a single summary instead of many low-level facts.

### Schema

The schema (`MEDICAL_GRAPHRAG_SCHEMA`) is a first-class runtime value, not a config file. It is injected into every extractor, resolver, summarizer, and prompt, and into BAML as dynamic enum types, so the LLM is constrained to the schema rather than merely prompted with it. Each type carries natural-language hints used to steer GLiNER and GLiREL, descriptions used in LLM prompts, and (for relations) the allowed head and tail label sets.

The default schema defines 13 entity types:

| | | | |
|---|---|---|---|
| `Disease` | `Drug` | `DrugClass` | `Symptom` |
| `Pathogen` | `AnatomicalStructure` | `Procedure` | `DiagnosticTest` |
| `RiskFactor` | `Gene` | `Protein` | `Pathway` |
| `MechanismOfAction` | | | |

and 25 relation types, grouped by clinical role:

| Group | Relations |
|-------|-----------|
| **Clinical** (disease-centered) | `HAS_SYMPTOM`, `TREATED_BY`, `DIAGNOSED_BY`, `CAUSED_BY`, `HAS_GENETIC_CAUSE`, `AFFECTS`, `HAS_COMPLICATION`, `DIFFERENTIAL_FOR`, `HAS_RISK_FACTOR` |
| **Pharmacological** (drug-centered) | `BELONGS_TO_CLASS`, `TARGETS`, `INHIBITS`, `ACTIVATES`, `METABOLIZED_BY`, `INTERACTS_WITH`, `CONTRAINDICATED_IN`, `CAUSES_ADVERSE_EFFECT`, `MONITORED_BY`, `HAS_MECHANISM` |
| **Molecular** | `ENCODES`, `PARTICIPATES_IN` |
| **Structural** | `IS_A`, `PART_OF`, `INNERVATED_BY`, `SUPPLIED_BY` |

The schema is inspired by SNOMED CT relationship types, the UMLS semantic network, and clinical reasoning patterns.

## Architecture

The system has two data flows: ingestion (text into a knowledge graph) and retrieval (a question into an answer). Both are composed from small components, LangGraph pipelines, and pluggable storage backends.

```
              MEDICAL CORPORA  (Textbooks · StatPearls · PubMed · Case Reports · Guidelines)
                                     │  async streaming
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│  EXTRACTION PIPELINE  (LangGraph StateGraph, graph_construction components)                │
│  chunk → [ GLiNER ‖ GLiREL ‖ LLM ] → combine → normalize → aggregate → resolve             │
│                                                  └─► detect communities → summarize        │
└────────────────────────────────────────────────────────────┬───────────────────────────────┘
                                                             │  KnowledgeGraph
                            ┌────────────────────────────────┴────────────────────────────┐
                            ▼                                                             ▼
              ┌──────────────────────────┐                        ┌────────────────────────────────┐
              │  Neo4j Graph DB          │                        │  Vector Store (Qdrant/Weaviate)│
              │  Document/Chunk/Entity/  │                        │  4 collections: entity ·       │
              │  Community/Report nodes  │                        │  relation · chunk · community  │
              └────────────┬─────────────┘                        └────────────────┬───────────────┘
                           │                                                       │
                           ▼                                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│  SEARCH ENGINE  (atomic methods + rerankers + data-only recipes)                           │
│  entity · relation · chunk · community · text2cypher   ──►  RRF / cross-encoder fusion     │
└────────────────────────────────────────────────────────────┬───────────────────────────────┘
                                                             │  search() exposed as agent tools
                                                             ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│  AGENTIC LOOP  (DeepAgents + ChatOpenAI)                                                  │
│  Orchestrator → Planner → Researcher×N (parallel) → Verifier (sufficiency gate) → Answer  │
│                                  └────────── re-plan on gaps ──────────┘                  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## Ingestion Pipeline

Corpus ingestion streams documents, chunks them, and runs a LangGraph pipeline built from small `graph_construction` components. State accumulates into a `KnowledgeGraph` that is written to Neo4j and the vector store. The stages run in order:

1. **Extract.** Each chunk is processed by up to three extractors that share the injected schema:
   - **GLiNER** runs batch zero-shot NER (`gliner-community/gliner_medium-v2.5`), steered by each entity type's natural-language label.
   - **GLiREL** runs zero-shot relation extraction (`jackboyla/glirel-large-v0`) over GLiNER's entity spans, so GLiREL requires GLiNER.
   - **LLM** runs a BAML extraction function with the valid entity and relation types injected as dynamic enums, constraining the model to the schema. Chunks are processed concurrently with bounded concurrency and isolated retries.
2. **Combine.** The three extractor outputs are merged per chunk using a configurable strategy: `union` (superset, merging provenance), `intersection` (only items every extractor found), `max_score` (highest-confidence version), or `gliner_primary` (GLiNER spans supplemented by the LLM).
3. **Normalize.** Within a chunk, entities are deduplicated by `(normalized name, label)` and relations by `(normalized head, normalized tail, type)`, merging sources, surface forms, chunk ids, scores, and properties. Low-confidence and too-short entities are filtered.
4. **Aggregate.** A deterministic, zero-LLM set-union across all chunks collapses the same keys into one cross-chunk candidate set.
5. **Resolve.** Entity and relation-type names are resolved in two stages: a deterministic deduplicator collapses exact and near-exact variants (SemHash, MinHash-LSH) and clusters the residual names; then, within each cluster, BM25 + cosine fusion retrieves candidates and an LLM selects exact duplicates and a single canonical alias. The result writes `canonical_name` and `aliases` onto entities and links relation endpoints to canonical entity ids.
6. **Detect communities.** Hierarchical Leiden runs over the resolved relation graph. Two interchangeable backends share one base class: graspologic-native (default) and Neo4j GDS. The output is a tree of communities with levels and parents.
7. **Summarize.** Communities are walked bottom-up by level; a degree-ranked, token-budgeted context is built for each and passed to an LLM that produces a titled report with structured findings and a clinical-importance rating.

Resolution runs before community detection so the graph is clustered over canonical entities rather than surface-form duplicates.

## Embedding Pipeline

After graph construction, the embeddable fields of each model are vectorised and upserted into four vector collections.

| Collection | Source | Text embedded |
|------------|--------|---------------|
| `am_diag_entity` | Entity | canonical name |
| `am_diag_relation` | Relation | three representations (below) |
| `am_diag_chunk` | Chunk | chunk text |
| `am_diag_community` | CommunityReport | report summary |

A single relation is embedded three ways, all keyed by the same relation id, because one vector cannot capture the predicate, the participants, and the full statement at once:

- **Edge fact** — the relation description (for example, *metformin treats type 2 diabetes*).
- **Edge type** — the predicate alone (for example, *TREATED_BY*).
- **Full SPO** — the subject–predicate–object sentence.

Collections support lazy creation, batch upsert, and native vector quantization. The transport record is collection-agnostic — the target collection is an upsert argument — so the same record type serves all four collections.

## Retrieval Pipeline

Retrieval uses a layered design: atomic methods (`vector`, `hybrid`, `fulltext`, `bfs`) compose into retrievers, which a `SearchEngine` fans out over according to a recipe and fuses with a pluggable reranker.

- **Entity retriever** runs hybrid search over entity names, optionally expanding from the matched seed entities with bounded, degree-aware BFS.
- **Relation retriever** runs hybrid search over the relation collection, then hydrates full edges (head, tail, type, description) from the graph.
- **Chunk retriever** runs hybrid search over raw passages.
- **Community retriever** searches community report summaries for thematic answers.
- **Text-to-Cypher retriever** has an LLM translate the question into a read-only Cypher query against the schema, with few-shot examples and bounded retry.

Recipes are data-only constants that select which methods to run and how to fuse them, so a new strategy requires no orchestration change.

| Recipe | Methods | Reranker |
|--------|---------|----------|
| `entity` / `relation` / `chunk` / `community` | single method | RRF |
| `hybrid_rrf` | entity + relation + chunk + community | RRF |
| `hybrid_cross_encoder` | all four + BFS | cross-encoder |
| `bfs_expand` | entity + BFS | RRF |
| `text2cypher` | text-to-Cypher | RRF |

Results from the recipe's methods are gathered concurrently and fused with Reciprocal Rank Fusion or a cross-encoder reranker (`zerank-2`); MMR and node-distance rerankers are also available. Fusion degrades gracefully, ignoring any method that returns no results.

## Agentic RAG Loop

The agent is a DeepAgents harness that coordinates three subagents with an orchestrator agent:

- **Planner** decomposes the clinical question into focused sub-questions, each mapped to a retrieval recipe. It has no tools and works from the question text.
- **Researcher** is spawned in parallel, one per sub-question. Each holds the retrieval tools (entity, relation, chunk, community, hybrid, text-to-Cypher search, and a community map-reduce tool) wrapping the `SearchEngine` with tenacity retries, and returns distilled evidence items with citations.
- **Verifier** reads the gathered evidence and returns a structured assessment — a coverage score, an evidence-depth score, missing pieces, targeted follow-ups, and unsupported claims. A deterministic numeric gate decides whether the evidence is sufficient.

When the gate fails, the targeted follow-ups seed another planning round, so the loop converges on missing information rather than repeating searches. When it passes (or iterations are exhausted), the orchestrator synthesises a structured `ClinicalAnswer` with answer text, source citations, a confidence score, clinical caveats, and an answerability flag.

## Structured LLM Output (BAML)

Every LLM interaction outside the agent loop is a typed BAML function with explicit inputs, outputs, retry policy, and provider configuration.

| BAML Function | Purpose |
|---------------|---------|
| `ExtractKnowledgeGraph` | Schema-constrained entity and relation extraction (dynamic enums) |
| `IdentifyClusterDuplicates` | Per-cluster entity/edge deduplication and canonical alias |
| `AreEntityMentionsSame` | Pairwise entity coreference arbitration |
| `GenerateCommunityReport` | Bottom-up community summary with findings and rating |
| `MapCommunityBatch` / `ReduceMapResponses` | Global map-reduce QA over community reports |
| `GenerateCypherQuery` | Natural language to validated read-only Cypher |

## Embedding Models and Rerankers

| Backend | Type | Notes |
|---------|------|-------|
| **Zembed-1** | Local with learned projections | Medical embeddings with learned dimension reduction. |
| **SentenceTransformers** | Local (CPU/GPU) | Zero-cost and private; configurable precision and dimensions. |
| **OpenAI** | API | General-purpose embeddings with no local storage. |
| **Cross-encoder reranker** (`zerank-2`) | Local / API | Post-retrieval relevance filtering with a configurable minimum score. |

## Datasets

We provide loaders for medical QA benchmarks across three evaluation formats.

### MCQ (Exact-Match Accuracy)

| Dataset | Loader | Format |
|---------|--------|--------|
| MedQA (USMLE) | `MedQADataset` | 4-option MCQ |
| MedMCQA | `MedMCQADataset` | 4-option MCQ |
| PubMedQA | `PubMedQADataset` | yes/no/maybe |
| MMLU-Med | `MMLUMedDataset` | 4-option MCQ |
| MMLU-Pro (Health) | `MMLUProHealthDataset` | 10-option MCQ |
| MedXpertQA (Text) | `MedXpertQADataset` | ~10-option MCQ |
| CareQA (MCQ) | `CareQADataset` | 4-option MCQ |
| NEJM Q&A | `NEJMQADataset` | 4–5-option MCQ |
| PubHealthBench | `PubHealthBenchDataset` | 4-option MCQ |
| SuperGPQA-Med | `SuperGPQAMedDataset` | up to 10-option MCQ |

### Rubric-Scored (LLM-as-Judge)

| Dataset | Loader | Format |
|---------|--------|--------|
| HealthBench | `HealthBenchDataset` | Multi-turn rubric-scored conversations |
| RAR-Med | `RARMedDataset` | Instance-specific rubrics per prompt |

### Open-Ended (LLM-as-Judge)

| Dataset | Loader | Format |
|---------|--------|--------|
| MedCaseReasoning | `MedCaseReasoningDataset` | Diagnosis from structured case prompts |
| CareQA (Reasoning) | `CareQAReasoningDataset` | Open-ended clinical questions |
| PubHealthBench (Freeform) | `PubHealthBenchFreeformDataset` | Free-form public-health answers |
| NEJM Diagnostic Reasoning | `NEJMDiagnosticDataset` | Diagnosis from full CPC vignettes |

## Corpora

We provide streaming loaders for medical text corpora.

| Corpus | Loader | Hugging Face Dataset |
|--------|--------|----------------------|
| USMLE Textbooks | `TextbooksCorpusLoader` | `MedRAG/textbooks` |
| StatPearls | `StatPearlsCorpusLoader` | `awinml/statpearls` |
| PubMed Abstracts | `PubMedCorpusLoader` | `MedRAG/pubmed` |
| PMC Case Reports | `PubmedCaseReportsCorpusLoader` | `zou-lab/MedCaseReasoning` |
| Meditron Clinical Guidelines | `ClinicalGuidelinesCorpusLoader` | `epfl-llm/guidelines` |

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/avnlp/agentic-med-diag.git
cd agentic-med-diag
pip install uv && uv sync
```

**Prerequisites:** Python ≥ 3.11, Neo4j (AuraDB or self-hosted), a vector store (Qdrant or Weaviate), and an OpenAI-compatible LLM endpoint.

## Usage

### Environment Setup

Create a `.env` file with the required credentials. Settings are env-overridable per subsystem, for example:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
QDRANT_URL=http://localhost:6333
AGENT_BASE_URL=https://api.openai.com/v1
AGENT_API_KEY=your_api_key
AGENT_MODEL=gpt-4o
```

### Running the Pipelines

`am_diag/pipelines` provides two CLIs (registered as `[project.scripts]`) for end-to-end ingestion and QA.

```bash
# Ingestion — loops all 5 corpus loaders
uv run am-diag-ingest
uv run am-diag-ingest --corpus pubmed,statpearls --batch-size 50

# QA runner — loops all 16 datasets through the agent
uv run am-diag-qa
uv run am-diag-qa --datasets careqa,medqa --limit 10
```

Both CLIs create Neo4j, Qdrant, and an embedder from env settings, run the pipeline, and print a summary. The QA runner writes per-dataset JSON files to `results/` (configurable via `--output-dir` or `QA_OUTPUT_DIR`).

### Programmatic Usage

Ingest a corpus into the knowledge graph:

```python
from am_diag.loaders.corpus import StatPearlsCorpusLoader
from am_diag.db.graph import create_neo4j_client
from am_diag.vector.embedding import ZembedEmbedder
from am_diag.ingestion import run_corpus_ingestion

report = await run_corpus_ingestion(
    corpus_loader=StatPearlsCorpusLoader(),
    graph_store=create_neo4j_client(),
    vector_store=vector_store,
    embedder=ZembedEmbedder(),
    batch_size=100,
)
```

Search with multi-strategy retrieval:

```python
from am_diag.retrieval import SearchEngine, RetrievalConfig

engine = SearchEngine(
    config=RetrievalConfig(),
    vector_store=vector_store,
    graph_store=graph_store,
    embedder=embedder,
    schema=MEDICAL_GRAPHRAG_SCHEMA,
    reranker=reranker,
)
results = await engine.search("What treats hypertension in chronic kidney disease?",
                              recipe="hybrid_rrf")
```

Answer a clinical question with the agentic loop:

```python
from am_diag.agents import answer_question, AgentSettings

answer = await answer_question(
    "What are first-line treatments for hypertension in a patient with type 2 diabetes?",
    search_engine=engine,
    settings=AgentSettings(),
)
```

## Project Structure

```text
am_diag/
├── common/
│   ├── data_models/        # all domain models (Entity, Relation, Community, Chunk, ...)
│   ├── cypher/             # externalised .cypher files + loader
│   └── schema/             # MEDICAL_GRAPHRAG_SCHEMA
├── chunking/               # recursive-character + markitdown chunkers
├── graph_construction/
│   ├── extract/            # GLiNER, GLiREL, LLM extractors + combiner
│   ├── normalize.py        # per-chunk dedup/normalization
│   ├── aggregate.py        # cross-chunk set-union
│   ├── resolve/            # deterministic + cluster + LLM resolution
│   └── community/          # Leiden / GDS detection + summarization
├── ingestion/              # LangGraph extraction / embedding / search pipelines
├── pipelines/              # end-to-end ingestion + QA CLIs
├── db/
│   ├── graph/              # Neo4j client + record serialization
│   └── vector/             # Qdrant / Weaviate stores
├── vector/                 # embedders + rerankers
├── retrieval/              # methods · retrievers · rerankers · recipes · SearchEngine
├── agents/                 # DeepAgents harness
├── llm/                    # BAML sources + generated client
└── loaders/                # corpus loaders + dataset loaders
```

## Contributing

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## References

- [Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) — hierarchical communities and community reports
- [Graphiti](https://github.com/getzep/graphiti) — layered graph retrieval and search recipes
- [KG-Gen](https://github.com/stair-lab/kg-gen) — knowledge graph extraction with cluster + LLM alias deduplication
- [YouTu-GraphRAG](https://github.com/TencentCloudADP/youtu-graphrag) — hierarchical agentic graph retrieval
- [LightRAG](https://github.com/HKUDS/LightRAG) — dual-level graph + vector retrieval
- [PathRAG](https://github.com/BUPT-GAMMA/PathRAG) — relational-path pruning over the graph
- [Cognee](https://github.com/topoteretes/cognee) — DataPoint-based graph memory and ECL pipelines
- [MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) / [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) — medical RAG ablations and corpora
- [MEDITRON-70B](https://arxiv.org/abs/2311.16079) — medical pretraining with the GAP-Replay corpus
- [GLiNER](https://arxiv.org/abs/2311.08526) / [GLiREL](https://github.com/jackboyla/GLiREL) — generalist NER and relation extraction
- [BAML](https://boundaryml.com), [LangGraph](https://www.langchain.com/langgraph), [DeepAgents](https://github.com/langchain-ai/deepagents), [ZeroEntropy](https://www.zeroentropy.dev)

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
