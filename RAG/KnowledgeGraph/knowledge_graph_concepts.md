# Knowledge Graphs — Complete Guide (From Zero)

We'll use one running example throughout:

```
Domain:   A small company's employee database
Entities: People, Companies, Skills, Projects
Goal:     Store, query, and reason about relationships between them
```

---

# 1. What is a Knowledge Graph?

## The Core Idea

A Knowledge Graph is a way to store information as **things** (entities) and **relationships** between them — instead of rows and columns like a database.

```
Traditional Database (table):
┌────────┬───────────┬──────────┬─────────────┐
│ Name   │ Role      │ Company  │ Skill       │
├────────┼───────────┼──────────┼─────────────┤
│ Alice  │ Engineer  │ Acme     │ Python      │
│ Bob    │ Manager   │ Acme     │ Leadership  │
│ Carol  │ Engineer  │ Beta Inc │ Python      │
└────────┴───────────┴──────────┴─────────────┘

Knowledge Graph (relationships):
  Alice ──works_at──→ Acme
  Alice ──has_skill──→ Python
  Alice ──has_role──→ Engineer
  Bob ──works_at──→ Acme
  Bob ──manages──→ Alice
  Bob ──has_skill──→ Leadership
  Carol ──works_at──→ Beta Inc
  Carol ──has_skill──→ Python
  Carol ──knows──→ Alice
```

The graph captures **relationships** that a table can't easily show — like "Bob manages Alice" or "Carol knows Alice".

## Why not just use a table?

```
Question: "Who does Alice know that also knows Python?"

Table approach:
  - Need complex JOINs across multiple tables
  - Gets messy with many relationship types
  - Slow for deeply connected queries

Knowledge Graph approach:
  - Just follow the edges: Alice → knows → Carol → has_skill → Python ✓
  - Natural, fast, intuitive
```

---

# 2. The Three Building Blocks

Every Knowledge Graph is made of just three things:

## Nodes (Entities)

Things that exist — people, places, concepts, objects.

```
Nodes in our example:
  [Alice]  [Bob]  [Carol]  [Acme]  [Beta Inc]  [Python]  [Leadership]  [Engineer]  [Manager]

Each node can have properties:
  Alice { age: 30, location: "NYC" }
  Acme  { founded: 2010, industry: "Tech" }
```

## Edges (Relationships)

Connections between nodes — always have a **direction** and a **label**.

```
  Alice ──works_at──→ Acme        (Alice works at Acme, not Acme works at Alice)
  Bob ──manages──→ Alice          (Bob manages Alice)
  Alice ──has_skill──→ Python     (Alice has the skill Python)

Edge direction matters:
  Alice ──manages──→ Bob    ≠    Bob ──manages──→ Alice
```

## Triplets (Subject → Predicate → Object)

The fundamental unit of a Knowledge Graph. Every piece of information is stored as a triplet.

```
(Subject)     (Predicate)     (Object)
─────────     ───────────     ────────
 Alice         works_at        Acme
 Alice         has_skill       Python
 Alice         has_role        Engineer
 Bob           works_at        Acme
 Bob           manages         Alice
 Bob           has_skill       Leadership
 Carol         works_at        Beta Inc
 Carol         has_skill       Python
 Carol         knows           Alice

Read as: "Alice works_at Acme", "Bob manages Alice"
```

**Everything in a Knowledge Graph is triplets.** If you can express information as (Subject, Predicate, Object), you can put it in a Knowledge Graph.

---

# 3. Types of Knowledge Graphs

## Directed vs Undirected

```
Directed (most Knowledge Graphs):
  Alice ──manages──→ Bob          (one-way: Alice manages Bob)

Undirected:
  Alice ──colleagues──── Bob      (two-way: they are colleagues of each other)
```

## Common Types

| Type | What it stores | Example |
|------|---------------|---------|
| General Knowledge Graph | World knowledge | Google Knowledge Graph, Wikidata, DBpedia |
| Domain Knowledge Graph | Specific field | Medical KG (diseases → symptoms → treatments) |
| Enterprise Knowledge Graph | Company data | Employees → projects → skills → departments |
| Temporal Knowledge Graph | Time-based facts | "Alice worked_at Acme from 2020 to 2023" |

---

# 4. How Knowledge Graphs are Built

## Step 1: Define your entities and relationships

```
Entities (node types):
  - Person: Alice, Bob, Carol
  - Company: Acme, Beta Inc
  - Skill: Python, Leadership, Java
  - Project: ProjectX, ProjectY

Relationships (edge types):
  - works_at: Person → Company
  - has_skill: Person → Skill
  - manages: Person → Person
  - works_on: Person → Project
  - requires: Project → Skill
```

## Step 2: Extract triplets from data

From raw text:
```
Text: "Alice is an engineer at Acme. She knows Python and works on ProjectX."

Extracted triplets:
  (Alice, has_role, Engineer)
  (Alice, works_at, Acme)
  (Alice, has_skill, Python)
  (Alice, works_on, ProjectX)
```

This extraction can be done:
- **Manually** — human reads and creates triplets
- **Rule-based** — regex/NLP patterns (e.g., "X works at Y" → (X, works_at, Y))
- **LLM-based** — ask an LLM to extract triplets from text (most common today)

## Step 3: Store in a graph database or structure

```
Options:
  - Neo4j (most popular graph database)
  - NetworkX (Python library, good for small graphs)
  - Amazon Neptune (managed graph database on AWS)
  - LlamaIndex KnowledgeGraphIndex (for RAG applications)
```

## Step 4: Query the graph

```
Query: "What skills does Alice have?"
Traversal: Alice → has_skill → [Python]

Query: "Who at Acme knows Python?"
Traversal: [?] → works_at → Acme  AND  [?] → has_skill → Python
Answer: Alice ✓ (works at Acme + knows Python)

Query: "What skills does ProjectX need that Carol has?"
Traversal: ProjectX → requires → [skills] ∩ Carol → has_skill → [skills]
```

---

# 5. Knowledge Graphs + RAG (Why it matters)

## Traditional RAG (Vector Search only)

```
User asks: "Who should I assign to ProjectX?"

Vector RAG:
  1. Embed the question
  2. Search vector store for similar chunks
  3. Might find: "ProjectX requires Python and Java"
  4. Might find: "Alice knows Python"
  5. But MISSES: "Alice works at Acme which owns ProjectX"
     → no connection between chunks!

Problem: Vector search finds similar TEXT, not connected FACTS
```

## Knowledge Graph RAG (Graph + Vector Search)

```
User asks: "Who should I assign to ProjectX?"

KG-RAG:
  1. Find ProjectX in the graph
  2. Follow edges: ProjectX → requires → [Python, Java]
  3. Follow edges: [?] → has_skill → Python → [Alice, Carol]
  4. Follow edges: [?] → has_skill → Java → [Alice]
  5. Check: Alice → works_at → Acme → owns → ProjectX ✓
  6. Answer: "Alice — she knows both Python and Java, and works at Acme which owns ProjectX"

Result: Connected reasoning across multiple hops!
```

## When to use which?

| Scenario | Use Vector RAG | Use KG-RAG |
|----------|---------------|------------|
| Simple Q&A ("What is X?") | ✓ | |
| Relationship queries ("How is X related to Y?") | | ✓ |
| Multi-hop reasoning ("Who knows someone who...") | | ✓ |
| Large unstructured text | ✓ | |
| Structured domain knowledge | | ✓ |
| Both structured + unstructured | | ✓ (hybrid) |

---

# 6. Triplet Extraction — How LLMs Build Knowledge Graphs

## The key step: turning text into triplets

```
Input text:
  "Alice is a senior engineer at Acme Corp. She has been working on ProjectX
   since 2022. The project requires expertise in Python and machine learning.
   Bob, her manager, approved the project budget last quarter."

LLM extracts triplets:
  (Alice, has_role, Senior Engineer)
  (Alice, works_at, Acme Corp)
  (Alice, works_on, ProjectX)
  (ProjectX, started, 2022)
  (ProjectX, requires, Python)
  (ProjectX, requires, Machine Learning)
  (Bob, manages, Alice)
  (Bob, approved_budget_for, ProjectX)
```

## How LlamaIndex does it automatically

```python
from llama_index.core import KnowledgeGraphIndex

# LlamaIndex reads your documents, sends chunks to LLM, extracts triplets
graph_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,    # limit triplets per chunk (controls cost/quality)
    include_embeddings=True       # also create embeddings for hybrid search
)

# What happens internally:
# Chunk 1: "Alice is an engineer at Acme" → LLM → (Alice, is, Engineer), (Alice, works_at, Acme)
# Chunk 2: "Bob manages Alice"            → LLM → (Bob, manages, Alice)
# ... for all chunks
```

## max_triplets_per_chunk explained

```
max_triplets_per_chunk = 2:
  Chunk: "Alice is an engineer at Acme. She knows Python and Java and works on ProjectX."
  Extracts only 2: (Alice, is, Engineer), (Alice, works_at, Acme)
  Misses: Python, Java, ProjectX relationships

max_triplets_per_chunk = 5:
  Same chunk → extracts all 5 relationships
  More complete but: slower, more LLM calls, higher cost

Trade-off:
  Low (1-2):  Fast, cheap, but incomplete graph
  Medium (3-5): Good balance
  High (10+): Complete but slow and expensive
```

---

# 7. Graph Traversal — How Queries Work

## Single-hop query (one edge)

```
Query: "Where does Alice work?"

Traversal:
  Alice ──works_at──→ ?

  Start at node "Alice"
  Follow edge "works_at"
  Arrive at "Acme"

Answer: Acme
```

## Multi-hop query (multiple edges)

```
Query: "What skills are needed for the project Alice works on?"

Traversal:
  Alice ──works_on──→ ProjectX ──requires──→ ?

  Start at "Alice"
  Hop 1: Follow "works_on" → arrive at "ProjectX"
  Hop 2: Follow "requires" → arrive at ["Python", "Machine Learning"]

Answer: Python and Machine Learning
```

## Complex query (multiple paths)

```
Query: "Find people who know Python AND work at the same company as Bob"

Traversal:
  Path 1: ? ──has_skill──→ Python           → [Alice, Carol]
  Path 2: Bob ──works_at──→ Acme            → Acme
  Path 3: ? ──works_at──→ Acme              → [Alice, Bob]

  Intersection: [Alice, Carol] ∩ [Alice, Bob] = [Alice]

Answer: Alice
```

## This is why KGs are powerful for RAG

Vector search can find "Alice knows Python" and "Bob works at Acme" separately, but it **can't connect them** to answer "Who knows Python at Bob's company?" — that requires **graph traversal**.

---

# 8. Visualization — Seeing the Graph

## Why visualize?

A graph with 100+ nodes is impossible to understand as a list of triplets. Visualization makes patterns visible.

## Common layouts

```
Spring Layout (force-directed):
  - Nodes repel each other like magnets
  - Edges pull connected nodes together
  - Result: clusters of related nodes form naturally

         [Python] ←── [Alice] ──→ [Acme]
            ↑              ↑          ↑
         [Carol]      [manages]    [Bob]

Hierarchical Layout (tree-like):
  - Top-down structure
  - Good for org charts, taxonomies

              [Acme]
             /      \
          [Alice]  [Bob]
           /    \      \
      [Python] [ProjectX] [Leadership]

Circular Layout:
  - All nodes on a circle
  - Edges cross through the middle
  - Good for seeing overall connectivity
```

## Tools for visualization

| Tool | Best for | Language |
|------|----------|---------|
| PyVis | Interactive HTML graphs | Python |
| NetworkX + Matplotlib | Static graph images | Python |
| Neo4j Browser | Large graph databases | Cypher queries |
| D3.js | Custom web visualizations | JavaScript |

---

# 9. Knowledge Graphs in Production — Key Concepts

## Entity Resolution

The same thing can be mentioned in different ways. Entity resolution merges them.

```
Without entity resolution:
  (Alice Smith, works_at, Acme)
  (A. Smith, has_skill, Python)
  (alice, works_on, ProjectX)
  → 3 separate nodes for the same person!

With entity resolution:
  Alice Smith = A. Smith = alice → merged into ONE node [Alice Smith]
  (Alice Smith, works_at, Acme)
  (Alice Smith, has_skill, Python)
  (Alice Smith, works_on, ProjectX)
```

## Ontology (Schema for your graph)

Defines what types of nodes and edges are allowed — like a schema for a database.

```
Ontology:
  Node types:   Person, Company, Skill, Project
  Edge types:   works_at (Person → Company)
                has_skill (Person → Skill)
                manages (Person → Person)
                requires (Project → Skill)

Without ontology:
  (Alice, employed_by, Acme)
  (Bob, works_for, Acme)
  (Carol, is_at, Acme)
  → 3 different edge labels for the same relationship!

With ontology:
  All normalized to: works_at
  (Alice, works_at, Acme)
  (Bob, works_at, Acme)
  (Carol, works_at, Acme)
```

## Graph Embeddings

Just like word embeddings turn words into vectors, graph embeddings turn **nodes and edges** into vectors.

```
Node embedding:
  Alice   → [0.8, 0.2, 0.5, ...]    (captures: engineer, Python, Acme)
  Bob     → [0.7, 0.3, 0.4, ...]    (captures: manager, Leadership, Acme)
  Carol   → [0.8, 0.1, 0.6, ...]    (captures: engineer, Python, Beta Inc)

  Alice and Carol are close in vector space (both engineers with Python)
  Alice and Bob are somewhat close (same company)

Use cases:
  - Find similar entities (nearest neighbors)
  - Link prediction ("Alice probably knows Carol" — close vectors + shared skills)
  - Cluster detection (group similar nodes)
```

---

# 10. Knowledge Graph vs Vector Store vs Relational DB

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Feature         │ Relational DB    │ Vector Store     │ Knowledge Graph  │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Stores          │ Rows & columns   │ Vectors          │ Nodes & edges    │
│ Query style     │ SQL              │ Similarity search│ Graph traversal  │
│ Best for        │ Structured data  │ Semantic search  │ Relationships    │
│ Finds           │ Exact matches    │ Similar content  │ Connected facts  │
│ Multi-hop       │ Complex JOINs    │ Can't do it      │ Natural & fast   │
│ Schema          │ Rigid (tables)   │ None (flexible)  │ Flexible ontology│
│ Example query   │ SELECT * WHERE   │ "Find similar to"│ "How is X        │
│                 │ name='Alice'     │ this text"       │  related to Y?"  │
│ Scales to       │ Billions of rows │ Billions vectors │ Billions of edges│
│ Example tool    │ PostgreSQL       │ Pinecone, Chroma │ Neo4j, Neptune   │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**In production RAG, you often combine all three:**
```
User query: "Who should lead the Python migration project?"

1. Vector Store → finds relevant documents about Python migration
2. Knowledge Graph → finds people with Python skills, their roles, availability
3. Relational DB → checks project deadlines, budget, team capacity

Combined answer: "Alice — she has Python expertise (KG), led similar migrations (vector),
                  and has availability next quarter (DB)"
```

---

# 11. Frequently Asked Questions

## Q: How is a Knowledge Graph different from a regular graph?

**A:** A regular graph just has nodes and edges. A Knowledge Graph adds **meaning** — nodes have types (Person, Company), edges have labels (works_at, manages), and everything is stored as semantic triplets.

```
Regular graph:    A → B → C        (what does this mean? no idea)
Knowledge Graph:  Alice →works_at→ Acme →located_in→ NYC   (clear meaning)
```

## Q: How big can a Knowledge Graph get?

**A:**
```
Small:    100-1000 nodes        (company team graph)
Medium:   10K-100K nodes        (enterprise knowledge base)
Large:    1M-100M nodes         (Wikidata has ~100M items)
Massive:  1B+ nodes             (Google Knowledge Graph has 500B+ facts)
```

## Q: Can I build a Knowledge Graph without an LLM?

**A:** Yes. Three approaches:
```
1. Manual:     Human creates triplets (accurate but slow)
2. Rule-based: NLP patterns extract triplets (fast but limited)
3. LLM-based:  LLM extracts triplets from text (flexible, most common today)

For production: usually LLM extraction + human review for quality
```

## Q: What is the difference between a Knowledge Graph and an Ontology?

**A:**
```
Ontology = the SCHEMA (what types of things and relationships are allowed)
  "A Person can work_at a Company"
  "A Project can require a Skill"

Knowledge Graph = the DATA (actual instances following the schema)
  "Alice works_at Acme"
  "ProjectX requires Python"

Ontology is the blueprint. Knowledge Graph is the building.
```

## Q: How do you update a Knowledge Graph when information changes?

**A:**
```
Alice gets promoted:
  Old: (Alice, has_role, Engineer)
  New: (Alice, has_role, Senior Engineer)

Options:
  1. Delete old triplet, add new one (simple)
  2. Add timestamp: (Alice, has_role, Engineer, until=2024)
                    (Alice, has_role, Senior Engineer, from=2024)
     → Temporal Knowledge Graph (keeps history)
```

## Q: What are the limitations of Knowledge Graphs?

**A:**
```
1. Building is expensive    → extracting triplets from millions of docs takes time/money
2. Incomplete by nature     → can't capture every relationship
3. Schema design is hard    → wrong ontology = useless graph
4. Ambiguity               → "Apple" = company or fruit? (entity resolution needed)
5. Maintenance              → knowledge changes, graph must be updated
6. Doesn't understand text  → needs vector search for semantic understanding
   → That's why KG + Vector Store (hybrid) is the production standard
```

## Q: What is a "hop" in graph queries?

**A:** Each edge you follow is one hop.
```
1-hop: Alice → works_at → Acme                          (direct relationship)
2-hop: Alice → works_at → Acme → located_in → NYC       (2 edges followed)
3-hop: Alice → knows → Carol → works_at → Beta → located_in → London

More hops = deeper reasoning, but also slower and noisier
Production systems usually limit to 2-3 hops
```

## Q: How does Knowledge Graph RAG handle questions the graph can't answer?

**A:**
```
Hybrid approach:
  1. Try graph traversal first → if answer found, return it
  2. If not found in graph → fall back to vector search
  3. Combine both results → send to LLM for final answer

Example:
  Q: "What is Alice's opinion on remote work?"
  Graph: no "opinion" edges → can't answer
  Vector: finds document where Alice wrote about remote work → answers it
```

---

# 12. Rules & Constraints

## Hard Rules

| # | Rule | What happens if broken |
|---|------|----------------------|
| 1 | Every edge must connect exactly 2 nodes | Crash — edge with missing node is invalid |
| 2 | Triplets must be (Subject, Predicate, Object) | Data becomes meaningless without structure |
| 3 | Node IDs must be unique | Duplicate IDs merge unrelated entities |
| 4 | Edge direction matters in directed graphs | "Alice manages Bob" ≠ "Bob manages Alice" |

## Soft Rules (best practices)

| # | Rule | What happens if broken |
|---|------|----------------------|
| 5 | Define an ontology before building | Inconsistent edge labels, messy graph |
| 6 | Resolve entities (merge duplicates) | Same person appears as multiple nodes |
| 7 | Limit triplets per chunk (2-5) | Too many = noisy/expensive, too few = incomplete |
| 8 | Validate extracted triplets | LLM may hallucinate fake relationships |
| 9 | Keep edge labels consistent | "works_at" vs "employed_by" vs "works_for" = chaos |
| 10 | Add embeddings for hybrid search | Graph-only search misses semantic similarity |

---

# 13. Inputs, Parameters & Configuration

## What inputs does a Knowledge Graph RAG system need?

### Building the Graph
```
┌──────────────────────┬──────────────────────────────────────────────┐
│ Input                │ Example                                      │
├──────────────────────┼──────────────────────────────────────────────┤
│ Documents            │ PDFs, text files, web pages, API responses   │
│ (raw source data)    │ "Alice is an engineer at Acme..."            │
├──────────────────────┼──────────────────────────────────────────────┤
│ Ontology (optional)  │ Node types: [Person, Company, Skill]         │
│ (schema definition)  │ Edge types: [works_at, has_skill, manages]   │
├──────────────────────┼──────────────────────────────────────────────┤
│ LLM for extraction   │ GPT-4, Claude, or local model               │
│ (extracts triplets)  │ Reads text → outputs (S, P, O) triplets     │
├──────────────────────┼──────────────────────────────────────────────┤
│ Embedding model      │ text-embedding-3-small, all-MiniLM-L6-v2    │
│ (for hybrid search)  │ Converts nodes/text to vectors               │
└──────────────────────┴──────────────────────────────────────────────┘
```

### Querying the Graph
```
┌──────────────────────┬──────────────────────────────────────────────┐
│ Input                │ Example                                      │
├──────────────────────┼──────────────────────────────────────────────┤
│ User query           │ "Who at Acme knows Python?"                  │
│ (natural language)   │                                              │
├──────────────────────┼──────────────────────────────────────────────┤
│ Graph index          │ The built knowledge graph with triplets      │
│ (pre-built)          │                                              │
├──────────────────────┼──────────────────────────────────────────────┤
│ LLM for generation   │ Takes retrieved subgraph + query → answer    │
│ (generates answer)   │                                              │
└──────────────────────┴──────────────────────────────────────────────┘
```

## Parameters & Hyperparameters

### Graph Construction Parameters

| Parameter | What it controls | Typical value | How to think about it |
|-----------|-----------------|---------------|----------------------|
| `max_triplets_per_chunk` | How many triplets to extract from each text chunk | 2-5 | Higher = more complete graph but slower and more expensive |
| `chunk_size` | Size of text chunks before triplet extraction | 512-1024 tokens | Bigger chunks = more context for extraction, but may miss details |
| `chunk_overlap` | Overlap between consecutive chunks | 50-200 tokens | Prevents cutting relationships at chunk boundaries |
| `include_embeddings` | Whether to also create vector embeddings | True | Enables hybrid search (graph + vector). Almost always True |
| `embedding_model` | Which model creates the vectors | text-embedding-3-small | Balances quality vs cost. Larger models = better but more expensive |

### Query Parameters

| Parameter | What it controls | Typical value | How to think about it |
|-----------|-----------------|---------------|----------------------|
| `similarity_top_k` | Number of similar results to retrieve | 3-10 | Higher = more context but may include noise |
| `temperature` | Randomness of LLM response | 0.0-0.3 | Low = precise/factual. High = creative. For RAG, keep low |
| `max_tokens` | Maximum length of generated answer | 512-2048 | Depends on expected answer length |
| `max_hops` | How many edges to traverse | 2-3 | More hops = deeper reasoning but slower and noisier |
| `response_mode` | How to synthesize the answer | "tree_summarize" | Options: compact, refine, tree_summarize |

### Graph Database Parameters (for production)

| Parameter | What it controls | Typical value | How to think about it |
|-----------|-----------------|---------------|----------------------|
| `graph_store` | Where to store the graph | Neo4j, Neptune | In-memory (NetworkX) for dev, database for production |
| `batch_size` | Triplets inserted per batch | 100-1000 | Higher = faster ingestion but more memory |
| `index_type` | How nodes are indexed for lookup | B-tree, hash | Affects query speed on large graphs |

### Parameters vs Hyperparameters

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Hyperparameters (YOU decide)         Parameters (SYSTEM learns) │
│  ─────────────────────────           ──────────────────────────  │
│  max_triplets_per_chunk = 3          Node embeddings             │
│  chunk_size = 512                    Edge weights (if weighted)  │
│  similarity_top_k = 5               Entity resolution mappings  │
│  temperature = 0.1                   LLM extraction patterns    │
│  max_hops = 2                                                    │
│  embedding_model = "..."                                         │
│                                                                  │
│  Set BEFORE building/querying        Learned/computed during     │
│  Fixed for a given setup             graph construction          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
