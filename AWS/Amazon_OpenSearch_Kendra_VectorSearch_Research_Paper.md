# Research Paper: Amazon OpenSearch, Amazon Kendra & Vector Search for RAG

## Abstract

This paper provides an in-depth analysis of Amazon OpenSearch Service and Amazon Kendra — two AWS services that power search and retrieval — and their critical relationship to vector search, embeddings, and Retrieval Augmented Generation (RAG). It explains how vector databases work, compares OpenSearch vs Kendra for different use cases, and shows how they integrate with Amazon Bedrock Knowledge Bases to build GenAI applications.

---

## 1. The Search Problem in the AI Era

Traditional search uses **keyword matching** — if you search "how to fix login error", it looks for documents containing those exact words. But what if the answer is in a document that says "troubleshooting authentication failures"? Keyword search misses it.

**Vector search** solves this by understanding **meaning** (semantics), not just words.

```
Traditional Search:  "fix login error" → matches documents with "fix", "login", "error"
Vector Search:       "fix login error" → matches documents about authentication problems
                     (even if they don't contain those exact words)
```

---

## 2. What is Vector Search?

### 2.1 How It Works

```
Step 1: EMBED — Convert text to numbers (vectors)
   "The cat sat on the mat" → [0.23, -0.45, 0.67, ..., 0.12]  (1024 numbers)

Step 2: STORE — Save vectors in a vector database

Step 3: SEARCH — Convert query to vector, find nearest neighbors
   "Where did the feline rest?" → [0.21, -0.43, 0.65, ..., 0.14]  (similar vector!)
   → Finds "The cat sat on the mat" because vectors are close in space
```

### 2.2 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Embedding** | A numerical representation (vector) of text/image that captures its meaning |
| **Vector** | An array of numbers (e.g., 1024 floats) representing a piece of content |
| **Dimensions** | How many numbers in the vector (256, 512, 1024, 1536) |
| **Similarity** | How "close" two vectors are (cosine similarity, dot product, Euclidean distance) |
| **k-NN** | k-Nearest Neighbors — find the k most similar vectors to a query |
| **HNSW** | Hierarchical Navigable Small World — fast approximate nearest neighbor algorithm |
| **Index** | Data structure that organizes vectors for efficient search |
| **RAG** | Retrieval Augmented Generation — feed retrieved context to an LLM for accurate answers |

### 2.3 Why Vector Search Matters for GenAI

LLMs (like Claude, GPT) have a knowledge cutoff and don't know your private data. RAG solves this:

```
User Question → Embed Question → Vector Search (find relevant docs) → Feed docs + question to LLM → Accurate Answer
```

Without RAG: LLM hallucinates or says "I don't know"
With RAG: LLM answers accurately using YOUR data

---

## 3. Amazon OpenSearch Service

### 3.1 Overview

Amazon OpenSearch Service is a managed search and analytics engine (fork of Elasticsearch) that supports full-text search, log analytics, AND vector search (k-NN).

### 3.2 Deployment Options

| Option | Description | Best For |
|--------|-------------|----------|
| **Managed Domains** | You choose instance types, storage, configure clusters | Full control, predictable workloads |
| **Serverless** | No infrastructure management, auto-scales OCUs | Variable workloads, no ops |

### 3.3 OpenSearch Serverless Collection Types

| Type | Purpose | Storage | k-NN Support |
|------|---------|---------|--------------|
| **Search** | Full-text search (e-commerce, CMS) | Hot only | ❌ |
| **Time Series** | Log analytics, metrics | Hot + Warm | ❌ |
| **Vector Search** | Semantic search, RAG, embeddings | Hot only | ✅ |

### 3.4 Vector Search in OpenSearch

OpenSearch supports the `knn_vector` field type with up to **16,000 dimensions**.

**Search Methods:**
- **Exact k-NN**: Brute-force, 100% accurate, slow on large datasets
- **Approximate k-NN (HNSW)**: Fast, ~95-99% accurate, scales to billions of vectors

**Distance Metrics:**
- Cosine similarity (most common for text)
- Euclidean distance (L2)
- Dot product (for normalized vectors)

**GPU Acceleration**: NextGen vector search collections use GPU to build indexes faster.

### 3.5 OpenSearch Vector Search — Configuration Example

```python
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# Connect to OpenSearch Serverless
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, 
                   'us-east-1', 'aoss', session_token=credentials.token)

client = OpenSearch(
    hosts=[{'host': '<collection-endpoint>', 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    connection_class=RequestsHttpConnection
)

# Create vector index
index_body = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512
        }
    },
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "content_vector": {
                "type": "knn_vector",
                "dimension": 1024,          # Match your embedding model
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            "metadata": {"type": "keyword"}
        }
    }
}

client.indices.create(index="knowledge-base", body=index_body)
```

### 3.6 Indexing Documents with Embeddings

```python
import boto3
import json

# Generate embedding using Amazon Titan
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def get_embedding(text):
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    )
    return json.loads(response['body'].read())['embedding']

# Index a document
doc = {
    "content": "Amazon OpenSearch supports vector search using k-NN algorithms.",
    "content_vector": get_embedding("Amazon OpenSearch supports vector search using k-NN algorithms."),
    "metadata": "opensearch-docs"
}

client.index(index="knowledge-base", body=doc)
```

### 3.7 Searching with Vectors

```python
# Semantic search
query_text = "How do I do similarity search?"
query_vector = get_embedding(query_text)

search_body = {
    "size": 5,
    "query": {
        "knn": {
            "content_vector": {
                "vector": query_vector,
                "k": 5
            }
        }
    }
}

results = client.search(index="knowledge-base", body=search_body)
for hit in results['hits']['hits']:
    print(f"Score: {hit['_score']:.3f} | {hit['_source']['content'][:100]}")
```

### 3.8 S3 Vector Engine (Cost-Optimized)

OpenSearch can offload vector storage to S3 for cost-effective large-scale vector search:

```python
# Create index with S3 vector engine
index_body = {
    "settings": {
        "index": {
            "knn": True,
            "knn.vector_engine": "s3vector"  # Store vectors in S3!
        }
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "s3vector"}
            }
        }
    }
}
```

---

## 4. Amazon Kendra

### 4.1 Overview

Amazon Kendra is a managed **intelligent search** service that uses NLP and deep learning to understand natural language queries. Unlike OpenSearch (which you configure), Kendra is a higher-level service — you connect data sources and it handles indexing, ranking, and retrieval automatically.

### 4.2 How Kendra Differs from OpenSearch

| Aspect | Kendra | OpenSearch |
|--------|--------|------------|
| **Abstraction Level** | High (managed search) | Low (you configure everything) |
| **Query Type** | Natural language questions | Structured queries (DSL) + k-NN |
| **Indexing** | Automatic (connect data sources) | Manual (you index documents) |
| **Ranking** | ML-powered semantic ranking | BM25 + k-NN scoring |
| **Data Sources** | 40+ built-in connectors | You ingest data yourself |
| **Access Control** | Built-in ACL filtering | You implement |
| **Setup Time** | Minutes (connect & go) | Hours/days (configure cluster) |
| **Customization** | Limited | Full control |
| **Cost** | Higher (managed intelligence) | Lower (infrastructure only) |

### 4.3 Kendra Editions

| Edition | Purpose | Availability | Bedrock Integration |
|---------|---------|--------------|---------------------|
| **GenAI Enterprise** | Highest accuracy, hybrid search, RAG-optimized | Production | ✅ (Bedrock Knowledge Bases) |
| **Basic Enterprise** | Semantic search, high availability | Production | ❌ |
| **Basic Developer** | Proof of concept, testing | Development only | ❌ |

### 4.4 Kendra Data Source Connectors (40+)

| Category | Connectors |
|----------|-----------|
| **Cloud Storage** | Amazon S3, Google Drive, OneDrive |
| **Collaboration** | Confluence, SharePoint, Slack, Microsoft Teams |
| **CRM** | Salesforce, ServiceNow |
| **Databases** | Amazon RDS, Aurora |
| **Websites** | Web Crawler (with auth support) |
| **Code** | GitHub, GitLab |
| **Others** | Quip, Jira, Zendesk, Box, Dropbox |

### 4.5 Query Types

| Query Type | Example | How Kendra Handles It |
|-----------|---------|----------------------|
| **Factoid** | "What is the return policy?" | Returns exact answer snippet |
| **Descriptive** | "How do I set up VPN?" | Returns relevant passage/document |
| **Keyword** | "quarterly report 2024" | Semantic + keyword matching |
| **Conversational** | "Tell me more about that" | Context-aware follow-up |

### 4.6 Kendra + Bedrock Knowledge Bases

Kendra GenAI indexes integrate directly with Amazon Bedrock Knowledge Bases:

```
Documents → Kendra GenAI Index (automatic indexing + semantic ranking)
                    ↓
Bedrock Knowledge Base (connects to Kendra as retriever)
                    ↓
User Query → Retrieve from Kendra → Augment prompt → LLM generates answer
```

This gives you the best of both worlds: Kendra's 40+ connectors + Bedrock's LLM generation.

---

## 5. Vector Search Comparison: All AWS Options

### 5.1 AWS Vector Database Options

| Service | Type | Max Dimensions | Algorithms | Serverless | Best For |
|---------|------|---------------|------------|------------|----------|
| **OpenSearch Serverless** | Vector collection | 16,000 | HNSW, IVF | ✅ | RAG, semantic search at scale |
| **OpenSearch Managed** | k-NN plugin | 16,000 | HNSW, IVF, Faiss | ❌ (managed) | Full control, hybrid search |
| **Amazon S3 Vectors** | Object storage | Varies | Flat, HNSW | ✅ | Cost-optimized, large-scale |
| **Amazon Aurora pgvector** | PostgreSQL extension | 2,000 | IVFFlat, HNSW | ❌ | Existing PostgreSQL apps |
| **Amazon Neptune Analytics** | Graph + vector | 65,536 | — | ✅ | Graph + vector combined |
| **Amazon MemoryDB** | Redis-compatible | 32,768 | HNSW, Flat | ❌ | Ultra-low latency |
| **Amazon Kendra** | Managed search | Internal | Proprietary | ✅ | Enterprise search, 40+ connectors |

### 5.2 Decision Guide: Which Vector Store?

| Scenario | Recommended | Why |
|----------|-------------|-----|
| RAG with Bedrock Knowledge Bases (simple) | **OpenSearch Serverless** | Native integration, auto-created by Bedrock console |
| Enterprise search with ACLs + 40 data sources | **Kendra GenAI** | Built-in connectors, access control, no vector management |
| Existing PostgreSQL application | **Aurora pgvector** | No new infrastructure, familiar SQL |
| Ultra-low latency (<1ms) | **MemoryDB** | In-memory, Redis-compatible |
| Cost-optimized billions of vectors | **S3 Vectors** or **OpenSearch S3 engine** | Cheapest storage |
| Graph relationships + vector search | **Neptune Analytics** | Combined graph traversal + similarity |
| Full control over search pipeline | **OpenSearch Managed** | Custom analyzers, hybrid scoring |
| Maximum accuracy, zero config | **Kendra GenAI** | ML-powered ranking, no tuning needed |

---

## 6. RAG Architecture Patterns

### 6.1 Pattern 1: Bedrock KB + OpenSearch Serverless (Most Common)

```
Documents (S3) → Bedrock KB (chunks + embeds) → OpenSearch Serverless (stores vectors)
                                                          ↓
User Query → Bedrock KB (embeds query) → OpenSearch (k-NN search) → Top-K results
                                                          ↓
                                              LLM (generates answer with context)
```

**Setup**: Bedrock console auto-creates OpenSearch Serverless collection.

### 6.2 Pattern 2: Bedrock KB + Kendra GenAI (Enterprise)

```
40+ Data Sources → Kendra GenAI Index (auto-indexes with connectors)
                                    ↓
User Query → Bedrock KB → Kendra (semantic retrieval) → Top-K results
                                    ↓
                        LLM (generates answer with citations)
```

**Best for**: Organizations with data in SharePoint, Confluence, Salesforce, etc.

### 6.3 Pattern 3: Custom RAG with OpenSearch (Full Control)

```python
import boto3, json
from opensearchpy import OpenSearch

# 1. Embed the query
bedrock = boto3.client('bedrock-runtime')
query = "What is our refund policy?"
query_embedding = get_embedding(query)

# 2. Search OpenSearch
results = opensearch_client.search(
    index="company-docs",
    body={
        "size": 3,
        "query": {"knn": {"embedding": {"vector": query_embedding, "k": 3}}}
    }
)

# 3. Build context from results
context = "\n".join([hit['_source']['content'] for hit in results['hits']['hits']])

# 4. Generate answer with LLM
response = bedrock.invoke_model(
    modelId="anthropic.claude-sonnet-4-20250514-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{
            "role": "user",
            "content": f"Based on this context:\n{context}\n\nAnswer: {query}"
        }]
    })
)

answer = json.loads(response['body'].read())['content'][0]['text']
print(answer)
```

---

## 7. OpenSearch vs Kendra — When to Use Which

### 7.1 Use Kendra When...

- ✅ You need enterprise search with 40+ data source connectors
- ✅ You need built-in access control (ACL) filtering
- ✅ You want zero vector management (no embeddings, no indexing)
- ✅ You need factoid answers (not just document retrieval)
- ✅ You're building with Bedrock Knowledge Bases and want highest accuracy
- ✅ Your data is in SharePoint, Confluence, Salesforce, ServiceNow
- ✅ You want ML-powered ranking without tuning

### 7.2 Use OpenSearch When...

- ✅ You need full control over search pipeline and scoring
- ✅ You're building custom RAG with your own embedding models
- ✅ You need hybrid search (keyword BM25 + vector k-NN combined)
- ✅ You have log analytics + search in the same platform
- ✅ You need sub-second latency at massive scale
- ✅ You want to use OpenSearch Dashboards for visualization
- ✅ Cost is a priority (OpenSearch is cheaper than Kendra)
- ✅ You need custom analyzers, tokenizers, or scoring functions

### 7.3 Use Both Together When...

- ✅ Kendra for enterprise search (employees finding documents)
- ✅ OpenSearch for RAG backend (GenAI applications)
- ✅ Kendra for intelligent ranking + OpenSearch for log analytics
- ✅ Kendra re-ranks OpenSearch results (Kendra Intelligent Ranking)

---

## 8. Pricing Comparison

| Service | Pricing Model | Approximate Cost |
|---------|--------------|------------------|
| **Kendra GenAI Enterprise** | Per index + per query | ~$1,008/month base + $0.35/1000 queries |
| **Kendra Developer** | Per index | ~$810/month base |
| **OpenSearch Serverless** | Per OCU-hour | ~$0.24/OCU-hour (min 2 OCUs = ~$350/month) |
| **OpenSearch Managed** | Per instance-hour + storage | From ~$0.036/hr (t3.small) |
| **S3 Vectors** | Per storage + requests | ~$0.023/GB + $0.0004/1000 requests |
| **Aurora pgvector** | Per instance + storage | From ~$0.08/hr (db.t3.medium) |

**Cost Winner by Scale:**
- Small (< 100K docs): Aurora pgvector or OpenSearch Managed (t3)
- Medium (100K–10M docs): OpenSearch Serverless
- Large (10M+ docs, enterprise): Kendra GenAI (if you need connectors) or OpenSearch (if you manage yourself)
- Massive (billions of vectors): S3 Vectors + OpenSearch S3 engine

---

## 9. User Story: "DataCorp" — Building an AI Knowledge Assistant

### The Problem

DataCorp has 500,000 documents across SharePoint, Confluence, S3, and a PostgreSQL database. Employees spend 3 hours/day searching for information. They want an AI assistant that answers questions accurately from their data.

### The Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATACORP AI ASSISTANT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DATA SOURCES:                                                   │
│  ├── SharePoint (HR policies, procedures)     ──┐               │
│  ├── Confluence (engineering docs, runbooks)   ──┤               │
│  ├── S3 (PDFs, reports, contracts)            ──┼── KENDRA      │
│  └── Web Crawler (internal wiki)              ──┘  GenAI Index  │
│                                                       │          │
│  RETRIEVAL LAYER:                                     ↓          │
│  ├── Amazon Kendra GenAI Index (enterprise search + ACLs)       │
│  │       ↕ (connected as retriever)                             │
│  └── Amazon Bedrock Knowledge Base                              │
│              │                                                   │
│  GENERATION LAYER:                                               │
│  └── Amazon Bedrock (Claude Sonnet 4) → Generates answers       │
│              │                                                   │
│  INTERFACE:                                                      │
│  ├── Slack Bot (Amazon Lex + Lambda)                            │
│  ├── Web Chat (Bedrock Agent)                                   │
│  └── Internal Portal (React + API Gateway)                      │
│                                                                  │
│  SECURITY:                                                       │
│  └── Kendra ACLs ensure users only see docs they have access to │
└─────────────────────────────────────────────────────────────────┘
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| Time to find information | 3 hours/day | 30 seconds |
| Questions answered correctly | N/A | 92% accuracy |
| Employee satisfaction | 2.8/5 | 4.6/5 |
| Support tickets (internal IT) | 500/week | 150/week |

---

## 10. Conclusion

Amazon OpenSearch and Amazon Kendra serve complementary roles in the search and retrieval ecosystem:

- **OpenSearch** = Infrastructure-level search engine with full control over vector search, k-NN algorithms, hybrid scoring, and custom pipelines. Ideal for developers building custom RAG systems.
- **Kendra** = Application-level intelligent search with 40+ connectors, built-in ACLs, ML-powered ranking, and zero vector management. Ideal for enterprise search and Bedrock KB integration.
- **Vector Search** = The bridge between traditional search and GenAI. It enables semantic understanding by converting text to numerical representations and finding similar content by meaning, not keywords.

For RAG applications, the choice depends on your data sources, control requirements, and budget. Most organizations use OpenSearch Serverless (via Bedrock KB auto-creation) for GenAI apps and Kendra for enterprise employee search.

---

## References

1. OpenSearch Vector Search: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/vector-search.html
2. OpenSearch Serverless: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-overview.html
3. OpenSearch k-NN: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html
4. OpenSearch S3 Vector Engine: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/s3-vector-opensearch-integration-engine.html
5. What is Amazon Kendra: https://docs.aws.amazon.com/kendra/latest/dg/what-is-kendra.html
6. Kendra Index Types: https://docs.aws.amazon.com/kendra/latest/dg/hiw-index-types.html
7. Bedrock KB + Kendra: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-kendra-genai-index.html
8. Choosing a Vector Database for RAG: https://docs.aws.amazon.com/prescriptive-guidance/latest/choosing-an-aws-vector-database-for-rag-use-cases/introduction.html
9. OpenSearch Serverless Vector Search: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-vector-search.html

---

*Paper generated based on AWS official documentation. Features and pricing are subject to change.*
