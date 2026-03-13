# Complete Chunking Methods Guide

## Overview
Your RAG pipeline now has 7 chunking methods, each optimized for different use cases.

---

## 1. Character-based Chunking
```python
chunks = chunker.chunk_with_metadata(data, method='characters', chunk_size=500, overlap=100)
```

**How it works:** Fixed-size splitting by character count with overlap

**Pros:**
- Fast and simple
- Predictable chunk sizes
- Good for token limit management

**Cons:**
- Splits mid-sentence/mid-word
- No semantic awareness
- Poor retrieval quality

**Use case:** Quick prototyping, testing

---

## 2. Word-based Chunking
```python
chunks = chunker.chunk_with_metadata(data, method='words', chunk_size=100, overlap=20)
```

**How it works:** Splits by word count

**Pros:**
- Doesn't split words
- Simple and fast

**Cons:**
- May split mid-sentence
- No semantic awareness

**Use case:** Better than character-based, still basic

---

## 3. Sentence-based Chunking
```python
chunks = chunker.chunk_with_metadata(data, method='sentences', sentences_per_chunk=5)
```

**How it works:** Groups N sentences together

**Pros:**
- Respects sentence boundaries
- More semantic than word/character

**Cons:**
- Fixed sentence count (not size-aware)
- May group unrelated sentences

**Use case:** When sentences are self-contained units

---

## 4. Paragraph-based Chunking
```python
chunks = chunker.chunk_with_metadata(data, method='paragraph', max_chunk_size=1000)
```

**How it works:** Groups paragraphs until size limit

**Pros:**
- Respects document structure
- Natural boundaries

**Cons:**
- Assumes paragraphs = semantic units
- Not truly semantic

**Use case:** Well-formatted documents with clear paragraph structure

---

## 5. Recursive Chunking ⭐ (NEW)
```python
chunks = chunker.chunk_with_metadata(
    data, 
    method='recursive',
    chunk_size=1000,
    overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]  # Optional custom separators
)
```

**How it works:** 
1. Try splitting by paragraphs (\n\n)
2. If chunk too large → split by lines (\n)
3. If still too large → split by sentences (.)
4. If still too large → split by words ( )
5. Last resort → split by characters ("")

**Pros:**
- Respects natural text boundaries
- Gracefully handles edge cases
- Better than fixed-size methods
- Customizable separator hierarchy

**Cons:**
- Slightly slower than simple methods
- Not embedding-based

**Use case:** 
- General-purpose RAG (best default choice)
- Mixed content types
- Code chunking (custom separators)

**Similar to:** LangChain's RecursiveCharacterTextSplitter

---

## 6. Similarity-based Chunking ⭐⭐ (NEW - TRUE SEMANTIC)
```python
chunks = chunker.chunk_with_metadata(
    data,
    method='similarity',
    similarity_threshold=0.5,  # 0.0-1.0 (lower = more splits)
    max_chunk_size=1500,
    model_name='all-MiniLM-L6-v2'  # Embedding model
)
```

**How it works:**
1. Split text into sentences
2. Generate embeddings for each sentence
3. Calculate cosine similarity between consecutive sentences
4. Create new chunk when similarity drops below threshold (topic change)

**Pros:**
- TRUE semantic chunking
- Detects topic boundaries
- High-quality retrieval
- Embedding-based

**Cons:**
- Slower (needs to generate embeddings)
- Requires sentence-transformers library
- First run downloads model (~80MB)

**Use case:**
- High-quality production RAG
- Topic-based retrieval
- When quality > speed

**Parameters:**
- `similarity_threshold`: Lower = stricter boundaries (more chunks)
  - 0.3 = very strict (many small chunks)
  - 0.5 = balanced (recommended)
  - 0.7 = loose (fewer large chunks)

---

## 7. Hierarchical Chunking ⭐⭐⭐ (NEW - BEST FOR PRODUCTION)
```python
chunks = chunker.chunk_with_metadata(
    data,
    method='hierarchical',
    parent_size=2000,
    child_size=500,
    overlap=50
)
```

**How it works:**
1. Create large parent chunks (broad context)
2. Split each parent into smaller child chunks
3. Maintain parent-child relationships

**Pros:**
- Best of both worlds
- Precise retrieval (child) + rich context (parent)
- Excellent for complex queries
- Enables re-ranking strategies

**Cons:**
- More complex to implement in vector DB
- Requires storing parent references

**Use case:**
- Production RAG systems
- Complex multi-hop queries
- When you need both precision and context

**RAG Workflow:**
```python
# 1. INDEXING
for chunk in chunks:
    embedding = embed(chunk['content'])  # Embed child
    vector_db.store(
        embedding=embedding,
        metadata={
            'child_id': chunk['chunk_id'],
            'parent_id': chunk['parent_id'],
            'parent_content': chunk['parent_content']
        }
    )

# 2. RETRIEVAL
results = vector_db.search(query_embedding, top_k=3)

# 3. AUGMENTATION
context = ""
for result in results:
    context += f"Precise: {result['content']}\n"
    context += f"Context: {result['parent_content']}\n\n"

# 4. GENERATION
llm.generate(query, context)
```

---

## Comparison Table

| Method | Speed | Quality | Semantic | Use Case |
|--------|-------|---------|----------|----------|
| Character | ⚡⚡⚡ | ⭐ | ❌ | Testing |
| Word | ⚡⚡⚡ | ⭐⭐ | ❌ | Basic |
| Sentence | ⚡⚡ | ⭐⭐ | ⚠️ | Simple docs |
| Paragraph | ⚡⚡ | ⭐⭐⭐ | ⚠️ | Formatted docs |
| **Recursive** | ⚡⚡ | ⭐⭐⭐⭐ | ⚠️ | **General RAG** |
| **Similarity** | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | **High-quality RAG** |
| **Hierarchical** | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ | **Production RAG** |

---

## Recommendations

### For Quick Prototypes:
```python
chunks = chunker.chunk_with_metadata(data, method='recursive', chunk_size=1000)
```

### For Production RAG:
```python
# Option 1: Similarity-based (topic-aware)
chunks = chunker.chunk_with_metadata(
    data, 
    method='similarity',
    similarity_threshold=0.5,
    max_chunk_size=1500
)

# Option 2: Hierarchical (best quality)
chunks = chunker.chunk_with_metadata(
    data,
    method='hierarchical',
    parent_size=2000,
    child_size=500
)
```

### For Code:
```python
chunks = chunker.chunk_with_metadata(
    data,
    method='recursive',
    separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]
)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**New dependencies added:**
- sentence-transformers (for similarity chunking)
- scikit-learn (for cosine similarity)
- numpy (for array operations)

---

## Test Files

1. `test_recursive_chunking.py` - Demonstrates recursive chunking
2. `test_similarity_chunking.py` - Demonstrates similarity chunking
3. `test_all_chunking.py` - Compares all 7 methods
4. `hierarchical_example.py` - Visual hierarchical explanation

Run any test:
```bash
python test_all_chunking.py
```

---

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests to see differences
3. Choose method based on your use case
4. Integrate into your RAG pipeline
5. Experiment with parameters (chunk_size, overlap, threshold)

**Pro tip:** Start with recursive chunking, upgrade to similarity/hierarchical for production.
