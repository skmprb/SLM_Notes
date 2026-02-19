# Chapter 3: Building Index-Based RAG with LlamaIndex, Deep Lake, and OpenAI

## 1. Why Index-Based RAG?

**Core Advancement**: Indexes add transparency, speed, and precision to RAG systems.

**Key Benefits**:
1. **Traceability**: Track responses back to exact source data and location
2. **Speed**: Direct access via indices vs sequential vector comparison
3. **Precision**: Organized nodes with metadata for accurate retrieval
4. **Control**: Identify and fix data issues causing poor outputs
5. **Transparency**: Mitigates bias, hallucinations, copyright concerns

**Evolution**: From vector-based similarity → Index-based structured retrieval

---

## 2. Vector-Based vs Index-Based Search

| Feature | Vector-Based | Index-Based |
|---------|-------------|-------------|
| **Flexibility** | High | Medium (precomputed) |
| **Speed** | Slower (large datasets) | Fast, optimized |
| **Scalability** | Limited by real-time processing | Highly scalable |
| **Complexity** | Simpler setup | Requires indexing step |
| **Updates** | Easy to update | Requires re-indexing |
| **Traceability** | Limited | Full source tracking |

---

## 3. Architecture: Three-Pipeline System

### Pipeline #1: Data Collection & Preparation (D1-D2)
- Collect documents from sources (GitHub, Wikipedia)
- Clean and prepare data
- **Key Enhancement**: Store each document separately with metadata
- Enable full traceability to source files

### Pipeline #2: Data Embedding & Storage (D2-D3)
- Chunk data (automated optimization)
- Embed using OpenAI models (seamless)
- Load into Deep Lake vector store
- Create indexes over documents

### Pipeline #3: Index-Based RAG (D4, G1-G4, E1)
- Query vector store with user input
- Retrieve relevant nodes using indexes
- Augment input with retrieved data
- Generate response with LLM
- Evaluate with time and score metrics

---

## 4. Dataset Structure (4 Tensors)

### Standard Configuration:
1. **text** (str): Content of text chunks
2. **metadata** (json): Source filename, file path, node information
3. **embedding** (float32): Vector representation (e.g., 1536 dimensions)
4. **id** (str): Unique identifier (auto-populated)

### Metadata Components:
- `file_path`: Path to source file
- `file_name`: Name of source file
- `file_type`: Type (e.g., text/plain)
- `file_size`: Size in bytes
- `creation_date`: File creation date
- `last_modified_date`: Last modification date
- `_node_content`: Detailed node information
  - `id_`: Unique node identifier
  - `text`: Actual text content
  - `start_char_idx`: Starting character index
  - `end_char_idx`: Ending character index
  - `document_id`: Source document identifier
  - `relationships`: Links to other nodes

---

## 5. Integration: LlamaIndex + Deep Lake + OpenAI

### Seamless Package:
```python
!pip install llama-index-vector-stores-deeplake==0.1.6
!pip install deeplake==3.9.8
!pip install llama-index==0.10.64
```

### Key Advantage:
- **Encapsulated functionality**: Chunking, embedding, indexing automated
- **No manual configuration**: Default optimized settings
- **Customizable**: Can override defaults when needed
- **Industrial-grade**: Production-ready integration

### Import:
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
```

---

## 6. Four Index Types

### 1. Vector Store Index
**Mechanism**: Embedding-based similarity search in vector space

**Implementation**:
```python
from llama_index.core import VectorStoreIndex
vector_store_index = VectorStoreIndex.from_documents(documents)
vector_query_engine = vector_store_index.as_query_engine(
    similarity_top_k=k, 
    temperature=temp, 
    num_output=mt
)
```

**Characteristics**:
- Fast retrieval
- Semantic similarity-based
- Best for: Similar meaning searches

### 2. Tree Index
**Mechanism**: Hierarchical structure with summaries at higher levels, details at lower levels

**Implementation**:
```python
from llama_index.core import TreeIndex
tree_index = TreeIndex.from_documents(documents)
tree_query_engine = tree_index.as_query_engine(
    similarity_top_k=k, 
    temperature=temp, 
    num_output=mt
)
```

**Characteristics**:
- Optimized hierarchy (not classical tree)
- LLM acts like multiple-choice question answering
- Traverses relevant nodes only
- Best for: Large datasets, hierarchical data

### 3. List Index (Summary Index)
**Mechanism**: LLM evaluates each document independently, assigns relevance scores

**Implementation**:
```python
from llama_index.core import ListIndex
list_index = ListIndex.from_documents(documents)
list_query_engine = list_index.as_query_engine(
    similarity_top_k=k, 
    temperature=temp, 
    num_output=mt
)
```

**Characteristics**:
- Prompt-based selection (not rule-based)
- Each document scored independently
- Can synthesize from multiple nodes
- Best for: Comprehensive document review

### 4. Keyword Table Index
**Mechanism**: Extract keywords, organize in table, link to document IDs

**Implementation**:
```python
from llama_index.core import KeywordTableIndex
keyword_index = KeywordTableIndex.from_documents(documents)
keyword_query_engine = keyword_index.as_query_engine(
    similarity_top_k=k, 
    temperature=temp, 
    num_output=mt
)
```

**Characteristics**:
- Keyword → Document ID mapping
- Fast keyword-based retrieval
- Best for: Specific term searches

---

## 7. Query Parameters

### Standard Configuration:
```python
user_input = "How do drones identify vehicles?"

# Parameters
k = 3           # Top-k most probable responses (ranking)
temp = 0.1      # Low = precise, High (0.9) = creative
mt = 1024       # Max output tokens
```

### Query Execution:
```python
import time
start_time = time.time()
response = query_engine.query(user_input)
end_time = time.time()
elapsed_time = end_time - start_time
```

---

## 8. Data Collection & Preparation

### Document Loading:
```python
from llama_index.core import SimpleDirectoryReader

# Load documents with metadata
documents = SimpleDirectoryReader("./data/").load_data()

# Access document info
documents[0]  # Shows file_path, file_name, file_type, etc.
```

### Cleaning Function:
```python
import re
from bs4 import BeautifulSoup

def clean_text(content):
    content = re.sub(r'\[\d+\]', '', content)  # Remove [1], [2]
    content = re.sub(r'[^\w\s\.]', '', content)  # Remove punctuation
    return content

def fetch_and_clean(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('div', {'class': 'mw-parser-output'})
    text = content.get_text(separator=' ', strip=True)
    return clean_text(text)
```

### Save Separately:
```python
output_dir = './data/'
for url in urls:
    article_name = url.split('/')[-1].replace('.html', '')
    filename = os.path.join(output_dir, article_name + '.txt')
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(fetch_and_clean(url))
```

---

## 9. Vector Store Creation & Population

### Create Vector Store:
```python
from llama_index.core import StorageContext

vector_store_path = "hub://user/dataset_name"
dataset_path = "hub://user/dataset_name"

# Create vector store
vector_store = DeepLakeVectorStore(
    dataset_path=dataset_path, 
    overwrite=True  # True = create new, False = append
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index over documents
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)
```

### Load Dataset:
```python
import deeplake
ds = deeplake.load(dataset_path)

# Check size
ds_size_mb = ds.size_approx() / 1048576
ds_size_gb = ds.size_approx() / 1073741824
```

---

## 10. Query Response & Source Tracing

### Query Function:
```python
import pandas as pd
import textwrap

def index_query(input_query):
    response = vector_query_engine.query(input_query)
    print(textwrap.fill(str(response), 100))
    
    node_data = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        node_info = {
            'Node ID': node.id_,
            'Score': node_with_score.score,
            'Text': node.text
        }
        node_data.append(node_info)
    
    df = pd.DataFrame(node_data)
    return df, response
```

### Trace to Source:
```python
# Get node ID
nodeid = response.source_nodes[0].node_id

# Get full text of node
full_text = response.source_nodes[0].get_text()

# Get chunk size
for node_with_score in response.source_nodes:
    node = node_with_score.node
    chunk_size = len(node.text)
    print(f"Node ID: {node.id_}, Chunk Size: {chunk_size} characters")
```

---

## 11. Optimized Chunking

### Automated Variable Chunking:
- LlamaIndex determines chunk size automatically
- Variable sizes (e.g., 1806 to 4417 characters)
- Semantic optimization (not linear cutting)
- Balances context and retrieval efficiency

### Example Output:
```
Node ID: 83a135c6-dddd-402e-9423-d282e6524160, Chunk Size: 4417 chars
Node ID: 7b7b55fe-0354-45bc-98da-0a715ceaaab0, Chunk Size: 1806 chars
Node ID: 18528a16-ce77-46a9-bbc6-5e8f05418d95, Chunk Size: 3258 chars
```

---

## 12. Evaluation Metrics

### Metric 1: Weighted Average Performance
```python
import numpy as np

def info_metrics(response):
    scores = [node.score for node in response.source_nodes if node.score]
    if scores:
        weights = np.exp(scores) / np.sum(np.exp(scores))
        perf = np.average(scores, weights=weights) / elapsed_time
    else:
        perf = 0
    
    print(f"Average score: {np.mean(scores):.4f}")
    print(f"Query execution time: {elapsed_time:.4f} seconds")
    print(f"Performance metric: {perf:.4f}")
```

**Formula**: Performance = (Weighted Average Score) / Elapsed Time

### Metric 2: Cosine Similarity
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    return cosine_similarity([embeddings1], [embeddings2])[0][0]

# Calculate
similarity_score = calculate_cosine_similarity_with_embeddings(
    user_input, 
    str(response)
)
performance = similarity_score / elapsed_time
```

---

## 13. Performance Comparison (Example Results)

### Vector Store Index:
- Query time: ~0.88 seconds
- Average score: 0.8374
- Performance metric: 0.6312
- **Best for**: Speed and accuracy balance

### Tree Index:
- Query time: ~4.34 seconds
- Cosine similarity: 0.731
- Performance metric: 0.1686
- **Best for**: Hierarchical data, large datasets

### List Index:
- Query time: ~16.31 seconds
- Cosine similarity: 0.775
- Performance metric: 0.0475
- **Best for**: Comprehensive document analysis

### Keyword Index:
- Query time: ~2.43 seconds
- Cosine similarity: 0.801
- Performance metric: 0.3299
- **Best for**: Keyword-specific searches

**Note**: Times vary due to stochastic algorithms and dataset complexity.

---

## 14. Keyword Index Structure

### Extract and Visualize:
```python
import pandas as pd

# Extract keyword table
data = []
for keyword, doc_ids in keyword_index.index_struct.table.items():
    for doc_id in doc_ids:
        data.append({"Keyword": keyword, "Document ID": doc_id})

df = pd.DataFrame(data)
```

**Structure**: Each keyword → Document ID(s) mapping

---

## 15. Key Technical Insights

### Seamless Integration Benefits:
1. **No manual chunking**: Automated optimization
2. **No embedding configuration**: Default models work well
3. **No index creation code**: Built-in functionality
4. **Customizable**: Override defaults when needed

### LLM Evolution:
- From "magic" to industrialized components
- Multi-functional: Embedding, ranking, conversation
- Seamless integration like GPUs in computers
- Multiple providers: OpenAI, Cohere, AI21, Hugging Face

### Real-World Data Challenges:
- Heterogeneous sources (GitHub, Wikipedia)
- Unstructured formats
- Variable quality
- Noisy data (unrelated content)
- **Solution**: Robust RAG pipelines with indexing

---

## 16. Best Practices

1. **Separate Pipelines**: Enable parallel team development
2. **Metadata Enrichment**: Ensure full traceability
3. **Test Multiple Index Types**: No silver bullet—project-dependent
4. **Monitor Performance**: Time + quality metrics
5. **Optimize Chunking**: Let LlamaIndex auto-optimize initially
6. **Trace Responses**: Use node IDs to verify sources
7. **Handle Noise**: Build robust systems for imperfect data
8. **Freeze Versions**: Prevent dependency conflicts
9. **Evaluate Stochastic Variance**: Run multiple times
10. **Balance Speed vs Accuracy**: Choose index type accordingly

---

## 17. Key Takeaways

1. **Indexes enable full traceability** from response to source data
2. **Four index types** serve different use cases (vector, tree, list, keyword)
3. **LlamaIndex + Deep Lake + OpenAI** = seamless integration
4. **Automated chunking** optimizes semantic retrieval
5. **Performance metrics** combine quality and speed
6. **Tree index** creates optimized hierarchies (not classical trees)
7. **List index** uses LLM for independent document scoring
8. **Keyword index** provides fast term-based retrieval
9. **Metadata** critical for transparency and control
10. **No universal best index**—test for each project

---

## 18. Formulas & Metrics

### Performance Metric:
```
Performance = (Weighted Average Score) / Elapsed Time
```

### Weighted Average:
```
weights = exp(scores) / sum(exp(scores))
weighted_avg = average(scores, weights=weights)
```

### Cosine Similarity:
```
similarity = (A · B) / (||A|| × ||B||)
Range: [-1, 1]
```

### Dataset Size:
```
size_mb = bytes / 1048576
size_gb = bytes / 1073741824
```

---

## Interview-Ready Q&A

**Q: What's the main advantage of index-based RAG?**
Full traceability—track responses to exact source data, plus faster retrieval via direct access vs sequential comparison.

**Q: Four index types and when to use each?**
1) Vector: Semantic similarity, fast
2) Tree: Hierarchical data, large datasets
3) List: Comprehensive analysis, independent scoring
4) Keyword: Term-specific searches

**Q: How does Tree Index work?**
Creates optimized hierarchy with summaries at higher levels, details at lower. LLM selects best path like multiple-choice question.

**Q: What's in dataset metadata?**
File path, name, type, size, dates, node content (ID, text, indices, relationships), document ID.

**Q: LlamaIndex + Deep Lake integration benefits?**
Automated chunking, embedding, indexing—production-ready with minimal code, customizable when needed.

**Q: How to evaluate index performance?**
Weighted average score / elapsed time + cosine similarity between input and response.

**Q: Why separate documents in data collection?**
Enable traceability—each file has metadata linking responses back to exact source.

**Q: Automated vs manual chunking?**
Automated: Variable sizes, semantic optimization. Manual: Fixed sizes, more control.

---

## Tools & Technologies

**Frameworks**: LlamaIndex, Deep Lake, OpenAI
**Models**: GPT-4o, all-MiniLM-L6-v2 (Sentence Transformers)
**Data Processing**: BeautifulSoup, Requests, Regex
**Evaluation**: scikit-learn, Sentence Transformers, NumPy, Pandas
**Storage**: Deep Lake vector stores (serverless)
**Indexing**: Vector, Tree, List, Keyword


---

## Yes/No Questions with Answers

**Q1: Do indexes increase precision and speed in retrieval-augmented generative AI?**
Yes, indexes make retrieval faster and more accurate.

**Q2: Can indexes offer traceability for RAG outputs?**
Yes, indexes allow tracing back to the exact data source.

**Q3: Is index-based search slower than vector-based search for large datasets?**
No, index-based search is faster and optimized for large datasets.

**Q4: Does LlamaIndex integrate seamlessly with Deep Lake and OpenAI?**
Yes, LlamaIndex, Deep Lake, and OpenAI work well together.

**Q5: Are tree, list, vector, and keyword indexes the only types of indexes?**
No, these are common, but other types exist as well.

**Q6: Does the keyword index rely on semantic understanding to retrieve data?**
No, it retrieves based on keywords, not semantics.

**Q7: Is LlamaIndex capable of automatically handling chunking and embedding?**
Yes, LlamaIndex automates these processes for easier data management.

**Q8: Are metadata enhancements crucial for ensuring the traceability of RAG-generated outputs?**
Yes, metadata helps trace back to the source of the generated content.

**Q9: Can real-time updates easily be applied to an index-based search system?**
Indexes often require re-indexing for updates. However, some modern indexing systems have been designed to handle real-time or near-real-time updates more efficiently.

**Q10: Is cosine similarity a metric used in this chapter to evaluate query accuracy?**
Yes, cosine similarity helps assess the relevance of query results.
