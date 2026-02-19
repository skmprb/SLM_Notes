# Chapter 7: Building Scalable Knowledge-Graph-Based RAG with Wikipedia API and LlamaIndex

## 1. Why Knowledge Graphs?

**Core Challenge**: Scaled datasets become difficult to manage—visualizing relationships is key

**Knowledge Graph Solution**:
- Visual representation of data relationships
- Nodes = entities (e.g., marketing strategies)
- Edges = connections between entities
- Helps understand how data fits together

**Use Case**: Marketing knowledge base for upskilling students

---

## 2. Three-Pipeline Architecture

### Pipeline 1: Data Collection & Preparation
- Wikipedia API retrieves URLs and metadata
- Automated topic-based collection
- Citations and summaries extracted
- Data cleaned and aggregated

### Pipeline 2: Deep Lake Vector Store
- Automated chunking (seamless)
- OpenAI embedding (built-in)
- Upsert to vector store
- Transparent data structure

### Pipeline 3: Knowledge Graph Index-Based RAG
- LlamaIndex builds graph index automatically
- Visual graph display
- Query with LLM (OpenAI)
- Re-ranking and metrics

---

## 3. Knowledge Graph Fundamentals

### Graph Components:
- **Nodes (Vertices)**: Entities/concepts
- **Edges (Arcs)**: Relationships/connections
- **Directed**: Edges have direction (A → B)
- **Triplets**: Subject-Predicate-Object relationships

### Example:
```
Pairs: [('a', 'b'), ('b', 'e'), ('e', 'm')]
Friends: {('a', 'b'), ('b', 'e')}
Result: Solid lines = friends, dashed = not friends
```

### Layouts:
- **Spring**: Nodes attracted by edges, repel each other
- **Hierarchical**: Tree-like structure
- **Circular**: Nodes arranged in circle

---

## 4. Pipeline 1: Wikipedia API Implementation

### Install & Setup:
```python
!pip install Wikipedia-API==0.6.0
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='Knowledge/1.0 (your_email@example.com)'
)
```

### Retrieve Page Summary:
```python
topic = "Marketing"
page = wiki.page(topic)

if page.exists():
    summary = page.summary
    print(f"Tokens: {nb_tokens(summary)}")
    print(summary)
    print(page.fullurl)
```

### Extract Links & Summaries:
```python
links = page.links
urls = []
counter = 0
maxl = 100  # Max links to retrieve

for link in links:
    try:
        counter += 1
        print(f"Link {counter}: {link}")
        summary = wiki.page(link).summary
        url = wiki.page(link).fullurl
        urls.append(url)
        print(f"Summary: {summary}")
        if counter >= maxl:
            break
    except:
        pass
```

### Generate Citations File:
```python
from datetime import datetime

fname = f"{filename}_citations.txt"
with open(fname, "w") as file:
    file.write(f"Citation. In Wikipedia. Retrieved {datetime.now()}\n")
    file.write(f"Root page: {page.fullurl}\n")
    for link in links:
        # Write citation for each link
```

### Save URLs:
```python
ufname = f"{filename}_urls.txt"
with open(ufname, 'w') as file:
    for url in urls:
        file.write(url + '\n')
```

---

## 5. Pipeline 2: Deep Lake Vector Store

### Configuration:
```python
graph_name = "Marketing"
db = "hub://user/marketing01"
vector_store_path = db
dataset_path = db
pop_vs = True  # Activate upserting
ow = True      # Overwrite existing dataset
```

### Load URLs & Fetch Content:
```python
# Download URL list
directory = "Chapter07/citations"
file_name = f"{graph_name}_urls.txt"
download(directory, file_name)

# Read URLs
with open(file_name, 'r') as f:
    urls = f.readlines()

# Fetch and clean content (same as Chapter 3)
```

### Automated Chunking & Embedding:
- Deep Lake handles chunking automatically
- OpenAI embedding built-in
- No manual configuration needed

### Dataset Structure:
```
ID: Unique identifier
Metadata:
  - file_path
  - file_name
  - file_type
  - file_size
  - creation_date
  - last_modified_date
Text: Document content
Embedding: Vector representation
```

---

## 6. Pipeline 3: Knowledge Graph Index

### Create Graph Index:
```python
from llama_index.core import KnowledgeGraphIndex
import time

start_time = time.time()

graph_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    include_embeddings=True
)

elapsed_time = time.time() - start_time
print(f"Index creation time: {elapsed_time:.4f} seconds")
# Output: ~371.98 seconds
```

### Configure Query Engine:
```python
k = 3           # Top-k similar results
temp = 0.1      # Temperature (low = precise)
mt = 1024       # Max output tokens

graph_query_engine = graph_index.as_query_engine(
    similarity_top_k=k,
    temperature=temp,
    num_output=mt
)
```

---

## 7. Display Knowledge Graph

### Generate Interactive Graph:
```python
from pyvis.network import Network

g = graph_index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)

# Customize appearance
for node in net.nodes:
    node['color'] = 'lightgray'
    node['size'] = 10

for edge in net.edges:
    edge['color'] = 'black'
    edge['width'] = 1

# Save and display
fgraph = f"Knowledge_graph_{graph_name}.html"
net.write_html(fgraph)

# Display in notebook
from IPython.display import HTML
with open(fgraph, 'r') as file:
    html_content = file.read()
display(HTML(html_content))
```

---

## 8. Query Knowledge Graph

### Execute Query:
```python
def execute_query(user_input):
    start_time = time.time()
    response = graph_query_engine.query(user_input)
    elapsed_time = time.time() - start_time
    print(f"Query execution time: {elapsed_time:.4f} seconds")
    return response

user_query = "What is the primary goal of marketing for the consumer market?"
response = execute_query(user_query)
print(response)
```

**Example Output**:
```
Query execution time: 2.4789 seconds
The primary goal of marketing for the consumer market is to effectively 
satisfy and retain customers...
```

---

## 9. Re-Ranking Implementation

### Similarity Function:
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    return cosine_similarity([embeddings1], [embeddings2])[0][0]
```

### Re-Rank Results:
```python
user_query = "Which experts are often associated with marketing theory?"
response = execute_query(user_query)

best_score = 0
best_text = ""

for idx, node_with_score in enumerate(response.source_nodes):
    text1 = user_query
    text2 = node_with_score.node.get_text()
    
    similarity_score = calculate_cosine_similarity_with_embeddings(text1, text2)
    print(f"Rank {idx}: Score {similarity_score}")
    
    if similarity_score > best_score:
        best_score = similarity_score
        best_text = text2

print(f"Best Rank: {best_rank}")
print(f"Best Score: {best_score}")
print(best_text)
```

**Result**: Re-ranked response may provide more specific information (e.g., expert names)

---

## 10. Evaluation Metrics

### Run Multiple Examples:
```python
scores = []      # Cosine similarity scores
rscores = []     # Human feedback scores

# Example 1
user_query = "Which experts are often associated with marketing theory?"
response = execute_query(user_query)

text1 = str(response)
text2 = user_query
similarity_score = calculate_cosine_similarity_with_embeddings(text1, text2)
scores.append(similarity_score)

human_feedback = 0.75
rscores.append(human_feedback)

# Repeat for 10 examples...
```

### Calculate Statistics:
```python
import numpy as np

mean_score = np.mean(scores)
median_score = np.median(scores)
std_deviation = np.std(scores)
variance = np.var(scores)
min_score = np.min(scores)
max_score = np.max(scores)
range_score = max_score - min_score
percentile_25 = np.percentile(scores, 25)
percentile_75 = np.percentile(scores, 75)
iqr = percentile_75 - percentile_25
```

### Example Results (10 samples):
```
Cosine Similarity Scores: [0.809, 0.720, 0.760, 0.851, 0.546, 0.696, ...]
Human Feedback Scores: [0.75, 0.5, 0.8, 0.9, 0.65, 0.8, 0.9, 0.2, 0.2, 0.9]

Mean: 0.68
Median: 0.71
Standard Deviation: 0.15
Variance: 0.02
Minimum: 0.45
Maximum: 0.90
Range: 0.46
25th Percentile (Q1): 0.56
75th Percentile (Q3): 0.80
Interquartile Range (IQR): 0.24
```

---

## 11. Metric Interpretations

### Central Tendency:
- **Mean (0.68)**: Average score, typical performance
- **Median (0.71)**: Middle value, less affected by outliers

### Variability:
- **Standard Deviation (0.15)**: Moderate spread around mean
- **Variance (0.02)**: Low variance indicates consistency
- **Range (0.46)**: Span from lowest to highest
- **IQR (0.24)**: Middle 50% spread

### Extremes:
- **Minimum (0.45)**: Lowest performance
- **Maximum (0.90)**: Best performance

### Distribution:
- **Q1 (0.56)**: 25% of scores below this
- **Q3 (0.80)**: 75% of scores below this

---

## 12. Key Technical Insights

### Automated Chunking:
- Deep Lake handles automatically
- No manual chunk size configuration
- Optimized for semantic search

### Knowledge Graph Benefits:
- Visual relationship mapping
- Semantic connections visible
- Better understanding of data structure
- Improved query relevance

### LlamaIndex Integration:
- Automatic graph index creation
- Built-in LLM functionality (OpenAI)
- Seamless embedding and querying
- No manual prompt engineering needed

### Re-Ranking Value:
- May surface more specific information
- Provides alternative perspectives
- User preference varies (short vs long answers)
- Complements initial LLM response

---

## 13. Best Practices

1. **Topic Selection**: Choose clear, well-defined Wikipedia topics
2. **Citation Ethics**: Always save and reference Wikipedia citations
3. **URL Management**: Store URLs for traceability
4. **Automated Processing**: Leverage built-in chunking/embedding
5. **Graph Visualization**: Use interactive HTML for exploration
6. **Multiple Metrics**: Combine cosine similarity + human feedback
7. **Re-Ranking**: Test alternative result orderings
8. **Scalability**: Process one topic at a time for memory efficiency
9. **Performance Tracking**: Measure time at each stage
10. **Human Validation**: Essential for complex responses

---

## 14. Workflow Summary

**Step 1**: Select Wikipedia topic (e.g., "Marketing")
**Step 2**: Retrieve page summary and verify
**Step 3**: Extract links (up to maxl limit)
**Step 4**: Generate citations file
**Step 5**: Save URLs to file
**Step 6**: Fetch and clean URL content
**Step 7**: Upsert to Deep Lake vector store
**Step 8**: Create knowledge graph index (~372 sec)
**Step 9**: Configure query engine (k, temp, mt)
**Step 10**: Display interactive graph (HTML)
**Step 11**: Execute queries (~2.5 sec)
**Step 12**: Re-rank results if needed
**Step 13**: Calculate metrics (10+ examples)
**Step 14**: Evaluate with human feedback

---

## 15. Key Takeaways

1. **Knowledge graphs visualize** data relationships effectively
2. **Wikipedia API** automates large-scale data collection
3. **Three pipelines** enable modular, scalable architecture
4. **Deep Lake** handles chunking/embedding seamlessly
5. **LlamaIndex** auto-generates knowledge graph index
6. **Interactive graphs** (PyVis) aid exploration
7. **Re-ranking** can improve result relevance
8. **Multiple metrics** provide comprehensive evaluation
9. **Human feedback** essential for accuracy assessment
10. **Scalable design** processes topics incrementally

---

## Interview-Ready Q&A

**Q: What is a knowledge graph?**
Collection of nodes (entities) connected by edges (relationships), visually mapping data connections.

**Q: Why use Wikipedia API?**
Automated retrieval of structured data with metadata, citations, and URLs for ethical sourcing.

**Q: Three pipeline components?**
1) Data collection (Wikipedia API), 2) Vector store (Deep Lake), 3) Graph index RAG (LlamaIndex).

**Q: How does LlamaIndex create graph?**
Automatically extracts triplets (subject-predicate-object) from documents, builds index with embeddings.

**Q: What's max_triplets_per_chunk?**
Limits triplets per chunk (e.g., 2) to optimize memory and processing time.

**Q: Re-ranking purpose?**
Reorder results by recalculating similarity, may surface more specific/relevant information.

**Q: Key metrics for evaluation?**
Mean, median (central tendency), std dev, variance (variability), min/max (extremes), percentiles (distribution).

**Q: Why human feedback scores?**
Mathematical metrics don't capture accuracy—human judgment essential for complex responses.

**Q: Graph index creation time?**
~372 seconds for marketing dataset, varies by size and complexity.

**Q: Query performance?**
~2.5 seconds per query, constant regardless of graph size.

---

## Tools & Technologies

**Data Collection**: Wikipedia API (wikipediaapi)
**Vector Store**: Deep Lake (Activeloop)
**Graph Framework**: LlamaIndex (KnowledgeGraphIndex)
**Visualization**: PyVis (Network), NetworkX
**Embedding**: OpenAI (text-embedding-3-small)
**Generation**: OpenAI (GPT-4o via LlamaIndex)
**Evaluation**: Sentence Transformers (all-MiniLM-L6-v2), scikit-learn
**Data Processing**: pandas, NumPy, NLTK (tokenization)


---

## Yes/No Questions with Answers

**Q1: Does the chapter focus on building a scalable knowledge graph-based RAG system using the Wikipedia API and LlamaIndex?**
Yes, it details creating a knowledge graph-based RAG system using these tools.

**Q2: Is the primary use case discussed in the chapter related to healthcare data management?**
No, the primary use case discussed is related to marketing and other domains.

**Q3: Does Pipeline 1 involve collecting and preparing documents from Wikipedia using an API?**
Yes, Pipeline 1 automates document collection and preparation using the Wikipedia API.

**Q4: Is Deep Lake used to create a relational database in Pipeline 2?**
No, Deep Lake is used to create and populate a vector store, not a relational database.

**Q5: Does Pipeline 3 utilize LlamaIndex to build a knowledge graph index?**
Yes, Pipeline 3 uses LlamaIndex to build a knowledge graph index automatically.

**Q6: Is the system designed to only handle a single specific topic, such as marketing, without flexibility?**
No, the system is flexible and can handle various topics beyond marketing.

**Q7: Does the chapter describe how to retrieve URLs and metadata from Wikipedia pages?**
Yes, it explains the process of retrieving URLs and metadata using the Wikipedia API.

**Q8: Is a GPU required to run the pipelines described in the chapter?**
No, the pipelines are designed to run efficiently using only a CPU.

**Q9: Does the knowledge graph index visually map out relationships between pieces of data?**
Yes, the knowledge graph index visually displays semantic relationships in the data.

**Q10: Is human intervention required at every step to query the knowledge graph index?**
No, querying the knowledge graph index is automated, with minimal human intervention needed.
