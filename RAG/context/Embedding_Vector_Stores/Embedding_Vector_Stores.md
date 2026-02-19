# Chapter 2: RAG Embedding Vector Stores with Deep Lake and OpenAI

## 1. Why Embeddings and Vector Stores?

**Core Challenge**: Generative AI models need efficient access to large datasets beyond training data.

**Embeddings vs Keywords**:
- **Keywords**: Rigid, exact matching, less storage
- **Embeddings**: Semantic meaning, context-aware, nuanced retrieval, better results

**Solution**: Transform data into high-dimensional vectors capturing semantic essence → Store in vector stores for fast retrieval

---

## 2. From Raw Data to Embeddings

### Transformation Process:
1. **Raw Data** → Text, images, audio (books, articles, blogs, pictures, songs)
2. **Cleaning** → Remove noise, prepare data
3. **Chunking** → Break into manageable pieces (e.g., 1000 characters)
4. **Embedding** → Convert to vectors using models (e.g., OpenAI `text-embedding-3-small`)
5. **Storage** → Load into vector stores (Deep Lake, Pinecone, Chroma)

### Key Advantage - Transparency:
- **Text**: Fully traceable to source data
- **Embeddings**: Directly visible and linked (vs parametric models where data is buried in weights)
- RAG = game changer for AI transparency

---

## 3. Vector Store Structure

### Dataset Components (4 Tensors):
1. **text**: Chunk content (string)
2. **embedding**: Vector representation (float32, e.g., 1536 dimensions)
3. **metadata**: Source information (JSON)
4. **id**: Unique identifier (string)

### Example Dataset:
```
Dataset: hub://user/space_exploration_v1
- text: (839, 1) str
- metadata: (839, 1) str  
- embedding: (839, 1536) float32
- id: (839, 1) str
```

---

## 4. RAG Pipeline Architecture

### Why Separate Components?

**5 Key Benefits**:
1. **Specialization**: Teams focus on expertise (data collection, embedding, generation)
2. **Scalability**: Upgrade components independently
3. **Parallel Development**: Teams work simultaneously without dependencies
4. **Maintenance**: Fix one component without disrupting others
5. **Security**: Separate authorization/access per component

### Three-Component Pipeline:

**Component #1: Data Collection & Preparation (D1-D2)**
- Team #1 responsibility
- Collect raw data from sources
- Clean and prepare (remove noise, references)
- Output: Prepared text files

**Component #2: Data Embedding & Storage (D2-D3)**
- Team #2 responsibility
- Chunk data (e.g., 1000 characters)
- Embed using OpenAI models
- Load into vector store (Deep Lake)
- Verify dataset structure

**Component #3: Augmented Generation (D4, G1-G4, E1)**
- Team #3 responsibility
- Process user input
- Query vector store
- Augment input with retrieved data
- Generate output with GPT-4o
- Evaluate results

---

## 5. Implementation Details

### Environment Setup

**Key Packages** (freeze versions to prevent conflicts):
```python
beautifulsoup4==4.12.3  # Web scraping
requests==2.31.0         # HTTP requests
deeplake==3.9.18         # Vector store
openai==1.40.3           # Embeddings & generation
sentence-transformers    # Semantic similarity
```

**Authentication**:
- OpenAI API key
- Activeloop API token
- Store securely (Google Drive, env files)

**Critical Setup** (Google Colab + Activeloop):
```python
# DNS configuration for Activeloop
with open('/etc/resolv.conf', 'w') as file:
    file.write("nameserver 8.8.8.8")
```

---

## 6. Component #1: Data Collection & Preparation

### Implementation:
```python
import requests
from bs4 import BeautifulSoup
import re

# Clean function
def clean_text(content):
    content = re.sub(r'\[\d+\]', '', content)  # Remove [1], [2]
    return content

# Fetch and clean
def fetch_and_clean(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('div', {'class': 'mw-parser-output'})
    # Remove bibliography, references
    text = content.get_text(separator=' ', strip=True)
    return clean_text(text)

# Save to file
with open('llm.txt', 'w', encoding='utf-8') as file:
    for url in urls:
        file.write(fetch_and_clean(url) + '\n')
```

### Example Data Sources (Space Exploration):
- Wikipedia articles: Space exploration, Apollo program, Hubble, Mars rover, ISS, SpaceX, Juno, Voyager, Galileo, Kepler

---

## 7. Component #2: Data Embedding & Storage

### Chunking:
```python
CHUNK_SIZE = 1000
with open('llm.txt', 'r') as f:
    text = f.read()
chunked_text = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
```

### Embedding Function:
```python
def embedding_function(texts, model="text-embedding-3-small"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    return [data.embedding for data in 
            openai.embeddings.create(input=texts, model=model).data]
```

**Model Choice**: `text-embedding-3-small`
- Efficient and fast
- Balanced detail vs computational load
- Check dimensions and pricing before use

### Vector Store Operations:
```python
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore

# Create/load vector store
vector_store_path = "hub://user/dataset_name"
vector_store = VectorStore(path=vector_store_path)

# Add data
vector_store.add(
    text=chunked_text,
    embedding_function=embedding_function,
    embedding_data=chunked_text,
    metadata=[{"source": "llm.txt"}] * len(chunked_text)
)

# Load dataset
ds = deeplake.load(vector_store_path)

# Check size
ds_size_mb = ds.size_approx() / 1048576
ds_size_gb = ds.size_approx() / 1073741824
```

---

## 8. Component #3: Augmented Generation

### Query & Retrieval:
```python
# User input
user_prompt = "Tell me about space exploration on the Moon and Mars."

# Search vector store
search_results = vector_store.search(
    embedding_data=user_prompt,
    embedding_function=embedding_function
)

# Extract top result
top_text = search_results['text'][0].strip()
top_score = search_results['score'][0]
top_metadata = search_results['metadata'][0]['source']
```

### Augmented Input:
```python
augmented_input = user_prompt + " " + top_text
```

### Generation with GPT-4o:
```python
from openai import OpenAI
client = OpenAI()

def call_gpt4_with_full_text(itext):
    text_input = '\n'.join(itext)
    prompt = f"Please summarize or elaborate: {text_input}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a space expert."},
            {"role": "assistant", "content": "You can read technical docs."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

# Generate response
gpt4_response = call_gpt4_with_full_text(augmented_input)
```

---

## 9. Evaluation Metrics

### Method 1: TF-IDF Cosine Similarity
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# User prompt vs response: 0.396 (low)
# Augmented input vs response: 0.857 (better)
```

**Limitation**: TF-IDF relies on exact vocabulary overlap, misses semantic meaning.

### Method 2: Sentence Transformers (Better)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    return cosine_similarity([embeddings1], [embeddings2])[0][0]

# Augmented input vs response: 0.739
```

**Advantage**: Captures semantic relationships, synonyms, contextual usage.

**Model Choice**: `all-MiniLM-L6-v2`
- Compact and efficient
- Strong performance for sentence embeddings
- Fast generation

---

## 10. Key Technical Insights

### Chunking Strategy:
- **Size**: 1000 characters (adjustable)
- **Purpose**: Optimize processing, embedding, loading
- **Benefit**: Easier querying, detailed embeddings
- **Advanced**: Automated seamless chunking (Chapter 7)

### Cross-Platform Challenges:
- **Version Conflicts**: Dependencies may clash
- **Solution**: Freeze package versions
- **Trade-off**: Stability vs bug fixes/updates
- **Example**: Pillow 10.x.x required for Deep Lake (Google Colab had 9.x.x)
- **Strategy**: Continual quality control process

### Vector Store Features:
- **Serverless**: Create/access via API (Activeloop)
- **Visual Interface**: Query and explore datasets online
- **Indexing**: Optimized retrieval methods
- **Scalability**: Handle high-dimensional data efficiently

---

## 11. Practical Workflow

### Team #1 (Data Collection):
1. Define data sources (URLs, databases)
2. Scrape/collect raw data
3. Clean (remove noise, references, bibliography)
4. Save prepared text → `llm.txt`
5. Verify quality

### Team #2 (Embedding & Storage):
1. Download prepared data
2. Chunk into fixed sizes
3. Embed using OpenAI model
4. Create/load vector store
5. Add embeddings with metadata
6. Verify dataset structure and size

### Team #3 (Generation):
1. Load vector store
2. Process user input
3. Embed user query
4. Search vector store (retrieve top results)
5. Augment input with retrieved text
6. Generate response with GPT-4o
7. Evaluate with similarity metrics
8. Format and display output

---

## 12. Key Formulas & Concepts

### Cosine Similarity:
```
similarity = (A · B) / (||A|| × ||B||)
```
- Range: [-1, 1]
- 1 = identical, 0 = orthogonal, -1 = opposite

### TF-IDF (Term Frequency-Inverse Document Frequency):
- Measures word importance in document relative to corpus
- Good for keyword matching
- Misses semantic relationships

### Sentence Embeddings:
- Capture semantic meaning beyond keywords
- Better for context-aware retrieval
- Models: MiniLM, BERT variants

### Dataset Size Calculation:
```python
size_mb = bytes / 1048576
size_gb = bytes / 1073741824
```

---

## 13. Best Practices

1. **Separate Pipeline Components**: Enable parallel development, easier maintenance
2. **Freeze Package Versions**: Prevent dependency conflicts
3. **Verify Data Quality**: Check at each pipeline stage
4. **Use Semantic Embeddings**: Better than keyword matching
5. **Choose Right Embedding Model**: Balance efficiency, accuracy, cost
6. **Implement Multiple Metrics**: TF-IDF + Sentence Transformers
7. **Secure API Keys**: Store safely, never hardcode
8. **Monitor Dataset Size**: Track storage costs
9. **Test Retrieval Quality**: Evaluate top results before generation
10. **Format Outputs**: Use textwrap, markdown for readability

---

## 14. Key Takeaways

1. **Embeddings capture semantic meaning** better than keywords
2. **Vector stores enable fast, context-aware retrieval** from large datasets
3. **Three-component pipeline** allows parallel team development
4. **Chunking optimizes** processing and retrieval
5. **OpenAI text-embedding-3-small** balances efficiency and quality
6. **Deep Lake provides transparency** (visible text + embeddings)
7. **Augmented input significantly improves** generation quality
8. **Sentence Transformers outperform TF-IDF** for semantic similarity
9. **Version management critical** for cross-platform stability
10. **RAG transparency** enables traceability (vs black-box parametric models)

---

## Interview-Ready Q&A

**Q: Why use embeddings over keywords?**
Embeddings capture semantic meaning and context, enabling nuanced retrieval vs rigid keyword matching.

**Q: What are the 3 RAG pipeline components?**
1) Data collection & preparation, 2) Data embedding & storage, 3) Augmented generation.

**Q: What's in a vector store dataset?**
4 tensors: text (content), embedding (vectors), metadata (source), id (unique identifier).

**Q: Why separate pipeline components?**
Specialization, scalability, parallel development, independent maintenance, security.

**Q: TF-IDF vs Sentence Transformers?**
TF-IDF: keyword overlap, exact matching. Sentence Transformers: semantic relationships, context-aware.

**Q: What is chunking and why?**
Breaking text into fixed sizes (e.g., 1000 chars) for optimized processing, embedding, and retrieval.

**Q: How to evaluate RAG output?**
Cosine similarity between augmented input and generated response using embeddings.

**Q: Key challenge in RAG implementation?**
Cross-platform package dependencies and version conflicts—freeze versions for stability.

---

## Tools & Technologies

**Vector Stores**: Activeloop Deep Lake, Pinecone, Chroma
**Embedding Models**: OpenAI text-embedding-3-small, Sentence Transformers (all-MiniLM-L6-v2)
**Generation Models**: OpenAI GPT-4o
**Data Processing**: BeautifulSoup, Requests, Regex
**Evaluation**: scikit-learn (TF-IDF, cosine similarity), Sentence Transformers
**Environment**: Python, Google Colab, pip

---

## Yes/No Questions with Answers

**Q1: Do embeddings convert text into high-dimensional vectors for faster retrieval in RAG?**
Yes, embeddings create vectors that capture the semantic meaning of text.

**Q2: Are keyword searches more effective than embeddings in retrieving detailed semantic content?**
No, embeddings are more context-aware than rigid keyword searches.

**Q3: Is it recommended to separate RAG pipelines into independent components?**
Yes, this allows parallel development and easier maintenance.

**Q4: Does the RAG pipeline consist of only two main components?**
No, the pipeline consists of three components – data collection, embedding, and generation.

**Q5: Can Activeloop Deep Lake handle both embedding and vector storage?**
Yes, it stores embeddings efficiently for quick retrieval.

**Q6: Is the text-embedding-3-small model from OpenAI used to generate embeddings in this chapter?**
Yes, this model is chosen for its balance between detail and computational efficiency.

**Q7: Are data embeddings visible and directly traceable in an RAG-driven system?**
Yes, unlike parametric models, embeddings in RAG are traceable to the source.

**Q8: Can a RAG pipeline run smoothly without splitting into separate components?**
Splitting an RAG pipeline into components improves specialization, scalability, and security, which helps a system run smoothly. Simpler RAG systems may still function effectively without explicit component separation, although it may not be the optimal setup.

**Q9: Is chunking large texts into smaller parts necessary for embedding and storage?**
Yes, chunking helps optimize embedding and improves the efficiency of queries.

**Q10: Are cosine similarity metrics used to evaluate the relevance of retrieved information?**
Yes, cosine similarity helps measure how closely retrieved data matches the query.
