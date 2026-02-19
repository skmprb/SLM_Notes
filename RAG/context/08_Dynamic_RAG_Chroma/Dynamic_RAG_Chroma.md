# Chapter 8: Dynamic RAG with Chroma and Hugging Face Llama

## 1. Core Concept: Dynamic RAG for Real-Time Decision Making

**Definition**: Dynamic RAG = Temporary data collections created daily for specific meetings, avoiding long-term storage overhead while providing real-time, relevant information retrieval.

**Key Principle**: Data permanence is shifting - not all information needs indefinite storage. Focus on precise, pertinent data for specific needs at specific times (daily briefings, critical meetings).

**Use Case**: Daily meeting preparation where teams need rapid access to fresh datasets (10,000+ documents) without web searches or permanent storage costs.

## 2. Dynamic RAG Architecture Components

**Four-Stage Pipeline**:

1. **Temporary Chroma Collection Creation (D1, D2, D3, E2)**
   - Created each morning for that day's meeting only
   - No post-meeting persistence
   - Prevents long-term data clutter

2. **Embedding Relevant Data (D1, D2, D3, E2)**
   - Embeds critical data: customer support interactions, medical reports, scientific facts
   - Tailored to meeting agenda
   - Can include human feedback from documents and other generative AI systems

3. **Pre-Meeting Data Validation (D4)**
   - Batch queries run before meeting starts
   - Ensures data accuracy and alignment with objectives
   - Facilitates smooth, informed discussion

4. **Real-Time Query Handling (G1, G2, G3, G4)**
   - Handles spontaneous participant queries during meeting
   - Single question triggers specific information retrieval
   - Augments Llama input to generate dynamic flashcards
   - Provides concise, accurate responses

**Ecosystem Label**: D-G-E-T (Data-Generation-Embedding-Transformation)

## 3. Architecture Benefits

**Efficiency & Cost-Effectiveness**:
- Chroma for temporary storage = lightweight system
- Llama for response generation = no ongoing storage costs
- Ideal for frequently refreshed data without long-term storage needs
- Convincing for decision-makers wanting lean systems

**Flexibility**:
- Ephemeral nature allows daily new data integration
- Most up-to-date information always available
- Valuable in fast-paced environments with rapid information changes

**Scalability**:
- Not just increasing data volumes
- Ability to apply framework across wide range of domains and situations
- Adaptable to various datasets that can be embedded and queried effectively

**User-Friendliness**:
- Straightforward design accessible to non-technical users
- Reliable answers quickly
- Enhances user engagement and satisfaction with cost-effective, transparent, lightweight AI

## 4. Application Domains

**Customer Support**: Daily updated FAQs accessed real-time for quick customer responses

**Healthcare**: Medical teams use latest research and patient data for complex health questions during meetings

**Finance**: Financial analysts query latest market data for informed investment decisions and strategies

**Education**: Educators access latest educational resources and research to enhance learning

**Tech Support**: IT teams use updated technical documentation to solve issues and guide users

**Sales & Marketing**: Quick access to latest product information and market trends for client queries and strategizing

**Hard Science**: Daily briefings with scientific datasets (physics, chemistry, biology) for experimental decisions

## 5. Environment Setup: Open-Source Stack

**Hugging Face Installation**:
```python
# Token storage (Google Drive or manual entry)
from google.colab import drive
drive.mount('/content/drive')
f = open("drive/MyDrive/files/hf_token.txt", "r")
access_token = f.readline().strip()
os.environ['HF_TOKEN'] = access_token

# Install packages
!pip install datasets==2.20.0
!pip install transformers==4.41.2
!pip install accelerate==0.31.0  # GPU acceleration, mixed precision
```

**Chroma Installation**:
```python
!pip install chromadb==0.5.3
# Includes ONNX Runtime (onnxruntime-1.18.0)
```

**spaCy Installation** (for similarity scoring):
```python
!python -m spacy download en_core_web_md
```

**Key Dependencies**:
- **ONNX Runtime**: Cross-platform ML model accelerator, fully integrated with Chroma
- **accelerate**: PyTorch GPU acceleration, mixed precision, accelerated processing
- **spaCy en_core_web_md**: Medium-sized English model for NLP tasks, balanced speed/accuracy

## 6. Llama 2 Configuration

**Model Initialization**:
```python
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,  # Half-precision (16-bit) for faster, lighter processing
    device_map="auto"  # Auto-distribute across available GPUs
)
```

**Pipeline Parameters**:
- **torch.float16**: Half-precision (16 bits vs 32 bits) = reduced memory + faster computation
- **device_map="auto"**: Automatically determines best device (CPU/GPU/multi-GPU), distributes layers efficiently

## 7. SciQ Dataset: Hard Science Questions

**Dataset Overview**:
- **Source**: Welbl, Liu, Gardner (2017) - crowdsourced science questions
- **Size**: 13,679 multiple-choice questions (10,481 with support after filtering)
- **Domains**: Physics, chemistry, biology
- **Purpose**: Train NLP models for science exams

**Dataset Structure**:
- **question**: Query text
- **correct_answer**: Validated answer
- **support**: Explanatory content
- **distractor columns**: Wrong answers (dropped in processing)

**Data Preparation**:
```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("sciq", split="train")
filtered_dataset = dataset.filter(lambda x: x["support"] != "" and x["correct_answer"] != "")
df = pd.DataFrame(filtered_dataset)

# Drop distractors
df.drop(columns=['distractor3', 'distractor1', 'distractor2'], inplace=True)

# Merge correct_answer + support into completion
df['completion'] = df['correct_answer'] + " because " + df['support']
df.dropna(subset=['completion'], inplace=True)
```

**Final Shape**: (10481, 4) - question, correct_answer, support, completion

## 8. Chroma Collection Management

**Collection Creation**:
```python
import chromadb

client = chromadb.Client()
collection_name = "sciq_supports6"

# Check if collection exists
collections = client.list_collections()
collection_exists = any(collection.name == collection_name for collection in collections)

# Create if doesn't exist
if not collection_exists:
    collection = client.create_collection(collection_name)
```

**Collection Dictionary Structure**:
- **ids**: Unique identifiers for each item
- **embeddings**: Embedded vectors of documents
- **documents**: Completion column (merged correct_answer + support)
- **metadatas**: Additional data (type: "completion")
- **uris**: Resource identifiers
- **data**: Stored content
- **included**: Fields included in response

## 9. Embedding Model: all-MiniLM-L6-v2

**Model Selection**:
```python
model_name = "all-MiniLM-L6-v2"  # Default Chroma model
```

**Model Architecture** (Wang et al., 2021):
- **Compression Method**: Knowledge distillation through teacher-student models
- **Teacher Model**: Large, complex models (BERT, RoBERTa, XLM-R) with high accuracy
- **Student Model**: Smaller all-MiniLM-L6-v2 mimics teacher with fewer parameters
- **Distillation Focus**: Self-attention relationships between transformer components
- **Flexibility**: Variable attention heads between teacher and student

**Key Advantages**:
- Accelerates embedding and querying process
- Significantly fewer parameters than teacher models
- Lower computational expense
- Optimal for dynamic RAG scenarios

**Embedding Dimensions**: 384-dimensional space (vs 1,526 for OpenAI text-embedding-ada-002)

**Dense vs Sparse Vectors**:
- **Dense Vectors** (all-MiniLM-L6-v2): Use all dimensions to encode information, capture nuanced semantic relationships
- **Sparse Vectors** (BoW): Don't capture word order or context, less effective for LLM training

## 10. Embedding and Upserting Process

**Data Embedding**:
```python
ldf = len(df)
nb = ldf  # Number of questions to embed
start_time = time.time()

# Convert to list
completion_list = df["completion"][:nb].astype(str).tolist()

# Upsert to collection (only if doesn't exist)
if not collection_exists:
    collection.add(
        ids=[str(i) for i in range(0, nb)],
        documents=completion_list,
        metadatas=[{"type": "completion"} for _ in range(0, nb)]
    )

response_time = time.time() - start_time
print(f"Response Time: {response_time:.2f} seconds")
```

**Performance**: ~234 seconds for 10,000+ documents (with GPU)

**Embedding Verification**:
```python
result = collection.get(include=['embeddings'])
first_embedding = result['embeddings'][0]
embedding_length = len(first_embedding)  # Output: 384
```

**ONNX Integration**: Model loaded from `/root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz`

## 11. Batch Query Validation

**Query Execution**:
```python
start_time = time.time()

results = collection.query(
    query_texts=df["question"][:nb],
    n_results=1  # Most relevant document per question
)

response_time = time.time() - start_time  # ~199 seconds for 10,000+ queries
```

**Similarity Scoring with spaCy**:
```python
import spacy
import numpy as np

nlp = spacy.load('en_core_web_md')

def simple_text_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    vector1 = doc1.vector
    vector2 = doc2.vector
    
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0.0
    
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity
```

**Accuracy Calculation**:
```python
acc_counter = 0
threshold = 0.7

for i, q in enumerate(df['question'][:nb]):
    original_completion = df['completion'][i]
    retrieved_document = results['documents'][i][0]
    similarity_score = simple_text_similarity(original_completion, retrieved_document)
    
    if similarity_score > 0.7:
        acc_counter += 1

acc = acc_counter / nb
```

**Results**: Overall similarity score = 1.00 (all 10,481 queries returned relevant results)

## 12. Real-Time Query During Meetings

**Prompt Variants**:
```python
# Initial question (exact dataset text)
prompt = "Millions of years ago, plants used energy from the sun to form what?"

# Variant 1 (similar - likely user phrasing)
# prompt = "Eons ago, plants used energy from the sun to form what?"

# Variant 2 (divergent - challenging)
# prompt = "Eons ago, plants used sun energy to form what?"
```

**Query Execution**:
```python
import time
import textwrap

start_time = time.time()

results = collection.query(
    query_texts=[prompt],
    n_results=1
)

response_time = time.time() - start_time  # ~0.03 seconds

if results['documents'] and len(results['documents'][0]) > 0:
    wrapped_question = textwrap.fill(prompt, width=70)
    wrapped_document = textwrap.fill(results['documents'][0][0], width=70)
    print(f"Question: {wrapped_question}")
    print(f"Retrieved document: {wrapped_document}")
```

**Performance**: 0.03 seconds response time for single query

**Human Control Requirement**: More divergent queries (variant 2) = more challenging for system stability. Careful, continual improvements needed.

## 13. RAG with Llama 2 Generation

**Llama Configuration Function**:
```python
def LLaMA2(prompt):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        temperature=0.5,
        repetition_penalty=2.0,
        truncation=True
    )
    return sequences
```

**Parameter Definitions**:
- **do_sample=True**: Enables stochastic sampling for varied outputs
- **top_k=10**: Limits to top 10 most likely next tokens
- **num_return_sequences=1**: Returns one sequence per prompt
- **eos_token_id**: End-of-sequence token marker
- **max_new_tokens=100**: Maximum 100 tokens beyond input prompt
- **temperature=0.5**: Less random, more focused responses with some diversity
- **repetition_penalty=2.0**: Discourages token repetition for diverse text
- **truncation=True**: Ensures output doesn't exceed max_new_tokens

**Augmented Prompt Creation**:
```python
iprompt = 'Read the following input and write a summary for beginners: '
lprompt = iprompt + " " + results['documents'][0][0]

start_time = time.time()
response = LLaMA2(lprompt)

for seq in response:
    generated_part = seq['generated_text'].replace(iprompt, '').strip()

response_time = time.time() - start_time  # ~5.91 seconds
```

**Performance**: 5.91 seconds for Llama 2 completion generation

**Alternative**: Can replace with GPT-4o for superior output quality if needed (pragmatic approach - performance over pure open-source)

## 14. Collection Lifecycle Management

**Session Time Tracking**:
```python
# At session start (after environment installation)
session_start_time = time.time()

# At session end
end_time = time.time() - session_start_time
print(f"Session preparation time: {end_time:.2f} seconds")
```

**Total Session Time**: ~780 seconds (< 15 minutes) for full run with no human intervention

**Collection Deletion**:
```python
# Manual deletion
client.delete_collection(collection_name)

# Or close session to delete temporary collection

# Verify deletion
collections = client.list_collections()
collection_exists = any(collection.name == collection_name for collection in collections)
print("Collection exists:", collection_exists)  # False after deletion
```

**Temporary Nature**: Collection deleted after session ends, no long-term storage costs or space usage

## 15. Performance Metrics Summary

**Embedding Performance**:
- Documents: 10,481
- Embedding time: ~234 seconds
- Model: all-MiniLM-L6-v2 (384 dimensions)

**Query Performance**:
- Batch queries: 10,481 questions
- Query time: ~199 seconds
- Single query: ~0.03 seconds
- Overall similarity score: 1.00 (100% relevant results)

**Generation Performance**:
- Llama 2 response time: ~5.91 seconds
- Model: meta-llama/Llama-2-7b-chat-hf
- Max tokens: 100

**Total Session Time**: ~780 seconds (13 minutes) - fits dynamic RAG constraints with room for multiple tweaking runs before meetings

**Accuracy Threshold**: Similarity score > 0.7 considered accurate

## Interview-Ready Q&A

**Q1: What is Dynamic RAG?**
A: Temporary data collections created daily for specific meetings, avoiding long-term storage while providing real-time relevant information retrieval.

**Q2: What are the four stages of Dynamic RAG architecture?**
A: (1) Temporary Chroma collection creation, (2) Embedding relevant data, (3) Pre-meeting data validation, (4) Real-time query handling.

**Q3: Why use all-MiniLM-L6-v2 over larger models?**
A: Knowledge-distilled student model with 384 dimensions, fewer parameters, faster embedding/querying, optimal for dynamic RAG scenarios.

**Q4: What is the difference between dense and sparse vectors?**
A: Dense vectors use all dimensions to encode information with nuanced semantic relationships; sparse vectors (like BoW) don't capture word order or context.

**Q5: How does knowledge distillation work in MiniLM?**
A: Teacher model (BERT/RoBERTa) with high accuracy transfers knowledge to smaller student model (all-MiniLM-L6-v2) that mimics performance with fewer parameters.

**Q6: What is ONNX Runtime's role?**
A: Cross-platform ML model accelerator providing flexible interface for hardware-specific optimization (CPUs, GPUs, accelerators), fully integrated with Chroma.

**Q7: What are Llama 2 pipeline key parameters?**
A: torch.float16 (half-precision for speed), device_map="auto" (auto GPU distribution), temperature (randomness control), repetition_penalty (diversity).

**Q8: How is accuracy measured in batch validation?**
A: spaCy cosine similarity between original completion and retrieved document; threshold > 0.7 considered accurate.

**Q9: What is the SciQ dataset?**
A: 13,679 crowdsourced multiple-choice science questions (physics, chemistry, biology) with validated answers and support content.

**Q10: Why is Dynamic RAG cost-effective?**
A: Temporary local collections (no storage costs), open-source tools (Chroma, Llama), lightweight models, no long-term data management overhead.

**Q11: What happens to collections after meetings?**
A: Deleted automatically when session closes or manually with client.delete_collection() - no persistent storage.

**Q12: Can Dynamic RAG use proprietary models?**
A: Yes - pragmatic approach prioritizes performance; can replace Llama 2 with GPT-4o if superior output needed.

**Q13: What are Dynamic RAG application domains?**
A: Customer support, healthcare, finance, education, tech support, sales/marketing, hard science research.

**Q14: How fast is single query retrieval?**
A: ~0.03 seconds for semantic vector search in 10,000+ document collection.

**Q15: What is the total preparation time?**
A: ~13 minutes (780 seconds) for full run, leaving time for multiple tweaking runs before meetings.

## Tools & Technologies

**Vector Database**: Chroma (open-source, AI-native, local storage)

**Embedding Model**: all-MiniLM-L6-v2 (384 dimensions, knowledge-distilled)

**LLM**: meta-llama/Llama-2-7b-chat-hf (Hugging Face)

**Dataset**: SciQ (13,679 science questions, Hugging Face)

**NLP Library**: spaCy en_core_web_md (similarity scoring)

**ML Framework**: PyTorch (torch.float16 half-precision)

**Acceleration**: Hugging Face accelerate, ONNX Runtime

**Data Processing**: pandas, datasets (Hugging Face)

**Tokenization**: Hugging Face AutoTokenizer

**Pipeline**: Hugging Face transformers.pipeline

**Environment**: Google Colab (GPU recommended)

**Alternative LLM**: OpenAI GPT-4o (for superior output quality)

**Compression Standard**: ONNX (Open Neural Network Exchange)

**Teacher Models**: BERT, RoBERTa, XLM-R (for MiniLM distillation)

**Time Tracking**: Python time module

**Text Formatting**: textwrap module


---

## Yes/No Questions with Answers

**Q1: Does the script ensure that the Hugging Face API token is never hardcoded directly into the notebook for security reasons?**
Yes, the script provides methods to either use Google Drive or manual input for API token handling, thus avoiding hardcoding.

**Q2: In the chapter's program, is the accelerate library used to facilitate the deployment of machine learning models on cloud-based platforms?**
No, the accelerate library is used to run models on local resources such as multiple GPUs, TPUs, and CPUs, not specifically cloud platforms.

**Q3: Is user authentication, apart from the API token, required to access the Chroma database in this script?**
No, the script does not detail additional authentication mechanisms beyond using an API token to access Chroma.

**Q4: Does the notebook use Chroma for temporary storage of vectors during the dynamic retrieval process?**
Yes, the script employs Chroma for storing vectors temporarily to enhance the efficiency of data retrieval.

**Q5: Is the notebook configured to use real-time acceleration of queries through GPU optimization?**
Yes, the accelerate library is used to ensure that the notebook can leverage GPU resources for optimizing queries, which is particularly useful in dynamic retrieval settings.

**Q6: Can this notebook's session time measurements help in optimizing the dynamic RAG process?**
Yes, by measuring session time, the notebook provides insights that can be used to optimize the dynamic RAG process, ensuring efficient runtime performance.

**Q7: Does the script demonstrate Chroma's capability to integrate with machine learning models for enhanced retrieval performance?**
Yes, the integration of Chroma with the Llama model in this script highlights its capability to enhance retrieval performance by using advanced machine learning techniques.

**Q8: Does the script include functionality to adjust the parameters of the Chroma database based on session performance metrics?**
Yes, the notebook potentially allows adjustments to be made based on performance metrics, such as session time, which can influence how the notebook is built and adjust the process, depending on the project.
