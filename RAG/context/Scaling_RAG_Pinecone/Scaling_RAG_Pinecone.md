# Chapter 6: Scaling RAG Bank Customer Data with Pinecone

## 1. Why Scaling Matters

**Core Challenge**: Scaling isn't just adding more data—it fundamentally changes application behavior

**Three Key Aspects**:
1. **Right Amount**: Finding optimal data volume, not maximum
2. **New Features**: Handling bigger loads requires different capabilities
3. **Cost & Speed**: Performance constraints when scaling

**Use Case**: Bank customer churn reduction through personalized recommendations

---

## 2. Scaling Architecture: Three Pipelines

### Pipeline 1: Data Collection & Preparation
- Download Kaggle Bank Customer Churn dataset (10,000 records)
- Perform Exploratory Data Analysis (EDA)
- Apply k-means clustering
- Identify churn patterns

### Pipeline 2: Scaling Pinecone Index
- Chunk and embed data
- Upsert to Pinecone vector store
- Scale to 1,000,000+ vectors
- Optimize for speed and cost

### Pipeline 3: RAG Generative AI
- Query Pinecone with target vectors
- Augment user input
- Generate recommendations with GPT-4o
- Reduce customer churn

---

## 3. Technology Stack

**Pinecone Benefits**:
- Serverless architecture (no server management)
- Pay-as-you-go pricing
- Scalable infrastructure (AWS us-east-1)
- Consistent query performance at scale

**OpenAI Integration**:
- text-embedding-3-small for embeddings
- GPT-4o for generation
- Lightweight development stack

**Key Advantage**: Minimal external libraries, simplified pipeline

---

## 4. Dataset: Bank Customer Churn

**Source**: Kaggle (10,000 customer records)

**Key Columns** (after optimization):
- CustomerId: Unique identifier
- CreditScore: Credit worthiness
- Age: Customer age
- Tenure: Years with bank
- Balance: Account balance
- NumOfProducts: Products purchased
- HasCrCard: Credit card ownership
- IsActiveMember: Activity status
- EstimatedSalary: Income level
- Exited: Churned (1) or not (0)
- Complain: Complaint filed (1) or not (0)
- Satisfaction Score: Complaint resolution rating
- Card Type: Card category
- Point Earned: Credit card rewards

**Removed Columns** (for optimization):
- RowNumber: Redundant
- Surname: Privacy/anonymization
- Gender: Ethical considerations
- Geography: Avoid cultural bias

---

## 5. Pipeline 1: EDA Implementation

### Install Kaggle:
```python
!pip install kaggle
import kaggle
kaggle.api.authenticate()
```

### Download Dataset:
```python
!kaggle datasets download -d radheshyamkollipara/bank-customer-churn

import zipfile
with zipfile.ZipFile('/content/bank-customer-churn.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')
```

### Load & Optimize:
```python
import pandas as pd

data1 = pd.read_csv('/content/Customer-Churn-Records.csv')

# Drop unnecessary columns
data1.drop(columns=['RowNumber', 'Surname', 'Gender', 'Geography'], inplace=True)

# Save optimized dataset
data1.to_csv('data1.csv', index=False)
```

---

## 6. EDA Insights

### Key Finding: Complaints → Churn
```python
sum_exited = data1['Exited'].sum()
sum_complain = data1['Complain'].sum()
percentage = (sum_complain / sum_exited) * 100

# Output: 100.29% correlation
```

**Insight**: Customers who complain almost always leave the bank

### Age Analysis:
```python
# Customers 50+ less likely to churn
age_threshold = 50
age_50_plus_exited = data1[(data1['Age'] >= age_threshold) & (data1['Exited'] == 1)].shape[0]
percentage_age = (age_50_plus_exited / sum_exited) * 100

# Output: 31.11% (younger customers more likely to churn)
```

### Salary Analysis:
```python
# Salary threshold: $100,000
salary_threshold = 100000
high_salary_exited = data1[(data1['EstimatedSalary'] > salary_threshold) & (data1['Exited'] == 1)].shape[0]
percentage_salary = (high_salary_exited / sum_exited) * 100

# Output: 51% (not significant)
```

### Correlation Heatmap:
```python
import seaborn as sns
import matplotlib.pyplot as plt

numerical_columns = data1.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, 8))
sns.heatmap(data1[numerical_columns].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

**Result**: Highest correlation between Complain and Exited

---

## 7. K-Means Clustering

### Data Preparation:
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

data2 = data1.copy()
features = data2[['CreditScore', 'Age', 'EstimatedSalary', 'Exited', 'Complain', 'Point Earned']]

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

### Find Optimal Clusters:
```python
for n_clusters in range(2, 5):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    db_index = davies_bouldin_score(features_scaled, cluster_labels)
    print(f'n_clusters={n_clusters}, silhouette={silhouette_avg:.4f}, DB={db_index:.4f}')
```

**Results**:
- n=2: silhouette=0.6129, DB=0.6144 (best)
- n=3: silhouette=0.3391, DB=0.9876
- n=4: silhouette=0.3243, DB=1.0234

### Apply Best Model:
```python
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
data2['class'] = kmeans.fit_predict(features_scaled)

# Cluster 0: Complained + Exited
# Cluster 1: Satisfied + Stayed
```

---

## 8. Pipeline 2: Pinecone Implementation

### Install Environment:
```python
!pip install openai==1.40.3
!pip install pinecone-client==5.0.1

import os
import openai
from pinecone import Pinecone, ServerlessSpec

# Initialize
os.environ['OPENAI_API_KEY'] = API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
```

### Chunking Strategy:
```python
# Each customer record = one chunk
output_lines = []
for index, row in data1.iterrows():
    row_data = [f"{col}: {row[col]}" for col in data1.columns]
    line = ' '.join(row_data)
    output_lines.append(line)

chunks = output_lines.copy()
# Total: 10,000 chunks
```

### Embedding:
```python
from openai import OpenAI

client = OpenAI()
embedding_model = "text-embedding-3-small"

def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Batch embedding (1000 chunks at a time)
embeddings = []
chunk_start = 0
chunk_end = 1000
pause_time = 3  # Avoid rate limits

while chunk_end <= len(chunks):
    chunks_to_embed = chunks[chunk_start:chunk_end]
    for chunk in chunks_to_embed:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    time.sleep(pause_time)
    chunk_start += 1000
    chunk_end += 1000

# Time: ~2689 seconds for 10,000 embeddings
```

### Duplicate for Scaling:
```python
dsize = 5  # Duplicate 5x
duplicated_chunks = []
duplicated_embeddings = []

for i in range(len(chunks)):
    for _ in range(dsize):
        duplicated_chunks.append(chunks[i])
        duplicated_embeddings.append(embeddings[i])

# Total: 50,000 vectors
```

---

## 9. Create Pinecone Index

### Initialize:
```python
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'bank-index-50000'
cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Create index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=1536,  # text-embedding-3-small dimension
        metric='cosine',
        spec=spec
    )
    time.sleep(1)

index = pc.Index(index_name)
```

### Upsert Data:
```python
import sys

def get_batch_size(data, limit=4000000):  # 4MB limit
    total_size = 0
    batch_size = 0
    for item in data:
        item_size = sum([sys.getsizeof(v) for v in item.values()])
        if total_size + item_size > limit:
            break
        total_size += item_size
        batch_size += 1
    return batch_size

def batch_upsert(data):
    total = len(data)
    i = 0
    while i < total:
        batch_size = get_batch_size(data[i:])
        batch = data[i:i + batch_size]
        if batch:
            index.upsert(vectors=batch)
            i += batch_size
            print(f"Upserted {i}/{total} items...")
        else:
            break

# Prepare data
ids = [str(i) for i in range(1, len(duplicated_chunks) + 1)]
data_for_upsert = [
    {"id": str(id), "values": emb, "metadata": {"text": chunk}}
    for id, (chunk, emb) in zip(ids, zip(duplicated_chunks, duplicated_embeddings))
]

# Upsert
batch_upsert(data_for_upsert)
# Time: ~560 seconds for 50,000 vectors (~56 sec per 10k)
```

---

## 10. Query Pinecone Index

### Test Query:
```python
query_text = "Customer Robertson CreditScore 632 Age 21 Tenure 2 Balance 0"
query_embedding = get_embedding(query_text)

query_results = index.query(
    vector=query_embedding,
    top_k=1,
    include_metadata=True
)

# Display results
for match in query_results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Text: {match['metadata']['text']}")

# Time: 0.74 seconds (constant even at 1M+ vectors)
```

---

## 11. Pipeline 3: RAG with GPT-4o

### Target Vector (Market Segment):
```python
# Target: Age ~42, Salary $100k+, Complained, Exiting
query_text = "Customer Henderson CreditScore 599 Age 37 Tenure 2 Balance 0 EstimatedSalary 101348.88 Exited 1 Complain 1"

query_embedding = get_embedding(query_text)

query_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# Extract relevant texts
relevant_texts = [match['metadata']['text'] for match in query_results['matches']]
combined_text = '\n'.join(relevant_texts)
```

### Augmented Prompt:
```python
query_prompt = "I have this customer bank record with interesting profiles. Write an engaging email with recommendations: "
itext = query_prompt + query_text + combined_text
```

### Generate with GPT-4o:
```python
from openai import OpenAI

client = OpenAI()
gpt_model = "gpt-4o"

response = client.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": "You are a community manager who can write engaging emails."},
        {"role": "user", "content": itext}
    ],
    temperature=0,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(response.choices[0].message.content)
# Time: ~2.83 seconds
```

**Example Output**:
```
Subject: Exclusive Benefits Await You at Our Bank!

Dear Valued Customer,

We hope this email finds you well. Based on your profile, we have identified several opportunities:

1. Personalized Financial Advice
2. Exclusive Rewards and Offers (DIAMOND cardholder)
3. Enhanced Credit Options
4. Complimentary Financial Health Check
5. Loyalty Programs

To explore these benefits, please visit...
```

---

## 12. Scaling Challenges & Solutions

### Cost Management:
- **OpenAI**: Monitor billing, rate limits
- **Pinecone**: Track read/write units, storage costs
- **Strategy**: Optimize data size before upserting

### Performance Optimization:
- **Embedding**: Batch processing (1000 chunks)
- **Upserting**: 4MB batch limit, ~56 sec per 10k vectors
- **Querying**: Constant <1 sec even at 1M+ vectors

### Model Selection:
- **Embedding**: text-embedding-3-small (balance efficiency/cost)
- **Generation**: GPT-4o (latest, most efficient)
- **Continual Monitoring**: Models evolve, update regularly

---

## 13. Key Formulas & Metrics

### Silhouette Score:
```
Range: [-1, 1]
High value = well-separated, cohesive clusters
Best: 0.6129 (n=2 clusters)
```

### Davies-Bouldin Index:
```
Lower = better clustering
Best: 0.6144 (n=2 clusters)
```

### Cosine Similarity:
```
similarity = (A · B) / (||A|| × ||B||)
Range: [0, 1]
Used for vector matching in Pinecone
```

### Performance Metrics:
```
Embedding: ~2689 sec for 10k vectors
Upserting: ~56 sec per 10k vectors (linear)
Querying: ~0.74 sec (constant at any scale)
Generation: ~2.83 sec per request
```

---

## 14. Best Practices

1. **Optimize Before Upserting**: Remove unnecessary columns
2. **Batch Processing**: Avoid rate limits, improve efficiency
3. **Monitor Costs**: Real-time billing tracking essential
4. **Test Scaling**: Use synthetic data to extrapolate performance
5. **Structured Chunking**: One record = one chunk for traceability
6. **Consistent Embedding**: Same model for dataset and queries
7. **Target Vectors**: Represent market segments, not individual queries
8. **Human Validation**: Marketing team workshops for target design
9. **Performance Tracking**: Measure time at each pipeline stage
10. **Model Updates**: Continually evaluate new OpenAI models

---

## 15. Key Takeaways

1. **Scaling changes application behavior** fundamentally
2. **EDA reveals patterns** before complex ML (complaints → churn)
3. **K-means clustering validates** relationships (2 clusters optimal)
4. **Pinecone scales linearly** for upserting, constant for querying
5. **Target vectors** represent market segments, not questions
6. **Augmented prompts** combine target + similar profiles
7. **GPT-4o generates** personalized recommendations
8. **Cost monitoring essential** when scaling to 1M+ vectors
9. **Batch processing** avoids rate limits, improves efficiency
10. **Human insights** guide target vector design

---

## Interview-Ready Q&A

**Q: Why Pinecone for scaling?**
Serverless architecture, pay-as-you-go, constant query time even at 1M+ vectors, no server management.

**Q: Key EDA insight?**
100.29% correlation between complaints and churn—customers who complain almost always leave.

**Q: Why k-means with n=2?**
Best silhouette score (0.6129) and Davies-Bouldin index (0.6144), clearly separates complainers from satisfied customers.

**Q: Embedding strategy?**
Batch 1000 chunks at a time, pause 3 sec between batches to avoid rate limits, use text-embedding-3-small for efficiency.

**Q: Upserting performance?**
Linear: ~56 seconds per 10,000 vectors. 4MB batch limit. Total 560 sec for 50k vectors.

**Q: Query performance at scale?**
Constant: ~0.74 seconds regardless of index size (tested at 1M+ vectors).

**Q: What's a target vector?**
Market segment profile (age, salary, complaint status) used to find similar customers for targeted recommendations.

**Q: Why remove columns?**
Optimize data size → reduce storage costs, faster processing, avoid bias (gender, geography).

**Q: Scaling challenges?**
Cost monitoring, rate limits, performance optimization, model selection, error amplification.

---

## Tools & Technologies

**Vector Store**: Pinecone (serverless, AWS us-east-1)
**Embedding**: OpenAI text-embedding-3-small (1536 dimensions)
**Generation**: OpenAI GPT-4o
**Data Source**: Kaggle Bank Customer Churn dataset
**ML**: scikit-learn (k-means, StandardScaler, metrics)
**Visualization**: seaborn, matplotlib
**Data Processing**: pandas, NumPy


---

## Yes/No Questions with Answers

**Q1: Does using a Kaggle dataset typically involve downloading and processing real-world data for analysis?**
Yes, Kaggle datasets are used for practical, real-world data analysis and modeling.

**Q2: Is Pinecone capable of efficiently managing large-scale vector storage for AI applications?**
Yes, Pinecone is designed for large-scale vector storage, making it suitable for complex AI tasks.

**Q3: Can k-means clustering help validate relationships between features such as customer complaints and churn?**
Yes, k-means clustering is useful for identifying and validating patterns in datasets.

**Q4: Does leveraging over a million vectors in a database hinder the ability to personalize customer interactions?**
No, handling large volumes of vectors allows for more personalized and targeted customer interactions.

**Q5: Is the primary objective of using generative AI in business applications to automate and improve decision-making processes?**
Yes, generative AI aims to automate and refine decision-making in various business applications.

**Q6: Are lightweight development environments advantageous for rapid prototyping and application development?**
Yes, they streamline development processes, making it easier and faster to test and deploy applications.

**Q7: Can Pinecone's architecture automatically scale to accommodate increasing data loads without manual intervention?**
Yes, Pinecone's serverless architecture supports automatic scaling to handle larger data volumes efficiently.

**Q8: Is generative AI typically employed to create dynamic content and recommendations based on user data?**
Yes, generative AI is often used to generate customized content and recommendations dynamically.

**Q9: Does the integration of AI technologies such as Pinecone and OpenAI require significant manual configuration and maintenance?**
No, these technologies are designed to minimize manual efforts in configuration and maintenance through automation.

**Q10: Are projects that use vector databases and AI expected to effectively handle complex queries and large datasets?**
Yes, vector databases combined with AI are particularly well-suited for complex queries and managing large datasets.
