# RAG Mathematics - Mathematical Concepts in Retrieval-Augmented Generation

## 1. Vector Mathematics Fundamentals

### Vector Representation
A vector **v** in n-dimensional space is represented as:
```
v = [v₁, v₂, v₃, ..., vₙ]
```

### Vector Operations

#### Vector Addition
```
v + u = [v₁ + u₁, v₂ + u₂, ..., vₙ + uₙ]
```

#### Scalar Multiplication
```
c × v = [c×v₁, c×v₂, ..., c×vₙ]
```

#### Vector Magnitude (L2 Norm)
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

## 2. Distance Metrics and Similarity Measures

### Cosine Similarity
Most commonly used in RAG systems:
```
cosine_similarity(v, u) = (v · u) / (||v|| × ||u||)

Where v · u = v₁×u₁ + v₂×u₂ + ... + vₙ×uₙ (dot product)
```

**Range**: [-1, 1]
- 1: Identical direction
- 0: Orthogonal (no similarity)
- -1: Opposite direction

### Euclidean Distance
```
euclidean_distance(v, u) = √[(v₁-u₁)² + (v₂-u₂)² + ... + (vₙ-uₙ)²]
```

### Manhattan Distance (L1 Distance)
```
manhattan_distance(v, u) = |v₁-u₁| + |v₂-u₂| + ... + |vₙ-uₙ|
```

### Dot Product Similarity
```
dot_product(v, u) = v₁×u₁ + v₂×u₂ + ... + vₙ×uₙ
```

## 3. Embedding Mathematics

### Embedding Transformation
Text → Embedding function f(text) → Vector
```
f: Text → ℝⁿ
where ℝⁿ is n-dimensional real number space
```

### Dimensionality Reduction

#### Principal Component Analysis (PCA)
Reduces dimensionality while preserving variance:
```
Y = XW
where:
- X: original data matrix (m × n)
- W: transformation matrix (n × k)
- Y: reduced data matrix (m × k)
- k < n (reduced dimensions)
```

#### Singular Value Decomposition (SVD)
```
X = UΣVᵀ
where:
- U: left singular vectors
- Σ: diagonal matrix of singular values
- V: right singular vectors
```

## 4. Approximate Nearest Neighbor (ANN) Mathematics

### HNSW (Hierarchical Navigable Small World)

#### Graph Construction
- **Layer probability**: P(layer = l) = (1/ln(2))^l
- **Maximum layer**: mₗ = ⌊-ln(uniform(0,1)) × mₗ⌋

#### Search Complexity
- **Time Complexity**: O(log N) on average
- **Space Complexity**: O(N × M) where M is maximum connections

### IVF (Inverted File Index)

#### Clustering
Uses k-means clustering:
```
Minimize: Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
where:
- k: number of clusters
- Cᵢ: cluster i
- μᵢ: centroid of cluster i
```

#### Search Process
1. Find nearest cluster centroids
2. Search only within selected clusters
3. Reduces search space from N to N/k (approximately)

### Product Quantization (PQ)

#### Vector Decomposition
```
x = [x₁, x₂, ..., xₘ] where each xⱼ ∈ ℝᵈ/ᵐ
```

#### Quantization Error
```
Error = ||x - x̂||² where x̂ is quantized vector
```

## 5. Retrieval Scoring Mathematics

### TF-IDF (Term Frequency-Inverse Document Frequency)
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

where:
TF(t,d) = (number of times term t appears in document d) / (total terms in d)
IDF(t) = log(N / |{d ∈ D : t ∈ d}|)

N: total number of documents
D: document collection
```

### BM25 (Best Matching 25)
```
BM25(q,d) = Σᵢ₌₁ⁿ IDF(qᵢ) × [f(qᵢ,d) × (k₁ + 1)] / [f(qᵢ,d) + k₁ × (1 - b + b × |d|/avgdl)]

where:
- f(qᵢ,d): frequency of term qᵢ in document d
- |d|: length of document d
- avgdl: average document length
- k₁, b: tuning parameters (typically k₁=1.2, b=0.75)
```

## 6. Evaluation Metrics Mathematics

### Precision
```
Precision = |Relevant ∩ Retrieved| / |Retrieved|
```

### Recall
```
Recall = |Relevant ∩ Retrieved| / |Relevant|
```

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Mean Reciprocal Rank (MRR)
```
MRR = (1/|Q|) × Σᵢ₌₁|Q| (1/rankᵢ)
where rankᵢ is the rank of first relevant result for query i
```

### Mean Average Precision (MAP)
```
MAP = (1/|Q|) × Σᵢ₌₁|Q| AP(qᵢ)

where AP(q) = (1/|Relevant|) × Σₖ₌₁ⁿ P(k) × rel(k)
- P(k): precision at rank k
- rel(k): 1 if item at rank k is relevant, 0 otherwise
```

### Normalized Discounted Cumulative Gain (NDCG)
```
NDCG@k = DCG@k / IDCG@k

DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)

where:
- relᵢ: relevance score of item at position i
- IDCG@k: ideal DCG (best possible ranking)
```

## 7. Optimization Mathematics

### Gradient Descent for Embedding Training
```
θₜ₊₁ = θₜ - α∇L(θₜ)
where:
- θ: model parameters
- α: learning rate
- L: loss function
- ∇L: gradient of loss function
```

### Loss Functions

#### Contrastive Loss
```
L = (1/2N) × Σᵢ₌₁ᴺ [y × d² + (1-y) × max(0, margin - d)²]
where:
- d: distance between embeddings
- y: 1 if similar, 0 if dissimilar
- margin: threshold parameter
```

#### Triplet Loss
```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

## 8. Chunking Mathematics

### Fixed Chunking
```
Number of chunks = ⌈Document_length / Chunk_size⌉
```

### Sliding Window Chunking
```
Number of chunks = ⌈(Document_length - Chunk_size) / Stride⌉ + 1
where Stride = Chunk_size - Overlap
```

### Overlap Calculation
```
Overlap_ratio = Overlap_tokens / Chunk_size
Recommended: 0.1 ≤ Overlap_ratio ≤ 0.2
```

## 9. Performance Metrics

### Latency Calculation
```
Total_latency = Embedding_time + Retrieval_time + Generation_time
```

### Throughput Calculation
```
Throughput = Number_of_requests / Time_period
```

### Memory Usage for Embeddings
```
Memory = Number_of_vectors × Dimensions × Bytes_per_dimension
Example: 1M vectors × 768 dimensions × 4 bytes = 3.072 GB
```

## 10. Statistical Measures

### Embedding Quality Assessment

#### Intrinsic Dimensionality
```
ID ≈ μ²/2σ²
where μ and σ are mean and standard deviation of pairwise distances
```

#### Clustering Coefficient
```
C = (Number of triangles × 3) / Number of connected triples
```

### Distribution Analysis

#### Normal Distribution for Similarity Scores
```
f(x) = (1/σ√(2π)) × e^(-½((x-μ)/σ)²)
```

#### Confidence Intervals
```
CI = x̄ ± z_(α/2) × (σ/√n)
where:
- x̄: sample mean
- z_(α/2): critical value
- σ: standard deviation
- n: sample size
```

## 11. Information Theory in RAG

### Entropy
```
H(X) = -Σᵢ p(xᵢ) × log₂(p(xᵢ))
```

### Mutual Information
```
I(X;Y) = Σᵢ Σⱼ p(xᵢ,yⱼ) × log₂(p(xᵢ,yⱼ)/(p(xᵢ)×p(yⱼ)))
```

### Cross-Entropy Loss
```
L = -Σᵢ yᵢ × log(ŷᵢ)
where yᵢ is true label and ŷᵢ is predicted probability
```

## 12. Advanced Mathematical Concepts

### Learning-to-Rank Mathematics

#### Pointwise Ranking
```
L = Σᵢ₌₁ⁿ loss(f(xᵢ), yᵢ)
where f(xᵢ) is predicted relevance and yᵢ is true relevance
```

#### Pairwise Ranking (RankNet)
```
P(dᵢ > dⱼ) = 1 / (1 + e^(-σ(sᵢ - sⱼ)))
where sᵢ, sⱼ are relevance scores for documents i, j
```

#### Listwise Ranking (ListNet)
```
L = -Σₖ P(πₖ) × log(P̂(πₖ))
where π is a permutation and P̂ is predicted probability
```

### Multi-Modal Embedding Mathematics

#### Cross-Modal Similarity
```
sim(text, image) = cosine_similarity(f_text(text), f_image(image))
where f_text and f_image are embedding functions
```

#### Contrastive Learning Loss
```
L = -log(exp(sim(aᵢ, pᵢ)/τ) / Σⱼ exp(sim(aᵢ, nⱼ)/τ))
where aᵢ is anchor, pᵢ is positive, nⱼ are negatives, τ is temperature
```

### Graph-Enhanced RAG Mathematics

#### Graph Attention Networks
```
eᵢⱼ = a(Whᵢ, Whⱼ)
αᵢⱼ = softmax(eᵢⱼ) = exp(eᵢⱼ) / Σₖ∈Nᵢ exp(eᵢₖ)
h'ᵢ = σ(Σⱼ∈Nᵢ αᵢⱼWhⱼ)
```

#### Random Walk on Graphs
```
P(vₜ₊₁ = j | vₜ = i) = Aᵢⱼ / Σₖ Aᵢₖ
where A is adjacency matrix
```

### Federated Learning Mathematics

#### Federated Averaging
```
w^(t+1) = Σₖ₌₁ᴷ (nₖ/n) × wₖ^(t+1)
where nₖ is local data size, n is total data size
```

#### Differential Privacy
```
Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S] + δ
where D, D' differ by one record, ε is privacy budget
```

### Compression Mathematics

#### Vector Quantization
```
q(x) = argmin_c ||x - c||²
where c ∈ C is codebook entry
```

#### Huffman Coding Entropy
```
H = -Σᵢ p(xᵢ) × log₂(p(xᵢ))
where p(xᵢ) is probability of symbol xᵢ
```

### Attention Mechanisms

#### Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax(QK^T/√dₖ)V
where Q, K, V are query, key, value matrices
```

#### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

### Uncertainty Quantification

#### Bayesian Neural Networks
```
p(y|x,D) = ∫ p(y|x,w)p(w|D)dw
where w are network weights, D is training data
```

#### Monte Carlo Dropout
```
μ ≈ (1/T) × Σₜ₌₁ᵀ f(x,wₜ)
σ² ≈ (1/T) × Σₜ₌₁ᵀ f(x,wₜ)² - μ²
```

### Reinforcement Learning for RAG

#### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
where α is learning rate, γ is discount factor
```

#### Policy Gradient
```
∇θ J(θ) = E[∇θ log π(a|s,θ) × R]
where π is policy, R is reward
```

## Summary

These mathematical concepts form the foundation of RAG systems:
- **Vector mathematics** enables semantic representation
- **Distance metrics** measure similarity between concepts
- **ANN algorithms** provide efficient search capabilities
- **Evaluation metrics** quantify system performance
- **Optimization techniques** improve model quality
- **Statistical measures** ensure robust performance
- **Advanced techniques** enable sophisticated RAG variants
- **Multi-modal mathematics** supports diverse content types
- **Graph mathematics** enables knowledge graph integration
- **Privacy mathematics** ensures secure deployments

Understanding these mathematical principles helps in designing, implementing, and optimizing effective RAG systems for complex real-world applications.