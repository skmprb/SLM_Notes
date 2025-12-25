# Small Language Model (SLM) Implementation Notes

## Overview
This notebook implements a Small Language Model from scratch using PyTorch, trained on the TinyStories dataset. The model follows the GPT (Generative Pre-trained Transformer) architecture.

## 1. Data Preparation and Tokenization

### Dataset
- **TinyStories Dataset**: ~2 million training stories, ~20,000 validation stories
- Purpose: Generate coherent short stories for children

### Tokenization with BPE (Byte Pair Encoding)
**Mathematical Concept**: Subword tokenization algorithm

```python
encoding = tiktoken.get_encoding("gpt2")
```

**Key Benefits**:
- Handles out-of-vocabulary words
- Balances between character and word-level tokenization
- Vocabulary size: 50,257 tokens

### Data Storage Optimization
- **Memory Mapping**: Uses `np.memmap` for efficient disk-based data access
- **Binary Format**: Stores tokenized data in `.bin` files
- **Batching**: Processes data in chunks of 1024 batches

**Mathematical Representation**:
- Each token ID: 16-bit unsigned integer (2^16 = 65,536 possible tokens)
- Context window: 128 tokens

## 2. Model Architecture - GPT Implementation

### Core Components

#### 2.1 Layer Normalization
**Mathematical Formula**:
```
LayerNorm(x) = γ * (x - μ) / σ + β
```
Where:
- μ = mean across features
- σ = standard deviation across features  
- γ, β = learnable parameters

**Purpose**: Stabilizes training and prevents vanishing gradients

#### 2.2 Multi-Head Self-Attention
**Mathematical Foundation**:

**Attention Mechanism**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Implementation Details**:
- Number of heads: 6
- Embedding dimension: 384
- Head dimension: 384/6 = 64

**Causal Masking**: Prevents attention to future tokens
```python
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
```

#### 2.3 Feed-Forward Network (MLP)
**Architecture**:
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

**Dimensions**:
- Input: 384
- Hidden: 4 × 384 = 1536 (standard 4x expansion)
- Output: 384

**GELU Activation**: Gaussian Error Linear Unit
```
GELU(x) = x * Φ(x)
```
Where Φ(x) is the cumulative distribution function of standard normal distribution.

#### 2.4 Transformer Block
**Residual Connections**:
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Mathematical Benefit**: Enables gradient flow through deep networks, preventing vanishing gradients.

### Model Configuration
```python
config = GPTConfig(
    vocab_size=50257,      # GPT-2 tokenizer vocabulary
    block_size=128,        # Context length
    n_layer=6,            # Number of transformer blocks
    n_head=6,             # Number of attention heads
    n_embd=384,           # Embedding dimension
    dropout=0.1,          # Regularization
    bias=True             # Use bias terms
)
```

## 3. Training Process

### 3.1 Loss Function
**Cross-Entropy Loss**:
```
L = -∑(y_i * log(ŷ_i))
```

Where:
- y_i = true token (one-hot encoded)
- ŷ_i = predicted probability distribution

### 3.2 Optimization Strategy

#### AdamW Optimizer
**Mathematical Update Rule**:
```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2
θ_t = θ_{t-1} - α * (m_t / (√v_t + ε) + λ * θ_{t-1})
```

**Parameters**:
- Learning rate (α): 1e-4
- β_1 = 0.9, β_2 = 0.95
- Weight decay (λ): 0.1
- ε = 1e-9

#### Learning Rate Scheduling
1. **Linear Warmup** (1000 steps): Gradually increases learning rate
2. **Cosine Annealing**: Smoothly decreases learning rate
```
lr_t = lr_min + (lr_max - lr_min) * (1 + cos(π * t / T)) / 2
```

### 3.3 Training Configuration
- **Batch Size**: 32
- **Context Length**: 128 tokens
- **Gradient Accumulation**: 32 steps
- **Max Iterations**: 10,000
- **Evaluation Interval**: 500 steps

### 3.4 Automatic Mixed Precision (AMP)
**Purpose**: Dynamically uses FP16 for matrix operations and FP32 for numerical stability
**Benefit**: Faster training with reduced memory usage

## 4. Mathematical Concepts in Detail

### 4.1 Positional Encoding
**Learned Positional Embeddings**:
```python
pos_emb = self.transformer.wpe(pos)  # Learnable position embeddings
tok_emb = self.transformer.wte(idx)  # Token embeddings
x = tok_emb + pos_emb
```

### 4.2 Weight Initialization
**Xavier/Glorot Normal Initialization**:
```
W ~ N(0, 0.02²)
```

**Scaled Initialization for Residual Layers**:
```
W ~ N(0, (0.02 / √(2 * n_layers))²)
```

### 4.3 Gradient Clipping
**Purpose**: Prevents exploding gradients
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

## 5. Text Generation

### 5.1 Autoregressive Generation
**Process**:
1. Start with prompt tokens
2. Predict next token probability distribution
3. Sample from distribution
4. Append to sequence
5. Repeat

### 5.2 Sampling Strategies
**Temperature Scaling**:
```
p_i = exp(logit_i / T) / ∑exp(logit_j / T)
```

**Top-k Sampling**: Only consider k most likely tokens

## 6. Performance Metrics

### Training Results
- **Final Training Loss**: ~2.96
- **Final Validation Loss**: ~2.97
- **Training Time**: ~2 hours on T4 GPU
- **Total Parameters**: ~10M (estimated)

### Loss Progression
The model shows consistent learning with decreasing loss over 10,000 iterations, indicating successful convergence.

## 7. Key Mathematical Insights

1. **Attention Mechanism**: Enables the model to focus on relevant parts of the input sequence
2. **Residual Connections**: Allow deep networks to train effectively by providing gradient highways
3. **Layer Normalization**: Stabilizes training by normalizing activations
4. **Causal Masking**: Ensures autoregressive property for language modeling
5. **Weight Tying**: Shares parameters between token embedding and output projection layers

## 8. Implementation Optimizations

1. **Flash Attention**: Uses PyTorch's optimized attention implementation when available
2. **Memory Mapping**: Efficient data loading without RAM overflow
3. **Gradient Accumulation**: Simulates larger batch sizes with limited memory
4. **Mixed Precision**: Balances speed and numerical stability

This implementation demonstrates a complete pipeline for training a small but functional language model, incorporating modern deep learning techniques and mathematical foundations essential for transformer-based architectures.