# Attention Optimizations

## 🎯 Overview

Standard attention has O(n²) complexity in both memory and computation, limiting scalability. Various optimization techniques have been developed to make attention more efficient while maintaining performance.

## ⚡ FlashAttention

### Core Innovation

FlashAttention reduces memory usage from O(n²) to O(n) by avoiding materialization of the full attention matrix through **tiling** and **recomputation**.

### Mathematical Foundation

Standard attention computes:
```
S = QK^T / √d_k
P = softmax(S)
O = PV
```

FlashAttention computes the same result but in blocks:

```python
def flash_attention_forward(Q, K, V, block_size=64):
    """Simplified FlashAttention implementation."""
    N, d = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros(N)  # Row sums for softmax
    m = torch.full((N,), -float('inf'))  # Row maxes for softmax
    
    # Process in blocks
    for j in range(0, N, block_size):
        # Load K, V blocks
        K_j = K[j:j+block_size]
        V_j = V[j:j+block_size]
        
        for i in range(0, N, block_size):
            # Load Q block
            Q_i = Q[i:i+block_size]
            
            # Compute attention scores for block
            S_ij = torch.matmul(Q_i, K_j.T) / math.sqrt(d)
            
            # Online softmax update
            m_new = torch.maximum(m[i:i+block_size], torch.max(S_ij, dim=1)[0])
            l_new = torch.exp(m[i:i+block_size] - m_new) * l[i:i+block_size] + \
                    torch.sum(torch.exp(S_ij - m_new.unsqueeze(1)), dim=1)
            
            # Update output
            O[i:i+block_size] = (O[i:i+block_size] * torch.exp(m[i:i+block_size] - m_new).unsqueeze(1) * 
                                 l[i:i+block_size].unsqueeze(1) + 
                                 torch.matmul(torch.exp(S_ij - m_new.unsqueeze(1)), V_j)) / l_new.unsqueeze(1)
            
            # Update statistics
            m[i:i+block_size] = m_new
            l[i:i+block_size] = l_new
    
    return O

class FlashAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Apply FlashAttention for each head
        outputs = []
        for h in range(self.num_heads):
            head_output = flash_attention_forward(
                Q[:, :, h, :].squeeze(), 
                K[:, :, h, :].squeeze(), 
                V[:, :, h, :].squeeze(),
                self.block_size
            )
            outputs.append(head_output)
        
        # Concatenate heads
        output = torch.cat(outputs, dim=-1)
        return self.W_o(output)
```

### Key Benefits
- **Memory Efficiency**: O(n) instead of O(n²) memory
- **Speed**: Faster for long sequences due to better memory access
- **Exact**: Mathematically equivalent to standard attention
- **Hardware Optimized**: Designed for GPU memory hierarchy

## 🕸️ Sparse Attention

### Motivation

Most attention weights are small and can be pruned without significant performance loss.

### Sparse Attention Patterns

**1. Fixed Patterns**
```python
def create_sparse_mask(seq_len, pattern='local'):
    """Create sparse attention masks."""
    mask = torch.zeros(seq_len, seq_len)
    
    if pattern == 'local':
        # Local attention window
        window_size = 128
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
            
    elif pattern == 'strided':
        # Strided attention
        stride = 64
        for i in range(seq_len):
            # Attend to every stride-th position
            for j in range(0, seq_len, stride):
                mask[i, j] = 1
            # Plus local window
            start = max(0, i - 32)
            end = min(seq_len, i + 33)
            mask[i, start:end] = 1
            
    elif pattern == 'random':
        # Random sparse connections
        sparsity = 0.1  # 10% connections
        num_connections = int(seq_len * sparsity)
        for i in range(seq_len):
            connections = torch.randperm(seq_len)[:num_connections]
            mask[i, connections] = 1
    
    return mask

class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_pattern='local'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sparsity_pattern = sparsity_pattern
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create sparse mask
        mask = create_sparse_mask(seq_len, self.sparsity_pattern)
        mask = mask.to(x.device)
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sparse mask
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and output
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

**2. Learned Sparsity**
```python
class LearnedSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_ratio=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sparsity_ratio = sparsity_ratio
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Learnable sparsity gates
        self.sparsity_gate = nn.Linear(d_model, num_heads)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Compute sparsity gates
        gate_scores = self.sparsity_gate(x)  # [batch, seq_len, num_heads]
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply learned sparsity
        for h in range(self.num_heads):
            head_gates = gate_scores[:, :, h]  # [batch, seq_len]
            
            # Top-k selection for sparsity
            k = int(seq_len * self.sparsity_ratio)
            _, top_indices = torch.topk(head_gates, k, dim=-1)
            
            # Create mask
            mask = torch.zeros_like(head_gates)
            mask.scatter_(-1, top_indices, 1)
            
            # Apply mask to attention scores
            scores[:, h] = scores[:, h].masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

## 📏 Linear Attention

### Core Idea

Approximate attention using kernel methods to achieve O(n) complexity:

```
Attention(Q,K,V) ≈ φ(Q)(φ(K)^T V)
```

Where φ is a feature map function.

### Implementation

```python
def elu_feature_map(x):
    """ELU-based feature map for linear attention."""
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, feature_map=elu_feature_map):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_map = feature_map
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply feature maps
        Q_prime = self.feature_map(Q)
        K_prime = self.feature_map(K)
        
        # Linear attention computation
        # O(n) complexity: φ(Q)(φ(K)^T V)
        KV = torch.matmul(K_prime.transpose(-2, -1), V)  # [batch, heads, d_k, d_k]
        output = torch.matmul(Q_prime, KV)  # [batch, heads, seq_len, d_k]
        
        # Normalization
        normalizer = torch.matmul(Q_prime, K_prime.sum(dim=-2, keepdim=True).transpose(-2, -1))
        output = output / (normalizer + 1e-6)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)

class CausalLinearAttention(nn.Module):
    """Causal version of linear attention for autoregressive models."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Apply feature maps
        Q_prime = elu_feature_map(Q)
        K_prime = elu_feature_map(K)
        
        # Causal linear attention using cumulative sums
        outputs = []
        kv_state = torch.zeros(batch_size, self.num_heads, self.d_k, self.d_k, device=x.device)
        k_state = torch.zeros(batch_size, self.num_heads, self.d_k, device=x.device)
        
        for t in range(seq_len):
            # Update states
            kv_state += torch.einsum('bhd,bhe->bhde', K_prime[:, t], V[:, t])
            k_state += K_prime[:, t]
            
            # Compute output for current timestep
            output_t = torch.einsum('bhd,bhde->bhe', Q_prime[:, t], kv_state)
            normalizer = torch.einsum('bhd,bhd->bh', Q_prime[:, t], k_state).unsqueeze(-1)
            output_t = output_t / (normalizer + 1e-6)
            
            outputs.append(output_t)
        
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, heads, d_k]
        output = output.view(batch_size, seq_len, self.d_model)
        
        return self.W_o(output)
```

## 🪟 Sliding Window Attention

### Concept

Limit attention to a fixed-size window around each position, reducing complexity from O(n²) to O(n×w) where w is window size.

### Implementation

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=256):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def create_sliding_window_mask(self, seq_len):
        """Create sliding window attention mask."""
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            # Define window boundaries
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        
        return mask
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create sliding window mask
        mask = self.create_sliding_window_mask(seq_len).to(x.device)
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sliding window mask
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and output
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)

class AdaptiveSlidingWindow(nn.Module):
    """Adaptive sliding window that adjusts size based on content."""
    
    def __init__(self, d_model, num_heads, min_window=64, max_window=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.min_window = min_window
        self.max_window = max_window
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Window size predictor
        self.window_predictor = nn.Linear(d_model, 1)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Predict adaptive window sizes
        window_logits = self.window_predictor(x).squeeze(-1)  # [batch, seq_len]
        window_sizes = torch.sigmoid(window_logits) * (self.max_window - self.min_window) + self.min_window
        window_sizes = window_sizes.int()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create adaptive masks for each batch item
        outputs = []
        for b in range(batch_size):
            # Create position-specific mask
            mask = torch.zeros(seq_len, seq_len, device=x.device)
            
            for i in range(seq_len):
                window_size = window_sizes[b, i].item()
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = 1
            
            # Compute attention for this batch item
            scores = torch.matmul(Q[b], K[b].transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attention = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention, V[b])
            
            outputs.append(output)
        
        output = torch.stack(outputs, dim=0)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(output)
```

## 📊 Performance Comparison

### Complexity Analysis

```python
class AttentionComplexityAnalyzer:
    def __init__(self):
        self.methods = {
            'standard': self.standard_complexity,
            'flash': self.flash_complexity,
            'sparse': self.sparse_complexity,
            'linear': self.linear_complexity,
            'sliding_window': self.sliding_window_complexity
        }
    
    def standard_complexity(self, seq_len, d_model, sparsity=1.0, window_size=None):
        """Standard attention complexity."""
        return {
            'memory': seq_len ** 2,
            'compute': seq_len ** 2 * d_model,
            'parameters': 4 * d_model ** 2  # Q, K, V, O projections
        }
    
    def flash_complexity(self, seq_len, d_model, sparsity=1.0, window_size=None):
        """FlashAttention complexity."""
        return {
            'memory': seq_len,  # O(n) memory
            'compute': seq_len ** 2 * d_model,  # Same compute, better memory access
            'parameters': 4 * d_model ** 2
        }
    
    def sparse_complexity(self, seq_len, d_model, sparsity=0.1, window_size=None):
        """Sparse attention complexity."""
        return {
            'memory': seq_len ** 2 * sparsity,
            'compute': seq_len ** 2 * d_model * sparsity,
            'parameters': 4 * d_model ** 2
        }
    
    def linear_complexity(self, seq_len, d_model, sparsity=1.0, window_size=None):
        """Linear attention complexity."""
        return {
            'memory': seq_len * d_model,
            'compute': seq_len * d_model ** 2,
            'parameters': 4 * d_model ** 2
        }
    
    def sliding_window_complexity(self, seq_len, d_model, sparsity=1.0, window_size=256):
        """Sliding window attention complexity."""
        return {
            'memory': seq_len * window_size,
            'compute': seq_len * window_size * d_model,
            'parameters': 4 * d_model ** 2
        }
    
    def compare_methods(self, seq_len=4096, d_model=768):
        """Compare all methods."""
        results = {}
        
        for method_name, method_func in self.methods.items():
            if method_name == 'sparse':
                results[method_name] = method_func(seq_len, d_model, sparsity=0.1)
            elif method_name == 'sliding_window':
                results[method_name] = method_func(seq_len, d_model, window_size=256)
            else:
                results[method_name] = method_func(seq_len, d_model)
        
        return results
```

### Benchmark Results

```python
def benchmark_attention_methods():
    """Benchmark different attention methods."""
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    d_model = 768
    
    analyzer = AttentionComplexityAnalyzer()
    
    results = {}
    for seq_len in seq_lengths:
        results[seq_len] = analyzer.compare_methods(seq_len, d_model)
    
    # Print memory usage comparison
    print("Memory Usage (relative to standard attention):")
    print("Seq Len\tStandard\tFlash\tSparse\tLinear\tSliding")
    
    for seq_len in seq_lengths:
        standard_mem = results[seq_len]['standard']['memory']
        
        row = f"{seq_len}\t1.00\t"
        row += f"{results[seq_len]['flash']['memory'] / standard_mem:.2f}\t"
        row += f"{results[seq_len]['sparse']['memory'] / standard_mem:.2f}\t"
        row += f"{results[seq_len]['linear']['memory'] / standard_mem:.2f}\t"
        row += f"{results[seq_len]['sliding_window']['memory'] / standard_mem:.2f}"
        
        print(row)
```

## 🎯 Method Selection Guide

### Use Case Recommendations

**FlashAttention**
- ✅ Drop-in replacement for standard attention
- ✅ Long sequences with sufficient GPU memory
- ✅ Training large models

**Sparse Attention**
- ✅ Very long sequences (>8K tokens)
- ✅ Known attention patterns
- ⚠️ May lose some performance

**Linear Attention**
- ✅ Extremely long sequences (>16K tokens)
- ✅ Streaming applications
- ⚠️ Approximation may affect quality

**Sliding Window Attention**
- ✅ Local dependencies are most important
- ✅ Consistent performance across lengths
- ✅ Simple implementation

### Hybrid Approaches

```python
class HybridAttention(nn.Module):
    """Combines multiple attention mechanisms."""
    
    def __init__(self, d_model, num_heads, config):
        super().__init__()
        self.config = config
        
        # Initialize different attention types
        if config.get('use_flash', False):
            self.flash_attn = FlashAttention(d_model, num_heads)
        
        if config.get('use_sparse', False):
            self.sparse_attn = SparseAttention(d_model, num_heads)
        
        if config.get('use_sliding', False):
            self.sliding_attn = SlidingWindowAttention(d_model, num_heads)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Choose attention method based on sequence length
        if seq_len <= 2048 and hasattr(self, 'flash_attn'):
            return self.flash_attn(x)
        elif seq_len <= 8192 and hasattr(self, 'sparse_attn'):
            return self.sparse_attn(x)
        elif hasattr(self, 'sliding_attn'):
            return self.sliding_attn(x)
        else:
            # Fallback to standard attention
            return self.standard_attention(x)
```

## 📚 Summary

### Key Insights

**FlashAttention**
- Memory-efficient without approximation
- Best for moderate to long sequences
- Hardware-optimized implementation

**Sparse Attention**
- Significant memory savings
- Pattern-dependent performance
- Good for structured data

**Linear Attention**
- True linear complexity
- Approximation trade-offs
- Best for very long sequences

**Sliding Window Attention**
- Simple and effective
- Consistent performance
- Good locality bias

### Selection Criteria
- **Sequence length**: Primary factor in method selection
- **Hardware constraints**: Memory and compute limitations
- **Quality requirements**: Exact vs. approximate attention
- **Implementation complexity**: Development and maintenance costs

### Future Directions
- **Adaptive methods**: Dynamic selection based on content
- **Hardware co-design**: Optimizing for specific accelerators
- **Learned patterns**: AI-discovered attention patterns
- **Multi-scale attention**: Combining different granularities