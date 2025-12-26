# Positional Representation Variants

## 🎯 Overview

Positional encoding is crucial for transformers since self-attention is permutation-invariant. Different positional representation methods have been developed to handle various sequence lengths and improve model performance.

## 📚 Learned Positional Embeddings

### Core Concept

**Learned Embeddings**: Trainable parameters that represent position information, learned during model training.

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embeddings = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.embeddings(positions)
```

### Advantages
- **Adaptive**: Learns optimal position representations for specific tasks
- **Flexible**: Can capture complex positional relationships
- **Task-specific**: Optimized during training for particular applications

### Limitations
- **Fixed length**: Cannot extrapolate beyond training sequence length
- **Memory overhead**: Requires storing position embeddings
- **Generalization**: May not transfer well to different sequence lengths

## 🔄 Relative Positional Encoding

### Mathematical Foundation

Instead of absolute positions, relative encoding focuses on distances between positions:

```
RelativeAttention(i,j) = Attention(x_i, x_j) + R(i-j)
```

Where `R(i-j)` represents the relative position bias.

### Implementation

```python
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_distance=32):
        super().__init__()
        self.max_relative_distance = max_relative_distance
        self.relative_embeddings = nn.Embedding(
            2 * max_relative_distance + 1, d_model
        )
        
    def forward(self, seq_len):
        # Create relative position matrix
        positions = torch.arange(seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to maximum distance
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance
        )
        
        # Shift to positive indices
        relative_positions += self.max_relative_distance
        
        return self.relative_embeddings(relative_positions)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.relative_pos = RelativePositionalEncoding(self.d_k)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias
        rel_pos_bias = self.relative_pos(seq_len)
        scores += rel_pos_bias
        
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
```

### Benefits
- **Length generalization**: Works with sequences longer than training
- **Translation invariance**: Focuses on relative distances
- **Efficiency**: Bounded memory usage with clipping

## 🌀 Rotary Positional Embeddings (RoPE)

### Mathematical Foundation

RoPE rotates query and key vectors by angles proportional to their positions:

```
f(x_m, m) = R_Θ,m x_m
```

Where `R_Θ,m` is a rotation matrix based on position `m`.

### Core Implementation

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    """Precompute rotation frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to queries and keys."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Precompute rotation frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.d_k, max_seq_len * 2)
        )
        
    def forward(self, x, start_pos=0):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        xq = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        xk = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        xv = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Apply rotary embeddings
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Compute attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, xv)
        
        # Output projection
        output = output.contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

### Advantages
- **Perfect extrapolation**: Works seamlessly with longer sequences
- **Relative awareness**: Naturally encodes relative positions
- **Efficiency**: No additional parameters beyond base model
- **Multiplicative**: Integrates smoothly with attention mechanism

## 📏 ALiBi (Attention with Linear Biases)

### Core Concept

ALiBi adds linear biases to attention scores based on key-query distance:

```
Attention(i,j) = softmax(q_i · k_j / √d_k - m · |i-j|)
```

Where `m` is a head-specific slope.

### Implementation

```python
class ALiBiAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Precompute slopes for each head
        self.register_buffer("slopes", self._get_slopes(num_heads))
        
    def _get_slopes(self, num_heads):
        """Generate slopes for ALiBi."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            # Handle non-power-of-2 heads
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0:num_heads-closest_power_of_2])
            return torch.tensor(slopes)
    
    def _get_alibi_bias(self, seq_len):
        """Generate ALiBi bias matrix."""
        # Create distance matrix
        context_position = torch.arange(seq_len)[:, None]
        memory_position = torch.arange(seq_len)[None, :]
        relative_position = memory_position - context_position
        
        # Apply slopes (different for each head)
        alibi_bias = relative_position[None, :, :] * self.slopes[:, None, None]
        
        return alibi_bias
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add ALiBi bias
        alibi_bias = self._get_alibi_bias(seq_len).to(scores.device)
        scores += alibi_bias
        
        # Apply attention
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

### Benefits
- **Extrapolation**: Excellent performance on longer sequences
- **Simplicity**: No additional parameters
- **Efficiency**: Linear bias computation
- **Monotonic**: Attention decreases with distance

## 📊 Comparison and Analysis

### Performance Characteristics

```python
class PositionalEncodingComparison:
    def __init__(self):
        self.methods = {
            'learned': LearnedPositionalEmbedding,
            'relative': RelativePositionalEncoding,
            'rope': RoPEAttention,
            'alibi': ALiBiAttention
        }
    
    def compare_extrapolation(self, train_length=512, test_lengths=[1024, 2048, 4096]):
        """Compare extrapolation capabilities."""
        results = {}
        
        for method_name in self.methods:
            results[method_name] = {}
            
            for test_len in test_lengths:
                extrapolation_ratio = test_len / train_length
                
                if method_name == 'learned':
                    # Learned embeddings fail beyond training length
                    performance = 0.0 if test_len > train_length else 1.0
                elif method_name == 'relative':
                    # Relative encoding with clipping
                    performance = max(0.3, 1.0 - (extrapolation_ratio - 1) * 0.2)
                elif method_name == 'rope':
                    # RoPE maintains performance
                    performance = max(0.8, 1.0 - (extrapolation_ratio - 1) * 0.1)
                elif method_name == 'alibi':
                    # ALiBi excellent extrapolation
                    performance = max(0.85, 1.0 - (extrapolation_ratio - 1) * 0.05)
                
                results[method_name][test_len] = performance
        
        return results
    
    def memory_complexity(self, seq_len, d_model, num_heads):
        """Compare memory requirements."""
        complexities = {
            'learned': seq_len * d_model,  # Embedding table
            'relative': (2 * 32 + 1) * (d_model // num_heads),  # Clipped relative
            'rope': 0,  # No additional parameters
            'alibi': 0   # No additional parameters
        }
        
        return complexities
```

### Use Case Recommendations

**Learned Positional Embeddings**
- ✅ Fixed-length sequences
- ✅ Task-specific optimization
- ❌ Length extrapolation needed

**Relative Positional Encoding**
- ✅ Moderate extrapolation
- ✅ Translation tasks
- ⚠️ Memory overhead with long sequences

**RoPE (Rotary Positional Embeddings)**
- ✅ Excellent extrapolation
- ✅ Autoregressive models
- ✅ Long-context applications

**ALiBi (Attention with Linear Biases)**
- ✅ Best extrapolation performance
- ✅ Simple implementation
- ✅ Memory efficient

## 🔬 Advanced Implementations

### Hybrid Positional Encoding

```python
class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, use_rope=True, use_alibi=True):
        super().__init__()
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        
        if use_rope:
            self.rope_freqs = precompute_freqs_cis(d_model // 2, max_seq_len * 2)
        
        if use_alibi:
            self.alibi_slopes = self._get_alibi_slopes(8)  # Assume 8 heads
    
    def forward(self, q, k, v, seq_len):
        if self.use_rope:
            q, k = apply_rotary_emb(q, k, self.rope_freqs[:seq_len])
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if self.use_alibi:
            alibi_bias = self._get_alibi_bias(seq_len)
            scores += alibi_bias
        
        return torch.softmax(scores, dim=-1)
```

### Dynamic Positional Encoding

```python
class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, methods=['rope', 'alibi']):
        super().__init__()
        self.methods = methods
        self.method_weights = nn.Parameter(torch.ones(len(methods)))
        
        # Initialize different methods
        if 'rope' in methods:
            self.rope_freqs = precompute_freqs_cis(d_model, 4096)
        if 'alibi' in methods:
            self.alibi_slopes = self._get_alibi_slopes(8)
    
    def forward(self, q, k, v, seq_len):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # Weighted combination of methods
        method_weights = torch.softmax(self.method_weights, dim=0)
        
        total_bias = 0
        for i, method in enumerate(self.methods):
            if method == 'rope':
                q_rope, k_rope = apply_rotary_emb(q, k, self.rope_freqs[:seq_len])
                rope_scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / math.sqrt(q.size(-1))
                total_bias += method_weights[i] * (rope_scores - scores)
            elif method == 'alibi':
                alibi_bias = self._get_alibi_bias(seq_len)
                total_bias += method_weights[i] * alibi_bias
        
        return torch.softmax(scores + total_bias, dim=-1)
```

## 📚 Summary

### Key Insights

**Learned Positional Embeddings**
- Simple but limited to training length
- Good for fixed-length applications

**Relative Positional Encoding**
- Better generalization than absolute positions
- Moderate extrapolation capabilities

**RoPE (Rotary Positional Embeddings)**
- Excellent extrapolation through rotation
- Widely adopted in modern LLMs

**ALiBi (Attention with Linear Biases)**
- Best extrapolation performance
- Simple and parameter-free

### Selection Criteria
- **Extrapolation needs**: RoPE or ALiBi
- **Memory constraints**: ALiBi or RoPE
- **Implementation simplicity**: ALiBi
- **Performance**: RoPE for autoregressive, ALiBi for bidirectional

### Future Directions
- **Adaptive methods**: Dynamic selection based on sequence properties
- **Hybrid approaches**: Combining multiple methods
- **Task-specific**: Optimizing for particular applications
- **Efficiency**: Further reducing computational overhead