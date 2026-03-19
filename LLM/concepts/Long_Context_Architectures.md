# Long-Context Architectures

## 🎯 Overview

Long-context architectures extend transformers' ability to process sequences far beyond traditional limits. These approaches combine memory mechanisms, recurrence, and hierarchical processing to handle contexts of tens of thousands of tokens efficiently.

## 🧠 Memory-Augmented Transformers

### Core Concept

Memory-augmented transformers add external memory banks to store and retrieve information across long sequences, enabling models to maintain context beyond their immediate attention window.

### External Memory Architecture

```python
class ExternalMemoryBank(nn.Module):
    def __init__(self, memory_size, memory_dim, num_heads=8):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Initialize memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_keys = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Attention for memory access
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        
        # Memory update mechanism
        self.update_gate = nn.Linear(memory_dim * 2, memory_dim)
        self.write_gate = nn.Linear(memory_dim, memory_size)
        
    def read_memory(self, query):
        """Read from memory using attention mechanism."""
        batch_size, seq_len, _ = query.shape
        
        # Project query
        Q = self.query_proj(query)  # [batch, seq_len, dim]
        K = self.key_proj(self.memory_keys).unsqueeze(0)  # [1, memory_size, dim]
        V = self.value_proj(self.memory).unsqueeze(0)  # [1, memory_size, dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.memory_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Read from memory
        memory_output = torch.matmul(attention_weights, V)
        
        return memory_output, attention_weights
    
    def write_memory(self, content, write_strength):
        """Write new content to memory."""
        batch_size = content.size(0)
        
        # Compute write weights
        write_weights = torch.softmax(self.write_gate(write_strength), dim=-1)
        
        # Update memory (simplified)
        for b in range(batch_size):
            # Weighted update of memory slots
            update_content = content[b].mean(dim=0)  # Average over sequence
            
            # Find memory slot to update (highest write weight)
            slot_idx = torch.argmax(write_weights[b])
            
            # Update memory with gating
            old_memory = self.memory[slot_idx]
            gate = torch.sigmoid(self.update_gate(torch.cat([old_memory, update_content])))
            self.memory.data[slot_idx] = gate * update_content + (1 - gate) * old_memory

class MemoryAugmentedTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, memory_size=1024):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # External memory
        self.memory_bank = ExternalMemoryBank(memory_size, d_model, num_heads)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MemoryAugmentedLayer(d_model, num_heads, self.memory_bank)
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Process through layers with memory access
        for layer in self.layers:
            x = layer(x)
        
        return self.layer_norm(x)

class MemoryAugmentedLayer(nn.Module):
    def __init__(self, d_model, num_heads, memory_bank):
        super().__init__()
        self.d_model = d_model
        self.memory_bank = memory_bank
        
        # Standard self-attention
        self.self_attention = nn.MultiheadAttention(d_model, num_heads)
        
        # Memory cross-attention
        self.memory_attention = nn.MultiheadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Gating for memory integration
        self.memory_gate = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Memory attention
        memory_content, memory_weights = self.memory_bank.read_memory(x)
        
        # Integrate memory content
        combined = torch.cat([x, memory_content], dim=-1)
        gate = torch.sigmoid(self.memory_gate(combined))
        x_with_memory = gate * memory_content + (1 - gate) * x
        x = self.norm2(x + x_with_memory)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        # Update memory with current content
        write_strength = torch.mean(x, dim=1)  # Average pooling for write signal
        self.memory_bank.write_memory(x, write_strength)
        
        return x
```

### Compressive Memory

```python
class CompressiveMemory(nn.Module):
    """Compressive memory that summarizes old information."""
    
    def __init__(self, d_model, memory_size, compression_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.compression_ratio = compression_ratio
        
        # Recent memory (full resolution)
        self.recent_memory = []
        self.max_recent = memory_size // compression_ratio
        
        # Compressed memory (lower resolution)
        self.compressed_memory = []
        self.max_compressed = memory_size - self.max_recent
        
        # Compression network
        self.compressor = nn.Sequential(
            nn.Linear(d_model * compression_ratio, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def add_memory(self, new_content):
        """Add new content to memory with compression."""
        # Add to recent memory
        self.recent_memory.append(new_content)
        
        # Check if recent memory is full
        if len(self.recent_memory) > self.max_recent:
            # Compress oldest recent memories
            to_compress = self.recent_memory[:self.compression_ratio]
            self.recent_memory = self.recent_memory[self.compression_ratio:]
            
            # Compress multiple items into one
            compressed_item = self.compress_memories(to_compress)
            self.compressed_memory.append(compressed_item)
            
            # Limit compressed memory size
            if len(self.compressed_memory) > self.max_compressed:
                self.compressed_memory.pop(0)
    
    def compress_memories(self, memories):
        """Compress multiple memory items into one."""
        # Concatenate memories
        concatenated = torch.cat(memories, dim=-1)
        
        # Apply compression network
        compressed = self.compressor(concatenated)
        
        return compressed
    
    def retrieve_memory(self, query, top_k=5):
        """Retrieve most relevant memories."""
        all_memories = self.recent_memory + self.compressed_memory
        
        if not all_memories:
            return None
        
        # Compute similarities
        similarities = []
        for memory in all_memories:
            sim = torch.cosine_similarity(query.mean(dim=1), memory.mean(dim=1))
            similarities.append(sim.mean().item())
        
        # Get top-k most similar
        top_indices = sorted(range(len(similarities)), 
                           key=lambda i: similarities[i], reverse=True)[:top_k]
        
        retrieved = [all_memories[i] for i in top_indices]
        return torch.cat(retrieved, dim=1) if retrieved else None
```

## 🔄 Recurrence in Transformers

### Recurrent Memory Transformer

```python
class RecurrentMemoryTransformer(nn.Module):
    """Transformer with recurrent memory mechanism."""
    
    def __init__(self, d_model, num_heads, num_layers, segment_length=512):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.segment_length = segment_length
        
        # Transformer layers
        self.layers = nn.ModuleList([
            RecurrentTransformerLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        # Memory states for each layer
        self.memory_states = [None] * num_layers
    
    def forward(self, x, reset_memory=False):
        """Process input with recurrent memory."""
        if reset_memory:
            self.memory_states = [None] * self.num_layers
        
        # Process in segments
        seq_len = x.size(1)
        outputs = []
        
        for start in range(0, seq_len, self.segment_length):
            end = min(start + self.segment_length, seq_len)
            segment = x[:, start:end]
            
            # Process segment through layers
            for i, layer in enumerate(self.layers):
                segment, new_memory = layer(segment, self.memory_states[i])
                self.memory_states[i] = new_memory
            
            outputs.append(segment)
        
        return torch.cat(outputs, dim=1)

class RecurrentTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, memory_length=256):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        
        # Attention mechanisms
        self.self_attention = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Memory update gate
        self.memory_gate = nn.Linear(d_model, d_model)
    
    def forward(self, x, memory=None):
        batch_size, seq_len, _ = x.shape
        
        # Prepare extended context with memory
        if memory is not None:
            extended_context = torch.cat([memory, x], dim=1)
        else:
            extended_context = x
        
        # Self-attention over extended context
        attn_out, _ = self.self_attention(x, extended_context, extended_context)
        x = self.norm1(x + attn_out)
        
        # Cross-attention with memory (if available)
        if memory is not None:
            cross_attn_out, _ = self.cross_attention(x, memory, memory)
            x = self.norm2(x + cross_attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        # Update memory
        new_memory = self.update_memory(x, memory)
        
        return x, new_memory
    
    def update_memory(self, current_states, old_memory):
        """Update memory with current states."""
        # Use gating to control memory update
        gate = torch.sigmoid(self.memory_gate(current_states))
        
        # Select states to add to memory (e.g., last few tokens)
        memory_candidates = current_states[:, -self.memory_length//4:]
        
        if old_memory is not None:
            # Combine old memory with new candidates
            combined_memory = torch.cat([old_memory, memory_candidates], dim=1)
            
            # Truncate to memory length
            if combined_memory.size(1) > self.memory_length:
                # Keep most recent memories
                new_memory = combined_memory[:, -self.memory_length:]
            else:
                new_memory = combined_memory
        else:
            new_memory = memory_candidates
        
        return new_memory
```

### Transformer-XL Style Recurrence

```python
class TransformerXLLayer(nn.Module):
    """Transformer-XL layer with segment-level recurrence."""
    
    def __init__(self, d_model, num_heads, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention with relative positioning
        self.attention = RelativeMultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, memory=None):
        # Concatenate memory with current input
        if memory is not None:
            cat_input = torch.cat([memory, x], dim=1)
        else:
            cat_input = x
        
        # Self-attention
        attn_out = self.attention(x, cat_input, cat_input)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class TransformerXL(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 segment_length=512, memory_length=512):
        super().__init__()
        self.segment_length = segment_length
        self.memory_length = memory_length
        self.num_layers = num_layers
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerXLLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Memory for each layer
        self.memories = [None] * num_layers
    
    def forward(self, input_ids, reset_memory=False):
        if reset_memory:
            self.memories = [None] * self.num_layers
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Process through layers with memory
        new_memories = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x, self.memories[i])
            
            # Update memory for this layer
            if self.memories[i] is not None:
                new_memory = torch.cat([self.memories[i], x], dim=1)
            else:
                new_memory = x
            
            # Truncate memory to maximum length
            if new_memory.size(1) > self.memory_length:
                new_memory = new_memory[:, -self.memory_length:].detach()
            
            new_memories.append(new_memory)
        
        # Update memories
        self.memories = new_memories
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
```

## 🏗️ Hierarchical Attention

### Multi-Scale Attention

```python
class HierarchicalAttention(nn.Module):
    """Hierarchical attention operating at multiple scales."""
    
    def __init__(self, d_model, num_heads, scales=[1, 4, 16]):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        self.num_heads = num_heads
        
        # Attention layers for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads // len(scales))
            for _ in scales
        ])
        
        # Pooling layers for downsampling
        self.pooling_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=scale, stride=scale)
            for scale in scales[1:]  # Skip scale=1
        ])
        
        # Upsampling layers
        self.upsampling_layers = nn.ModuleList([
            nn.ConvTranspose1d(d_model, d_model, kernel_size=scale, stride=scale)
            for scale in scales[1:]
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * len(scales), d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                # Full resolution attention
                attn_out, _ = self.scale_attentions[i](x, x, x)
                scale_outputs.append(attn_out)
            else:
                # Downsample, attend, upsample
                x_pooled = self.pooling_layers[i-1](x.transpose(1, 2)).transpose(1, 2)
                
                # Attention at reduced resolution
                attn_out, _ = self.scale_attentions[i](x_pooled, x_pooled, x_pooled)
                
                # Upsample back to original resolution
                upsampled = self.upsampling_layers[i-1](attn_out.transpose(1, 2)).transpose(1, 2)
                
                # Ensure correct sequence length
                if upsampled.size(1) != seq_len:
                    upsampled = F.interpolate(
                        upsampled.transpose(1, 2), 
                        size=seq_len, 
                        mode='linear'
                    ).transpose(1, 2)
                
                scale_outputs.append(upsampled)
        
        # Fuse multi-scale outputs
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.fusion(fused)
        
        return output

class PyramidTransformer(nn.Module):
    """Pyramid-style transformer with hierarchical processing."""
    
    def __init__(self, d_model, num_heads, num_layers, pyramid_levels=3):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        
        # Pyramid layers (coarse to fine)
        self.pyramid_layers = nn.ModuleList()
        
        for level in range(pyramid_levels):
            layer_d_model = d_model // (2 ** level)  # Reduce dimension at higher levels
            layer = nn.ModuleList([
                HierarchicalTransformerLayer(layer_d_model, num_heads, level)
                for _ in range(num_layers // pyramid_levels)
            ])
            self.pyramid_layers.append(layer)
        
        # Cross-level connections
        self.cross_level_projections = nn.ModuleList([
            nn.Linear(d_model // (2 ** level), d_model // (2 ** (level + 1)))
            for level in range(pyramid_levels - 1)
        ])
        
        # Final fusion
        self.final_projection = nn.Linear(
            sum(d_model // (2 ** level) for level in range(pyramid_levels)),
            d_model
        )
    
    def forward(self, x):
        pyramid_outputs = []
        current_input = x
        
        # Process through pyramid levels
        for level, layers in enumerate(self.pyramid_layers):
            # Process at current level
            level_output = current_input
            for layer in layers:
                level_output = layer(level_output)
            
            pyramid_outputs.append(level_output)
            
            # Prepare input for next level (downsample and project)
            if level < len(self.pyramid_layers) - 1:
                # Downsample by factor of 2
                downsampled = F.avg_pool1d(
                    level_output.transpose(1, 2), 
                    kernel_size=2, 
                    stride=2
                ).transpose(1, 2)
                
                # Project to lower dimension
                current_input = self.cross_level_projections[level](downsampled)
        
        # Upsample and fuse all levels
        target_seq_len = x.size(1)
        upsampled_outputs = []
        
        for level, output in enumerate(pyramid_outputs):
            if output.size(1) != target_seq_len:
                # Upsample to target length
                upsampled = F.interpolate(
                    output.transpose(1, 2),
                    size=target_seq_len,
                    mode='linear'
                ).transpose(1, 2)
            else:
                upsampled = output
            
            upsampled_outputs.append(upsampled)
        
        # Fuse all levels
        fused = torch.cat(upsampled_outputs, dim=-1)
        final_output = self.final_projection(fused)
        
        return final_output

class HierarchicalTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, level):
        super().__init__()
        self.level = level
        
        # Attention with level-specific modifications
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

### Longformer-Style Attention

```python
class LongformerAttention(nn.Module):
    """Longformer-style attention with local + global patterns."""
    
    def __init__(self, d_model, num_heads, window_size=512, num_global_tokens=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        
        # Standard projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, global_token_ids=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create attention mask
        attention_mask = self.create_longformer_mask(seq_len, global_token_ids)
        attention_mask = attention_mask.to(x.device)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and output
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
    
    def create_longformer_mask(self, seq_len, global_token_ids=None):
        """Create Longformer attention mask with local + global patterns."""
        mask = torch.zeros(seq_len, seq_len)
        
        # Local attention windows
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        
        # Global attention
        if global_token_ids is not None:
            for global_id in global_token_ids:
                if global_id < seq_len:
                    # Global tokens attend to all positions
                    mask[global_id, :] = 1
                    # All positions attend to global tokens
                    mask[:, global_id] = 1
        else:
            # Default: first few tokens are global
            for i in range(min(self.num_global_tokens, seq_len)):
                mask[i, :] = 1
                mask[:, i] = 1
        
        return mask
```

## 📊 Performance Analysis

### Complexity Comparison

```python
class LongContextComplexityAnalyzer:
    def analyze_complexity(self, seq_len, d_model, memory_size=1024, window_size=512):
        """Analyze complexity of different long-context methods."""
        
        results = {
            'standard_transformer': {
                'memory': seq_len ** 2,
                'compute': seq_len ** 2 * d_model
            },
            'memory_augmented': {
                'memory': seq_len * d_model + memory_size * d_model,
                'compute': seq_len * memory_size * d_model + seq_len ** 2 * d_model
            },
            'recurrent_transformer': {
                'memory': seq_len * d_model + memory_size * d_model,
                'compute': seq_len * d_model ** 2  # Linear in sequence length
            },
            'hierarchical_attention': {
                'memory': seq_len * d_model * 3,  # Multiple scales
                'compute': seq_len * d_model * 3  # Approximately
            },
            'longformer': {
                'memory': seq_len * window_size,
                'compute': seq_len * window_size * d_model
            }
        }
        
        return results
```

### Use Case Recommendations

**Memory-Augmented Transformers**
- ✅ Need to maintain long-term context
- ✅ Document-level understanding
- ⚠️ Additional memory overhead

**Recurrent Transformers**
- ✅ Streaming applications
- ✅ Very long sequences
- ✅ Constant memory usage

**Hierarchical Attention**
- ✅ Multi-scale patterns
- ✅ Document structure awareness
- ⚠️ Implementation complexity

**Longformer-Style**
- ✅ Local + global attention needs
- ✅ Document processing
- ✅ Good balance of efficiency and performance

## 📚 Summary

### Key Innovations

**Memory-Augmented Transformers**
- External memory banks for long-term storage
- Attention-based memory access
- Compressive memory for efficiency

**Recurrence in Transformers**
- Segment-level processing with memory
- Transformer-XL style recurrence
- Constant memory complexity

**Hierarchical Attention**
- Multi-scale processing
- Pyramid architectures
- Local + global attention patterns

### Selection Criteria
- **Sequence length**: Primary factor in architecture choice
- **Memory constraints**: Available computational resources
- **Pattern types**: Local vs. global dependencies
- **Streaming requirements**: Real-time vs. batch processing

### Future Directions
- **Adaptive architectures**: Dynamic selection of attention patterns
- **Learned hierarchies**: AI-discovered multi-scale structures
- **Efficient memory**: Better compression and retrieval mechanisms
- **Hardware optimization**: Co-design with specialized accelerators

Long-context architectures represent the frontier of scaling transformer models to handle increasingly complex and lengthy inputs while maintaining computational efficiency.