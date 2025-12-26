# LLM Variables & Parameters Reference

## Model Architecture Parameters

### Core Dimensions
- **`d_model`** - Model dimension (hidden size)
  - Typical values: 512, 768, 1024, 2048, 4096
  - Controls model capacity and memory usage

- **`n_heads`** - Number of attention heads
  - Typical values: 8, 12, 16, 32
  - Must divide `d_model` evenly

- **`n_layers`** - Number of transformer layers
  - Typical values: 6, 12, 24, 48, 96
  - Deeper models = more capacity but slower

- **`d_ff`** - Feed-forward dimension
  - Usually 4 × `d_model`
  - Controls FFN capacity

### Vocabulary & Context
- **`vocab_size`** - Size of token vocabulary
  - Typical values: 30K, 50K, 100K, 250K
  - Affects embedding layer size

- **`max_seq_len`** - Maximum sequence length
  - Typical values: 512, 1024, 2048, 4096, 8192
  - Determines context window

- **`block_size`** - Training sequence length
  - Usually same as `max_seq_len`
  - Affects memory usage during training

## Training Parameters

### Learning Rate
- **`learning_rate`** - Base learning rate
  - Typical values: 1e-4 to 1e-3 for pretraining
  - 1e-5 to 1e-4 for fine-tuning

- **`warmup_steps`** - Learning rate warmup period
  - Typical values: 1000-10000 steps
  - Prevents early training instability

- **`lr_decay`** - Learning rate decay factor
  - Cosine, linear, or exponential decay
  - Helps convergence

### Batch Processing
- **`batch_size`** - Training batch size
  - Typical values: 32, 64, 128, 256
  - Limited by GPU memory

- **`micro_batch_size`** - Gradient accumulation unit
  - Used with gradient accumulation
  - Actual batch size = micro_batch × accumulation_steps

- **`gradient_accumulation_steps`** - Steps before weight update
  - Simulates larger batch sizes
  - Typical values: 1, 2, 4, 8

### Optimization
- **`beta1`** - Adam optimizer momentum
  - Default: 0.9
  - Controls momentum for gradients

- **`beta2`** - Adam optimizer second moment
  - Default: 0.999
  - Controls momentum for squared gradients

- **`epsilon`** - Adam optimizer epsilon
  - Default: 1e-8
  - Prevents division by zero

- **`weight_decay`** - L2 regularization strength
  - Typical values: 0.01, 0.1
  - Prevents overfitting

### Gradient Control
- **`max_grad_norm`** - Gradient clipping threshold
  - Typical values: 1.0, 5.0
  - Prevents exploding gradients

## Inference Parameters

### Decoding Strategy
- **`temperature`** - Sampling randomness
  - Range: 0.1 to 2.0
  - Lower = more deterministic

- **`top_k`** - Top-k sampling parameter
  - Typical values: 10, 50, 100
  - 0 = disabled

- **`top_p`** - Nucleus sampling threshold
  - Range: 0.1 to 1.0
  - Typical: 0.9, 0.95

- **`repetition_penalty`** - Penalty for repeated tokens
  - Range: 1.0 to 1.2
  - 1.0 = no penalty

### Beam Search
- **`num_beams`** - Number of beams
  - Typical values: 1, 4, 8
  - 1 = greedy decoding

- **`length_penalty`** - Length normalization
  - Range: 0.5 to 2.0
  - Balances length vs quality

### Generation Control
- **`max_new_tokens`** - Maximum tokens to generate
  - Typical values: 50, 100, 512, 1024
  - Controls output length

- **`min_length`** - Minimum generation length
  - Prevents very short outputs
  - Usually much smaller than max

- **`do_sample`** - Enable sampling vs greedy
  - Boolean: True/False
  - True for creative tasks

## Attention Parameters

### Standard Attention
- **`d_k`** - Key/Query dimension
  - Usually `d_model / n_heads`
  - Affects attention computation

- **`d_v`** - Value dimension
  - Usually same as `d_k`
  - Can be different for efficiency

### Positional Encoding
- **`max_position_embeddings`** - Max position index
  - Usually same as `max_seq_len`
  - For learned positions

- **`rope_theta`** - RoPE base frequency
  - Default: 10000
  - Affects position encoding

### Attention Optimizations
- **`attention_dropout`** - Attention dropout rate
  - Typical values: 0.0, 0.1
  - Regularization for attention

- **`use_cache`** - Enable KV caching
  - Boolean for inference
  - Speeds up generation

## Regularization Parameters

### Dropout
- **`dropout`** - General dropout rate
  - Typical values: 0.0, 0.1, 0.2
  - Applied to various layers

- **`attention_dropout`** - Attention-specific dropout
  - Usually same as general dropout
  - Applied to attention weights

- **`hidden_dropout`** - Hidden layer dropout
  - Applied to FFN outputs
  - Prevents overfitting

### Layer Normalization
- **`layer_norm_eps`** - LayerNorm epsilon
  - Default: 1e-5 or 1e-6
  - Numerical stability

- **`use_rms_norm`** - Use RMSNorm instead of LayerNorm
  - Boolean parameter
  - More efficient normalization

## Memory & Efficiency

### Precision
- **`torch_dtype`** - Model precision
  - Options: float32, float16, bfloat16
  - Affects memory and speed

- **`use_flash_attention`** - Enable FlashAttention
  - Boolean parameter
  - Memory-efficient attention

### Quantization
- **`load_in_8bit`** - 8-bit quantization
  - Boolean parameter
  - Reduces memory usage

- **`load_in_4bit`** - 4-bit quantization
  - Even more memory efficient
  - Some quality loss

## Fine-tuning Parameters

### LoRA
- **`lora_r`** - LoRA rank
  - Typical values: 8, 16, 32, 64
  - Higher = more parameters

- **`lora_alpha`** - LoRA scaling factor
  - Usually 2 × `lora_r`
  - Controls adaptation strength

- **`lora_dropout`** - LoRA dropout rate
  - Typical values: 0.05, 0.1
  - Regularization for LoRA

### Training Control
- **`num_train_epochs`** - Number of training epochs
  - Typical values: 1, 3, 5
  - Depends on dataset size

- **`save_steps`** - Checkpoint saving frequency
  - Number of steps between saves
  - For recovery and evaluation

- **`eval_steps`** - Evaluation frequency
  - Steps between evaluations
  - Monitors training progress

## Environment Variables

### Hardware
- **`CUDA_VISIBLE_DEVICES`** - GPU selection
  - Comma-separated GPU IDs
  - Controls GPU usage

- **`OMP_NUM_THREADS`** - CPU threads
  - Number of CPU threads
  - Affects CPU performance

### Memory Management
- **`PYTORCH_CUDA_ALLOC_CONF`** - CUDA memory config
  - Memory allocation strategy
  - Helps with OOM issues

---

## Parameter Relationships

### Memory Usage
```
Model Memory ≈ (vocab_size × d_model + n_layers × d_model²) × precision
```

### Training Memory
```
Training Memory ≈ Model Memory × (3 + gradient_accumulation_steps)
```

### Attention Complexity
```
Attention Memory ≈ batch_size × n_heads × seq_len² × d_k
```

## Common Configurations

### Small Model (125M parameters)
```
d_model: 768
n_heads: 12
n_layers: 12
d_ff: 3072
vocab_size: 50257
max_seq_len: 1024
```

### Medium Model (350M parameters)
```
d_model: 1024
n_heads: 16
n_layers: 24
d_ff: 4096
vocab_size: 50257
max_seq_len: 2048
```

### Large Model (1.3B parameters)
```
d_model: 2048
n_heads: 32
n_layers: 24
d_ff: 8192
vocab_size: 50257
max_seq_len: 2048
```