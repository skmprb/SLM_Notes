# 🛠️ Tokenization Implementation Parameters

## Core Tokenizer Parameters

### 1. **Vocabulary Size (`vocab_size`)**
```python
vocab_size = 30000  # Number of unique tokens
```
- **Range**: 1K - 200K+
- **Common values**: 30K-50K for most LLMs
- **Impact**: Memory usage, model size, coverage

### 2. **Special Tokens**
```python
special_tokens = {
    "pad_token": "[PAD]",      # Padding sequences
    "unk_token": "[UNK]",      # Unknown words
    "cls_token": "[CLS]",      # Classification token
    "sep_token": "[SEP]",      # Separator token
    "mask_token": "[MASK]",    # Masked language modeling
    "bos_token": "<s>",        # Beginning of sequence
    "eos_token": "</s>"        # End of sequence
}
```

### 3. **Subword Algorithm Parameters**

#### **BPE (Byte Pair Encoding)**
```python
bpe_params = {
    "merges": 30000,           # Number of merge operations
    "min_frequency": 2,        # Minimum token frequency
    "end_of_word_suffix": "</w>",
    "dropout": 0.1             # BPE dropout for robustness
}
```

#### **SentencePiece**
```python
sentencepiece_params = {
    "model_type": "bpe",       # bpe, unigram, char, word
    "vocab_size": 32000,
    "character_coverage": 0.9995,
    "split_by_whitespace": True,
    "byte_fallback": True,
    "normalization_rule_name": "nmt_nfkc"
}
```

#### **WordPiece (BERT-style)**
```python
wordpiece_params = {
    "vocab_size": 30522,
    "max_input_chars_per_word": 100,
    "continuing_subword_prefix": "##",
    "do_lower_case": True
}
```

### 4. **Sequence Processing Parameters**
```python
sequence_params = {
    "max_length": 512,         # Maximum sequence length
    "truncation": True,        # Truncate long sequences
    "padding": "max_length",   # Padding strategy
    "return_attention_mask": True,
    "return_token_type_ids": True
}
```

### 5. **Text Preprocessing Parameters**
```python
preprocessing_params = {
    "do_lower_case": False,    # Convert to lowercase
    "strip_accents": None,     # Remove accents
    "clean_text": True,        # Clean invisible chars
    "handle_chinese_chars": True,
    "tokenize_chinese_chars": True
}
```

## Implementation Examples

### **Hugging Face Transformers**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    vocab_size=30522,
    max_length=512,
    padding="max_length",
    truncation=True,
    do_lower_case=True
)
```

### **Custom BPE Training**
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

### **SentencePiece Training**
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=0.9995,
    split_by_whitespace=True,
    byte_fallback=True
)
```

## Key Configuration Decisions

### **For Different Use Cases:**

| Use Case | Vocab Size | Algorithm | Special Considerations |
|----------|------------|-----------|----------------------|
| **English NLP** | 30K-50K | WordPiece/BPE | Standard special tokens |
| **Multilingual** | 50K-100K | SentencePiece | High character coverage |
| **Code + Text** | 50K+ | BPE/Byte-level | Preserve code syntax |
| **Domain-specific** | 20K-40K | Custom BPE | Domain vocabulary |
| **Memory-constrained** | 10K-20K | Aggressive BPE | Accept longer sequences |

### **Performance vs. Quality Trade-offs:**
```python
# High quality, more memory
config_quality = {
    "vocab_size": 50000,
    "character_coverage": 0.9999,
    "model_type": "bpe"
}

# Balanced approach
config_balanced = {
    "vocab_size": 32000,
    "character_coverage": 0.9995,
    "model_type": "bpe"
}

# Memory efficient
config_efficient = {
    "vocab_size": 16000,
    "character_coverage": 0.995,
    "model_type": "bpe"
}
```