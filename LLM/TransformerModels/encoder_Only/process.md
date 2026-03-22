# Encoder-Only

## Step 1. Text Corpus
- Input:
    - Pre-training: Unlabeled raw text for learning language structure via **Masked Language Modeling (MLM)** — predict masked words from context
    - Fine-tuning: **Labeled data as (text, label) pairs** for classification tasks (e.g., sentiment: "I love this movie" → positive)
- Output:
    - Text data with labels: `[(text, label), ...]` where label is 0=negative, 1=positive etc.

## Step 2. Tokenization
Tokenization — Building Vocabulary & Converting Text to IDs with BERT-style special tokens
- Input:
    - Raw text data
- Process:
    - Build vocabulary mapping unique words → integer IDs
    - Wrap each sentence with special tokens: `[CLS] tokens... [SEP]`
    - `[CLS]` (ID=1): Classification token at position 0 — its output becomes the **sentence-level representation**
    - `[SEP]` (ID=2): Separator / end of sentence marker
    - `[PAD]` (ID=0): Padding shorter sequences to equal length
    - `[UNK]` (ID=3): Unknown / out-of-vocabulary token
    - `[MASK]` (ID=4): Used in MLM pre-training to mask tokens
    - Single vocabulary (not separate src/tgt like encoder-decoder)
    - No `<sos>` token — encoder-only doesn't generate tokens
- Output:
    - Vocabulary: A mapping of unique tokens to their corresponding IDs (word2idx, idx2word)
    - Tokenized Text: `[CLS_IDX, token_id1, token_id2, ..., SEP_IDX]`

## Step 3. Embedding
Embedding — Converting Token IDs to Scaled Dense Vectors
- Input:
    - Tokenized text as a sequence of token IDs: `(batch, seq_len)`
- Process:
    - Learnable lookup table: `nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)`
    - Scaled by `√(d_model)` to balance magnitude with positional encoding that gets added next
    - Only **one** embedding layer needed (single vocabulary, unlike encoder-decoder which needs two)
- Output:
    - Dense vectors: `(batch, seq_len, d_model)` — each token is now a dense vector (e.g., 256 dims)

## Step 4. Positional Encoding
Positional Encoding — Injecting Order Information via Sinusoidal Functions
- Input:
    - Token embeddings: `(batch, seq_len, d_model)`
- Process:
    - Uses sinusoidal functions: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(...)`
    - **Added** to embeddings (not concatenated)
    - Not learned — fixed sinusoidal pattern (BERT uses learned PE, but concept is the same)
    - Key: **No causal mask** — encoder-only is **BIDIRECTIONAL** (every token sees every other token)
    - This is the key advantage over decoder-only: `[CLS]` can attend to the entire sentence in both directions
- Output:
    - Position-aware embeddings: `(batch, seq_len, d_model)` — now contains both semantic meaning and position info

## Step 5. Encoder Stack
A stack of N identical layers (e.g., N=4), each containing:
1. **Multi-Head Self-Attention** — every token attends to every other token (bidirectional, no causal mask!)
2. **Feed-Forward Network** — position-wise non-linear transformation (Linear → ReLU → Linear)
3. **Residual connections + Layer Normalization** around each sub-layer

- Input:
    - Position-aware embeddings: `(batch, seq_len, d_model)`
    - Optional padding mask: ensures padded tokens don't affect attention scores
- Process:
    - Each layer: `src → Self-Attention → Add & Norm → FFN → Add & Norm → output`
    - **No causal mask** → fully bidirectional attention (vs decoder which is causal/masked)
    - Token 3 sees tokens 1,2,3,4,5... (decoder-only would only see 1,2,3)
    - `[CLS]` at position 0 aggregates information from ALL tokens via self-attention
- Output:
    - Contextualized embeddings: `(batch, seq_len, d_model)` — same shape but deeply contextualized

## Step 6. Training Head
Classification Head — Takes `[CLS]` output and predicts class
- Input:
    - Encoder output: `(batch, seq_len, d_model)`
- Process:
    - Extract **only `[CLS]` token output** at position 0: `encoder_output[:, 0, :]` → `(batch, d_model)`
    - Pass through head: `Dropout → Linear(d_model, d_model) → Tanh → Dropout → Linear(d_model, n_classes)`
    - `[CLS]` has attended to every token bidirectionally → it represents the whole sentence
    - Head is simple — the encoder does the heavy lifting
    - Other tasks use different heads: NER → Linear on all tokens, MLM → Linear(d_model, vocab_size) on masked tokens
- Output:
    - Logits: `(batch, n_classes)` — one prediction per input sentence (e.g., 2 classes for positive/negative)

## Step 7. Loss
Cross-Entropy Loss — Measuring Classification Error
- Input:
    - Logits from training head: `(batch, n_classes)`
    - True labels: `(batch,)` — integer class IDs
- Process:
    - `nn.CrossEntropyLoss()` — computes `-log(probability of correct class)`
    - Much simpler than seq2seq loss — one prediction per sentence, not per token
    - No `ignore_index` needed (no padding in labels, unlike encoder-decoder)
    - If model is confident and correct → low loss. If wrong → high loss
- Output:
    - Loss value: single scalar measuring how wrong the predictions are

## Step 8. Backpropagation & Training
Full Training Loop — Forward → Loss → Backward → Update
- Input:
    - Loss value from Step 7
    - Full assembled model: `Embedding + PositionalEncoding + Encoder + ClassificationHead`
- Process:
    - `logits = model(input_batch, pad_mask)` — forward pass
    - `loss = criterion(logits, label_batch)` — compute loss
    - `optimizer.zero_grad()` — reset gradients
    - `loss.backward()` — compute gradients via backpropagation
    - `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — gradient clipping to prevent exploding gradients
    - `optimizer.step()` — update weights (Adam optimizer, lr=1e-3)
    - **Padding mask** ensures padded tokens don't affect attention scores
- Output:
    - Updated model parameters — accuracy should reach ~100% on small datasets

## Step 9. Saved Weights
Saving the Trained Model to Disk
- Input:
    - Trained model parameters
- Process:
    - Save **4 files**:
        1. `model.pth` — model weights via `torch.save(model.state_dict())`
        2. `config.json` — architecture config (vocab_size, d_model, n_heads, n_layers, d_ff, n_classes, dropout)
        3. `vocab.json` — tokenizer vocabulary (word2idx mapping)
        4. `labels.json` — label mapping (e.g., {0: "negative", 1: "positive"})
    - Complete model = **Weights + Config + Vocab + Labels**
- Output:
    - Saved model directory with all 4 files — ready for inference without retraining

## Step 10. Inference
Classification & Sentence Embeddings — Single Forward Pass
- Input:
    - Saved model weights loaded from disk
    - New unseen text data
- Process:
    - `model.eval()` + `@torch.no_grad()` — disables dropout, skips gradient computation
    - Rebuild tokenizer from saved vocab, rebuild model from saved config, load weights
    - **No autoregressive loop!** Unlike decoder models, encoder-only runs **once** per input
    - **Task 1 — Classification**: tokenize → forward pass → `[CLS]` → head → softmax → predicted class + confidence
    - **Task 2 — Sentence Embeddings**: tokenize → forward pass → `[CLS]` output directly → use for cosine similarity, retrieval, clustering
- Output:
    - Classification: predicted label with confidence score (e.g., "positive", 99.99%)
    - Embeddings: `(batch, d_model)` vectors for similarity/retrieval tasks
