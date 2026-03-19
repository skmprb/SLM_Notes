# Transformer Architecture — Encoder-Decoder (Step by Step)

We'll use one running example throughout all steps:

```
Task:       English → French translation
Source:     "I am a student"
Target:     "je suis un étudiant"
d_model:    512  (embedding dimension)
n_heads:    8    (number of attention heads)
vocab_size: 10000
```

---

# TRAINING PHASE

## Step 1. Input Text (Corpus Preparation)

**What:** Prepare sentence pairs — each pair has a source (input) and target (expected output).

**Example:**
```
corpus = [
    ("I am a student",    "je suis un étudiant"),
    ("hello",             "bonjour"),
    ("thank you",         "merci"),
]
```

**Output:** A list of (source_text, target_text) pairs ready for the pipeline.

---

## Step 2. Tokenization

**What:** Convert raw text into token IDs (numbers) that the model can process.
Each language has its own tokenizer with its own vocabulary.

**Special tokens:**
| Token   | Meaning              | Purpose                                      |
|---------|----------------------|----------------------------------------------|
| `<pad>` | Padding              | Makes all sequences the same length           |
| `<sos>` | Start of Sequence    | Tells the decoder "start generating"          |
| `<eos>` | End of Sequence      | Tells the model "sentence is complete"        |
| `<unk>` | Unknown              | Replaces words not in the vocabulary          |

**Example:**
```
Source: "I am a student"
  tokens:    ["I", "am", "a", "student"]
  token_ids: [12,   45,  7,   892]

Target: "je suis un étudiant"
  tokens:    ["je", "suis", "un", "étudiant"]
  token_ids: [15,    67,    23,   541]
```

**Output:**
- source_token_ids: `[12, 45, 7, 892]`
- target_token_ids: `[15, 67, 23, 541]`

---

## Step 3. Token Embedding

**What:** Convert each token ID into a dense vector of size `d_model` (512).
This is a learnable lookup table — each word gets its own 512-dimensional vector that captures its meaning.

**Example:**
```
token_id 12 ("I")       → [0.23, -0.45, 0.12, ..., 0.78]   (512 numbers)
token_id 45 ("am")      → [0.11, 0.33, -0.67, ..., 0.54]   (512 numbers)
token_id 7  ("a")       → [-0.09, 0.21, 0.44, ..., -0.31]  (512 numbers)
token_id 892 ("student") → [0.56, -0.12, 0.89, ..., 0.03]  (512 numbers)
```

**Output:**
- source_embedding: shape `(1, 4, 512)` → (batch_size, seq_len, d_model)
- target_embedding: shape `(1, 4, 512)`

**Note:** At this point, the word "student" has the same embedding regardless of where it appears in the sentence. Position info comes next.

---

## Step 4. Positional Encoding

**What:** Add position information to each embedding so the model knows the order of words.
Without this, "I am a student" and "student a am I" would look the same to the model.

**How:** Uses sine and cosine waves of different frequencies to create a unique pattern for each position.

**Example:**
```
Position 0 ("I"):       PE = [0.00, 1.00, 0.00, 1.00, ...]   (512 numbers)
Position 1 ("am"):      PE = [0.84, 0.54, 0.01, 1.00, ...]
Position 2 ("a"):       PE = [0.91, -0.42, 0.02, 0.99, ...]
Position 3 ("student"): PE = [0.14, -0.99, 0.03, 0.99, ...]

source_pos_embedding = source_embedding + positional_encoding
  "I"       → [0.23+0.00, -0.45+1.00, ...] = [0.23, 0.55, ...]
  "am"      → [0.11+0.84, 0.33+0.54, ...]  = [0.95, 0.87, ...]
```

**Output:**
- source_pos_embedding: shape `(1, 4, 512)` — same shape, but now contains position info
- target_pos_embedding: shape `(1, 4, 512)`

---

## Step 5. Encoder Stack

**What:** Processes the source sentence to build a deep understanding of it.
The encoder has N layers (e.g., 6), each with the same structure.

**Each encoder layer has:**
```
Source Input
    │
    ├─→ Multi-Head Self-Attention ─→ Add & LayerNorm
    │       (each word looks at ALL other source words
    │        to understand context)
    │
    └─→ Feed-Forward Network ──────→ Add & LayerNorm ─→ Output
            (processes each position independently
             to add non-linearity)
```

**Self-Attention example:**
```
Input: "I am a student"

When processing "am":
  Q (Query):  "am" asks — "who is relevant to me?"
  K (Key):    all words offer — "here's what I represent"
  V (Value):  all words offer — "here's my information"

  Attention scores:
    "am" → "I"       = 0.35  (high — "I" is the subject of "am")
    "am" → "am"      = 0.25
    "am" → "a"       = 0.10  (low — not very relevant)
    "am" → "student" = 0.30  (medium — "am" relates to "student")

  Result: "am" now carries context from all words, weighted by relevance
```

**Multi-Head:** Instead of one attention, we run 8 parallel attentions (heads), each focusing on different relationships (grammar, meaning, position, etc.), then combine them.

**Add & LayerNorm (Residual Connection):**
- Add: `output = input + attention_output` — preserves original information
- LayerNorm: normalizes values to prevent them from getting too large or small

**Output:**
- encoder_output: shape `(1, 4, 512)` — each word now contains context from the entire source sentence

---

## Step 6. Encoder Output (Memory)

**What:** The encoder output is stored as "memory" — this is the encoder's understanding of the source sentence.

**Example:**
```
memory = encoder_output    shape: (1, 4, 512)

  "I"       → [rich contextual vector knowing it's the subject]
  "am"      → [rich contextual vector knowing it's a verb linked to "I" and "student"]
  "a"       → [rich contextual vector knowing it's an article before "student"]
  "student" → [rich contextual vector knowing it's the object, linked to "I am"]
```

The decoder will use this memory to understand the source sentence while generating the translation.

---

## Step 7. Decoder Stack

**What:** Generates the target sentence using the memory from the encoder.
The decoder also has N layers (e.g., 6), but each layer has THREE sub-layers.

**Input:**
- target_pos_embedding (target words with position info)
- memory (encoder output)
- causal_mask (prevents looking at future target words)

**Each decoder layer has:**
```
Target Input
    │
    ├─→ Masked Self-Attention ────→ Add & LayerNorm
    │       (Q=K=V from target, future words MASKED)
    │       "What have I generated so far?"
    │
    ├─→ Cross-Attention ──────────→ Add & LayerNorm
    │       (Q from target, K & V from encoder memory)
    │       "What part of the source is relevant for my next word?"
    │
    └─→ Feed-Forward Network ─────→ Add & LayerNorm ─→ Output
```

### Causal Mask (prevents cheating)

During training, we feed the entire target sentence at once for speed. But the model should learn to predict left-to-right, so we mask future words.

**Example:** Target = `[<sos>, je, suis, un]`
```
              <sos>   je    suis    un
  <sos>    [   ✓      ✗      ✗      ✗  ]  ← can only see <sos>
  je       [   ✓      ✓      ✗      ✗  ]  ← can see <sos>, je
  suis     [   ✓      ✓      ✓      ✗  ]  ← can see <sos>, je, suis
  un       [   ✓      ✓      ✓      ✓  ]  ← can see everything before it

  ✓ = True (allowed to attend)
  ✗ = False (blocked — attention score set to -infinity → softmax gives 0)
```

**How the mask is built:**
```python
1. torch.ones(4, 4)                    # matrix of all 1s
2. torch.triu(..., diagonal=1)         # keep upper triangle (future positions) = 1, rest = 0
3. .bool()                             # convert to True/False
4. ~mask                               # flip — lower triangle = True (allowed), upper = False (blocked)
```

### Masked Self-Attention example:
```
When predicting position 2 (should predict "suis"):
  Can see: <sos>, je  (positions 0, 1)
  Cannot see: suis, un (positions 2, 3 — future)

  "je" attends to "<sos>" and itself → understands "I've started with je"
```

### Cross-Attention example:
```
When generating "suis" (French for "am"):
  Q (Query) comes from decoder: "suis" asks — "what source word should I focus on?"
  K, V come from encoder memory: all source words offer their information

  Attention scores:
    "suis" → "I"       = 0.15
    "suis" → "am"      = 0.60  ← highest! "suis" is the French translation of "am"
    "suis" → "a"       = 0.05
    "suis" → "student" = 0.20

  Result: "suis" pulls most information from "am" in the source
```

**Output:**
- decoder_output: shape `(1, 4, 512)` — contextual representation for each target position

---

## Step 8. Output Head (Linear + Softmax)

**What:** Converts the decoder output into a probability distribution over the entire target vocabulary.
This tells us: for each position, what is the probability of each word being the next word?

**How:**
1. Linear layer: projects from `d_model` (512) → `vocab_size` (10000) — these raw scores are called **logits**
2. Softmax (during inference): converts logits into probabilities that sum to 1

**Example:**
```
decoder_output for position 1: [0.56, -0.12, 0.89, ..., 0.03]  (512 dims)
                                        ↓ Linear layer
logits:                         [1.2, -0.5, 0.1, ..., 8.7, ..., -2.3]  (10000 dims)
                                                        ↑
                                              index 67 = "suis" has highest score

After softmax:
  P("je")       = 0.02
  P("suis")     = 0.85  ← highest probability
  P("un")       = 0.01
  P("étudiant") = 0.03
  ...all 10000 words sum to 1.0

Predicted token: argmax → index 67 → "suis" ✓
```

**Output:**
- logits: shape `(1, 4, 10000)` → (batch, target_seq_len, vocab_size)
- predicted_token_ids: `[15, 67, 23, 541]` → `["je", "suis", "un", "étudiant"]`

---

## Step 9. Loss Function (Cross-Entropy Loss)

**What:** Measures how wrong the model's predictions are compared to the correct answers.
The lower the loss, the better the model is predicting.

**How it works:**
- Takes the predicted probability distribution (logits) and the correct token ID
- Calculates: `loss = -log(probability of correct token)`
- If the model is confident and correct → low loss. If wrong → high loss.

**Example:**
```
Position 1 — correct answer: "suis" (token_id 67)

  If model predicts P("suis") = 0.85  → loss = -log(0.85) = 0.16  (low — good!)
  If model predicts P("suis") = 0.10  → loss = -log(0.10) = 2.30  (high — bad!)
  If model predicts P("suis") = 0.01  → loss = -log(0.01) = 4.60  (very high — terrible!)
```

**Special settings:**
- `ignore_index=PAD_IDX` — skip `<pad>` tokens in loss calculation (padding is not real content)
- `label_smoothing=0.1` — instead of saying the correct answer is 100% right, we say it's 90% right and spread 10% across other words. This prevents the model from becoming overconfident.

**Output:**
- loss: a single number (scalar), e.g., `2.45`

---

## Step 10. Backpropagation & Optimization

### Assembling the full model (EncoderDecoderTransformer)
All previous steps are connected into one model:
```
Source → Embedding → Positional Encoding → Encoder → Memory
                                                        ↓
Target → Embedding → Positional Encoding → Decoder (uses Memory) → Output Head → Logits
```

### Batch Preparation
Before training, we prepare the data:
```
Source:       "I am a student"
  src_ids:    [12, 45, 7, 892, <eos>]              ← add <eos> at end

Target Input (fed to decoder):
  tgt_input:  [<sos>, 15, 67, 23, 541]             ← add <sos> at start (teacher forcing)

Target Labels (correct answers to compare against):
  tgt_labels: [15, 67, 23, 541, <eos>]             ← add <eos> at end

Padding: shorter sentences get <pad> tokens so all sequences are the same length
  e.g., [12, 45, <eos>, <pad>, <pad>]
```

**Teacher forcing:** During training, we feed the correct previous words to the decoder (not its own predictions). This makes training faster and more stable.

### Optimizer & Loss
- **Adam optimizer:** decides how much to adjust each weight
    - `lr=1e-4` — learning rate (step size)
    - `betas=(0.9, 0.98)` — momentum settings tuned for transformers
- **CrossEntropyLoss:** measures prediction error (see Step 9)

### Training Loop
```
for each epoch (e.g., 100 rounds):

    1. Generate causal mask     → block future words in decoder
    2. Forward pass             → src & tgt go through the model → get logits
    3. Compute loss             → compare logits with correct answers
    4. optimizer.zero_grad()    → clear old gradients from previous round
    5. loss.backward()          → backpropagation: compute gradient for every weight
                                   (how much each weight contributed to the error)
    6. clip_grad_norm_(1.0)     → cap gradients to prevent exploding values
    7. optimizer.step()         → update weights: weight = weight - lr × gradient
```

**Example of loss decreasing over training:**
```
Epoch  20 | Loss: 2.1534   (model is mostly guessing)
Epoch  40 | Loss: 0.8721   (model is learning patterns)
Epoch  60 | Loss: 0.3245   (model is getting good)
Epoch  80 | Loss: 0.1102   (model is nearly correct)
Epoch 100 | Loss: 0.0451   (model has learned the translations)
```

**Output:**
- Model weights are updated and improved after each epoch
- Loss decreases over time → model is learning

---

## Step 11. Save Model Weights

**What:** Save everything needed to rebuild and use the model later.

**Saved files:**
```
trained_enc_dec_model/
  ├── config.json       → model settings (d_model, n_heads, n_layers, etc.)
  ├── model.pth         → all learned weights (the knowledge)
  ├── src_vocab.json    → source language vocabulary (word → id mapping)
  └── tgt_vocab.json    → target language vocabulary (word → id mapping)
```

---

# INFERENCE PHASE

## Step 12. Inference — Generate Translation Word by Word

**What:** Use the trained model to translate a new sentence. Unlike training, the model generates one word at a time.

### Load the saved model
```
1. Read config.json           → get model architecture settings
2. Rebuild tokenizers         → load src_vocab.json and tgt_vocab.json
3. Rebuild model + load weights → create model from config, load model.pth
4. model.eval()               → switch to inference mode (disables dropout)
```

### Greedy Decoding (token-by-token generation)

**How it works:**
```
Input: "I am a student"

Step 1: Tokenize source → [12, 45, 7, 892, <eos>]
Step 2: Encode ONCE     → memory (encoder's understanding of the source)
Step 3: Start decoder with [<sos>]

Step 4: Loop — generate one word at a time:

  Iteration 1:
    Decoder input: [<sos>]
    Decoder looks at memory + [<sos>] → predicts "je"
    Generated so far: [<sos>, je]

  Iteration 2:
    Decoder input: [<sos>, je]
    Decoder looks at memory + [<sos>, je] → predicts "suis"
    Generated so far: [<sos>, je, suis]

  Iteration 3:
    Decoder input: [<sos>, je, suis]
    Decoder looks at memory + [<sos>, je, suis] → predicts "un"
    Generated so far: [<sos>, je, suis, un]

  Iteration 4:
    Decoder input: [<sos>, je, suis, un]
    Decoder looks at memory + [<sos>, je, suis, un] → predicts "étudiant"
    Generated so far: [<sos>, je, suis, un, étudiant]

  Iteration 5:
    Decoder input: [<sos>, je, suis, un, étudiant]
    Decoder looks at memory + [...] → predicts <eos>
    STOP! Sentence is complete.

Step 5: Remove <sos>, convert IDs to words → "je suis un étudiant"
```

**"Greedy"** means: at each step, pick the word with the highest probability. Simple but not always optimal (beam search explores multiple paths for better results).

### Key difference: Training vs Inference
```
Training:    Feed the ENTIRE correct target at once    → fast, parallel
             [<sos>, je, suis, un, étudiant]             (teacher forcing)

Inference:   Feed only what's generated SO FAR          → slow, sequential
             [<sos>] → [<sos>, je] → [<sos>, je, suis] → ...
```

**Output:**
- The translated sentence: `"je suis un étudiant"`
