# Decoder-Only Transformer — Concepts Reference

Every concept used in the decoder-only transformer pipeline, explained with intuition.

---

## 1. Pre-training Objective

### Causal Language Modeling (CLM)
- **Story**: Imagine you're reading a book one word at a time, and after each word you try to guess the next word. You can only use the words you've already read — no peeking ahead. "The cat sat on the..." — you guess "mat". That's exactly what CLM does. The model reads left-to-right and predicts the next token at every position using only the past.
- **What**: The pre-training objective for decoder-only models. Given tokens `[t1, t2, t3]`, predict `[t2, t3, t4]`. Every token is both input AND label — just shifted by one position.
- **Example**: Input: `[the, cat, sat, on, the, mat]` → Target: `[cat, sat, on, the, mat, <eos>]`
- **Why it works**: The text supervises itself — no labels needed. This is why GPT can be trained on massive unlabeled internet text.
- **Used by**: GPT, GPT-2, GPT-3, LLaMA, Mistral, Falcon
- **Contrast with MLM**: Encoder-only models use Masked Language Modeling (predict masked tokens using both left AND right context). CLM is unidirectional (left-to-right only).

### Self-Supervised Learning
- **Story**: A child learns language just by listening to people talk — nobody gives them flashcards with labels. They hear "the cat sat on the..." thousands of times and learn to predict "mat". Decoder-only models learn the same way — raw text is the only teacher. The supervision comes from the text itself (shifted by 1), not from human-provided labels.
- **What**: Training where the labels come from the data itself, not from human annotation. In CLM, the target is just the input shifted by one position.
- **Why**: Unlabeled text is virtually unlimited (internet, books, code). Labeled data is expensive and scarce.

### Teacher Forcing (Training vs Inference)
- **Story**: A driving instructor sits next to a student. During training, the instructor keeps the car on the road even when the student makes mistakes — the student always sees the correct road ahead (teacher forcing). During the actual driving test, the student is alone — if they turn wrong, they keep going from that wrong turn. In training, the model always sees the real tokens. In inference, it sees its own (possibly wrong) predictions.
- **What**: During training, the model receives the **actual** previous tokens as input (not its own predictions). During inference, it feeds its own generated tokens back as input.
- **Why**: Training is stable and parallelizable — all positions computed at once. Inference is sequential — one token at a time.

---

## 2. Tokenization Concepts

### Vocabulary (word2idx / idx2word)
- **Story**: Think of a phone contacts list. Every person has a name ("Alice") and a number (42). When you want to call Alice, you look up her number. When someone calls you from number 42, you look up the name. That's exactly what vocabulary does — it's a two-way phone book between words and numbers.
- **What**: A bidirectional mapping between words and integer IDs. `word2idx = {"the": 3, "cat": 4}`, `idx2word = {3: "the", 4: "cat"}`.
- **Why**: Neural networks operate on numbers. The vocabulary is the bridge between human-readable text and model-readable integers.

### Special Tokens (Decoder-Only)
- **Story**: When you write a letter, you add a period at the end to mark "I'm done." Special tokens are like those markers — they tell the model where things start, end, or need special treatment. Decoder-only is simpler than encoder-only — no `[CLS]` or `[SEP]` needed because the model generates, it doesn't classify.

| Token | ID | Purpose |
|-------|----|---------|
| `<pad>` | 0 | Pads shorter sequences to equal length in a batch |
| `<eos>` | 1 | End of sequence — model learns to stop generating here |
| `<unk>` | 2 | Replaces out-of-vocabulary words |

- **No `[CLS]`**: Decoder-only doesn't classify — it generates. No need for a sentence-level summary token.
- **No `[SEP]`**: No sentence pairs to separate.
- **No `[MASK]`**: No masked language modeling — CLM predicts the next token, not a masked one.

### Input/Target Shifting
- **Story**: Imagine a conveyor belt of words. You take a photo of the belt (that's your input). Then you shift the belt one position to the left and take another photo (that's your target). The two photos are almost identical — just offset by one. That's how decoder-only creates training data from raw text — no labels needed, just shift.
- **What**: For a sentence `[the, cat, sat, on, the, mat, <eos>]`:
  - Input:  `[the, cat, sat, on, the, mat]` ← everything except last
  - Target: `[cat, sat, on, the, mat, <eos>]` ← everything except first
- **At position i**: The model sees tokens `0..i` and predicts the single token at `i+1`.
- **Why**: This creates supervised training data from raw text — the text supervises itself.

### Padding
- **Story**: You're packing boxes for shipping. All boxes must be the same size to fit on the truck. Some items are small, so you fill the extra space with bubble wrap. Padding is the bubble wrap — it fills shorter sentences with `<pad>` tokens so all sentences in a batch are the same length. The model knows to ignore the bubble wrap.
- **What**: Adding `<pad>` tokens to shorter sequences so all sequences in a batch have the same length.
- **Why**: GPUs process batches in parallel — all tensors in a batch must have the same shape.
- **ignore_index=PAD_IDX**: Tells CrossEntropyLoss to skip padded positions when computing loss.

### BPE (Byte Pair Encoding) / WordPiece
- **Story**: Imagine you're playing Scrabble and you don't have the tiles for "unhappiness". But you DO have tiles for "un", "happi", and "ness". So you build the word from smaller pieces. That's what BPE does — it breaks unknown words into smaller known pieces so the model never has to say "I don't know this word."
- **What**: Sub-word tokenization algorithms used in production models. GPT-2 uses BPE, LLaMA uses SentencePiece.
- **Our notebook**: Uses simple word-level tokenization for clarity. Production models use sub-word tokenization.


---

## 3. Embedding Concepts

### Dense Vector
- **Story**: Imagine describing a person with just 3 numbers: height, weight, age. Those 3 numbers capture a lot about the person in a compact way. A dense vector does the same for words — it describes a word using 256 (or 768) numbers. Words with similar meanings get similar numbers, so "cat" and "dog" end up close together, while "cat" and "flew" are far apart.
- **What**: A fixed-size array of floating-point numbers (e.g., 256 dimensions) that represents a token's meaning in continuous space.
- **Why**: Unlike one-hot vectors (sparse, high-dimensional, no relationships), dense vectors are compact and capture semantic similarity.
- **Shape**: `(batch_size, seq_len, d_model)` — e.g., `(1, 6, 256)` means 1 sentence, 6 tokens, 256 dimensions per token.

### d_model
- **Story**: Think of d_model as the "vocabulary" the model uses to describe each word internally. If d_model=3, the model can only describe words using 3 numbers — very limited. If d_model=768, it has 768 numbers to capture subtle differences. More dimensions = more nuance, like describing a color with just "red/blue/green" vs using the full RGB spectrum.
- **What**: The dimensionality of the model's internal representations. Every vector flowing through the transformer has this size.
- **Values**: GPT-2 small = 768, GPT-2 medium = 1024, our notebook = 256 (for demo).

### nn.Embedding
- **Story**: Picture a giant spreadsheet with one row per word in your vocabulary. Row 4 is "cat" and contains 256 numbers that represent what "cat" means. When the model sees token ID 4, it just looks up row 4 and grabs those 256 numbers. During training, the model adjusts these numbers to make them more useful. That spreadsheet IS the embedding layer.
- **What**: PyTorch's learnable lookup table. Maps integer IDs to dense vectors. `nn.Embedding(vocab_size=20, d_model=256)` creates a 20×256 matrix.
- **padding_idx=0**: Ensures the `<pad>` token always maps to a zero vector (not learned).
- **How it works**: `embedding[token_id]` = row `token_id` from the weight matrix. Gradients flow back to update these rows during training.

### √d_model Scaling
- **Story**: Imagine two people talking — one whispering (embedding) and one shouting (positional encoding). You can't hear the whisperer. Scaling by √d_model is like giving the whisperer a microphone so both voices are equally loud. This way, the model can hear both the word's meaning AND its position clearly.
- **What**: Embeddings are multiplied by `√d_model` (e.g., √256 = 16) before adding positional encoding.
- **Why**: Embedding values are typically small (initialized near 0). Positional encoding values are in [-1, 1]. Without scaling, positional encoding would dominate.

---

## 4. Positional Encoding Concepts

### Sinusoidal Positional Encoding
- **Story**: Imagine a row of students sitting in a classroom. The teacher can see all of them at once but can't tell who's sitting where. So each student holds up a unique clock showing a different time. Student 1 shows 1:00, student 2 shows 2:00, etc. Now the teacher knows the order. Sinusoidal PE does the same — it gives each position a unique "wave pattern" so the model knows which word came first, second, third.
- **What**: A fixed (non-learned) signal added to embeddings that encodes position using sine and cosine functions at different frequencies.
- **Formula**:
  ```
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```
- **Why sinusoidal**: (1) Can generalize to longer sequences than seen during training. (2) Relative positions can be represented as linear functions of absolute positions. (3) No extra learnable parameters.

### Learned vs Sinusoidal vs RoPE
- **Story**: Three ways to tell students their seat number. Sinusoidal: give each a fixed clock pattern (our notebook). Learned: let each student pick their own badge number during training (GPT-2). RoPE: rotate each student's information based on their position — nearby students have similar rotations (LLaMA, Mistral). All achieve the same goal — the model knows token order.
- **What**:
  - **Sinusoidal** (original Transformer): Fixed, no learnable parameters. Used in our notebook.
  - **Learned** (GPT-2): A second `nn.Embedding` for positions. Learnable but limited to max_len seen during training.
  - **RoPE** (LLaMA, Mistral): Rotary Position Embedding — encodes position by rotating Q and K vectors. Better for long sequences.

### register_buffer
- **Story**: You have a toolbox. Some tools you sharpen over time (learnable parameters). But the ruler never changes — it's always 30cm. You still keep it in the toolbox so it travels with you. `register_buffer` is how you put the ruler (positional encoding) in the toolbox (model) without sharpening it (no gradient updates).
- **What**: `self.register_buffer('pe', pe)` stores a tensor as part of the model but **not as a learnable parameter**. It moves with the model to GPU, gets saved/loaded, but doesn't receive gradients.
- **Why**: Positional encoding is fixed — we don't want the optimizer to update it.


---

## 5. Attention Concepts

### Self-Attention
- **Story**: You're at a dinner party and someone says "I went to the bank to get some water." To understand "bank", you look around the sentence for clues — "water" tells you it's a riverbank, not a financial bank. Self-attention is the model doing exactly this: every word looks at every other word in the sentence to figure out what it means in this specific context.
- **What**: A mechanism where every token in a sequence computes a weighted sum of other tokens' representations. Each token "looks at" other tokens to build context.
- **Formula**: `Attention(Q, K, V) = softmax(QK^T / √d_k) × V`

### Causal (Masked) Self-Attention — THE Key Concept
- **Story**: Imagine you're watching a movie for the first time. At minute 30, you can only remember what happened in minutes 1–30. You can't know what happens at minute 60 — it hasn't happened yet. Causal attention enforces this rule on the model: when processing token 3, it can only "see" tokens 0, 1, 2, 3. It cannot peek at tokens 4, 5, 6. This is what makes decoder-only models autoregressive — they generate left-to-right, never looking ahead.
- **What**: A lower-triangular mask applied to attention scores so that position `i` can only attend to positions `0..i`. Future positions are blocked by setting their scores to `-inf` before softmax.
- **The mask visualized**:
  ```
  Token:    the  cat  sat  on   the  mat
  the    [  ✅   ❌   ❌   ❌   ❌   ❌  ]  ← sees only itself
  cat    [  ✅   ✅   ❌   ❌   ❌   ❌  ]  ← sees 'the', 'cat'
  sat    [  ✅   ✅   ✅   ❌   ❌   ❌  ]  ← sees 'the', 'cat', 'sat'
  on     [  ✅   ✅   ✅   ✅   ❌   ❌  ]
  the    [  ✅   ✅   ✅   ✅   ✅   ❌  ]
  mat    [  ✅   ✅   ✅   ✅   ✅   ✅  ]  ← sees everything before it
  ```
- **Why**: During generation, future tokens don't exist yet. The mask ensures training matches inference — the model never cheats by looking ahead.
- **How implemented**: `scores.masked_fill(mask == 0, float('-inf'))` → after softmax, `-inf` becomes 0 probability.
- **Contrast with encoder-only**: BERT uses **bidirectional** attention (no mask) — every token sees every other token. That's great for understanding but impossible for generation.

### Query (Q), Key (K), Value (V)
- **Story**: You walk into a library looking for a book about cooking (that's your **Query**). Each book on the shelf has a title on its spine (that's the **Key**). You compare your query against each title to find matches. When you find a match, you pull the book off the shelf and read its contents (that's the **Value**). Attention works the same way.
- **What**: Three different linear projections of the input:
  - **Query (Q)**: "What am I looking for?" — the token asking the question
  - **Key (K)**: "What do I contain?" — the token being compared against
  - **Value (V)**: "What information do I provide?" — the actual content to aggregate
- **How**: `Q = W_q × input`, `K = W_k × input`, `V = W_v × input` where W_q, W_k, W_v are learned weight matrices.
- **In decoder-only**: Q, K, V all come from the **same** input (self-attention). In encoder-decoder, the decoder's cross-attention uses Q from decoder, K and V from encoder.

### Scaled Dot-Product Attention
- **Story**: Imagine scoring how similar two things are by multiplying their features together. If each thing has 256 features, the total score gets really big just because there are so many features — not because the things are actually more similar. Dividing by √256 = 16 corrects for this, keeping scores fair regardless of dimensions.
- **What**: `scores = QK^T / √d_k` — dot product of queries and keys, scaled by √d_k.
- **Why scale**: Without scaling, dot products grow large with high dimensions, pushing softmax into regions with tiny gradients (vanishing gradient problem).

### Multi-Head Attention
- **Story**: When you read a sentence, your brain processes multiple things at once — grammar, meaning, emotion, references. Multi-head attention is like having 4 different readers, each focusing on a different aspect. One head might notice that "it" refers to "the cat", another might notice that "sat" is the verb. Then all their notes are combined.
- **What**: Instead of one attention computation, split Q/K/V into `n_heads` parallel "heads", each attending to different aspects, then concatenate results.
- **Example**: With d_model=256 and n_heads=4, each head works with d_k=64 dimensions.
- **Formula**: `MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_o`

### No Cross-Attention (Decoder-Only vs Encoder-Decoder)
- **Story**: In an encoder-decoder model, the decoder is like a translator who reads the original text (encoder output) while writing the translation. They constantly look back at the source. In a decoder-only model, there's no "source" to look back at — the model only talks to itself. It generates purely from its own context, no external reference.
- **What**: Decoder-only has **2 sub-layers** per layer: Causal Self-Attention + FFN. Encoder-decoder has **3 sub-layers**: Causal Self-Attention + Cross-Attention + FFN.
- **Why no cross-attention**: There's no encoder to attend to. The model's only input is the sequence itself.

| | Decoder-Only | Encoder-Only | Encoder-Decoder (decoder) |
|--|-------------|-------------|--------------------------|
| Self-Attention | **Causal** (masked) | **Bidirectional** (no mask) | **Causal** (masked) |
| Cross-Attention | ❌ None | ❌ None | ✅ Attends to encoder |
| Sub-layers/layer | **2** | **2** | **3** |

### Softmax
- **Story**: You rate 5 dishes: pasta=8, pizza=9, salad=2, soup=3, steak=7. Softmax turns these raw scores into percentages that add up to 100%: pizza gets ~40%, salad gets ~1%. It's a way of saying "how much should I pay attention to each option?"
- **What**: Converts raw scores (logits) into a probability distribution that sums to 1. `softmax(x_i) = e^(x_i) / Σ(e^(x_j))`.
- **Where used**: (1) In attention — converts attention scores to weights. (2) In output — converts logits to token probabilities.


---

## 6. Feed-Forward Network (FFN) Concepts

### Position-wise FFN
- **Story**: After the team meeting (attention), each person goes back to their desk to think individually. They expand their notes (d_model → d_ff, making them 4× bigger), process and refine their thoughts, then compress them back into a summary (d_ff → d_model). This individual thinking time is the FFN — it processes each token's information independently after attention has gathered context.
- **What**: A two-layer neural network applied independently to each token position: `FFN(x) = Linear₂(ReLU(Linear₁(x)))`.
- **Dimensions**: `Linear₁: d_model → d_ff` (expand), `Linear₂: d_ff → d_model` (compress). Typically d_ff = 4 × d_model.
- **Why**: Attention captures relationships between tokens. FFN processes each token's representation independently, adding non-linearity and computational depth.

### ReLU (Rectified Linear Unit)
- **Story**: Think of a bouncer at a club. If your number is positive, you get in (pass through unchanged). If your number is negative, you're turned away (set to zero). That's ReLU — it keeps the good stuff and throws away the bad. This simple rule gives neural networks the ability to learn complex, non-linear patterns.
- **What**: Activation function: `ReLU(x) = max(0, x)`. Outputs x if positive, 0 if negative.
- **Why**: Without activation functions, stacking linear layers would just be one big linear layer — no ability to learn complex patterns.
- **Note**: GPT-2 uses GELU (smoother variant), LLaMA uses SwiGLU. Our notebook uses ReLU for simplicity.

---

## 7. Normalization & Regularization Concepts

### Layer Normalization (LayerNorm)
- **Story**: Imagine a class of students where some score 95/100 and others score 950/1000 on different tests. You can't compare them directly. LayerNorm converts everyone's scores to the same scale (mean=0, std=1) so they're comparable. In a transformer, values flowing through layers can drift to wildly different ranges. LayerNorm resets them to a stable range after each step.
- **What**: Normalizes the values across the feature dimension (d_model) for each token independently. Makes mean ≈ 0 and std ≈ 1, then applies learned scale and shift.
- **Why**: Stabilizes training by keeping activations in a consistent range.
- **Where**: Applied after each sub-layer (attention, FFN) in combination with residual connections.

### Residual Connection (Skip Connection)
- **Story**: You're writing an essay draft. Instead of rewriting from scratch each time, you keep the original and just add corrections on top: `final = original + corrections`. If the corrections are bad, the original still survives. Residual connections work the same way — the input passes through unchanged AND through the sub-layer, then both are added together.
- **What**: `output = LayerNorm(x + sublayer(x))` — the input is added directly to the sub-layer's output.
- **Why**: Solves the **vanishing gradient problem** in deep networks. Gradients can flow directly through the skip connection, making it easier to train deep stacks (4, 6, 12+ layers).
- **Intuition**: The sub-layer only needs to learn the "residual" (what to add/change), not the entire transformation.

### Dropout
- **Story**: A football team always passes to their star player. But what if the star gets injured? The team collapses. The coach starts randomly benching the star during practice, forcing other players to step up. Dropout does this to neurons — randomly disables some during training so the model becomes robust.
- **What**: During training, randomly sets a fraction of values to 0 (e.g., 10%). During inference, does nothing.
- **Where used**: After attention, after FFN, after positional encoding.
- **Why**: Prevents overfitting by forcing the model to not rely on any single neuron.

### Gradient Clipping
- **Story**: You're driving downhill and the car starts going too fast. You tap the brakes to keep a safe speed. Gradient clipping does the same — if the gradients get dangerously large, it scales them down to a safe maximum.
- **What**: `clip_grad_norm_(parameters, max_norm=1.0)` — if the total gradient norm exceeds max_norm, scale all gradients down proportionally.
- **Why**: Transformers can have **exploding gradients**, causing unstable training. Clipping keeps them bounded.


---

## 8. Loss & Optimization Concepts

### Cross-Entropy Loss (at Every Position)
- **Story**: You ask a friend to guess which card you're holding from a deck of 20. If they say "Ace" with 90% confidence and they're right, small penalty (low loss). If they're wrong, big penalty (high loss). In decoder-only, this happens at **every position** simultaneously — the model guesses the next token at every step, and the loss is the average of all those guesses.
- **What**: `Loss = -log(P(correct_next_token))` averaged across all positions.
- **How computed**:
  ```
  Position 0: model predicts "cat"  → loss₀
  Position 1: model predicts "sat"  → loss₁
  Position 2: model predicts "on"   → loss₂
  ...
  Total loss = average(loss₀ + loss₁ + ... + loss₅)
  ```
- **Shape**: Logits flattened to `(batch*seq_len, vocab_size)` vs targets flattened to `(batch*seq_len)`.
- **ignore_index=PAD_IDX**: Padded positions are excluded from loss computation.
- **Contrast**: Encoder-only computes loss once per sentence (on [CLS]). Decoder-only computes loss at every position.

### Logits
- **Story**: Before a judge announces the winner of a cooking competition, they have raw scores on their notepad: Chef A = 8.5, Chef B = 3.2, Chef C = 7.1. These raw scores are logits — they haven't been converted to percentages yet. Softmax turns them into probabilities. PyTorch's CrossEntropyLoss takes the raw scores directly (it does softmax internally).
- **What**: The raw, unnormalized output scores from the LM Head. Shape: `(batch, seq_len, vocab_size)`.
- **Why raw**: CrossEntropyLoss in PyTorch expects logits (applies softmax internally for numerical stability).

### Adam Optimizer
- **Story**: Imagine hiking down a mountain in fog. Basic SGD takes equal-sized steps in the steepest direction. Adam is smarter — it remembers which direction you've been going (momentum) and adjusts step size per dimension. If you've been going steeply downhill in one direction, it takes bigger steps there. If the terrain is flat, smaller steps.
- **What**: Adaptive Moment Estimation — maintains per-parameter learning rates based on first moment (mean) and second moment (variance) of gradients.
- **Why Adam**: (1) Adapts learning rate per parameter. (2) Momentum smooths out noisy gradients. (3) Works well out-of-the-box for transformers.
- **Key hyperparameters**: `lr=1e-3` (learning rate), `betas=(0.9, 0.999)`, `eps=1e-8`.

### Learning Rate
- **Story**: You're adjusting the volume on a speaker. Turn the knob too much (high lr) and it blasts from silent to deafening — you overshoot. Turn it too little (low lr) and you're there all day making tiny adjustments.
- **What**: Controls how big each weight update step is. `new_weight = old_weight - lr × gradient`.
- **Our notebook**: `lr=1e-3` (0.001). Production GPT models use learning rate warmup + decay schedules.

### Backpropagation
- **Story**: You bake a cake and it tastes bad (high loss). You trace back: "Was it the oven temperature? The sugar amount? The mixing time?" You figure out how much each ingredient contributed to the bad taste and adjust them for next time. Backpropagation does exactly this — it traces the error backwards through every layer.
- **What**: Algorithm that computes gradients of the loss with respect to every parameter by applying the chain rule backwards through the computation graph.
- **How in PyTorch**: `loss.backward()` computes all gradients, `optimizer.step()` updates weights, `optimizer.zero_grad()` resets gradients for the next iteration.

---

## 9. Model Architecture Concepts

### DecoderOnlyTransformer (Full Model)
- **Story**: Think of the full model as a factory assembly line with 4 stations: (1) Look up what each word means (Embedding), (2) Stamp each word with its position (PE), (3) Let words discuss with each other in a controlled way — no peeking ahead (Decoder Stack), (4) Based on the discussion, guess the next word (LM Head). The whole factory processes one sentence and outputs a prediction for every position.
- **What**: The complete model: `Embedding → Positional Encoding → Decoder Stack (N layers) → LM Head`.
- **Forward pass**: `input_ids → embedded → pos_encoded → decoder_output → logits`.

### Decoder Layer (Single Layer)
- **Story**: Each station on the assembly line does two things: (1) workers discuss with each other to share information, but only with workers who came before them (causal self-attention), (2) each worker individually refines their piece (FFN). After 4 stations, a rough understanding becomes a polished prediction.
- **What**: One unit of the decoder stack containing:
  1. Causal Self-Attention + Residual + LayerNorm
  2. FFN + Residual + LayerNorm
- **Only 2 sub-layers**: No cross-attention (that's encoder-decoder only).
- **Stacking**: GPT-2 small = 12 layers, GPT-2 medium = 24, LLaMA-7B = 32. Our notebook = 4.

### LM Head (Language Model Head)
- **Story**: After all the discussion and processing, someone needs to make the final call: "What's the next word?" The LM Head is that decision-maker. It takes the refined understanding of each position (a 256-dim vector) and scores every word in the vocabulary (20 scores). The highest score wins.
- **What**: `Linear(d_model → vocab_size)` — projects decoder output to vocabulary logits at every position.
- **Output shape**: `(batch, seq_len, vocab_size)` — a probability distribution over the vocabulary at each position.

### Weight Tying
- **Story**: The embedding layer converts word IDs to vectors (20 → 256). The LM Head converts vectors back to word scores (256 → 20). They're doing the reverse of each other — so why not share the same weight matrix? It's like using the same dictionary for both looking up definitions and writing definitions. Saves parameters and often improves performance.
- **What**: `lm_head.projection.weight = embedding.embedding.weight` — the embedding matrix and LM Head share the same weight matrix.
- **Why**: (1) Reduces parameters significantly. (2) Ensures the output space is consistent with the input space. (3) Used by GPT-2, LLaMA, and most modern models.

### state_dict
- **Story**: A state_dict is like a recipe card for a trained chef. The card doesn't describe HOW to cook (that's the code/architecture). It lists the exact amounts of every ingredient the chef has perfected (the learned weights). You can give this card to any chef who knows the same technique (same architecture) and they'll make the exact same dish.
- **What**: A Python dictionary mapping parameter names to their tensor values. `model.state_dict()` returns it, `model.load_state_dict()` loads it.
- **Saved files**: `model.pth` (weights) + `config.json` (architecture) + `vocab.json` (tokenizer).

### model.eval() vs model.train()
- **Story**: A student behaves differently during practice (training) vs the real exam (inference). During practice, they skip some questions randomly to challenge themselves (dropout). During the exam, they give full effort — no skipping. `model.train()` is practice mode. `model.eval()` is exam mode.
- **What**: Switches the model between training and evaluation modes.
- **Differences**: `eval()` disables dropout. `train()` enables it.
- **Always pair with**: `torch.no_grad()` during inference to skip gradient computation (saves memory and speed).


---

## 10. Inference Concepts

### Autoregressive Generation
- **Story**: Decoder-only is like a painter — they paint one stroke at a time, each stroke depending on the previous ones. Encoder-only is like a photographer — one click and you have the whole picture. For generation tasks ("write me a poem"), you need the painter. The model generates one token, appends it to the input, and repeats until it decides to stop.
- **What**: Generate text one token at a time in a loop:
  1. Start with prompt: `[the, cat]`
  2. Model predicts next token: `sat`
  3. Append: `[the, cat, sat]`
  4. Model predicts next: `on`
  5. Repeat until `<eos>` or max length
- **Why slow**: Each token requires a full forward pass. Generating N tokens = N forward passes. Training processes all positions in parallel (one pass), but inference is sequential.

### Training vs Inference
| | Training | Inference |
|--|---------|-----------|
| Input | Full sequence (teacher forcing) | Prompt + generated so far |
| Runs | Once per batch (parallel) | N times (one per token) |
| Target | Known (shifted input) | Unknown (being generated) |
| Mask | All positions at once | Grows by 1 each step |
| Speed | Fast (parallel) | Slow (sequential) |

### Greedy Decoding
- **Story**: At every fork in the road, you always take the path that looks best right now. You never consider that a slightly worse path now might lead to a much better destination later. That's greedy decoding — always pick the highest probability token.
- **What**: At each step, pick the token with the highest probability: `next_token = logits.argmax()`.
- **Pros**: Deterministic, fast, simple.
- **Cons**: Can be repetitive and boring. May miss better sequences that start with a lower-probability token.

### Temperature
- **Story**: Temperature is like a creativity dial. Turn it down (0.1) and the model becomes very focused and predictable — always picking the obvious next word. Turn it up (2.0) and the model becomes wild and creative — picking surprising, sometimes nonsensical words. At 1.0, it's the default balance.
- **What**: Scale logits before softmax: `logits = logits / temperature`.
- **Low temperature (0.1)**: Sharpens distribution → model is very confident → picks the top token almost always.
- **High temperature (2.0)**: Flattens distribution → all tokens become more equally likely → more random/creative output.
- **Temperature = 1.0**: No change (default).

### Top-k Sampling
- **Story**: Instead of always picking the best restaurant (greedy), you narrow it down to the top 3 restaurants and randomly pick one. Sometimes you end up at the second-best place, which turns out to be a great surprise. Top-k sampling does this — it keeps only the top k most likely tokens and samples from them.
- **What**: Keep only the top `k` tokens by probability, zero out the rest, then sample from the remaining distribution.
- **Example**: k=3 means the model only considers the 3 most likely next tokens, then randomly picks one based on their probabilities.
- **Why**: Adds diversity while avoiding very unlikely (nonsensical) tokens.

### Top-p (Nucleus) Sampling
- **Story**: Instead of always keeping exactly 3 options (top-k), you keep however many options it takes to cover 90% of the probability. If one token has 95% probability, you only keep that one. If the top 10 tokens each have ~9%, you keep all 10. It adapts to the situation.
- **What**: Keep the smallest set of tokens whose cumulative probability ≥ p (e.g., p=0.9). Sample from that set.
- **Why**: More adaptive than top-k. When the model is confident, it considers fewer options. When uncertain, it considers more.

### Decoding Strategies Summary
| Strategy | How | Best for |
|----------|-----|----------|
| **Greedy** | Always pick highest prob | Deterministic, factual tasks |
| **Top-k** | Sample from top k tokens | Creative text with controlled diversity |
| **Top-p** | Sample from smallest set with cumulative prob ≥ p | Best diversity/quality balance |
| **Temperature** | Scale logits before sampling | Controls creativity (combine with top-k/top-p) |

---

## 11. Hyperparameters Summary

| Hyperparameter | Our Notebook | GPT-2 Small | GPT-2 Medium | LLaMA-7B |
|---------------|-------------|-------------|--------------|----------|
| d_model | 256 | 768 | 1024 | 4096 |
| n_heads | 4 | 12 | 16 | 32 |
| n_layers | 4 | 12 | 24 | 32 |
| d_ff | 1024 | 3072 | 4096 | 11008 |
| vocab_size | 20 | 50,257 | 50,257 | 32,000 |
| dropout | 0.1 | 0.1 | 0.1 | 0.0 |
| Parameters | ~3.2M | ~124M | ~355M | ~7B |

---

## 12. File Formats

| Format | Extension | Used by | Notes |
|--------|-----------|---------|-------|
| PyTorch checkpoint | `.pth` | PyTorch native | Our notebook uses this |
| HuggingFace binary | `.bin` | HuggingFace Transformers | PyTorch weights in HF format |
| SafeTensors | `.safetensors` | HuggingFace (recommended) | Faster loading, no arbitrary code execution risk |
| Config | `.json` | All frameworks | Architecture hyperparameters |
| Vocabulary | `.json` / `.txt` | Tokenizer | word2idx mapping |
| GGUF | `.gguf` | llama.cpp, Ollama | Quantized format for CPU/edge inference |

---

## 13. Real-World Deployment Example

### Scenario: AI Writing Assistant for a Content Platform

**Goal**: Generate blog post drafts, email replies, and creative content from user prompts in real-time.

### Step 1: Pre-trained Model Selection
- Use **LLaMA-2-7B** or **Mistral-7B** from HuggingFace — already pre-trained on trillions of tokens.
- Don't train from scratch — transfer learning saves months of compute and millions of dollars.

### Step 2: Fine-tuning (Optional)
| Detail | Value |
|--------|-------|
| Base model | `mistral-7b-v0.1` (7B params) |
| Fine-tuning method | LoRA / QLoRA (parameter-efficient) |
| Training data | 50,000 high-quality writing examples |
| Fine-tuning epochs | 1–3 |
| Batch size | 4 (with gradient accumulation) |
| Learning rate | 2e-5 with cosine decay |
| Hardware | 1× NVIDIA A10G GPU (24GB) |
| Fine-tuning time | ~4–8 hours |
| Fine-tuning cost (AWS) | ~$10–20 on `ml.g5.xlarge` |

### Step 3: Deployment Options on AWS

#### Option A: Real-time API (Amazon SageMaker)
```
User prompt → API Gateway → SageMaker Endpoint → LLM → generated text → Response
```
| Detail | Value |
|--------|-------|
| Instance | `ml.g5.2xlarge` (1 A10G GPU, 24GB) |
| Latency | ~500ms–2s (depends on output length) |
| Throughput | ~10–50 requests/sec (with batching) |
| Cost | ~$1.52/hr = ~$1,094/month (24/7) |
| Best for | Real-time generation, chatbots |

#### Option B: Amazon Bedrock (Managed)
```
User prompt → Bedrock API → Foundation Model → generated text → Response
```
| Detail | Value |
|--------|-------|
| Model | Claude, Llama, Mistral (hosted by AWS) |
| Latency | ~500ms–3s |
| Cost | Pay per token (~$0.0008/1K input tokens) |
| Best for | No infrastructure management, quick start |

#### Option C: Batch Processing (SageMaker Batch Transform)
```
S3 bucket (prompts CSV) → SageMaker Batch Transform → S3 bucket (generated text)
```
| Detail | Value |
|--------|-------|
| Instance | `ml.g5.2xlarge` |
| Throughput | ~1,000 prompts/hour |
| Cost | ~$0.05 per 100 prompts |
| Best for | Bulk content generation, nightly jobs |

### Step 4: Model Optimization for Production

| Technique | What it does | Speedup | Quality loss |
|-----------|-------------|---------|-------------|
| **Quantization (INT4/INT8)** | Reduce precision | 2–4× faster, 50–75% less memory | <1–2% quality drop |
| **KV Cache** | Cache key/value tensors from previous tokens | 2–5× faster inference | None |
| **vLLM / TGI** | Optimized serving with continuous batching | 3–10× throughput | None |
| **Speculative decoding** | Use small model to draft, large model to verify | 2–3× faster | None |
| **GGUF quantization** | For CPU/edge deployment | Runs on CPU/laptop | 1–3% quality drop |
| **LoRA/QLoRA** | Fine-tune only small adapter weights | 10–100× less memory for training | <1% quality drop |

### Step 5: Cost Summary

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Fine-tuning (QLoRA) | 1× A10G GPU | 4–8 hours | ~$10–20 |
| Deployment (real-time, 24/7) | 1× A10G GPU | Monthly | ~$1,094/month |
| Deployment (Bedrock) | Managed | Per token | ~$0.0008/1K tokens |
| Deployment (batch, nightly) | 1× A10G GPU | 1 hour/day | ~$45/month |

### Common Use Cases for Decoder-Only Models

| Use Case | Example |
|----------|---------|
| Text generation | Blog posts, emails, stories |
| Chatbots / assistants | Customer support, coding help |
| Code generation | GitHub Copilot, Amazon Q Developer |
| Summarization | Condense long documents |
| Translation | "Translate to French: Hello" |
| Question answering | "What is the capital of France?" |
| Reasoning | Chain-of-thought problem solving |
| Instruction following | "Write a Python function that..." |

### When NOT to Use Decoder-Only
- **Pure classification** (use encoder-only: BERT — faster, cheaper for understanding tasks)
- **Semantic search / embeddings** (use encoder-only: sentence-BERT)
- **Structured seq-to-seq** (use encoder-decoder: T5, BART — better for translation, structured output)

Decoder-only = **generation**. If your task is "produce text from a prompt", decoder-only is the right choice.

---

## 15. Decoder-Only Model Types & Training Objectives

Not all decoder-only models are trained the same way. The architecture is the same (causal self-attention + FFN), but the **training objective** and **masking strategy** differ. CLM is the dominant approach, but there are alternatives — each with trade-offs.

### Type 1: Causal Language Modeling (CLM) — The Standard
- **Story**: You're reading a book left-to-right, one word at a time. After each word, you guess the next one. You can NEVER peek ahead. This is the simplest and most natural way to train a text generator — just predict the next token.
- **What**: Predict the next token at every position using only previous tokens. Strict left-to-right, no future peeking.
- **Masking**: Lower-triangular causal mask — position `i` sees only `0..i`.
- **Training data**: Raw text, no special formatting needed.
- **Example**:
  ```
  Input:  [the, cat, sat, on, the, mat]
  Target: [cat, sat, on, the, mat, <eos>]
  ```
- **Used by**: GPT, GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Falcon, Phi
- **Why dominant**: (1) Simplest to implement. (2) Scales beautifully — just add more data and parameters. (3) Naturally supports generation. (4) No special data formatting needed.

### Type 2: Prefix Language Modeling (Prefix LM)
- **Story**: Imagine reading a question with full context ("What is the capital of France?") — you can see all words in the question at once, like an encoder. But when writing the answer ("The capital is Paris"), you write left-to-right, one word at a time, like a decoder. Prefix LM combines both: bidirectional attention on the prefix (input), causal attention on the generation part.
- **What**: Split the input into a **prefix** (bidirectional attention, like an encoder) and a **generation part** (causal attention, like a decoder). The prefix tokens can see each other freely; the generation tokens can only see the prefix + previous generation tokens.
- **Masking**:
  ```
  Prefix: "What is the capital of France?"
  Generate: "The capital is Paris"

  Prefix tokens:    [✅ ✅ ✅ ✅ ✅ ✅]  ← bidirectional (see all prefix tokens)
  Generate tokens:  [✅ ✅ ✅ ✅ ✅ ✅ | ✅ ❌ ❌ ❌]  ← causal (see prefix + past only)
  ```
- **Used by**: U-PaLM, some variants of T5 (when used as decoder-only)
- **Why not dominant**: (1) Requires knowing where the prefix ends — needs special data formatting. (2) More complex masking logic. (3) CLM with instruction tuning achieves similar results more simply.

### Type 3: Instruction-Tuned / Chat Models (CLM + Fine-tuning)
- **Story**: A base CLM model is like a parrot — it can continue any text but doesn't follow instructions well. Instruction tuning is like training the parrot to be a helpful assistant. You show it thousands of (instruction, response) pairs, and it learns to follow commands. The architecture and masking are still CLM — only the training data changes.
- **What**: Start with a CLM pre-trained model, then fine-tune on (instruction, response) pairs. The model learns to follow instructions, answer questions, and be helpful.
- **Training data format**:
  ```
  <|system|>You are a helpful assistant.
  <|user|>What is the capital of France?
  <|assistant|>The capital of France is Paris.
  ```
- **Loss**: Often computed only on the **response** tokens (not the instruction), so the model focuses on generating good answers.
- **Used by**: ChatGPT, LLaMA-Chat, Mistral-Instruct, Phi-3-Instruct
- **Why popular**: Makes CLM models actually useful for real tasks. Without instruction tuning, CLM models just autocomplete text.

### Type 4: RLHF / DPO Aligned Models (CLM + Alignment)
- **Story**: After instruction tuning, the model can follow instructions but might still say harmful or unhelpful things. RLHF is like having a human teacher grade the model's responses: "This answer is helpful" (reward) vs "This answer is harmful" (penalty). The model learns to maximize the reward — producing responses humans prefer.
- **What**: After instruction tuning, further train the model using human preference data. A reward model scores responses, and the LLM is optimized to produce higher-scoring responses.
- **Methods**:
  - **RLHF** (Reinforcement Learning from Human Feedback): Train a reward model, then use PPO to optimize the LLM.
  - **DPO** (Direct Preference Optimization): Skip the reward model — directly optimize from preference pairs (chosen vs rejected responses).
- **Used by**: ChatGPT (RLHF), LLaMA-2-Chat (RLHF), Zephyr (DPO)
- **Why important**: Makes models safe, helpful, and aligned with human values.

### Type 5: Mixture of Experts (MoE) — Architecture Variant
- **Story**: Instead of one giant brain processing everything, imagine a team of specialists. For a cooking question, the cooking expert activates. For a math question, the math expert activates. Only 2 out of 8 experts work on each token — the rest stay idle. This gives you the knowledge of a huge model but the compute cost of a small one.
- **What**: Replace the single FFN in each layer with multiple "expert" FFNs. A router network picks the top-k experts for each token. Only the selected experts compute — the rest are skipped.
- **Architecture**: Same causal attention, but FFN becomes `Router → Top-k Experts → Combine`.
- **Used by**: Mixtral 8x7B (8 experts, 2 active), GPT-4 (rumored MoE), DeepSeek-MoE
- **Why used**: 8x7B has 47B total parameters but only uses ~13B per token — fast inference with large capacity.

### Comparison: Why CLM Dominates

| Type | Architecture | Training Objective | Complexity | When to Use |
|------|-------------|-------------------|------------|-------------|
| **CLM** | Causal mask | Next token prediction | ⭐ Simplest | Base pre-training, text completion |
| **Prefix LM** | Hybrid mask (bi + causal) | Next token (generation part only) | ⭐⭐ Medium | When input context is fixed and known |
| **Instruction-Tuned** | Causal mask (same as CLM) | CLM on (instruction, response) pairs | ⭐⭐ Medium | Chatbots, assistants, following commands |
| **RLHF/DPO** | Causal mask (same as CLM) | Human preference optimization | ⭐⭐⭐ Complex | Safety, helpfulness, alignment |
| **MoE** | Causal mask + expert routing | CLM (same objective) | ⭐⭐⭐ Complex | Large-scale models with efficiency needs |

### Why CLM Won Over Alternatives

1. **Simplicity**: Just predict the next token. No special masking, no prefix boundaries, no reward models needed for base training.
2. **Scalability**: "Scaling laws" (Chinchilla, GPT-4) showed that CLM + more data + more params = better performance. No architectural tricks needed.
3. **Emergent abilities**: Large CLM models spontaneously develop abilities (reasoning, translation, coding) without being explicitly trained for them.
4. **Instruction tuning fixes the gap**: Prefix LM's advantage (bidirectional context on input) is largely matched by instruction-tuned CLM models that learn to "understand" the prompt through training data alone.
5. **Universal interface**: CLM treats everything as text completion — Q&A, translation, summarization, coding — all become "complete this text." No task-specific architecture needed.

### The Typical Production Pipeline
```
Step 1: CLM Pre-training          → Base model (raw text, trillions of tokens)
Step 2: Instruction Fine-tuning    → Instruction model (100K+ instruction pairs)
Step 3: RLHF / DPO Alignment      → Aligned model (human preference data)
Step 4: (Optional) MoE             → Efficient large-scale model
```
This is exactly how ChatGPT, LLaMA-Chat, and Mistral-Instruct are built.

---

## 16. Key Differences: Decoder-Only vs Encoder-Only vs Encoder-Decoder

| Feature | Decoder-Only | Encoder-Only | Encoder-Decoder |
|---------|-------------|-------------|----------------|
| Attention | **Causal** (masked) | Bidirectional | Both |
| Task | **Generation** | Understanding | Seq-to-Seq |
| Pre-training | CLM (next token) | MLM (masked token) | Span corruption / denoising |
| Inference | **Autoregressive loop** | Single pass | Encode once + decode loop |
| Output | Generated text | Embeddings / classes | Generated text |
| Sub-layers | **2** (self-attn + FFN) | 2 (self-attn + FFN) | 3 (self + cross + FFN) |
| Cross-attention | **❌ No** | ❌ No | ✅ Yes |
| Special tokens | `<eos>`, `<pad>` | `[CLS]`, `[SEP]`, `[MASK]` | `<eos>`, `<pad>`, `<bos>` |
| Examples | GPT, LLaMA, Mistral | BERT, RoBERTa | T5, BART, mBART |
