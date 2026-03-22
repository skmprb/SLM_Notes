# Encoder-Only Transformer — Concepts Reference

Every concept used in the encoder-only transformer pipeline, explained with intuition.

---

## 1. Pre-training & Fine-tuning

### Masked Language Modeling (MLM)
- **Story**: Imagine you're reading a book and someone covers a word with their thumb. You can still guess the word because you can read the words before AND after the covered word. That's exactly what MLM does — it hides some words and asks the model to guess them using the surrounding context from both sides.
- **What**: The pre-training objective for encoder-only models. Randomly mask ~15% of tokens in a sentence, and the model predicts the original tokens from context.
- **Example**: Input: "The cat [MASK] on the mat" → Model predicts: "sat"
- **Why it works**: Forces the model to understand **bidirectional context** — it must use both left ("The cat") and right ("on the mat") to predict the masked word.
- **Used by**: BERT, RoBERTa, ALBERT, DistilBERT
- **Contrast with CLM**: Decoder-only models use Causal Language Modeling (predict next token left-to-right only). MLM is bidirectional, CLM is unidirectional.

### Fine-tuning
- **Story**: Think of a medical student. They first spend years learning general medicine (pre-training). Then they specialize in cardiology by studying heart cases (fine-tuning). They don't forget general medicine — they just add specialized knowledge on top. Same with BERT: it learns general English first, then you teach it your specific task.
- **What**: Taking a pre-trained model and training it further on a **task-specific labeled dataset** (e.g., sentiment classification).
- **Why**: Pre-training learns general language understanding. Fine-tuning adapts it to your specific task with much less data and compute.
- **Example**: Pre-trained BERT (trained on Wikipedia) → fine-tune on 8 movie reviews with sentiment labels → sentiment classifier.

### Transfer Learning
- **Story**: You learned to ride a bicycle as a kid. When you try a motorcycle for the first time, you don't start from zero — you already know balance, steering, braking. You "transfer" that knowledge. Transfer learning is the same idea: a model trained on millions of sentences already "knows" English, so it only needs a few examples to learn a new task.
- **What**: Using knowledge learned from one task (pre-training on massive text) and applying it to another (classification, NER, etc.).
- **Why**: Training from scratch requires millions of examples. Transfer learning lets you get good results with hundreds or thousands.

---

## 2. Tokenization Concepts

### Vocabulary (word2idx / idx2word)
- **Story**: Think of a phone contacts list. Every person has a name ("Alice") and a number (42). When you want to call Alice, you look up her number. When someone calls you from number 42, you look up the name. That's exactly what vocabulary does — it's a two-way phone book between words and numbers.
- **What**: A bidirectional mapping between words and integer IDs. `word2idx = {"cat": 5, "dog": 6}`, `idx2word = {5: "cat", 6: "dog"}`.
- **Why**: Neural networks operate on numbers. The vocabulary is the bridge between human-readable text and model-readable integers.

### Special Tokens
- **Story**: When you write a letter, you don't just write the message. You add a greeting at the top, a signature at the bottom, and "P.S." for extra notes. Special tokens are like those — they're extra markers added to the text that tell the model where things start, end, or need special treatment.

| Token | Purpose | When used |
|-------|---------|-----------|
| `[PAD]` (ID=0) | Pads shorter sequences to equal length in a batch | Training & inference |
| `[CLS]` (ID=1) | Classification token — its output = sentence representation | Encoder-only specific |
| `[SEP]` (ID=2) | Marks sentence boundary / separates sentence pairs | Encoder-only specific |
| `[UNK]` (ID=3) | Replaces out-of-vocabulary words | When unseen words appear |
| `[MASK]` (ID=4) | Replaces tokens during MLM pre-training | Pre-training only |

### BPE (Byte Pair Encoding) / WordPiece
- **Story**: Imagine you're playing Scrabble and you don't have the tiles for "unhappiness". But you DO have tiles for "un", "happi", and "ness". So you build the word from smaller pieces you already know. That's what BPE/WordPiece does — it breaks unknown words into smaller known pieces so the model never has to say "I don't know this word".
- **What**: Sub-word tokenization algorithms used in production models. They break words into smaller pieces: "unhappiness" → ["un", "##happi", "##ness"].
- **Why**: Handles unknown words gracefully. A word-level tokenizer would map "unhappiness" to `[UNK]` if not in vocab. BPE breaks it into known sub-words.
- **Our notebook**: Uses simple word-level tokenization for clarity. Production BERT uses WordPiece.

### Padding
- **Story**: You're packing boxes for shipping. All boxes must be the same size to fit on the truck. Some items are small, so you fill the extra space with bubble wrap. Padding is the bubble wrap — it fills shorter sentences with `[PAD]` tokens so all sentences in a batch are the same length. The model knows to ignore the bubble wrap.
- **What**: Adding `[PAD]` tokens to shorter sequences so all sequences in a batch have the same length.
- **Why**: GPUs process batches in parallel — all tensors in a batch must have the same shape.
- **Padding mask**: A boolean tensor (True where real tokens, False where padded) passed to attention so padded positions are ignored.

---

## 3. Embedding Concepts

### Dense Vector
- **Story**: Imagine describing a person with just 3 numbers: height, weight, age. Those 3 numbers capture a lot about the person in a compact way. A dense vector does the same for words — it describes a word using 256 (or 768) numbers. Words with similar meanings get similar numbers, so "happy" and "joyful" end up close together, while "happy" and "sad" are far apart.
- **What**: A fixed-size array of floating-point numbers (e.g., 256 dimensions) that represents a token's meaning in continuous space.
- **Why**: Unlike one-hot vectors (sparse, high-dimensional, no relationships), dense vectors are compact and capture semantic similarity — "king" and "queen" are close in vector space.
- **Shape**: `(batch_size, seq_len, d_model)` — e.g., `(8, 9, 256)` means 8 sentences, 9 tokens each, 256 dimensions per token.

### d_model
- **Story**: Think of d_model as the "vocabulary" the model uses to describe each word internally. If d_model=3, the model can only describe words using 3 numbers — very limited. If d_model=768, it has 768 numbers to capture subtle differences between words. More dimensions = more nuance, like describing a color with just "red/blue/green" vs using the full RGB spectrum.
- **What**: The dimensionality of the model's internal representations. Every vector flowing through the transformer has this size.
- **Values**: BERT-base = 768, BERT-large = 1024, our notebook = 256 (for demo).
- **Why it matters**: Larger d_model = more capacity to represent nuance, but more compute and memory.

### nn.Embedding
- **Story**: Picture a giant spreadsheet with one row per word in your vocabulary. Row 5 is "cat" and contains 256 numbers that represent what "cat" means. When the model sees token ID 5, it just looks up row 5 and grabs those 256 numbers. During training, the model adjusts these numbers to make them more useful. That spreadsheet IS the embedding layer.
- **What**: PyTorch's learnable lookup table. Maps integer IDs to dense vectors. `nn.Embedding(vocab_size=33, d_model=256)` creates a 33×256 matrix.
- **padding_idx=0**: Ensures the `[PAD]` token always maps to a zero vector (not learned).
- **How it works**: `embedding[token_id]` = row `token_id` from the weight matrix. Gradients flow back to update these rows during training.

### √d_model Scaling
- **Story**: Imagine two people talking at the same time — one whispering (embedding) and one shouting (positional encoding). You can't hear the whisperer. Scaling by √d_model is like giving the whisperer a microphone so both voices are equally loud. This way, the model can hear both the word's meaning AND its position clearly.
- **What**: Embeddings are multiplied by `√d_model` (e.g., √256 = 16) before adding positional encoding.
- **Why**: Embedding values are typically small (initialized near 0). Positional encoding values are in [-1, 1]. Without scaling, positional encoding would dominate. Scaling brings embeddings to a comparable magnitude.

---

## 4. Positional Encoding Concepts

### Sinusoidal Positional Encoding
- **Story**: Imagine a row of students sitting in a classroom. The teacher can see all of them at once but can't tell who's sitting where — they all look the same from the front. So each student holds up a unique clock showing a different time. Student 1 shows 1:00, student 2 shows 2:00, etc. Now the teacher knows the order. Sinusoidal PE does the same — it gives each position a unique "wave pattern" so the model knows which word came first, second, third, etc.
- **What**: A fixed (non-learned) signal added to embeddings that encodes position using sine and cosine functions at different frequencies.
- **Formula**:
  ```
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```
- **Why sinusoidal**: (1) Can generalize to longer sequences than seen during training. (2) Relative positions can be represented as linear functions of absolute positions. (3) No extra learnable parameters.
- **Contrast**: BERT uses **learned** positional embeddings (a second nn.Embedding for positions). Same purpose, different implementation.

### register_buffer
- **Story**: You have a toolbox. Some tools you sharpen over time (learnable parameters). But the ruler never changes — it's always 30cm. You still keep it in the toolbox so it travels with you. `register_buffer` is how you put the ruler (positional encoding) in the toolbox (model) without sharpening it (no gradient updates).
- **What**: `self.register_buffer('pe', pe)` stores a tensor as part of the model but **not as a learnable parameter**. It moves with the model to GPU, gets saved/loaded, but doesn't receive gradients.
- **Why**: Positional encoding is fixed — we don't want the optimizer to update it.

### Dropout (on PE)
- **Story**: During a team exam, the teacher randomly tells some students to stay quiet. The remaining students must answer without relying on those specific teammates. This forces everyone to be independently capable. Dropout does the same — it randomly silences some numbers during training so the model can't over-rely on any single feature.
- **What**: Randomly zeroes some elements of the input tensor with probability `p` (e.g., 0.1 = 10%).
- **Why**: Regularization — prevents the model from relying too heavily on any single position or feature. Only active during `model.train()`, disabled during `model.eval()`.

---

## 5. Attention Concepts

### Self-Attention
- **Story**: You're at a dinner party and someone says "I went to the bank to get some water." To understand "bank", you look around the sentence for clues — "water" tells you it's a riverbank, not a financial bank. Self-attention is the model doing exactly this: every word looks at every other word in the sentence to figure out what it means in this specific context.
- **What**: A mechanism where every token in a sequence computes a weighted sum of all other tokens' representations. Each token "looks at" every other token to build context.
- **Example**: In "The bank by the river", the word "bank" attends strongly to "river" to understand it means "riverbank" not "financial bank".
- **Formula**: `Attention(Q, K, V) = softmax(QK^T / √d_k) × V`

### Query (Q), Key (K), Value (V)
- **Story**: You walk into a library looking for a book about cooking (that's your **Query**). Each book on the shelf has a title on its spine (that's the **Key**). You compare your query against each title to find matches. When you find a match, you pull the book off the shelf and read its contents (that's the **Value**). Attention works the same way — Q asks "what am I looking for?", K says "here's what I have", and V provides the actual information.
- **What**: Three different linear projections of the input, each serving a different role:
  - **Query (Q)**: "What am I looking for?" — the token asking the question
  - **Key (K)**: "What do I contain?" — the token being compared against
  - **Value (V)**: "What information do I provide?" — the actual content to aggregate
- **How**: `Q = W_q × input`, `K = W_k × input`, `V = W_v × input` where W_q, W_k, W_v are learned weight matrices.
- **Analogy**: Like a search engine — Q is your search query, K is the index of documents, V is the document content. The attention score (Q·K) determines which documents (V) are most relevant.

### Scaled Dot-Product Attention
- **Story**: Imagine you're scoring how similar two things are by multiplying their features together. If each thing has 256 features, the total score gets really big just because there are so many features — not because the things are actually more similar. Dividing by √256 = 16 corrects for this, like grading on a curve so the scores stay fair regardless of how many features you have.
- **What**: `scores = QK^T / √d_k` — dot product of queries and keys, scaled by √d_k.
- **Why scale**: Without scaling, dot products grow large with high dimensions, pushing softmax into regions with tiny gradients (vanishing gradient problem). Dividing by √d_k keeps values in a reasonable range.

### Multi-Head Attention
- **Story**: When you read a sentence, your brain processes multiple things at once — grammar, meaning, emotion, references. Multi-head attention is like having 4 (or 8 or 12) different readers, each focusing on a different aspect. One head might notice that "it" refers to "the cat", another might notice that "sat" is the verb, another might capture the sentiment. Then all their notes are combined into one complete understanding.
- **What**: Instead of one attention computation, split Q/K/V into `n_heads` parallel "heads", each attending to different aspects of the input, then concatenate results.
- **Example**: With d_model=256 and n_heads=4, each head works with d_k=64 dimensions.
- **Why**: Different heads can learn different types of relationships — one head might capture syntax, another semantics, another coreference.
- **Formula**: `MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_o`

### Bidirectional Attention (No Mask)
- **Story**: Imagine reading a mystery novel. An encoder-only model reads the WHOLE book at once — it already knows the ending when it reads the first page. So when it sees a clue on page 1, it immediately connects it to the reveal on page 200. A decoder-only model reads page by page and can't peek ahead. For understanding tasks ("what is this text about?"), reading everything at once is a huge advantage.
- **What**: In encoder-only models, every token can attend to every other token — both left and right context. No causal mask is applied.
- **Why**: This is the key advantage of encoder-only over decoder-only. "The cat sat on the [MASK]" — the model can use both "The cat sat on" AND nothing after to predict "mat". But in a sentence like "I went to the bank to deposit money", "bank" can see "deposit" and "money" to the right to understand it's a financial bank.
- **Contrast**: Decoder-only uses a **causal mask** (lower-triangular) so token i can only see tokens 0..i.

### Padding Mask
- **Story**: In a classroom of 10 seats, only 7 students showed up. The 3 empty seats have nobody in them. When the teacher asks "raise your hand if you agree", she should only count the 7 real students, not the empty seats. The padding mask tells the model "these positions are empty — ignore them completely."
- **What**: A boolean mask that marks which positions are real tokens (True) vs padding (False). Shape: `(batch, 1, 1, seq_len)`.
- **Why**: Without it, padded positions would participate in attention, corrupting the representations. The mask sets attention scores for padded positions to `-inf`, so after softmax they become 0.

### Softmax
- **Story**: You're at a restaurant and rate 5 dishes: pasta=8, pizza=9, salad=2, soup=3, steak=7. Softmax turns these raw scores into percentages that add up to 100%: pizza gets the biggest share (~40%), salad gets almost nothing (~1%). It's a way of saying "how much should I pay attention to each option?" — the best option gets most of the attention.
- **What**: Converts raw scores (logits) into a probability distribution that sums to 1. `softmax(x_i) = e^(x_i) / Σ(e^(x_j))`.
- **Where used**: (1) In attention — converts attention scores to weights. (2) In output — converts logits to class probabilities.
- **Property**: Amplifies differences — the highest score gets most of the probability mass.

---

## 6. Feed-Forward Network (FFN) Concepts

### Position-wise FFN
- **Story**: After the team meeting (attention), each person goes back to their desk to think individually. They expand their notes (d_model → d_ff, making them 4× bigger), process and refine their thoughts, then compress them back into a summary (d_ff → d_model). This individual thinking time is the FFN — it processes each token's information independently after attention has gathered context from other tokens.
- **What**: A two-layer neural network applied independently to each token position: `FFN(x) = Linear₂(ReLU(Linear₁(x)))`.
- **Dimensions**: `Linear₁: d_model → d_ff` (expand), `Linear₂: d_ff → d_model` (compress). Typically d_ff = 4 × d_model.
- **Why**: Attention captures relationships between tokens. FFN processes each token's representation independently, adding non-linearity and computational depth.

### ReLU (Rectified Linear Unit)
- **Story**: Think of a bouncer at a club. If your number is positive, you get in (pass through unchanged). If your number is negative, you're turned away (set to zero). That's ReLU — it keeps the good stuff and throws away the bad. This simple rule is what gives neural networks the ability to learn complex, non-linear patterns. Without it, the whole network would just be one big multiplication.
- **What**: Activation function: `ReLU(x) = max(0, x)`. Outputs x if positive, 0 if negative.
- **Why**: Introduces non-linearity. Without activation functions, stacking linear layers would just be one big linear layer — no ability to learn complex patterns.
- **Note**: BERT actually uses GELU (Gaussian Error Linear Unit), a smoother variant. Our notebook uses ReLU for simplicity.

---

## 7. Normalization & Regularization Concepts

### Layer Normalization (LayerNorm)
- **Story**: Imagine a class of students where some score 95/100 and others score 950/1000 on different tests. You can't compare them directly. LayerNorm is like converting everyone's scores to the same scale (mean=0, std=1) so they're comparable. In a transformer, values flowing through layers can drift to wildly different ranges. LayerNorm resets them to a stable range after each step, keeping training smooth.
- **What**: Normalizes the values across the feature dimension (d_model) for each token independently. Makes mean ≈ 0 and std ≈ 1, then applies learned scale and shift.
- **Why**: Stabilizes training by keeping activations in a consistent range. Without it, values can explode or vanish as they pass through many layers.
- **Where**: Applied after each sub-layer (attention, FFN) in combination with residual connections.

### Residual Connection (Skip Connection)
- **Story**: You're writing an essay draft. Instead of rewriting the whole thing from scratch each time, you keep the original and just add corrections on top: `final = original + corrections`. If the corrections are bad, the original still survives. Residual connections work the same way — the input passes through unchanged AND through the sub-layer, then both are added together. This means even if a layer learns nothing useful, the information still flows through.
- **What**: `output = LayerNorm(x + sublayer(x))` — the input is added directly to the sub-layer's output.
- **Why**: Solves the **vanishing gradient problem** in deep networks. Gradients can flow directly through the skip connection, making it easier to train deep stacks (4, 6, 12+ layers).
- **Intuition**: The sub-layer only needs to learn the "residual" (what to add/change), not the entire transformation.

### Dropout
- **Story**: A football team always passes to their star player. But what if the star gets injured? The team collapses. The coach starts randomly benching the star during practice, forcing other players to step up. Now the whole team is strong, not just one player. Dropout does this to neurons — randomly disables some during training so the model becomes robust and doesn't over-rely on any single feature.
- **What**: During training, randomly sets a fraction of values to 0 (e.g., 10%). During inference, does nothing.
- **Where used in our model**: After attention, after FFN, in the classification head.
- **Why**: Prevents overfitting by forcing the model to not rely on any single neuron. Acts as an ensemble of sub-networks.

### Gradient Clipping
- **Story**: You're driving downhill and the car starts going too fast. You tap the brakes to keep a safe speed. Gradient clipping does the same for training — if the gradients (the "speed" of weight updates) get dangerously large, it scales them down to a safe maximum. Without it, one bad batch could send the model's weights flying off into nonsense.
- **What**: `clip_grad_norm_(parameters, max_norm=1.0)` — if the total gradient norm exceeds max_norm, scale all gradients down proportionally.
- **Why**: Transformers can have **exploding gradients** (gradients become extremely large), causing unstable training. Clipping keeps them bounded.

---

## 8. Loss & Optimization Concepts

### Cross-Entropy Loss
- **Story**: You ask a friend to guess which card you're holding from a deck of 52. If they say "Ace of Spades" with 90% confidence and they're right, you give them a small penalty (low loss). If they say "Ace of Spades" with 90% confidence and they're WRONG, you give them a big penalty (high loss). Cross-entropy is this penalty system — it punishes the model more when it's confidently wrong and rewards it when it's confidently right.
- **What**: `Loss = -log(P(correct_class))`. Measures how far the predicted probability distribution is from the true label.
- **Example**: If true label is "positive" (class 1) and model predicts P(positive) = 0.9, loss = -log(0.9) = 0.105 (low). If P(positive) = 0.1, loss = -log(0.1) = 2.302 (high).
- **For encoder-only**: Computed once per sentence (on [CLS] output). For decoder/encoder-decoder: computed at every token position.

### Logits
- **Story**: Before a judge announces the winner of a cooking competition, they have raw scores on their notepad: Chef A = 8.5, Chef B = 3.2, Chef C = 7.1. These raw scores are logits — they haven't been converted to percentages yet. Softmax turns them into "Chef A wins with 65% probability". PyTorch's CrossEntropyLoss takes the raw scores directly (it does the softmax internally for better numerical precision).
- **What**: The raw, unnormalized output scores from the model before softmax. Shape: `(batch, n_classes)` for classification.
- **Why raw**: CrossEntropyLoss in PyTorch expects logits (it applies softmax internally for numerical stability).

### Adam Optimizer
- **Story**: Imagine you're hiking down a mountain in fog (finding the lowest loss). Basic SGD is like taking equal-sized steps in the steepest direction. Adam is smarter — it remembers which direction you've been going (momentum) and adjusts step size per dimension. If you've been going steeply downhill in one direction, it takes bigger steps there. If the terrain is flat, it takes smaller steps. It's like having a GPS that adapts to the terrain.
- **What**: Adaptive Moment Estimation — an optimizer that maintains per-parameter learning rates based on first moment (mean) and second moment (variance) of gradients.
- **Why Adam over SGD**: (1) Adapts learning rate per parameter — parameters that update rarely get larger steps. (2) Momentum smooths out noisy gradients. (3) Works well out-of-the-box for transformers.
- **Key hyperparameters**: `lr=1e-3` (learning rate), `betas=(0.9, 0.999)` (momentum decay rates), `eps=1e-8` (numerical stability).

### Learning Rate
- **Story**: You're adjusting the volume on a speaker. Turn the knob too much (high learning rate) and it blasts from silent to deafening — you overshoot. Turn it too little (low learning rate) and you're there all day making tiny adjustments. The learning rate is how big each adjustment step is. For fine-tuning a pre-trained model, you use a tiny learning rate (2e-5) because the model is already close to good — you just need small tweaks.
- **What**: Controls how big each weight update step is. `new_weight = old_weight - lr × gradient`.
- **Too high**: Training diverges (loss explodes). **Too low**: Training is extremely slow.
- **Our notebook**: `lr=1e-3` (0.001). Production BERT uses `lr=2e-5` for fine-tuning (much smaller because the model is already pre-trained).

### Backpropagation
- **Story**: You bake a cake and it tastes bad (high loss). You trace back: "Was it the oven temperature? The sugar amount? The mixing time?" You figure out how much each ingredient contributed to the bad taste and adjust them for next time. Backpropagation does exactly this — it traces the error backwards through every layer and calculates how much each weight contributed to the mistake, so the optimizer knows what to adjust.
- **What**: Algorithm that computes gradients of the loss with respect to every parameter by applying the chain rule backwards through the computation graph.
- **How in PyTorch**: `loss.backward()` computes all gradients, `optimizer.step()` updates weights, `optimizer.zero_grad()` resets gradients for the next iteration.

---

## 9. Model Architecture Concepts

### [CLS] Token
- **Story**: Imagine a team meeting where one person is the designated note-taker. They don't contribute their own ideas — they just listen to everyone else and write a summary. The [CLS] token is that note-taker. It sits at position 0, listens to every other token through self-attention across all layers, and by the end, its output is a single vector that summarizes the entire sentence. You then hand that summary to the classification head to make a decision.
- **What**: A special token prepended to every input. After passing through the encoder, its output vector at position 0 serves as the **sentence-level representation**.
- **Why it works**: Through self-attention across all layers, [CLS] aggregates information from every token in the sentence. By the final layer, it "knows" the entire sentence.
- **Used for**: Classification (pass through a head), sentence similarity (compare [CLS] vectors via cosine similarity), retrieval (use as document embedding).

### Encoder Layer
- **Story**: Think of an assembly line in a factory. Each station (layer) does two things: (1) workers discuss with each other to share information (self-attention), (2) each worker individually refines their piece (FFN). After 4 stations, a rough piece of metal becomes a polished product. Each encoder layer adds another level of understanding — early layers capture basic grammar, later layers capture meaning and nuance.
- **What**: One unit of the encoder stack containing: Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm.
- **Stacking**: BERT-base has 12 layers, BERT-large has 24. Our notebook uses 4. Deeper = better understanding but more compute.

### state_dict
- **Story**: A state_dict is like a recipe card for a trained chef. The card doesn't describe HOW to cook (that's the code/architecture). It lists the exact amounts of every ingredient the chef has perfected over years of practice (the learned weights). You can give this card to any chef who knows the same cooking technique (same architecture) and they'll make the exact same dish.
- **What**: A Python dictionary mapping parameter names to their tensor values. `model.state_dict()` returns it, `model.load_state_dict()` loads it.
- **Why**: Separates the model architecture (code) from the learned weights (data). You can save/load weights independently.

### model.eval() vs model.train()
- **Story**: A student behaves differently during practice (training) vs the real exam (inference). During practice, they might skip some questions randomly to challenge themselves (dropout). During the exam, they give it their full effort — no skipping. `model.train()` is practice mode (dropout on, learning). `model.eval()` is exam mode (dropout off, full performance).
- **What**: Switches the model between evaluation and training modes.
- **Differences**: `eval()` disables dropout and changes BatchNorm behavior. `train()` enables them.
- **Always pair with**: `torch.no_grad()` during inference to skip gradient computation (saves memory and speed).

---

## 10. Inference Concepts

### Cosine Similarity
- **Story**: Two arrows pointing in the same direction are similar, even if one is longer than the other. Two arrows pointing in opposite directions are dissimilar. Cosine similarity measures the angle between two arrows (vectors), ignoring their length. If "I love this movie" and "This film is wonderful" point in the same direction in vector space, they're similar (+1). If "I love this" and "I hate this" point in opposite directions, they're dissimilar (-1).
- **What**: Measures the angle between two vectors. `cos_sim(A, B) = (A·B) / (||A|| × ||B||)`. Range: [-1, 1].
- **+1**: Vectors point in the same direction (very similar). **0**: Orthogonal (unrelated). **-1**: Opposite directions (very dissimilar).
- **Used for**: Sentence similarity, semantic search, document retrieval. Compare [CLS] embeddings of two sentences.

### Sentence Embedding
- **Story**: Imagine compressing an entire paragraph into a single fingerprint. Two paragraphs about the same topic would have similar fingerprints, even if they use completely different words. The [CLS] token's output is that fingerprint — a single vector (256 or 768 numbers) that captures the essence of the entire sentence. You can compare fingerprints to find similar documents, cluster them into topics, or use them for search.
- **What**: A single fixed-size vector (d_model dimensions) that represents the meaning of an entire sentence.
- **How**: The [CLS] token's output from the encoder. It has attended to all tokens bidirectionally.
- **Use cases**: Semantic search, clustering, duplicate detection, recommendation systems.

### Single Forward Pass (No Autoregressive Loop)
- **Story**: Encoder-only is like a photographer — one click and you have the whole picture. Decoder-only (GPT) is like a painter — they paint one stroke at a time, each stroke depending on the previous ones. For understanding tasks ("is this review positive?"), you just need the photograph. For generation tasks ("write me a poem"), you need the painter. That's why encoder-only inference is fast — one pass, done.
- **What**: Encoder-only models process the entire input in one shot. Feed in text → get output. Done.
- **Contrast**: Decoder-only (GPT) needs N forward passes to generate N tokens. Encoder-decoder needs 1 encoder pass + N decoder passes.
- **Why faster**: No sequential dependency. The entire computation is parallelizable.

---

## 11. Hyperparameters Summary

| Hyperparameter | Our Notebook | BERT-base | BERT-large |
|---------------|-------------|-----------|------------|
| d_model | 256 | 768 | 1024 |
| n_heads | 4 | 12 | 16 |
| n_layers | 4 | 12 | 24 |
| d_ff | 1024 | 3072 | 4096 |
| vocab_size | 33 | 30,522 | 30,522 |
| dropout | 0.1 | 0.1 | 0.1 |
| Parameters | ~3.2M | ~110M | ~340M |

---

## 12. File Formats

| Format | Extension | Used by | Notes |
|--------|-----------|---------|-------|
| PyTorch checkpoint | `.pth` | PyTorch native | Our notebook uses this |
| HuggingFace binary | `.bin` | HuggingFace Transformers | PyTorch weights in HF format |
| SafeTensors | `.safetensors` | HuggingFace (recommended) | Faster loading, no arbitrary code execution risk |
| Config | `.json` | All frameworks | Architecture hyperparameters |
| Vocabulary | `.json` / `.txt` | Tokenizer | word2idx mapping |
| ONNX | `.onnx` | Cross-platform | For deployment on non-PyTorch runtimes |

---

## 13. Real-World Deployment Example

### Scenario: Customer Review Sentiment Classifier for an E-commerce Platform

**Goal**: Classify incoming product reviews as positive/negative in real-time to flag issues and track satisfaction.

### Step 1: Pre-trained Model Selection
- Use **BERT-base** (110M params) from HuggingFace — already pre-trained on English Wikipedia + BookCorpus.
- Don't train from scratch — transfer learning saves weeks of compute.

### Step 2: Fine-tuning
| Detail | Value |
|--------|-------|
| Base model | `bert-base-uncased` (110M params) |
| Training data | 25,000 labeled reviews (Amazon/Yelp) |
| Fine-tuning epochs | 3–5 |
| Batch size | 32 |
| Learning rate | 2e-5 (small — model is already pre-trained) |
| Hardware | 1× NVIDIA T4 GPU (16GB) |
| Fine-tuning time | ~30 minutes |
| Fine-tuning cost (AWS) | ~$0.50 on `ml.g4dn.xlarge` ($0.736/hr) |

### Step 3: Deployment Options on AWS

#### Option A: Real-time API (Amazon SageMaker)
```
User review → API Gateway → SageMaker Endpoint → BERT model → positive/negative → Response
```
| Detail | Value |
|--------|-------|
| Instance | `ml.g4dn.xlarge` (1 T4 GPU) |
| Latency | ~10–20ms per request |
| Throughput | ~500 requests/sec |
| Cost | ~$0.736/hr = ~$530/month (24/7) |
| Best for | Real-time classification, low latency |

#### Option B: Serverless (AWS Lambda + ONNX)
```
User review → API Gateway → Lambda → ONNX Runtime → positive/negative → Response
```
| Detail | Value |
|--------|-------|
| Convert model | PyTorch → ONNX (smaller, CPU-optimized) |
| Runtime | ONNX Runtime on Lambda (CPU) |
| Latency | ~50–100ms per request |
| Cost | ~$0.20 per 1M requests (pay per use) |
| Best for | Low traffic, cost-sensitive, bursty workloads |

#### Option C: Batch Processing (AWS Batch / SageMaker Batch Transform)
```
S3 bucket (reviews CSV) → SageMaker Batch Transform → S3 bucket (results CSV)
```
| Detail | Value |
|--------|-------|
| Instance | `ml.g4dn.xlarge` |
| Throughput | ~50,000 reviews/minute |
| Cost | ~$0.02 per 10,000 reviews |
| Best for | Nightly batch processing, analytics dashboards |

### Step 4: Model Optimization for Production

| Technique | What it does | Speedup | Quality loss |
|-----------|-------------|---------|-------------|
| **ONNX export** | Convert to optimized runtime | 2–3× | None |
| **Quantization (INT8)** | Reduce precision from FP32 to INT8 | 2–4× | <1% accuracy drop |
| **Distillation** | Train a smaller model (DistilBERT, 66M params) to mimic BERT | 2× faster, 40% smaller | ~1–2% accuracy drop |
| **Pruning** | Remove unimportant weights | 1.5–2× | <1% accuracy drop |
| **TorchScript** | JIT compile the model | 1.2–1.5× | None |

### Step 5: Cost Summary

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Fine-tuning | 1× T4 GPU | 30 min | ~$0.50 |
| Deployment (real-time, 24/7) | 1× T4 GPU | Monthly | ~$530/month |
| Deployment (serverless) | Lambda CPU | Per request | ~$0.20/1M requests |
| Deployment (batch, nightly) | 1× T4 GPU | 10 min/day | ~$3.70/month |

### Common Use Cases for Encoder-Only Models

| Use Case | Task Type | Example |
|----------|-----------|---------|
| Sentiment analysis | Classification | "Great product!" → positive |
| Spam detection | Classification | "Buy now! Free money!" → spam |
| Named Entity Recognition (NER) | Token classification | "John works at Amazon" → [PERSON, O, O, ORG] |
| Semantic search | Embedding + similarity | "How to return an item?" → find similar FAQ |
| Duplicate detection | Embedding + similarity | "Is this review a duplicate?" |
| Text clustering | Embedding + clustering | Group similar support tickets |
| Question answering (extractive) | Span extraction | Context + question → extract answer span |
| Toxicity detection | Classification | Flag harmful content in comments |

### When NOT to Use Encoder-Only
- **Text generation** (use decoder-only: GPT, LLaMA)
- **Translation** (use encoder-decoder: T5, BART)
- **Summarization** (use encoder-decoder or decoder-only)
- **Chatbots** (use decoder-only)

Encoder-only = **understanding**. If your task is "read text and decide something", encoder-only is the right choice.
