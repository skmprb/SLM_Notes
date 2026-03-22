NN is an function approximator
it learns "x" -> input and "y" -> output

NN can learn anything by given enough data and compute power.
This can happen by single concept called neuron

neuron :
A single neuron does this:
y = 𝜎(w^T x + b)

Where:

x → input vector
w → weights (learnable parameters)
b → bias
σ → activation function
y → output

Intuition:

w → what to pay attention to
b → shift the decision boundary
σ → introduces non-linearity (very important)

Without σ, the network is just a linear model → useless for complex problems.

These neurons scale by adding layers (stack of neurons)
        Input layer -> hidden layers -> output layer

Forwards Pass
1. data flows left -> right
2. each layer transforms the data  (input layer, layer1, layer2, ... , output layer)
3. final output is produced by (matrix multiplication + activation )

Loss Function
Measures how well the network is performing
Common loss functions:
- Mean Squared Error (MSE) for regression
- Cross-Entropy Loss for classification

Backpropagation
the real engine (where the learning happens)
1. compute the loss using the chain rule of calculus
2. update the weights by learning rate. the gradient descent algorithm
   # The gradient tells us the direction of steepest increase in the loss function
   # By moving in the opposite direction (negative gradient), we minimize the loss
   # Think of it like rolling a ball downhill to find the lowest point (minimum loss)   w = w - α * ∇w (where α is learning rate and ∇w is gradient of loss w.r.t weights)
   b = b - α * ∇b (where ∇b is gradient of loss w.r.t bias)

Training loop
1. initialize weights and biases
2. for each epoch:
   - forward pass (compute output)
   - compute loss
   - backpropagation (update weights and biases)
3. repeat until convergence (loss is low or max epochs reached)

types of neural networks
1. Feedforward Neural Networks (FNN): data flows in one direction (input -> output)
2. Convolutional Neural Networks (CNN): great for image data (uses convolutional layers)
3. Recurrent Neural Networks (RNN): designed for sequential data (uses loops to maintain memory)
4. Transformers: state-of-the-art for NLP (uses self-attention mechanism)

common problems faced
1. Overfitting: model performs well on training data but poorly on unseen data (solution: regularization, dropout, more data)
2. Uderfitting :model has not trained on enough feautures. the model is trained on ver simple data with less features which is hard to find any pattern in them.
2. Vanishing/Exploding Gradients: gradients become too small or too large during backpropagation (solution: use ReLU activation, batch normalization)
3. Computational Cost: training deep networks can be expensive (solution: use GPUs, distributed training)
4. Hyperparameter Tuning: finding the right learning rate, batch size, etc. can be challenging (solution: grid search, random search, Bayesian optimization)
5. Data Quality: poor quality data can lead to bad performance (solution: data cleaning, augmentation)



In summary, neural networks are powerful tools for function approximation and learning complex patterns in data. They consist of layers of neurons that transform input data through weighted connections and activation functions. The training process involves forward passes to compute outputs, loss calculation to measure performance, and backpropagation to update weights and biases based on the computed gradients. Different types of neural networks are suited for various tasks, and common challenges include overfitting, vanishing/exploding gradients, computational cost, hyperparameter tuning, and data quality issues.




# Activation Functions (ReLU, GELU)
Activation functions introduce non-linearity to neural networks, enabling them to learn complex patterns
- ReLU (Rectified Linear Unit): f(x) = max(0, x) - simple, fast, helps with vanishing gradient problem
- GELU (Gaussian Error Linear Unit): f(x) = x * Φ(x) where Φ is cumulative distribution function of standard normal
- GELU is smoother than ReLU and often performs better in transformer models

# Optimizers (Adam, SGD)
- Optimizers determine how weights are updated during training based on computed gradients
- SGD (Stochastic Gradient Descent): w = w - α * ∇w - simple but can be slow to converge
- Adam (Adaptive Moment Estimation): combines momentum and adaptive learning rates for each parameter
- Adam maintains moving averages of gradients and their squared values for more efficient training

# Regularization (Dropout, L2)
- Regularization techniques prevent overfitting by adding constraints or noise during training
- Dropout: randomly sets some neurons to zero during training, forcing network to not rely on specific neurons
- L2 Regularization: adds penalty term λ||w||² to loss function, encouraging smaller weights
- Both help the model generalize better to unseen data

# Initialization Strategies
- Proper weight initialization is crucial for successful training and convergence
- Xavier/Glorot: initializes weights based on number of input and output neurons to maintain gradient flow
- He initialization: designed for ReLU activations, scales weights by sqrt(2/fan_in)
- Poor initialization can lead to vanishing/exploding gradients or slow convergence

# Gradients
- At its core, a gradient is the partial derivative of the loss function with respect to each parameter (weight or bias)
- Mathematically: ∇w = ∂L/∂w (partial derivative of loss L with respect to weight w)
- It represents the rate of change of the loss function as we change a specific parameter
- The gradient is a vector that points in the direction of steepest increase of the loss function
- Each element in the gradient vector corresponds to how much the loss would change if we slightly increased that specific parameter- Gradients are computed during backpropagation to update weights and biases

# Linear Unit to its core- A linear unit is a simple neuron without an activation function: y = w^T x + b
- It can only model linear relationships between input and output
- Not suitable for complex tasks like image recognition or natural language processing

# Non-Linear Unit
- A non-linear unit includes an activation function: y = σ(w^T x + b)
- It can model complex relationships and patterns in data
- Essential for deep learning, as it allows the network to learn non-linear mappings from inputs to outputs


# hidden state

## The Story of Hidden State: Memory in Neural Networks
 
- Imagine you're reading a book chapter by chapter. As you progress, you don't forget what happened earlier - 
- you carry forward the context and understanding from previous chapters to make sense of the current one.
- This is exactly what hidden state does in Recurrent Neural Networks (RNNs).

## The Journey of Hidden State:
- 1. At the beginning (t=0), the hidden state starts as zeros - like a blank slate
- 2. When the first word comes in, the RNN processes it along with the blank hidden state
- 3. The network produces two things: an output for that word AND an updated hidden state
- 4. This updated hidden state now contains "memories" of the first word
- 5. When the second word arrives, it's processed together with this memory-filled hidden state
- 6. The cycle continues: each new word + previous memories → new output + updated memories

 Real Example - Processing "The cat sat on the mat":
 Step 1: "The" + empty_memory → output_1 + memory_of_"The"
 Step 2: "cat" + memory_of_"The" → output_2 + memory_of_"The_cat" 
 Step 3: "sat" + memory_of_"The_cat" → output_3 + memory_of_"The_cat_sat"
 And so on...

## Why Hidden State Matters:
- Without it, each word would be processed in isolation (like having amnesia)
- With it, the network understands context: "bank" in "river bank" vs "money bank"
- It enables the network to handle variable-length sequences
- It's the key ingredient that makes RNNs suitable for language, time series, and sequential data

## The Problems with Hidden State that Led to Attention:

### 1. Information Bottleneck Problem:
- The hidden state is a fixed-size vector (e.g., 512 dimensions) that must compress ALL previous information
- For long sequences, early information gets "squeezed out" or forgotten
- Example: In a 1000-word document, information from the first paragraph is mostly lost by the end

### 2. Vanishing Gradient Problem:
- During backpropagation, gradients become exponentially smaller as they travel back through time
- The network struggles to learn long-range dependencies
- Words at the beginning of a sentence have minimal impact on words at the end

### 3. Sequential Processing Limitation:
- RNNs must process words one by one, in order (can't parallelize)
- This makes training very slow, especially for long sequences
- You can't process word 100 until you've processed words 1-99

### 4. Context Dilution:
- As the sequence gets longer, the hidden state becomes a "blurry average" of all previous words
- Important information from specific positions gets diluted
- The network can't easily "look back" to specific relevant parts of the input

### 5. Fixed Context Window:
- The hidden state has a practical limit on how far back it can remember
- Beyond a certain sequence length (typically 100-200 tokens), performance degrades significantly
- Long documents or conversations become problematic

## How Attention Solved These Problems:
- Attention allows the model to directly access ANY previous position in the sequence
- No information bottleneck - each position can attend to all others simultaneously
- Parallel processing - all positions can be computed at once
- Selective focus - the model can choose which parts of the input are most relevant
- No vanishing gradients across positions - direct connections to all previous states

## Technical Details:
- Hidden state is typically a vector (e.g., 256 or 512 dimensions)
- It gets updated at each time step using: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
- The same hidden state vector flows through the entire sequence
- Different sequences start with fresh hidden states
- In RNNs, the hidden state is a vector that captures information about the previous inputs in a sequence
- It allows the network to maintain memory and context across time steps- In RNNs, the hidden state is a vector that captures information about the previous inputs in a sequence
- It allows the network to maintain memory and context across time steps


# LSTM
- LSTM (Long Short-Term Memory) is a type of RNN designed to address the vanishing gradient problem
- It introduces a more complex architecture with gates to control the flow of information
- LSTM has three gates: input gate, forget gate, and output gate
- These gates allow the LSTM to selectively remember or forget information, enabling it to capture long-range dependencies in sequences

# GRU
- GRU (Gated Recurrent Unit) is a simplified version of LSTM that combines the input and forget gates into a single update gate
- It has fewer parameters than LSTM, making it faster to train while still addressing the vanishing gradient problem




Lets start from subset of AI
- Machine Learning (ML): algorithms that learn from data to make predictions or decisions
  Types:
  - Supervised Learning: learns from labeled data (classification, regression)
  - Unsupervised Learning: finds patterns in unlabeled data (clustering, dimensionality reduction)
  - Reinforcement Learning: learns through interaction with environment using rewards/penalties
  - Semi-supervised Learning: combines labeled and unlabeled data for training

- Deep Learning (DL): a subset of ML that uses neural networks with multiple layers to learn from data
  Types:
  - Feedforward Neural Networks: data flows in one direction
  - Convolutional Neural Networks (CNNs): specialized for image processing
  - Recurrent Neural Networks (RNNs): designed for sequential data
  - Generative Adversarial Networks (GANs): two networks competing to generate realistic data

- Neural Networks (NN): a type of DL model inspired by the structure of the human brain, consisting of layers of interconnected neurons that process and learn from data
  Types:
  - Perceptron: single layer neural network
  - Multi-layer Perceptron (MLP): multiple hidden layers
  - Autoencoders: encode input to lower dimension then decode back
  - Radial Basis Function Networks: use radial basis functions as activation

- Natural Language Processing (NLP): field of AI focused on enabling computers to understand, interpret, and generate human language
  Types:
  - Text Classification: categorizing text into predefined classes
  - Named Entity Recognition (NER): identifying entities like names, locations
  - Sentiment Analysis: determining emotional tone of text
  - Machine Translation: translating text between languages
  - Question Answering: providing answers to questions based on context
  - Text Generation: creating human-like text

- Transformers: a specific architecture of neural networks that uses self-attention mechanisms, particularly effective for natural language processing tasks
  Types:
  - BERT: bidirectional encoder for understanding context
  - GPT: generative pre-trained transformer for text generation
  - T5: text-to-text transfer transformer
  - Vision Transformer (ViT): applies transformer architecture to image processing


AI
 └── Machine Learning
      └── Deep Learning
           └── Neural Networks
                └── Transformers
                     ├── Encoder (used by BERT)
                     ├── Decoder (used by GPT)
                     └── Encoder-Decoder (used by T5, BART)


# Encoder
- The encoder processes the input data and generates a representation (embedding) that captures its meaning
- Used in models like BERT for understanding context and relationships in text

# Encoder-Only Pipeline (BERT, RoBERTa, ALBERT, DistilBERT)

## Training Pipeline
`Text Corpus → Tokenization → Embedding → Positional Encoding → Encoder Stack → Training Head → Loss → Backpropagation → Saved Weights`

## Inference Pipeline
`User Text → Tokenization → Embedding → Positional Encoding → Encoder Stack → Output Representation / Embedding / Classification`

## The Story of Encoder-Only Pipeline: From Raw Text to Understanding

### Step 1: Text Corpus - The Beginning of Understanding
1. Text Corpus: labeled data for fine-tuning (e.g., ("I love this movie", positive)) or unlabeled text for pre-training (MLM)
- Why: Encoder-only models learn from (text, label) pairs during fine-tuning, and unlabeled text during pre-training via Masked Language Modeling (MLM)
- What it solves: Provides the training signal — the model learns language structure and task-specific patterns
- How: Pre-training uses MLM (fill in masked words), fine-tuning uses labeled data (classification, NER, etc.)
- Where: This is the entry point — no target sequence to generate, just classify/tag/embed the input

### Step 2: Tokenization - Breaking Down the Language Barrier
2. Tokenization: split text into tokens and wrap with special tokens → [CLS] tokens... [SEP]
- Why: Neural networks need numbers, not strings. [CLS] token aggregates information from all tokens via self-attention and becomes the sentence representation for classification
- What it solves: Converts text into token IDs with BERT-style special tokens ([PAD], [CLS], [SEP], [UNK], [MASK])
- How: Uses algorithms like BPE or WordPiece. Single vocabulary (not separate src/tgt like encoder-decoder)
- Where: No `<sos>` token — encoder-only doesn't generate tokens, unlike decoder models

### Step 3: Embedding - Giving Meaning to Symbols
3. Embedding: convert token IDs to dense vectors scaled by √(d_model)
- Why: Token IDs are arbitrary integers with no semantic meaning — embeddings place tokens in continuous space where similar words are nearby
- What it solves: Maps discrete tokens to high-dimensional vectors (e.g., 256 or 768 dimensions)
- How: Learned lookup table (nn.Embedding). Only one embedding layer needed (single vocabulary)
- Where: Output shape: (batch, seq_len, d_model) — each token is now a dense vector

### Step 4: Positional Encoding - Adding the Dimension of Order
4. Positional Encoding: add sinusoidal position information to embeddings
- Why: Transformers process all tokens in parallel — no inherent notion of order. "not good" ≠ "good not"
- What it solves: Restores positional information. Encoder-only is BIDIRECTIONAL — no causal mask, every token sees every other token
- How: Sinusoidal functions (or learned embeddings in BERT). Added to embeddings, not concatenated
- Where: This is the key advantage — [CLS] can attend to the entire sentence in both directions

### Step 5: Encoder Stack - The Heart of Understanding
5. Encoder Stack: N layers of (Multi-Head Self-Attention + Feed-Forward Network) with residual connections + LayerNorm
- Why: Self-attention builds contextualized representations — same word gets different vectors in different contexts. Bidirectional attention (no mask!) means full context in both directions. Stacking layers = deeper understanding (syntax → semantics)
- What it solves: Captures long-range dependencies bidirectionally. Token 3 sees tokens 1,2,3,4,5... (unlike decoder which only sees 1,2,3)
- How: Each layer has 2 sub-layers: (1) Multi-Head Self-Attention (Q=K=V=input), (2) Position-wise FFN (Linear → ReLU → Linear)
- Where: [CLS] output at position 0 aggregates the entire sentence's meaning. Output shape unchanged: (batch, seq_len, d_model) but deeply contextualized

### Step 6: Training Head - Adapting to the Task
6. Training Head: task-specific layer on top of encoder output
- Why: The encoder produces general-purpose representations — the head adapts them to a specific task. For classification, only the [CLS] token's output is needed (it has seen the whole sentence)
- What it solves: Classification → Linear(d_model, n_classes) on [CLS] output. NER → Linear(d_model, n_tags) on all token outputs. MLM → Linear(d_model, vocab_size) on masked tokens
- How: Simple head (Dropout → Linear → Tanh → Linear) — the encoder does the heavy lifting
- Where: Output: (batch, n_classes) → one prediction per input sentence

### Step 7: Loss - Measuring Classification Error
7. Loss: Cross-Entropy Loss comparing predicted class probabilities vs true label
- Why: We need a single number to tell the model how wrong it is. -log(probability of correct class)
- What it solves: Much simpler than seq2seq loss — one prediction per sentence, not per token. No ignore_index needed (no padding in labels)
- How: CrossEntropyLoss on (batch, n_classes) logits vs (batch,) labels
- Where: If model is confident and correct → low loss. If wrong → high loss

### Step 8: Backpropagation & Training - Learning from Mistakes
8. Backpropagation: compute gradients of loss w.r.t. all parameters, then update weights
- Why: The loss tells us how wrong the model is, gradients tell us the direction to adjust each parameter
- What it solves: Full model assembled: Embedding + PE + Encoder + Classification Head. Padding mask ensures padded tokens don't affect attention
- How: Forward → Loss → loss.backward() → clip_grad_norm → optimizer.step()
- Where: Training loop runs for N epochs until accuracy converges (~100% on small datasets)

### Step 9: Saved Weights - Persisting the Trained Model
9. Saved Weights: save model weights (.pth), config (.json), tokenizer vocab (.json), and label mapping (.json)
- Why: Training is expensive — save once, load for inference without retraining
- What it solves: Complete model = Weights + Config + Vocab + Labels. Can be shared, deployed, or fine-tuned further
- How: torch.save(model.state_dict()) for weights, json.dump for config/vocab/labels
- Where: This completes the Training Pipeline

### Step 10: Inference - Classification & Embeddings
10. Inference: load saved model and use for classification, embeddings, or similarity
- Why: model.eval() + torch.no_grad() disables dropout and skips gradient computation
- What it solves: Classification (sentiment prediction), Sentence Embeddings ([CLS] output for similarity/retrieval)
- How: Single forward pass — no autoregressive loop! Unlike decoder models, encoder-only runs ONCE per input
- Where: [CLS] → Classification Head → softmax → predicted class, OR [CLS] output directly → cosine similarity for retrieval


# decoder
- The decoder generates output based on the encoded representation from the encoder
- Used in models like GPT for text generation and T5 for text-to-text tasks

# decoder pipeline
## The Story of Decoder Pipeline: From Understanding to Generation

### Step 1: Input - The Seed of Creation
1. Input: encoded representation from the encoder (e.g., contextualized embeddings)
- Why: The decoder needs a rich understanding of the input to generate meaningful output
- What it solves: Provides the necessary context and information for generation
- How: Takes the output from the encoder as the starting point for generation
- Where: This is the entry point for the decoder - it relies on the encoder's understanding to create output

### Step 2: Masked Multi-Head Self-Attention - Focusing on the Future
2. Masked Multi-Head Self-Attention: compute attention scores while preventing access to future tokens
- Why: During generation, the model should not "cheat" by looking at future tokens it hasn't generated yet
- What it solves: Ensures the autoregressive nature of generation, maintaining causality
- How: Applies a mask to the attention scores to block access to future positions
- Where: Critical for language generation tasks to maintain coherence and logical flow

### Step 3: Encoder-Decoder Attention - Bridging Understanding and Creation
3. Encoder-Decoder Attention: compute attention scores between decoder states and encoder outputs
- Why: The decoder needs to reference the encoder's understanding to generate relevant output
- What it solves: Allows the decoder to focus on relevant parts of the input when generating each token
- How: Each decoder position attends to all encoder positions to gather necessary information
- Where: This is what allows the model to generate contextually relevant output based on the input

### Step 4: Feed-Forward Network - Refining the Creation
4. Feed-Forward Network: apply non-linear transformations to the attention output
- Why: Similar to the encoder, we need to process and refine the information before generating output
- What it solves: Adds computational depth and non-linearity to transform attention patterns into meaningful output
- How: Two linear layers with activation function (usually ReLU or GELU) in between
- Where: Applied to each position independently, allowing for position-specific processing

### Step 5: Output - The Final Creation
5. Output: generated token probabilities (e.g., for next word prediction)
- Why: We need to convert the decoder's internal representations into actual tokens that can be output
- What it solves: Provides the mechanism for generating text based on the decoder's understanding and attention
- How: A linear layer followed by softmax to produce probabilities for each token in the vocabulary
- Where: This is the final step in the decoder pipeline, producing the output that can be used for generation or other tasks


# decoder only model
- A decoder-only model consists solely of the decoder architecture without a separate encoder
- Examples include GPT (Generative Pre-trained Transformer) which uses only the decoder for autoregressive text generation
- In a decoder-only model, the input is typically a sequence of tokens (e.g., a prompt), and the model generates the next token in the sequence based on the previous tokens
- Decoder-only models are designed for tasks like text generation, where the model generates output based on a given prompt or context

--------------------------------------------------------------------------

Yes — here are the **pipelines only**.

## 1. Encoder-only model

### Training

`Text Corpus → Tokenization → Embedding → Positional Encoding → Encoder Stack → Training Head → Loss → Backpropagation → Saved Weights (.pth/.bin/.safetensors)`

### Inference

`User Text → Tokenization → Embedding → Positional Encoding → Encoder Stack → Output Representation / Embedding / Classification`

---

## 2. Decoder-only model

### Training

`Text Corpus → Tokenization → Embedding → Positional Encoding → Decoder Stack (Masked Self-Attention + FFN) → Next Token Prediction Head → Loss → Backpropagation → Saved Weights (.pth/.bin/.safetensors)`

### Inference

`User Prompt → Tokenization → Embedding → Positional Encoding → Decoder Stack → Next Token Prediction → Loop Until End Token`

---

## 3. Encoder–Decoder model

### Training

`Input Text Corpus → Tokenization → Embedding → Positional Encoding → Encoder Stack → Encoder Output → Decoder Stack (Masked Self-Attention + Cross-Attention + FFN) → Output Token Prediction Head → Loss → Backpropagation → Saved Weights (.pth/.bin/.safetensors)`

### Inference

`User Input → Tokenization → Embedding → Positional Encoding → Encoder Stack → Encoder Output → Decoder Stack → Token-by-Token Generation`

---

## Model file

`Trained Weights + Architecture Config + Tokenizer Files = Model`
