## Step 1. Text corpus
Raw Text for Causal Language Modeling
 - Input
    - Raw unlabeled text, no pairs, no labels
    - "The cat is on the roof."
    - "The dog is in the yard."
    - "The bird is in the tree."


## Step 2. Tokenization
Tokenization
 - Input
    - Raw text
 - Output
    - Token IDs (using vocabulary built from corpus)
    - Special tokens: `<pad>`(0), `<eos>`(1), `<unk>`(2)
    - Creates input and target by shifting the full sequence by 1 position:
      - Input:  `[the, cat, sat, on, the, mat]` ← everything except last
      - Target: `[cat, sat, on, the, mat, <eos>]` ← everything except first

## Step 3. Embeddings
 - Input
    - Token ids
 - Output
    - Dense vector representations of tokens (embeddings)

## Step 4. Positional Embedding
 - Input
    - Token embeddings
 - Output
    - Dense vector + position of the tokens in it (sinusoidal positional encoding added to embeddings)

## Stack 5. Decoder Stack

A stack of N identical layers, each containing:
1. **Masked (Causal) Multi-Head Self-Attention** — each token can only attend to itself and **previous** tokens
2. **Feed-Forward Network** — position-wise non-linear transformation
3. **Residual connections + Layer Normalization**

- Input
    - Token embeddings with positional information + causal mask
- Output
    - Contextualized token representations after passing through multiple layers of self-attention and feed-forward networks

## Step 6. Next token prediction head
- Input
    - Contextualized token representations from the Decoder stack
    - Decoder output shape: `(batch, seq_len, d_model)`
- Output
    - Probability distribution over the vocabulary for the next token prediction
    - Logits shape: `(batch, seq_len, vocab_size)`
    - Uses weight tying: embedding and LM head share the same weight matrix

## Step 7. Cross Entropy Loss
**Cross-Entropy Loss** computed at **every position** — comparing the predicted next token vs the actual next token.
- Input
    - Predicted token probabilities (logits) from the Next Token Prediction Head
    - Actual next token labels (shifted input tokens)
- Output
    - Scalar loss value that quantifies how well the model's predictions match the actual next tokens in the training data


## Step 8. Backpropagation and Training
 - Input
    - Input batch and target batch (padded to same length)
 - Output
    - Loss (CrossEntropyLoss with `ignore_index=PAD_IDX`)
    - Gradient clipping (`clip_grad_norm_`) to prevent exploding gradients
    - Updates weights via Adam optimizer and trains model


## Step 9. Saved weigths
- Input
    - Trained model weights after training
- Output
    - Saved model weights that can be loaded for inference or further training

## Step 10. Inferences
- Input
    - New input text (prompt)
    - Trained model weights
- Output
    - Generated text based on the input prompt, using the trained model to predict the next tokens iteratively until a stopping criterion is met (e.g., end token, max length)