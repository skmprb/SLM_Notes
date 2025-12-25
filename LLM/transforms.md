# Understanding Recurrent Models and Their Limitations

## How Recurrent Models Work

Recurrent neural networks (RNNs), LSTMs, and GRUs process sequences (like sentences) one step at a time. At each step t, the model:

1. Takes the current input (word or token)
2. Looks at the previous hidden state h_{t-1}
3. Produces a new hidden state h_t

The computation flows sequentially:
```
h_1 → h_2 → h_3 → ... → h_T
```

Each step depends on the previous step.

## The Core Problem: Sequential Dependency

Because each hidden state requires the previous one, you **cannot compute steps in parallel**. You can't jump to h_50 until h_49 is complete.

### This Creates Bottlenecks:
- **Slower training** - no parallelization across sequence steps
- **Limited sequence length** - memory constraints from sequential processing
- **Smaller batch sizes** - memory usage grows with sequence length

## Attempted Solutions

Researchers have tried various optimization techniques:

- **Factorization**: Breaking computations into cheaper pieces
- **Conditional computation**: Only computing relevant parts based on input

While these improve efficiency somewhat, they don't solve the fundamental issue.

## The Fundamental Constraint

No matter what optimizations are applied, RNNs remain fundamentally sequential:
- h_1 must be computed before h_2
- h_2 must be computed before h_3
- And so on...

This sequential dependency cannot be eliminated from the RNN architecture.

## Concrete Example: Sequential Addition

Consider processing the numbers `[2, 5, 3, 7]`:

```
Step 1: Start → read 2 → state = 2
Step 2: state 2 → read 5 → state = 7  
Step 3: state 7 → read 3 → state = 10
Step 4: state 10 → read 7 → state = 17
```

You cannot skip to step 4 without completing steps 1-3 first.

### Scaling the Problem

With 1000 numbers instead of 4:
- Must process one by one
- Cannot parallelize across the sequence
- Training becomes prohibitively slow

## Visual Comparison

**RNN (Sequential Processing):**
```
h_1 → h_2 → h_3 → h_4 → ... → h_1000
```
*Only one computation at a time*

**Transformer (Parallel Processing):**
```
h_1   h_2   h_3   h_4   ...   h_1000
```
*All positions computed simultaneously*

---

# Training Methods of Different Models

Feed-forward networks process all input at once and have no memory, so they can't handle sequences by themselves.

Recurrent models (RNN/LSTM/GRU) read input one step at a time, keeping a memory of previous steps — but this makes them slow and not parallelizable.

CNNs process data in parallel using local windows, good for patterns but not long-range relationships.

Transformers look at all tokens at the same time using attention, capturing long-range relationships and scaling easily — which is why they power modern LLMs.

---

# The Evolution: Why We Moved from RNNs to CNNs to Transformers

## RNNs were too slow — even with attention

Attention improved how RNNs *use* memory, **but it did NOT remove the core bottleneck:**

### RNNs must process sequences *one token at a time*
### Cannot be parallelized
### Extremely slow for long sequences
### Impossible to scale to millions of tokens

So even though RNNs had memory, they were **too sequential**.

**CNNs = massively parallel → much faster.**

## CNNs *simulate* long-range memory using layers

Even though CNNs don't have a recurrent hidden state, they can still model long dependencies by stacking layers.

Each layer expands the "receptive field"—the range a token can interact with.

For example:
* Layer 1: each token sees its 3 neighbors
* Layer 2: can see neighbors-of-neighbors
* Layer 3: can see even farther

After many layers, a CNN can connect two very distant tokens.

So CNNs don't have a "hidden state," but they **build memory through depth**.

## CNNs allow full parallelism (the big win)

Unlike RNNs, CNNs let you compute:
* all positions
* all convolutions
* all layers

**at the same time.**

This made training much faster, which was critical before GPUs got extremely powerful.

## Attention alone wasn't enough yet

Attention helps with long-range dependencies, but **if you put attention on top of an RNN**, it's still stuck in:
* step-by-step processing
* long training times
* difficulty scaling
* limited sequence lengths

So the community tried:

### RNN + Attention
→ still sequential, still slow

### CNN + Attention
→ fully parallel, much faster
→ but long-range memory requires many layers

## Transformers = best of both worlds

Transformers solved BOTH problems:
* **parallelism** like CNNs
* **long-range memory** like attention
* **constant** (not growing) steps to connect distant tokens

Transformers removed:
* hidden state bottleneck (RNN problem)
* deep-layer dependency buildup (CNN problem)

**One-sentence answer**: We moved from RNNs to CNNs because RNNs were too slow and sequential, even with attention; CNNs allowed full parallel computation, and although they lacked a hidden state, they could still capture long-range dependencies through stacked layers — making them a faster alternative until the Transformer solved everything.

---

# NLP Timeline: The Clear Path to Transformers

## CNNs Background
The idea of CNNs first appeared in 1980 with Kunihiko Fukushima's Neocognitron. This was the first model to use convolution + pooling ideas.

Modern CNNs were introduced in 1989–1998 by Yann LeCun with LeNet-1 → LeNet-5, which became the foundation of today's CNNs.

## ① RNNs (1997 → 2014)

**Used because:**
* They have a hidden state → can model sequences naturally

**Problem:**
* **Too slow**, must process *one token at a time*
* Can't parallelize
* Hard to capture long-range dependencies
* Training collapses on long sentences

## ② RNN + Attention (2014 → 2016)

**What attention fixed:**
* Helped the decoder look back at ALL encoder states
* Made long-range dependencies learnable
* Improved translation massively

**But:**
* The RNN was *still sequential*
* Training remained slow
* Long sequences still hard
* Couldn't scale to massive datasets

⟶ **Attention improved quality but didn't fix speed.**

## ③ CNN-based sequence models (2016 → 2017)

Models: **WaveNet, ByteNet, ConvS2S, Extended Neural GPU**

**Why CNNs were tried:**
* CNNs are **fully parallel** (unlike RNNs)
* Much faster for long sequences
* Deep layers give a "receptive field" that grows with depth
  → can simulate long-range memory

**But:**
* To connect distant tokens, CNNs need **many layers**
* Long-range dependencies still not efficient
* Memory grows slowly with depth
* More layers → bigger models → more compute

⟶ **CNNs fixed speed but not long-range dependency efficiency.**

## ④ Transformers (2017 → now)

**Breakthrough idea:**
* Use **only attention**, no RNN, no CNN
* Every token can see every other token in **one step**
  → constant-time global communication
* Fully parallel
* Handles long-range relationships perfectly

**Result:**
* Unprecedented scalability
* Massive speedups
* Allowed training on billions–trillions of tokens
* Enabled GPT, BERT, LLaMA, Claude, etc.

⟶ **Transformers solved *both* problems at once: sequential bottlenecks + long-range dependencies.**

**One-sentence summary**: NLP moved from RNNs to CNNs to Transformers because attention improved quality but RNNs were still too slow, CNNs were fast but struggled with long-range dependencies, and Transformers finally combined parallelism with perfect long-range modeling.

---

# Attention Mechanisms Explained

## Self-Attention Mechanism

Self-attention (also called intra-attention) is a technique where a model looks at different positions within the same sequence—for example, different words in the same sentence—to figure out how they relate to each other. By doing this, the model builds a richer, context-aware representation of the sequence.

This mechanism has been successfully used in many tasks, such as:
* Reading comprehension – understanding a passage and answering questions about it
* Abstractive summarization – generating a summary that uses new wording rather than copying
* Textual entailment – deciding whether one sentence logically follows from another
* Universal sentence representation learning – creating general-purpose embeddings for sentences

**In short**: Self-attention helps the model understand how each part of a sequence depends on the others, and this has proven useful across many natural language tasks.

## Attention Before Transformers

Attention mechanisms let a model focus on any part of a sequence, no matter how far away it is. For example, in a long sentence, attention lets the model directly connect word 1 with word 20.

Before Transformers, attention was usually added on top of recurrent models (RNNs/LSTMs). So the RNN would still process the sequence step-by-step, but attention would help it look back at earlier states more effectively.

**In short**: Attention helped RNNs handle long-range dependencies, but it did NOT remove the slow, sequential nature of RNNs.

## End-to-End Memory Networks

End-to-end memory networks use a type of recurrent attention mechanism—meaning they repeatedly apply attention over memories—rather than processing sequences step by step using an RNN. These memory networks have shown good performance on tasks like:
* Simple question answering (where the language and reasoning are limited)
* Language modeling (predicting the next word in a sequence)

However, the authors point out something important:

**The Transformer is different and new**

They say that, as far as they know, the Transformer is the first model for sequence-to-sequence tasks (transduction tasks like translation) that:
* Depends entirely on self-attention for both input and output representations
* Does not use:
  * sequence-aligned RNNs (like LSTMs or GRUs)
  * convolutions (like in ConvS2S or ByteNet)

In other words, the Transformer removes recurrence and convolution completely and replaces them with self-attention everywhere.

---

# Key Takeaway

Recurrent models' sequential nature creates an insurmountable parallelization barrier. Each hidden state's dependency on the previous one makes long sequences computationally expensive and slow to train.

This fundamental limitation motivated the development of **Transformers**, which eliminate sequential dependencies and enable full parallelization across sequence positions, leading to the current era of large language models and advanced AI systems.