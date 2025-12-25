# 📚 NLP & ML Abbreviations Glossary

## Core Machine Learning

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **AI** | Artificial Intelligence | Computer systems that can perform tasks requiring human intelligence |
| **ML** | Machine Learning | Algorithms that learn patterns from data without explicit programming |
| **DL** | Deep Learning | Neural networks with multiple layers for complex pattern recognition |
| **NN** | Neural Network | Computing system inspired by biological neural networks |
| **CNN** | Convolutional Neural Network | Deep learning architecture for processing grid-like data (images) |
| **RNN** | Recurrent Neural Network | Neural network designed for sequential data processing |
| **LSTM** | Long Short-Term Memory | RNN variant that can learn long-term dependencies |
| **GRU** | Gated Recurrent Unit | Simplified version of LSTM with fewer parameters |
| **GAN** | Generative Adversarial Network | Two neural networks competing to generate realistic data |
| **VAE** | Variational Autoencoder | Generative model that learns data distributions |

## Natural Language Processing

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **NLP** | Natural Language Processing | AI field focused on human-computer language interaction |
| **NLU** | Natural Language Understanding | Subset of NLP focused on comprehending text meaning |
| **NLG** | Natural Language Generation | AI system that produces human-like text |
| **LLM** | Large Language Model | Massive neural networks trained on vast text corpora |
| **GPT** | Generative Pre-trained Transformer | Autoregressive language model architecture |
| **BERT** | Bidirectional Encoder Representations from Transformers | Bidirectional transformer for understanding context |
| **T5** | Text-to-Text Transfer Transformer | Unified framework treating all NLP tasks as text generation |
| **ELECTRA** | Efficiently Learning an Encoder that Classifies Token Replacements Accurately | Pre-training method using replaced token detection |
| **RoBERTa** | Robustly Optimized BERT Pretraining Approach | Improved BERT training methodology |
| **ALBERT** | A Lite BERT | Parameter-efficient version of BERT |

## Tokenization & Text Processing

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **BPE** | Byte Pair Encoding | Subword tokenization algorithm merging frequent character pairs |
| **SentencePiece** | Sentence Piece | Language-independent subword tokenizer |
| **WordPiece** | Word Piece | Subword tokenization used in BERT |
| **OOV** | Out-of-Vocabulary | Words not present in the model's vocabulary |
| **UNK** | Unknown | Token representing unrecognized words |
| **PAD** | Padding | Special token to make sequences equal length |
| **CLS** | Classification | Special token for classification tasks |
| **SEP** | Separator | Token to separate different text segments |
| **MASK** | Mask | Token used in masked language modeling |
| **BOS** | Beginning of Sequence | Token marking sequence start |
| **EOS** | End of Sequence | Token marking sequence end |

## Transformer Architecture

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **MHA** | Multi-Head Attention | Attention mechanism with multiple parallel heads |
| **SA** | Self-Attention | Attention mechanism within the same sequence |
| **FFN** | Feed-Forward Network | Fully connected layers in transformer blocks |
| **PE** | Positional Encoding | Method to inject sequence position information |
| **LayerNorm** | Layer Normalization | Normalization technique applied to layer inputs |
| **QKV** | Query, Key, Value | Three matrices used in attention computation |
| **MHSA** | Multi-Head Self-Attention | Self-attention with multiple parallel heads |

## Training & Optimization

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **SGD** | Stochastic Gradient Descent | Optimization algorithm using random data samples |
| **Adam** | Adaptive Moment Estimation | Optimization algorithm with adaptive learning rates |
| **AdamW** | Adam with Weight Decay | Adam optimizer with decoupled weight decay |
| **LR** | Learning Rate | Step size for parameter updates during training |
| **BS** | Batch Size | Number of samples processed before updating parameters |
| **Epoch** | Epoch | One complete pass through the training dataset |
| **Dropout** | Dropout | Regularization technique randomly setting neurons to zero |
| **L1/L2** | L1/L2 Regularization | Penalty terms to prevent overfitting |
| **EMA** | Exponential Moving Average | Smoothing technique for model parameters |
| **WD** | Weight Decay | Regularization technique shrinking model weights |

## Evaluation Metrics

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **BLEU** | Bilingual Evaluation Understudy | Metric for evaluating machine translation quality |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | Metric for text summarization evaluation |
| **METEOR** | Metric for Evaluation of Translation with Explicit ORdering | Translation evaluation considering synonyms and stemming |
| **BERTScore** | BERT Score | Evaluation metric using BERT embeddings for similarity |
| **GLUE** | General Language Understanding Evaluation | Benchmark for evaluating NLP models |
| **SuperGLUE** | Super General Language Understanding Evaluation | More challenging version of GLUE benchmark |
| **SQuAD** | Stanford Question Answering Dataset | Reading comprehension benchmark |
| **CoLA** | Corpus of Linguistic Acceptability | Grammar acceptability classification task |

## Model Architecture Components

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **MLP** | Multi-Layer Perceptron | Fully connected neural network with multiple layers |
| **ResNet** | Residual Network | Architecture using skip connections to enable deeper networks |
| **Attention** | Attention Mechanism | Method to focus on relevant parts of input sequence |
| **Embedding** | Embedding Layer | Dense vector representation of discrete tokens |
| **Softmax** | Soft Maximum | Function converting logits to probability distribution |
| **ReLU** | Rectified Linear Unit | Activation function returning max(0, x) |
| **GELU** | Gaussian Error Linear Unit | Smooth activation function used in transformers |
| **Swish** | Swish Activation | Smooth activation function: x * sigmoid(x) |

## Data & Preprocessing

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **Corpus** | Text Corpus | Large collection of written texts for training |
| **Dataset** | Data Set | Collection of data used for training or evaluation |
| **Preprocessing** | Data Preprocessing | Cleaning and preparing raw data for model training |
| **Augmentation** | Data Augmentation | Techniques to artificially increase dataset size |
| **Normalization** | Text Normalization | Standardizing text format (case, punctuation, etc.) |
| **Stemming** | Stemming | Reducing words to their root form |
| **Lemmatization** | Lemmatization | Reducing words to their dictionary form |
| **POS** | Part-of-Speech | Grammatical category of words (noun, verb, etc.) |
| **NER** | Named Entity Recognition | Identifying and classifying named entities in text |
| **Chunking** | Text Chunking | Grouping words into meaningful phrases |

## Hardware & Infrastructure

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **GPU** | Graphics Processing Unit | Specialized processor for parallel computations |
| **TPU** | Tensor Processing Unit | Google's custom chip for machine learning workloads |
| **CPU** | Central Processing Unit | Main processor handling general computing tasks |
| **VRAM** | Video Random Access Memory | Memory on graphics cards for storing data |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's parallel computing platform |
| **FP16** | 16-bit Floating Point | Half-precision format to reduce memory usage |
| **FP32** | 32-bit Floating Point | Single-precision floating point format |
| **Mixed Precision** | Mixed Precision Training | Using both FP16 and FP32 for efficient training |

## Advanced Techniques

| Abbreviation | Full Form | Explanation |
|--------------|-----------|-------------|
| **LoRA** | Low-Rank Adaptation | Parameter-efficient fine-tuning method |
| **QLoRA** | Quantized Low-Rank Adaptation | LoRA combined with quantization for efficiency |
| **PEFT** | Parameter-Efficient Fine-Tuning | Methods to fine-tune models with fewer parameters |
| **RLHF** | Reinforcement Learning from Human Feedback | Training method using human preferences |
| **PPO** | Proximal Policy Optimization | Reinforcement learning algorithm |
| **DPO** | Direct Preference Optimization | Alternative to RLHF for preference learning |
| **RAG** | Retrieval-Augmented Generation | Combining retrieval with generation for better responses |
| **ICL** | In-Context Learning | Learning from examples provided in the input |
| **CoT** | Chain-of-Thought | Prompting technique encouraging step-by-step reasoning |
| **Few-Shot** | Few-Shot Learning | Learning from very few examples |
| **Zero-Shot** | Zero-Shot Learning | Performing tasks without specific training examples |