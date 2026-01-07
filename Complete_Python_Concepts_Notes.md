# Complete Python Concepts Notes

## Table of Contents
1. [String Methods & Text Processing](#1-string-methods--text-processing)
2. [NumPy Fundamentals](#2-numpy-fundamentals)
3. [Matrix Operations & Neural Networks](#3-matrix-operations--neural-networks)
4. [NumPy Advanced Functions](#4-numpy-advanced-functions)
5. [Advanced Python Syntax](#5-advanced-python-syntax)
6. [Type Hints & Annotations](#6-type-hints--annotations)
7. [Lambda Functions](#7-lambda-functions)
8. [Dictionary & List Comprehensions](#8-dictionary--list-comprehensions)
9. [Classes & Object-Oriented Programming](#9-classes--object-oriented-programming)
10. [Collections Module](#10-collections-module)
11. [Tuples & Advanced Data Structures](#11-tuples--advanced-data-structures)
12. [Enumerate & Iteration](#12-enumerate--iteration)
13. [Practical Examples](#13-practical-examples)

---

## 1. String Methods & Text Processing

### 1.1 `str.translate()` and `str.maketrans()`

**Purpose**: Remove or replace characters in strings efficiently

**What `str.maketrans()` Does**:
`str.maketrans()` creates a translation table - a mapping that tells Python how to transform characters.

**Syntax**: `str.maketrans(from, to, delete)`
- **from**: Characters to replace FROM
- **to**: Characters to replace TO  
- **delete**: Characters to DELETE

**How It Works**:
```python
# Example: Remove punctuation
translator = str.maketrans('', '', string.punctuation)
# Creates mapping: {33: None, 34: None, 44: None, ...}
# Meaning: ASCII codes of punctuation → None (delete)

# Example: Replace vowels
replace_table = str.maketrans('aeiou', '12345')
# Creates: {97: 49, 101: 50, 105: 51, 111: 52, 117: 53}
# Meaning: 'a'→'1', 'e'→'2', 'i'→'3', 'o'→'4', 'u'→'5'
```

**Key Point**: `maketrans()` creates the "rules" - `translate()` applies them.

**Important Limitation**: Works only with **individual characters**, not whole words.

**For Word Replacement, Use**:
- **`str.replace()`**: Simple word replacement
- **`re.sub()`**: Pattern-based word replacement

```python
# ✅ Character-level (translate works)
translator = str.maketrans('aeiou', '12345')
text = "hello"
result = text.translate(translator)  # "h2ll4"

# ❌ Word-level (translate won't work)
# Can't do: str.maketrans('hello', 'goodbye')

# ✅ Use replace() for words
text = "hello world hello"
result = text.replace('hello', 'goodbye')  # "goodbye world goodbye"

# ✅ Use re.sub() for pattern matching
import re
text = "The cat and the dog"
result = re.sub(r'\bcat\b', 'dog', text)  # "The dog and the dog"
```

```python
import string

# Basic usage
text = "Hello, World! How are you?"

# Remove punctuation
translator = str.maketrans('', '', string.punctuation)
# string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
clean_text = text.translate(translator)
print(clean_text)  # "Hello World How are you"

# Replace characters
replace_table = str.maketrans('aeiou', '12345')
replaced = text.translate(replace_table)
print(replaced)  # "H2ll4, W4rld! H4w 1r2 y45?"

# Remove specific characters
remove_vowels = str.maketrans('', '', 'aeiouAEIOU')
no_vowels = text.translate(remove_vowels)
print(no_vowels)  # "Hll, Wrld! Hw r y?"
```

**Advanced Usage**:
```python
# Custom translation for text preprocessing
def clean_text_for_nlp(text):
    # Remove punctuation and digits
    translator = str.maketrans('', '', string.punctuation + string.digits)
    return text.translate(translator).lower().strip()

text = "Hello123, World! @#$"
clean = clean_text_for_nlp(text)
print(clean)  # "hello world"
```

### 1.2 `string.punctuation`

**Contains all ASCII punctuation characters**:
```python
import string

print(string.punctuation)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# Common uses
def remove_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation)

def count_punctuation(text):
    return sum(1 for char in text if char in string.punctuation)

text = "Hello, world! How are you?"
print(remove_punctuation(text))  # "Hello world How are you"
print(count_punctuation(text))   # 3
```

### 1.3 `split()` Method

**Basic splitting**:
```python
text = "Natural language processing is amazing"
words = text.split()  # Default: split on whitespace
print(words)  # ['Natural', 'language', 'processing', 'is', 'amazing']

# Split with custom delimiter
csv_data = "apple,banana,cherry,date"
fruits = csv_data.split(',')
print(fruits)  # ['apple', 'banana', 'cherry', 'date']

# Limit splits
text = "one-two-three-four-five"
limited = text.split('-', 2)  # Split only first 2 occurrences
print(limited)  # ['one', 'two', 'three-four-five']
```

**Advanced splitting**:
```python
import re

def advanced_tokenize(text):
    # Split on whitespace and punctuation, keep punctuation
    tokens = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
    return tokens

text = "Hello, world! How are you?"
tokens = advanced_tokenize(text)
print(tokens)  # ['hello', ',', 'world', '!', 'how', 'are', 'you', '?']
```

**How `re.findall()` Works**:
- **`re.findall(pattern, string)`**: Finds ALL matches of pattern in string, returns as list
- **Pattern `r'\b\w+\b|[.,!?;]'`**:
  - `\b\w+\b`: Matches whole words (word boundaries + word characters)
  - `|`: OR operator
  - `[.,!?;]`: Matches specific punctuation
- **Spaces are ignored**: Not in pattern, so they're skipped
- **Word boundaries (`\b`)**: Automatically separate words from punctuation

**Result**: Words and punctuation become separate tokens, perfect for NLP tasks.

---

## 2. NumPy Fundamentals

### 2.1 NumPy Import Convention

```python
import numpy as np  # Standard convention
```

### 2.2 `np.ndarray` - N-dimensional Arrays

**Creating arrays**:
```python
# 1D arrays
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d.shape)  # (5,)
print(arr_1d.ndim)   # 1

# 2D arrays
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d.shape)  # (2, 3)
print(arr_2d.ndim)   # 2

# 3D arrays
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr_3d.shape)  # (2, 2, 2)
print(arr_3d.ndim)   # 3
```

**Array properties**:
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"Shape: {arr.shape}")      # (2, 3)
print(f"Size: {arr.size}")        # 6
print(f"Dtype: {arr.dtype}")      # int64
print(f"Dimensions: {arr.ndim}")  # 2
print(f"Item size: {arr.itemsize}") # 8 bytes
```

### 2.3 `np.zeros()` and Array Initialization

```python
# Create arrays filled with zeros
zeros_1d = np.zeros(5)           # [0. 0. 0. 0. 0.]
zeros_2d = np.zeros((3, 4))      # 3x4 matrix of zeros
zeros_3d = np.zeros((2, 3, 4))   # 2x3x4 tensor of zeros

# Specify data type
zeros_int = np.zeros(5, dtype=int)     # [0 0 0 0 0]
zeros_bool = np.zeros(3, dtype=bool)   # [False False False]

# Other initialization functions
ones = np.ones((2, 3))           # Matrix of ones
full = np.full((2, 3), 7)        # Matrix filled with 7
empty = np.empty((2, 3))         # Uninitialized matrix
identity = np.eye(3)             # 3x3 identity matrix
```

**Practical usage in ML**:
```python
# Initialize weight matrices
def initialize_weights(input_size, output_size):
    # Xavier initialization
    weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
    bias = np.zeros(output_size)
    return weights, bias

# Create one-hot encoding matrix
def create_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

labels = [0, 1, 2, 1, 0]
one_hot = create_one_hot(labels, 3)
print(one_hot)
```

---

## 3. Matrix Operations & Neural Networks

### 3.1 Matrix Multiplication with `@` Operator

**The `@` operator performs matrix multiplication (introduced in Python 3.5)**:

```python
import numpy as np

# @ is equivalent to np.dot() for 2D arrays
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# These are equivalent:
result1 = A @ B
result2 = np.dot(A, B)
result3 = np.matmul(A, B)

print(result1)
# [[19 22]
#  [43 50]]

# Matrix-vector multiplication
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([1, 2, 3])
result = matrix @ vector  # Shape: (2, 3) @ (3,) = (2,)
print(result)  # [14 32]
```

**Dimension rules for matrix multiplication**:
```python
# (m, n) @ (n, p) = (m, p)
# Inner dimensions must match!

A = np.random.randn(3, 4)  # 3 rows, 4 columns
B = np.random.randn(4, 5)  # 4 rows, 5 columns
C = A @ B                  # Result: (3, 5)

# Common error:
# A @ A  # Error! (3, 4) @ (3, 4) - inner dimensions don't match
```

### 3.2 Neural Network Weight Matrices

**Understanding weight matrix naming conventions**:

```python
class NeuralNetworkWeights:
    def __init__(self, input_size, hidden_size, output_size):
        # Weight matrix naming convention:
        # W[destination][source] - from source to destination
        
        # Wxh: from input (x) to hidden (h)
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.1
        
        # Whh: from hidden (h) to hidden (h) - recurrent connection
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        
        # Why: from hidden (h) to output (y)
        self.Why = np.random.randn(hidden_size, output_size) * 0.1
        
        # Bias vectors
        self.bh = np.zeros((1, hidden_size))   # Hidden bias
        self.by = np.zeros((1, output_size))   # Output bias

# Example dimensions for RNN:
input_size = 10   # Input features
hidden_size = 20  # Hidden units
output_size = 5   # Output classes

weights = NeuralNetworkWeights(input_size, hidden_size, output_size)
print(f"Wxh shape: {weights.Wxh.shape}")  # (10, 20)
print(f"Whh shape: {weights.Whh.shape}")  # (20, 20)
print(f"Why shape: {weights.Why.shape}")  # (20, 5)
```

### 3.3 RNN Forward Pass Computation

**Step-by-step breakdown of RNN computation**:

```python
def rnn_step_detailed(input_t, h_prev, Wxh, Whh, Why, bh, by):
    """
    Detailed RNN step computation
    
    Args:
        input_t: Current input (batch_size, input_size)
        h_prev: Previous hidden state (batch_size, hidden_size)
        Wxh: Input-to-hidden weights (input_size, hidden_size)
        Whh: Hidden-to-hidden weights (hidden_size, hidden_size)
        Why: Hidden-to-output weights (hidden_size, output_size)
        bh: Hidden bias (1, hidden_size)
        by: Output bias (1, output_size)
    """
    
    # Step 1: Linear transformation of input
    input_contribution = input_t @ Wxh  # (batch, input) @ (input, hidden)
    print(f"Input contribution shape: {input_contribution.shape}")
    
    # Step 2: Linear transformation of previous hidden state
    hidden_contribution = h_prev @ Whh  # (batch, hidden) @ (hidden, hidden)
    print(f"Hidden contribution shape: {hidden_contribution.shape}")
    
    # Step 3: Combine inputs and add bias
    linear_output = input_contribution + hidden_contribution + bh
    print(f"Linear output shape: {linear_output.shape}")
    
    # Step 4: Apply activation function
    h_new = np.tanh(linear_output)
    print(f"New hidden state shape: {h_new.shape}")
    
    # Step 5: Compute output
    output = h_new @ Why + by  # (batch, hidden) @ (hidden, output)
    print(f"Output shape: {output.shape}")
    
    return output, h_new

# Example usage
batch_size, input_size, hidden_size, output_size = 2, 3, 4, 2

input_t = np.random.randn(batch_size, input_size)
h_prev = np.random.randn(batch_size, hidden_size)
Wxh = np.random.randn(input_size, hidden_size) * 0.1
Whh = np.random.randn(hidden_size, hidden_size) * 0.1
Why = np.random.randn(hidden_size, output_size) * 0.1
bh = np.zeros((1, hidden_size))
by = np.zeros((1, output_size))

output, h_new = rnn_step_detailed(input_t, h_prev, Wxh, Whh, Why, bh, by)
```

### 3.4 Activation Functions

**Understanding `np.tanh()` and other activations**:

```python
# Hyperbolic tangent (tanh)
x = np.linspace(-3, 3, 100)
tanh_x = np.tanh(x)

# Properties of tanh:
# - Range: [-1, 1]
# - Zero-centered (unlike sigmoid)
# - Derivative: 1 - tanh²(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Other common activations
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage
logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
probs = softmax(logits)
print(f"Softmax probabilities:\n{probs}")
print(f"Row sums: {np.sum(probs, axis=1)}")  # Should be [1, 1]
```

### 3.5 Broadcasting in Neural Networks

**Understanding how bias addition works**:

```python
# Broadcasting allows operations between arrays of different shapes
hidden_states = np.random.randn(32, 128)  # (batch_size, hidden_size)
bias = np.random.randn(1, 128)            # (1, hidden_size)

# Broadcasting: (32, 128) + (1, 128) = (32, 128)
result = hidden_states + bias

# This is equivalent to:
bias_expanded = np.repeat(bias, 32, axis=0)  # (32, 128)
result_manual = hidden_states + bias_expanded

print(f"Results are equal: {np.allclose(result, result_manual)}")

# Common broadcasting patterns in neural networks:
batch_size, seq_len, hidden_size = 16, 10, 64

# Pattern 1: Adding bias to all time steps
hidden_sequence = np.random.randn(batch_size, seq_len, hidden_size)
bias = np.random.randn(1, 1, hidden_size)
result = hidden_sequence + bias  # Broadcasts to (16, 10, 64)

# Pattern 2: Element-wise operations
weights = np.random.randn(hidden_size, hidden_size)
input_batch = np.random.randn(batch_size, hidden_size)
output = input_batch @ weights  # (16, 64) @ (64, 64) = (16, 64)
```

---

## 4. NumPy Advanced Functions

### 4.1 Random Number Generation

**`np.random.randn()` - Standard Normal Distribution**:

```python
# Generate random numbers from standard normal distribution (mean=0, std=1)
samples = np.random.randn(3, 4)  # 3x4 matrix
print(f"Mean: {np.mean(samples):.3f}")  # Should be close to 0
print(f"Std: {np.std(samples):.3f}")    # Should be close to 1

# Common usage in neural networks - Xavier/Glorot initialization
def xavier_init(input_size, output_size):
    """Xavier initialization for better gradient flow"""
    limit = np.sqrt(6.0 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

# He initialization (for ReLU networks)
def he_init(input_size, output_size):
    """He initialization for ReLU networks"""
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

# Example
weights = np.random.randn(784, 128) * 0.01  # Small random weights
print(f"Weight matrix shape: {weights.shape}")
print(f"Weight statistics: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
```

**`np.random.uniform()` - Uniform Distribution**:

```python
# Generate random numbers from uniform distribution
uniform_samples = np.random.uniform(low=0, high=1, size=(3, 3))
print(f"Uniform samples:\n{uniform_samples}")

# Common usage: random phase for sine waves
def generate_sine_wave(length, frequency=1.0, phase=None):
    if phase is None:
        phase = np.random.uniform(0, 2*np.pi)  # Random starting phase
    
    t = np.linspace(0, 4*np.pi, length)
    return np.sin(frequency * t + phase)

# Generate multiple sine waves with random phases
waves = []
for i in range(5):
    wave = generate_sine_wave(100)
    waves.append(wave)

waves = np.array(waves)
print(f"Generated {len(waves)} sine waves, shape: {waves.shape}")
```

**Other random functions**:

```python
# Set seed for reproducibility
np.random.seed(42)

# Random integers
random_ints = np.random.randint(0, 10, size=(3, 3))

# Random choice from array
colors = ['red', 'green', 'blue', 'yellow']
random_colors = np.random.choice(colors, size=5, replace=True)

# Random permutation
indices = np.random.permutation(10)  # Shuffle indices 0-9
print(f"Shuffled indices: {indices}")

# Random sampling without replacement
data = np.arange(100)
sampled = np.random.choice(data, size=10, replace=False)
print(f"Random sample: {sampled}")
```

### 4.2 `np.linspace()` - Linear Spacing

**Generate evenly spaced numbers**:

```python
# Basic usage
linear_points = np.linspace(0, 10, 11)  # 11 points from 0 to 10
print(f"Linear points: {linear_points}")
# [0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]

# Common usage: time series generation
def create_time_series(start_time, end_time, num_points):
    time_points = np.linspace(start_time, end_time, num_points)
    return time_points

# Generate sine wave data
time = np.linspace(0, 4*np.pi, 1000)  # 1000 points from 0 to 4π
sine_wave = np.sin(time)
cosine_wave = np.cos(time)

print(f"Time range: {time[0]:.2f} to {time[-1]:.2f}")
print(f"Number of points: {len(time)}")

# Include/exclude endpoint
with_endpoint = np.linspace(0, 1, 5, endpoint=True)   # [0.   0.25 0.5  0.75 1.  ]
without_endpoint = np.linspace(0, 1, 5, endpoint=False) # [0.  0.2 0.4 0.6 0.8]

print(f"With endpoint: {with_endpoint}")
print(f"Without endpoint: {without_endpoint}")
```

**Practical applications**:

```python
# Create training data for sequence models
def generate_sequence_data(num_sequences, seq_length):
    """Generate synthetic sequence data"""
    sequences = []
    
    for i in range(num_sequences):
        # Random starting phase
        phase = np.random.uniform(0, 2*np.pi)
        
        # Create time points
        t = np.linspace(0, 4*np.pi, seq_length + 1)
        
        # Generate sequence (sine wave with noise)
        sequence = np.sin(t + phase) + 0.1 * np.random.randn(seq_length + 1)
        
        # Input: all but last point, Target: all but first point
        X = sequence[:-1]
        y = sequence[1:]
        
        sequences.append((X, y))
    
    return sequences

# Generate training data
training_data = generate_sequence_data(num_sequences=100, seq_length=50)
print(f"Generated {len(training_data)} sequences")
print(f"Each sequence: input shape {training_data[0][0].shape}, target shape {training_data[0][1].shape}")

# Frequency analysis
def analyze_frequencies(signal, sample_rate):
    """Analyze frequency components of a signal"""
    fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Only positive frequencies
    positive_freq_idx = frequencies > 0
    return frequencies[positive_freq_idx], np.abs(fft[positive_freq_idx])

# Example
sample_rate = 100  # Hz
t = np.linspace(0, 1, sample_rate, endpoint=False)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*10*t)  # 5Hz + 10Hz

freqs, amplitudes = analyze_frequencies(signal, sample_rate)
print(f"Dominant frequencies: {freqs[np.argsort(amplitudes)[-2:]]}")
```

### 4.3 Array Shape Manipulation

**Understanding array dimensions in neural networks**:

```python
# Common array shapes in deep learning
batch_size = 32
seq_length = 10
input_size = 128
hidden_size = 256

# Sequence data: (batch_size, seq_length, input_size)
sequence_data = np.random.randn(batch_size, seq_length, input_size)
print(f"Sequence data shape: {sequence_data.shape}")

# For RNN processing, often need: (seq_length, batch_size, input_size)
rnn_input = sequence_data.transpose(1, 0, 2)
print(f"RNN input shape: {rnn_input.shape}")

# Reshape operations
flattened = sequence_data.reshape(batch_size, -1)  # Flatten sequence
print(f"Flattened shape: {flattened.shape}")  # (32, 1280)

# Restore original shape
restored = flattened.reshape(batch_size, seq_length, input_size)
print(f"Restored shape: {restored.shape}")
print(f"Shapes match: {np.array_equal(sequence_data, restored)}")
```

**Advanced indexing and slicing**:

```python
# Boolean indexing
data = np.random.randn(100)
positive_values = data[data > 0]  # Only positive values
print(f"Original size: {len(data)}, Positive values: {len(positive_values)}")

# Fancy indexing
indices = [0, 2, 4, 6, 8]
selected = data[indices]
print(f"Selected values: {selected}")

# Multi-dimensional indexing
matrix = np.random.randn(5, 5)
diagonal = matrix[np.arange(5), np.arange(5)]  # Diagonal elements
print(f"Diagonal: {diagonal}")

# Conditional operations
clipped = np.where(data > 1, 1, np.where(data < -1, -1, data))  # Clip to [-1, 1]
print(f"Clipped range: [{np.min(clipped):.2f}, {np.max(clipped):.2f}]")
```

### 4.4 Mathematical Operations

**Element-wise vs. matrix operations**:

```python
# Element-wise operations (broadcasting)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

element_wise_mult = A * B  # Element-wise multiplication
matrix_mult = A @ B        # Matrix multiplication

print(f"Element-wise multiplication:\n{element_wise_mult}")
print(f"Matrix multiplication:\n{matrix_mult}")

# Statistical operations
data = np.random.randn(1000, 50)  # 1000 samples, 50 features

# Along different axes
mean_per_feature = np.mean(data, axis=0)  # Shape: (50,)
mean_per_sample = np.mean(data, axis=1)   # Shape: (1000,)
overall_mean = np.mean(data)              # Scalar

print(f"Mean per feature shape: {mean_per_feature.shape}")
print(f"Mean per sample shape: {mean_per_sample.shape}")
print(f"Overall mean: {overall_mean:.4f}")

# Normalization (common in ML)
def normalize_data(data, axis=0):
    """Normalize data to zero mean and unit variance"""
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

normalized = normalize_data(data, axis=0)
print(f"Normalized data - Mean: {np.mean(normalized, axis=0)[:5]}")
print(f"Normalized data - Std: {np.std(normalized, axis=0)[:5]}")
```

---

## 5. Advanced Python Syntax

### 3.1 N-grams with `tuple(tokens[i:i + n])`

**Basic n-gram extraction**:
```python
def get_ngrams(tokens, n):
    """Extract n-grams from token list"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams

# Example usage
tokens = ['the', 'quick', 'brown', 'fox', 'jumps']

unigrams = get_ngrams(tokens, 1)
print("Unigrams:", unigrams)
# [('the',), ('quick',), ('brown',), ('fox',), ('jumps',)]

bigrams = get_ngrams(tokens, 2)
print("Bigrams:", bigrams)
# [('the', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumps')]

trigrams = get_ngrams(tokens, 3)
print("Trigrams:", trigrams)
# [('the', 'quick', 'brown'), ('quick', 'brown', 'fox'), ('brown', 'fox', 'jumps')]
```

**Advanced n-gram processing**:
```python
from collections import Counter

class NGramProcessor:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = Counter()
    
    def fit(self, texts):
        """Train on multiple texts"""
        for text in texts:
            tokens = text.lower().split()
            ngrams = self.get_ngrams(tokens)
            self.ngram_counts.update(ngrams)
    
    def get_ngrams(self, tokens):
        return [tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]
    
    def most_common(self, k=10):
        return self.ngram_counts.most_common(k)

# Usage
processor = NGramProcessor(n=2)
texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the cat and the dog played"
]
processor.fit(texts)
print(processor.most_common(5))
```

---

## 6. Type Hints & Annotations

### 4.1 Basic Type Hints

```python
from typing import List, Dict, Tuple, Optional, Union, Any

# Basic types
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add_numbers(a: int, b: int) -> int:
    return a + b

def calculate_average(numbers: List[float]) -> float:
    return sum(numbers) / len(numbers)

# Optional types
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # Returns str or None

# Union types
def process_id(user_id: Union[int, str]) -> str:
    return str(user_id)
```

### 4.2 Complex Type Hints

```python
from typing import Dict, List, Tuple, Callable

# Complex data structures
def process_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Convert word->id mapping to id->word mapping"""
    return {v: k for k, v in vocab.items()}

def get_word_pairs(word: List[str]) -> Dict[Tuple[str, str], int]:
    """Get adjacent word pairs and their counts"""
    pairs = {}
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

# Function type hints
def apply_function(data: List[int], func: Callable[[int], int]) -> List[int]:
    """Apply function to each element in list"""
    return [func(x) for x in data]

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()
```

### 4.3 Class Type Hints

```python
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Token:
    text: str
    pos: int
    label: Optional[str] = None

class Tokenizer:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
    
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize text into Token objects"""
        tokens = []
        words = text.split()
        for i, word in enumerate(words):
            token = Token(text=word, pos=i)
            tokens.append(token)
        return tokens
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.vocab.get(token, 0) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert IDs back to tokens"""
        return [self.reverse_vocab.get(id, '<UNK>') for id in ids]
```

---

## 7. Lambda Functions

### 5.1 Basic Lambda Functions

```python
# Basic syntax: lambda arguments: expression

# Simple operations
square = lambda x: x ** 2
print(square(5))  # 25

add = lambda x, y: x + y
print(add(3, 4))  # 7

# Multiple arguments
multiply_three = lambda x, y, z: x * y * z
print(multiply_three(2, 3, 4))  # 24
```

### 5.2 Lambda with Built-in Functions

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# Map to squares
squares = list(map(lambda x: x ** 2, numbers))
print(squares)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Sort by custom criteria
words = ['python', 'java', 'c', 'javascript', 'go']
sorted_by_length = sorted(words, key=lambda x: len(x))
print(sorted_by_length)  # ['c', 'go', 'java', 'python', 'javascript']

# Reduce (from functools)
from functools import reduce
product = reduce(lambda x, y: x * y, [1, 2, 3, 4, 5])
print(product)  # 120
```

### 5.3 Lambda in Data Processing

```python
# Sorting complex data structures
students = [
    {'name': 'Alice', 'grade': 85, 'age': 20},
    {'name': 'Bob', 'grade': 90, 'age': 19},
    {'name': 'Charlie', 'grade': 78, 'age': 21}
]

# Sort by grade (descending)
by_grade = sorted(students, key=lambda x: x['grade'], reverse=True)
print([s['name'] for s in by_grade])  # ['Bob', 'Alice', 'Charlie']

# Sort by multiple criteria
by_age_then_grade = sorted(students, key=lambda x: (x['age'], -x['grade']))

# Lambda with max/min
best_student = max(students, key=lambda x: x['grade'])
print(best_student['name'])  # Bob

# Lambda in list comprehensions with conditions
high_performers = [s for s in students if (lambda x: x['grade'] > 80)(s)]
```

### 5.4 Lambda in NLP/ML Context

```python
from collections import Counter

# Word frequency analysis
def analyze_text(text):
    words = text.lower().split()
    
    # Filter out short words
    long_words = list(filter(lambda w: len(w) > 3, words))
    
    # Get word frequencies
    word_freq = Counter(long_words)
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words

# N-gram probability calculation
def calculate_ngram_probs(ngrams, counts):
    total = sum(counts.values())
    return {ngram: count/total for ngram, count in counts.items()}

# Sort n-grams by probability
ngram_counts = {('the', 'cat'): 5, ('cat', 'sat'): 3, ('sat', 'on'): 2}
sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
```

---

## 8. Dictionary & List Comprehensions

### 6.1 Dictionary Comprehensions

**Basic syntax**: `{key_expr: value_expr for item in iterable}`

```python
# Basic dictionary comprehension
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# With condition
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)  # {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}

# From existing dictionary
original = {'a': 1, 'b': 2, 'c': 3}
doubled = {k: v*2 for k, v in original.items()}
print(doubled)  # {'a': 2, 'b': 4, 'c': 6}
```

### 6.2 Reverse Dictionary Mapping

```python
# Common pattern: reverse vocab mapping
vocab = {'cat': 0, 'dog': 1, 'bird': 2, 'fish': 3}
reverse_vocab = {v: k for k, v in vocab.items()}
print(reverse_vocab)  # {0: 'cat', 1: 'dog', 2: 'bird', 3: 'fish'}

# Filter during reversal
filtered_reverse = {v: k for k, v in vocab.items() if v > 1}
print(filtered_reverse)  # {2: 'bird', 3: 'fish'}

# Transform keys and values
upper_vocab = {k.upper(): v+100 for k, v in vocab.items()}
print(upper_vocab)  # {'CAT': 100, 'DOG': 101, 'BIRD': 102, 'FISH': 103}
```

### 6.3 Advanced Dictionary Comprehensions

```python
# Nested dictionary comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = {i: {j: matrix[j][i] for j in range(len(matrix))} 
              for i in range(len(matrix[0]))}

# Group by condition
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
grouped = {len(word): [w for w in words if len(w) == len(word)] 
           for word in words}

# Word frequency from text
text = "the quick brown fox jumps over the lazy dog"
word_freq = {word: text.split().count(word) for word in set(text.split())}
print(word_freq)
```

### 6.4 List Comprehensions

```python
# Basic list comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# Nested list comprehensions
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
print(matrix)  # [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# Flatten nested lists
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Conditional expressions in comprehensions
numbers = range(-5, 6)
abs_values = [x if x >= 0 else -x for x in numbers]
```

---

## 9. Classes & Object-Oriented Programming

### 7.1 Basic Class Structure

```python
class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    def __init__(self, name: str, age: int):
        """Constructor method - called when creating new instance"""
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
        self.friends = []
    
    def introduce(self) -> str:
        """Instance method - first parameter is always self"""
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    def add_friend(self, friend: 'Person') -> None:
        """Method that modifies instance state"""
        self.friends.append(friend)
        friend.friends.append(self)  # Mutual friendship
    
    def __str__(self) -> str:
        """String representation of object"""
        return f"Person(name='{self.name}', age={self.age})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return f"Person('{self.name}', {self.age})"

# Usage
alice = Person("Alice", 25)
bob = Person("Bob", 30)

print(alice.introduce())  # Hi, I'm Alice and I'm 25 years old
alice.add_friend(bob)
print(f"Alice's friends: {[str(f) for f in alice.friends]}")
```

### 7.2 Understanding `self`

```python
class Counter:
    def __init__(self, initial_value: int = 0):
        self.value = initial_value  # self refers to the specific instance
    
    def increment(self):
        self.value += 1  # Modify this instance's value
        return self  # Return self for method chaining
    
    def decrement(self):
        self.value -= 1
        return self
    
    def get_value(self):
        return self.value

# Each instance has its own state
counter1 = Counter(10)
counter2 = Counter(20)

counter1.increment().increment()  # Method chaining
counter2.decrement()

print(counter1.get_value())  # 12
print(counter2.get_value())  # 19

# self is automatically passed
# counter1.increment() is equivalent to Counter.increment(counter1)
```

### 7.3 Advanced Class Features

```python
from typing import List, Optional

class BankAccount:
    # Class variables
    bank_name = "Python Bank"
    interest_rate = 0.02
    
    def __init__(self, account_holder: str, initial_balance: float = 0):
        # Private attributes (convention: prefix with _)
        self._account_holder = account_holder
        self._balance = initial_balance
        self._transaction_history: List[str] = []
    
    @property
    def balance(self) -> float:
        """Getter for balance (read-only access)"""
        return self._balance
    
    @property
    def account_holder(self) -> str:
        """Getter for account holder"""
        return self._account_holder
    
    def deposit(self, amount: float) -> None:
        """Deposit money to account"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self._balance += amount
        self._transaction_history.append(f"Deposited ${amount:.2f}")
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw money from account"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if amount > self._balance:
            return False  # Insufficient funds
        
        self._balance -= amount
        self._transaction_history.append(f"Withdrew ${amount:.2f}")
        return True
    
    def apply_interest(self) -> None:
        """Apply interest to account"""
        interest = self._balance * self.interest_rate
        self._balance += interest
        self._transaction_history.append(f"Interest applied: ${interest:.2f}")
    
    @classmethod
    def create_savings_account(cls, holder: str, initial_deposit: float):
        """Class method - alternative constructor"""
        account = cls(holder, initial_deposit)
        account.interest_rate = 0.05  # Higher interest for savings
        return account
    
    @staticmethod
    def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
        """Static method - utility function"""
        return principal * (1 + rate) ** years
    
    def __len__(self) -> int:
        """Return number of transactions"""
        return len(self._transaction_history)
    
    def __str__(self) -> str:
        return f"Account({self._account_holder}: ${self._balance:.2f})"

# Usage examples
account = BankAccount("Alice", 1000)
account.deposit(500)
account.withdraw(200)
account.apply_interest()

print(account)  # Account(Alice: $1326.00)
print(f"Transactions: {len(account)}")  # Transactions: 3

# Class method usage
savings = BankAccount.create_savings_account("Bob", 5000)

# Static method usage
future_value = BankAccount.calculate_compound_interest(1000, 0.05, 10)
print(f"Future value: ${future_value:.2f}")
```

### 7.4 Inheritance

```python
class Animal:
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species
    
    def make_sound(self) -> str:
        return "Some generic animal sound"
    
    def info(self) -> str:
        return f"{self.name} is a {self.species}"

class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name, "Canine")  # Call parent constructor
        self.breed = breed
    
    def make_sound(self) -> str:  # Override parent method
        return "Woof!"
    
    def fetch(self) -> str:
        return f"{self.name} is fetching the ball!"

class Cat(Animal):
    def __init__(self, name: str, indoor: bool = True):
        super().__init__(name, "Feline")
        self.indoor = indoor
    
    def make_sound(self) -> str:
        return "Meow!"
    
    def purr(self) -> str:
        return f"{self.name} is purring contentedly"

# Usage
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", indoor=True)

print(dog.info())        # Buddy is a Canine
print(dog.make_sound())  # Woof!
print(dog.fetch())       # Buddy is fetching the ball!

print(cat.info())        # Whiskers is a Feline
print(cat.make_sound())  # Meow!
print(cat.purr())        # Whiskers is purring contentedly
```

---

## 10. Collections Module

### 8.1 `defaultdict`

**Purpose**: Dictionary that provides default values for missing keys

```python
from collections import defaultdict

# Basic usage
dd = defaultdict(int)  # Default value is 0
dd['a'] += 1  # No KeyError, creates 'a' with value 0, then increments
dd['b'] += 5
print(dict(dd))  # {'a': 1, 'b': 5}

# Different default types
dd_list = defaultdict(list)
dd_list['fruits'].append('apple')
dd_list['fruits'].append('banana')
dd_list['vegetables'].append('carrot')
print(dict(dd_list))  # {'fruits': ['apple', 'banana'], 'vegetables': ['carrot']}

dd_set = defaultdict(set)
dd_set['colors'].add('red')
dd_set['colors'].add('blue')
dd_set['colors'].add('red')  # Duplicate ignored
print(dict(dd_set))  # {'colors': {'red', 'blue'}}
```

**Practical examples**:
```python
# Word frequency counting
def count_words(text):
    word_count = defaultdict(int)
    for word in text.split():
        word_count[word] += 1
    return dict(word_count)

text = "the quick brown fox jumps over the lazy dog the fox"
frequencies = count_words(text)
print(frequencies)  # {'the': 3, 'quick': 1, 'brown': 1, 'fox': 2, ...}

# Group items by property
def group_by_length(words):
    groups = defaultdict(list)
    for word in words:
        groups[len(word)].append(word)
    return dict(groups)

words = ['cat', 'dog', 'elephant', 'bird', 'fish']
grouped = group_by_length(words)
print(grouped)  # {3: ['cat', 'dog'], 8: ['elephant'], 4: ['bird', 'fish']}

# N-gram counting
def count_ngrams(tokens, n=2):
    ngram_counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngram_counts[ngram] += 1
    return dict(ngram_counts)

tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
bigrams = count_ngrams(tokens, 2)
print(bigrams)  # {('the', 'cat'): 1, ('cat', 'sat'): 1, ('sat', 'on'): 1, ...}
```

### 8.2 `Counter`

**Purpose**: Dictionary subclass for counting hashable objects

```python
from collections import Counter

# Basic usage
text = "hello world"
char_count = Counter(text)
print(char_count)  # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

# Most common elements
print(char_count.most_common(3))  # [('l', 3), ('o', 2), ('h', 1)]

# Count words
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
word_count = Counter(words)
print(word_count)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})
```

**Advanced Counter operations**:
```python
# Counter arithmetic
c1 = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
c2 = Counter(['a', 'b', 'b', 'd'])

print(c1 + c2)  # Counter({'b': 5, 'a': 3, 'c': 1, 'd': 1})
print(c1 - c2)  # Counter({'c': 1, 'b': 1})
print(c1 & c2)  # Intersection: Counter({'a': 1, 'b': 2})
print(c1 | c2)  # Union: Counter({'b': 3, 'a': 2, 'c': 1, 'd': 1})

# Update counter
c1.update(['d', 'd', 'e'])
print(c1)  # Counter({'b': 3, 'a': 2, 'd': 2, 'c': 1, 'e': 1})

# Get count (returns 0 for missing keys)
print(c1['missing'])  # 0

# Total count
print(sum(c1.values()))  # Total number of elements
```

**NLP applications**:
```python
def analyze_text_statistics(text):
    """Comprehensive text analysis using Counter"""
    words = text.lower().split()
    
    # Word frequencies
    word_freq = Counter(words)
    
    # Character frequencies (excluding spaces)
    char_freq = Counter(char for char in text.lower() if char != ' ')
    
    # Word length distribution
    length_dist = Counter(len(word) for word in words)
    
    return {
        'total_words': len(words),
        'unique_words': len(word_freq),
        'most_common_words': word_freq.most_common(5),
        'most_common_chars': char_freq.most_common(5),
        'word_length_dist': dict(length_dist)
    }

text = "The quick brown fox jumps over the lazy dog. The fox is quick."
stats = analyze_text_statistics(text)
for key, value in stats.items():
    print(f"{key}: {value}")
```

### 8.3 Other Collections

```python
from collections import deque, namedtuple, OrderedDict

# deque - double-ended queue
dq = deque([1, 2, 3])
dq.appendleft(0)  # Add to left
dq.append(4)      # Add to right
print(dq)         # deque([0, 1, 2, 3, 4])

# namedtuple - tuple with named fields
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)   # 1 2
print(p[0], p[1]) # 1 2 (still works like tuple)

# OrderedDict - maintains insertion order (less needed in Python 3.7+)
od = OrderedDict()
od['first'] = 1
od['second'] = 2
od['third'] = 3
print(list(od.keys()))  # ['first', 'second', 'third']
```

---

## 11. Tuples & Advanced Data Structures

### 9.1 Tuple Basics

```python
# Creating tuples
empty_tuple = ()
single_item = (1,)  # Note the comma!
coordinates = (3, 4)
rgb_color = (255, 128, 0)

# Tuple unpacking
x, y = coordinates
r, g, b = rgb_color
print(f"Point: ({x}, {y}), Color: RGB({r}, {g}, {b})")

# Multiple assignment
a, b, c = 1, 2, 3  # Actually creates tuple (1, 2, 3) then unpacks

# Swapping variables
a, b = b, a
```

### 9.2 Tuples as Dictionary Keys

```python
# Tuples are immutable, so they can be dictionary keys
coordinate_data = {
    (0, 0): "origin",
    (1, 0): "right",
    (0, 1): "up",
    (1, 1): "diagonal"
}

print(coordinate_data[(1, 1)])  # diagonal

# N-gram storage
bigram_counts = {
    ('the', 'cat'): 5,
    ('cat', 'sat'): 3,
    ('sat', 'on'): 2
}

# Word pair relationships
word_pairs = {
    ('king', 'queen'): 'royal_pair',
    ('man', 'woman'): 'gender_pair',
    ('hot', 'cold'): 'opposite_pair'
}
```

### 9.3 Advanced Tuple Operations

```python
# Tuple methods
numbers = (1, 2, 3, 2, 4, 2, 5)
print(numbers.count(2))  # 3
print(numbers.index(3))  # 2

# Tuple concatenation and repetition
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
combined = tuple1 + tuple2  # (1, 2, 3, 4, 5, 6)
repeated = tuple1 * 3       # (1, 2, 3, 1, 2, 3, 1, 2, 3)

# Tuple slicing
data = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
print(data[2:5])    # (2, 3, 4)
print(data[::2])    # (0, 2, 4, 6, 8)
print(data[::-1])   # (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
```

### 9.4 Named Tuples for Structured Data

```python
from collections import namedtuple
from typing import NamedTuple

# Using collections.namedtuple
Person = namedtuple('Person', ['name', 'age', 'city'])
alice = Person('Alice', 30, 'New York')
print(alice.name)  # Alice
print(alice._asdict())  # {'name': 'Alice', 'age': 30, 'city': 'New York'}

# Using typing.NamedTuple (preferred for type hints)
class Point(NamedTuple):
    x: float
    y: float
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

point = Point(3.0, 4.0)
print(point.distance_from_origin())  # 5.0

# Token representation
class Token(NamedTuple):
    text: str
    start: int
    end: int
    label: str = 'O'  # Default value

token = Token('hello', 0, 5)
print(token)  # Token(text='hello', start=0, end=5, label='O')
```

---

## 12. Enumerate & Iteration

### 10.1 Basic `enumerate`

```python
# Basic usage
fruits = ['apple', 'banana', 'cherry', 'date']

for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# Output:
# 0: apple
# 1: banana
# 2: cherry
# 3: date

# Start from different number
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}: {fruit}")
# Output:
# 1: apple
# 2: banana
# 3: cherry
# 4: date
```

### 10.2 Advanced Enumerate Usage

```python
# Create index mapping
words = ['cat', 'dog', 'bird', 'fish']
word_to_index = {word: i for i, word in enumerate(words)}
print(word_to_index)  # {'cat': 0, 'dog': 1, 'bird': 2, 'fish': 3}

# Find positions of specific items
text = "the quick brown fox jumps over the lazy dog"
words = text.split()
the_positions = [i for i, word in enumerate(words) if word == 'the']
print(the_positions)  # [0, 6]

# Process with both index and value
def process_tokens_with_position(tokens):
    processed = []
    for i, token in enumerate(tokens):
        # Add position information to each token
        processed_token = {
            'text': token,
            'position': i,
            'is_first': i == 0,
            'is_last': i == len(tokens) - 1
        }
        processed.append(processed_token)
    return processed

tokens = ['hello', 'world', 'how', 'are', 'you']
result = process_tokens_with_position(tokens)
for item in result:
    print(item)
```

### 10.3 Enumerate with Multiple Iterables

```python
# Using zip with enumerate
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
cities = ['NYC', 'LA', 'Chicago']

for i, (name, age, city) in enumerate(zip(names, ages, cities)):
    print(f"{i+1}. {name} ({age}) lives in {city}")
# Output:
# 1. Alice (25) lives in NYC
# 2. Bob (30) lives in LA
# 3. Charlie (35) lives in Chicago

# Enumerate with dictionary items
person_data = {'name': 'Alice', 'age': 30, 'city': 'NYC', 'job': 'Engineer'}
for i, (key, value) in enumerate(person_data.items()):
    print(f"{i}: {key} = {value}")
```

### 10.4 Practical Applications

```python
# Text processing with line numbers
def process_file_with_line_numbers(text):
    lines = text.split('\n')
    processed_lines = []
    
    for line_num, line in enumerate(lines, start=1):
        if line.strip():  # Skip empty lines
            processed_line = f"Line {line_num}: {line.strip()}"
            processed_lines.append(processed_line)
    
    return processed_lines

# Create training data with indices
def create_training_data(sentences, labels):
    training_data = []
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        training_example = {
            'id': i,
            'text': sentence,
            'label': label,
            'length': len(sentence.split())
        }
        training_data.append(training_example)
    return training_data

sentences = ["I love Python", "Python is great", "Programming is fun"]
labels = ["positive", "positive", "positive"]
training_data = create_training_data(sentences, labels)
for example in training_data:
    print(example)

# Batch processing with enumerate
def process_in_batches(data, batch_size=3):
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    
    for batch_num, batch in enumerate(batches):
        print(f"Processing batch {batch_num + 1}: {batch}")
        # Process batch here
    
    return batches

data = list(range(10))
process_in_batches(data, batch_size=3)
```

---

## 13. Practical Examples

### 11.1 Complete Text Processing Pipeline

```python
import string
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np

class TextProcessor:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.word_counts = Counter()
        self.ngram_counts = defaultdict(int)
    
    def clean_text(self, text: str) -> str:
        """Clean text using translate method"""
        # Remove punctuation and digits
        translator = str.maketrans('', '', string.punctuation + string.digits)
        cleaned = text.translate(translator).lower().strip()
        return re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize cleaned text"""
        cleaned = self.clean_text(text)
        return cleaned.split()
    
    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """Build vocabulary from texts"""
        # Count all words
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
        
        # Build vocab with frequent words
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in self.word_counts.most_common():
            if count >= min_freq:
                self.vocab[word] = len(self.vocab)
        
        # Create reverse mapping
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Extract n-grams from tokens"""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    
    def count_ngrams(self, texts: List[str], n: int = 2) -> Dict[Tuple[str, ...], int]:
        """Count n-grams in texts"""
        ngram_counts = defaultdict(int)
        for text in texts:
            tokens = self.tokenize(text)
            ngrams = self.get_ngrams(tokens, n)
            for ngram in ngrams:
                ngram_counts[ngram] += 1
        return dict(ngram_counts)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to numerical representation"""
        tokens = self.tokenize(text)
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return np.array(encoded)
    
    def decode_sequence(self, sequence: np.ndarray) -> str:
        """Decode numerical sequence back to text"""
        tokens = [self.reverse_vocab.get(idx, '<UNK>') for idx in sequence]
        return ' '.join(tokens)
    
    def get_statistics(self, texts: List[str]) -> Dict:
        """Get comprehensive text statistics"""
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
        
        char_counts = Counter(''.join(all_tokens))
        length_dist = Counter(len(token) for token in all_tokens)
        
        return {
            'total_texts': len(texts),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'avg_text_length': np.mean([len(self.tokenize(text)) for text in texts]),
            'vocab_size': len(self.vocab),
            'most_common_words': self.word_counts.most_common(10),
            'most_common_chars': char_counts.most_common(10),
            'token_length_dist': dict(length_dist)
        }

# Usage example
processor = TextProcessor()

# Sample texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "The cat sat on the mat and looked at the rat.",
    "Python is a powerful programming language for data science."
]

# Build vocabulary
processor.build_vocab(texts, min_freq=1)

# Get statistics
stats = processor.get_statistics(texts)
for key, value in stats.items():
    print(f"{key}: {value}")

# Encode and decode example
sample_text = "The cat sat on the mat"
encoded = processor.encode_text(sample_text)
decoded = processor.decode_sequence(encoded)
print(f"\nOriginal: {sample_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

# N-gram analysis
bigrams = processor.count_ngrams(texts, n=2)
trigrams = processor.count_ngrams(texts, n=3)

print(f"\nMost common bigrams:")
sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
for bigram, count in sorted_bigrams[:5]:
    print(f"  {bigram}: {count}")
```

### 11.2 Simple Language Model Implementation

```python
class SimpleLanguageModel:
    def __init__(self, n: int = 2):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def train(self, texts: List[str]) -> None:
        """Train the language model"""
        for text in texts:
            # Simple tokenization
            tokens = ['<BOS>'] * (self.n - 1) + text.lower().split() + ['<EOS>']
            self.vocab.update(tokens)
            
            # Count n-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1]
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
    
    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Calculate P(word|context) with Laplace smoothing"""
        ngram = context + (word,)
        
        numerator = self.ngram_counts[ngram] + 1  # Laplace smoothing
        denominator = self.context_counts[context] + len(self.vocab)
        
        return numerator / denominator if denominator > 0 else 1.0 / len(self.vocab)
    
    def generate_text(self, max_length: int = 10) -> str:
        """Generate text using the model"""
        context = tuple(['<BOS>'] * (self.n - 1))
        generated = []
        
        for _ in range(max_length):
            # Get possible next words
            candidates = []
            probabilities = []
            
            for word in self.vocab:
                if word not in ['<BOS>', '<EOS>']:
                    prob = self.probability(word, context)
                    candidates.append(word)
                    probabilities.append(prob)
            
            if not candidates:
                break
            
            # Sample next word
            probabilities = np.array(probabilities)
            probabilities = probabilities / np.sum(probabilities)
            next_word = np.random.choice(candidates, p=probabilities)
            
            if next_word == '<EOS>':
                break
            
            generated.append(next_word)
            context = context[1:] + (next_word,)
        
        return ' '.join(generated)

# Train and use the model
model = SimpleLanguageModel(n=2)
model.train(texts)

print("Generated text samples:")
for i in range(3):
    generated = model.generate_text(max_length=8)
    print(f"  {i+1}: {generated}")
```

This comprehensive guide covers all the Python concepts you mentioned with practical examples and real-world applications. Each concept is explained with increasing complexity, from basic usage to advanced applications in text processing and machine learning contexts.