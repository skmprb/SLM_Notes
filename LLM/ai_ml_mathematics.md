# Essential Mathematics for AI/ML

## Linear Algebra

### Vectors
- **Definition**: Arrays of numbers representing magnitude and direction
- **Operations**: Addition, scalar multiplication, dot product
- **Applications**: Word embeddings, feature representations

### Matrices
- **Definition**: 2D arrays of numbers
- **Operations**: Multiplication, transpose, inverse
- **Applications**: Weight matrices, transformations, attention scores

### Dot Product
```
a · b = Σ(ai × bi)
```
- **Use**: Measuring similarity, computing attention weights
- **Geometric meaning**: Projection of one vector onto another

### Matrix Multiplication
```
C = A × B where Cij = Σ(Aik × Bkj)
```
- **Applications**: Neural network forward pass, attention computation

## Probability and Statistics

### Probability Distributions
- **Softmax**: Converts logits to probabilities
```
softmax(xi) = e^xi / Σ(e^xj)
```
- **Applications**: Classification outputs, attention weights

### Expected Value
```
E[X] = Σ(xi × P(xi))
```
- **Use**: Understanding model predictions, loss functions

### Variance and Standard Deviation
```
Var(X) = E[(X - μ)²]
σ = √Var(X)
```
- **Applications**: Weight initialization, normalization

## Calculus

### Derivatives
- **Chain Rule**: ∂f/∂x = (∂f/∂u)(∂u/∂x)
- **Applications**: Backpropagation, gradient computation

### Partial Derivatives
```
∂f/∂xi while holding other variables constant
```
- **Use**: Computing gradients for optimization

### Gradients
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```
- **Applications**: Parameter updates, gradient descent

## Optimization

### Gradient Descent
```
θ = θ - α∇J(θ)
```
- **α**: Learning rate
- **J(θ)**: Loss function
- **Applications**: Training neural networks

### Loss Functions
- **Mean Squared Error**: MSE = (1/n)Σ(yi - ŷi)²
- **Cross-Entropy**: CE = -Σ(yi log(ŷi))
- **Applications**: Measuring prediction errors

## Information Theory

### Entropy
```
H(X) = -Σ(P(xi) log P(xi))
```
- **Meaning**: Measure of uncertainty/information content
- **Applications**: Loss functions, attention mechanisms

### KL Divergence
```
KL(P||Q) = Σ(P(xi) log(P(xi)/Q(xi)))
```
- **Use**: Measuring difference between distributions

## Transformer-Specific Mathematics

### Attention Score Computation
```
Attention(Q,K,V) = softmax(QK^T/√dk)V
```
- **Q**: Query matrix
- **K**: Key matrix  
- **V**: Value matrix
- **dk**: Dimension of key vectors

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

### Positional Encoding
```
PE(pos,2i) = sin(pos/10000^(2i/dmodel))
PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
```

## RNN Mathematics

### Hidden State Update
```
ht = tanh(Wxh·xt + Whh·ht-1 + bh)
```
- **xt**: Input at time t
- **ht-1**: Previous hidden state
- **W, b**: Weight matrices and biases

### LSTM Gates
```
ft = σ(Wf·[ht-1, xt] + bf)  # Forget gate
it = σ(Wi·[ht-1, xt] + bi)  # Input gate
ot = σ(Wo·[ht-1, xt] + bo)  # Output gate
```

## Activation Functions

### Common Functions
- **ReLU**: f(x) = max(0, x)
- **Sigmoid**: σ(x) = 1/(1 + e^(-x))
- **Tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

### Properties
- **Non-linearity**: Enables learning complex patterns
- **Differentiability**: Required for backpropagation

## Normalization

### Layer Normalization
```
LN(x) = γ((x - μ)/σ) + β
```
- **μ**: Mean across features
- **σ**: Standard deviation
- **γ, β**: Learnable parameters

### Batch Normalization
```
BN(x) = γ((x - μB)/σB) + β
```
- **μB, σB**: Batch statistics

## Distance Metrics

### Euclidean Distance
```
d(x,y) = √Σ(xi - yi)²
```

### Cosine Similarity
```
cos(θ) = (x·y)/(||x|| ||y||)
```
- **Range**: [-1, 1]
- **Applications**: Text similarity, recommendation systems

### Manhattan Distance
```
d(x,y) = Σ|xi - yi|
```

## Key Mathematical Concepts for Understanding

### Why These Matter:
1. **Linear Algebra**: Foundation for all neural network operations
2. **Calculus**: Essential for training (backpropagation)
3. **Probability**: Understanding model outputs and uncertainty
4. **Optimization**: How models learn from data
5. **Information Theory**: Measuring and comparing information content

### Practical Applications:
- **Attention mechanisms** use dot products and softmax
- **Training** relies on gradients and optimization
- **Embeddings** are vectors in high-dimensional space
- **Similarity** measured using cosine similarity or distance metrics