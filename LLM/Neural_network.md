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
2. Uderfitting :model has not trained on enough data.
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
# At its core, a gradient is the partial derivative of the loss function with respect to each parameter (weight or bias)
# Mathematically: ∇w = ∂L/∂w (partial derivative of loss L with respect to weight w)
# It represents the rate of change of the loss function as we change a specific parameter
# The gradient is a vector that points in the direction of steepest increase of the loss function
# Each element in the gradient vector corresponds to how much the loss would change if we slightly increased that specific parameter- Gradients are computed during backpropagation to update weights and biases

# Linear Unit to its core- A linear unit is a simple neuron without an activation function: y = w^T x + b
- It can only model linear relationships between input and output
- Not suitable for complex tasks like image recognition or natural language processing

# Non-Linear Unit
- A non-linear unit includes an activation function: y = σ(w^T x + b)
- It can model complex relationships and patterns in data
- Essential for deep learning, as it allows the network to learn non-linear mappings from inputs to outputs





Lets start from subset of AI
- Machine Learning (ML): algorithms that learn from data to make predictions or decisions
- Deep Learning (DL): a subset of ML that uses neural networks with multiple layers to learn from data
- Neural Networks (NN): a type of DL model inspired by the structure of the human brain, consisting of layers of interconnected neurons that process and learn from data
- Transformers: a specific architecture of neural networks that uses self-attention mechanisms, particularly effective for natural language processing tasks
