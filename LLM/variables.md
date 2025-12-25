# Machine Learning Concepts

## 1) Encoding
**What it is:** Converting data into numbers that computers can understand
- Turns text/words into numerical vectors
- Like translating human language to computer language
- Example: Word "cat" becomes [0.2, 0.8, 0.1]

## 2) Decoding
**What it is:** Converting numbers back to human-readable format
- Opposite of encoding
- Takes model predictions and turns them into text/results
- Example: [0.9, 0.1] becomes "positive sentiment"

## 3) Cosine Similarity
**What it is:** Measures how similar two things are
- Compares angles between vectors (like comparing directions)
- Score from -1 (opposite) to 1 (identical)
- Used to find similar documents or words

## 4) Softmax
**What it is:** Converts raw scores into probabilities
- Makes sure all outputs add up to 100%
- Helps models make confident predictions
- Example: [2, 1, 3] becomes [24%, 9%, 67%]

## 5) LSTMs (Long Short-Term Memory)
**What it is:** A type of neural network that remembers important information
- Good at understanding sequences (like sentences or time series)
- Has "memory gates" to decide what to remember/forget
- Better than basic networks at handling long texts

## 6) State of the Art
**What it is:** The current best method or model available
- Highest performance on standard tests
- Latest breakthrough technology
- Example: ChatGPT is state-of-the-art for conversational AI