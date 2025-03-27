import numpy as np

class SimpleRNN:
    def __init__(self, vocab_size, hidden_dim=3):
        # Initialize weights randomly
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Input → Hidden weights
        self.Wxh = np.random.randn(hidden_dim, vocab_size) * 0.01
        # Hidden → Hidden weights
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        # Hidden → Output weights
        self.Why = np.random.randn(vocab_size, hidden_dim) * 0.01
        
        # Biases
        self.bh = np.zeros(hidden_dim)  # Hidden bias
        self.by = np.zeros(vocab_size)  # Output bias

    def forward(self, inputs):
        # One-hot encode the inputs
        x = np.zeros((len(inputs), self.vocab_size))
        for i, idx in enumerate(inputs):
            x[i, idx] = 1
        
        # Initialize hidden state
        h = np.zeros(self.hidden_dim)
        
        # Recurrent forward pass
        for xt in x:
            # Compute new hidden state
            h = np.tanh(np.dot(self.Wxh, xt) + np.dot(self.Whh, h) + self.bh)
        
        # Compute output logits
        logits = np.dot(self.Why, h) + self.by
        
        # Apply softmax
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

# Example usage
vocab = ["the", "cat", "sat", "on", "mat", "."]
rnn = SimpleRNN(len(vocab))
input_indices = [vocab.index(word) for word in ["the", "cat", "sat", "on", "the"]]
probs = rnn.forward(input_indices)
print("Probabilities:", dict(zip(vocab, probs)))