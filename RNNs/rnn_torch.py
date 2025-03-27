import torch
import torch.nn as nn
import numpy as np

# Define the vocabulary
vocab = ["the", "cat", "sat", "on", "mat"]  # Example vocabulary

class PyTorchRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding layer to convert one-hot encoded input to dense vector
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
        # RNN layer
        self.rnn = nn.RNN(vocab_size, hidden_dim, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Create one-hot encoding
        x_onehot = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        
        # Pass through RNN
        out, _ = self.rnn(x_onehot)
        
        # Get the last time step's output and pass through fully connected layer
        return torch.softmax(self.fc(out[:, -1, :]), dim=1)

# Example usage
model = PyTorchRNN(len(vocab))
input_indices = torch.tensor([vocab.index(word) for word in ["the", "cat", "sat", "on", "the"]])
probs = model(input_indices).squeeze().detach().numpy()
print("Probabilities:", dict(zip(vocab, probs)))