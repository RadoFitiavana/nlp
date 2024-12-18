import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
from collections import Counter

# SkipGram MLP model definition
class SkipGramMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramMLP, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, 256)
        self.output = nn.Linear(256, vocab_size)
    
    def forward(self, center):
        x = self.embeddings(center)
        x = torch.relu(self.hidden(x))
        out = self.output(x)
        return out


# Function to train the SkipGram MLP model
def trainEmbeddingMLP(model, dataloader, vocab_size, n_epochs=5, device='cpu'):
    model.to(device)  # Move the model to the selected device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(n_epochs):
        total_loss = 0
        model.train()  # Set the model to training mode
        for target, context in dataloader:
            target, context = target.to(device), context.to(device)  # Move data to device

            optimizer.zero_grad()
            outputs = model(target)

            # Calculate loss
            loss = criterion(outputs, context)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")

# Generate center-context pairs for SkipGram model
def generate_pairs(vocab, sentences, window_size=2):
    pairs = []
    for sentence in sentences:
        for center_idx in range(len(sentence)):
            window = sentence[max(center_idx - window_size, 0): min(center_idx + window_size + 1, len(sentence))]
            for context_word in window:
                if sentence[center_idx] != context_word:
                    pairs.append((vocab[sentence[center_idx]], vocab[context_word]))
    return pairs


# Tokenization of input text
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    return text.split()


# Dataset for SkipGram model
class Word2VecDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])


# LSTM classifier with transfer learning from SkipGram embeddings
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

    def preprocess_data(self, texts, vocab, max_length):
        sequences = [
            [vocab[word] for word in text if word in vocab]
            for text in texts
        ]
        sequences = [
            seq[:max_length] + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length]
            for seq in sequences
        ]
        return torch.tensor(sequences)
