import torch
from torch.utils.data import DataLoader
from embedding import SkipGramMLP, trainEmbeddingMLP, Word2VecDataset, LSTMClassifier, tokenize, generate_pairs
import numpy as np
from datasets import load_dataset
from collections import Counter

# Load the IMDB dataset from Hugging Face Datasets
def load_imdb_data():
    # Download the IMDB dataset from Hugging Face
    dataset = load_dataset("imdb")
    texts = dataset['train']['text']  # Reviews text from the train set
    labels = dataset['train']['label']  # Labels (0 for negative, 1 for positive)
    return texts, labels

# Prepare vocabulary from the training texts
def build_vocab(texts):
    all_words = [tokenize(text) for text in texts]
    flat_words = [word for sublist in all_words for word in sublist]
    vocab = {word: idx for idx, (word, _) in enumerate(Counter(flat_words).items())}
    return vocab

# Prepare the dataset for Word2Vec
def prepare_dataset(texts, vocab, window_size=2):
    pairs = generate_pairs(vocab, [tokenize(text) for text in texts], window_size)
    dataset = Word2VecDataset(pairs)
    return dataset

# Main function
def main():
    # Step 1: Load IMDB dataset
    texts, labels = load_imdb_data()

    # Step 2: Build the vocabulary
    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    embedding_dim = 50

    # Step 3: Prepare dataset for SkipGram model
    dataset = prepare_dataset(texts, vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Step 4: Initialize and train SkipGram MLP model
    embedding_model = SkipGramMLP(vocab_size, embedding_dim)
    trainEmbeddingMLP(embedding_model, dataloader, vocab_size, n_epochs=5)

    # Step 5: Extract the learned embeddings
    pretrained_embeddings = embedding_model.embeddings.weight.data

    # Step 6: Train LSTM Classifier using pretrained embeddings
    hidden_dim = 128
    output_dim = 1  # Binary classification (positive/negative sentiment)
    lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings)

    # Step 7: Prepare data for LSTM classifier
    max_length = 200
    train_data = lstm_model.preprocess_data(texts, vocab, max_length)

    # Step 8: Train LSTM model
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    for epoch in range(5):
        lstm_model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(data)
            loss = criterion(outputs.view(-1), torch.tensor(labels).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    main()
