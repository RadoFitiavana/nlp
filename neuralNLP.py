import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from embedding import SkipGramMLP, trainEmbeddingMLP, generate_pairs, Word2VecDataset, LSTMClassifier, tokenize, preprocess_data
from torchtext.datasets import IMDB
import re

# --- Step 1: Train SkipGram Embeddings ---
def train_skipgram_embeddings():
    print("Training SkipGram Embeddings...")

    # Load IMDB dataset and tokenize
    train_iter = IMDB(split='train')
    tokenized_texts = [tokenize(text) for label, text in train_iter]

    # Build vocabulary
    all_words = [word for tokens in tokenized_texts for word in tokens]
    vocab = {word: idx for idx, (word, _) in enumerate(Counter(all_words).items(), 1)}
    vocab_size = len(vocab) + 1  # Include padding token

    # Generate center-context pairs
    pairs = generate_pairs(vocab, tokenized_texts, window_size=2)

    # Prepare dataset and dataloader
    dataset = Word2VecDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize and train SkipGramMLP
    embedding_dim = 100
    skipgram_model = SkipGramMLP(vocab_size, embedding_dim)
    trainEmbeddingMLP(n_epochs=5, model=skipgram_model, dataloader=dataloader)

    # Extract learned embeddings
    learned_embeddings = skipgram_model.embeddings.weight.detach()
    print("SkipGram Embeddings Trained Successfully!\n")
    return learned_embeddings, vocab

# --- Step 2: Train LSTM Classifier ---
def train_lstm_classifier(embeddings, vocab):
    print("Training LSTM Classifier with Pretrained Embeddings...")

    # Load IMDB dataset and tokenize
    train_iter, test_iter = IMDB(split=('train', 'test'))
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for label, text in train_iter:
        train_texts.append(tokenize(text))
        train_labels.append(1 if label == 'pos' else 0)

    for label, text in test_iter:
        test_texts.append(tokenize(text))
        test_labels.append(1 if label == 'pos' else 0)

    # Set max_len to the minimum document length
    min_doc_length = min([len(text) for text in train_texts])
    print(f"Setting max_len to minimum document length: {min_doc_length}")
    max_len = min_doc_length

    # Preprocess data
    X_train = preprocess_data(train_texts, vocab, max_len)
    y_train = torch.tensor(train_labels, dtype=torch.float32)

    X_test = preprocess_data(test_texts, vocab, max_len)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    # Initialize LSTM model
    embedding_dim = embeddings.size(1)
    hidden_dim = 128
    output_dim = 1
    lstm_model = LSTMClassifier(len(vocab), embedding_dim, hidden_dim, output_dim, embeddings)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 5
    for epoch in range(n_epochs):
        lstm_model.train()
        optimizer.zero_grad()

        outputs = lstm_model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    lstm_model.eval()
    with torch.no_grad():
        predictions = lstm_model(X_test).squeeze()
        predictions = (predictions >= 0.5).float()
        accuracy = (predictions == y_test).sum().item() / len(y_test)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# --- Main Function ---
if __name__ == "__main__":
    # Step 1: Train SkipGram Embeddings
    embeddings, vocab = train_skipgram_embeddings()

    # Step 2: Train LSTM Classifier using Pretrained Embeddings
    train_lstm_classifier(embeddings, vocab)
