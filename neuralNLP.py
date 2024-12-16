# neuralNLP.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator
from embedding import SkipGramMLP, trainEmbeddingMLP, LSTMClassifier, preprocess_data
import embedding

# Step 1: Load the IMDB dataset using torchtext
TEXT = Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False, is_target=True)

# Load and split the IMDB dataset into training and test sets
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary using pre-trained embeddings like GloVe, or create it from the IMDB corpus
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)

# Step 2: Generate center-context pairs and train the SkipGram MLP model
# Get the vocabulary and tokenize the text
vocab = TEXT.vocab.stoi  # Indexing for vocabulary
corpus = [vars(example)["text"] for example in train_data]  # List of tokenized sentences

# Generate center-context pairs for the SkipGram model
pairs = embedding.generate_pairs(vocab, corpus)
dataset = embedding.Word2VecDataset(pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train the SkipGram MLP model
embedding_dim = 100  # Define embedding dimension
skipgram_mlp = SkipGramMLP(len(TEXT.vocab), embedding_dim)
trainEmbeddingMLP(skipgram_mlp, dataloader)

# Get the learned embeddings from the SkipGram MLP
pretrained_embeddings = skipgram_mlp.embeddings.weight.data

# Step 3: Initialize and fine-tune the LSTM model with transfer learning
hidden_dim = 128
output_dim = 1  # Binary classification (positive or negative sentiment)
lstm_classifier = LSTMClassifier(len(TEXT.vocab), embedding_dim, hidden_dim, output_dim, pretrained_embeddings)

# Step 4: Prepare the IMDB data for the LSTM model
# We will use BucketIterator to batch the data with sorted sequences
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size=32, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    sort_within_batch=True,
    sort_key=lambda x: len(x.text)
)

# Step 5: Training the LSTM model
optimizer = optim.Adam(lstm_classifier.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

def train_lstm_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        optimizer.zero_grad()
        
        # The LSTM model expects the text data in the form (text, lengths)
        text, text_lengths = batch.text
        labels = batch.label
        
        predictions = model(text).squeeze(1)  # Get the predictions from the LSTM
        loss = criterion(predictions, labels.float())
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Function to calculate accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc

# Training Loop
n_epochs = 5
for epoch in range(n_epochs):
    train_loss, train_acc = train_lstm_model(lstm_classifier, train_iterator, optimizer, criterion)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

