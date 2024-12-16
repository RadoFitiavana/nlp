import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.datasets import IMDB
from embedding import SkipGramMLP, LSTMClassifier, trainEmbeddingMLP, Word2VecDataset, tokenize
from collections import Counter

# Function to build vocabulary
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {word: i for i, (word, _) in enumerate(counter.items(), 1)}
    vocab['<pad>'] = 0  # Add padding token
    return vocab

# Function to prepare the dataset
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize and convert to indices
        tokens = tokenize(text)
        indices = [self.vocab.get(word, 0) for word in tokens]  # 0 is for unknown words
        # Pad/truncate the sequences
        indices = indices[:self.max_length] + [0] * (self.max_length - len(indices))
        return torch.tensor(indices), torch.tensor(label)

# Load the IMDB dataset
train_iter, test_iter = IMDB(split='train'), IMDB(split='test')

# Prepare the dataset
train_texts, train_labels = zip(*[(text, label) for label, text in train_iter])
test_texts, test_labels = zip(*[(text, label) for label, text in test_iter])

# Build vocabulary
vocab = build_vocab(train_texts)

# Set parameters
max_length = 100  # Max sequence length

# Create the dataset and dataloaders
train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_length)
test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

vocab_size = len(vocab)
# Initialize and train the SkipGram MLP model for embeddings
embedding_model = SkipGramMLP(len(vocab), embedding_dim=100)
trainEmbeddingMLP(embedding_model, train_loader, vocab_size, n_epochs=5)

# Get the learned embeddings from the SkipGram model
pretrained_embeddings = embedding_model.embeddings.weight.data

# Initialize the LSTM classifier with the pretrained embeddings
lstm_model = LSTMClassifier(len(vocab), embedding_dim=100, hidden_dim=128, output_dim=1, pretrained_embeddings=pretrained_embeddings)

# Train the LSTM classifier
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# Training loop for the LSTM
def train_lstm_classifier(model, train_loader, val_loader, n_epochs=5):
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {total_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for texts, labels in val_loader:
                outputs = model(texts)
                predicted = (outputs.squeeze() > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            accuracy = 100 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")

# Train the model
train_lstm_classifier(lstm_model, train_loader, test_loader, n_epochs=5)
