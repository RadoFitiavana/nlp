import torch
from torchtext.datasets import IMDB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from bow import CountBow, TfidfBow, Ngram
import numpy as np

# --- Utility Functions ---
def load_imdb_data():
    """Load IMDB dataset and prepare texts and labels."""
    print("Loading IMDB Dataset...")
    train_iter, test_iter = IMDB(split=('train', 'test'))
    texts, labels = [], []
    for label, text in train_iter:
        texts.append(text)
        labels.append(1 if label == 'pos' else 0)
    for label, text in test_iter:
        texts.append(text)
        labels.append(1 if label == 'pos' else 0)
    print(f"Loaded {len(texts)} samples.\n")
    return texts, labels


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train logistic regression and evaluate its performance."""
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def prepare_bow_features(bow_method, train_texts, test_texts, method="default", win_size=2):
    """
    Prepare Bag of Words features based on the chosen method.
    
    Parameters:
    - bow_method: Bow object (CountBow, TfidfBow, Ngram)
    - train_texts: List of training texts
    - test_texts: List of test texts
    - method: 'default', 'freq', or 'norm'
    - win_size: Window size for Ngram (default=2)
    """
    print(f"Vectorizing with {bow_method.__class__.__name__} using method '{method}'...")

    # Select vectorization method
    if isinstance(bow_method, Ngram):
        if method == "norm":
            bow_method.vectorizeNorm(train_texts, win_size=win_size)
        else:
            bow_method.vectorize(train_texts, win_size=win_size)
    else:
        if method == "freq" and hasattr(bow_method, "vectorizeFreq"):
            bow_method.vectorizeFreq(train_texts)
        elif method == "norm" and hasattr(bow_method, "vectorizeNorm"):
            bow_method.vectorizeNorm(train_texts)
        else:
            bow_method.vectorize(train_texts)

    # Sparse matrix for training data
    X_train = bow_method.embedding.copy()

    if isinstance(bow_method, Ngram):
        if method == "norm":
            bow_method.vectorizeNorm(test_texts, win_size=win_size)
        else:
            bow_method.vectorize(test_texts, win_size=win_size)
    else:
        if method == "freq" and hasattr(bow_method, "vectorizeFreq"):
            bow_method.vectorizeFreq(train_texts)
        elif method == "norm" and hasattr(bow_method, "vectorizeNorm"):
            bow_method.vectorizeNorm(test_texts)
        else:
            bow_method.vectorize(test_texts)
    # Transform the test data explicitly
    X_test = bow_method.embedding.copy()
    
    print(f"Train Feature Matrix Shape: {X_train.shape}")
    print(f"Test Feature Matrix Shape: {X_test.shape}\n")
    return X_train, X_test


def main():
    # Load data
    texts, labels = load_imdb_data()

    # Split into train and test sets
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # CountBoW (default)
    count_bow = CountBow()
    X_train_count, X_test_count = prepare_bow_features(count_bow, X_train_texts, X_test_texts)
    train_logistic_regression(X_train_count, y_train, X_test_count, y_test)

    # CountBoW with Frequency Normalization
    X_train_count_freq, X_test_count_freq = prepare_bow_features(count_bow, X_train_texts, X_test_texts, method="freq")
    train_logistic_regression(X_train_count_freq, y_train, X_test_count_freq, y_test)

    # TfidfBoW (default)
    tfidf_bow = TfidfBow()
    X_train_tfidf, X_test_tfidf = prepare_bow_features(tfidf_bow, X_train_texts, X_test_texts)
    train_logistic_regression(X_train_tfidf, y_train, X_test_tfidf, y_test)

    # TfidfBoW with Normalization
    X_train_tfidf_norm, X_test_tfidf_norm = prepare_bow_features(tfidf_bow, X_train_texts, X_test_texts, method="norm")
    train_logistic_regression(X_train_tfidf_norm, y_train, X_test_tfidf_norm, y_test)

    # Ngram (default, win_size=2)
    ngram_bow = Ngram()
    X_train_ngram, X_test_ngram = prepare_bow_features(ngram_bow, X_train_texts, X_test_texts, win_size=2)
    train_logistic_regression(X_train_ngram, y_train, X_test_ngram, y_test)

    # Ngram with Normalization (win_size=2)
    X_train_ngram_norm, X_test_ngram_norm = prepare_bow_features(ngram_bow, X_train_texts, X_test_texts, method="norm", win_size=2)
    train_logistic_regression(X_train_ngram_norm, y_train, X_test_ngram_norm, y_test)


if __name__ == "__main__":
    main()
