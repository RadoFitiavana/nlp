# logisticNLP.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
from bow import CountBow, TfidfBow, Ngram
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the IMDB dataset
def load_imdb_data():
    # Load the IMDB dataset from Keras
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    
    # Convert the sequence of integers to text by mapping indices to words
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

    # Decode the reviews into text
    train_corpus = [decode_review(review) for review in train_data]
    test_corpus = [decode_review(review) for review in test_data]
    
    # Combine the training and test sets
    corpus = train_corpus + test_corpus
    labels = np.concatenate([train_labels, test_labels])
    
    return corpus, labels

# Function to train and evaluate a model
def train_and_evaluate(vectorizer, corpus, labels):
    # Vectorize the corpus
    vectorizer.vectorize(corpus)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(vectorizer.embedding, labels, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Load dataset
corpus, labels = load_imdb_data()

# Initialize the vectorizers
count_vectorizer = CountBow()
tfidf_vectorizer = TfidfBow()
ngram_vectorizer = Ngram()

# Train and evaluate each model
print("Training and evaluating CountBow model...")
count_accuracy = train_and_evaluate(count_vectorizer, corpus, labels)
print(f"CountBow Model Accuracy: {count_accuracy:.4f}")

print("Training and evaluating TfidfBow model...")
tfidf_accuracy = train_and_evaluate(tfidf_vectorizer, corpus, labels)
print(f"TfidfBow Model Accuracy: {tfidf_accuracy:.4f}")

print("Training and evaluating Ngram model...")
ngram_accuracy = train_and_evaluate(ngram_vectorizer, corpus, labels)
print(f"Ngram Model Accuracy: {ngram_accuracy:.4f}")
