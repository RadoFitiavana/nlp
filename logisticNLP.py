# logisticNLP.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from bow import CountBow, TfidfBow, Ngram

# Assuming you have the IMDB dataset loaded into `corpus` and `labels`
# Example:
# corpus = ['text1', 'text2', ...]
# labels = [0, 1, ...]

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
