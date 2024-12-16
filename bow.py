from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from abc import ABC, abstractmethod


class Bow(ABC):
    def __init__(self):
        self.embedding = None

    @abstractmethod
    def vectorize(self, corpus):
        pass

        
class CountBow(Bow):
    def vectorize(self, corpus):
        vectorizer = CountVectorizer()
        self.embedding = vectorizer.fit_transform(corpus)

    def vectorizeFreq(self, corpus):
        self.vectorize(corpus)
        row_sums = self.embedding.sum(axis=1).A1
        self.embedding = self.embedding.multiply(1.0 / row_sums[:, np.newaxis])


class TfidfBow(Bow):
    def vectorize(self, corpus):
        vectorizer = TfidfVectorizer()
        self.embedding = vectorizer.fit_transform(corpus)
    
    def vectorizeNorm(self, corpus):
        self.vectorize(corpus)
        row_sums = self.embedding.max(axis=1).A1
        self.embedding = self.embedding.multiply(1.0 / row_sums[:, np.newaxis])


class Ngram():
    def __init__(self):
        self.embedding = None

    def vectorize(self, corpus, win_size=2):
        vectorizer = TfidfVectorizer(ngram_range=(win_size,win_size))
        self.embedding = vectorizer.fit_transform(corpus)

    def vectorizeNorm(self, corpus, win_size=2):
        self.vectorize(corpus,win_size=win_size)
        row_sums = self.embedding.max(axis=1).A1
        self.embedding = self.embedding.multiply(1.0 / row_sums[:, np.newaxis])