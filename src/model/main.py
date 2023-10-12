from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib
from typing import Type


def make_model():
    pipeline = make_pipeline()
    return Model(pipeline)


def make_pipeline():
    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", RandomForestClassifier()),
    ])


class Model:
    
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
    
    def predict(self, X):
        return self.pipeline.predict(X)

    def dump(self, filename):
        joblib.dump(self.pipeline, filename)
    
    def load(self, filename):
        self.pipeline = joblib.load(filename)
    
