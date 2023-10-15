from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
from typing import Type
import re
import json
import unidecode



import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Télécharger le corpus français
nltk.download('stopwords')

# Obtenir les stopwords français
french_stopwords = set(stopwords.words('french'))

# Initialiser le stemmer français
stemmer = SnowballStemmer('french')




def make_model():

    # Chargement du fichier de configuration
    with open("src/config.json", "r") as file:
        config = json.load(file)

    pipeline = make_pipeline(config=config)
    return Model(pipeline)



def make_pipeline(config):

    feature_extraction_method = config["feature_extraction"]["method"]
    
    
    
    if feature_extraction_method == "CountVectorizer":
        vectorizer = CountVectorizer(
            preprocessor=preprocess_text
        )

    elif feature_extraction_method == "TfidfVectorizer":
        vectorizer = TfidfVectorizer(
            stop_words=list(french_stopwords),
            preprocessor=preprocess_text,
            ngram_range=(1, 1)
            )

    else:
        raise ValueError(f"Unknown feature extraction method: {feature_extraction_method}")
    
    model_type = config["model"]["type"]
    
    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ("vectorizer", vectorizer),
        ("model", RandomForestClassifier()),
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

    def evaluate(self,X,y):
        # Scikit learn has function for cross validation
        scores = cross_val_score(self.pipeline, X, y, scoring="accuracy")
        print(f"Got accuracy {100 * np.mean(scores)}%")
        return scores
    



def preprocess_text(text):
    """
    Preprocess the input text by tokenizing, removing stopwords, and stemming.

    Parameters:
    - text (str): The input text to preprocess.

    Returns:
    - str: The preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()

    # Retirer la ponctuation et les nombres
    text = re.sub(r'[^\w\s]', '', text)

    # Remove accents
    text = unidecode.unidecode(text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text, language='french')
    
    # Remove non-alphabetic characters and stopwords
    clean_tokens = [
        token for token in tokens 
        if token.isalpha() and token not in french_stopwords
    ]

    print(clean_tokens)
    
    return ' '.join(clean_tokens)












