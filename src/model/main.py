from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


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

    vectorizer_method = config["vectorizer"]["method"]
    features_method = config["features"]
    
    
    
    
    if vectorizer_method == "CountVectorizer":
        vectorizer = CountVectorizer(
            stop_words=preprocessed_stopwords,
            preprocessor=preprocess_text
        )

    elif vectorizer_method == "TfidfVectorizer":
        vectorizer = TfidfVectorizer(
            stop_words=preprocessed_stopwords,
            preprocessor=preprocess_text,
            ngram_range=(1, 1)
            )

    else:
        raise ValueError(f"Unknown feature extraction method: {vectorizer_method}")
    


    
    # Define preprocessing for numeric and text features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_method),
            ('text', vectorizer, 'video_name')
        ]
    )
    
    model_type = config["model"]["type"]
    
    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Combine preprocessing with classifier
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
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

    # Chargement du fichier de configuration
    with open("src/config.json", "r") as file:
        config = json.load(file)

    preprocess_operations = config["preprocess_operations"]


    if "lowercase" in preprocess_operations:
        # Convert text to lowercase
        text = text.lower()

    if "punctuation" in preprocess_operations:
        # Retirer la ponctuation
        text = re.sub(r'[^\w\s]', '', text)

    if "accent" in preprocess_operations:
        # Remove accents
        text = unidecode.unidecode(text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text, language='french')
    

    if "stemming" in preprocess_operations:
        clean_tokens = [
            stemmer.stem(token) for token in tokens 
            if token.isalpha() and token not in french_stopwords
        ]
    else:
        clean_tokens = [
            token for token in tokens 
            if token.isalpha() and token not in french_stopwords
        ]

    # print(clean_tokens)
    
    return ' '.join(clean_tokens)

# Appliquer la même prétraitement aux stop words qu'au texte
preprocessed_stopwords = list(set(preprocess_text(word) for word in french_stopwords))












