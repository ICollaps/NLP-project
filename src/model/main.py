from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split , cross_val_score , cross_val_predict
from sklearn.metrics import classification_report, accuracy_score , make_scorer
import numpy as np
import joblib


from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import pandas as pd

def make_model():


    model = Model(model_type="RandomForestClassifier")
    
    return model

    


def prepare_data(X:pd.DataFrame,y:pd.Series , task:str , features_list: list):

    if task == "is_name":
        
        X, y = rebalance_dataset(X=X , y=y)

    X_combined, vectorizer = vectorize(X=X , features_list= features_list)

    return X_combined, y , vectorizer


def rebalance_dataset(X: pd.DataFrame, y: pd.Series):
    # Create a DataFrame by combining X and y
    df = pd.concat([X, y], axis=1)

    # Separate the majority and minority classes
    df_majority = df[df.is_name == 0]
    df_minority = df[df.is_name == 1]


    # Downsample the majority class
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    
    # Combine downsampled majority class and minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    
    # Separate X and y again
    X_downsampled = df_downsampled.drop('is_name', axis=1)
    y_downsampled = df_downsampled['is_name']

    return X_downsampled, y_downsampled



def split_data(X_combined_data , y: pd.Series):

    X_train, X_test, y_train, y_test = train_test_split(X_combined_data, y , test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def vectorize(X: pd.DataFrame , features_list: list):

    tfidf_vectorizer = TfidfVectorizer()

    X_sparse_matrix = tfidf_vectorizer.fit_transform(X['X'])

    # features_list = ['is_capitalized', 'length', 'has_non_alpha']
    # features_list = ['length', 'video_name_word_count']
    # features_list = []

    features = X[features_list].astype(float)

    features_sparse = csr_matrix(features.values)

    X_combined = hstack([X_sparse_matrix, features_sparse])

    return X_combined , tfidf_vectorizer


class Model:
    
    def __init__(self, model_type:str):
        self.model_type = model_type
        if model_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(random_state=42, n_jobs=1)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def test(self, y_test, y_pred):

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f'Accuracy : {accuracy}')
        print(f'Classification Report {class_report}')

    
    def evaluate(self, X, y, cv=5):
        # Obtenir les prédictions pour chaque split de la cross-validation
        y_pred = cross_val_predict(self.model, X, y, cv=cv)

        # Calculer l'accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f'Cross-validated accuracy: {accuracy * 100:.2f}%')

        # Calculer le rapport de classification
        class_report = classification_report(y, y_pred)
        print(f'Classification Report:\n{class_report}')
    

    def dump(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)
        
        

