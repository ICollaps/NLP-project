# make_features.py
import pandas as pd

def make_features(df : pd.DataFrame, task: str):
    
    y = get_y(df, task)
    X = get_X(df, task)

    # features_list = ['length', 'video_name_word_count']
    # features_list = ['is_capitalized', 'length', 'has_non_alpha']
    features_list = []

    X = create_features(X=X , features_list=features_list)
    


    return X, y


def get_y(df : pd.DataFrame, task: str):

    if task == "is_comic_video":
        y = df["is_comic"]

    elif task == "is_name":
        y = df["is_name"]

    elif task == "find_comic_name":
        y = df["comic_name"]

    else:
        raise ValueError("Unknown task")

    return y


def get_X(df: pd.DataFrame, task: str):

    if task == "is_comic_video":
        X = df[['video_name']].copy()
        X = X.rename(columns={'video_name': 'X'})

    elif task == "is_name":
        X = df[['token']].copy()
        X = X.rename(columns={'token': 'X'})

    elif task == "find_comic_name":
        X = df[['tokens']].copy()
        X = X.rename(columns={'tokens': 'X'})
    
    else:
        raise ValueError("Unknown task")

    return X


def create_features(
        X : pd.DataFrame,
        features_list : list[str]
    ):
    if 'is_capitalized' in features_list:
        X['is_capitalized'] = X['X'].apply(lambda x: x.istitle())

    if 'length' in features_list:
        X['length'] = X['X'].apply(len)

    if 'has_non_alpha' in features_list:
        X['has_non_alpha'] = X['X'].apply(lambda x: not x.isalpha())

    if 'video_name_word_count' in features_list:
        X['video_name_word_count'] = X['X'].apply(lambda x: len(x.split()))
    
    return X

