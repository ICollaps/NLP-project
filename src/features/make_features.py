

def make_features(df, task):
    y = get_output(df, task)
    X = df[['video_name']].copy()
    
    # Feature: Length of the video name
    X['video_name_length'] = X['video_name'].apply(len)
    
    # Feature: Number of words in the video name
    X['video_name_word_count'] = X['video_name'].apply(lambda x: len(x.split()))
    
    # Feature: Number of uppercase letters
    X['num_uppercase'] = X['video_name'].apply(lambda x: sum(1 for c in x if c.isupper()))
    
    # Feature: Presence of numbers
    X['has_number'] = X['video_name'].apply(lambda x: int(any(char.isdigit() for char in x)))

    
    return X, y




def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y






