import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import ast

def make_dataset(filename , task):
    
    dataset_instance = Dataset(filename=filename , task=task)
    dataset_instance.load()
    dataset = dataset_instance.pre_process()

    return dataset




class Dataset:
    
    def __init__(self, filename : str, task : str):
        self.filename = filename
        self.task = task

    def load(self):
        self.df = pd.read_csv(self.filename)
        
    
    def pre_process(self):

        
        self.df['tokens'] = self.df['tokens'].apply(ast.literal_eval)

        if self.task == 'is_comic_video':
            self.dataset = self.df

        elif self.task == "is_name":

            self.df['is_name'] = self.df['is_name'].apply(ast.literal_eval)
            
            # Expand the rows so that each row corresponds to a single token and its associated label
            expanded_rows = []
            for tokens, labels in zip(self.df['tokens'], self.df['is_name']):
                for token, label in zip(tokens, labels):
                    expanded_rows.append([token, label])
            
            self.dataset = pd.DataFrame(expanded_rows, columns=['token', 'is_name'])
        
        elif self.task == "find_comic_name":

            label_encoder = LabelEncoder()
            self.df['comic_name'] = label_encoder.fit_transform(self.df['comic_name'].astype(str))

            self.df['tokens'].apply(lambda tokens: ' '.join(tokens))
            
            self.df['tokens'] = self.df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
            
            self.dataset = self.df

        return self.dataset


def tokenize(text:str):

    pattern = r"\b\w+[-’]?\w*\b|[-’]"
    tokens = re.findall(pattern, text)

    # Flatten the list of tuples returned by findall and remove empty strings
    tokens = [token for token in tokens if token]

    return tokens
