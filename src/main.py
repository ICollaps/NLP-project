# main.py

import click
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from joblib import dump , load



from data.make_dataset import make_dataset , tokenize
from features.make_features import make_features , get_config
from model.main import make_model , prepare_data , split_data

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/model/trained_models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):

    # Read CSV
    click.echo("Trying to make dataset...")
    dataset = make_dataset(filename=input_filename, task=task)
    click.echo("Dataset done")

    click.echo("Trying to get features config...")
    features_list = get_config()
    click.echo("Config done")

    # Make features (tokenization, lowercase, stopwords, stemming...)
    click.echo("Trying to make features...")
    X, y = make_features(df=dataset, task=task , features_list=features_list)
    click.echo("Features done")


    # Rebalance dataset, vectorise ...
    click.echo("Trying to prepare data...")
    X_combined, y , vectorizer = prepare_data(X=X, y=y , task=task , features_list=features_list)
    click.echo("Data preparation done")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_combined, y)

    # Object with .fit, .predict methods
    click.echo("Trying to make model...")
    model = make_model()

    click.echo("Trying to fit model...")
    model.fit(X_train=X_train , y_train=y_train)
    click.echo("Model done")

    model.dump(filename=model_dump_filename)
    dump(vectorizer, f'src/model/training_vectorizer/{task}_vectorizer.joblib')



@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/model/trained_models/dump.json", help="File to dump model")
@click.option("--output_filename", default="src/data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):

    # Read CSV
    click.echo("Trying to make dataset...")
    dataset = make_dataset(filename=input_filename, task=task)
    click.echo("Dataset done")

    click.echo("Trying to get features config...")
    features_list = get_config()
    click.echo("Config done")

    # Make features (tokenization, lowercase, stopwords, stemming...)
    click.echo("Trying to make features...")
    X, y = make_features(df=dataset, task=task , features_list=features_list)
    click.echo("Features done")
    
    # Rebalance dataset, vectorise ...
    click.echo("Trying to prepare data...")
    X_combined, y , vectorizer = prepare_data(X=X, y=y , task=task , features_list=features_list)
    click.echo("Data preparation done")

    X_train, X_test, y_train, y_test = split_data(X_combined, y)

    # Object with .fit, .predict methods
    click.echo("Trying to make model...")
    model = make_model()

    # Load pretrained and dumped model
    click.echo("Trying to load model...") 
    model.load(filename=model_dump_filename)
    click.echo("Model loaded...")
    
    # Make predictions
    click.echo("Trying to predict...")
    y_pred = model.predict(X_test)
    click.echo("Predictions done...")

    click.echo("Trying to test model...")
    model.test(y_test=y_test, y_pred=y_pred)
    click.echo("Test done...")
    
    # Save predictions to CSV
    click.echo("Trying to save predictions...")
    output_df = pd.DataFrame({"prediction": y_pred})
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):

    # Read CSV
    click.echo("Trying to make dataset...")
    dataset = make_dataset(filename=input_filename, task=task)
    click.echo("Dataset done")

    click.echo("Trying to get features config...")
    features_list = get_config()
    click.echo("Config done")

    # Make features (tokenization, lowercase, stopwords, stemming...)
    click.echo("Trying to make features...")
    X, y = make_features(df=dataset, task=task , features_list=features_list)
    click.echo("Features done")

    # Rebalance dataset, vectorise ...
    click.echo("Trying to prepare data...")
    X_combined, y, vectorizer  = prepare_data(X=X, y=y , task=task , features_list=features_list)
    click.echo("Data preparation done")

    # Object with .fit, .predict methods
    click.echo("Trying to make model...")
    model = make_model()
    click.echo("Model done")

    # Perform k-fold cross validation
    click.echo("Trying to evaluate model...")
    model.evaluate(X=X_combined , y=y)


@click.command()
@click.option("--video_name" , default="Seins en quête d’une libération - La chronique d'Isabelle Sorente")
@click.option("--is_comic_video_model_path", default="src/model/trained_models/is_comic_video_model.json", help="is_comic_video task trained model")
@click.option("--is_name_model_path", default="src/model/trained_models/is_name_model.json", help="is_name task trained model")
@click.option("--find_comic_name_model_path", default="src/model/trained_models/find_comic_name_model.json", help="find_comic_name task trained model")

def use(video_name:str, is_comic_video_model_path:str , is_name_model_path:str , find_comic_name_model_path:str):

    click.echo(f"Choised video_name : {video_name}")
    
    click.echo(f"Trying to load is_comic_video model and vectorizer from {is_comic_video_model_path} ")
    is_comic_video_model = make_model()
    is_comic_video_model.load(filename=is_comic_video_model_path)
    is_comic_video_vectorizer = load('src/model/training_vectorizer/is_comic_video_vectorizer.joblib')
    click.echo(f"is_comic_video model and vectorizer loaded from {is_comic_video_model_path} ")

    click.echo(f"Trying to load is_name model and vectorizer from {is_name_model_path} ")
    is_name_model = make_model()
    is_name_model.load(filename=is_name_model_path)
    is_name_vectorizer = load('src/model/training_vectorizer/is_name_vectorizer.joblib')
    click.echo(f"is_name model loaded from {is_name_model_path} ")

    click.echo(f"Trying to load find_comic_name model from {find_comic_name_model_path} ")
    find_comic_name_model = make_model()
    find_comic_name_model.load(filename=find_comic_name_model_path)
    find_comic_name_vectorizer = load('src/model/training_vectorizer/find_comic_name_vectorizer.joblib')
    click.echo(f"find_comic_name model loaded from {find_comic_name_model_path} ")

    tokens = tokenize(text=video_name)

    is_comic_video_features = is_comic_video_vectorizer.transform([video_name])
    is_comic_video = is_comic_video_model.predict(is_comic_video_features)[0]

    if is_comic_video == 1:
        click.echo("This video name was predicted as comic")

        is_name_features = is_name_vectorizer.transform(tokens)
        is_name = is_name_model.predict(is_name_features)

        if 1 in is_name:
            click.echo("A name was predicted in the video name")
            indices = [i for i, val in enumerate(is_name) if val == 1]

            click.echo(f"Names find by index for the video name: {video_name}")   
            for i in indices:
                click.echo(tokens[i])

            # click.echo("Names find by find_comic_name model :")  
            # find_comic_name_features = find_comic_name_vectorizer.transform(tokens)
            # comic_name = find_comic_name_model.predict(find_comic_name_features)
            # print(comic_name)



cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)
cli.add_command(use)


if __name__ == "__main__":
    cli()
