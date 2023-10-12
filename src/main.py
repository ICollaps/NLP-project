import click
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from data.make_dataset import make_dataset
from features.make_features import make_features
from model.main import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/model/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model()
    model.fit(X, y)

    return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/model/dump.json", help="File to dump model")
@click.option("--output_filename", default="src/data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):

    # Load testing data
    df = make_dataset(input_filename)
    X, y = make_features(df, task)
    
    # Load the trained model
    model = make_model()
    model.load(model_dump_filename)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Evaluate the model using the evaluate method
    model.evaluate(X, y)
    
    # Save predictions to CSV
    output_df = pd.DataFrame({"prediction": predictions})
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):

    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return model.evaluate(X, y)



cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
