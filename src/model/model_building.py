# src/model/model_building.py
# --------------------------------------------------
# This script does 2 things:
# 1. Convert cleaned comment text into numbers using
#    TF-IDF (Term Frequency - Inverse Document Frequency)
# 2. Train a LightGBM classifier on those numbers to
#    predict the sentiment category (positive/neutral/negative)
#
# It saves two files in the project root folder:
# - tfidf_vectorizer.pkl  (the text-to-numbers converter)
# - lgbm_model.pkl        (the trained model)
# Both files are needed together to make predictions later.
# --------------------------------------------------

import pandas as pd
import pickle
import yaml
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer


def load_params(params_path):
    # Open params.yaml and load all the settings
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    return params


def load_data(file_path):
    # Read the cleaned training data
    df = pd.read_csv(file_path)

    # Fill any empty cells with an empty string, so nothing breaks later
    df = df.fillna("")

    return df


def apply_tfidf(train_data, max_features, ngram_range):
    # TfidfVectorizer converts text into a matrix of numbers
    # max_features: how many unique words/phrases to keep (the most important ones)
    # ngram_range: whether to look at single words, pairs of words, or triples of words
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    # Get the comment text and the true label (category) as separate lists
    X_train_text = train_data["clean_comment"].values
    y_train = train_data["category"].values

    # Learn the vocabulary from the training text, and convert it into numbers
    X_train_tfidf = vectorizer.fit_transform(X_train_text)

    # Save the vectorizer so we can reuse the SAME vocabulary later
    # (during evaluation, and when the app makes real predictions)
    with open("tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    return X_train_tfidf, y_train


def train_lgbm(X_train, y_train, learning_rate, max_depth, n_estimators):
    # Create a LightGBM classifier for a 3-class problem
    # (negative, neutral, positive)
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        is_unbalance=True,
        class_weight="balanced",
        reg_alpha=0.1,       # helps prevent overfitting (L1 regularization)
        reg_lambda=0.1,      # helps prevent overfitting (L2 regularization)
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators
    )

    # Train the model on our TF-IDF numbers and true labels
    model.fit(X_train, y_train)

    return model


def save_model(model, file_path):
    # Save the trained model to a file using pickle
    # (pickle turns a python object into bytes that can be saved and reloaded later)
    with open(file_path, "wb") as file:
        pickle.dump(model, file)


def main():
    # Step 1: Load settings from params.yaml
    params = load_params("params.yaml")
    max_features = params["model_building"]["max_features"]
    ngram_range = tuple(params["model_building"]["ngram_range"])
    learning_rate = params["model_building"]["learning_rate"]
    max_depth = params["model_building"]["max_depth"]
    n_estimators = params["model_building"]["n_estimators"]

    # Step 2: Load the cleaned training data
    train_data = load_data("data/interim/train_processed.csv")

    # Step 3: Convert the text into TF-IDF numbers
    X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

    # Step 4: Train the LightGBM model
    model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

    # Step 5: Save the trained model
    save_model(model, "lgbm_model.pkl")

    print("Model building complete! lgbm_model.pkl and tfidf_vectorizer.pkl saved in project root folder.")


if __name__ == "__main__":
    main()
