# src/model/model_evaluation.py
# --------------------------------------------------
# This script checks how well our trained model performs
# on data it has NEVER seen before (the test set).
# It calculates precision, recall, and f1-score for each
# sentiment category, and saves a confusion matrix image
# so we can visually see where the model gets confused.
# --------------------------------------------------

import pandas as pd
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    # Read the cleaned test data
    df = pd.read_csv(file_path)

    # Fill any empty cells with an empty string
    df = df.fillna("")

    return df


def load_model(model_path):
    # Load the trained model back from its pickle file
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def load_vectorizer(vectorizer_path):
    # Load the saved TF-IDF vectorizer back from its pickle file
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)
    return vectorizer


def evaluate_model(model, X_test, y_test):
    # Ask the model to predict sentiment for the test data
    y_pred = model.predict(X_test)

    # classification_report gives precision, recall, and f1-score per class
    report = classification_report(y_test, y_pred, output_dict=True)

    # confusion_matrix shows how many predictions were correct or incorrect, per class
    cm = confusion_matrix(y_test, y_pred)

    return report, cm


def save_confusion_matrix(cm, output_path):
    # Draw the confusion matrix as a heatmap image and save it as a PNG file
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Test Data")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(output_path)
    plt.close()


def save_report(report, output_path):
    # Save the classification report as a JSON file so we can inspect it later
    with open(output_path, "w") as file:
        json.dump(report, file, indent=4)


def main():
    # Step 1: Load the trained model and vectorizer
    model = load_model("lgbm_model.pkl")
    vectorizer = load_vectorizer("tfidf_vectorizer.pkl")

    # Step 2: Load the cleaned test data
    test_data = load_data("data/interim/test_processed.csv")

    # Step 3: Convert test comments into TF-IDF numbers using the SAME vectorizer
    X_test_tfidf = vectorizer.transform(test_data["clean_comment"].values)
    y_test = test_data["category"].values

    # Step 4: Evaluate the model
    report, cm = evaluate_model(model, X_test_tfidf, y_test)

    # Step 5: Save the results
    save_report(report, "reports/metrics.json")
    save_confusion_matrix(cm, "reports/figures/confusion_matrix.png")

    print("Model evaluation complete! Check reports/metrics.json and reports/figures/confusion_matrix.png")


if __name__ == "__main__":
    main()
