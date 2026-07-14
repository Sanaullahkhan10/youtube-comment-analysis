# src/model/model_evaluation.py
# --------------------------------------------------
# This script checks how well our trained model performs
# on data it has NEVER seen before (the test set), and logs
# everything (parameters, metrics, model, confusion matrix)
# to MLflow so we can track and compare experiment runs.
# --------------------------------------------------

import pandas as pd
import pickle
import json
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_params(params_path):
    # Open params.yaml and load all the settings
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    return params


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
    # Point MLflow to a local SQLite database file to store experiment metadata.
    # This is MLflow's current recommended default (the old "file:./mlruns" style
    # is now in maintenance mode and no longer supported by the UI/server).
    # Later, in the AWS deployment phase, this will point to a remote server instead.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("dvc-pipeline-runs")

    # Start a new MLflow run - everything inside this "with" block belongs to this run
    with mlflow.start_run() as run:
        # Step 1: Load settings from params.yaml and log them to MLflow
        params = load_params("params.yaml")
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Step 2: Load the trained model and vectorizer
        model = load_model("lgbm_model.pkl")
        vectorizer = load_vectorizer("tfidf_vectorizer.pkl")

        # Step 3: Load the cleaned test data
        test_data = load_data("data/interim/test_processed.csv")

        # Step 4: Convert test comments into TF-IDF numbers using the SAME vectorizer
        X_test_tfidf = vectorizer.transform(test_data["clean_comment"].values)
        y_test = test_data["category"].values

        # Step 5: Evaluate the model
        report, cm = evaluate_model(model, X_test_tfidf, y_test)

        # Step 6: Save the results locally as files
        save_report(report, "reports/metrics.json")
        save_confusion_matrix(cm, "reports/figures/confusion_matrix.png")

        # Step 7: Log precision, recall, and f1-score for each class to MLflow
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"test_{label}_precision", metrics["precision"])
                mlflow.log_metric(f"test_{label}_recall", metrics["recall"])
                mlflow.log_metric(f"test_{label}_f1_score", metrics["f1-score"])

        # Step 8: Log the confusion matrix image and the vectorizer file as MLflow artifacts
        mlflow.log_artifact("reports/figures/confusion_matrix.png")
        mlflow.log_artifact("tfidf_vectorizer.pkl")

        # Step 9: Log the trained model itself to MLflow.
        # We explicitly mark these types as "trusted" because MLflow now uses
        # the safer skops format (instead of raw pickle) to save models, which
        # requires a whitelist of any non-standard types used inside the model.
        mlflow.sklearn.log_model(
            model,
            name="lgbm_model",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier"
            ]
        )

        # Step 10: Add descriptive tags so this run is easy to identify later
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("task", "Sentiment Analysis")
        mlflow.set_tag("dataset", "YouTube Comments")

        # Step 11: Save this run's ID and model path - we will need this
        # later when registering the model in the MLflow Model Registry
        model_info = {
            "run_id": run.info.run_id,
            "model_path": "lgbm_model"
        }
        with open("experiment_info.json", "w") as file:
            json.dump(model_info, file, indent=4)

        print("Model evaluation complete! Metrics logged to MLflow, saved to reports/metrics.json")


if __name__ == "__main__":
    main()
