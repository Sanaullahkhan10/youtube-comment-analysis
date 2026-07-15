# scripts/test_model_performance.py
# --------------------------------------------------
# This test checks that the model's accuracy, precision,
# recall, and f1-score on the held-out test set are all
# above a minimum acceptable threshold. If a newly trained
# model performs worse than this, the test fails and the
# CI/CD pipeline will stop before promoting a bad model.
# --------------------------------------------------

import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Minimum acceptable scores - if the model falls below any of
# these, something has clearly gone wrong with training
EXPECTED_ACCURACY = 0.40
EXPECTED_PRECISION = 0.40
EXPECTED_RECALL = 0.40
EXPECTED_F1 = 0.40


def test_model_performance_meets_threshold():
    model_name = "yt_chrome_plugin_model"

    # Load the current "staging" model
    model_uri = f"models:/{model_name}@staging"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the same TF-IDF vectorizer used during training
    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    # Load the held-out test data (data the model has never seen)
    test_data = pd.read_csv("data/interim/test_processed.csv")
    test_data = test_data.fillna("")

    # Convert the test comments into TF-IDF numbers
    X_test = vectorizer.transform(test_data["clean_comment"].values).toarray()
    y_test = test_data["category"].values

    # Ask the model to predict on the whole test set
    y_pred = model.predict(X_test)

    # Calculate the standard classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)

    # Check that every metric meets the minimum threshold
    assert accuracy >= EXPECTED_ACCURACY, f"Accuracy too low: {accuracy}"
    assert precision >= EXPECTED_PRECISION, f"Precision too low: {precision}"
    assert recall >= EXPECTED_RECALL, f"Recall too low: {recall}"
    assert f1 >= EXPECTED_F1, f"F1 score too low: {f1}"

    print(f"Performance test passed - accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
