# scripts/test_model_signature.py
# --------------------------------------------------
# This test checks that our model can actually make a
# prediction on a realistic input, and that the output
# shape matches what we expect. This catches problems
# like a mismatched vectorizer or a broken preprocessing
# step before they ever reach production.
# --------------------------------------------------

import pickle
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def test_model_predicts_on_sample_input():
    model_name = "yt_chrome_plugin_model"

    # Load the current "staging" model
    model_uri = f"models:/{model_name}@staging"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the same TF-IDF vectorizer used during training
    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    # Create one dummy comment and convert it into TF-IDF numbers,
    # exactly the way real incoming comments are processed
    sample_text = "this is a great video"
    sample_input = vectorizer.transform([sample_text]).toarray()

    # Ask the model to predict on this single sample
    prediction = model.predict(sample_input)

    # The number of predictions returned should match the number
    # of rows we sent in (1 row in, 1 prediction out)
    assert len(prediction) == sample_input.shape[0], "Output row count does not match input row count"

    print(f"Model processed a sample input correctly. Prediction: {prediction}")
