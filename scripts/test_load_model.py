# scripts/test_load_model.py
# --------------------------------------------------
# This test checks that our trained model can actually be
# loaded from the MLflow registry using the "staging" alias.
# If this test fails, it usually means either the model was
# never registered, or the registry/alias name is wrong.
# --------------------------------------------------

import mlflow

# Point MLflow to the same local SQLite database used during
# training and registration
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def test_load_staging_model():
    model_name = "yt_chrome_plugin_model"

    # Build the special MLflow URI that points to whichever model
    # version currently has the "staging" alias
    model_uri = f"models:/{model_name}@staging"

    # Try to load the model - if this line fails, the test fails
    model = mlflow.pyfunc.load_model(model_uri)

    # Make sure something actually got loaded (not None)
    assert model is not None, "Model failed to load from the registry"

    print(f"Model '{model_name}' loaded successfully from the 'staging' alias.")
