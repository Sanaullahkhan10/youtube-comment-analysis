# src/model/register_model.py
# --------------------------------------------------
# This script takes the model we already logged to MLflow
# (in the evaluation step) and formally registers it in the
# MLflow Model Registry under a fixed name, then assigns it
# the "staging" alias - meaning "this is a candidate, not
# live in production yet".
#
# Note: MLflow used to use fixed "stages" (Staging/Production/
# Archived) for this, but that system is deprecated. Aliases
# are the modern replacement - a version can have any named
# alias, and promoting a model is just re-pointing an alias
# to a different version.
# --------------------------------------------------

import json
import mlflow

# Point to the same local SQLite database used in evaluation
# (later this will point to a remote AWS-hosted MLflow server)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def load_model_info(file_path):
    # Read the run_id and model_path that we saved during evaluation
    with open(file_path, "r") as file:
        model_info = json.load(file)
    return model_info


def register_model(model_name, model_info):
    # Build the special MLflow URI that points to this exact model,
    # inside this exact run
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    # Register the model under a fixed name in the Model Registry
    # (if this name does not exist yet, MLflow creates it as version 1;
    # if it already exists, MLflow adds a new version automatically)
    model_version = mlflow.register_model(model_uri, model_name)

    # Assign the "staging" alias to this version, marking it as a
    # candidate model that still needs to pass tests before going live
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=model_version.version
    )

    return model_version


def main():
    # Step 1: Load the run_id saved by the evaluation script
    model_info = load_model_info("experiment_info.json")

    # Step 2: Register the model under a fixed name
    model_name = "yt_chrome_plugin_model"
    model_version = register_model(model_name, model_info)

    print(f"Model registered successfully! Name: {model_name}, Version: {model_version.version}, Alias: staging")


if __name__ == "__main__":
    main()
