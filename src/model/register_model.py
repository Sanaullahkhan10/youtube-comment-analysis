# src/model/register_model.py
# --------------------------------------------------
# This script takes the model we already logged to MLflow
# (in the evaluation step) and formally registers it in the
# MLflow Model Registry under a fixed name, with the status
# "Staging" - meaning "this is a candidate, not live yet".
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

    # Move this specific version into the "Staging" stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    return model_version


def main():
    # Step 1: Load the run_id saved by the evaluation script
    model_info = load_model_info("experiment_info.json")

    # Step 2: Register the model under a fixed name
    model_name = "yt_chrome_plugin_model"
    model_version = register_model(model_name, model_info)

    print(f"Model registered successfully! Name: {model_name}, Version: {model_version.version}, Stage: Staging")


if __name__ == "__main__":
    main()
