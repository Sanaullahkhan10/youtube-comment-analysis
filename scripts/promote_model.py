# scripts/promote_model.py
# --------------------------------------------------
# This script "promotes" the model that currently has the
# "staging" alias by also giving it the "production" alias.
# This marks it as the official, approved version. It only
# runs after all the tests in the CI/CD pipeline have passed.
# --------------------------------------------------

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def promote_model():
    model_name = "yt_chrome_plugin_model"
    client = mlflow.tracking.MlflowClient()

    # Find out which version currently has the "staging" alias
    staging_version = client.get_model_version_by_alias(model_name, "staging")

    # Give that same version the "production" alias too.
    # If another version already had the "production" alias,
    # MLflow automatically moves the alias off the old version -
    # only one version can hold a given alias at a time.
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=staging_version.version
    )

    print(f"Model '{model_name}' version {staging_version.version} promoted to 'production' alias.")


if __name__ == "__main__":
    promote_model()
