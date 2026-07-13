# src/data/data_ingestion.py
# --------------------------------------------------
# This script does 4 things:
# 1. Load settings from params.yaml
# 2. Download the raw comments data from the internet
# 3. Clean the data (remove empty/duplicate rows)
# 4. Split data into train/test and save it in data/raw
# --------------------------------------------------

import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split


def load_params(params_path):
    # Open the params.yaml file
    # yaml.safe_load converts it into a python dictionary (a small table)
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    return params


def load_data(data_url):
    # pandas read_csv can directly read a CSV file from an internet URL
    # and turn it into a DataFrame (a table-like structure)
    df = pd.read_csv(data_url)
    return df


def clean_data(df):
    # Remove rows where any column has a missing (empty) value
    df = df.dropna()

    # Remove rows that are exact duplicates of another row
    df = df.drop_duplicates()

    # Remove rows where the clean_comment column is just empty spaces
    df = df[df["clean_comment"].str.strip() != ""]

    return df


def save_data(train_data, test_data, output_folder):
    # Create the data/raw folder if it does not already exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the train and test data as separate CSV files
    train_data.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    test_data.to_csv(os.path.join(output_folder, "test.csv"), index=False)


def main():
    # Step 1: Load settings from params.yaml
    params = load_params("params.yaml")
    test_size = params["data_ingestion"]["test_size"]

    # Step 2: Download the original raw data from the internet
    data_url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    df = load_data(data_url)

    # Step 3: Clean the data
    clean_df = clean_data(df)

    # Step 4: Split the data into train and test sets
    train_data, test_data = train_test_split(clean_df, test_size=test_size, random_state=42)

    # Step 5: Save both files into the data/raw folder
    save_data(train_data, test_data, "data/raw")

    print("Data ingestion complete! train.csv and test.csv saved in data/raw folder.")


if __name__ == "__main__":
    main()
