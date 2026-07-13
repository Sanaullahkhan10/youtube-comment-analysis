# src/data/data_preprocessing.py
# --------------------------------------------------
# This script cleans up the raw comment text so a
# machine learning model can understand it better.
# It takes data/raw/train.csv and data/raw/test.csv,
# cleans the text in each row, and saves the result
# into data/interim/train_processed.csv and
# data/interim/test_processed.csv
# --------------------------------------------------

import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download the NLTK data files needed for stopwords and lemmatization
# (this only downloads once, then it stays cached on your computer)
nltk.download("stopwords")
nltk.download("wordnet")

# Create one lemmatizer object we can reuse for every comment
lemmatizer = WordNetLemmatizer()

# Get the list of common English stopwords (words like "the", "is", "a")
# But we REMOVE a few important words from this list, because words like
# "not" and "but" completely change the meaning of a sentence, and we
# do not want to lose them during sentiment analysis.
stop_words = set(stopwords.words("english"))
important_words = ["not", "but", "however", "no", "yet"]
for word in important_words:
    stop_words.discard(word)


def clean_comment(comment):
    # Turn all text to lowercase, so "Good" and "good" are treated the same
    comment = comment.lower()

    # Remove extra spaces from the start and end of the comment
    comment = comment.strip()

    # Replace newline characters with a single space
    comment = re.sub(r"\n", " ", comment)

    # Remove any character that is not a letter, number, space, or basic punctuation
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    # Split the comment into a list of separate words
    words = comment.split()

    # Build a new list of words, keeping only the ones that are NOT stopwords
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    # Turn each remaining word into its root form (lemmatization)
    # Example: "running" becomes "run", "better" becomes "good"
    lemmatized_words = []
    for word in filtered_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    # Join the words back into a single cleaned sentence
    cleaned_comment = " ".join(lemmatized_words)

    return cleaned_comment


def normalize_text(df):
    # Apply the clean_comment function to every row in the clean_comment column
    df["clean_comment"] = df["clean_comment"].apply(clean_comment)
    return df


def save_data(train_data, test_data, output_folder):
    # Create the data/interim folder if it does not already exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the cleaned train and test data as separate CSV files
    train_data.to_csv(os.path.join(output_folder, "train_processed.csv"), index=False)
    test_data.to_csv(os.path.join(output_folder, "test_processed.csv"), index=False)


def main():
    # Step 1: Load the raw train and test data
    train_data = pd.read_csv("data/raw/train.csv")
    test_data = pd.read_csv("data/raw/test.csv")

    # Step 2: Clean the text in both datasets
    train_processed = normalize_text(train_data)
    test_processed = normalize_text(test_data)

    # Step 3: Save the cleaned data into data/interim
    save_data(train_processed, test_processed, "data/interim")

    print("Data preprocessing complete! train_processed.csv and test_processed.csv saved in data/interim folder.")


if __name__ == "__main__":
    main()
