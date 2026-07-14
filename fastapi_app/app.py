# fastapi_app/app.py
# --------------------------------------------------
# This is the backend API that serves our sentiment
# analysis model to the outside world (for example, to
# a Chrome extension, or to anyone testing with Postman).
#
# It exposes two endpoints for now:
# - /predict                  : takes comments, returns sentiment
# - /predict_with_timestamps  : same, but keeps timestamps too
# --------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import re
import pickle
import mlflow
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

# CORS lets a webpage (like a Chrome extension) running on a different
# origin/domain send requests to this API without the browser blocking it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create one lemmatizer we can reuse for every comment
lemmatizer = WordNetLemmatizer()

# Get the list of common English stopwords, but keep a few important
# ones that change the meaning of a sentence (same logic as data_preprocessing.py)
stop_words = set(stopwords.words("english"))
important_words = ["not", "but", "however", "no", "yet"]
for word in important_words:
    stop_words.discard(word)


def preprocess_comment(comment):
    # This is the exact same cleaning logic used in data_preprocessing.py.
    # It is critical that we clean incoming comments the SAME way we
    # cleaned the training data, otherwise the model will get confused.
    comment = comment.lower()
    comment = comment.strip()
    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    words = comment.split()

    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    lemmatized_words = []
    for word in filtered_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))

    cleaned_comment = " ".join(lemmatized_words)
    return cleaned_comment


def load_model_and_vectorizer():
    # Point MLflow to the same local database used during training
    # (later, in the AWS phase, this will point to a remote server)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Load the model version that currently has the "staging" alias.
    # If a new model gets promoted to this alias later, this app will
    # automatically use the new one next time it restarts - no code change needed.
    model_uri = "models:/yt_chrome_plugin_model@staging"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the TF-IDF vectorizer from the local pickle file
    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


# Load the model and vectorizer ONCE, when the app starts.
# (Loading them inside every single request would be very slow.)
model, vectorizer = load_model_and_vectorizer()


@app.get("/")
def home():
    return "Welcome to the YouTube comment sentiment analysis API"


@app.post("/predict")
def predict(payload: dict):
    comments = payload.get("comments")

    if not comments:
        return {"error": "No comments provided"}

    try:
        # Clean every comment the same way the training data was cleaned
        preprocessed_comments = []
        for comment in comments:
            preprocessed_comments.append(preprocess_comment(comment))

        # Convert the cleaned comments into TF-IDF numbers using the SAME
        # vectorizer that was fitted during training
        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()

        # Ask the model to predict a sentiment for each comment
        predictions = model.predict(transformed_comments)

        # Convert predictions into plain strings (so JSON can send them back)
        prediction_list = []
        for prediction in predictions:
            prediction_list.append(str(prediction))

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Build the final response: each comment paired with its predicted sentiment
    response = []
    for comment, sentiment in zip(comments, prediction_list):
        response.append({"comment": comment, "sentiment": sentiment})

    return response


@app.post("/predict_with_timestamps")
def predict_with_timestamps(payload: dict):
    comments_data = payload.get("comments")

    if not comments_data:
        return {"error": "No comments provided"}

    try:
        # Pull out the comment text and the timestamp from each item
        comments = []
        timestamps = []
        for item in comments_data:
            comments.append(item["text"])
            timestamps.append(item["timestamp"])

        # Clean every comment the same way the training data was cleaned
        preprocessed_comments = []
        for comment in comments:
            preprocessed_comments.append(preprocess_comment(comment))

        # Convert the cleaned comments into TF-IDF numbers
        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()

        # Ask the model to predict a sentiment for each comment
        predictions = model.predict(transformed_comments)

        # Convert predictions into plain strings
        prediction_list = []
        for prediction in predictions:
            prediction_list.append(str(prediction))

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Build the final response: comment + sentiment + timestamp, together
    response = []
    for comment, sentiment, timestamp in zip(comments, prediction_list, timestamps):
        response.append({"comment": comment, "sentiment": sentiment, "timestamp": timestamp})

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
