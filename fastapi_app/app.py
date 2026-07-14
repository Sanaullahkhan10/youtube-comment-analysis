# fastapi_app/app.py
# --------------------------------------------------
# This is the backend API that serves our sentiment
# analysis model to the outside world (for example, to
# a Chrome extension, or to anyone testing with Postman).
#
# Endpoints:
# - /predict                  : takes comments, returns sentiment
# - /predict_with_timestamps  : same, but keeps timestamps too
# - /generate_chart           : returns a pie chart image (PNG)
# - /generate_wordcloud       : returns a word cloud image (PNG)
# - /generate_trend_graph     : returns a sentiment-over-time line graph (PNG)
# --------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend since this runs on a server, not a desktop

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import re
import io
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
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
    # Load the trained model and vectorizer directly from local pickle files.
    #
    # Note: we are NOT loading the model from the MLflow registry here.
    # MLflow's local (file-based) artifact storage bakes in an ABSOLUTE
    # file path tied to this specific computer (e.g. C:\PROJECTS\...).
    # That path does not exist inside a Docker container, which has its
    # own separate filesystem - so registry loading would break as soon
    # as this app runs inside a container.
    #
    # Once we have a remote MLflow server running on AWS (a later step),
    # we will switch back to loading via the registry, because a network
    # URL (like http://our-server:5000) works identically everywhere,
    # unlike a local file path.
    with open("lgbm_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer

# Load the model and vectorizer ONCE, when the app starts.
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
        preprocessed_comments = []
        for comment in comments:
            preprocessed_comments.append(preprocess_comment(comment))

        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()
        predictions = model.predict(transformed_comments)

        prediction_list = []
        for prediction in predictions:
            prediction_list.append(str(prediction))

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

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
        comments = []
        timestamps = []
        for item in comments_data:
            comments.append(item["text"])
            timestamps.append(item["timestamp"])

        preprocessed_comments = []
        for comment in comments:
            preprocessed_comments.append(preprocess_comment(comment))

        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()
        predictions = model.predict(transformed_comments)

        prediction_list = []
        for prediction in predictions:
            prediction_list.append(str(prediction))

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    response = []
    for comment, sentiment, timestamp in zip(comments, prediction_list, timestamps):
        response.append({"comment": comment, "sentiment": sentiment, "timestamp": timestamp})

    return response

@app.post("/generate_chart")
def generate_chart(payload: dict):
    sentiment_counts = payload.get("sentiment_counts")

    if not sentiment_counts:
        return {"error": "No sentiment counts provided"}

    try:
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0))
        ]

        if sum(sizes) == 0:
            return {"error": "Sentiment counts sum to zero"}

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"}
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return {"error": f"Chart generation failed: {str(e)}"}

@app.post("/generate_wordcloud")
def generate_wordcloud(payload: dict):
    comments = payload.get("comments")

    if not comments:
        return {"error": "No comments provided"}

    try:
        preprocessed_comments = []
        for comment in comments:
            preprocessed_comments.append(preprocess_comment(comment))

        text = " ".join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return {"error": f"Word cloud generation failed: {str(e)}"}

@app.post("/generate_trend_graph")
def generate_trend_graph(payload: dict):
    sentiment_data = payload.get("sentiment_data")

    if not sentiment_data:
        return {"error": "No sentiment data provided"}

    try:
        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        monthly_counts = df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return {"error": f"Trend graph generation failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
