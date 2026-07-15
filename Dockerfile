FROM python:3.12-slim

WORKDIR /app

# lightgbm needs the libgomp1 system library (OpenMP runtime) to work correctly
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install them - this is a Docker best practice
# for layer caching: this expensive step only re-runs when requirements.txt
# actually changes, not every time the application code changes.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Download the NLTK data needed for text preprocessing
RUN python -m nltk.downloader stopwords wordnet

# Copy only the API code into the container
COPY fastapi_app/ /app/

# Copy the trained model and vectorizer, needed to make predictions
COPY lgbm_model.pkl /app/lgbm_model.pkl
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

EXPOSE 5000
CMD ["python", "app.py"]
