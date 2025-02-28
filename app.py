import os
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article

# Initialize Flask App
app = Flask(__name__)

# Load the saved model and TF-IDF vectorizer
model_path = Path('src/model/fake_news_detector.mod')

with open(model_path, "rb") as f:
    model = joblib.load(f)

def get_vectorizer():
    fake_df = pd.read_csv("src/data/Fake.csv")
    real_df = pd.read_csv("src/data/True.csv")

    df = pd.concat([fake_df, real_df], axis=0)
    df = df.drop_duplicates().reset_index(drop=True)

    df['text_length'] = df['text'].str.strip().str.len().astype(int)
    df = df[(df["text_length"] > 50) & (df["text_length"] < 5000)].reset_index(drop=True)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    # Fit and transform the text data
    tfidf_vectorizer.fit(df["text"])

    return tfidf_vectorizer

vectorizer = get_vectorizer()

# Initialize LIME Explainer
explainer = LimeTextExplainer(class_names=["Real", "Fake"])

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(e)
        return None

# Model inference function
def predict_text(text):
    text_vector = vectorizer.transform([text]).toarray()  # Convert text to TF-IDF feature vector
    prediction = model.predict(text_vector)  # Get single prediction
    return int(prediction)  # Convert to integer (0 = Real, 1 = Fake)

# Function to explain prediction using LIME
def explain_prediction(text):
    def predictor(texts):
        text_vectors = vectorizer.transform(texts).toarray()  # Convert batch of texts to TF-IDF
        preds = model.predict_proba(text_vectors)  # Get class probabilities
        return preds  

    explanation = explainer.explain_instance(text, predictor, num_features=10, num_samples=1000)
    return explanation.as_list()

# API Route: Predict Fake or Real
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        url = data.get("url", "")

        if url:  # If a URL is provided, extract article text
            text = extract_text_from_url(url)
            if not text:
                print("Failed to extract text from URL")
                return jsonify({"error": "Failed to extract text from URL"}), 200

        text_vector = vectorizer.transform([text]).toarray()
        prediction = model.predict(text_vector)

        if prediction == 1:
            return jsonify({"prediction": "Fake", "text": text}), 200
        else:
            return jsonify({"prediction": "Real", "text": text}), 200

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 200

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
