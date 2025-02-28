# Fake News Detector

## Overview
This project is a machine learning-powered web application that detects whether a given news article is real or fake. Users can input a news article URL or paste the article text, and the model will predict its authenticity.

## Features
- Accepts both article URLs and raw text as input.
- Uses a Logistic Regression model with an IDF vectorizer for classification.
- A simple and user-friendly web interface.
- Deployed as a Flask web application.

## Tech Stack
- **Backend:** Flask, Gunicorn
- **Machine Learning:** Scikit-learn (Logistic Regression with TF-IDF Vectorizer)
- **Frontend:** HTML, CSS, JavaScript (jQuery)
- **Deployment:** Render

## Data & Model
The model was trained on a dataset of labeled news articles, using TF-IDF vectorization for text representation. Despite good performance on train-test data, real-world articles posed challenges, highlighting the limitations of the dataset and model generalization.

## Installation & Setup
### Prerequisites
- Python 3.8+
- pip
- virtualenv (optional)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application locally:
   ```bash
   python app.py
   ```

5. Access the web app at `http://127.0.0.1:5000`

## Usage
1. Enter a news article URL or paste article text into the input box.
2. Click the **Predict** button.
3. View the prediction result (Real or Fake).

## Deployment
The application is deployed on Render. To redeploy:
- Ensure `gunicorn` is installed (`pip install gunicorn`).
- Add `requirements.txt` and `Procfile` for deployment.
- Set environment variables if needed.
- Deploy using Render's automated deployment pipeline.

## Challenges & Learnings
- **Model Generalization:** The Logistic Regression model performed well on the training dataset but struggled with newer articles, suggesting a need for continual retraining and dataset expansion.
- **Text Extraction from URLs:** Some websites blocked automated content extraction, leading to errors for certain URLs.
- **Alternative Approaches:** Future improvements could explore deep learning models (e.g., BERT) or ensembles for better robustness.

## Future Improvements
- **Model Enhancement:** Experiment with transformer-based models (e.g., BERT, RoBERTa).
- **Data Augmentation:** Collect and label more recent articles for training.
- **Explainability:** Implement interpretable AI techniques (e.g., SHAP, LIME) for model insights.
- **UI Improvements:** Enhance user experience and error handling in the frontend.

## Conclusion
This project showcases the power of NLP and machine learning in detecting fake news. While it highlights the potential of ML-based solutions, it also underscores the challenges of real-world deployment and model generalization. The project is a great learning experience, and there is scope for further improvements in the future.

