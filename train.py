import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_word_prediction_model(data_path: str, model_path: str, ngram_range=(1, 3)):
    """
    Train a word prediction model and save it to a file, using an n-gram range.
    
    Args:
        data_path (str): Path to the dataset (CSV file).
        model_path (str): Path to save the trained model (Pickle file).
        ngram_range (tuple): The n-gram range for the CountVectorizer (e.g., (1, 3)).
    """
    # Load dataset
    print("Loading dataset...")
    data = pd.read_csv(data_path)

    # Ensure required columns exist
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    X, y = data['text'], data['label']

    # Split into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with text vectorization and Naive Bayes classification
    print(f"Creating pipeline with n-gram range {ngram_range}...")
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=ngram_range)),  # Dynamic n-gram range
        ('classifier', MultinomialNB())
    ])

    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the model
    print(f"Saving model to {model_path}...")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    # Update these paths as needed
    DATA_PATH = "extended_dataset.csv"
    MODEL_PATH = "models/word_prediction_model.pkl"
    
    # Define the desired n-gram range (e.g., unigrams, bigrams, and trigrams)
    NGRAM_RANGE = (1, 3)

    train_word_prediction_model(DATA_PATH, MODEL_PATH, ngram_range=NGRAM_RANGE)

