import joblib

model_path = "models/word_prediction_model.pkl"
model = joblib.load(model_path)

input_text = "The quick brown fox"
prediction = model.predict([input_text])
print(f"Prediction: {prediction}")
