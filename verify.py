from model.singleton_model_loader import SingletonModelLoader

# Test loading the model and predicting
model_loader = SingletonModelLoader("models/word_prediction_model.pkl")
print("Model loaded successfully.")

# Test prediction
try:
    test_input = "example input"
    prediction = model_loader.predict(test_input)
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Error during prediction: {e}")
