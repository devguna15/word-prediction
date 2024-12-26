import joblib
import threading

class SingletonModelLoader:
    """
    Singleton class to ensure the model is loaded only once into memory.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_path):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SingletonModelLoader, cls).__new__(cls)
                cls._instance._initialize(model_path)
        return cls._instance

    def _initialize(self, model_path):
        """Private method to initialize the model loader."""
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path):
        """Load the trained model from the specified path."""
        try:
            print(f"Attempting to load model from: {model_path}")
            model = joblib.load(model_path)  # Use joblib for loading
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, input_data):
        """Perform inference using the loaded model."""
        if self.model is None:
            raise ValueError("Model is not loaded.")
        try:
            return self.model.predict([input_data])
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
