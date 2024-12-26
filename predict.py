from model.singleton_model_loader import SingletonModelLoader
import logging

logger = logging.getLogger(__name__)

class WordPredictor:
    """
    A class for making predictions using the trained word prediction model.
    """

    def __init__(self, model_path: str):
        """
        Initialize the WordPredictor with the SingletonModelLoader.
        Args:
            model_path (str): Path to the trained model file.
        """
        self.model_loader = SingletonModelLoader(model_path)

    def predict(self, input_text: str):
        """
        Predict the next word or class for the given input text.
        Args:
            input_text (str): The input text for prediction.
        Returns:
            dict: A dictionary with the prediction and associated probability.
        """
        if not input_text.strip():
            raise ValueError("Input text cannot be empty.")

        try:
            # Perform prediction
            prediction = self.model_loader.predict(input_text)

            # Assuming the model also provides probability scores
            probability = getattr(self.model_loader.model, "predict_proba", None)
            if probability:
                prob_scores = probability([input_text])[0]
                return {"prediction": prediction[0], "probabilities": prob_scores.tolist()}
            else:
                logger.warning("Model does not support probability predictions.")
                return {"prediction": prediction[0]}
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
