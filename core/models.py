import os
import joblib
from tensorflow.keras.models import load_model

class Bundle:
    def __init__(self, model_path: str, scaler_path: str, encoder_path: str):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
        self.encoder = joblib.load(encoder_path) if encoder_path and os.path.exists(encoder_path) else None
        self.input_shape = self.model.input_shape
