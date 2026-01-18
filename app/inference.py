import onnxruntime as rt
from sentence_transformers import SentenceTransformer
import numpy as np

class SentimentModel:
    def __init__(self, model_path="model.onnx"):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.session = rt.InferenceSession(model_path)
        self.classes = ["Negative", "Neutral", "Positive"]

    def predict(self, sentence: str):
        embeddings = self.embed_model.encode([sentence])
        output = self.session.run(None, {"input": embeddings})
        result_idx = output[0][0].argmax()
        return self.classes[result_idx]

sentiment_model = None
