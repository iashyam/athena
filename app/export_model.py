import mlflow.pytorch
import torch 
import onnxruntime as rt
from sentence_transformers import SentenceTransformer
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "workspace.default.NN_With_Augmentation_ONNX"
model_uri = f"models:/{model_name}/latest"
try:
    model = rt.InferenceSession(model_uri)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def predict(sentence: str):
    embeddings = embed_model.encode([sentence])
    output = model.run(None, {"input": embeddings})
    classes = ["Negative", "Neutral", "Positive"]
    return classes[output[0][0].argmax()]

while True:
    sentence = input("Enter a sentence: ")
    if sentence == "exit":
        break
    print(predict(sentence))