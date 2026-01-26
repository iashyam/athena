from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.inference import SentimentModel
import mlflow
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()

# Set Databricks credentials for MLflow
os.environ["DATABRICKS_HOST"] = os.environ.get("MLFLOW_URI")
os.environ["DATABRICKS_TOKEN"] = os.environ.get("databricks_token")

# Define global model variable
model_wrapper = {}


model_name = "workspace.default.NN_With_Augmentation_ONNX"
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

try:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise Exception(f"No versions found for model {model_name}")
    latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0].version
    print(f"Using latest model version: {latest_version}")
    model_uri = f"models:/{model_name}/{latest_version}"
except Exception as e:
    print(f"Error finding latest model version: {e}")
    raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing models...")
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    model_path = os.path.join(local_path, "model.onnx")
    print(f"Model downloaded to: {model_path}")
    model_wrapper["model"] = SentimentModel(model_path)
    yield
    model_wrapper.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentRequest(BaseModel):
    sentence: str

@app.get("/")
def read_root():
    return FileResponse('app/static/index.html')

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: SentimentRequest):
    result = model_wrapper["model"].predict(request.sentence)
    return {"prediction": result}
