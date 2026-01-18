import torch
import onnx
import mlflow.onnx
from mlflow.models import infer_signature

def export_to_onnx(model, input_sample, model_name="model", artifact_path="model.onnx"):
    """
    Exports a PyTorch model to ONNX format and logs it to MLflow.
    """
    
    input_sample = input_sample.to("cpu")
    model = model.to("cpu")
    model.eval()

    torch.onnx.export(
        model,
        input_sample,
        artifact_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    name = model_name.split(".")[-1] 
    onnx_model = onnx.load(artifact_path)
    # Infer signature
    with torch.no_grad():
        output = model(input_sample)
    signature = infer_signature(input_sample.numpy(), output.numpy())

    mlflow.onnx.log_model(
        onnx_model,
        name=name,
        registered_model_name=model_name,
        input_example=input_sample.numpy(),
        signature=signature,
    )
    
    print(f"Model exported to {artifact_path} and logged to MLflow as {model_name}")
