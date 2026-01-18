
from src.utils.export import export_to_onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import mlflow
import mlflow.pytorch
import mlflow.onnx
import time
from typing import Tuple
import os

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/shyam10kwd@gmail.com/athena/sentiment-analysis")

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: Optimizer, 
    device: torch.device
) -> Tuple[float, float]:
    
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if isinstance(criterion, nn.CrossEntropyLoss):
             labels = labels.long()

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
        
        preds = torch.argmax(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()

def validate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                 labels = labels.long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    run_name: str = "Training Run",
    model_name: str = "SentimentModel"
):
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        print(f"Starting run: {run_name}")
        
        # Log parameters
        mlflow.log_param("device", str(device))
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("loss_function", criterion.__class__.__name__)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", num_epochs)
        
        start_time = time.time()
        model = model.to(device)

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print("-" * 30)
            
            # Log metrics history
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f}s")
        
        # Log final summary metrics
        mlflow.log_metric("total_training_time", total_time)
        mlflow.log_metric("final_val_acc", val_acc)
        
        # Log the trained model as onnx for inference
        x, _ = next(iter(train_loader))
        
        model_name = f"workspace.default.{model_name}"
        export_to_onnx(model, x, model_name=model_name)
        
    return model
