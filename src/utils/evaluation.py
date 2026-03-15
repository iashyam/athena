import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import mlflow

def plot_confusion_matrix(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save to file
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Log to MLflow if active
    if mlflow.active_run():
        mlflow.log_artifact("confusion_matrix.png")
        
    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def calculate_validation_accuracy(model, dataloader, device):
    """
    Calculates the validation accuracy using a PyTorch model and DataLoader.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    print(f"Final Validation Accuracy: {accuracy:.4f}")

    if mlflow.active_run():
        mlflow.log_metric("final_validation_accuracy", accuracy)
    elif mlflow.last_active_run():
        with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
            mlflow.log_metric("final_validation_accuracy", accuracy)

    return accuracy

