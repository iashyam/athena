import torch
import pandas as pd
from src.data.datasets import TextDataset, TextDatasetWithPretrainedEmbeddings
from src.utils.text_preprocessing import build_vocabulary
from src.train.train import train_loop, validate
from src.models.linear_model import LinearSentimentModel, SimpleNN, NNWithPretrainedEmbeddings
from src.models.transformer_model import TransformerSentimentModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split

from src.data.download_data import download_data
from src.utils.evaluation import plot_confusion_matrix, calculate_validation_accuracy
import yaml

# Load parameters
with open("parameter.yaml", "r") as f:
    config = yaml.safe_load(f)

# Sentence Transformer initialization
model = SentenceTransformer(config['embeddings']['model_name'])

def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    rows = len(dataset)
    train_split = int(train_ratio * rows)
    val_split = int(val_ratio * rows)
    test_split = rows - train_split - val_split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_split, val_split, test_split])

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
   
def main():
    # Attempt to download the data if it's missing
    print("Checking and downloading datasets if necessary...")
    download_data()

    train_df = pd.read_csv(config['data']['train_csv'])
    val_df = pd.read_csv(config['data']['val_csv'])
    test_df = pd.read_csv(config['data']['test_csv'])

    # Initialize Datasets with Pretrained Embeddings (Compute once)
    print("Encoding Training Data...")
    train_dataset = TextDatasetWithPretrainedEmbeddings(train_df, cache_file=config['embeddings']['train_cache'], model_name=config['embeddings']['model_name'])
    print("Encoding Validation Data...")
    val_dataset = TextDatasetWithPretrainedEmbeddings(val_df, cache_file=config['embeddings']['val_cache'], model_name=config['embeddings']['model_name'])
    print("Encoding Test Data...")
    test_dataset = TextDatasetWithPretrainedEmbeddings(test_df, cache_file=config['embeddings']['test_cache'], model_name=config['embeddings']['model_name'])

    embedding_dim = train_dataset[0][0].shape[0]
    batch_size = config['training']['batch_size']

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = train_df['label'].nunique()
    
    # Initialize Model
    model = NNWithPretrainedEmbeddings(
        embed_dim=embedding_dim, 
        hidden_dim=config['model']['hidden_dim'], 
        num_classes=num_classes
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=float(config['training']['weight_decay'])
    )

    # Train
    train_loop(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        device = config['training']['device'], 
        num_epochs = config['training']['epochs'],
        run_name = config['training']['run_name'],
        model_name = config['training']['model_name']
    )

    # Evaluation
    print("\nRunning Evaluation...")
    # Class names in alphabetical order (as per cat.codes)
    plot_confusion_matrix(model, test_loader, config['training']['device'], [str(i) for i in range(num_classes)])
    val_acc = calculate_validation_accuracy(model, val_loader, config['training']['device'])
    with open("accuracy.log", 'a') as f:
        f.write(f'{val_acc}\n')

if __name__ == "__main__":
    main()

