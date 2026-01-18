import torch
import pandas as pd
from src.data.datasets import TextDataset, TextDatasetWithPretrainedEmbeddings
from src.utils.text_preprocessing import build_vocabulary
from src.train.train import train_loop, validate
from src.models.linear_model import LinearSentimentModel, SimpleNN, NNWithPretrainedEmbeddings
from src.models.transformer_model import TransformerSentimentModel

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    train_split = int(train_ratio * rows)
    val_split = int(val_ratio * rows)
    test_split = rows - train_split - val_split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_split, val_split, test_split])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
   

BATCH_SIZE = 64
EPOCHS = 20

train_df = pd.read_csv("data/processed/sentiment train_df.csv")
val_df = pd.read_csv("data/processed/sentiment val_df.csv")
test_df = pd.read_csv("data/processed/sentiment test_df.csv")

# Initialize Datasets with Pretrained Embeddings (Compute once)
print("Encoding Training Data...")
train_dataset = TextDatasetWithPretrainedEmbeddings(train_df, cache_file="data/processed/train_embeddings.pt")
print("Encoding Validation Data...")
val_dataset = TextDatasetWithPretrainedEmbeddings(val_df, cache_file="data/processed/val_embeddings.pt")
print("Encoding Test Data...")
test_dataset = TextDatasetWithPretrainedEmbeddings(test_df, cache_file="data/processed/test_embeddings.pt")

embedding_dim = train_dataset[0][0].shape[0]

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = train_df['label'].nunique()
# Initialize Model
model = NNWithPretrainedEmbeddings(embed_dim=embedding_dim, hidden_dim=128, num_classes=num_classes)
# model = TransformerSentimentModel(embed_dim=embedding_dim, hidden_dim=128, n_head=8, num_classes=num_classes)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5) # Added weight decay

# Train
train_loop(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    criterion = criterion,
    optimizer = optimizer,
    device = "cpu", 
    num_epochs = EPOCHS,
    run_name = "Augmented_Data_Pretrained_Embeddings_Run_small_epochs",
    model_name = "NN_With_Augmentation_ONNX"
)

# Evaluation
print("\nRunning Evaluation...")
from src.utils.evaluation import plot_confusion_matrix
# Class names in alphabetical order (as per cat.codes)
# class_names = ['Anger', 'Calm', 'Curiosity', 'Fear', 'Hope', 'Joy', 'Love', 'Sadness', 'Shame', 'Surprise']
plot_confusion_matrix(model, test_loader, "cpu",[str(i) for i in range(num_classes)])

