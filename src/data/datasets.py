import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class TextDataset(Dataset):
    """
    A simple dataset that returns raw text and labels.
    """
    def __init__(self, df, text_col='text', label_col='label'):
        # Ensure texts are strings and handle potential NaNs
        self.texts = [str(t) if pd.notna(t) else "" for t in df[text_col].tolist()]
        self.labels = torch.tensor(df[label_col].tolist(), dtype=torch.long)
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class TextDatasetWithPretrainedEmbeddings(Dataset):
    """
    A dataset that computes (or loads cached) sentence embeddings for the text.
    """
    def __init__(self, df, cache_file, text_col='text', label_col='label', model_name='all-MiniLM-L6-v2'):
        self.labels = torch.tensor(df[label_col].tolist(), dtype=torch.long)
        
        # Check if cache exists to avoid recomputing embeddings
        if os.path.exists(cache_file):
            print(f"Loading embeddings from cache: {cache_file}")
            # weights_only=True is recommended for security when loading tensors
            self.embeddings = torch.load(cache_file, weights_only=True)
        else:
            print(f"Computing embeddings and saving to cache: {cache_file}")
            # Ensure texts are strings and handle potential NaNs
            texts = [str(t) if pd.notna(t) else "" for t in df[text_col].tolist()]
            
            # Load the embedding model
            model = SentenceTransformer(model_name)
            
            # Encode texts -> Returns a numpy array
            embeddings_np = model.encode(texts, show_progress_bar=True)
            self.embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
            
            # Ensure cache directory exists, then save
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                
            torch.save(self.embeddings, cache_file)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
