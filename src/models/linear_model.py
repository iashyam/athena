import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearSentimentModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linearModel = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.mean(dim=1)
        return self.linearModel(x)

class SimpleNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_Len]
        embedded = self.embedding(x) # [Batch, Seq_Len, Embed_Dim]
        
        # Global Average Pooling
        pooled = embedded.mean(dim=1) # [Batch, Embed_Dim]
        
        return self.network(pooled)

class NNWithPretrainedEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        return self.network(x)

