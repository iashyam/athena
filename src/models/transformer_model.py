from torch import nn
import torch


class TransformerSentimentModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, n_head: int, num_classes: int):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)