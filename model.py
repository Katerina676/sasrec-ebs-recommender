import torch
import torch.nn as nn
import numpy as np


class SASRecModel(nn.Module):
    def __init__(self, cnt_item, max_seq_len=50, hidden_dim=64,
                 num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()

        self.max_seq_len = max_seq_len

        # эмбеддинги
        self.item_emb = nn.Embedding(cnt_item + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # трансформер
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward= 4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # нормализация и выход
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, cnt_item + 1)

        self._init_weights()