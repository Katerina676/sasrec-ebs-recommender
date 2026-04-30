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

        # нормализация эмбеддингов
        self.emb_norm = nn.LayerNorm(hidden_dim)

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

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, item_indices):
        batch_size, seq_len = item_indices.shape

        # эмбеддинги
        item_emb = self.item_emb(item_indices)
        pos = torch.arange(seq_len, device=item_indices.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)

        x = item_emb + pos_emb
        x = self.emb_norm(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_indices.device), diagonal=1
        ).bool()

        # трансформер
        x = self.transformer(x, mask=causal_mask, is_causal=False)

        # выход
        scores = self.output(x)
        return scores

    def predict_next(self, history_user, top_k=10):
        # предсказания топ 10 следующих треков
        self.eval()
        with torch.no_grad():
            hist = history_user[-self.max_seq_len:]
            padded = [0] * (self.max_seq_len - len(hist)) + hist
            tensor = torch.tensor([padded], dtype=torch.long,
                                    device=next(self.parameters()).device)
            all_scores = self.forward(tensor)
            scores = all_scores[0, -1, :].cpu().numpy()
            rec_ids = np.argsort(scores)[::-1][:top_k]
            rec_scores = scores[rec_ids]
        return rec_ids, rec_scores


def negative_sampling_loss(scores, targets, weights, cnt_item, neg_samples=100):
    # функция потерь с Negative Sampling
    batch_size = scores.shape[0]

    pos_scores = scores[torch.arange(batch_size), targets]
    negative_indices = torch.randint(1, cnt_item, (batch_size, neg_samples),
                                device=scores.device)
    negative_scores = scores.gather(1, negative_indices)

    pos_loss = -torch.nn.functional.logsigmoid(pos_scores)
    neg_loss = -torch.nn.functional.logsigmoid(-negative_scores).sum(dim=1)

    weighted_loss = (pos_loss + neg_loss) * weights
    return weighted_loss.mean()

