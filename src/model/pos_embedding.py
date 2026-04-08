import torch.nn as nn


class POSEmbedding(nn.Module):
    """Learnable POS-tag embedding table."""

    def __init__(self, n_pos: int, pos_dim: int = 50, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(n_pos, pos_dim, padding_idx=pad_idx)

    def forward(self, upos_ids):
        # upos_ids: (batch, seq_len) → (batch, seq_len, pos_dim)
        return self.embedding(upos_ids)
