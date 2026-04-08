import torch
import torch.nn as nn


class CharCNN(nn.Module):
    """Character-level CNN encoder.

    Each word is represented by max-pooling over CNN filters applied to
    its character embeddings.
    """

    def __init__(
        self,
        n_chars: int,
        char_embed_dim: int = 50,
        out_dim: int = 50,
        kernel_sizes: tuple = (3, 5),
        pad_idx: int = 0,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(n_chars, char_embed_dim, padding_idx=pad_idx)

        # Each kernel produces out_dim // len(kernel_sizes) channels;
        # the last kernel absorbs any remainder so total = out_dim.
        n_kernels = len(kernel_sizes)
        base = out_dim // n_kernels
        channels = [base] * n_kernels
        channels[-1] += out_dim - sum(channels)  # handle remainder

        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, ch, k, padding=k // 2)
            for k, ch in zip(kernel_sizes, channels)
        ])
        self.out_dim = out_dim

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch * seq_len, max_word_len) int64

        Returns:
            (batch * seq_len, out_dim) float
        """
        # (BL, W) → (BL, char_embed_dim, W)
        x = self.char_embedding(char_ids).transpose(1, 2)

        # Apply each conv and max-pool over the time dimension
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))           # (BL, ch, W')
            h = h.max(dim=-1).values          # (BL, ch)
            pooled.append(h)

        return torch.cat(pooled, dim=-1)       # (BL, out_dim)
