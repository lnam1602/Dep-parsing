import torch
import torch.nn as nn
from .encoder import Encoder
from .biaffine import Biaffine


class DependencyParser(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", hidden_dim=768, n_labels=50, dropout=0.33):
        super().__init__()

        self.encoder = Encoder(model_name=model_name)
        encoder_dim = self.encoder.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.root = nn.Parameter(torch.zeros(1, 1, encoder_dim))

        self.arc_mlp_head = nn.Linear(encoder_dim, hidden_dim)
        self.arc_mlp_dep = nn.Linear(encoder_dim, hidden_dim)
        self.rel_mlp_head = nn.Linear(encoder_dim, hidden_dim)
        self.rel_mlp_dep = nn.Linear(encoder_dim, hidden_dim)
        self.activation = nn.ReLU()

        self.arc_biaffine = Biaffine(hidden_dim, 1)
        self.rel_biaffine = Biaffine(hidden_dim, n_labels)

    def _gather_word_representations(self, token_representations, word_starts):
        batch_size, _, hidden_size = token_representations.size()
        safe_word_starts = word_starts.clamp(min=0)
        gather_index = safe_word_starts.unsqueeze(-1).expand(batch_size, safe_word_starts.size(1), hidden_size)
        word_representations = token_representations.gather(1, gather_index)
        invalid_positions = word_starts.eq(-1).unsqueeze(-1)
        return word_representations.masked_fill(invalid_positions, 0.0)

    def forward(self, input_ids, attention_mask, word_starts, word_mask):
        x = self.encoder(input_ids, attention_mask)
        x = self._gather_word_representations(x, word_starts)

        root = self.root.expand(x.size(0), -1, -1)
        x = torch.cat((root, x), dim=1)
        token_mask = torch.cat((torch.ones(x.size(0), 1, device=x.device, dtype=torch.bool), word_mask), dim=1)

        arc_head = self.dropout(self.activation(self.arc_mlp_head(x)))
        arc_dep = self.dropout(self.activation(self.arc_mlp_dep(x)))
        rel_head = self.dropout(self.activation(self.rel_mlp_head(x)))
        rel_dep = self.dropout(self.activation(self.rel_mlp_dep(x)))

        arc_scores = self.arc_biaffine(arc_dep, arc_head).squeeze(-1)
        rel_scores = self.rel_biaffine(rel_dep, rel_head)

        head_mask = token_mask.unsqueeze(1)
        arc_scores = arc_scores.masked_fill(~head_mask, -1e9)
        rel_scores = rel_scores.masked_fill(~head_mask.unsqueeze(-1), -1e9)

        seq_len = arc_scores.size(1)
        diagonal_mask = torch.eye(seq_len, device=arc_scores.device, dtype=torch.bool).unsqueeze(0)
        arc_scores[:, 1:, :] = arc_scores[:, 1:, :].masked_fill(diagonal_mask[:, 1:, :], -1e9)

        return arc_scores, rel_scores, token_mask
