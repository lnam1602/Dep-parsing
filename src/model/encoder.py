import torch.nn as nn
from transformers import AutoModel


class Encoder(nn.Module):
    def __init__(self, model_name="vinai/phobert-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
