import torch
import torch.nn as nn


class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        x_dim = in_dim + int(bias_x)
        y_dim = in_dim + int(bias_y)
        self.U = nn.Parameter(torch.zeros(x_dim, out_dim, y_dim))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x, y):
        if self.bias_x:
            ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
            x = torch.cat((x, ones), dim=-1)
        if self.bias_y:
            ones = torch.ones(*y.shape[:-1], 1, device=y.device, dtype=y.dtype)
            y = torch.cat((y, ones), dim=-1)
        return torch.einsum("bxi,ioj,byj->bxyo", x, self.U, y)
