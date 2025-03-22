from torch import nn
import torch
from src.models.msistftef2.modules.modules import WN


class PriorNN(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, p_dropout=0.2)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, mask, g=None, t=1.0):
    x = self.enc(x, mask, g=g)
    x = self.proj(x) * mask
    m, logs = torch.split(x, self.out_channels, dim=1)
    z =  torch.distributions.Normal(m, torch.exp(logs)*t).rsample()*mask
    return z, m, logs, mask
