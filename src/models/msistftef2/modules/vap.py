from torch import nn
import torch
from src.models.msistftef2.modules.modules import WN

LOSS_DUR_SCALE = 0.1

class VarationalAlignmentPredictor(nn.Module):
    def __init__(self, filter_channels, kernel_size, n_layers, duration_offset=1.0, gin_channels=0):
        super().__init__()
        self.conv_y = nn.Conv1d(2, filter_channels, 1)
        if gin_channels > 0:
            self.conv_g = nn.Conv1d(gin_channels, filter_channels, 1)
        self.encoder = WN(filter_channels, kernel_size, 1, n_layers, filter_channels)
        self.proj_z = nn.Conv1d(filter_channels, 2*filter_channels, 1)
        self.decoder = WN(filter_channels, kernel_size, 1, n_layers, gin_channels=filter_channels, p_dropout=0.5)
        self.out = nn.Conv1d(filter_channels, 2, 1)
        self.duration_offset = duration_offset

    def forward(self, x, x_mask, e_a, b_a, g=None):
        # x:(B, 192, TT), x_mask:(B, 1, TT)
        # e_a:(B, TT), e_b:(B, TT) => (B, 2, TT)
        y_input = torch.detach(torch.cat([e_a.unsqueeze(1), b_a.unsqueeze(1)], 1)) * x_mask
        y = self.conv_y(y_input) # (B, 2, TT) => (B, filter_channels;192, TT)

        if g is not None:
            x = x + self.conv_g(g)
        x = torch.detach(x)*x_mask
        y_enc = self.encoder(y, x_mask, g=x) # (B, filter_channels, TT)
        stats = self.proj_z(y_enc) # (B, 2*filter_channels, TT)
        m, logs = torch.split(stats, stats.size(1)//2, dim=1) # (B, filter_channels, TT), (B, filter_channels, TT)

        z = torch.distributions.Normal(m, torch.exp(logs*0.5)).rsample()*x_mask # (B, filter_channels, TT)
        dec_y = self.decoder(z,x_mask, g=x) # (B, filter_channels, TT)
        y_hat = self.out(dec_y) # (B, 2, TT)
        mask_sum = torch.clamp(torch.sum(x_mask), min=1e-4)
        loss_dur = torch.sum(torch.abs((torch.log(torch.clamp(y_input + self.duration_offset,min=1e-8))*x_mask - y_hat*x_mask)) * LOSS_DUR_SCALE) / (mask_sum * LOSS_DUR_SCALE)
        loss_kl = torch.sum( torch.distributions.kl_divergence(torch.distributions.Normal(m, torch.exp(logs*0.5)),
                      torch.distributions.Normal(torch.zeros_like(m), torch.ones_like(m)))* x_mask) / mask_sum
        if torch.isnan(loss_dur).any():
            print("loss_dur is nan")
            loss_dur = torch.zeros_like(loss_dur)
        if torch.isnan(loss_kl).any():
            print("loss_kl is nan")
            loss_kl = torch.zeros_like(loss_kl)
        return loss_dur, loss_kl
    
    def infer(self, x, x_mask, t_a=1.0, g=None):
        if g is not None:
            x = x + self.conv_g(g)
        x = x * x_mask
        z = torch.randn_like(x)* t_a
        dec_y = self.decoder(z, x_mask, g=x)
        y_hat = self.out(dec_y)
        return torch.clamp(torch.exp(y_hat) - self.duration_offset, min=0) * x_mask