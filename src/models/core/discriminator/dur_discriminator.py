import torch
import torch.nn as nn
import torch.nn.functional as F

class DurationDiscriminator(nn.Module):  # vits2
    # TODO : not using "spk conditioning" for now according to the paper.
    # Can be a better discriminator if we use it.
    def __init__(
        self, in_channels, filter_channels, kernel_size=3, p_dropout=0.1, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        #assert p_dropout == 0.0, "Not implemented yet."
        #assert gin_channels == 0, "Not implemented yet."

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pre_out_conv_2(x * x_mask)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        assert g is None, "Not implemented yet."
        x = torch.detach(x)
        x = self.conv_1(x * x_mask)
        x = F.leaky_relu(x, negative_slope=0.1)

        x = self.conv_2(x * x_mask)
        x = F.leaky_relu(x, negative_slope=0.1)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs
