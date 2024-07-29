import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import Optional, Tuple

from src.models.core.components.utils.ops import LayerNorm, init_weights, LoRALinear1d
from src.models.core.components.flytts.stft import OnnxSTFT, TorchSTFT


def get_padding(kernel_size, dilation=1):
    return int(((kernel_size - 1) * dilation) / 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

# z, speaker_id_embeddedを入力にとり音声を生成するネットワーク
class VocosDecoder(torch.nn.Module):
    def __init__(
        self,
        speaker_id_embedding_dim=0,  # 話者idの埋め込み先のベクトルの大きさ
        in_z_channel=192,  # 入力するzのchannel数
        hidden_dim=512,  # 入力されたzと埋め込み済み話者idの両者のチャネル数を、まず最初にconvを適用させることによってupsample_initial_channelに揃える
        itnermediate_dim=1536,
        num_layer: int = 6,
        gen_istft_n_fft=1024,
        gen_istft_hop_sizes=256,
        is_onnx=False
    ):  # 各ResnetBlockのdilation
        super(VocosDecoder, self).__init__()

        self.speaker_id_embedding_dim = speaker_id_embedding_dim  # 話者idの埋め込み先のベクトルの大きさ
        self.in_z_channel = in_z_channel  # 入力するzのchannel数
        self.hidden_dim = hidden_dim  # 入力されたzと埋め込み済み話者idの両者のチャネル数を、まず最初にconvを適用させることによってupsample_initial_channelに揃える 
        self.num_layer = num_layer
        layer_scale_init_value = 1 / num_layer
        out_channel = gen_istft_n_fft + 2
        
        self.embed = nn.Conv1d(
            self.in_z_channel,
            self.hidden_dim,
            kernel_size=7,
            padding=3
        )
        
        # 入力された埋め込み済み話者idに最初に適用するネットワーク
        if speaker_id_embedding_dim != 0:
            self.cond = LoRALinear1d(self.hidden_dim, self.hidden_dim, self.speaker_id_embedding_dim, self.speaker_id_embedding_dim // 4)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=hidden_dim,
                    intermediate_dim=itnermediate_dim,
                    layer_scale_init_value=layer_scale_init_value
                )
                for _ in range(num_layer)
            ]
        )
        # resnet_blocks_channels = 32
        self.post_n_fft = gen_istft_n_fft
        self.final_layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.out_conv = nn.Conv1d(hidden_dim, out_channel, kernel_size=1)
        self.pad = nn.ReflectionPad1d([1, 0])
        if is_onnx:
            self.stft = OnnxSTFT(
                filter_length=gen_istft_n_fft,
                hop_length=gen_istft_hop_sizes,
                win_length=gen_istft_n_fft,
            )
        else:
            self.stft = TorchSTFT(
                filter_length=gen_istft_n_fft,
                hop_length=gen_istft_hop_sizes,
                win_length=gen_istft_n_fft,
            )

        self.apply(self._init_weights)
        
    def forward(self, z, speaker_id_embedded):
        if speaker_id_embedded is None:
            x = self.embed(z)
        else:
            # z, speaker_id_embedded両者のchannel数をconv1dによって揃える
            x = self.cond(self.embed(z), speaker_id_embedded)
        # 各Deconv1d層の適用
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pad(x)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp(max=1e2)
        phase = math.pi * torch.sin(phase)
        out = self.stft.inverse(mag, phase).to(x.device)
        
        return out
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    z_channel = 192
    x = torch.randn(2, z_channel, 1280)
    g = torch.randn(2, 192, 1)
    import time
    s = time.time()
    for _ in range(1):
        model = VocosDecoder(
            speaker_id_embedding_dim=192,
            in_z_channel=z_channel,
        ).cpu()
        
        o = model(x, g)
    print(o.shape) # torch.Size([2, 1, 32768]) (256倍にアップサンプリングされている)
    print(time.time() - s, "sec")