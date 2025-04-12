import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from src.models.core.components.flytts.stft import OnnxSTFT
from src.models.core.components.flytts.decode import get_padding
import math

LRELU_SLOPE = 0.1

class ConvNextBottleNeck(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilatoin: int = 1,
        hidden_dim_rate: int = 4,
        layer_scale_init_value: float = -1.0
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilatoin, padding=get_padding(kernel_size, dilatoin),
                groups=channels
            )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        hidden_dim = int(channels * hidden_dim_rate)
        self.pw_conv1 = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(hidden_dim, channels)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        x = residual + x
        return x
class ConvNextBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilatoin: tuple = (1, 3, 5),
        hidden_dim_rate: int = 3
    ) -> None:
        super().__init__()
        self.conv_list = nn.ModuleList([
            ConvNextBottleNeck(channels, kernel_size, dilatoin=dilatoin[i], hidden_dim_rate=hidden_dim_rate)
            for i in range(len(dilatoin))
        ])
    
    def forward(self, x: torch.Tensor, x_maxk=None) -> torch.Tensor:
        if x_maxk is not None:
            x = x * x_maxk
        for conv in self.conv_list:
            x = conv(x)
        if x_maxk is not None:
            x = x * x_maxk
        return x

class Conv1dGNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv1dGNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Conv1dTransposeGNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv1dTransposeGNorm, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    

class LoRALinear1d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.r = r
        self.main_fc = Conv1dGNorm(in_channels, out_channels, 1)
        self.adapter_in = Conv1dGNorm(info_channels, in_channels * r, 1)
        self.adapter_out = Conv1dGNorm(info_channels, out_channels * r, 1)
    def forward(self, x, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        x = self.main_fc(x) + torch.einsum("brl,brc->bcl", torch.einsum("bcl,bcr->brl", x, a_in), a_out)
        return x

class Multistream_iSTFT_Generator(torch.nn.Module):
    def __init__(
        self, 
        initial_channel, 
        block_type:str = "resblock", 
        block_kernel_sizes=[3, 7, 11],
        block_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        upsample_rates=[4, 4],
        upsample_initial_channel=512, 
        upsample_kernel_sizes=[16, 16],
        gen_istft_n_fft=16, 
        gen_istft_hop_size=4, 
        subbands=4, 
        gin_channels=0,
        convnext_hidden_dim_rate=3
    ):
        super(Multistream_iSTFT_Generator, self).__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(block_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1dGNorm(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        if block_type == "convnext":
            block_module = ConvNextBlock
        else:
            raise ValueError("block_type should be resblock")
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # upsample rate is controlled by stride
            self.ups.append(Conv1dTransposeGNorm(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2))

        self.blocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(block_kernel_sizes, block_dilation_sizes)):
                if block_type == "convnext":
                    self.blocks.append(block_module(ch, k, d, hidden_dim_rate=convnext_hidden_dim_rate))
                else:
                    raise ValueError("block_type should be resblock")
        self.post_n_fft = gen_istft_n_fft
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []

        self.subband_conv_post = Conv1dGNorm(ch, self.subbands*(self.post_n_fft + 2), 7, 1, padding=3)
        
        
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.multistream_conv_post = Conv1dGNorm(4, 1, kernel_size=63, bias=False, padding=get_padding(63, 1))
        
        self.stft = OnnxSTFT(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft,
        )
        
        self.gin_cond = None
        if gin_channels > 0:
            self.gin_cond = LoRALinear1d(
                upsample_initial_channel, 
                upsample_initial_channel,
                gin_channels,
                gin_channels // 4
            )


    def forward(self, x, speaker_id_embedded=None):
        # x: [B, 192, length]
        g = speaker_id_embedded
        self.stft.to(x.device)
        # add speaker embedding
        if g is None:
            x = self.conv_pre(x)
        elif self.gin_cond is not None:
            x = self.gin_cond(self.conv_pre(x), g) #[B, ch, length]
        else:
            raise ValueError("g is not None but gin_channels is 0")
        # gin x: [B, 512, length]
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            # upsample: 1: [B, 256, length*4], 2: [B, 128, length*16]
            
       
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.blocks[i*self.num_kernels+j](x)
                else:
                    xs += self.blocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            # block: [B, 256, length*16], [B, 128, length*64]
        x = F.leaky_relu(x)
        x = self.reflection_pad(x) # [B, 128, length*64 + 1]
        x = self.subband_conv_post(x) # [B, 4*(n_fft:16 + 2):72, length*64 + 1]
        x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1]//self.subbands, x.shape[-1])) # [B, 4, 18, length*64 + 1]
        spec = torch.exp(x[:,:,:self.post_n_fft // 2 + 1, :]) # [B, 4, 9, length*64 + 1]
        phase = math.pi*torch.sin(x[:,:, self.post_n_fft // 2 + 1:, :]) # [B, 4, 9, length*64 + 1]
        y_mb_hat = self.stft.inverse(
            torch.reshape(spec, (spec.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1])), 
            torch.reshape(phase, (phase.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1]))
        ) # [B*4, 1, length*64]
        y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1])) # [B, 4, 1, length*64]
        y_mb_hat = y_mb_hat.squeeze(-2) # [B, 4, length*64]
        y_mb_hat = F.conv_transpose1d(y_mb_hat, self.updown_filter.to(x.device) * self.subbands, stride=self.subbands) # [B, 4, length*64*4]
        y_g_hat = self.multistream_conv_post(y_mb_hat) # [B, 1, length*64*4]
        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.blocks:
            if hasattr(l, "remove_weight_norm"):
                l.remove_weight_norm()

        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.subband_conv_post)
        remove_weight_norm(self.multistream_conv_post)
if __name__ == "__main__":
    z_channel, gin_channel = 192, 192
    batch_size = 1
    x = torch.randn(batch_size, z_channel, 861)
    g = torch.randn(batch_size, gin_channel, 1)
    import time
    
    check_num = 3
    model = Multistream_iSTFT_Generator(
        initial_channel=192,
        block_type="resblock",
        block_kernel_sizes=[3, 7, 11],
        block_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        upsample_rates=[4, 4],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16],
        gen_istft_n_fft=16,
        gen_istft_hop_size=4,
        subbands=4,
        gin_channels=gin_channel
    ).cpu()
    
    s = time.time()
    for _ in range(check_num):
        o = model(x, g)
    print(o[0].shape, o[1].shape) # torch.Size([2, 1, 32768]) (256倍にアップサンプリングされている)
    print((time.time() - s) / check_num) # 0.002s
    
    model = Multistream_iSTFT_Generator(
        initial_channel=192,
        block_type="convnext",
        block_kernel_sizes=[7, 7, 11],
        block_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        upsample_rates=[4, 4],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16],
        gen_istft_n_fft=16,
        gen_istft_hop_size=4,
        subbands=4,
        gin_channels=gin_channel,
        convnext_hidden_dim_rate=3
    ).cpu()
    
    s = time.time()
    for _ in range(check_num):
        o = model(x, g)
    print(o[0].shape, o[1].shape) # torch.Size([2, 1, 32768]) (256倍にアップサンプリングされている)
    print((time.time() - s) / check_num) # 0.002s
    
    model = Multistream_iSTFT_Generator(
        initial_channel=192,
        block_type="convnext",
        block_kernel_sizes=[7, 7, 11],
        block_dilation_sizes=[(1, 3), (1, 3), (1, 3)],
        upsample_rates=[4, 4],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16],
        gen_istft_n_fft=16,
        gen_istft_hop_size=4,
        subbands=4,
        gin_channels=gin_channel,
        convnext_hidden_dim_rate=2
    ).cpu()
    
    s = time.time()
    for _ in range(check_num):
        o = model(x, g)
    print(o[0].shape, o[1].shape) # torch.Size([2, 1, 32768]) (256倍にアップサンプリングされている)
    print((time.time() - s) / check_num) # 0.002s