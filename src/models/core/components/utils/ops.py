import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, pad: int):
        super().__init__()
        assert isinstance(kernel, int), f"kernel is int for 1d conv."
        assert isinstance(pad, int), f"pad is int for 1d conv."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._kernel = (1, 2)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Sequential()

        self.block1 = nn.Sequential(
            nn.LeakyReLU(0.2), nn.Conv1d(in_channels, in_channels, kernel_size=kernel, padding=pad)
        )

        self.block2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True), nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        s = self.skip(x)
        s = F.avg_pool2d(s, self._kernel, stride=self._kernel, padding=(0, 0))

        o = self.block1(x)
        o = F.avg_pool2d(o, self._kernel, stride=self._kernel, padding=(0, 0))
        o = self.block2(o)

        return s + o


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, spk_emb_channels=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spk_emb_channels = spk_emb_channels

        if in_channels != out_channels:
            raise NotImplementedError()
        else:
            self.skip = nn.Sequential()

        self.block1 = nn.Sequential(
            nn.ReflectionPad1d((dilation, dilation)),
            nn.Conv1d(in_channels, out_channels * 2, kernel_size=3, dilation=dilation),
        )

        if spk_emb_channels is not None:
            self.spk_emb_block = nn.Conv1d(spk_emb_channels, out_channels * 2, kernel_size=1)

        self._tanh = nn.Tanh()
        self._sigmoid = nn.Sigmoid()
        self.block2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=(dilation))

    def forward(self, x, spk_emb=None):
        o = self.block1(x)
        if spk_emb is not None:
            o = o + self.spk_emb_block(spk_emb)
        o = self._tanh(o[:, : self.out_channels, ...]) * self._sigmoid(o[:, self.out_channels :, ...])
        o = self.block2(o)
        s = self.skip(x)
        return o + s


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spk_emb_channels, r, num_resblock) -> None:
        super().__init__()
        # out_channels = mult * ngf
        self.gelu = nn.GELU()
        self.deconv1 = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=r * 2, stride=r, padding=(r // 2 + r % 2)
        )

        resblocks = []
        for i in range(num_resblock):
            resblocks.append(ResnetBlock(out_channels, out_channels, 3**i, spk_emb_channels))
        self.resblocks = nn.ModuleList(resblocks)

    def forward(self, x, spk_emb):
        # rが偶数なので、入力も偶数が良い
        x = self.gelu(x)
        x = self.deconv1(x)
        for resblock in self.resblocks:
            x = resblock(x, spk_emb)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r, num_resblock) -> None:
        super().__init__()

        resblocks = []
        for i in range(num_resblock):
            resblocks.append(ResnetBlock(in_channels, in_channels, 3**i, None))
        self.resblocks = nn.ModuleList(resblocks)
        self.gelu = nn.GELU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=r * 2, stride=r, padding=(r // 2 + r % 2))

    def forward(self, x):
        # rが偶数なので、入力も偶数が良い
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.gelu(x)
        return self.conv(x)

class LoRALinear1d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.r = r
        self.main_fc = weight_norm(nn.Conv1d(in_channels, out_channels, 1))
        self.adapter_in = nn.Conv1d(info_channels, in_channels * r, 1)
        self.adapter_out = nn.Conv1d(info_channels, out_channels * r, 1)
        nn.init.normal_(self.adapter_in.weight.data, 0, 0.01)
        nn.init.constant_(self.adapter_out.weight.data, 1e-6)
        init_weights(self.main_fc)
        self.adapter_in = weight_norm(self.adapter_in)
        self.adapter_out = weight_norm(self.adapter_out)

    def forward(self, x, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        x = self.main_fc(x) + torch.einsum("brl,brc->bcl", torch.einsum("bcl,bcr->brl", x, a_in), a_out)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.main_fc)
        remove_weight_norm(self.adapter_in)
        remove_weight_norm(self.adapter_out)

if __name__ == "__main__":
    import torch

    def test_resblock():
        x = torch.ones((2, 32, 1000))  # waaveform
        spk_emb = torch.ones((2, 128, 1))  # spkemb

        rb = ResnetBlock(32, 32, 3, 128)
        o = rb(x, spk_emb)
        print(o.shape)
        o = rb(x)
        print(o.shape)

    def test_residualblock():
        x = torch.ones((2, 250, 250))  # mel spec
        x = nn.Conv1d(250, 32, kernel_size=3, padding=1)(x)
        rb = ResidualBlock(32, 128, 3, 1)
        o = rb(x)
        print(o.shape)

    def test_upblock():
        x = torch.ones((2, 32, 1000))
        db = DownBlock(32, 12, 2, 2)
        x = db(x)
        print(x.shape)
        spk_emb = torch.ones((1, 128, 1))
        ub = UpBlock(12, 32, 128, 2, 2)
        o = ub(x, spk_emb)
        print(o.shape)

    test_resblock()
    test_residualblock()
    test_upblock()