import torch
from torch import nn
import numpy as np
from src.models.core.components.wn import gated_activation_unit

class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels =hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                        dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        # x.size(), x_mask.size() : torch.Size([batch_size, 192, length(可変)]) torch.Size([batch_size, 1, length])
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            else:
                g_l = torch.zeros_like(x_in)
            # gated_activation_unitの適用　通す情報と通さない情報をフィルタリングすることで学習効率を改善する役割を果たすとされる
            acts = gated_activation_unit(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                # 特徴量の前半はxと加算
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts) * x_mask
                # 特徴量の後半は出力用tensorに加算
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)

class HybridAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(2)*0.1)
        self.linear_hidden = nn.Conv1d(channels,channels*2,1)
        self.linear_out = nn.Conv1d(channels*2,channels,1)

    def forward(self, e, a, b, x_h, text_mask=None, mel_mask=None, max_length=1000, min_length=10):
        if mel_mask is None : # for inference only
            length = torch.round(b[:,-1]).squeeze().item() + 1
            length = min_length if length < min_length else length
            length = max_length if length > max_length else length
        else:
            length = mel_mask.size(-1)
        q = torch.arange(0, length).unsqueeze(0).repeat(e.size(0),1).type_as(e)
        if mel_mask is not None:
            q = q*mel_mask.float()
        energies_e = -1 * (q.unsqueeze(-1) - e.unsqueeze(1))**2
        energies_boundary = -1 * (
            torch.abs(q.unsqueeze(-1) - a.unsqueeze(1)) +
            torch.abs(q.unsqueeze(-1) - b.unsqueeze(1)) -
            (b.unsqueeze(1) - a.unsqueeze(1)) )**2
        energies_all = torch.cat([energies_e.unsqueeze(1), energies_boundary.unsqueeze(1)], 1)
        real_sigma = torch.clamp(self.sigma, max=3, min=1e-6)
        energies = energies_all * real_sigma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        if text_mask is not None:
            energies = energies.masked_fill(~(text_mask.unsqueeze(1).unsqueeze(1)), -float('inf'))
        attns = torch.softmax(energies, dim=-1) # (B, 2, TS, TT)
        h = self.linear_hidden(x_h) # (B, 192, TT) => (B, 384, TT)
        h = h.view(h.size(0), 2, h.size(1)//2, -1).transpose(2,3) # B, 2, TT, 192
        out = torch.matmul(attns, h) # (B, 2, TS, TT) x (B, 2, TT, 192) => (B, 2, TS, 192)
        out = out.transpose(2,3).contiguous().view(h.size(0), h.size(3)*2, -1) # (B, 192*2, TS)
        out = self.linear_out(out) # (B, 192*2, TS) => (B, 192, TS)
        return out, attns, real_sigma

class AttentionPI(nn.Module):
    def __init__(self, channels, attntion_heads):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(attntion_heads)*0.1)
        self.attntion_heads = attntion_heads
        self.linear_hidden = nn.Conv1d(channels,channels*2,1)
        self.linear_out = nn.Conv1d(channels*2,channels,1)

    def forward(self, pi, p,  x_h, text_mask=None, mel_mask=None):
        energies = -1 * (pi.unsqueeze(-1) - p.unsqueeze(1))**2
        real_sigma = torch.clamp(self.sigma, max=3, min=1e-6)
        energies = energies.unsqueeze(1) * real_sigma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        if text_mask is not None:
            energies = energies.masked_fill(~(text_mask.unsqueeze(1).unsqueeze(1)), -float('inf'))
        attns = torch.softmax(energies, dim=-1)
        h = self.linear_hidden(x_h)
        h = h.view(h.size(0), self.attntion_heads, h.size(1)//self.attntion_heads, -1).transpose(2,3)
        out = torch.matmul(attns, h)
        out = out.transpose(2,3).contiguous().view(h.size(0), h.size(3)*self.attntion_heads, -1)
        out = self.linear_out(out)
        return out, attns, real_sigma

class AttentionOperator(nn.Module):
    def __init__(self, hid, n_position=1000):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, hid))

    def forward(self, x_h, y_h, x_mask, y_mask, sigma):
        return self.compute_e_and_boundaries(x_h, y_h, x_mask, y_mask, sigma)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(hid_j // 2)/ d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def compute_PI(self, x_h, y_h, x_mask, y_mask):
        # pos_table: (1, n_position:1000, hid:192)
        # x_h: (B, 192, TT), y_h: (B, 192, TS)
        x_h = x_h + self.pos_table[:,:x_h.size(-1)].clone().detach().transpose(1, 2)
        y_h = y_h + self.pos_table[:,:y_h.size(-1)].clone().detach().transpose(1, 2)
        # (B, TS, 192)*(B,192, TT) => scores: (B, TS, TT) dot product
        scores = torch.bmm(y_h.transpose(1,2), x_h) / np.sqrt(float(x_h.size(1)))
        scores = scores.masked_fill(~x_mask.unsqueeze(1), -float('inf'))
        alpha = torch.softmax(scores, dim=-1) # (B, TS, TT)
        p = torch.arange(0, alpha.size(-1)).type_as(alpha).unsqueeze(0) * x_mask.float() # (B, TT)
        pi_dummy = torch.bmm(alpha, p.unsqueeze(-1)).squeeze(-1) # (B, TS)
        delta_pi = torch.relu(pi_dummy[:,1:]-pi_dummy[:, :-1]) # (B, TS-1)
        delta_pi = torch.cat([torch.zeros(alpha.size(0), 1).type_as(alpha), delta_pi], -1) * y_mask.float()
        pi_f = torch.cumsum(delta_pi, -1)* y_mask.float() # (B, TS)
        delta_pi_inverse = torch.flip(delta_pi, [-1])
        pi_b = - torch.flip(torch.cumsum(delta_pi_inverse, -1), [-1])
        pi = pi_f + pi_b
        last_pi, _ = torch.max(pi, dim=-1)
        last_pi = torch.clamp(last_pi, min=1e-8).unsqueeze(1)
        first_pi = pi[:,0:1]
        x_lengths = torch.sum(x_mask.float(), -1)
        max_pi = x_lengths.unsqueeze(-1) -1 
        pi = (pi - first_pi) / (last_pi - first_pi) * max_pi # (B, TS)
        # pi は、p(0, 1, 2, ...TT-1)の位置に対応するy_hの位置を表す
        # pi (0, 0.2046, ..., 5.135, ...TT-1)
        return pi, p

    def compute_e_and_boundaries(self, x_h, y_h, x_mask, y_mask, sigma=0.2):
        # x_h: (B, 192, TT), y_h: (B, 192, TS)
        # pi: (B, TS), p: (B, TT) piは、テキストの対応位置を表す
        pi, p = self.compute_PI(x_h, y_h, x_mask, y_mask)
        # energies: (B, TT, TS) 位置の差を計算
        energies = -1 * (pi.unsqueeze(1) - p.unsqueeze(-1))**2 * sigma
        energies = energies.masked_fill(
            ~(y_mask.unsqueeze(1).repeat(1, energies.size(1), 1)), -float('inf')
        )
        beta = torch.softmax(energies, dim=2)
        q = torch.arange(0, y_mask.size(-1)).unsqueeze(0).type_as(pi) * y_mask.float() # (B, TS)
        e = torch.bmm(beta, q.unsqueeze(-1)).squeeze(-1) * x_mask.float() # (B, TT)

        boundary_idx = torch.clamp(p-0.5, min=0)
        energies_a = -1 *(pi.unsqueeze(1) - boundary_idx.unsqueeze(-1))**2 * sigma
        energies_a = energies_a.masked_fill(
            ~(y_mask.unsqueeze(1).repeat(1, energies_a.size(1), 1)), -float('inf')
        )
        beta_a = torch.softmax(energies_a, dim=2) # (B, TT, TS)
        a = torch.bmm(beta_a, q.unsqueeze(-1)).squeeze(-1) * x_mask.float()
        a_real = torch.cat([torch.zeros(a.size(0), 1).type_as(a), a[:,1:]], -1) # (B, TT)

        max_x = torch.sum(x_mask, dim=-1) - 1
        max_y = torch.sum(y_mask, dim=-1) - 1
        b = torch.cat([a_real[:,1:], torch.zeros(a.size(0), 1).type_as(a)], -1) # (B, TT)
        b_real = b.scatter_(1, max_x.unsqueeze(1), max_y.unsqueeze(1).type_as(b)) # (B, TT)
        return e, a_real, b_real