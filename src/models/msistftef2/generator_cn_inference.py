import math

import torch
import torch.nn as nn

from src.config.model.vits import netG
from src.config.dataset.default import DatasetConfig
from src.tts.phonome.param import (
    num_accents,
    num_phonomes
)

from src.models.msistftef2.modules import (
    TextEncoder, EncoderCFG, PosteriorEncoder, 
    AttentionOperator, HybridAttention, PriorNN, 
    VarationalAlignmentPredictor
)
from src.models.core.components.mbistft.modules import Multistream_iSTFT_Generator as Decoder
from src.models.core.components.utils.commons import sequence_mask
from src.models.core.components.generate_path import generate_path
from src.tts.utils.audio import rand_slice_segments
try:
    from src.models.core import monotonic_align
except ImportError:
    monotonic_align = None
    print("Ineference mode is not need for monotonic alignment.")
    


class Generator(nn.Module):
    def __init__(self,
                cfg: netG,
                data_cfg: DatasetConfig
        ):
        super(Generator, self).__init__()
        self._cfg = cfg
        self._n_mels = data_cfg.mel_bins
        self._segment_size = data_cfg.segment_size // data_cfg.hop_length
        self._n_speaker = data_cfg.n_speaker
        self._n_phonome = num_phonomes()
        self._n_accents = num_accents() if data_cfg.accent_split else 0
        
        self._use_noise_scaled_mas = cfg.use_noise_scaled_mas
        self._mas_noise_scale_initial = cfg.mas_nosie_scale_initial
        self._mas_noise_scale_delta = cfg.mas_noise_scale_delta
        self.current_mas_noise_scale = self._mas_noise_scale_initial
        
        self.text_encoder = TextEncoder(
            vocab_size=self._n_phonome,
            accent_position_size=self._n_accents,
            input_embedding_dim=cfg.phoneme_embedding_dim,
            enccfg=EncoderCFG()
        )
        
        self.speaker_embedding = nn.Embedding(
            num_embeddings=self._n_speaker,
            embedding_dim=cfg.speaker_id_embedding_dim
        )
        
        #z, speaker_id_embeddedを入力にとり音声を生成するネットワーク
        self.decoder = Decoder(
            initial_channel=cfg.z_channels,
            block_type="convnext",
            block_kernel_sizes=[7, 7, 11],
            block_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            upsample_rates=[4, 4],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16],
            gen_istft_n_fft=16,
            gen_istft_hop_size=4,
            subbands=4,
            gin_channels=cfg.speaker_id_embedding_dim,
            convnext_hidden_dim_rate=3
        )
        self.attn = HybridAttention(cfg.z_channels)
        self.prior_nn1 = PriorNN(cfg.z_channels, cfg.hidden_channels, 5, 1, cfg.prior_nn_layers1, gin_channels=cfg.speaker_id_embedding_dim)
        self.prior_nn2 = PriorNN(cfg.z_channels, cfg.hidden_channels, 5, 1, cfg.prior_nn_layers2, gin_channels=cfg.speaker_id_embedding_dim)
        self.vap = VarationalAlignmentPredictor(
            cfg.hidden_channels, 3, n_layers=cfg.vap_layers, gin_channels=cfg.speaker_id_embedding_dim
        )

    def forward(
        self,
        text_padded,
        text_lengths,
        accent_pos_padded,
        speaker_id,
        speaker_emb=None,
        t1=0.7,
        t2=0.7,
        length_scale=1.0,
        ta=0.7,
    ):
        # text(音素)の内容をTextEncoderに通す
        # text_encoded: (B, 192, TT), text_mask: (B, 1, TT)
        text_encoded, text_mask = self.text_encoder(
            text_padded, text_lengths, accent_pos_padded
        ) # m_p,logs_pは音素の情報を表す
        
        # 話者埋め込み (B, 128, 1)
        if speaker_emb is None:
            speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(-1)
        vap_outs = self.vap.infer(text_encoded, text_mask, t_a=t1, g=speaker_emb)
        b = torch.cumsum(vap_outs[:,1,:], dim=1) * length_scale
        a = torch.cat([torch.zeros(b.size(0), 1).type_as(b), b[:,:-1]], -1)
        e = a + vap_outs[:,0,:] * length_scale
        x_align, attns, real_sigma = self.attn(e, a, b, text_encoded, text_mask=None, mel_mask=None, max_length = self._cfg.attn_max_len)
        y_mask = torch.ones(x_align.size(0), 1, x_align.size(-1)).type_as(text_mask)
        z_1, _, _, _= self.prior_nn1(x_align, y_mask, g=speaker_emb, t=t1)
        z_2, _, _, _= self.prior_nn2(z_1, y_mask, g=speaker_emb, t=t2)
        o, _ = self.decoder(z_2, speaker_id_embedded=None)
        return o, attns, y_mask
    
    def update_current_mas_noise_scale(self):
        if self._use_noise_scaled_mas:
            self.current_mas_noise_scale = max(0, self.current_mas_noise_scale - self._mas_noise_scale_delta)


if __name__ == "__main__":
    t = Generator(
        netG(),
        DatasetConfig()
    )
    
    t_lengths = torch.Tensor([6, 10])
    text = torch.ones((2, 10), dtype=torch.long)
    accent =  torch.ones((2, 10), dtype=torch.long)
    spec = torch.ones((2, 513, 25), dtype=torch.float32)
    spec_lengths = torch.Tensor([23, 25])
    spk_ids = torch.Tensor([0, 0]).to(torch.long)
    (
        wave_fake,
        (loss_a, loss_a_kl),
        attns,
        ids_slice,
        text_mask,
        spec_mask,
        (m1, logs1, m_q1, logs_q1),
        (m2, logs2, m_q2, logs_q2),
        _,
        real_sigma
    ) = t(text, t_lengths, accent, spec, spec_lengths, spk_ids)
    print("---")
    print(wave_fake.shape) # torch.Size([2, 1, 2560]) 10x256
    print(loss_a.shape, loss_a_kl.shape)