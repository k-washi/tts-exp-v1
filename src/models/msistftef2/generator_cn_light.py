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
        
        # linear spectrogramと埋め込み済み話者idを入力にとりEncodeを実行、zを出力するモデル
        self.posterior_encoder = PosteriorEncoder(
            speaker_id_embedding_dim=cfg.speaker_id_embedding_dim,  # 話者idの埋め込み先のベクトルの大きさ xxx Transform Vits
            in_spec_channels=cfg.spec_channels,  # 入力する線形スペクトログラムの縦軸(周波数)の次元
            out_z_channels=cfg.z_channels,  # PosteriorEncoderから出力されるzのchannel数
            # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
            phoneme_embedding_dim=cfg.phoneme_embedding_dim,
        )
        
        #z, speaker_id_embeddedを入力にとり音声を生成するネットワーク
        self.decoder = Decoder(
            initial_channel=cfg.z_channels,
            block_type="convnext",
            block_kernel_sizes=[7, 7, 11],
            block_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            upsample_rates=[4, 4],
            upsample_initial_channel=cfg.decoder_upsample_initial_channel,
            upsample_kernel_sizes=[16, 16],
            gen_istft_n_fft=16,
            gen_istft_hop_size=4,
            subbands=4,
            gin_channels=cfg.speaker_id_embedding_dim,
            convnext_hidden_dim_rate=2
        )
        self.attn = HybridAttention(cfg.z_channels)
        self.prior_nn1 = PriorNN(cfg.z_channels, cfg.hidden_channels, 5, 1, cfg.prior_nn_layers1, gin_channels=cfg.speaker_id_embedding_dim)
        self.prior_nn2 = PriorNN(cfg.z_channels, cfg.hidden_channels, 5, 1, cfg.prior_nn_layers2, gin_channels=cfg.speaker_id_embedding_dim)
        self.vap = VarationalAlignmentPredictor(
            cfg.hidden_channels, 3, n_layers=cfg.vap_layers, gin_channels=cfg.speaker_id_embedding_dim
        )
        self.attn_op = AttentionOperator(cfg.hidden_channels, n_position=cfg.attn_max_len)
        
    def forward(
        self,
        text_padded,
        text_lengths,
        accent_pos_padded,
        spec_padded,
        spec_lengths,
        speaker_id=torch.Tensor([0]).to(torch.long),
        speaker_emb=None,
        bi=False,
    ):
        """_summary_

        Args:
            text_padded (_type_): (B, TT)の音素の系列
            text_lengths (_type_): (B)の音素の系列の長さ
            accent_pos_padded (_type_): (B, TT)のアクセントの位置の系列
            spec_padded (_type_): (B, 513, TS)の線形スペクトログラム
            spec_lengths (_type_): (B)の線形スペクトログラムの長さ
            speaker_id (_type_, optional): (B)の話者ID. Defaults to torch.Tensor([0]).to(torch.long).
            speaker_emb (_type_, optional): _description_. Defaults to None.
            bi (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # text(音素)の内容をTextEncoderに通す
        # text_encoded: (B, 192, TT), text_mask: (B, 1, TT)
        text_encoded, text_mask = self.text_encoder(
            text_padded, text_lengths, accent_pos_padded
        ) # m_p,logs_pは音素の情報を表す
        
        # 話者埋め込み (B, 128, 1)
        if speaker_emb is None:
            speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(-1)
        
        #linear spectrogramと埋め込み済み話者idを入力にとりEncodeを実行、zを出力する
        # spec_padded: (B, 513, TS), spec_lengths: (B), speaker_emb: (B, 128, 1)
        # y_h: (B, 192, TS), z1: (B, 192, TS), z2: (B, 192, TS), 
        # m1: (B, 192, TS), logs1: (B, 192, TS),
        # m2: (B, 192, TS), logs2: (B, 192, TS), 
        # spec_mask: (B, 1, TS)
        y_h, z1, z2, m1, logs1, m2, logs2, spec_mask = self.posterior_encoder(spec_padded, spec_lengths, speaker_emb)
        
        text_mask_b, spec_mask_b = text_mask.squeeze(1).bool(), spec_mask.squeeze(1).bool()
        
        # attention
        # text_encoded: (B, 192, TT), y_h: (B, 192, TS)
        # e: (B, TT), a: (B, TT), b: (B, TT)
        e, a, b = self.attn_op(text_encoded, y_h, text_mask_b, spec_mask_b, sigma=0.5)
        # x_align: (B, 192, TS), attns: (B, 2, TS, TT), real_sigma: (B)
        x_align, attns, real_sigma = self.attn(
            e, a, b, text_encoded, text_mask_b, spec_mask_b, max_length = self._cfg.attn_max_len
        )
        
        # zの要素からランダムにself.segment_size個取り出しz_sliceとする
        z_slice, ids_slice = rand_slice_segments(
            z2, spec_lengths, self._segment_size
        )
        wave_fake, _ = self.decoder(z_slice, speaker_id_embedded=None)
        
        # prior
        z1_r, m_q1, logs_q1, spec_mask = self.prior_nn1(x_align, spec_mask, g=speaker_emb)
        _, m_q2, logs_q2, spec_mask = self.prior_nn2(z1, spec_mask, g=speaker_emb)

        e_a = e - a # (B, TT)
        b_a = b - a # (B, TT)
        loss_a, loss_a_kl = self.vap(text_encoded, text_mask, e_a, b_a, g=speaker_emb)
        if bi:
            _, m_q2_r, logs_q2_r, spec_mask = self.prior_nn2(z1_r, spec_mask, g=speaker_emb)
            return (
                wave_fake,
                (loss_a, loss_a_kl),
                attns,
                ids_slice,
                text_mask,
                spec_mask,
                (m1, logs1, m_q1, logs_q1),
                (m2, logs2, m_q2, logs_q2),
                (m2, logs2, m_q2_r, logs_q2_r),
                real_sigma
            )
        return (
                wave_fake,
                (loss_a, loss_a_kl),
                attns,
                ids_slice,
                text_mask,
                spec_mask,
                (m1, logs1, m_q1, logs_q1),
                (m2, logs2, m_q2, logs_q2),
                None,
                real_sigma
            )
    def text_to_speech(
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