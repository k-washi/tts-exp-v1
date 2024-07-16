import math

import torch
import torch.nn as nn

from src.config.model.vits import netG
from src.config.dataset.default import DatasetConfig
from src.tts.phonome.param import (
    num_accents,
    num_phonomes
)

from src.models.core.components.text_encoder import TextEncoder, EncoderCFG
from src.models.core.components.posterior_encoder import PosteriorEncoder
from src.models.core.components.decode import Decoder
from src.models.core.components.flow import Flow, FlowLayerType
from src.models.core.components.stochastic_duration_predictor import StochasticDurationPredictor
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
                        speaker_id_embedding_dim=cfg.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                        in_z_channel=cfg.z_channels #入力するzのchannel数
                        )
        if cfg.flow_layer_type == "FFTransformerCouplingLayer":
            flow_layer_type = FlowLayerType.FFTransformerCouplingLayer
        elif cfg.flow_layer_type == "FFTransformerCouplingLayer2":
            flow_layer_type = FlowLayerType.FFTransformerCouplingLayer2
        else:
            raise ValueError(f"Invalid flow_layer_type: {cfg.flow_layer_type}")
        self.flow = Flow(
                      speaker_id_embedding_dim=cfg.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                      in_z_channels=cfg.z_channels,#入力するzのchannel数
                      phoneme_embedding_dim=cfg.phoneme_embedding_dim,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                      layer_type=flow_layer_type,
                      n_resblocks=cfg.flow_n_resblocks
                    )
        self.stochastic_duration_predictor = StochasticDurationPredictor(
                      speaker_id_embedding_dim=cfg.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                      phoneme_embedding_dim=cfg.phoneme_embedding_dim,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                      filter_channels=cfg.duration_filter_channels,
                      kernel_size=cfg.duration_kernel_size,
                      p_dropout=cfg.duration_p_dropout,
                      n_flows=cfg.duration_n_flows
                    )
    
    def forward(
        self,
        text_padded,
        text_lengths,
        accent_pos_padded,
        spec_padded,
        spec_lengths,
        speaker_id=0,
        speaker_emb=None
    ):
        # text(音素)の内容をTextEncoderに通す
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(
            text_padded, text_lengths, accent_pos_padded
        )
        
        # 話者埋め込み
        if speaker_emb is None:
            #print(speaker_id.shape)
            speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(-1)
        
        #linear spectrogramと埋め込み済み話者idを入力にとりEncodeを実行、zを出力する
        z, m_q, logs_q, spec_mask = self.posterior_encoder(spec_padded, spec_lengths, speaker_emb)
        #zと埋め込み済み話者idを入力にとり、Monotonic Alignment Searchで用いる変数z_pを出力する
        z_p = self.flow(z, spec_mask, speaker_id_embedded=speaker_emb)
        # Monotonic Alignment Search(MAS)の実行　音素の情報と音声の情報を関連付ける役割を果たす
        # MASによって、尤度を最大にするようなpathを求める
        with torch.no_grad():
            # DPで用いる、各ノードの尤度を前計算しておく
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            
            # noise scaled mas
            if self._use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon
            # 不要なノードにマスクをかけた上でDPを実行
            MAS_node_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(
                spec_mask, -1
            )
            MAS_path = (
                monotonic_align.maximum_path(neg_cent, MAS_node_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )
        # text(音素)の各要素ごとに、音素長を計算(各音素長は整数)
        duration_of_each_phoneme = MAS_path.sum(2)
        # StochasticDurationPredictorを、音素列の情報から音素継続長を予測できるよう学習させる
        l_length = self.stochastic_duration_predictor(
                text_encoded,
                text_mask,
                duration_of_each_phoneme,
                speaker_id_embedded=None,
            )
        l_length = l_length / torch.sum(text_mask)
        logw = self.stochastic_duration_predictor(
            text_encoded,
            text_mask,
            speaker_id_embedded=None,
            reverse=True,
        )
        logw_ = torch.log(duration_of_each_phoneme + 1e-6) * text_mask
        
        
        m_p = torch.matmul(MAS_path.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )
        logs_p = torch.matmul(
            MAS_path.squeeze(1), logs_p.transpose(1, 2)
        ).transpose(1, 2)

        # zの要素からランダムにself.segment_size個取り出しz_sliceとする
        z_slice, ids_slice = rand_slice_segments(
            z, spec_lengths, self._segment_size
        )
        # z_sliceから音声波形を生成
        wav_fake = self.decoder(z_slice, speaker_id_embedded=None)

        return (
            wav_fake,
            l_length,
            ids_slice,
            text_mask,
            spec_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (text_encoded, logw, logw_)
        )
    def text_to_speech(
        self,
        text_padded,
        text_lengths,
        accent_pos_padded,
        speaker_id,
        speaker_emb=None,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
    ):
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(
            text_padded, text_lengths, accent_pos_padded
        )
        # 話者埋め込み
        if speaker_emb is None:
            speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(-1)

        logw = self.stochastic_duration_predictor(
            text_encoded,
            text_mask,
            speaker_id_embedded=None,
            reverse=True,
            noise_scale=noise_scale_w,
        )

        w = torch.exp(logw) * text_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        spec_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(
            text_mask.dtype
        )
        MAS_node_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(
            spec_mask, -1
        )
        MAS_path = generate_path(w_ceil, MAS_node_mask)

        m_p = torch.matmul(MAS_path.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )
        logs_p = torch.matmul(
            MAS_path.squeeze(1), logs_p.transpose(1, 2)
        ).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(
            z_p, spec_mask, speaker_id_embedded=speaker_emb, reverse=True
        )
        wav_fake = self.decoder(
            (z * spec_mask)[:, :, :max_len], speaker_id_embedded=None
        )
        return wav_fake
    
    def update_current_mas_noise_scale(self):
        if self._use_noise_scaled_mas:
            self.current_mas_noise_scale = max(0, self.current_mas_noise_scale - self._mas_noise_scale_delta)
            