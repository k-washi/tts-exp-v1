import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from src.config.model.vits import netG
from src.config.dataset.default import DatasetConfig
from src.tts.phonome.param import (
    num_accents,
    num_phonomes
)

from src.models.core.components.text_encoder import TextEncoder, EncoderCFG
from src.models.core.components.posterior_encoder import PosteriorEncoder
from src.models.core.components.mbistft.modules import Multistream_iSTFT_Generator as Decoder
from src.models.core.components.flow import Flow, FlowLayerType
from src.models.core.components.stochastic_duration_predictor import StochasticDurationPredictor
from src.models.core.components.utils.commons import sequence_mask

noise_scale=0.667
length_scale=1
noise_scale_w=0.8

class TextEncoderOnnx(TextEncoder):
    def __init__(
        self,
        vocab_size,
        accent_position_size,  # 音素の種類数 # Accent Position embedding種類 # 音素の埋め込みベクトルDim
        input_embedding_dim=192,
        enccfg: EncoderCFG = EncoderCFG(),
    ) -> None:
        TextEncoder.__init__(self, vocab_size, accent_position_size, input_embedding_dim, enccfg)
        self.__input_embedding_dim = input_embedding_dim
    
    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, accent_pos_padded=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input index tensor (B, T_text).
            x_lengths (Tensor): Length tensor (B,).
        Returns:
            Tensor: Encoded hidden representation (B, attention_dim, T_text).
            Tensor: Projected mean tensor (B, attention_dim, T_text).
            Tensor: Projected scale tensor (B, attention_dim, T_text).
            Tensor: Mask tensor for input tensor (B, 1, T_text).
        """

        if self.accent_emb is not None:
            accent_emb = self.accent_emb(accent_pos_padded) # (B, text_length, input_embedding_dim)
            accent_emb = self.accent_proj(accent_emb.transpose(1, 2)).transpose(1, 2)
            x = (self.emb(x) + accent_emb) * math.sqrt(self.__input_embedding_dim)
        else:
            x = self.emb(x) * math.sqrt(self.__input_embedding_dim)
     
        x_mask = (
            sequence_mask(x_lengths).to(device=x.device, dtype=x.dtype)
        ).unsqueeze(1)
        x, _ = self.encoder(x, x_mask)

        # convert the channel first (B, attention_dim, T_text)
        x = x.transpose(1, 2)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)
        return x, m, logs, x_mask

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    padding_shape = [[0, 0], [1, 0], [0, 0]]
    padding = [item for sublist in padding_shape[::-1] for item in sublist]

    path = path - F.pad(path, padding)[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path

class MsISTFTInfer(nn.Module):
    def __init__(self,
                cfg: netG,
                data_cfg: DatasetConfig
        ):
        super(MsISTFTInfer, self).__init__()
        self._cfg = cfg
        self._n_mels = data_cfg.mel_bins
        self._segment_size = data_cfg.segment_size // data_cfg.hop_length
        self._n_speaker = data_cfg.n_speaker
        self._n_phonome = num_phonomes()
        self._n_accents = num_accents()
        
        self.text_encoder = TextEncoderOnnx(
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
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16],
            gen_istft_n_fft=16,
            gen_istft_hop_size=4,
            subbands=4,
            gin_channels=cfg.speaker_id_embedding_dim,
            convnext_hidden_dim_rate=3
        )
        self.flow = Flow(
                        speaker_id_embedding_dim=cfg.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                        in_z_channels=cfg.z_channels,#入力するzのchannel数
                        phoneme_embedding_dim=cfg.phoneme_embedding_dim,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                        n_resblocks=cfg.flow_n_resblocks,
                        layer_type=FlowLayerType.ResidualCouplingLayer
                    )
        self.stochastic_duration_predictor = StochasticDurationPredictor(
                        speaker_id_embedding_dim=cfg.speaker_id_embedding_dim,#話者idの埋め込み先のベクトルの大きさ
                        phoneme_embedding_dim=cfg.phoneme_embedding_dim,#TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                        filter_channels=cfg.duration_filter_channels,
                        kernel_size=cfg.duration_kernel_size,
                        p_dropout=cfg.duration_p_dropout,
                        n_flows=cfg.duration_n_flows
                    )
    
    #@torch.inference_mode()
    def forward(
        self,
        text_padded,
        accent_pos_padded,
        speaker_id,
    ):

        max_len=None
        text_lengths = (torch.ones(text_padded.size(0)) * text_padded.size(-1)).to(text_padded.device)
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(
            text_padded, text_lengths, accent_pos_padded
        )

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
        wav_fake, _ = self.decoder(
            (z * spec_mask)[:, :, :max_len], speaker_id_embedded=None
        )
        return wav_fake
    

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    from src.config.config import Config
    from src.tts.phonome.utils import extract_symbols
    from src.tts.phonome.param import (
        symbol_preprocess,
        phonome_to_sequence,
        accent_to_sequence
    )
    from src.tts.utils.audio import save_wave
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkopint", type=str, default="./checkpoints/vits_best_model.pth")
    parser.add_argument("--text", type=str, default="今日はいい天気ですね。")
    parser.add_argument("--output", type=str, default="./data/output.wav")
    
    args = parser.parse_args()
    checkpoint_path = args.checkopint
    text = args.text
    output = args.output
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    cfg = Config()
    vits = MsISTFTInfer(cfg.model.net_g, cfg.data)
    vits.load_state_dict(torch.load(checkpoint_path))
    
    symbol_list = extract_symbols(text)
    phonome_list, accent_list = symbol_preprocess(symbol_list, add_blank=True)
    phonome_indexes = phonome_to_sequence(phonome_list)
    accent_indexes = accent_to_sequence(accent_list)
    print(phonome_indexes)
    print(accent_indexes)
    
    phonome_indexes = torch.tensor(phonome_indexes).unsqueeze(0)
    accent_indexes = torch.tensor(accent_indexes).unsqueeze(0)
    sid = torch.tensor([0])
    text_lengths = torch.tensor([len(phonome_indexes[0])])
    
    print(phonome_indexes.shape, accent_indexes.shape, sid.shape, text_lengths.shape)
    w = vits(phonome_indexes, accent_indexes, sid)
    w = w.squeeze(1) * 0.5
    print(w.shape)
    
    w = w.cpu()
    save_wave(w, output, cfg.data.sample_rate)