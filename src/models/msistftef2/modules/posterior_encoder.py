import torch
import torch.nn as nn

from src.models.msistftef2.modules.modules import WN
from src.models.core.components.utils.commons import sequence_mask

# linear spectrogramを入力にとりEncodeを実行、zを出力するモデル
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        speaker_id_embedding_dim,  # 話者idの埋め込み先のベクトルの大きさ
        in_spec_channels=513,  # 入力する線形スペクトログラムの縦軸(周波数)の次元
        out_z_channels=192,  # 出力するzのchannel数
        phoneme_embedding_dim=192,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        kernel_size=5,  # WN内のconv1dのカーネルサイズ
        dilation_rate=1,  # WN内のconv1dのdilationを決めるための数値
        n_resblocks=16,  # WN内で、ResidualBlockをいくつ重ねるか
    ):
        super(PosteriorEncoder, self).__init__()

        self.speaker_id_embedding_dim = speaker_id_embedding_dim  # 話者idの埋め込み先のベクトルの大きさ
        self.in_spec_channels = in_spec_channels  # 入力する線形スペクトログラムの縦軸(周波数)の次元
        self.out_z_channels = out_z_channels  # 出力するzのchannel数
        self.phoneme_embedding_dim = phoneme_embedding_dim  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        self.kernel_size = kernel_size  # WN内のconv1dのカーネルサイズ
        self.dilation_rate = dilation_rate  # WN内のconv1dのdilationを決めるための数値
        self.n_resblocks = n_resblocks  # WN内で、ResidualBlockをいくつ重ねるか

        # 入力スペクトログラムに対し前処理を行うネットワーク
        self.preprocess = nn.Conv1d(self.in_spec_channels, self.phoneme_embedding_dim, 1)
        # WNを用いて特徴量の抽出を行う　WNの詳細はwn.py参照
        # speaker_id_embedding_dim = 0
        self.wn = WN(
            self.phoneme_embedding_dim,
            self.kernel_size,
            self.dilation_rate,
            self.n_resblocks,
            gin_channels=self.speaker_id_embedding_dim
        )
        # ガウス分布の平均と分散を生成するネットワーク
        self.projection_h = nn.Conv1d(self.phoneme_embedding_dim, self.out_z_channels, 1)
        self.projection = nn.Conv1d(self.phoneme_embedding_dim, self.out_z_channels * 4, 1)

    def forward(self, spectrogram, spectrogram_lengths, speaker_id_embedded=None, t1=1.0, t2=1.0):
        spectrogram_mask = torch.unsqueeze(sequence_mask(spectrogram_lengths, spectrogram.size(2)), 1).to(
            spectrogram.dtype
        )
        # 入力スペクトログラムに対しConvを用いて前処理を行う
        x = self.preprocess(spectrogram) * spectrogram_mask
        # WNを用いて特徴量の抽出を行う
        x = self.wn(x, spectrogram_mask, speaker_id_embedded=speaker_id_embedded)
        # 出力された特徴マップをもとに統計量を生成
        y_h = self.projection_h(x) * spectrogram_mask
        z_stats = self.projection(x) * spectrogram_mask
        m1, logs1, m2, logs2 = torch.split(z_stats, self.out_z_channels, dim=1)
        z1 = torch.distributions.Normal(m1, torch.exp(logs1)*t1).rsample()*spectrogram_mask
        z2 = torch.distributions.Normal(m2, torch.exp(logs2)*t2).rsample()*spectrogram_mask
        return y_h, z1, z2, m1, logs1, m2, logs2, spectrogram_mask