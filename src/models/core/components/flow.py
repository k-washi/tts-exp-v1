import torch
import torch.nn as nn

from enum import Enum
from src.models.core.components.wn import WN
import src.models.core.components.attentions as attentions


# xをchannel方向に沿って要素を逆順に変換
class Flip(nn.Module):
    def forward(self, x, *args, **kwargs):
        x = torch.flip(x, [1])
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        in_channels=192,  # 入力Tensorのchannel数
        phoneme_embedding_dim=192,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        speaker_id_embedding_dim=0,  # 話者idの埋め込み先のベクトルの大きさ
        kernel_size=5,  # WN内のconv1dのカーネルサイズ
        dilation_rate=1,  # WN内のconv1dのdilationを決めるための数値
        n_resblocks=4,
    ):  # WN内で、ResidualBlockをいくつ重ねるか
        assert in_channels % 2 == 0, "in_channels should be divisible by 2"
        super().__init__()
        self.in_channels = in_channels  # 入力Tensorのchannel数
        self.half_channels = in_channels // 2  # 入力Tensorをchannelの次元に沿って半分ずつに分割するのに用いる
        self.phoneme_embedding_dim = phoneme_embedding_dim  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        self.kernel_size = kernel_size  # WN内のconv1dのカーネルサイズ
        self.dilation_rate = dilation_rate  # WN内のconv1dのdilationを決めるための数値
        self.n_resblocks = n_resblocks  # WN内で、ResidualBlockをいくつ重ねるか

        # 最初に適用するネットワーク
        self.pre = nn.Conv1d(self.half_channels, self.phoneme_embedding_dim, 1)
        # WNを用いて特徴量の抽出を行う　WNの詳細はwn.py参照
        self.wn = WN(
            self.phoneme_embedding_dim,
            self.kernel_size,
            self.dilation_rate,
            self.n_resblocks,
            speaker_id_embedding_dim=speaker_id_embedding_dim,
        )
        # 平均値を生成するネットワーク
        self.post = nn.Conv1d(self.phoneme_embedding_dim, self.half_channels, 1)
        # 初期化処理
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, speaker_id_embedded, reverse):
        # 入力Tensorであるxをchannelの次元に沿って半分ずつに分割、それぞれx0、x1とする
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        # x0に対しconv1dを適用
        h = self.pre(x0) * x_mask
        # WNを用いて特徴量の抽出を行う
        h = self.wn(h, x_mask, speaker_id_embedded=speaker_id_embedded)
        # 抽出した特徴量からx1の平均値をどれだけずらすか決める値を生成する
        x1_mean = self.post(h) * x_mask

        # 順伝搬時
        if not reverse:
            # x0から生成された平均値を用いてx1の平均をずらす
            x1 = x1_mean + x1 * x_mask
        # 逆伝搬時
        else:
            # x0から生成された平均値を用いてx1の平均をずらす
            x1 = (x1 - x1_mean) * x_mask

        # 2つをchannelの次元に沿って結合
        x = torch.cat([x0, x1], 1)
        return x

class ResidualCouplingTransformerLayer(nn.Module):
    def __init__(
        self,
        in_channels=192,  # 入力Tensorのchannel数
        phoneme_embedding_dim=192,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        speaker_id_embedding_dim=0,  # 話者idの埋め込み先のベクトルの大きさ
        kernel_size=5,  # WN内のconv1dのカーネルサイズ
        dilation_rate=1,  # WN内のconv1dのdilationを決めるための数値
        n_resblocks=2,
        p_dropout=0.0,
    ):  # WN内で、ResidualBlockをいくつ重ねるか
        assert in_channels % 2 == 0, "in_channels should be divisible by 2"
        super().__init__()
        self.in_channels = in_channels  # 入力Tensorのchannel数
        self.half_channels = in_channels // 2  # 入力Tensorをchannelの次元に沿って半分ずつに分割するのに用いる
        self.phoneme_embedding_dim = phoneme_embedding_dim  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        self.kernel_size = kernel_size  # WN内のconv1dのカーネルサイズ
        self.dilation_rate = dilation_rate  # WN内のconv1dのdilationを決めるための数値
        self.n_resblocks = n_resblocks  # WN内で、ResidualBlockをいくつ重ねるか

        # 最初に適用するネットワーク
        # https://github.com/p0p4k/vits2_pytorch/blob/1f4f3790568180f8dec4419d5cad5d0877b034bb/attentions.py#L559
        # g_inの処理を将来的に導入する

        self.pre_transformer = attentions.Encoder(
            hidden_channels=self.half_channels,
            filter_channels=self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=p_dropout,
            window_size=None
        )
        
        self.pre = nn.Conv1d(self.half_channels, self.phoneme_embedding_dim, 1)
        # WNを用いて特徴量の抽出を行う　WNの詳細はwn.py参照
        self.wn = WN(
            self.phoneme_embedding_dim,
            self.kernel_size,
            self.dilation_rate,
            self.n_resblocks,
            speaker_id_embedding_dim=speaker_id_embedding_dim,
        )
        #self.post_transformer = Encoder(
        #    attention_dim=self.phoneme_embedding_dim,
        #    attention_heads=2,
        #    linear_units=self.phoneme_embedding_dim,
        #    num_blocks=2,
        #    dropout_rate=p_dropout,
        #    normalize_before=True,
        #    positionwise_layer_type="conv1d",
        #    positionwise_conv_kernel_size=3,
        #    macaron_style=False,
        #    pos_enc_layer_type="rel_pos",
        #    selfattention_layer_type="rel_selfattn",
        #    activation_type="swish",
        #    use_cnn_module=False,
        #)
        # 平均値を生成するネットワーク
        self.post = nn.Conv1d(self.phoneme_embedding_dim, self.half_channels, 1)
        # 初期化処理
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, speaker_id_embedded, reverse):
        # 入力Tensorであるxをchannelの次元に沿って半分ずつに分割、それぞれx0、x1とする
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        
        x0 = self.pre_transformer(x0 * x_mask, x_mask) + x0
        # x0に対しconv1dを適用
        h = self.pre(x0) * x_mask
        # WNを用いて特徴量の抽出を行う
        h = self.wn(h, x_mask, speaker_id_embedded=speaker_id_embedded)
        # 抽出した特徴量からx1の平均値をどれだけずらすか決める値を生成する
        # h = self.post_transformer((h * x_mask).transpose(1, 2), x_mask)[0].transpose(1, 2) + h
        x1_mean = self.post(h) * x_mask

        # 順伝搬時
        if not reverse:
            # x0から生成された平均値を用いてx1の平均をずらす
            x1 = x1_mean + x1 * x_mask
        # 逆伝搬時
        else:
            # x0から生成された平均値を用いてx1の平均をずらす
            x1 = (x1 - x1_mean) * x_mask

        # 2つをchannelの次元に沿って結合
        x = torch.cat([x0, x1], 1)
        return x

class FFTransformerCouplingLayer(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        n_heads,
        p_dropout=0,
        filter_channels=768,
        mean_only=False,
        gin_channels=0,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            isflow=True,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, speaker_id_embedded=None, reverse=False):
        g = speaker_id_embedded
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h_ = self.enc(h, x_mask, g=g)
        h = h_ + h
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            #logdet = torch.sum(logs, [1, 2]
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
        return x

# Flowとは入出力が可逆なネットワーク
# zと埋め込み済み話者idを入力にとり、Monotonic Alignment Searchで用いる変数z_pを出力するネットワーク　逆も可能
class FlowLayerType(Enum):
    ResidualCouplingLayer = 0
    ResidualCouplingTransformerLayer = 1
    FFTransformerCouplingLayer = 2
    FFTransformerCouplingLayer2 = 3
    
class Flow(nn.Module):
    def __init__(
        self,
        speaker_id_embedding_dim=256,  # 話者idの埋め込み先のベクトルの大きさ
        in_z_channels=192,  # Flowに入力するzのchannel数
        phoneme_embedding_dim=192,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        n_flows=4,  # Flowを構成するf(x)をいくつ重ねるか
        kernel_size=5,  # WN内のconv1dのカーネルサイズ
        dilation_rate=1,  # WN内のconv1dのdilationを決めるための数値
        n_resblocks=4,
        layer_type=FlowLayerType.ResidualCouplingLayer,
    ):  # WN内で、ResidualBlockをいくつ重ねるか
        super().__init__()

        self.speaker_id_embedding_dim = speaker_id_embedding_dim  # 話者idの埋め込み先のベクトルの大きさ
        self.in_z_channels = in_z_channels  # 入力するzのchannel数
        self.phoneme_embedding_dim = phoneme_embedding_dim  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
        self.n_flows = n_flows  # Flowを構成するf(x)をいくつ重ねるか
        self.kernel_size = kernel_size  # WN内のconv1dのカーネルサイズ
        self.dilation_rate = dilation_rate  # WN内のconv1dのdilationを決めるための数値
        self.n_resblocks = n_resblocks  # WN内で、ResidualBlockをいくつ重ねるか

        # 各f(x)を生成、self.flowsに格納
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            if layer_type == FlowLayerType.ResidualCouplingLayer:
                self.flows.append(
                    ResidualCouplingLayer(
                        in_channels=self.in_z_channels,  # Flowに入力するzのchannel数
                        phoneme_embedding_dim=self.phoneme_embedding_dim,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                        speaker_id_embedding_dim=self.speaker_id_embedding_dim,  # 話者idの埋め込み先のベクトルの大きさ
                        kernel_size=self.kernel_size,  # WN内のconv1dのカーネルサイズ
                        dilation_rate=self.dilation_rate,  # WN内のconv1dのdilationを決めるための数値
                        n_resblocks=self.n_resblocks,  # WN内で、ResidualBlockをいくつ重ねるか
                    )
                )
            elif layer_type == FlowLayerType.ResidualCouplingTransformerLayer:
                self.flows.append(
                    ResidualCouplingTransformerLayer(
                        in_channels=self.in_z_channels,  # Flowに入力するzのchannel数
                        phoneme_embedding_dim=self.phoneme_embedding_dim,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                        speaker_id_embedding_dim=self.speaker_id_embedding_dim,  # 話者idの埋め込み先のベクトルの大きさ
                        kernel_size=self.kernel_size,  # WN内のconv1dのカーネルサイズ
                        dilation_rate=self.dilation_rate,  # WN内のconv1dのdilationを決めるための数値
                        n_resblocks=self.n_resblocks,  # WN内で、ResidualBlockをいくつ重ねるか
                        p_dropout=0.1
                    )
                )
            elif layer_type == FlowLayerType.FFTransformerCouplingLayer:
                self.flows.append(
                    FFTransformerCouplingLayer(
                        channels=self.in_z_channels,  # Flowに入力するzのchannel数
                        hidden_channels=self.phoneme_embedding_dim,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                        kernel_size=self.kernel_size,  # WN内のconv1dのカーネルサイズ
                        n_layers=1,  # WN内で、ResidualBlockをいくつ重ねるか
                        n_heads=4,
                        p_dropout=0.1,
                        filter_channels=768,
                        mean_only=True,
                        gin_channels=speaker_id_embedding_dim,
                    )
                )
            elif layer_type == FlowLayerType.FFTransformerCouplingLayer2:
                self.flows.append(
                    FFTransformerCouplingLayer(
                        channels=self.in_z_channels,  # Flowに入力するzのchannel数
                        hidden_channels=self.phoneme_embedding_dim,  # TextEncoderで作成した、埋め込み済み音素のベクトルの大きさ
                        kernel_size=self.kernel_size,  # WN内のconv1dのカーネルサイズ
                        n_layers=3,  # WN内で、ResidualBlockをいくつ重ねるか
                        n_heads=2,
                        p_dropout=0.1,
                        filter_channels=768,
                        mean_only=True,
                        gin_channels=speaker_id_embedding_dim,
                    )
                )
            else:
                raise ValueError(f"Invalid layer_type: {layer_type}")
            self.flows.append(Flip())

    def forward(self, z, z_mask, speaker_id_embedded, reverse=False):
        z_p = z
        z_p_mask = z_mask
        # 順方向
        if not reverse:
            for flow in self.flows:
                z_p = flow(z_p, z_p_mask, speaker_id_embedded=speaker_id_embedded, reverse=reverse)
        # 逆方向
        else:
            for flow in reversed(self.flows):
                z_p = flow(z_p, z_p_mask, speaker_id_embedded=speaker_id_embedded, reverse=reverse)
        return z_p