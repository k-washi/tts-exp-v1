from dataclasses import dataclass, field
from typing import List

@dataclass()
class netG():
    phoneme_embedding_dim:int = 192 #各音素の埋め込み先のベクトルの大きさ
    spec_channels:int = 513 #入力する線形スペクトログラムの縦軸(周波数)の次元
    z_channels:int = 192 #PosteriorEncoderから出力されるzのchannel数 (phonome_embedding_dimと合わせる)
    flow_n_resblocks:int = 4
    flow_layer_type:str = "FFTransformerCouplingLayer"
    speaker_id_embedding_dim:int = 128 #話者idの埋め込み先のベクトルの大きさ
    text_encoders_dropout_during_train:float = 0.1 #学習時のtext_encoderのdropoutの割合
    text_n_heads:int = 2 #self.encoder内の、transformerに似た構造のモジュールで使われている、MultiHeadAttentionのhead数
    text_n_layers:int = 6 #self.encoder内の、transformerに似た構造のモジュールをいくつ重ねるか
    text_kernel_size:int = 3 #self.encoder内の、transformerに似た構造のモジュールにあるFeedForwardNetworkのカーネルサイズ
    text_filter_channels:int = 768 #self.encoder内の、transformerに似た構造のモジュールにあるFeedForwardNetworkの隠れ層のチャネル数
    duration_filter_channels:int = 192
    duration_kernel_size:int = 3
    duration_p_dropout:float = 0.5
    duration_n_flows:int = 4
    use_noise_scaled_mas: bool = False
    mas_nosie_scale_initial: float = 0.01
    mas_noise_scale_delta: float = 2e-6

@dataclass
class netD():
    n_D_update_steps:int = 2
    
@dataclass
class OptimizerG():
    name: str = "AdamW"
    lr: float = 2e-4
    eps: float = 1e-9
    betas: List = field(default_factory=lambda: [0.9, 0.999])

@dataclass
class OptimizerD():
    name: str = "AdamW"
    lr: float = 2e-4
    eps: float = 1e-9
    betas: List= field(default_factory=lambda: [0.9, 0.999])

@dataclass
class SchedulerG():
    use: bool = True
    name: str = "linear_w_warmup"
    gamma: float = 0.999998
    interval: str = "step"
    warmup_epoch: int = 10
    

@dataclass
class SchedulerD():
    use: bool = True
    name: str = "linear_w_warmup"
    gamma: float = 0.999998
    interval: str = "step"
    warmup_epoch: int = 10

@dataclass
class GAdvLossConfig():
    average_by_discriminators: bool = False
    loss_type: str = "mse"

@dataclass
class DAdvLossConfig():
    average_by_discriminators: bool = False
    loss_type: str = "mse"

@dataclass
class WavLMAdvLossConfig():
    model:str = "microsoft/wavlm-base-plus"
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64
    sr: int = 16000
@dataclass
class FeatureMatchLossConfig():
    average_by_discriminators: bool = False
    average_by_layers: bool = False
    include_final_outputs: bool = True

@dataclass
class Loss():
    sisnr_loss_use: bool = False
    mel_loss_lambda: float = 45.0
    duration_loss_lambda: float = 1.0
    kl_loss_lambda: float = 1.0
    feature_loss_loss_lambda: float =  1.0
    adversarial_loss_G_lambda: float = 1.0
    adversarial_loss_D_lambda: float = 1.0
    sisnr_loss_lambda: float = 1.0
    g_adv_loss: GAdvLossConfig = field(default_factory=lambda:GAdvLossConfig())
    d_adv_loss: DAdvLossConfig = field(default_factory=lambda:DAdvLossConfig())
    feat_match_loss: FeatureMatchLossConfig = field(default_factory=lambda:FeatureMatchLossConfig())
    

@dataclass()
class ModelConfig:
    net_g:netG = field(default_factory=lambda:netG())
    net_d:netD = field(default_factory=lambda:netD())
    optim_g:OptimizerG = field(default_factory=lambda:OptimizerG())
    optim_d:OptimizerD = field(default_factory=lambda:OptimizerD())
    scheduler_g:SchedulerG = field(default_factory=lambda:SchedulerG())
    scheduler_d:SchedulerD = field(default_factory=lambda:SchedulerD())
    loss: Loss = field(default_factory=lambda:Loss())
    wavlm_d: WavLMAdvLossConfig = field(default_factory=lambda:WavLMAdvLossConfig())
    