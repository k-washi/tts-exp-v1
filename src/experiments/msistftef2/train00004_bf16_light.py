"""アクセント分割のVITS
入力形式: 音素, アクセント記号分割
MS-ISTFTを使った音声生成
ConvNextを使用して高速化
bf16で学習
"""

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dataclasses import asdict

from src.models.msistftef2.plmodule_cn_light import MsISTFTEF2Module
from src.dataset.pt.plmodule import TTSDataModule
from src.experiments.utils.pl_callbacks import CheckpointEveryEpoch


from src.utils.logger import get_logger
logger = get_logger(debug=True)

from src.config.config import Config, get_config
cfg:Config = get_config()

seed_everything(cfg.ml.seed)
##########
# PARAMS #
##########

VERSION = "00512"
EXP_ID = "msistft_ef2_accent_mblank_bf16_light"
WANDB_PROJECT_NAME = "vits-exp-v1"
IS_LOGGING = True
FAST_DEV_RUN = False

# PRETRAIN_GEN_MODEL = "logs/flytts_accent_mblank_noduradv_00310/ckpt/ckpt-4350/net_g.pth"
# PRETRAIN_D_MODEL = "logs/flytts_accent_mblank_noduradv_00310/ckpt/ckpt-4350/net_d.pth"

cfg.ml.num_epochs = 10000
cfg.ml.max_steps = 1000000
cfg.ml.batch_size = 32
cfg.ml.val_batch_size = 24
cfg.ml.num_workers = 8
cfg.ml.accumulate_grad_batches = 1
cfg.ml.grad_clip_val = 500
cfg.ml.check_val_every_n_epoch = 10
cfg.ml.early_stopping_patience = 500
cfg.ml.early_stopping_mode = "max"
cfg.ml.early_stopping_monitor = "val/speech_bert_score_f1"
cfg.ml.mix_precision = "bf16" # 16 or 32, bf16
cfg.ml.wav_save_every_n = 20 # 500個のテスト音声に対して1/10の50個を保存
cfg.ml.evaluator.speech_bert_score_model = "japanese-hubert-base" # 評価に用いるSSLモデル


cfg.dataset.add_blank_type = 2 # 0: なし, 1: 音素の前後に挿入, 2: モーラの前後に挿入
cfg.dataset.accent_split = True # アクセントを分割するか
cfg.dataset.accent_up_ignore = False # アクセント上昇を無視するか
cfg.dataset.use_distirubute_sampler = True # データセットの長さに応じてバッチを生成するか
cfg.dataset.segment_size = 16384
LOG_SAVE_DIR = f"logs/{EXP_ID}_{VERSION}"
cfg.path.model_save_dir = f"{LOG_SAVE_DIR}/ckpt"
cfg.path.val_out_dir = f"{LOG_SAVE_DIR}/val"

cfg.path.dataset_dir = "data/jsut"
cfg.path.train_file_path = f"{cfg.path.dataset_dir}/train.txt"
cfg.path.valid_file_path = f"{cfg.path.dataset_dir}/val.txt"

cfg.model.optim_g.lr = 2e-4
cfg.model.optim_g.eps = 1e-4
cfg.model.optim_d.lr = 2e-4
cfg.model.optim_d.eps = 1e-4
cfg.model.scheduler_g.use = True
cfg.model.scheduler_d.use = True

cfg.model.scheduler_g.warmup_epoch = 1
cfg.model.scheduler_d.warmup_epoch = 1

# VITS2
cfg.model.net_g.use_noise_scaled_mas = True
cfg.model.net_g.mas_nosie_scale_initial = 0.01
cfg.model.net_g.mas_noise_scale_delta = 5e-7
cfg.model.net_g.flow_n_resblocks = 4
cfg.model.wavlm_d.model = "rinna/japanese-hubert-large"
cfg.model.wavlm_d.hidden = 1024
cfg.model.wavlm_d.nlayers = 25

cfg.model.net_g.z_channels = 96
cfg.model.net_g.hidden_channels = 96
cfg.model.net_g.prior_nn_layers1 = 2
cfg.model.net_g.prior_nn_layers2 = 5
cfg.model.net_g.vap_layers = 2
cfg.model.net_g.phoneme_embedding_dim = cfg.model.net_g.hidden_channels
cfg.model.net_g.speaker_id_embedding_dim = cfg.model.net_g.hidden_channels
cfg.model.net_g.decoder_upsample_initial_channel = 384

def train():
    logger.info(f"Config: {cfg}")

    ################################
    # データセットとモデルの設定
    ################################
    dataset = TTSDataModule(cfg)
    cfg.dataset.train_dataset_num = dataset.train_dataset_num
    model = MsISTFTEF2Module(cfg)
    
    # model.net_g.load_state_dict(torch.load(PRETRAIN_GEN_MODEL))
    # model.net_d.load_state_dict(torch.load(PRETRAIN_D_MODEL))
    
    ################################
    # コールバックなど訓練に必要な設定
    ################################
    wandb_logger = None
    if IS_LOGGING:
        wandb_logger = WandbLogger(name=f"{EXP_ID}_{VERSION}", project=WANDB_PROJECT_NAME, config=asdict(cfg))
        wandb_logger.log_hyperparams(asdict(cfg))
    
    checkpoint_callback = CheckpointEveryEpoch(
        save_dir=cfg.path.model_save_dir,
        every_n_epochs=cfg.ml.check_val_every_n_epoch
    )
    callback_list = [
        checkpoint_callback, 
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor=cfg.ml.early_stopping_monitor,
            patience=cfg.ml.early_stopping_patience,
            mode=cfg.ml.early_stopping_mode
        )
    ]
    
    ################################
    # 訓練
    ################################
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
            precision=cfg.ml.mix_precision,
            accelerator=device,
            devices=cfg.ml.gpu_devices,
            max_epochs=cfg.ml.num_epochs,
            max_steps=cfg.ml.max_steps,
            #accumulate_grad_batches=cfg.ml.accumulate_grad_batches,
            profiler=cfg.ml.profiler,
            fast_dev_run=FAST_DEV_RUN,
            check_val_every_n_epoch=cfg.ml.check_val_every_n_epoch,
            logger=wandb_logger,
            callbacks=callback_list,
            num_sanity_val_steps=2
        )
    logger.debug("START TRAIN")
    trainer.fit(model, dataset)

if __name__ == "__main__":
    train()