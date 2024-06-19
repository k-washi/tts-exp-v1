"""アクセント分割のVITS
入力形式: 音素, アクセント記号分割
"""

import torch
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from dataclasses import asdict

from src.models.vits.plmodule import ViTSModule
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

VERSION = "00001"
EXP_ID = "vits_accent"
WANDB_PROJECT_NAME = "vits-exp-v1"
FAST_DEV_RUN = False

cfg.ml.num_epochs = 5000
cfg.ml.max_steps = 1000000
cfg.ml.batch_size = 24
cfg.ml.val_batch_size = 12
cfg.ml.num_workers = 8
cfg.ml.accumulate_grad_batches = 1
cfg.ml.check_val_every_n_epoch = 1
cfg.ml.mix_precision = 32 # 16 or 32, bf16
cfg.ml.wav_save_every_n = 20 # 500個のテスト音声に対して1/10の50個を保存
cfg.ml.evaluator.speech_bert_score_model = "japanese-hubert-base" # 評価に用いるSSLモデル

cfg.dataset.add_blank_type = 0 # 0: なし, 1: 音素の前後に挿入, 2: モーラの前後に挿入
cfg.dataset.accent_split = True # アクセントを分割するか
cfg.dataset.accent_up_ignore = False # アクセント上昇を無視するか
cfg.dataset.use_distirubute_sampler = True # データセットの長さに応じてバッチを生成するか

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

cfg.model.scheduler_g.warmup_epoch = 10
cfg.model.scheduler_d.warmup_epoch = 10


def train():
    logger.info(f"Config: {cfg}")

    ################################
    # データセットとモデルの設定
    ################################
    dataset = TTSDataModule(cfg)
    cfg.dataset.train_dataset_num = dataset.train_dataset_num
    model = ViTSModule(cfg)
    
    ################################
    # コールバックなど訓練に必要な設定
    ################################
    wandb_logger = WandbLogger(name=f"EXP_ID_{VERSION}", project=WANDB_PROJECT_NAME, config=asdict(cfg), group=EXP_ID)
    wandb_logger.log_hyperparams(asdict(cfg))
    
    checkpoint_callback = CheckpointEveryEpoch(
        save_dir=cfg.path.model_save_dir,
    )
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval='epoch')]
    
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
            accumulate_grad_batches=cfg.ml.accumulate_grad_batches,
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