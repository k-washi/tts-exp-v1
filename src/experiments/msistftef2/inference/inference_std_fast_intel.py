
import torch
from pathlib import Path
import time
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dataclasses import asdict
from src.dataset.pt.dataset import AudioTextDataset
from tqdm import tqdm
from src.dataset.pt.sampler import (
    collate_fn,
)

import intel_extension_for_pytorch as ipex

from src.models.msistftef2.generator_cn_inference_rmwn import Generator

from src.utils.logger import get_logger
logger = get_logger(debug=True)

from src.config.config import Config, get_config
cfg:Config = get_config()

seed_everything(cfg.ml.seed)

cfg.path.dataset_dir = "data/jsut"
cfg.dataset.add_blank_type = 2 # 0: なし, 1: 音素の前後に挿入, 2: モーラの前後に挿入
cfg.dataset.accent_split = True # アクセントを分割するか
cfg.dataset.accent_up_ignore = False # アクセント上昇を無視するか
cfg.ml.mix_precision = 32 # 16 or 32, bf16
cfg.model.net_g.flow_n_resblocks = 4


SR = 22050
m_dtype = torch.float32
G_MODEL = "logs/msistft_ef2_accent_mblank_00508/ckpt/ckpt-3350/net_g.pth"

valid_fp = "data/jsut/val.txt"
with open(valid_fp, 'r', encoding='utf-8') as f:
    valid_namelist = f.read().splitlines()
valid_namelist = valid_namelist[:20]
    
val_dataset = AudioTextDataset(
    cfg=cfg,
    data_list=valid_namelist,
)
val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                num_workers=2,
                batch_size=1,
                collate_fn=collate_fn,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )

model = Generator(cfg=cfg.model.net_g, data_cfg=cfg.dataset)
model.load_state_dict(torch.load(G_MODEL, map_location="cpu"), strict=False)
for p in model.parameters():
    p.requires_grad = False
# CPU inference
device = torch.device("cpu")
model.to(device, dtype=m_dtype)
model.eval()

param_num = sum(p.numel() for p in [
    *model.text_encoder.parameters(),
    *model.decoder.parameters(),
    *model.vap.parameters(),
    *model.prior_nn1.parameters(),
    *model.prior_nn2.parameters(),
])
print(f"Param Num: {param_num / 1e6:.2f}M")

model = ipex.optimize(model, dtype=torch.float32)
#model = torch.compile(model, backend="ipex")


# real time factor = 1秒の音声を処理するのにかかる時間
cpu_created_time = 0
cpu_inference_time = 0
for i, batch in tqdm(enumerate(val_dataloader)):
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=m_dtype):
            # batch = {k: v.to(device, dtype=m_dtype) for k, v in batch.items()}
            # batch = {k: v.to(device) for k, v in batch.items()}
            # print(batch)
            (
                real_wave_padded,
                wav_lengths,
                _,
                _,
                text_padded,
                text_lengths,
                accent_pos_padded,
                speaker_id,
            ) = batch
            
        
            s = time.time()
            fake_wave, _, _ = model(
                    text_padded,
                    text_lengths,
                    accent_pos_padded,
                    speaker_id
                )
            if i < 10:
                continue
            elapsed = time.time() - s
            # 22050Hzの出力
            ct = fake_wave.shape[-1] / SR
            cpu_created_time += ct
            cpu_inference_time += elapsed

logger.info(f"CPU inference time: {cpu_inference_time:.4f} sec")
logger.info(f"CPU created time: {cpu_created_time:.4f} sec")
logger.info(f"CPU RTF: {cpu_inference_time/cpu_created_time:.4f}")

