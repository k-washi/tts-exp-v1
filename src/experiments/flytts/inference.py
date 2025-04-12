
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

from src.models.flytts.plmodule_noduradv import ViTSModule

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
G_MODEL = "logs/flytts_accent_mblank_noduradv_00310/ckpt/ckpt-4350/net_g.pth"

valid_fp = "data/jsut/val.txt"
with open(valid_fp, 'r', encoding='utf-8') as f:
    valid_namelist = f.read().splitlines()
valid_namelist = valid_namelist[:100]
    
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

model = ViTSModule(cfg=cfg)


# GPU inference
val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                num_workers=2,
                batch_size=1,
                collate_fn=collate_fn,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )

# CPU inference
device = torch.device("cuda")
model.net_g.to(device, dtype=m_dtype)
model.net_g.eval()

# real time factor = 1秒の音声を処理するのにかかる時間
gpu_created_time = 0
gpu_inference_time = 0
gpu_tf_time = 0
count = 0
for i, batch in tqdm(enumerate(val_dataloader)):
    with torch.no_grad():
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
            
            torch.cuda.synchronize()
            s = time.time()
            text_padded = text_padded.to(device)
            text_lengths = text_lengths.to(device)
            accent_pos_padded = accent_pos_padded.to(device)
            speaker_id = speaker_id.to(device)
            
            torch.cuda.synchronize()
            elapsed = time.time() - s
            gpu_tf_time += elapsed
            count += 1
            
            torch.cuda.synchronize()
            s = time.time()
            fake_wave = model.net_g.text_to_speech(
                    text_padded,
                    text_lengths,
                    accent_pos_padded,
                    speaker_id
                )
            if i < 50:
                continue
            torch.cuda.synchronize()
            elapsed = time.time() - s
            # 22050Hzの出力
            ct = fake_wave.shape[-1] / SR
            gpu_created_time += ct
            gpu_inference_time += elapsed
            logger.info(f"GPU TF: {gpu_tf_time/count:.4f}")
            logger.info(f"RTF: {gpu_inference_time/gpu_created_time:.4f}")

print(f"GPU inference time: {gpu_inference_time:.8f} sec")
print(f"GPU created time: {gpu_created_time:.8f} sec")
print(f"GPU RTF: {gpu_inference_time/gpu_created_time :.8f}")
print(f"GPU TF: {gpu_tf_time/count:.8f} sec")
print(f"GPU TF count: {count}")

param_num = sum(p.numel() for p in [
    *model.net_g.text_encoder.parameters(),
    *model.net_g.decoder.parameters(),
    *model.net_g.flow.parameters(),
    *model.net_g.stochastic_duration_predictor.parameters(),
])
print(f"Param Num: {param_num / 1e6:.2f}M")