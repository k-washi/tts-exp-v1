from pathlib import Path
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from pytorch_lightning import LightningDataModule

from src.config.config import Config
from src.dataset.pt.sampler import (
    collate_fn,
    DistributedBucketSampler
)
from src.dataset.pt.dataset import AudioTextDataset


class TTSDataModule(LightningDataModule):
    def __init__(self, cfg: Config):
        super(TTSDataModule, self).__init__()
        
        with open(cfg.path.train_file_path, 'r', encoding='utf-8') as f:
            self.train_namelist = f.read().splitlines()
        with open(cfg.path.valid_file_path, 'r', encoding='utf-8') as f:
            self.valid_namelist = f.read().splitlines()
        
        self.train_dataset_num = len(self.train_namelist)
        self.cfg = cfg
        self.batch_size = cfg.ml.batch_size
        self.num_workers = cfg.ml.num_workers
    
    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = AudioTextDataset(
                self.cfg, self.train_namelist
            )
            self.val_dataset = AudioTextDataset(
                self.cfg, self.valid_namelist
            )
        
        else:
            raise RuntimeError(f'Invalid stage: {stage}')
    
    def train_dataloader(self):
        if self.cfg.dataset.use_distirubute_sampler:
            # [0.3, 5, 10, 16],:wavの長さを左の長さに沿って組みを組んでからBatchを生成
            # batch内のPadを減らす
            train_sampler = DistributedBucketSampler(
                self.train_dataset,
                self.batch_size,
                [0.5, 3, 5, 7, 9, 12, 16],
                num_replicas=1,
                rank=0,
                shuffle=True,
            )
            return torch.utils.data.DataLoader(
                self.train_dataset,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=False,
                batch_sampler=train_sampler
            )
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True
            
        )
    
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.val_dataset,
                    num_workers=self.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    shuffle=False,
                    batch_size=self.cfg.ml.val_batch_size,
                    drop_last=False
                )