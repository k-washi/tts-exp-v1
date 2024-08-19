import torch
import pytorch_lightning as pl
from pathlib import Path

class CheckpointEveryEpoch(pl.Callback):
    def __init__(self, save_dir, start_epoch=0, every_n_epochs=1):
        self.start_epoch = start_epoch
        self.save_dir = save_dir
        self.epochs = 0
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        self.epochs += 1
        if self.epochs == 1 or self.epochs >= self.start_epoch and self.epochs % self.every_n_epochs == 0:
            save_dir = Path(f"{self.save_dir}") / f"ckpt-{self.epochs}"
            save_dir.mkdir(exist_ok=True, parents=True)
            net_g_save_path = save_dir / "net_g.pth"
            net_d_save_path = save_dir / "net_d.pth"
            net_wavlm_d_save_path = save_dir / "wavlm_d.pth"
            
            if hasattr(trainer, "model"):
                if hasattr(trainer.model, "module"):
                    torch.save(trainer.model.module.net_g.state_dict(), net_g_save_path)
                    torch.save(trainer.model.module.net_d.state_dict(), net_d_save_path)
                    torch.save(trainer.model.module.wavlm_d.state_dict(), net_wavlm_d_save_path)
                else:
                    torch.save(trainer.model.net_g.state_dict(), net_g_save_path)
                    torch.save(trainer.model.net_d.state_dict(), net_d_save_path)
                    torch.save(trainer.model.wavlm_d.state_dict(), net_wavlm_d_save_path)
                return None


            print(trainer.model)
            raise ValueError("Model not found.")